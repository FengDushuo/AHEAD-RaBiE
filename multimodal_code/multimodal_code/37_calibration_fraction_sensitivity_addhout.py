#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration-fraction sensitivity for AddH-out few-shot validation.

This script varies how many AddH-out labels are allowed for target-domain
calibration and evaluates the remaining held-out AddH-out samples. It reuses
the few-shot residual calibration logic from 36_validate_addhout_fewshot_holdout.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def load_fewshot_module():
    path = Path(__file__).with_name("36_validate_addhout_fewshot_holdout.py")
    if not path.exists():
        raise SystemExit(f"[ERROR] missing dependency: {path}")
    spec = importlib.util.spec_from_file_location("addhout_fewshot36", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[ERROR] could not import dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="AddH-out calibration fraction sensitivity.")
    ap.add_argument("--pred-csv", default="outputs_addh_bidirectional_chemistry_prior/bidirectional_chemistry_addhout_predictions.csv")
    ap.add_argument("--labels-csv", default="outputs_addh_llm_element_priors/addhout_audit_labels.csv")
    ap.add_argument("--out-dir", default="outputs_addh_calibration_fraction_sensitivity")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--target-col", default="h_ads_excel")
    ap.add_argument("--anchor-col", default="pred_chem_spike_final")
    ap.add_argument("--compare-cols", default="pred_superblend_final")
    ap.add_argument("--reference-cols", default="pred_bidir_chem_final,pred_bidir_chem_conservative")
    ap.add_argument("--calibration-fracs", default="0.10,0.20,0.30,0.50,0.65,0.70,0.80")
    ap.add_argument("--n-repeats", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write-xlsx", action="store_true")
    return ap.parse_args()


def parse_fracs(raw: str) -> List[float]:
    vals: List[float] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = float(tok)
        if not (0.0 < v < 1.0):
            raise SystemExit(f"[ERROR] calibration fraction must be in (0,1): {v}")
        vals.append(v)
    if not vals:
        raise SystemExit("[ERROR] no calibration fractions supplied.")
    return sorted(dict.fromkeys(vals))


def material_stratified_calibration_split(
    df: pd.DataFrame, calibration_frac: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    cal_idx: List[int] = []
    for _, sub in df.groupby("material", dropna=False):
        idx = sub.index.to_numpy()
        n_cal = max(1, int(round(len(idx) * calibration_frac)))
        n_cal = min(n_cal, max(1, len(idx) - 1))
        cal_idx.extend(rng.choice(idx, size=n_cal, replace=False).tolist())
    cal_idx = np.asarray(sorted(set(cal_idx)), dtype=int)
    cal_set = set(cal_idx.tolist())
    test_idx = np.asarray([i for i in df.index if i not in cal_set], dtype=int)
    return cal_idx, test_idx


def summarize_with_sizes(metrics_df: pd.DataFrame, group_cols: Sequence[str], fewshot_module) -> pd.DataFrame:
    summary = fewshot_module.summarize_metric_table(metrics_df, group_cols)
    size_rows: List[Dict[str, object]] = []
    for key, sub in metrics_df.groupby(list(group_cols), dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row: Dict[str, object] = dict(zip(group_cols, key))
        row["n_calib_mean"] = float(pd.to_numeric(sub["n_calib"], errors="coerce").mean())
        row["n_test_mean"] = float(pd.to_numeric(sub["n_test"], errors="coerce").mean())
        size_rows.append(row)
    return summary.merge(pd.DataFrame(size_rows), on=list(group_cols), how="left")


def paired_improvement(metrics_df: pd.DataFrame, baseline_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for frac, sub in metrics_df.groupby("calibration_fraction", dropna=False):
        piv = sub.pivot_table(index="split_id", columns="pred_col", values="mae", aggfunc="first")
        if "fewshot_calibrated" not in piv.columns:
            continue
        for base in baseline_cols:
            if base not in piv.columns:
                continue
            delta = piv[base] - piv["fewshot_calibrated"]
            delta = delta.dropna()
            if len(delta) == 0:
                continue
            rows.append(
                {
                    "calibration_fraction": frac,
                    "baseline_col": base,
                    "n_splits": int(len(delta)),
                    "mean_mae_reduction": float(delta.mean()),
                    "median_mae_reduction": float(delta.median()),
                    "q05_mae_reduction": float(delta.quantile(0.05)),
                    "q95_mae_reduction": float(delta.quantile(0.95)),
                    "win_rate": float((delta > 0).mean()),
                    "fewshot_worse_count": int((delta < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    few = load_fewshot_module()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = few.read_table(Path(args.pred_csv))
    labels = few.read_table(Path(args.labels_csv))
    if args.id_col not in pred.columns or args.id_col not in labels.columns:
        raise SystemExit("[ERROR] both prediction and labels files must contain id column.")
    if args.target_col not in labels.columns:
        raise SystemExit(f"[ERROR] labels file missing target column: {args.target_col}")
    if args.anchor_col not in pred.columns:
        raise SystemExit(f"[ERROR] prediction file missing anchor column: {args.anchor_col}")

    data = pred.merge(labels[[args.id_col, args.target_col]], on=args.id_col, how="left")
    data = data.dropna(subset=[args.target_col, args.anchor_col]).copy().reset_index(drop=True)
    data = few.add_prediction_features(data, args.target_col, args.anchor_col)
    compare_cols = [c.strip() for c in args.compare_cols.split(",") if c.strip()]
    reference_cols = [c.strip() for c in args.reference_cols.split(",") if c.strip()]
    fracs = parse_fracs(args.calibration_fracs)
    rng = np.random.default_rng(args.seed)

    metric_rows: List[Dict[str, object]] = []
    detail_rows: List[pd.DataFrame] = []
    for frac in fracs:
        for rep in range(args.n_repeats):
            cal_idx, test_idx = material_stratified_calibration_split(data, frac, rng)
            rows, detail = few.evaluate_split(
                data,
                cal_idx,
                test_idx,
                args.anchor_col,
                args.target_col,
                "calibration_fraction_random",
                f"{frac:.3f}:{rep}",
                compare_cols,
            )
            for row in rows:
                row["calibration_fraction"] = float(frac)
            detail["calibration_fraction"] = float(frac)
            metric_rows.extend(rows)
            detail_rows.append(detail)

    metrics_df = pd.DataFrame(metric_rows)
    detail_df = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    summary = summarize_with_sizes(metrics_df, ["calibration_fraction", "pred_col"], few)
    paired = paired_improvement(metrics_df, [args.anchor_col] + [c for c in compare_cols if c])

    reference_rows: List[Dict[str, object]] = []
    for c in [args.anchor_col] + [x for x in compare_cols if x in data.columns] + [x for x in reference_cols if x in data.columns]:
        row = few.metrics(data[args.target_col], data[c])
        row.update({"pred_col": c, "note": "full_data_reference_not_holdout"})
        reference_rows.append(row)
    reference = pd.DataFrame(reference_rows)

    metrics_df.to_csv(out_dir / "calibration_fraction_split_metrics.csv", index=False)
    summary.to_csv(out_dir / "calibration_fraction_summary.csv", index=False)
    paired.to_csv(out_dir / "calibration_fraction_paired_improvement.csv", index=False)
    detail_df.to_csv(out_dir / "calibration_fraction_predictions.csv", index=False)
    reference.to_csv(out_dir / "full_data_reference_metrics.csv", index=False)
    if args.write_xlsx:
        with pd.ExcelWriter(out_dir / "calibration_fraction_sensitivity_report.xlsx") as w:
            summary.to_excel(w, sheet_name="summary", index=False)
            paired.to_excel(w, sheet_name="paired_improvement", index=False)
            metrics_df.to_excel(w, sheet_name="split_metrics", index=False)
            reference.to_excel(w, sheet_name="full_reference", index=False)
            detail_df.head(30000).to_excel(w, sheet_name="predictions_sample", index=False)

    manifest = {
        "script": Path(__file__).name,
        "dependency": "36_validate_addhout_fewshot_holdout.py",
        "pred_csv": args.pred_csv,
        "labels_csv": args.labels_csv,
        "anchor_col": args.anchor_col,
        "compare_cols": compare_cols,
        "full_data_reference_cols": reference_cols,
        "calibration_fractions": fracs,
        "n_labeled_rows": int(len(data)),
        "n_repeats_per_fraction": args.n_repeats,
        "seed": args.seed,
        "validation_claim": "calibration-size sensitivity for internal target-domain few-shot holdout",
        "test_labels_used_for_training": False,
    }
    (out_dir / "calibration_fraction_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[OK] wrote {out_dir}")
    print("[SUMMARY]")
    print(summary.to_string(index=False))
    print("[PAIRED IMPROVEMENT]")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
