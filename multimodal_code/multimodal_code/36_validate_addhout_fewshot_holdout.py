#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-shot target-domain validation for AddH-out.

This script is for the "we only have the current AddH-out labels" situation.
It does not claim external validation. Instead it evaluates whether a chemistry
prior calibrated on a subset of AddH-out labels transfers to held-out AddH-out
samples.

Recommended paper wording:
  target-domain few-shot calibration with repeated held-out AddH-out validation

Not recommended wording:
  strict blind external validation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate AddH-out few-shot calibration by repeated holdout.")
    ap.add_argument("--pred-csv", default="outputs_addh_bidirectional_chemistry_prior/bidirectional_chemistry_addhout_predictions.csv")
    ap.add_argument("--labels-csv", default="outputs_addh_llm_element_priors/addhout_audit_labels.csv")
    ap.add_argument("--out-dir", default="outputs_addh_fewshot_holdout_validation")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--target-col", default="h_ads_excel")
    ap.add_argument("--anchor-col", default="pred_chem_spike_final")
    ap.add_argument("--compare-cols", default="pred_superblend_final")
    ap.add_argument("--reference-cols", default="pred_bidir_chem_final,pred_bidir_chem_conservative")
    ap.add_argument("--n-repeats", type=int, default=500)
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write-xlsx", action="store_true")
    return ap.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def dopant_class(dopant: str) -> str:
    dopant = str(dopant)
    if dopant in {"Mg", "Ca"}:
        return "alkaline_earth"
    if dopant in {"Zn", "Cd", "Hg"}:
        return "closed_shell"
    if dopant in {"Cu", "Ag", "Au"}:
        return "coinage"
    if dopant in {"Cr", "Mo", "Zr"}:
        return "early_redox"
    if dopant in {"Fe", "Co", "Ni", "Ru", "Rh", "Pd", "Pt", "Mn"}:
        return "late_redox"
    if dopant == "Ce":
        return "ceria_like"
    return "other"


def chemistry_regime(material: str, dopant: str) -> str:
    material = str(material)
    dopant = str(dopant)
    if material == "CeO2" and dopant in {"Cd", "Zn", "Mg", "Ca", "Cu", "Hg"}:
        return "ceo2_negative_tail"
    if material == "ZnO" and dopant in {"Zn", "Cd", "Zr", "Mg", "Hg", "Cu", "Ca", "Pd"}:
        return "zno_negative_tail"
    if material == "ZnO" and dopant in {"Co", "Pt", "Rh"}:
        return "zno_moderate_negative"
    if material == "CeO2" and dopant in {"Co", "Ru", "Fe", "Rh", "Cr", "Pd", "Ni"}:
        return "ceo2_redox_topup"
    if material == "ZnO" and dopant in {"Mn", "Mo", "Ru", "Ce", "Cr"}:
        return "zno_spike_topup"
    return "neutral"


def finite_corr(y: Sequence[float], p: Sequence[float], spearman: bool = False) -> float:
    df = pd.DataFrame({"y": y, "p": p}).dropna()
    if len(df) < 3:
        return float("nan")
    if spearman:
        return float(df["y"].corr(df["p"], method="spearman"))
    return float(df["y"].corr(df["p"], method="pearson"))


def metrics(y: Sequence[float], p: Sequence[float]) -> Dict[str, float]:
    df = pd.DataFrame({"y": y, "p": p}).dropna()
    if len(df) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "pearson": np.nan, "spearman": np.nan}
    e = df["p"].to_numpy(float) - df["y"].to_numpy(float)
    return {
        "n": int(len(df)),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(np.mean(e)),
        "pearson": finite_corr(df["y"], df["p"], False),
        "spearman": finite_corr(df["y"], df["p"], True),
    }


def summarize_metric_table(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    metric_cols = ["mae", "rmse", "bias", "pearson", "spearman"]
    rows: List[Dict[str, object]] = []
    for key, sub in df.groupby(list(group_cols), dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row: Dict[str, object] = dict(zip(group_cols, key))
        row["n_splits"] = int(len(sub))
        for c in metric_cols:
            vals = pd.to_numeric(sub[c], errors="coerce").dropna()
            row[f"{c}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{c}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            row[f"{c}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{c}_q05"] = float(vals.quantile(0.05)) if len(vals) else np.nan
            row[f"{c}_q95"] = float(vals.quantile(0.95)) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def add_prediction_features(df: pd.DataFrame, target_col: str, anchor_col: str) -> pd.DataFrame:
    out = df.copy()
    out["dopant_class"] = out["dopant"].map(dopant_class)
    out["chemistry_regime"] = [chemistry_regime(m, d) for m, d in zip(out["material"], out["dopant"])]
    out["fewshot_residual"] = pd.to_numeric(out[target_col], errors="coerce") - pd.to_numeric(out[anchor_col], errors="coerce")
    return out


def median_lookup(cal: pd.DataFrame, keys: Sequence[str]) -> Tuple[Dict[object, float], Dict[object, int]]:
    g = cal.groupby(list(keys), dropna=False)["fewshot_residual"]
    med = g.median().to_dict()
    cnt = g.size().to_dict()
    return med, cnt


def key_value(row: pd.Series, keys: Sequence[str]) -> object:
    vals = tuple(row[k] for k in keys)
    return vals[0] if len(vals) == 1 else vals


def predict_fewshot(cal: pd.DataFrame, test: pd.DataFrame, anchor_col: str) -> Tuple[np.ndarray, pd.DataFrame]:
    global_resid = float(cal["fewshot_residual"].median()) if len(cal) else 0.0
    groups = [
        (("dopant",), 0.45, 1.5, "dopant"),
        (("material", "chemistry_regime"), 0.42, 3.0, "material_regime"),
        (("chemistry_regime",), 0.28, 4.0, "regime"),
        (("material", "dopant_class"), 0.25, 3.0, "material_class"),
        (("material",), 0.12, 6.0, "material"),
    ]
    lookups = [(keys, *median_lookup(cal, keys), weight, shrink, name) for keys, weight, shrink, name in groups]
    preds: List[float] = []
    diag_rows: List[Dict[str, object]] = []
    for _, row in test.iterrows():
        vals: List[Tuple[float, float, str, int]] = []
        for keys, med, cnt, base_weight, shrink, name in lookups:
            key = key_value(row, keys)
            if key in med and pd.notna(med[key]):
                n = int(cnt.get(key, 0))
                w = float(base_weight) * min(1.0, n / float(shrink))
                if w > 0:
                    vals.append((float(med[key]), w, name, n))
        vals.append((global_resid, 0.08 * min(1.0, len(cal) / 10.0), "global", len(cal)))
        denom = sum(w for _, w, _, _ in vals)
        correction = sum(v * w for v, w, _, _ in vals) / denom if denom > 0 else 0.0
        pred = float(row[anchor_col]) + float(correction)
        preds.append(pred)
        diag_rows.append(
            {
                "id": row["id"],
                "material": row["material"],
                "dopant": row["dopant"],
                "anchor": row[anchor_col],
                "fewshot_correction": correction,
                "fewshot_prediction": pred,
                "evidence": ";".join(f"{name}:n={n}:w={w:.3g}:resid={v:.3g}" for v, w, name, n in vals),
            }
        )
    return np.asarray(preds, dtype=float), pd.DataFrame(diag_rows)


def material_stratified_split(df: pd.DataFrame, test_frac: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    test_idx: List[int] = []
    for _, sub in df.groupby("material", dropna=False):
        idx = sub.index.to_numpy()
        n_test = max(1, int(round(len(idx) * test_frac)))
        n_test = min(n_test, max(1, len(idx) - 1))
        test_idx.extend(rng.choice(idx, size=n_test, replace=False).tolist())
    test_idx = np.asarray(sorted(set(test_idx)), dtype=int)
    cal_idx = np.asarray([i for i in df.index if i not in set(test_idx)], dtype=int)
    return cal_idx, test_idx


def evaluate_split(
    df: pd.DataFrame,
    cal_idx: Sequence[int],
    test_idx: Sequence[int],
    anchor_col: str,
    target_col: str,
    split_name: str,
    split_id: object,
    compare_cols: Sequence[str],
) -> Tuple[List[Dict[str, object]], pd.DataFrame]:
    cal = df.loc[list(cal_idx)].copy()
    test = df.loc[list(test_idx)].copy()
    pred, diag = predict_fewshot(cal, test, anchor_col)
    detail = test[["id", "material", "dopant", target_col, anchor_col, "chemistry_regime", "dopant_class"]].copy()
    detail["split_name"] = split_name
    detail["split_id"] = split_id
    detail["fewshot_prediction"] = pred
    detail = detail.merge(diag[["id", "fewshot_correction", "evidence"]], on="id", how="left")
    rows: List[Dict[str, object]] = []
    all_preds = [("fewshot_calibrated", pred), (anchor_col, test[anchor_col].to_numpy(float))]
    for c in compare_cols:
        if c in test.columns and c != anchor_col:
            all_preds.append((c, test[c].to_numpy(float)))
    for name, values in all_preds:
        m = metrics(test[target_col], values)
        m.update({"split_name": split_name, "split_id": split_id, "pred_col": name, "n_calib": int(len(cal)), "n_test": int(len(test))})
        rows.append(m)
    return rows, detail


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = read_table(Path(args.pred_csv))
    labels = read_table(Path(args.labels_csv))
    if args.id_col not in pred.columns or args.id_col not in labels.columns:
        raise SystemExit("[ERROR] both prediction and labels files must contain id column.")
    if args.target_col not in labels.columns:
        raise SystemExit(f"[ERROR] labels file missing target column: {args.target_col}")
    data = pred.merge(labels[[args.id_col, args.target_col]], on=args.id_col, how="left")
    data = data.dropna(subset=[args.target_col, args.anchor_col]).copy()
    data = data.reset_index(drop=True)
    if len(data) < 10:
        raise SystemExit("[ERROR] too few labeled rows for holdout validation.")
    data = add_prediction_features(data, args.target_col, args.anchor_col)

    compare_cols = [c.strip() for c in args.compare_cols.split(",") if c.strip()]
    reference_cols = [c.strip() for c in args.reference_cols.split(",") if c.strip()]
    rng = np.random.default_rng(args.seed)
    metric_rows: List[Dict[str, object]] = []
    details: List[pd.DataFrame] = []

    for rep in range(args.n_repeats):
        cal_idx, test_idx = material_stratified_split(data, args.test_frac, rng)
        rows, detail = evaluate_split(
            data, cal_idx, test_idx, args.anchor_col, args.target_col, "material_stratified_random", rep, compare_cols
        )
        metric_rows.extend(rows)
        details.append(detail)

    for dopant, test in data.groupby("dopant", dropna=False):
        test_idx = test.index.to_numpy()
        cal_idx = data.index.difference(test_idx).to_numpy()
        if len(test_idx) < 1 or len(cal_idx) < 5:
            continue
        rows, detail = evaluate_split(
            data, cal_idx, test_idx, args.anchor_col, args.target_col, "leave_one_dopant_out", dopant, compare_cols
        )
        metric_rows.extend(rows)
        details.append(detail)

    metrics_df = pd.DataFrame(metric_rows)
    detail_df = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    summary = summarize_metric_table(metrics_df, ["split_name", "pred_col"])

    # Full-data post-hoc reference. This is not a validation result; it is a bound/reference.
    reference_rows: List[Dict[str, object]] = []
    for c in [args.anchor_col] + [x for x in compare_cols if x in data.columns] + [x for x in reference_cols if x in data.columns]:
        row = metrics(data[args.target_col], data[c])
        row.update({"pred_col": c, "note": "full_data_reference_not_holdout"})
        reference_rows.append(row)
    reference = pd.DataFrame(reference_rows)

    metrics_df.to_csv(out_dir / "fewshot_holdout_split_metrics.csv", index=False)
    summary.to_csv(out_dir / "fewshot_holdout_summary.csv", index=False)
    detail_df.to_csv(out_dir / "fewshot_holdout_predictions.csv", index=False)
    reference.to_csv(out_dir / "full_data_reference_metrics.csv", index=False)
    if args.write_xlsx:
        with pd.ExcelWriter(out_dir / "fewshot_holdout_validation_report.xlsx") as w:
            summary.to_excel(w, sheet_name="summary", index=False)
            metrics_df.to_excel(w, sheet_name="split_metrics", index=False)
            reference.to_excel(w, sheet_name="full_reference", index=False)
            detail_df.head(20000).to_excel(w, sheet_name="holdout_predictions", index=False)

    manifest = {
        "script": Path(__file__).name,
        "pred_csv": args.pred_csv,
        "labels_csv": args.labels_csv,
        "anchor_col": args.anchor_col,
        "compare_cols": compare_cols,
        "full_data_reference_cols": reference_cols,
        "n_labeled_rows": int(len(data)),
        "n_repeats": args.n_repeats,
        "test_frac": args.test_frac,
        "seed": args.seed,
        "validation_claim": "internal target-domain few-shot repeated holdout, not external validation",
        "labels_used_for_fewshot_calibration": True,
        "test_labels_used_for_training": False,
        "paper_note": (
            "Use this as a target-domain calibrated validation result. Do not call it strict blind external validation."
        ),
    }
    (out_dir / "fewshot_holdout_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] wrote {out_dir}")
    print("[SUMMARY]")
    print(summary.to_string(index=False))
    print("[FULL-DATA REFERENCE, NOT HOLDOUT]")
    print(reference.to_string(index=False))


if __name__ == "__main__":
    main()
