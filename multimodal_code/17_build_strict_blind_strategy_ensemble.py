#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a final strict-blind strategy ensemble from multiple strict-blind prediction outputs.

This script does NOT use addH-out labels for selecting models, setting weights, or making
predictions. Labels are only used if --audit-with-labels is provided, and then only for
post-hoc reporting.

Inputs are directories such as:
  outputs_addh_strict_blind_diverse/
  outputs_addh_strict_blind_diverse_more_conservative/
  outputs_addh_strict_blind_oof/
  outputs_addh_strict_blind_stability/

Each directory should contain:
  strict_blind_addhout_predictions.csv
  strict_blind_selected_models.csv  (optional, for strategy quality weighting)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except Exception:
    mean_absolute_error = mean_squared_error = r2_score = None

PRED_CANDIDATES = [
    "pred_strict_blind_weighted",
    "pred_strict_blind",
    "pred_strict_blind_trimmed_mean",
    "pred_strict_blind_median",
    "pred",
    "pred_median",
    "pred_mean",
]

LABEL_COLS = {
    "h_ads_excel", "target", "target_computed",
    "abs_err_vs_h_ads_excel", "abs_err_vs_target", "abs_err_vs_target_computed",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Combine multiple strict-blind prediction strategies without using addH-out labels.")
    ap.add_argument("--strategy-dirs", nargs="+", required=True, help="Directories containing strict_blind_addhout_predictions.csv")
    ap.add_argument("--addhout-master-csv", default="outputs_addh_full_mm_envsplit/addH_out_master_normalized.csv")
    ap.add_argument("--out-dir", default="outputs_addh_strict_blind_final")
    ap.add_argument("--weight-mode", choices=["soft_inverse_oof", "rank", "uniform"], default="soft_inverse_oof")
    ap.add_argument("--min-coverage", type=float, default=0.80)
    ap.add_argument("--clip-to-source-range", action="store_true")
    ap.add_argument("--train-master-csv", default="outputs_addh_full_mm_envsplit/addH_master_target_weighted_mild.csv")
    ap.add_argument("--audit-with-labels", action="store_true")
    ap.add_argument("--target-col", default="h_ads_excel")
    ap.add_argument("--strict-output-no-labels", action="store_true")
    return ap.parse_args()


def detect_pred_col(df: pd.DataFrame) -> Optional[str]:
    for c in PRED_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = str(c).lower()
        if "pred" in lc and not any(x in lc for x in ["std", "min", "max", "err", "rank", "n_"]):
            return c
    return None


def read_strategy_quality(d: Path) -> Dict[str, float]:
    sel = d / "strict_blind_selected_models.csv"
    predp = d / "strict_blind_addhout_predictions.csv"
    q: Dict[str, float] = {"strategy_dir": str(d), "n_selected": 0, "mean_oof_rmse": np.nan, "mean_oof_mae": np.nan, "mean_pred_std": np.nan, "coverage": np.nan}
    if predp.exists():
        try:
            pdf = pd.read_csv(predp)
            pc = detect_pred_col(pdf)
            if pc:
                q["coverage"] = float(pd.to_numeric(pdf[pc], errors="coerce").notna().mean())
            for c in ["pred_strict_blind_std_across_models", "pred_std", "pred_abs_std"]:
                if c in pdf.columns:
                    q["mean_pred_std"] = float(pd.to_numeric(pdf[c], errors="coerce").mean())
                    break
        except Exception:
            pass
    if sel.exists():
        try:
            sdf = pd.read_csv(sel)
            q["n_selected"] = int(len(sdf))
            if "strict_blind_weight" in sdf.columns:
                w = pd.to_numeric(sdf["strict_blind_weight"], errors="coerce").fillna(0.0).to_numpy()
                if w.sum() <= 0:
                    w = np.ones(len(sdf), dtype=float)
            else:
                w = np.ones(len(sdf), dtype=float)
            for src, dst in [("oof_rmse", "mean_oof_rmse"), ("oof_mae", "mean_oof_mae")]:
                if src in sdf.columns:
                    x = pd.to_numeric(sdf[src], errors="coerce").to_numpy()
                    m = np.isfinite(x)
                    if m.any():
                        q[dst] = float(np.average(x[m], weights=w[m]))
            if "pred_std_mean_addhout_unlabeled" in sdf.columns:
                x = pd.to_numeric(sdf["pred_std_mean_addhout_unlabeled"], errors="coerce").to_numpy()
                m = np.isfinite(x)
                if m.any():
                    q["mean_member_pred_std"] = float(np.average(x[m], weights=w[m]))
        except Exception:
            pass
    return q


def make_weights(qdf: pd.DataFrame, mode: str) -> pd.Series:
    qdf = qdf.copy()
    ok = qdf["coverage"].fillna(0) > 0
    if not ok.any():
        return pd.Series(np.ones(len(qdf)) / max(len(qdf), 1), index=qdf.index)
    if mode == "uniform":
        w = pd.Series(0.0, index=qdf.index)
        w.loc[ok] = 1.0
    elif mode == "rank":
        score = pd.Series(0.0, index=qdf.index)
        if "mean_oof_rmse" in qdf:
            score += qdf["mean_oof_rmse"].rank(ascending=True, na_option="bottom")
        if "mean_pred_std" in qdf:
            score += qdf["mean_pred_std"].rank(ascending=True, na_option="bottom") * 0.5
        if "coverage" in qdf:
            score += qdf["coverage"].rank(ascending=False, na_option="bottom") * 0.5
        w = 1.0 / score.replace(0, np.nan)
        w = w.fillna(0.0)
    else:
        rmse = pd.to_numeric(qdf.get("mean_oof_rmse", np.nan), errors="coerce")
        uncertainty = pd.to_numeric(qdf.get("mean_pred_std", np.nan), errors="coerce")
        coverage = pd.to_numeric(qdf.get("coverage", np.nan), errors="coerce").fillna(0.0)
        rmse = rmse.fillna(rmse.median() if rmse.notna().any() else 3.0)
        uncertainty = uncertainty.fillna(uncertainty.median() if uncertainty.notna().any() else 1.0)
        raw = np.exp(-0.9 * rmse) * np.exp(-0.25 * uncertainty) * coverage.clip(0, 1)
        w = pd.Series(raw, index=qdf.index).fillna(0.0)
    w = w.where(ok, 0.0)
    if w.sum() <= 0:
        w.loc[ok] = 1.0
    return w / w.sum()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(args.addhout_master_csv)
    id_col = "id" if "id" in master.columns else master.columns[0]
    keep_cols = [
        "id", "material", "idx", "element", "dopant", "family_base", "family_base_miller", "miller",
        "site_type", "anchor_count", "slab_formula", "h_ads_excel", "target", "target_computed",
        "target_mismatch_excel_minus_computed", "contcar_path", "bare_contcar_path", "cif_path",
    ]
    keep_cols = [c for c in keep_cols if c in master.columns]
    base = master[keep_cols].drop_duplicates(id_col).copy()

    qualities: List[Dict[str, float]] = []
    pred_cols: List[str] = []
    merged = base.copy()

    for sd in args.strategy_dirs:
        d = Path(sd)
        p = d / "strict_blind_addhout_predictions.csv"
        if not p.exists():
            print(f"[WARN] missing predictions: {p}")
            continue
        df = pd.read_csv(p)
        pc = detect_pred_col(df)
        if not pc:
            print(f"[WARN] no prediction column in {p}")
            continue
        tag = d.name.replace("outputs_addh_", "").replace("strict_blind_", "sb_")
        pred_name = f"pred__{tag}"
        tmp = df[[id_col, pc]].copy().rename(columns={pc: pred_name})
        tmp[pred_name] = pd.to_numeric(tmp[pred_name], errors="coerce")
        merged = merged.merge(tmp, on=id_col, how="left")
        pred_cols.append(pred_name)
        q = read_strategy_quality(d)
        q["strategy_name"] = tag
        q["pred_col"] = pred_name
        q["path"] = str(p)
        qualities.append(q)
        print(f"[LOAD] {tag}: {p} pred_col={pc}")

    if not pred_cols:
        raise SystemExit("[ERROR] no valid strict-blind strategy predictions were loaded.")

    qdf = pd.DataFrame(qualities)
    qdf["strategy_weight"] = make_weights(qdf, args.weight_mode).values
    qdf.to_csv(out_dir / "strict_blind_strategy_weights.csv", index=False)

    # Build prediction matrix aligned to selected pred_cols.
    X = merged[pred_cols].apply(pd.to_numeric, errors="coerce")
    weights = np.array([float(qdf.loc[qdf["pred_col"] == c, "strategy_weight"].iloc[0]) for c in pred_cols], dtype=float)
    # zero weight if a sample missing a strategy, renormalize rowwise
    V = X.to_numpy(dtype=float)
    finite = np.isfinite(V)
    W = np.broadcast_to(weights.reshape(1, -1), V.shape).copy()
    W[~finite] = 0.0
    wsum = W.sum(axis=1)
    pred_weighted = np.full(V.shape[0], np.nan)
    ok = wsum > 0
    pred_weighted[ok] = np.nansum(np.where(finite, V, 0.0) * W, axis=1)[ok] / wsum[ok]

    merged["pred_strict_blind_strategy_weighted"] = pred_weighted
    merged["pred_strict_blind_strategy_median"] = np.nanmedian(V, axis=1)
    merged["pred_strict_blind_strategy_mean"] = np.nanmean(V, axis=1)
    # use weighted as primary; median fallback if all weights missing
    merged["pred_strict_blind_final"] = merged["pred_strict_blind_strategy_weighted"].fillna(merged["pred_strict_blind_strategy_median"])
    merged["pred_strict_blind_strategy_std"] = np.nanstd(V, axis=1)
    merged["pred_strict_blind_strategy_n"] = np.isfinite(V).sum(axis=1)

    if args.clip_to_source_range and Path(args.train_master_csv).exists():
        tr = pd.read_csv(args.train_master_csv)
        target_col = "target" if "target" in tr.columns else ("target_computed" if "target_computed" in tr.columns else None)
        if target_col:
            y = pd.to_numeric(tr[target_col], errors="coerce").dropna()
            if len(y):
                lo, hi = float(y.min()), float(y.max())
                merged["pred_strict_blind_final_unclipped"] = merged["pred_strict_blind_final"]
                merged["pred_strict_blind_final"] = merged["pred_strict_blind_final"].clip(lo, hi)
                print(f"[INFO] clipped final prediction to source range [{lo:.4f}, {hi:.4f}]")

    merged["strict_blind_final_rank"] = merged["pred_strict_blind_final"].rank(method="average", ascending=True)
    merged = merged.sort_values("strict_blind_final_rank", na_position="last")

    # Strict output: no labels unless audit requested and no strict-output-no-labels.
    strict = merged.copy()
    if args.strict_output_no_labels or not args.audit_with_labels:
        strict = strict.drop(columns=[c for c in LABEL_COLS if c in strict.columns], errors="ignore")
    strict.to_csv(out_dir / "strict_blind_strategy_ensemble_predictions.csv", index=False)
    strict.to_excel(out_dir / "strict_blind_strategy_ensemble_predictions.xlsx", index=False)

    if args.audit_with_labels:
        target_col = args.target_col
        if target_col in merged.columns:
            tmp = merged[[target_col, "pred_strict_blind_final"]].copy()
            tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")
            tmp["pred_strict_blind_final"] = pd.to_numeric(tmp["pred_strict_blind_final"], errors="coerce")
            tmp = tmp.dropna()
            audit = {}
            if len(tmp) and mean_absolute_error is not None:
                yt = tmp[target_col].to_numpy(float)
                yp = tmp["pred_strict_blind_final"].to_numpy(float)
                audit = {
                    "target_col": target_col,
                    "n": int(len(tmp)),
                    "mae": float(mean_absolute_error(yt, yp)),
                    "rmse": float(math.sqrt(mean_squared_error(yt, yp))),
                    "r2": float(r2_score(yt, yp)) if len(np.unique(yt)) > 1 else np.nan,
                    "pearson": float(pd.Series(yp).corr(pd.Series(yt), method="pearson")),
                    "spearman": float(pd.Series(yp).corr(pd.Series(yt), method="spearman")),
                    "bias": float(np.mean(yp - yt)),
                }
            pd.DataFrame([audit]).to_csv(out_dir / "strict_blind_strategy_ensemble_posthoc_audit.csv", index=False)
            merged.to_csv(out_dir / "strict_blind_strategy_ensemble_predictions_with_audit_labels.csv", index=False)
            print("[AUDIT]", audit)

    print("[OK] saved:", out_dir / "strict_blind_strategy_weights.csv")
    print("[OK] saved:", out_dir / "strict_blind_strategy_ensemble_predictions.csv")
    print("[OK] saved:", out_dir / "strict_blind_strategy_ensemble_predictions.xlsx")
    print("[INFO] strategies used:", ",".join(qdf["strategy_name"].astype(str).tolist()))
    print("[INFO] first rows:")
    show_cols = [
        "id", "material", "element", "dopant",
        "pred_strict_blind_final", "pred_strict_blind_strategy_weighted",
        "pred_strict_blind_strategy_median", "pred_strict_blind_strategy_std",
        "pred_strict_blind_strategy_n", "strict_blind_final_rank",
    ]
    show_cols = [c for c in show_cols if c in strict.columns]
    print(strict[show_cols].head(50).to_string(index=False))


if __name__ == "__main__":
    main()
