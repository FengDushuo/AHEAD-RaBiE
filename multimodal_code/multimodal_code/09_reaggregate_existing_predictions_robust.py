#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robustly re-aggregate existing multi-view predictions without retraining.

Inputs expected in --work-dir:
  - test_pred_all_runs.csv
  - addH_out_pred_all_runs.csv

Outputs:
  - robust_runs_used.csv
  - test_pred_oof_ensemble_robust.csv
  - test_pred_oof_ensemble_robust_metrics.json
  - addH_out_pred_ensemble_robust.csv
  - addH_out_top20_low_robust.csv
  - addH_out_top20_high_robust.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--method", default="median", choices=["median", "mean", "val_mae_weighted"])
    ap.add_argument("--max-val-mae", type=float, default=3.0)
    ap.add_argument("--max-test-rmse", type=float, default=10.0)
    ap.add_argument("--max-abs-pred", type=float, default=30.0)
    ap.add_argument("--min-runs", type=int, default=3, help="Fallback to all runs if filters keep fewer than this many runs.")
    ap.add_argument("--topk", type=int, default=20)
    return ap.parse_args()


def save_json(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def metrics(df: pd.DataFrame, pred_col="pred"):
    y = df["target"].to_numpy(dtype=float)
    p = df[pred_col].to_numpy(dtype=float)
    return {
        "n": int(len(df)),
        "mae": float(mean_absolute_error(y, p)) if len(df) else None,
        "rmse": float(np.sqrt(mean_squared_error(y, p))) if len(df) else None,
        "r2": float(r2_score(y, p)) if len(df) > 1 else None,
        "target_min": float(np.min(y)) if len(df) else None,
        "target_max": float(np.max(y)) if len(df) else None,
        "pred_min": float(np.min(p)) if len(df) else None,
        "pred_max": float(np.max(p)) if len(df) else None,
    }


def aggregate(sub: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "median":
        return sub.groupby("id", as_index=False)["pred"].median()
    if method == "mean":
        return sub.groupby("id", as_index=False)["pred"].mean()
    if method == "val_mae_weighted":
        tmp = sub[["id", "pred", "run_weight"]].copy()
        if "run_weight" not in tmp.columns or tmp["run_weight"].isna().all():
            tmp["run_weight"] = 1.0
        tmp["run_weight"] = pd.to_numeric(tmp["run_weight"], errors="coerce").fillna(1.0).clip(lower=1e-8)
        num = tmp.assign(wx=tmp["pred"] * tmp["run_weight"]).groupby("id", as_index=False)["wx"].sum()
        den = tmp.groupby("id", as_index=False)["run_weight"].sum()
        out = num.merge(den, on="id", how="inner")
        out["pred"] = out["wx"] / out["run_weight"].replace(0, np.nan)
        return out[["id", "pred"]]
    raise ValueError(method)


def main():
    args = parse_args()
    work = Path(args.work_dir).resolve()
    oof_path = work / "test_pred_all_runs.csv"
    out_path = work / "addH_out_pred_all_runs.csv"
    if not oof_path.exists():
        raise FileNotFoundError(oof_path)
    if not out_path.exists():
        raise FileNotFoundError(out_path)

    oof = pd.read_csv(oof_path)
    out = pd.read_csv(out_path)
    for df_name, df in [("test_pred_all_runs.csv", oof), ("addH_out_pred_all_runs.csv", out)]:
        if "run_tag" not in df.columns:
            raise ValueError(f"{df_name} missing run_tag")
        if "id" not in df.columns or "pred" not in df.columns:
            raise ValueError(f"{df_name} missing id/pred")
        df["id"] = df["id"].astype(str)
        df["pred"] = pd.to_numeric(df["pred"], errors="coerce")

    if "target" not in oof.columns:
        raise ValueError("test_pred_all_runs.csv must contain target")
    oof["target"] = pd.to_numeric(oof["target"], errors="coerce")
    oof = oof[np.isfinite(oof["target"]) & np.isfinite(oof["pred"])]
    out = out[np.isfinite(out["pred"])]

    stats = []
    for run_tag, sub in oof.groupby("run_tag"):
        sub = sub.copy()
        ae = (sub["pred"] - sub["target"]).abs()
        val_mae = np.nan
        if "val_mae_cal" in sub.columns:
            val_mae = float(pd.to_numeric(sub["val_mae_cal"], errors="coerce").dropna().median())
        stats.append({
            "run_tag": str(run_tag),
            "fold": int(sub["fold"].iloc[0]) if "fold" in sub.columns and pd.notna(sub["fold"].iloc[0]) else None,
            "seed": int(sub["seed"].iloc[0]) if "seed" in sub.columns and pd.notna(sub["seed"].iloc[0]) else None,
            "n": int(len(sub)),
            "val_mae_cal": val_mae,
            "test_mae": float(ae.mean()),
            "test_rmse": float(np.sqrt(np.mean((sub["pred"] - sub["target"]) ** 2))),
            "test_median_abs_err": float(ae.median()),
            "test_p95_abs_err": float(ae.quantile(0.95)),
            "test_max_abs_err": float(ae.max()),
            "max_abs_pred": float(sub["pred"].abs().max()),
        })
    stats_df = pd.DataFrame(stats)

    keep = pd.Series(True, index=stats_df.index)
    if args.max_val_mae is not None and "val_mae_cal" in stats_df.columns:
        keep &= stats_df["val_mae_cal"].isna() | (stats_df["val_mae_cal"] <= float(args.max_val_mae))
    if args.max_test_rmse is not None:
        keep &= stats_df["test_rmse"] <= float(args.max_test_rmse)
    if args.max_abs_pred is not None:
        keep &= stats_df["max_abs_pred"] <= float(args.max_abs_pred)

    stats_df["used_for_robust_ensemble"] = keep
    kept_runs = set(stats_df.loc[keep, "run_tag"].astype(str))
    if len(kept_runs) < int(args.min_runs):
        print(f"[WARN] filters kept only {len(kept_runs)} runs < min_runs={args.min_runs}; falling back to all runs.")
        kept_runs = set(stats_df["run_tag"].astype(str))
        stats_df["used_for_robust_ensemble"] = True

    stats_path = work / "robust_runs_used.csv"
    stats_df.sort_values(["used_for_robust_ensemble", "test_rmse"], ascending=[False, True]).to_csv(stats_path, index=False)

    oof_used = oof[oof["run_tag"].astype(str).isin(kept_runs)].copy()
    out_used = out[out["run_tag"].astype(str).isin(kept_runs)].copy()

    # Aggregate OOF separately within each held-out fold to avoid mixing duplicate fold predictions incorrectly.
    oof_parts = []
    group_cols = ["fold"] if "fold" in oof_used.columns else ["__all__"]
    if group_cols == ["__all__"]:
        oof_used["__all__"] = 0
    for _, sub in oof_used.groupby(group_cols):
        agg = aggregate(sub, args.method)
        truth = sub[["id", "target"]].drop_duplicates("id")
        joined = truth.merge(agg, on="id", how="inner")
        if "fold" in sub.columns:
            joined["fold"] = int(sub["fold"].iloc[0])
        oof_parts.append(joined)
    oof_ens = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame(columns=["id", "target", "pred"])
    oof_ens.to_csv(work / "test_pred_oof_ensemble_robust.csv", index=False)
    save_json(work / "test_pred_oof_ensemble_robust_metrics.json", metrics(oof_ens, "pred"))

    out_agg = aggregate(out_used, args.method).rename(columns={"pred": "pred_robust"})
    manifest = out_used.drop(columns=["pred"], errors="ignore").drop_duplicates("id")
    final = manifest.merge(out_agg, on="id", how="left")
    final = final.rename(columns={"pred_robust": "pred"})
    final.to_csv(work / "addH_out_pred_ensemble_robust.csv", index=False)
    final.sort_values("pred").head(int(args.topk)).to_csv(work / "addH_out_top20_low_robust.csv", index=False)
    final.sort_values("pred", ascending=False).head(int(args.topk)).to_csv(work / "addH_out_top20_high_robust.csv", index=False)

    print(f"[OK] run filter summary -> {stats_path}")
    print(f"[OK] robust OOF -> {work / 'test_pred_oof_ensemble_robust.csv'}")
    print(f"[OK] robust addH-out -> {work / 'addH_out_pred_ensemble_robust.csv'}")
    print("\n[RUNS]")
    with pd.option_context("display.max_columns", 80, "display.width", 180):
        print(stats_df.sort_values("test_rmse", ascending=False).to_string(index=False))
    print("\n[OOF robust metrics]")
    print(metrics(oof_ens, "pred"))


if __name__ == "__main__":
    main()
