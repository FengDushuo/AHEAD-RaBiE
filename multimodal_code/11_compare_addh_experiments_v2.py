#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare AddH multimodal experiment grid results.

It scans each experiment directory and collects:
- robust OOF metrics
- number of robust runs used
- addH-out posterior metrics if h_ads_excel/target_computed exists
- final-all seed results if present

Usage:
  python 11_compare_addh_experiments.py \
    --grid-root outputs_addh_expgrid \
    --output-csv outputs_addh_expgrid/experiment_comparison_summary.csv \
    --output-xlsx outputs_addh_expgrid/experiment_comparison_summary.xlsx
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd


def read_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def numeric(s):
    return pd.to_numeric(s, errors="coerce")


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = numeric(pd.Series(y_true))
    y_pred = numeric(pd.Series(y_pred))
    m = y_true.notna() & y_pred.notna()
    y_true = y_true[m].astype(float)
    y_pred = y_pred[m].astype(float)
    if len(y_true) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "pearson": np.nan, "spearman": np.nan}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    pearson = float(pd.Series(y_true).corr(pd.Series(y_pred), method="pearson")) if len(y_true) >= 2 else np.nan
    spearman = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")) if len(y_true) >= 2 else np.nan
    return {"n": int(len(y_true)), "mae": mae, "rmse": rmse, "r2": r2, "pearson": pearson, "spearman": spearman}


def addhout_metrics(p: Path, pred_col: str = "pred_median") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not p.exists():
        return out
    try:
        df = pd.read_csv(p)
    except Exception:
        return out
    if pred_col not in df.columns:
        if "pred" in df.columns:
            pred_col = "pred"
        else:
            return out

    out["addhout_rows"] = int(len(df))
    out["addhout_pred_min"] = float(numeric(df[pred_col]).min())
    out["addhout_pred_max"] = float(numeric(df[pred_col]).max())
    out["addhout_pred_mean"] = float(numeric(df[pred_col]).mean())
    out["addhout_pred_median"] = float(numeric(df[pred_col]).median())
    if "pred_std" in df.columns:
        out["addhout_pred_std_mean"] = float(numeric(df["pred_std"]).mean())
        out["addhout_pred_std_median"] = float(numeric(df["pred_std"]).median())
    else:
        out["addhout_pred_std_mean"] = np.nan
        out["addhout_pred_std_median"] = np.nan

    for true_col in ["h_ads_excel", "target_computed", "target"]:
        if true_col in df.columns:
            m = regression_metrics(df[true_col], df[pred_col])
            prefix = f"addhout_vs_{true_col}"
            for k, v in m.items():
                out[f"{prefix}_{k}"] = v

    # Top candidate summary: lower predicted adsorption energy
    if "id" in df.columns:
        top = df.sort_values(pred_col, ascending=True).head(10)
        out["top10_low_ids"] = ";".join(top["id"].astype(str).tolist())
    return out


def parse_exp_config(name: str) -> Dict[str, Any]:
    # Examples: t10_ssl_median_u2_wp05
    out: Dict[str, Any] = {}
    parts = name.split("_")
    out["target_abs_max"] = np.nan
    if parts and parts[0].startswith("t"):
        try:
            out["target_abs_max"] = float(parts[0][1:])
        except Exception:
            pass
    out["ssl"] = "ssl" in parts and "nossl" not in parts
    out["ensemble_label"] = "weighted" if "weighted" in parts else ("median" if "median" in parts else "")
    for p in parts:
        if p.startswith("u") and p[1:].isdigit():
            out["unfreeze_top_n"] = int(p[1:])
        if p.startswith("wp"):
            try:
                out["sampler_power"] = float(p[2:]) / 10.0
            except Exception:
                pass
    return out


def collect_one(exp_dir: Path) -> Dict[str, Any]:
    name = exp_dir.name
    row: Dict[str, Any] = {"experiment": name, "exp_dir": str(exp_dir)}
    row.update(parse_exp_config(name))
    cfg = read_json(exp_dir / "experiment_config.json")
    for k, v in cfg.items():
        if k not in row:
            row[k] = v

    work = exp_dir / "cv_work"
    cv_metrics = read_json(work / "test_pred_oof_ensemble_robust_metrics.json")
    for k, v in cv_metrics.items():
        row[f"oof_{k}"] = v

    runs_p = work / "robust_runs_used.csv"
    if runs_p.exists():
        try:
            runs = pd.read_csv(runs_p)
            row["runs_total"] = int(len(runs))
            if "used_for_robust_ensemble" in runs.columns:
                used = runs["used_for_robust_ensemble"].astype(str).str.lower().isin(["true", "1", "yes"])
                row["runs_used"] = int(used.sum())
                row["runs_used_frac"] = float(used.mean()) if len(used) else np.nan
            if "test_rmse" in runs.columns:
                row["run_test_rmse_min"] = float(numeric(runs["test_rmse"]).min())
                row["run_test_rmse_median"] = float(numeric(runs["test_rmse"]).median())
                row["run_test_rmse_max"] = float(numeric(runs["test_rmse"]).max())
        except Exception:
            pass

    byid_p = work / "addH_out_pred_ensemble_robust_final_by_id.csv"
    row.update(addhout_metrics(byid_p, pred_col="pred_median"))

    # final-all seed-level outputs if present
    final_rows: List[pd.DataFrame] = []
    for fp in sorted(exp_dir.glob("final_work_seed*/addH_out_pred_ensemble.csv")):
        try:
            d = pd.read_csv(fp).drop(columns=["eq_emb"], errors="ignore")
            d["source_final_file"] = str(fp)
            d["source_seed_dir"] = fp.parent.name
            final_rows.append(d)
        except Exception:
            pass
    if final_rows:
        final = pd.concat(final_rows, ignore_index=True)
        if "pred" in final.columns and "id" in final.columns:
            final["pred"] = numeric(final["pred"])
            key_cols = [c for c in final.columns if c not in {"pred", "eq_emb"}]
            info_cols = [c for c in [
                "material", "idx", "element", "dopant", "family_base", "family_base_miller",
                "miller", "site_type", "anchor_count", "slab_formula", "target", "h_ads_excel",
                "target_computed", "target_mismatch_excel_minus_computed"
            ] if c in final.columns]
            info = final.groupby("id", dropna=False)[info_cols].first().reset_index()
            stat = final.groupby("id", dropna=False)["pred"].agg(
                final_pred_median="median",
                final_pred_mean="mean",
                final_pred_std="std",
                final_pred_min="min",
                final_pred_max="max",
                final_n_runs="count",
            ).reset_index()
            fbyid = info.merge(stat, on="id", how="right")
            save = exp_dir / "final_all_addH_out_by_id.csv"
            fbyid.to_csv(save, index=False)
            fm = addhout_metrics(save, pred_col="final_pred_median")
            row.update({f"final_{k}": v for k, v in fm.items()})

    return row


def rank_score(df: pd.DataFrame) -> pd.Series:
    # Lower is better. Prefer addH-out posterior MAE if true h_ads_excel exists,
    # otherwise rely on robust OOF RMSE and run stability.
    components = []
    weights = []

    def add_rank(col, ascending=True, weight=1.0):
        if col in df.columns and df[col].notna().any():
            components.append(df[col].rank(ascending=ascending, na_option="bottom"))
            weights.append(weight)

    add_rank("oof_rmse", ascending=True, weight=0.35)
    add_rank("oof_mae", ascending=True, weight=0.25)
    add_rank("addhout_vs_h_ads_excel_mae", ascending=True, weight=0.25)
    add_rank("addhout_pred_std_mean", ascending=True, weight=0.10)
    add_rank("runs_used_frac", ascending=False, weight=0.05)

    if not components:
        return pd.Series(np.nan, index=df.index)

    score = sum(w * c for w, c in zip(weights, components)) / sum(weights)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-root", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-xlsx", default=None)
    args = ap.parse_args()

    grid = Path(args.grid_root).resolve()
    rows = []
    for exp_dir in sorted([p for p in grid.iterdir() if p.is_dir()]):
        # skip non-experiment dirs
        if not (exp_dir / "cv_work").exists() and not any(exp_dir.glob("final_work_seed*")):
            continue
        rows.append(collect_one(exp_dir))

    df = pd.DataFrame(rows)
    if df.empty:
        print("[WARN] no experiments found")
        df.to_csv(args.output_csv, index=False)
        return

    df["selection_score"] = rank_score(df)
    df = df.sort_values(["selection_score", "oof_rmse"], ascending=[True, True], na_position="last")

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("[OK] summary csv ->", out_csv)
    print(df.head(20).to_string(index=False))

    if args.output_xlsx:
        out_xlsx = Path(args.output_xlsx)
        try:
            with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="summary")
            print("[OK] summary xlsx ->", out_xlsx)
        except Exception as e:
            print("[WARN] failed to write xlsx:", e)


if __name__ == "__main__":
    main()
