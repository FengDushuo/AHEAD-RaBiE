#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect target outliers in addH_master.csv and generate a robust subset.

Default behavior
----------------
1) Load addH_master.csv
2) Focus on rows with non-null target; by default, also require parse_ok=True if the column exists
3) Detect outliers with an IQR rule on target
4) Save:
   - addH_master_outlier_report.csv
   - addH_master_robust.csv
   - addH_master_outlier_summary.json

Recommended first run
---------------------
python inspect_target_outliers_make_robust.py \
  --input-csv addH_master.csv \
  --output-csv addH_master_robust.csv

Useful alternatives
-------------------
# Stronger filtering with IQR
python inspect_target_outliers_make_robust.py \
  --input-csv addH_master.csv \
  --method iqr \
  --iqr-multiplier 2.5

# MAD-based robust filtering
python inspect_target_outliers_make_robust.py \
  --input-csv addH_master.csv \
  --method mad \
  --mad-z-threshold 4.5

# Clip target values instead of dropping rows
python inspect_target_outliers_make_robust.py \
  --input-csv addH_master.csv \
  --action clip \
  --method iqr

# Only inspect, do not drop or clip
python inspect_target_outliers_make_robust.py \
  --input-csv addH_master.csv \
  --action report_only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Inspect target outliers and build a robust addH_master.csv")
    ap.add_argument("--input-csv", required=True, help="Path to addH_master.csv")
    ap.add_argument("--output-csv", default="addH_master_robust.csv", help="Output robust CSV path")
    ap.add_argument(
        "--report-csv",
        default="addH_master_outlier_report.csv",
        help="Per-row outlier report CSV path",
    )
    ap.add_argument(
        "--summary-json",
        default="addH_master_outlier_summary.json",
        help="Summary JSON path",
    )
    ap.add_argument("--target-col", default="target", help="Target column name")
    ap.add_argument(
        "--id-col",
        default="id",
        help="ID column name used in report if present",
    )
    ap.add_argument(
        "--method",
        default="iqr",
        choices=["iqr", "mad", "quantile", "abs"],
        help="Outlier detection method",
    )
    ap.add_argument(
        "--action",
        default="drop",
        choices=["drop", "clip", "report_only"],
        help="How to create the robust output",
    )
    ap.add_argument(
        "--iqr-multiplier",
        type=float,
        default=3.0,
        help="IQR multiplier for lower/upper fences. 3.0 is conservative.",
    )
    ap.add_argument(
        "--mad-z-threshold",
        type=float,
        default=4.5,
        help="Modified z-score threshold for MAD method",
    )
    ap.add_argument(
        "--quantile-low",
        type=float,
        default=0.01,
        help="Lower quantile threshold for quantile method",
    )
    ap.add_argument(
        "--quantile-high",
        type=float,
        default=0.99,
        help="Upper quantile threshold for quantile method",
    )
    ap.add_argument(
        "--abs-low",
        type=float,
        default=None,
        help="Absolute lower bound for abs method",
    )
    ap.add_argument(
        "--abs-high",
        type=float,
        default=None,
        help="Absolute upper bound for abs method",
    )
    ap.add_argument(
        "--require-parse-ok",
        action="store_true",
        help="If set, require parse_ok=True when that column exists. Default behavior already does this if parse_ok exists.",
    )
    ap.add_argument(
        "--keep-all-columns",
        action="store_true",
        help="Keep all original columns in robust CSV. Default is also to keep all columns; this flag is kept for readability.",
    )
    return ap.parse_args()


def choose_usable_rows(df: pd.DataFrame, target_col: str, require_parse_ok: bool) -> pd.Series:
    mask = df[target_col].notna()
    if "parse_ok" in df.columns:
        if require_parse_ok or True:
            mask = mask & df["parse_ok"].fillna(False)
    return mask


def detect_iqr(y: pd.Series, multiplier: float) -> Tuple[pd.Series, Dict[str, float]]:
    q1 = float(y.quantile(0.25))
    q3 = float(y.quantile(0.75))
    iqr = q3 - q1
    low = q1 - multiplier * iqr
    high = q3 + multiplier * iqr
    is_outlier = (y < low) | (y > high)
    return is_outlier, {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "low_bound": low,
        "high_bound": high,
    }


def detect_mad(y: pd.Series, threshold: float) -> Tuple[pd.Series, Dict[str, float]]:
    med = float(np.median(y))
    abs_dev = np.abs(y - med)
    mad = float(np.median(abs_dev))
    if mad == 0:
        modified_z = pd.Series(np.zeros(len(y)), index=y.index, dtype=float)
    else:
        modified_z = 0.6745 * (y - med) / mad
    is_outlier = modified_z.abs() > threshold
    return is_outlier, {
        "median": med,
        "mad": mad,
        "mad_z_threshold": threshold,
    }


def detect_quantile(y: pd.Series, q_low: float, q_high: float) -> Tuple[pd.Series, Dict[str, float]]:
    low = float(y.quantile(q_low))
    high = float(y.quantile(q_high))
    is_outlier = (y < low) | (y > high)
    return is_outlier, {
        "quantile_low": q_low,
        "quantile_high": q_high,
        "low_bound": low,
        "high_bound": high,
    }


def detect_abs(y: pd.Series, low: float | None, high: float | None) -> Tuple[pd.Series, Dict[str, float | None]]:
    if low is None and high is None:
        raise ValueError("For method=abs, provide at least one of --abs-low or --abs-high")
    is_outlier = pd.Series(False, index=y.index)
    if low is not None:
        is_outlier = is_outlier | (y < low)
    if high is not None:
        is_outlier = is_outlier | (y > high)
    return is_outlier, {
        "low_bound": low,
        "high_bound": high,
    }


def clip_by_bounds(y: pd.Series, low: float | None, high: float | None) -> pd.Series:
    if low is None and high is None:
        return y.copy()
    return y.clip(lower=low, upper=high)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    report_csv = Path(args.report_csv).resolve()
    summary_json = Path(args.summary_json).resolve()

    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    df = pd.read_csv(input_csv)
    if args.target_col not in df.columns:
        raise ValueError(f"Missing target column: {args.target_col}")

    usable_mask = choose_usable_rows(df, args.target_col, args.require_parse_ok)
    usable = df.loc[usable_mask].copy()
    if usable.empty:
        raise ValueError("No usable rows remain after filtering target/parse_ok")

    y = usable[args.target_col].astype(float)

    if args.method == "iqr":
        outlier_mask, meta = detect_iqr(y, args.iqr_multiplier)
        low_bound = meta["low_bound"]
        high_bound = meta["high_bound"]
    elif args.method == "mad":
        outlier_mask, meta = detect_mad(y, args.mad_z_threshold)
        low_bound = None
        high_bound = None
    elif args.method == "quantile":
        outlier_mask, meta = detect_quantile(y, args.quantile_low, args.quantile_high)
        low_bound = meta["low_bound"]
        high_bound = meta["high_bound"]
    elif args.method == "abs":
        outlier_mask, meta = detect_abs(y, args.abs_low, args.abs_high)
        low_bound = meta.get("low_bound")
        high_bound = meta.get("high_bound")
    else:
        raise ValueError(args.method)

    usable = usable.copy()
    usable["is_outlier"] = outlier_mask.to_numpy()

    # Optional diagnostic scores
    med = float(np.median(y))
    mad = float(np.median(np.abs(y - med)))
    usable["target_abs"] = usable[args.target_col].abs()
    if mad == 0:
        usable["target_modified_z"] = 0.0
    else:
        usable["target_modified_z"] = 0.6745 * (usable[args.target_col] - med) / mad

    report_cols = [c for c in [args.id_col, "family_base", "family_base_miller", "dopant", args.target_col, "is_outlier", "target_abs", "target_modified_z", "notes"] if c in usable.columns]
    usable.sort_values([ "is_outlier", args.target_col ], ascending=[False, True]).to_csv(report_csv, index=False)

    # Build robust output
    robust = df.copy()
    robust["outlier_flag_target"] = False
    robust.loc[usable.index, "outlier_flag_target"] = usable["is_outlier"].to_numpy()

    if args.action == "drop":
        robust = robust.loc[~robust["outlier_flag_target"]].copy()
    elif args.action == "clip":
        if args.method in ("iqr", "quantile", "abs"):
            robust.loc[usable.index, args.target_col] = clip_by_bounds(
                usable[args.target_col].astype(float), low_bound, high_bound
            ).to_numpy()
        elif args.method == "mad":
            # MAD method has no natural hard bounds; keep original values and only flag
            pass
    elif args.action == "report_only":
        pass
    else:
        raise ValueError(args.action)

    robust.to_csv(output_csv, index=False)

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "report_csv": str(report_csv),
        "summary_json": str(summary_json),
        "target_col": args.target_col,
        "method": args.method,
        "action": args.action,
        "n_total_rows": int(len(df)),
        "n_usable_rows": int(len(usable)),
        "n_outliers": int(usable["is_outlier"].sum()),
        "outlier_fraction_usable": float(usable["is_outlier"].mean()),
        "target_min_usable": float(y.min()),
        "target_max_usable": float(y.max()),
        "target_mean_usable": float(y.mean()),
        "target_std_usable": float(y.std(ddof=1)) if len(y) > 1 else 0.0,
        "n_rows_in_robust_csv": int(len(robust)),
        "detection_meta": meta,
        "top_outlier_examples": usable.loc[usable["is_outlier"], report_cols].head(20).to_dict(orient="records"),
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[OK] input          ->", input_csv)
    print("[OK] report         ->", report_csv)
    print("[OK] robust csv     ->", output_csv)
    print("[OK] summary json   ->", summary_json)
    print("[INFO] usable rows  =", len(usable))
    print("[INFO] outliers     =", int(usable["is_outlier"].sum()))
    print("[INFO] action       =", args.action)
    if args.method in ("iqr", "quantile", "abs"):
        print("[INFO] low/high     =", low_bound, high_bound)


if __name__ == "__main__":
    main()
