#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose extreme prediction errors from multi-view CV outputs.

Reads per-run test_metrics.csv files produced by 05_run_*.py and writes:
  - prediction_run_stability_summary.csv
  - prediction_extreme_pairs.csv

This script does not modify any training/prediction files.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True, help="multiview_cv_mm_full_fixclip directory")
    ap.add_argument("--topn", type=int, default=80)
    ap.add_argument("--pred-abs-warn", type=float, default=30.0)
    ap.add_argument("--err-warn", type=float, default=20.0)
    return ap.parse_args()


def rmse(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.sqrt(np.mean((p - y) ** 2))) if len(y) else np.nan


def mae(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean(np.abs(p - y))) if len(y) else np.nan


def parse_run_tag(p: Path) -> tuple[str, int | None, int | None]:
    run_tag = p.parent.name
    m = re.search(r"fold(\d+)_seed(\d+)", run_tag)
    if m:
        return run_tag, int(m.group(1)), int(m.group(2))
    return run_tag, None, None


def main():
    args = parse_args()
    work = Path(args.work_dir).resolve()
    if not work.exists():
        raise FileNotFoundError(work)

    files = sorted(work.glob("fold*_seed*/test_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No fold*_seed*/test_metrics.csv under {work}")

    pair_parts = []
    summary_rows = []
    keep_meta_cols = [
        "id", "target", "pred", "abs_err", "run_tag", "fold", "seed",
        "family_base", "family_base_miller", "data_source", "dopant", "miller",
        "site_type", "anchor_count", "slab_formula", "contcar_path", "bare_contcar_path", "text",
    ]

    for p in files:
        run_tag, fold, seed = parse_run_tag(p)
        df = pd.read_csv(p)
        if not {"id", "target", "pred"}.issubset(df.columns):
            print(f"[WARN] skip {p}: missing id/target/pred")
            continue
        df = df.copy()
        df["run_tag"] = run_tag
        df["fold"] = fold
        df["seed"] = seed
        df["target"] = pd.to_numeric(df["target"], errors="coerce")
        df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
        df = df[np.isfinite(df["target"]) & np.isfinite(df["pred"])]
        df["abs_err"] = (df["pred"] - df["target"]).abs()
        pair_parts.append(df[[c for c in keep_meta_cols if c in df.columns]].copy())
        summary_rows.append({
            "run_tag": run_tag,
            "fold": fold,
            "seed": seed,
            "n": int(len(df)),
            "mae": mae(df["target"], df["pred"]),
            "rmse": rmse(df["target"], df["pred"]),
            "median_abs_err": float(df["abs_err"].median()) if len(df) else np.nan,
            "p90_abs_err": float(df["abs_err"].quantile(0.90)) if len(df) else np.nan,
            "p95_abs_err": float(df["abs_err"].quantile(0.95)) if len(df) else np.nan,
            "max_abs_err": float(df["abs_err"].max()) if len(df) else np.nan,
            "max_abs_pred": float(df["pred"].abs().max()) if len(df) else np.nan,
            f"n_abs_err_gt_{args.err_warn:g}": int((df["abs_err"] > args.err_warn).sum()),
            f"n_abs_pred_gt_{args.pred_abs_warn:g}": int((df["pred"].abs() > args.pred_abs_warn).sum()),
        })

    pairs = pd.concat(pair_parts, ignore_index=True) if pair_parts else pd.DataFrame()
    summary = pd.DataFrame(summary_rows).sort_values(["rmse", "max_abs_pred"], ascending=False)

    summary_path = work / "prediction_run_stability_summary.csv"
    pairs_path = work / "prediction_extreme_pairs.csv"
    summary.to_csv(summary_path, index=False)
    pairs.sort_values("abs_err", ascending=False).head(int(args.topn)).to_csv(pairs_path, index=False)

    print(f"[OK] summary -> {summary_path}")
    print(f"[OK] top pairs -> {pairs_path}")
    print("\n[TOP unstable runs]")
    with pd.option_context("display.max_columns", 50, "display.width", 180):
        print(summary.head(20).to_string(index=False))
    print("\n[TOP extreme prediction pairs]")
    show_cols = [c for c in ["run_tag", "fold", "seed", "id", "target", "pred", "abs_err", "family_base_miller", "data_source", "contcar_path"] if c in pairs.columns]
    with pd.option_context("display.max_columns", 50, "display.width", 220):
        print(pairs.sort_values("abs_err", ascending=False).head(int(args.topn))[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
