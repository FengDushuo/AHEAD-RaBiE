#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate addH-out per-run predictions into one row per id.

Usage:
  python 12_make_addhout_by_id.py \
    --input-csv addH_out_pred_ensemble_robust.csv \
    --output-csv addH_out_pred_ensemble_robust_final_by_id.csv \
    --clip-abs 10
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--clip-abs", type=float, default=None)
    args = ap.parse_args()

    p = Path(args.input_csv)
    df = pd.read_csv(p)
    df = df.drop(columns=["eq_emb"], errors="ignore")

    if "pred" not in df.columns:
        raise ValueError(f"No pred column found in {p}")
    if "id" not in df.columns:
        raise ValueError(f"No id column found in {p}")

    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    if args.clip_abs is not None:
        a = float(args.clip_abs)
        df["pred"] = df["pred"].clip(-a, a)

    key_cols = [
        "material", "idx", "element", "dopant",
        "family_base", "family_base_miller", "miller",
        "site_type", "anchor_count", "slab_formula",
        "target", "h_ads_excel", "target_computed",
        "target_mismatch_excel_minus_computed",
        "contcar_path", "bare_contcar_path", "cif_path"
    ]
    key_cols = [c for c in key_cols if c in df.columns]

    info = df.groupby("id", dropna=False)[key_cols].first().reset_index()
    stat = (
        df.groupby("id", dropna=False)["pred"]
          .agg(
              pred_median="median",
              pred_mean="mean",
              pred_std="std",
              pred_min="min",
              pred_max="max",
              n_runs="count",
          )
          .reset_index()
    )

    out = info.merge(stat, on="id", how="right")

    for true_col in ["h_ads_excel", "target", "target_computed"]:
        if true_col in out.columns:
            out[true_col] = pd.to_numeric(out[true_col], errors="coerce")
            out[f"abs_err_vs_{true_col}"] = (out["pred_median"] - out[true_col]).abs()

    out = out.sort_values("pred_median", ascending=True)
    save = Path(args.output_csv)
    save.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(save, index=False)

    print("[OK] saved:", save)
    print("[INFO] rows:", len(out))
    show_cols = [
        "id", "material", "element",
        "pred_median", "pred_mean", "pred_std", "pred_min", "pred_max", "n_runs",
        "h_ads_excel", "target_computed", "abs_err_vs_h_ads_excel"
    ]
    show_cols = [c for c in show_cols if c in out.columns]
    print(out[show_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
