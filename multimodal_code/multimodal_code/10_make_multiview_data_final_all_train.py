#!/usr/bin/env python3
from __future__ import annotations
"""Build a production final-all-training multi-view fold for addH-out prediction.

This script uses all clean addH/addH-2 rows for model development, keeping only a
small target-stratified group validation subset for calibration/early reliability.
It does NOT create an outer test split. Use it after OOF/CV diagnostics are done.

It dynamically reuses helpers from 04_make_multiview_data_cv_multimodal.py so the
schema, optional columns, addH-out SSL handling, and outlier filtering remain
consistent with the CV pipeline.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd


def _load_04_module():
    path = Path(__file__).with_name("04_make_multiview_data_cv_multimodal.py")
    if not path.exists():
        raise FileNotFoundError(f"Required helper file not found next to this script: {path}")
    spec = importlib.util.spec_from_file_location("mv04_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    ap = argparse.ArgumentParser(description="Build final all-clean train/val data for production addH-out prediction.")
    ap.add_argument("--addh-master-csv", required=True)
    ap.add_argument("--eq-emb-pkl", required=True)
    ap.add_argument("--addhout-master-csv", required=True)
    ap.add_argument("--addhout-eq-emb-pkl", required=True)
    ap.add_argument("--out-dir", required=True, help="Output CV-like root; will write fold_0/*.pkl")
    ap.add_argument("--group-col", default="family_base_miller")
    ap.add_argument("--val-frac", type=float, default=0.15, help="Fraction of groups held out for validation/calibration.")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--include-regress-eq-emb", action="store_true")
    ap.add_argument("--include-addhout-in-clip", action="store_true")
    ap.add_argument("--exclude-target-outliers", action="store_true")
    ap.add_argument("--target-abs-max", type=float, default=20.0)
    ap.add_argument("--stratify-bins", type=int, default=5)
    ap.add_argument("--stratify-target-col", default="target")
    ap.add_argument(
        "--extra-regress-cols",
        default="text_structured,text_raw,family_base,family_base_miller,dopant,miller,site_type,anchor_count,slab_formula,data_source,w_domain,outlier_flag_target,outlier_reason_target",
    )
    ap.add_argument(
        "--extra-clip-cols",
        default="text_structured,text_raw,family_base,family_base_miller,dopant,miller,site_type,anchor_count,slab_formula,data_source",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    h = _load_04_module()

    out_root = Path(args.out_dir)
    fold_dir = out_root / "fold_0"
    fold_dir.mkdir(parents=True, exist_ok=True)

    addh = pd.read_csv(args.addh_master_csv)
    emb = h._load_eq_emb(Path(args.eq_emb_pkl))
    usable_addh = h._filter_addh(
        addh,
        emb,
        args.group_col,
        exclude_target_outliers=args.exclude_target_outliers,
        target_abs_max=args.target_abs_max,
    )
    usable_addh[args.group_col] = usable_addh[args.group_col].astype(str)

    addhout = pd.read_csv(args.addhout_master_csv)
    addhout_emb = h._load_eq_emb(Path(args.addhout_eq_emb_pkl))
    addhout_usable = h._filter_addhout(
        addhout,
        addhout_emb,
        include_regress_eq_emb=args.include_regress_eq_emb,
        include_addhout_in_clip=args.include_addhout_in_clip,
    )

    extra_regress_cols = h._parse_csv_list(args.extra_regress_cols)
    extra_clip_cols = h._parse_csv_list(args.extra_clip_cols)

    train_groups, val_groups = h._choose_val_groups_from_train(
        outer_train_df=usable_addh,
        group_col=args.group_col,
        val_mode="stratified_group_holdout",
        val_frac=args.val_frac,
        seed=args.seed,
        outer_fold_idx=0,
        stratify_bins=args.stratify_bins,
        stratify_target_col=args.stratify_target_col,
    )

    split_df = usable_addh.copy()
    split_df["split"] = np.where(split_df[args.group_col].isin(val_groups), "val", "train")
    split_df["outer_fold"] = 0
    split_df["cv_mode"] = "final_all_train"
    split_df.to_csv(fold_dir / "addH_split_assignments.csv", index=False)

    clip_mandatory = ["id", "text", "target", "eq_emb"]
    clip_train = h._select_with_optional(split_df[split_df["split"] == "train"], clip_mandatory, extra_clip_cols)
    clip_val = h._select_with_optional(split_df[split_df["split"] == "val"], clip_mandatory, extra_clip_cols)

    n_addhout_clip_train = 0
    if args.include_addhout_in_clip:
        addhout_clip = h._prepare_clip_addhout(addhout_usable, optional_cols=extra_clip_cols)
        common_cols = h._unique_preserve_order(list(clip_train.columns) + list(addhout_clip.columns))
        for c in common_cols:
            if c not in clip_train.columns:
                clip_train[c] = np.nan
            if c not in addhout_clip.columns:
                addhout_clip[c] = np.nan
        clip_train = pd.concat([clip_train[common_cols], addhout_clip[common_cols]], ignore_index=True)
        n_addhout_clip_train = int(len(addhout_clip))

    regress_mandatory = ["id", "text", "target"] + (["eq_emb"] if args.include_regress_eq_emb else [])
    regress_train = h._select_with_optional(split_df[split_df["split"] == "train"], regress_mandatory, extra_regress_cols)
    regress_val = h._select_with_optional(split_df[split_df["split"] == "val"], regress_mandatory, extra_regress_cols)

    addhout_pred_mandatory = ["id", "text"] + (["eq_emb"] if args.include_regress_eq_emb else [])
    addhout_pred = h._select_with_optional(addhout_usable, addhout_pred_mandatory, extra_regress_cols)
    addhout_pred["target"] = 0.0
    ordered_cols = h._existing_cols(addhout_pred, ["id", "text"] + (["eq_emb"] if args.include_regress_eq_emb else []) + ["target"] + extra_regress_cols)
    addhout_pred = addhout_pred[ordered_cols]

    clip_train.to_pickle(fold_dir / "clip_train.pkl")
    clip_val.to_pickle(fold_dir / "clip_val.pkl")
    regress_train.to_pickle(fold_dir / "regress_train.pkl")
    regress_val.to_pickle(fold_dir / "regress_val.pkl")
    # A small placeholder keeps generic tooling happy; do not use --run-predict-test for this final fold.
    regress_val.head(0).to_pickle(fold_dir / "regress_test.pkl")
    addhout_pred.to_pickle(fold_dir / "addH_out_pred_input.pkl")
    addhout_usable.to_csv(fold_dir / "addH_out_pred_manifest.csv", index=False)

    train_stats = h._split_target_stats(split_df, args.group_col, train_groups, target_col=args.stratify_target_col)
    val_stats = h._split_target_stats(split_df, args.group_col, val_groups, target_col=args.stratify_target_col)
    summary = pd.DataFrame([{
        "fold": 0,
        "cv_mode": "final_all_train",
        "group_col": args.group_col,
        "train_groups": ",".join(map(str, sorted(train_groups))),
        "val_groups": ",".join(map(str, sorted(val_groups))),
        "n_train": int(len(regress_train)),
        "n_val": int(len(regress_val)),
        "n_addhout_pred": int(len(addhout_pred)),
        "n_addhout_clip_train": int(n_addhout_clip_train),
        "train_target_min": train_stats["target_min"],
        "train_target_max": train_stats["target_max"],
        "train_target_median": train_stats["target_median"],
        "val_target_min": val_stats["target_min"],
        "val_target_max": val_stats["target_max"],
        "val_target_median": val_stats["target_median"],
        "exclude_target_outliers": bool(args.exclude_target_outliers),
        "target_abs_max": float(args.target_abs_max) if args.target_abs_max is not None else None,
        "include_regress_eq_emb": bool(args.include_regress_eq_emb),
        "include_addhout_in_clip": bool(args.include_addhout_in_clip),
    }])
    summary.to_csv(out_root / "cv_summary.csv", index=False)

    meta = summary.iloc[0].to_dict()
    meta["note"] = "Final production fold: all clean source rows used except a small target-stratified group validation subset; no outer test split."
    with open(fold_dir / "fold_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("[OK] final all-train fold ->", fold_dir)
    print("[INFO] train rows =", len(regress_train), "val rows =", len(regress_val), "addH-out rows =", len(addhout_pred))
    print("[INFO] cv_summary ->", out_root / "cv_summary.csv")


if __name__ == "__main__":
    main()
