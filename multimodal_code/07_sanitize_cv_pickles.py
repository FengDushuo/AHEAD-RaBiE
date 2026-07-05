#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description="Remove stale extreme-target rows from existing multi-view CV pickle files.")
    ap.add_argument("--cv-root", required=True, help="Root directory containing fold_*/ pkl files from 04_make_multiview_data_cv_multimodal.py")
    ap.add_argument("--target-abs-max", type=float, default=20.0, help="Drop rows with abs(target) greater than this value.")
    ap.add_argument("--drop-outlier-flags", action="store_true", help="Also drop rows with outlier_flag_target=True when the column exists.")
    ap.add_argument("--dry-run", action="store_true", help="Only report; do not rewrite pkl files.")
    return ap.parse_args()


def coerce_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return (
        s.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes", "y"})
    )


def sanitize_pickle(path: Path, target_abs_max: Optional[float], drop_outlier_flags: bool, dry_run: bool) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False, "before": 0, "after": 0, "dropped_abs": 0, "dropped_flag": 0, "max_abs_target_before": None, "min_target_before": None, "max_target_before": None}

    df = pd.read_pickle(path)
    before = int(len(df))
    if before == 0 or "target" not in df.columns:
        return {"path": str(path), "exists": True, "before": before, "after": before, "dropped_abs": 0, "dropped_flag": 0, "max_abs_target_before": None, "min_target_before": None, "max_target_before": None}

    y = pd.to_numeric(df["target"], errors="coerce")
    keep = pd.Series(True, index=df.index)
    dropped_abs = 0
    dropped_flag = 0

    max_abs_before = float(y.abs().max()) if y.notna().any() else None
    min_before = float(y.min()) if y.notna().any() else None
    max_before = float(y.max()) if y.notna().any() else None

    if target_abs_max is not None:
        abs_keep = y.notna() & (y.abs() <= float(target_abs_max))
        dropped_abs = int((~abs_keep & keep).sum())
        keep = keep & abs_keep

    if drop_outlier_flags and "outlier_flag_target" in df.columns:
        flag_keep = ~coerce_bool_series(df["outlier_flag_target"])
        dropped_flag = int((~flag_keep & keep).sum())
        keep = keep & flag_keep

    out = df.loc[keep].copy()
    after = int(len(out))
    if after == 0:
        raise ValueError(f"{path}: sanitizing would remove all rows; inspect target values or relax threshold.")

    if after != before and not dry_run:
        backup = path.with_suffix(path.suffix + ".before_sanitize")
        if not backup.exists():
            df.to_pickle(backup)
        out.to_pickle(path)

    return {
        "path": str(path),
        "exists": True,
        "before": before,
        "after": after,
        "dropped_abs": dropped_abs,
        "dropped_flag": dropped_flag,
        "max_abs_target_before": max_abs_before,
        "min_target_before": min_before,
        "max_target_before": max_before,
    }


def main():
    args = parse_args()
    cv_root = Path(args.cv_root).resolve()
    fold_dirs = sorted([p for p in cv_root.iterdir() if p.is_dir() and p.name.startswith("fold_")], key=lambda p: int(p.name.split("_")[-1]))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found under {cv_root}")

    rows = []
    for fold_dir in fold_dirs:
        for name in ["clip_train.pkl", "clip_val.pkl", "regress_train.pkl", "regress_val.pkl", "regress_test.pkl"]:
            rows.append(sanitize_pickle(fold_dir / name, args.target_abs_max, args.drop_outlier_flags, args.dry_run))

    report = pd.DataFrame(rows)
    out_csv = cv_root / ("cv_pickle_sanitize_report.dryrun.csv" if args.dry_run else "cv_pickle_sanitize_report.csv")
    report.to_csv(out_csv, index=False)
    print("[OK] report ->", out_csv)
    print("[INFO] dry_run =", bool(args.dry_run))
    print("[INFO] target_abs_max =", args.target_abs_max)
    print("[INFO] drop_outlier_flags =", bool(args.drop_outlier_flags))
    print("[INFO] total dropped_abs =", int(report["dropped_abs"].sum()))
    print("[INFO] total dropped_flag =", int(report["dropped_flag"].sum()))
    print("[INFO] worst max_abs_target_before =", report["max_abs_target_before"].max())
    print(report.sort_values(["dropped_abs", "max_abs_target_before"], ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
