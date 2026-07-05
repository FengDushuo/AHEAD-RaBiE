#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_addh_master_tables_robust.py

Merge:
  - addH_master_drop8.csv
  - addH_master_ml2.csv

into:
  - addH_master_merged_robust.csv

Typical usage
-------------
python merge_addh_master_tables_robust.py \
  --old-csv addH_master_drop8.csv \
  --new-csv addH_master_ml2.csv \
  --output-csv addH_master_merged_robust.csv \
  --merged-raw-csv addH_master_merged_raw.csv \
  --outlier-method iqr \
  --outlier-action drop \
  --iqr-multiplier 3.0

What it does
------------
1) Read old/new master CSVs
2) Add / normalize a data_source column
3) Align columns and concatenate
4) Deduplicate by id with a simple quality-aware priority
5) Optionally save merged raw table
6) Apply outlier handling on merged target
7) Save final robust merged CSV + optional reports
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-csv", required=True, help="Existing training master table, e.g. addH_master_drop8.csv")
    ap.add_argument("--new-csv", required=True, help="Newly built master table, e.g. addH_master_ml2.csv")
    ap.add_argument("--output-csv", default="addH_master_merged_robust.csv")
    ap.add_argument("--merged-raw-csv", default="addH_master_merged_raw.csv", help="Optional merged raw CSV before robust filtering")
    ap.add_argument("--old-source-name", default="orig_addH")
    ap.add_argument("--new-source-name", default="ml2_addH")

    # dedup options
    ap.add_argument(
        "--dedup-policy",
        default="prefer_new",
        choices=["prefer_new", "prefer_old", "quality_then_new", "quality_then_old"],
        help="How to choose among duplicate ids."
    )

    # outlier handling
    ap.add_argument("--outlier-method", default="iqr", choices=["none", "iqr", "id_list"])
    ap.add_argument("--outlier-action", default="drop", choices=["flag", "drop"])
    ap.add_argument("--iqr-multiplier", type=float, default=3.0)
    ap.add_argument("--drop-ids", default="")
    ap.add_argument("--outlier-report-csv", default="addH_master_merged_outlier_report.csv")
    ap.add_argument("--outlier-summary-json", default="addH_master_merged_outlier_summary.json")
    return ap.parse_args()


def _split_drop_ids(raw: str) -> List[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _usable_target_mask(df: pd.DataFrame) -> pd.Series:
    mask = df["target"].notna()
    if "parse_ok" in df.columns:
        mask = mask & df["parse_ok"].fillna(False)
    return mask


def _iqr_outlier_mask(y: pd.Series, iqr_multiplier: float) -> Tuple[pd.Series, Dict[str, float]]:
    q1 = float(y.quantile(0.25))
    q3 = float(y.quantile(0.75))
    iqr = q3 - q1
    low = q1 - iqr_multiplier * iqr
    high = q3 + iqr_multiplier * iqr
    mask = (y < low) | (y > high)
    meta = {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "low_bound": low,
        "high_bound": high,
        "iqr_multiplier": float(iqr_multiplier),
    }
    return mask, meta


def apply_outlier_handling(
    df: pd.DataFrame,
    outlier_method: str,
    outlier_action: str,
    iqr_multiplier: float,
    drop_ids: List[str],
    report_csv: Optional[Path],
    summary_json: Optional[Path],
) -> pd.DataFrame:
    df = df.copy()
    df["outlier_flag_target"] = False
    df["outlier_reason_target"] = ""

    usable_mask = _usable_target_mask(df)
    usable = df.loc[usable_mask].copy()

    detection_meta: Dict[str, object] = {}
    outlier_index = pd.Index([], dtype=int)

    if outlier_method == "none":
        pass
    elif outlier_method == "id_list":
        if not drop_ids:
            raise ValueError("outlier_method=id_list requires non-empty --drop-ids")
        id_mask = usable["id"].astype(str).isin(drop_ids)
        outlier_index = usable.loc[id_mask].index
        detection_meta = {"drop_ids": drop_ids}
    elif outlier_method == "iqr":
        if usable.empty:
            raise ValueError("No usable rows available for IQR outlier detection")
        out_mask, detection_meta = _iqr_outlier_mask(usable["target"].astype(float), iqr_multiplier)
        outlier_index = usable.loc[out_mask].index
    else:
        raise ValueError(f"Unsupported outlier_method: {outlier_method}")

    if len(outlier_index) > 0:
        df.loc[outlier_index, "outlier_flag_target"] = True
        reason = outlier_method if outlier_method != "id_list" else "id_list"
        df.loc[outlier_index, "outlier_reason_target"] = reason

    if report_csv is not None:
        report_cols = [c for c in [
            "id", "family_base", "family_base_miller", "dopant", "data_source",
            "target", "outlier_flag_target", "outlier_reason_target", "notes"
        ] if c in df.columns]
        report_df = df.loc[usable_mask, report_cols].copy()
        report_df["target_abs"] = df.loc[usable_mask, "target"].abs().to_numpy()
        report_df.sort_values(["outlier_flag_target", "target"], ascending=[False, True]).to_csv(report_csv, index=False)

    summary = {
        "outlier_method": outlier_method,
        "outlier_action": outlier_action,
        "n_total_rows_before": int(len(df)),
        "n_usable_rows": int(usable_mask.sum()),
        "n_outliers": int(df["outlier_flag_target"].sum()),
        "drop_ids": drop_ids,
        "detection_meta": detection_meta,
    }

    if outlier_action == "drop":
        df = df.loc[~df["outlier_flag_target"]].copy()
    elif outlier_action == "flag":
        pass
    else:
        raise ValueError(f"Unsupported outlier_action: {outlier_action}")

    summary["n_total_rows_after"] = int(len(df))

    if summary_json is not None:
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] outlier_method={outlier_method} outlier_action={outlier_action}")
    print(f"[INFO] outliers flagged={summary['n_outliers']} rows_before={summary['n_total_rows_before']} rows_after={summary['n_total_rows_after']}")
    if report_csv is not None:
        print(f"[OK] outlier report saved -> {report_csv}")
    if summary_json is not None:
        print(f"[OK] outlier summary saved -> {summary_json}")

    return df


def normalize_source(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()
    if "data_source" not in df.columns:
        df["data_source"] = source_name
    else:
        df["data_source"] = df["data_source"].fillna(source_name).replace("", source_name)
    return df


def compute_quality_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    if "parse_ok" in df.columns:
        score += df["parse_ok"].fillna(False).astype(float) * 100.0
    if "target" in df.columns:
        score += df["target"].notna().astype(float) * 20.0
    if "text" in df.columns:
        score += df["text"].notna().astype(float) * 10.0
    if "contcar_path" in df.columns:
        score += df["contcar_path"].notna().astype(float) * 5.0
    if "vasprun_path" in df.columns:
        score += df["vasprun_path"].notna().astype(float) * 2.0
    if "oszicar_path" in df.columns:
        score += df["oszicar_path"].notna().astype(float) * 1.0
    return score


def deduplicate_by_id(df: pd.DataFrame, dedup_policy: str, old_source_name: str, new_source_name: str) -> pd.DataFrame:
    if "id" not in df.columns:
        raise ValueError("Merged dataframe has no 'id' column; cannot deduplicate.")

    df = df.copy()
    df["_quality_score"] = compute_quality_score(df)

    if dedup_policy == "prefer_new":
        source_rank = df["data_source"].map({new_source_name: 1, old_source_name: 0}).fillna(0)
        df["_source_rank"] = source_rank
        df = df.sort_values(["id", "_source_rank", "_quality_score"], ascending=[True, False, False])
    elif dedup_policy == "prefer_old":
        source_rank = df["data_source"].map({old_source_name: 1, new_source_name: 0}).fillna(0)
        df["_source_rank"] = source_rank
        df = df.sort_values(["id", "_source_rank", "_quality_score"], ascending=[True, False, False])
    elif dedup_policy == "quality_then_new":
        source_rank = df["data_source"].map({new_source_name: 1, old_source_name: 0}).fillna(0)
        df["_source_rank"] = source_rank
        df = df.sort_values(["id", "_quality_score", "_source_rank"], ascending=[True, False, False])
    elif dedup_policy == "quality_then_old":
        source_rank = df["data_source"].map({old_source_name: 1, new_source_name: 0}).fillna(0)
        df["_source_rank"] = source_rank
        df = df.sort_values(["id", "_quality_score", "_source_rank"], ascending=[True, False, False])
    else:
        raise ValueError(f"Unsupported dedup_policy: {dedup_policy}")

    deduped = df.drop_duplicates(subset=["id"], keep="first").copy()
    deduped = deduped.drop(columns=[c for c in ["_quality_score", "_source_rank"] if c in deduped.columns], errors="ignore")
    return deduped


def main():
    args = parse_args()

    old_csv = Path(args.old_csv).resolve()
    new_csv = Path(args.new_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    merged_raw_csv = Path(args.merged_raw_csv).resolve() if args.merged_raw_csv else None
    report_csv = Path(args.outlier_report_csv).resolve() if args.outlier_report_csv else None
    summary_json = Path(args.outlier_summary_json).resolve() if args.outlier_summary_json else None

    if not old_csv.exists():
        raise FileNotFoundError(old_csv)
    if not new_csv.exists():
        raise FileNotFoundError(new_csv)

    old_df = pd.read_csv(old_csv)
    new_df = pd.read_csv(new_csv)

    old_df = normalize_source(old_df, args.old_source_name)
    new_df = normalize_source(new_df, args.new_source_name)

    old_n = len(old_df)
    new_n = len(new_df)

    # align columns
    all_cols = sorted(set(old_df.columns) | set(new_df.columns))
    old_df = old_df.reindex(columns=all_cols)
    new_df = new_df.reindex(columns=all_cols)

    merged = pd.concat([old_df, new_df], ignore_index=True)
    merged_n_before_dedup = len(merged)

    deduped = deduplicate_by_id(
        df=merged,
        dedup_policy=args.dedup_policy,
        old_source_name=args.old_source_name,
        new_source_name=args.new_source_name,
    )
    deduped_n = len(deduped)

    if merged_raw_csv is not None:
        deduped.to_csv(merged_raw_csv, index=False)
        print(f"[OK] merged raw saved -> {merged_raw_csv}")

    robust = apply_outlier_handling(
        df=deduped,
        outlier_method=args.outlier_method,
        outlier_action=args.outlier_action,
        iqr_multiplier=args.iqr_multiplier,
        drop_ids=_split_drop_ids(args.drop_ids),
        report_csv=report_csv,
        summary_json=summary_json,
    )

    robust.to_csv(output_csv, index=False)

    print(f"[OK] robust merged saved -> {output_csv}")
    print(f"[INFO] old rows               = {old_n}")
    print(f"[INFO] new rows               = {new_n}")
    print(f"[INFO] merged rows pre-dedup  = {merged_n_before_dedup}")
    print(f"[INFO] rows after dedup       = {deduped_n}")
    print(f"[INFO] rows after robust step = {len(robust)}")

    if "data_source" in robust.columns:
        counts = robust["data_source"].fillna("unknown").value_counts(dropna=False)
        print("[INFO] final data_source counts:")
        print(counts.to_string())

    if "family_base" in robust.columns:
        print(f"[INFO] unique family_base = {robust['family_base'].nunique(dropna=True)}")
    if "dopant" in robust.columns:
        print(f"[INFO] unique dopant = {robust['dopant'].nunique(dropna=True)}")
    if "parse_ok" in robust.columns:
        print(f"[INFO] parse_ok count = {int(robust['parse_ok'].fillna(False).sum())}")
    if "target" in robust.columns:
        print(f"[INFO] target non-null count = {int(robust['target'].notna().sum())}")


if __name__ == "__main__":
    main()
