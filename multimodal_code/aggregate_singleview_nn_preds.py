#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_singleview_nn_preds.py

把 singleview_nn_preds 目录下分 fold 的 predictions.csv
自动汇总成：

1) test_pred_oof_ensemble.csv
2) addH_out_pred_ensemble.csv

同时也会输出：
- test_pred_all_runs.csv
- addH_out_pred_all_runs.csv
- nn_pred_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-root", required=True, help="例如 /data/home/terminator/RL/multi-view/singleview_nn_preds")
    ap.add_argument("--test-pattern", default="fold*_test/predictions.csv")
    ap.add_argument("--out-pattern", default="fold*_addhout/predictions.csv")
    ap.add_argument("--ensemble-method", default="mean", choices=["mean", "median"])
    ap.add_argument("--test-output", default="test_pred_oof_ensemble.csv")
    ap.add_argument("--out-output", default="addH_out_pred_ensemble.csv")
    ap.add_argument("--test-all-runs-output", default="test_pred_all_runs.csv")
    ap.add_argument("--out-all-runs-output", default="addH_out_pred_all_runs.csv")
    ap.add_argument("--summary-json", default="nn_pred_summary.json")
    ap.add_argument("--strict-test-target", action="store_true", help="若 test 预测文件中找不到 target 列则报错")
    return ap.parse_args()


def save_json(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_fold_from_path(path: Path) -> Optional[int]:
    s = str(path)
    m = re.search(r"fold[_\-]?(\d+)", s)
    if m:
        return int(m.group(1))
    return None


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def normalize_test_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    id_col = find_col(df, ["id", "ID", "sample_id"])
    pred_col = find_col(df, ["pred", "prediction", "y_pred", "pred_mean"])
    target_col = find_col(df, ["target", "y", "y_true", "label", "truth"])

    if id_col is None:
        raise ValueError(f"{path}: could not find id column")
    if pred_col is None:
        raise ValueError(f"{path}: could not find prediction column")

    out = pd.DataFrame({
        "id": df[id_col].astype(str),
        "pred": pd.to_numeric(df[pred_col], errors="coerce"),
    })

    if target_col is not None:
        out["target"] = pd.to_numeric(df[target_col], errors="coerce")
    else:
        out["target"] = np.nan

    out["fold"] = parse_fold_from_path(path)
    out["source_file"] = str(path)
    out = out.dropna(subset=["pred"])
    return out


def normalize_out_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    id_col = find_col(df, ["id", "ID", "sample_id"])
    pred_col = find_col(df, ["pred", "prediction", "y_pred", "pred_mean"])

    if id_col is None:
        raise ValueError(f"{path}: could not find id column")
    if pred_col is None:
        raise ValueError(f"{path}: could not find prediction column")

    out = pd.DataFrame({
        "id": df[id_col].astype(str),
        "pred": pd.to_numeric(df[pred_col], errors="coerce"),
    })
    out["fold"] = parse_fold_from_path(path)
    out["source_file"] = str(path)
    out = out.dropna(subset=["pred"])
    return out


def aggregate_test(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if "target" not in df.columns:
        raise ValueError("test dataframe missing target column")

    if method == "mean":
        pred_agg = df.groupby("id", as_index=False)["pred"].mean()
    else:
        pred_agg = df.groupby("id", as_index=False)["pred"].median()

    target_agg = df.groupby("id", as_index=False)["target"].mean()
    out = target_agg.merge(pred_agg, on="id", how="inner")
    return out[["id", "target", "pred"]].copy()


def aggregate_out(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "mean":
        pred_agg = df.groupby("id", as_index=False)["pred"].mean()
    else:
        pred_agg = df.groupby("id", as_index=False)["pred"].median()
    return pred_agg[["id", "pred"]].copy()


def main():
    args = parse_args()
    nn_root = Path(args.nn_root).resolve()
    if not nn_root.exists():
        raise FileNotFoundError(nn_root)

    test_files = sorted(nn_root.glob(args.test_pattern))
    out_files = sorted(nn_root.glob(args.out_pattern))

    if not test_files:
        raise FileNotFoundError(f"No test prediction files matched: {nn_root / args.test_pattern}")
    if not out_files:
        raise FileNotFoundError(f"No addH_out prediction files matched: {nn_root / args.out_pattern}")

    test_parts = [normalize_test_df(p) for p in test_files]
    out_parts = [normalize_out_df(p) for p in out_files]

    test_all = pd.concat(test_parts, axis=0, ignore_index=True)
    out_all = pd.concat(out_parts, axis=0, ignore_index=True)

    if args.strict_test_target and test_all["target"].isna().all():
        raise ValueError("All test prediction files are missing target values")

    test_all.to_csv(nn_root / args.test_all_runs_output, index=False)
    out_all.to_csv(nn_root / args.out_all_runs_output, index=False)

    test_oof = aggregate_test(test_all, method=args.ensemble_method)
    out_ens = aggregate_out(out_all, method=args.ensemble_method)

    test_oof.to_csv(nn_root / args.test_output, index=False)
    out_ens.to_csv(nn_root / args.out_output, index=False)

    summary = {
        "nn_root": str(nn_root),
        "ensemble_method": args.ensemble_method,
        "n_test_files": len(test_files),
        "n_out_files": len(out_files),
        "test_files": [str(p) for p in test_files],
        "out_files": [str(p) for p in out_files],
        "n_test_all_rows": int(len(test_all)),
        "n_out_all_rows": int(len(out_all)),
        "n_test_oof_rows": int(len(test_oof)),
        "n_out_ensemble_rows": int(len(out_ens)),
        "test_missing_target_rows": int(test_all["target"].isna().sum()) if "target" in test_all.columns else None,
    }
    save_json(nn_root / args.summary_json, summary)

    print(f"[OK] test all-runs  -> {nn_root / args.test_all_runs_output}")
    print(f"[OK] out  all-runs  -> {nn_root / args.out_all_runs_output}")
    print(f"[OK] test ensemble  -> {nn_root / args.test_output}")
    print(f"[OK] out  ensemble  -> {nn_root / args.out_output}")
    print(f"[OK] summary json   -> {nn_root / args.summary_json}")


if __name__ == "__main__":
    main()
