#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_cat_mv_base_and_calibrate.py

Global CatBoost + multiview base builder.

What it does
------------
1) Load CatBoost and multiview OOF / addH_out predictions
2) Fit a GLOBAL two-way base model on OOF overlap:
      pred_base = f(pred_cat, pred_mv)
   with method in {ridge, mean, weighted}
3) Optionally apply:
   - global calibration
   - group/material calibration
4) Export stable per-id base tables for:
   - source OOF ids
   - addH_out ids

Outputs
-------
out-dir/
  base_source_oof.csv
  base_addH_out.csv
  base_model_info.json
  base_oof_metrics_raw.json
  base_oof_metrics_final.json
  base_group_calibration.json
  addH_out_top20_low.csv
  addH_out_top20_high.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-root", required=True)
    ap.add_argument("--cat-root", required=True)
    ap.add_argument("--mv-root", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--base-method", default="ridge", choices=["ridge", "mean", "weighted"])
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--cat-weight", type=float, default=0.60)
    ap.add_argument("--mv-weight", type=float, default=0.40)

    ap.add_argument("--use-global-calibration", action="store_true")
    ap.add_argument("--global-calibration-mode", default="bias_only", choices=["bias_only", "affine"])

    ap.add_argument("--use-group-calibration", action="store_true")
    ap.add_argument("--group-col", default="family_base")
    ap.add_argument("--group-calibration-mode", default="bias_only", choices=["bias_only", "affine"])
    ap.add_argument("--min-group-size", type=int, default=20)

    ap.add_argument("--topk", type=int, default=20)
    return ap.parse_args()


def save_json(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def metrics_from_df(df: pd.DataFrame, pred_col: str = "pred") -> Dict[str, float]:
    y_true = df["target"].to_numpy(dtype=float)
    y_pred = df[pred_col].to_numpy(dtype=float)
    return {
        "n": int(len(df)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fit_calibration(y_pred: np.ndarray, y_true: np.ndarray, mode: str) -> Dict[str, float]:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    finite = np.isfinite(y_pred) & np.isfinite(y_true)
    y_pred = y_pred[finite]
    y_true = y_true[finite]
    if len(y_pred) == 0:
        return {"mode": mode, "a": 1.0, "b": 0.0}
    if mode == "bias_only":
        b = float(np.mean(y_true - y_pred))
        return {"mode": mode, "a": 1.0, "b": b}
    if len(y_pred) < 2 or float(np.std(y_pred)) < 1e-12:
        b = float(np.mean(y_true - y_pred))
        return {"mode": "affine", "a": 1.0, "b": b}
    a, b = np.polyfit(y_pred, y_true, deg=1)
    return {"mode": "affine", "a": float(a), "b": float(b)}


def apply_calibration_array(y_pred: np.ndarray, calib: Dict[str, float]) -> np.ndarray:
    return calib["a"] * np.asarray(y_pred, dtype=float) + calib["b"]


def _aggregate_all_runs_oof(df: pd.DataFrame) -> pd.DataFrame:
    need = {"id", "pred", "target"}
    if not need.issubset(df.columns):
        raise ValueError(f"All-runs OOF file missing columns: {sorted(need - set(df.columns))}")
    return df.groupby("id", as_index=False).agg(target=("target", "mean"), pred=("pred", "mean"))


def _aggregate_all_runs_out(df: pd.DataFrame) -> pd.DataFrame:
    need = {"id", "pred"}
    if not need.issubset(df.columns):
        raise ValueError(f"All-runs out file missing columns: {sorted(need - set(df.columns))}")
    return df.groupby("id", as_index=False).agg(pred=("pred", "mean"))


def _load_model_oof_out(root: Path, label: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    oof_candidates = [
        root / "test_pred_oof_ensemble.csv",
        root / "test_pred_all_runs.csv",
    ]
    out_candidates = [
        root / "addH_out_pred_ensemble.csv",
        root / "addH_out_pred_all_runs.csv",
    ]

    oof_df = None
    for p in oof_candidates:
        if p.exists():
            tmp = pd.read_csv(p)
            if p.name == "test_pred_all_runs.csv":
                tmp = _aggregate_all_runs_oof(tmp)
            else:
                need = {"id", "target", "pred"}
                if not need.issubset(tmp.columns):
                    raise ValueError(f"{p}: missing columns {sorted(need - set(tmp.columns))}")
                tmp = tmp[["id", "target", "pred"]].copy()
            oof_df = tmp
            break
    if oof_df is None:
        raise FileNotFoundError(f"{label}: could not find OOF file under {root}")

    out_df = None
    for p in out_candidates:
        if p.exists():
            tmp = pd.read_csv(p)
            if p.name == "addH_out_pred_all_runs.csv":
                tmp = _aggregate_all_runs_out(tmp)
            else:
                need = {"id", "pred"}
                if not need.issubset(tmp.columns):
                    raise ValueError(f"{p}: missing columns {sorted(need - set(tmp.columns))}")
                tmp = tmp[["id", "pred"]].copy()
            out_df = tmp
            break
    if out_df is None:
        raise FileNotFoundError(f"{label}: could not find addH-out file under {root}")

    oof_df["id"] = oof_df["id"].astype(str)
    out_df["id"] = out_df["id"].astype(str)
    oof_df = oof_df.rename(columns={"pred": f"pred_{label}"})
    out_df = out_df.rename(columns={"pred": f"pred_{label}"})
    return oof_df, out_df


def list_fold_dirs(cv_root: Path) -> List[Path]:
    cands = [p for p in cv_root.iterdir() if p.is_dir() and p.name.startswith("fold_")]
    cands.sort(key=lambda p: int(p.name.split("_")[-1]))
    return cands


def collect_source_meta(cv_root: Path) -> pd.DataFrame:
    fold_dirs = list_fold_dirs(cv_root)
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* dirs under {cv_root}")

    keep_cols_pref = [
        "id", "target", "family_base", "family_base_miller", "site_type",
        "miller", "dopant", "anchor_count", "data_source"
    ]
    parts = []
    for fold_dir in fold_dirs:
        for name in ["nn_train.pkl", "nn_val.pkl", "nn_test.pkl"]:
            p = fold_dir / name
            if not p.exists():
                continue
            df = pd.read_pickle(p)
            cols = [c for c in keep_cols_pref if c in df.columns]
            if "id" not in cols:
                continue
            tmp = df[cols].copy()
            tmp["id"] = tmp["id"].astype(str)
            parts.append(tmp)
    if not parts:
        raise FileNotFoundError("Could not collect source metadata from nn_train/val/test.pkl")
    out = pd.concat(parts, axis=0, ignore_index=True).drop_duplicates(subset=["id"], keep="last")
    return out


def collect_out_meta(cv_root: Path) -> pd.DataFrame:
    fold_dirs = list_fold_dirs(cv_root)
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* dirs under {cv_root}")

    keep_cols_pref = [
        "id", "family_base", "family_base_miller", "site_type",
        "miller", "dopant", "anchor_count", "data_source"
    ]
    parts = []
    for fold_dir in fold_dirs:
        for name in ["addH_out_nn_pred_input.pkl", "addH_out_pred_manifest.csv"]:
            p = fold_dir / name
            if not p.exists():
                continue
            if p.suffix == ".pkl":
                df = pd.read_pickle(p)
            else:
                df = pd.read_csv(p)
            cols = [c for c in keep_cols_pref if c in df.columns]
            if "id" not in cols:
                continue
            tmp = df[cols].copy()
            tmp["id"] = tmp["id"].astype(str)
            parts.append(tmp)
    if not parts:
        raise FileNotFoundError("Could not collect addH_out metadata")
    out = pd.concat(parts, axis=0, ignore_index=True).drop_duplicates(subset=["id"], keep="last")
    return out


def fit_base_model(df: pd.DataFrame, args) -> Dict[str, object]:
    X = df[["pred_cat", "pred_mv"]].to_numpy(dtype=float)
    y = df["target"].to_numpy(dtype=float)

    if args.base_method == "mean":
        return {"method": "mean"}

    if args.base_method == "weighted":
        s = float(args.cat_weight + args.mv_weight)
        wc = float(args.cat_weight) / s
        wm = float(args.mv_weight) / s
        return {"method": "weighted", "cat_weight": wc, "mv_weight": wm}

    model = Ridge(alpha=float(args.ridge_alpha), fit_intercept=True)
    model.fit(X, y)
    return {
        "method": "ridge",
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "alpha": float(args.ridge_alpha),
        "model": model,
    }


def apply_base(df: pd.DataFrame, base_info: Dict[str, object]) -> np.ndarray:
    x_cat = df["pred_cat"].to_numpy(dtype=float)
    x_mv = df["pred_mv"].to_numpy(dtype=float)
    if base_info["method"] == "mean":
        return ((x_cat + x_mv) / 2.0).astype(np.float32)
    if base_info["method"] == "weighted":
        return (float(base_info["cat_weight"]) * x_cat + float(base_info["mv_weight"]) * x_mv).astype(np.float32)
    model = base_info["model"]
    X = df[["pred_cat", "pred_mv"]].to_numpy(dtype=float)
    return model.predict(X).astype(np.float32)


def fit_group_calibration(df: pd.DataFrame, group_col: str, mode: str, min_group_size: int) -> Dict[str, Dict[str, float]]:
    group_map = {}
    for g, sub in df.groupby(group_col):
        sub = sub.dropna(subset=["target", "pred"])
        if len(sub) < int(min_group_size):
            continue
        group_map[str(g)] = fit_calibration(sub["pred"].to_numpy(dtype=float), sub["target"].to_numpy(dtype=float), mode=mode)
    return group_map


def apply_group_calibration(df: pd.DataFrame, group_col: str, group_map: Dict[str, Dict[str, float]]) -> np.ndarray:
    preds = df["pred"].to_numpy(dtype=float).copy()
    if group_col not in df.columns:
        return preds
    groups = df[group_col].fillna("__NA__").astype(str).tolist()
    for i, g in enumerate(groups):
        calib = group_map.get(g)
        if calib is not None:
            preds[i] = float(calib["a"] * preds[i] + calib["b"])
    return preds


def main():
    args = parse_args()
    cv_root = Path(args.cv_root).resolve()
    cat_root = Path(args.cat_root).resolve()
    mv_root = Path(args.mv_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_oof, cat_out = _load_model_oof_out(cat_root, "cat")
    mv_oof, mv_out = _load_model_oof_out(mv_root, "mv")

    source_meta = collect_source_meta(cv_root)
    out_meta = collect_out_meta(cv_root)

    oof = cat_oof.merge(mv_oof, on="id", how="inner")
    out = cat_out.merge(mv_out, on="id", how="inner")

    oof = oof.merge(source_meta, on="id", how="left")
    out = out.merge(out_meta, on="id", how="left")

    base_info = fit_base_model(oof, args)
    oof["pred_base_raw"] = apply_base(oof, base_info)
    out["pred_base_raw"] = apply_base(out, base_info)

    raw_metrics = metrics_from_df(oof.rename(columns={"pred_base_raw": "pred"}), pred_col="pred")
    save_json(out_dir / "base_oof_metrics_raw.json", raw_metrics)

    global_calib = {"mode": args.global_calibration_mode, "a": 1.0, "b": 0.0}
    if args.use_global_calibration:
        global_calib = fit_calibration(oof["pred_base_raw"].to_numpy(dtype=float), oof["target"].to_numpy(dtype=float), mode=args.global_calibration_mode)
    save_json(out_dir / "base_global_calibration.json", global_calib)

    oof["pred"] = apply_calibration_array(oof["pred_base_raw"].to_numpy(dtype=float), global_calib)
    out["pred"] = apply_calibration_array(out["pred_base_raw"].to_numpy(dtype=float), global_calib)

    group_map = {}
    if args.use_group_calibration:
        if args.group_col not in oof.columns:
            raise ValueError(f"group-col {args.group_col!r} not found in source metadata")
        group_map = fit_group_calibration(oof[[args.group_col, "target", "pred"]].copy(), args.group_col, args.group_calibration_mode, int(args.min_group_size))
        oof["pred"] = apply_group_calibration(oof[[args.group_col, "pred"]].copy(), args.group_col, group_map)
        if args.group_col in out.columns:
            out["pred"] = apply_group_calibration(out[[args.group_col, "pred"]].copy(), args.group_col, group_map)

    final_metrics = metrics_from_df(oof, pred_col="pred")
    save_json(out_dir / "base_oof_metrics_final.json", final_metrics)

    base_info_json = {k: v for k, v in base_info.items() if k != "model"}
    save_json(out_dir / "base_model_info.json", base_info_json)
    save_json(out_dir / "base_group_calibration.json", {
        "group_col": args.group_col,
        "group_calibration_mode": args.group_calibration_mode,
        "min_group_size": int(args.min_group_size),
        "groups": group_map,
    })

    keep_oof = ["id", "target", "pred_cat", "pred_mv", "pred_base_raw", "pred"]
    keep_oof += [c for c in ["family_base", "family_base_miller", "site_type", "miller", "dopant", "anchor_count", "data_source"] if c in oof.columns]
    keep_out = ["id", "pred_cat", "pred_mv", "pred_base_raw", "pred"]
    keep_out += [c for c in ["family_base", "family_base_miller", "site_type", "miller", "dopant", "anchor_count", "data_source"] if c in out.columns]

    oof[keep_oof].to_csv(out_dir / "base_source_oof.csv", index=False)
    out[keep_out].to_csv(out_dir / "base_addH_out.csv", index=False)

    out[["id", "pred"]].sort_values("pred").head(args.topk).to_csv(out_dir / "addH_out_top20_low.csv", index=False)
    out[["id", "pred"]].sort_values("pred", ascending=False).head(args.topk).to_csv(out_dir / "addH_out_top20_high.csv", index=False)

    summary = {
        "base_method": args.base_method,
        "ridge_alpha": float(args.ridge_alpha),
        "cat_weight": float(args.cat_weight),
        "mv_weight": float(args.mv_weight),
        "use_global_calibration": bool(args.use_global_calibration),
        "global_calibration_mode": args.global_calibration_mode,
        "use_group_calibration": bool(args.use_group_calibration),
        "group_col": args.group_col,
        "group_calibration_mode": args.group_calibration_mode,
        "min_group_size": int(args.min_group_size),
        "raw_oof_metrics": raw_metrics,
        "final_oof_metrics": final_metrics,
    }
    save_json(out_dir / "base_summary.json", summary)

    print("[DONE] global cat+mv base built ->", out_dir)
    print("[INFO] raw OOF metrics   =", raw_metrics)
    print("[INFO] final OOF metrics =", final_metrics)


if __name__ == "__main__":
    main()
