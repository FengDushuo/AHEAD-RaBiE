#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_multiview_meta_features.py

Build lightweight multiview-derived meta features for downstream models.

Main use
--------
Convert multiview outputs into tabular per-id features that can be merged into:
- 04_make_singleview_data_cv_meta_v2.py outputs
- CatBoost singleview pipeline
- singleview strong NN

This script is intentionally conservative:
- for source training rows, it uses OOF / held-out test predictions only
- for target rows (e.g. addH-out), it aggregates multiview predictions across runs
- optionally compresses multiview latent embeddings into PCA features

What it produces
----------------
1) multiview_train_features.csv
   one row per training id, based on OOF predictions

2) multiview_out_features.csv
   one row per target/out id, based on addH-out predictions

Feature families
----------------
A. Prediction statistics
   mv_pred_mean
   mv_pred_std
   mv_pred_min
   mv_pred_max
   mv_pred_range
   mv_pred_count

B. Optional latent PCA features
   mv_lat_pca_000, mv_lat_pca_001, ...

Expected multiview prediction inputs
------------------------------------
This script can read either:
1) global aggregated files:
   - test_pred_all_runs.csv
   - addH_out_pred_all_runs.csv

or
2) per-run files found recursively under --mv-root:
   - test_pred.csv
   - addH_out_pred.csv

The prediction CSVs should contain at least:
- id
- pred

Optional latent embedding inputs
--------------------------------
- --train-latent-pkl
- --out-latent-pkl

Each should be a pickle dict:
    id -> 1D vector

These vectors are PCA-compressed using training vectors only.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except Exception as e:
    raise SystemExit(f"scikit-learn is required: {e}")


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mv-root", required=True, help="Root directory containing multiview outputs")
    ap.add_argument("--train-manifest-csv", required=True, help="Training/source manifest with an id column")
    ap.add_argument("--out-manifest-csv", required=True, help="Target/out manifest with an id column")

    ap.add_argument("--train-pred-csv", default=None, help="Optional explicit OOF/test prediction CSV")
    ap.add_argument("--out-pred-csv", default=None, help="Optional explicit addH-out prediction CSV")

    ap.add_argument("--train-latent-pkl", default=None)
    ap.add_argument("--out-latent-pkl", default=None)
    ap.add_argument("--latent-pca-dim", type=int, default=16)
    ap.add_argument("--use-latent-scaler", action="store_true")

    ap.add_argument("--train-output-csv", default="multiview_train_features.csv")
    ap.add_argument("--out-output-csv", default="multiview_out_features.csv")
    ap.add_argument("--summary-json", default="multiview_meta_feature_summary.json")

    ap.add_argument("--strict", action="store_true")
    return ap.parse_args()


def save_json(path: Path, obj: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_run_tag(text: str) -> Tuple[Optional[int], Optional[int]]:
    s = str(text)
    m = re.search(r"fold[_\-]?(\d+).*?seed[_\-]?(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"fold[_\-]?(\d+)", s)
    if m:
        return int(m.group(1)), None
    return None, None


def normalize_pred_df(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    out = df.copy()
    if "id" not in out.columns or "pred" not in out.columns:
        raise ValueError(f"{origin}: prediction file must contain id and pred columns")

    p_fold, p_seed = parse_run_tag(origin)

    if "fold" not in out.columns:
        out["fold"] = p_fold
    elif p_fold is not None:
        out["fold"] = out["fold"].fillna(p_fold)

    if "seed" not in out.columns:
        out["seed"] = p_seed
    elif p_seed is not None:
        out["seed"] = out["seed"].fillna(p_seed)

    if "run_tag" not in out.columns:
        if p_fold is not None and p_seed is not None:
            out["run_tag"] = f"fold{p_fold}_seed{p_seed}"
        elif p_fold is not None:
            out["run_tag"] = f"fold{p_fold}"
        else:
            out["run_tag"] = Path(origin).stem

    out["id"] = out["id"].astype(str)
    out["pred"] = pd.to_numeric(out["pred"], errors="coerce")
    out = out.dropna(subset=["pred"])
    return out


def load_prediction_csvs(mv_root: Path, explicit_csv: Optional[str], kind: str) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []

    if explicit_csv is not None:
        p = Path(explicit_csv)
        if not p.exists():
            raise FileNotFoundError(p)
        parts.append(normalize_pred_df(pd.read_csv(p), origin=str(p)))
    else:
        if kind == "train":
            global_name = mv_root / "test_pred_all_runs.csv"
            if global_name.exists():
                parts.append(normalize_pred_df(pd.read_csv(global_name), origin=str(global_name)))
            else:
                for fp in mv_root.rglob("test_pred.csv"):
                    parts.append(normalize_pred_df(pd.read_csv(fp), origin=str(fp)))
        elif kind == "out":
            global_name = mv_root / "addH_out_pred_all_runs.csv"
            if global_name.exists():
                parts.append(normalize_pred_df(pd.read_csv(global_name), origin=str(global_name)))
            else:
                for fp in mv_root.rglob("addH_out_pred.csv"):
                    parts.append(normalize_pred_df(pd.read_csv(fp), origin=str(fp)))
        else:
            raise ValueError(f"Unsupported kind: {kind}")

    if not parts:
        raise FileNotFoundError(f"No multiview prediction files found for kind={kind} under {mv_root}")

    out = pd.concat(parts, axis=0, ignore_index=True)
    subset_cols = [c for c in ["id", "fold", "seed", "run_tag", "pred"] if c in out.columns]
    if subset_cols:
        out = out.drop_duplicates(subset=subset_cols, keep="last")
    return out


def aggregate_prediction_features(pred_df: pd.DataFrame, prefix: str = "mv_") -> pd.DataFrame:
    grouped = pred_df.groupby("id")["pred"]
    feat = grouped.agg(["mean", "std", "min", "max", "count"]).reset_index()
    feat = feat.rename(columns={
        "mean": f"{prefix}pred_mean",
        "std": f"{prefix}pred_std",
        "min": f"{prefix}pred_min",
        "max": f"{prefix}pred_max",
        "count": f"{prefix}pred_count",
    })
    feat[f"{prefix}pred_std"] = feat[f"{prefix}pred_std"].fillna(0.0)
    feat[f"{prefix}pred_range"] = feat[f"{prefix}pred_max"] - feat[f"{prefix}pred_min"]
    return feat


def load_latent_map(path: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{p}: expected pickle dict[id] -> vector")
    out = {}
    for k, v in obj.items():
        out[str(k)] = np.asarray(v, dtype=np.float32).reshape(-1)
    return out


def prepare_latent_features(
    train_ids: List[str],
    out_ids: List[str],
    train_lat_map: Optional[Dict[str, np.ndarray]],
    out_lat_map: Optional[Dict[str, np.ndarray]],
    pca_dim: int,
    use_scaler: bool,
    prefix: str = "mv_lat_",
):
    if train_lat_map is None or out_lat_map is None:
        return None, None, None

    train_vecs = [train_lat_map[i] for i in train_ids if i in train_lat_map]
    if not train_vecs:
        return None, None, None

    dims = sorted(set(v.shape[0] for v in train_vecs))
    if len(dims) != 1:
        raise ValueError(f"Inconsistent training latent dimensions: {dims}")

    X_train_fit = np.stack(train_vecs, axis=0)
    scaler = None
    if use_scaler:
        scaler = StandardScaler()
        X_train_fit = scaler.fit_transform(X_train_fit)

    n_comp = int(min(pca_dim, X_train_fit.shape[0] - 1, X_train_fit.shape[1]))
    n_comp = max(2, n_comp)
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(X_train_fit)

    def _transform_ids(id_list: List[str], lat_map: Dict[str, np.ndarray]):
        cols = [f"{prefix}pca_{i:03d}" for i in range(n_comp)]
        rows = []
        for sid in id_list:
            rec = {"id": sid}
            if sid not in lat_map:
                for c in cols:
                    rec[c] = np.nan
            else:
                v = lat_map[sid].reshape(1, -1)
                if scaler is not None:
                    v = scaler.transform(v)
                z = pca.transform(v).reshape(-1)
                for c, val in zip(cols, z):
                    rec[c] = float(val)
            rows.append(rec)
        return pd.DataFrame(rows), cols

    train_df, cols = _transform_ids(train_ids, train_lat_map)
    out_df, _ = _transform_ids(out_ids, out_lat_map)
    meta = {
        "n_components": n_comp,
        "cols": cols,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "use_scaler": bool(use_scaler),
    }
    return train_df, out_df, meta


def ensure_manifest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns:
        raise ValueError(f"{path}: expected id column")
    df["id"] = df["id"].astype(str)
    return df


def main():
    args = parse_args()

    mv_root = Path(args.mv_root).resolve()
    train_manifest = ensure_manifest(args.train_manifest_csv)
    out_manifest = ensure_manifest(args.out_manifest_csv)

    train_pred_df = load_prediction_csvs(mv_root, args.train_pred_csv, kind="train")
    out_pred_df = load_prediction_csvs(mv_root, args.out_pred_csv, kind="out")

    train_pred_feat = aggregate_prediction_features(train_pred_df, prefix="mv_")
    out_pred_feat = aggregate_prediction_features(out_pred_df, prefix="mv_")

    train_out = train_manifest[["id"]].copy().merge(train_pred_feat, on="id", how="left")
    out_out = out_manifest[["id"]].copy().merge(out_pred_feat, on="id", how="left")

    latent_meta = None
    train_lat_map = load_latent_map(args.train_latent_pkl)
    out_lat_map = load_latent_map(args.out_latent_pkl)
    if train_lat_map is not None and out_lat_map is not None:
        train_lat_df, out_lat_df, latent_meta = prepare_latent_features(
            train_ids=train_manifest["id"].astype(str).tolist(),
            out_ids=out_manifest["id"].astype(str).tolist(),
            train_lat_map=train_lat_map,
            out_lat_map=out_lat_map,
            pca_dim=int(args.latent_pca_dim),
            use_scaler=bool(args.use_latent_scaler),
            prefix="mv_lat_",
        )
        if train_lat_df is not None:
            train_out = train_out.merge(train_lat_df, on="id", how="left")
        if out_lat_df is not None:
            out_out = out_out.merge(out_lat_df, on="id", how="left")

    pred_cols = [c for c in train_out.columns if c.startswith("mv_pred")]
    train_missing = int(train_out[pred_cols].isna().all(axis=1).sum()) if pred_cols else int(len(train_out))
    out_missing = int(out_out[pred_cols].isna().all(axis=1).sum()) if pred_cols else int(len(out_out))

    if args.strict:
        if train_missing > 0:
            raise ValueError(f"Missing multiview prediction features for {train_missing} training ids")
        if out_missing > 0:
            raise ValueError(f"Missing multiview prediction features for {out_missing} out ids")

    train_out.to_csv(args.train_output_csv, index=False)
    out_out.to_csv(args.out_output_csv, index=False)

    summary = {
        "mv_root": str(mv_root),
        "n_train_manifest": int(len(train_manifest)),
        "n_out_manifest": int(len(out_manifest)),
        "n_train_pred_rows": int(len(train_pred_df)),
        "n_out_pred_rows": int(len(out_pred_df)),
        "n_train_feature_rows": int(len(train_out)),
        "n_out_feature_rows": int(len(out_out)),
        "train_missing_prediction_features": int(train_missing),
        "out_missing_prediction_features": int(out_missing),
        "prediction_feature_cols": [c for c in train_out.columns if c.startswith("mv_pred")],
        "latent_meta": latent_meta,
    }
    save_json(Path(args.summary_json), summary)

    print(f"[OK] train features -> {args.train_output_csv}")
    print(f"[OK] out features   -> {args.out_output_csv}")
    print(f"[OK] summary json   -> {args.summary_json}")
    print(f"[INFO] train missing prediction features = {train_missing}")
    print(f"[INFO] out missing prediction features   = {out_missing}")


if __name__ == "__main__":
    main()
