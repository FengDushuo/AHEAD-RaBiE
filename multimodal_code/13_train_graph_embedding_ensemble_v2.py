#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph-embedding ensemble for addH/addH-2 -> addH-out prediction.

This script trains classical/ML regressors directly on existing graph embeddings
(eq_emb = concat(addH_emb, bare_emb, addH_emb - bare_emb)) stored in the CV pkl
files produced by 04_make_multiview_data_cv_multimodal.py.

It does NOT rerun FAIR-Chem, CLIP, or RoBERTa. It is designed as a fast, strong
baseline/stacking layer for small-sample addH adsorption-energy prediction.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV, Ridge, HuberRegressor, ElasticNetCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def read_pickle(path: Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        # common forms: {"data": df}, {"df": df}, or column dict
        for k in ["data", "df", "records"]:
            if k in obj:
                v = obj[k]
                if isinstance(v, pd.DataFrame):
                    return v.copy()
                return pd.DataFrame(v)
        return pd.DataFrame(obj)
    raise TypeError(f"Unsupported pickle object in {path}: {type(obj)}")


def write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def find_out_pkl(fold_dir: Path) -> Optional[Path]:
    candidates = [
        "regress_out.pkl", "regress_addhout.pkl", "regress_addh_out.pkl",
        "addhout_predict.pkl", "addhout_pred.pkl", "addH_out.pkl", "predict_out.pkl",
        "regress_predict_out.pkl", "regress_infer.pkl", "addhout.pkl",
    ]
    for name in candidates:
        p = fold_dir / name
        if p.exists():
            return p
    pkls = sorted(fold_dir.glob("*.pkl"))
    bad = {"regress_train.pkl", "regress_val.pkl", "regress_test.pkl", "clip_train.pkl", "clip_val.pkl"}
    for p in pkls:
        lname = p.name.lower()
        if p.name in bad:
            continue
        if any(s in lname for s in ["out", "addhout", "addh_out", "predict"]):
            return p
    return None


def load_emb_pkl(path: Path) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                out[str(k)] = np.asarray(v, dtype=np.float32).reshape(-1)
            except Exception:
                pass
        return out
    if isinstance(obj, pd.DataFrame):
        id_col = "id" if "id" in obj.columns else obj.columns[0]
        emb_col = "eq_emb" if "eq_emb" in obj.columns else None
        if emb_col is None:
            for c in obj.columns:
                if "emb" in c.lower():
                    emb_col = c
                    break
        if emb_col is None:
            raise ValueError(f"Cannot find embedding column in {path}")
        return {str(r[id_col]): parse_emb_value(r[emb_col]) for _, r in obj.iterrows()}
    raise TypeError(f"Unsupported embedding pkl type: {type(obj)}")


def parse_emb_value(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32).reshape(-1)
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=np.float32).reshape(-1)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return np.zeros((0,), dtype=np.float32)
        try:
            val = ast.literal_eval(s)
            return np.asarray(val, dtype=np.float32).reshape(-1)
        except Exception:
            # numpy array string like "[ 1.0  2.0 ...]"
            s2 = s.strip("[]")
            arr = np.fromstring(s2.replace("\n", " "), sep=" ", dtype=np.float32)
            return arr.reshape(-1)
    try:
        return np.asarray(x, dtype=np.float32).reshape(-1)
    except Exception:
        return np.zeros((0,), dtype=np.float32)


def ensure_eq_emb(df: pd.DataFrame, emb_map: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
    if "eq_emb" in df.columns:
        return df
    if emb_map is None:
        raise ValueError("DataFrame has no eq_emb column and no embedding map was provided.")
    if "id" not in df.columns:
        raise ValueError("DataFrame has no id column for joining embedding map.")
    df = df.copy()
    df["eq_emb"] = df["id"].astype(str).map(emb_map)
    return df


def select_feature(v: np.ndarray, mode: str) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if mode in ["full", "all", "dual"]:
        return v
    if len(v) % 3 != 0:
        # Cannot split addH/bare/delta; fall back to full.
        return v
    d = len(v) // 3
    addh = v[:d]
    bare = v[d:2*d]
    delta = v[2*d:3*d]
    if mode == "addh":
        return addh
    if mode == "bare":
        return bare
    if mode == "delta":
        return delta
    if mode == "addh_delta":
        return np.concatenate([addh, delta])
    if mode == "bare_delta":
        return np.concatenate([bare, delta])
    if mode == "addh_bare":
        return np.concatenate([addh, bare])
    raise ValueError(f"Unknown feature_mode: {mode}")


def matrix_from_df(df: pd.DataFrame, feature_mode: str = "full") -> Tuple[np.ndarray, np.ndarray]:
    if "eq_emb" not in df.columns:
        raise ValueError("eq_emb column is missing.")
    ids = df["id"].astype(str).to_numpy() if "id" in df.columns else np.array([str(i) for i in range(len(df))])
    vecs = []
    ok = []
    for i, x in enumerate(df["eq_emb"].values):
        v = parse_emb_value(x)
        v = select_feature(v, feature_mode)
        if v.size == 0 or not np.isfinite(v).any():
            ok.append(False)
            vecs.append(None)
        else:
            ok.append(True)
            vecs.append(v)
    # pick common dimension
    dims = [len(v) for v in vecs if v is not None]
    if not dims:
        raise ValueError("No valid embeddings found.")
    dim = max(set(dims), key=dims.count)
    X = np.zeros((len(vecs), dim), dtype=np.float32)
    valid = np.zeros(len(vecs), dtype=bool)
    for i, v in enumerate(vecs):
        if v is None:
            continue
        if len(v) >= dim:
            X[i] = v[:dim]
            valid[i] = True
        else:
            X[i, :len(v)] = v
            valid[i] = True
    return X, valid


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan}
    yt = y_true[m]
    yp = y_pred[m]
    return {
        "n": int(m.sum()),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(math.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)) if len(np.unique(yt)) > 1 else np.nan,
        "target_min": float(np.min(yt)),
        "target_max": float(np.max(yt)),
        "pred_min": float(np.min(yp)),
        "pred_max": float(np.max(yp)),
    }


def pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    if np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = pd.Series(np.asarray(a, dtype=float))
    b = pd.Series(np.asarray(b, dtype=float))
    m = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3:
        return np.nan
    if a[m].nunique() <= 1 or b[m].nunique() <= 1:
        return np.nan
    return float(a[m].corr(b[m], method="spearman"))


def build_preprocessor(pca_dim: Optional[int]) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if pca_dim and pca_dim > 0:
        steps.append(("pca", PCA(n_components=pca_dim, random_state=42)))
    return Pipeline(steps)


def available_model_names(requested: Sequence[str]) -> List[str]:
    out = []
    for name in requested:
        n = name.strip().lower()
        if not n:
            continue
        if n == "catboost":
            try:
                import catboost  # noqa: F401
                out.append(n)
            except Exception:
                print("[WARN] catboost not installed; skip catboost")
        elif n == "xgboost":
            try:
                import xgboost  # noqa: F401
                out.append(n)
            except Exception:
                print("[WARN] xgboost not installed; skip xgboost")
        elif n == "lightgbm":
            try:
                import lightgbm  # noqa: F401
                out.append(n)
            except Exception:
                print("[WARN] lightgbm not installed; skip lightgbm")
        else:
            out.append(n)
    # unique preserve order
    seen = set(); uniq = []
    for n in out:
        if n not in seen:
            uniq.append(n); seen.add(n)
    return uniq


def make_model(name: str, seed: int) -> RegressorMixin:
    name = name.lower()
    if name == "ridge":
        return RidgeCV(alphas=np.logspace(-4, 4, 17))
    if name == "ridge_strong":
        return Ridge(alpha=10.0, random_state=seed)
    if name == "huber":
        return HuberRegressor(alpha=1e-4, epsilon=1.35, max_iter=1000)
    if name == "elasticnet":
        return ElasticNetCV(l1_ratio=[0.05, 0.1, 0.3, 0.5, 0.8], alphas=np.logspace(-4, 1, 20), cv=3, max_iter=5000, random_state=seed)
    if name == "extratrees":
        return ExtraTreesRegressor(n_estimators=800, max_features="sqrt", min_samples_leaf=2, random_state=seed, n_jobs=-1)
    if name == "extratrees_deep":
        return ExtraTreesRegressor(n_estimators=1000, max_features=0.7, min_samples_leaf=1, random_state=seed, n_jobs=-1)
    if name == "rf":
        return RandomForestRegressor(n_estimators=600, max_features="sqrt", min_samples_leaf=2, random_state=seed, n_jobs=-1)
    if name == "gbrt":
        return GradientBoostingRegressor(n_estimators=350, learning_rate=0.025, max_depth=2, min_samples_leaf=3, subsample=0.8, random_state=seed)
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.025, max_leaf_nodes=15, l2_regularization=0.05, random_state=seed)
    if name == "mlp":
        return MLPRegressor(hidden_layer_sizes=(256, 128), activation="relu", alpha=1e-3, learning_rate_init=3e-4,
                            batch_size=32, max_iter=800, early_stopping=True, validation_fraction=0.15, random_state=seed)
    if name == "mlp_small":
        return MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", alpha=3e-3, learning_rate_init=3e-4,
                            batch_size=32, max_iter=800, early_stopping=True, validation_fraction=0.15, random_state=seed)
    if name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=1200, depth=4, learning_rate=0.025, loss_function="MAE",
                                 random_seed=seed, verbose=False, allow_writing_files=False, l2_leaf_reg=5.0)
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=700, learning_rate=0.025, max_depth=3, subsample=0.85, colsample_bytree=0.8,
                            reg_lambda=5.0, objective="reg:squarederror", random_state=seed, n_jobs=-1)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=700, learning_rate=0.025, num_leaves=15, subsample=0.85,
                             colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, n_jobs=-1, verbose=-1)
    raise ValueError(f"Unknown model: {name}")


def parse_seeds(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def list_folds(cv_root: Path) -> List[Path]:
    folds = sorted([p for p in cv_root.glob("fold_*") if p.is_dir()], key=lambda p: int(re.sub(r"\D", "", p.name) or 0))
    if not folds and (cv_root / "regress_train.pkl").exists():
        return [cv_root]
    return folds


def add_meta_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df["split_origin"] = prefix
    return df


def maybe_sanitize(df: pd.DataFrame, target_abs_max: Optional[float], drop_outlier_flags: bool) -> pd.DataFrame:
    if "target" not in df.columns:
        return df
    out = df.copy()
    y = pd.to_numeric(out["target"], errors="coerce")
    keep = y.notna()
    if target_abs_max and target_abs_max > 0:
        keep &= y.abs() <= float(target_abs_max)
    if drop_outlier_flags and "outlier_flag_target" in out.columns:
        flag = out["outlier_flag_target"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
        keep &= ~flag
    return out.loc[keep].reset_index(drop=True)


def calibrate_pred(y_val: np.ndarray, p_val: np.ndarray, p_other: np.ndarray, mode: str) -> Tuple[np.ndarray, Dict[str, float]]:
    y_val = np.asarray(y_val, dtype=float)
    p_val = np.asarray(p_val, dtype=float)
    m = np.isfinite(y_val) & np.isfinite(p_val)
    if m.sum() < 3 or mode == "none":
        return p_other, {"calib_a": 1.0, "calib_b": 0.0}
    if mode == "bias":
        b = float(np.median(y_val[m] - p_val[m]))
        return p_other + b, {"calib_a": 1.0, "calib_b": b}
    if mode == "linear":
        lr = LinearRegression().fit(p_val[m].reshape(-1, 1), y_val[m])
        a = float(lr.coef_[0]); b = float(lr.intercept_)
        return a * p_other + b, {"calib_a": a, "calib_b": b}
    raise ValueError(f"Unknown calibration mode: {mode}")


def aggregate_predictions(df: pd.DataFrame, pred_col: str, id_col: str = "id", method: str = "median", weight_col: Optional[str] = None) -> pd.DataFrame:
    """Aggregate many fold/seed/model predictions to one row per id.

    Supported methods:
      median          robust default
      mean            arithmetic mean
      trimmed_mean    mean after 10--90% trimming when enough runs are available
      val_mae_weighted weight = 1/(val_mae + 1e-6)^2, clipped to avoid a single run dominating
    """
    rows = []
    meta_cols = [c for c in df.columns if c not in [pred_col, "target", "abs_err", "model_name", "seed", "fold", "run_tag", "val_mae", "val_rmse", "test_mae", "test_rmse", "used"]]
    for gid, g in df.groupby(id_col, dropna=False):
        vals_all = pd.to_numeric(g[pred_col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(vals_all)
        vals = vals_all[finite]
        if len(vals) == 0:
            pred = np.nan
        elif method == "mean":
            pred = float(np.mean(vals))
        elif method == "trimmed_mean":
            if len(vals) >= 10:
                lo, hi = np.quantile(vals, [0.10, 0.90])
                vv = vals[(vals >= lo) & (vals <= hi)]
                pred = float(np.mean(vv)) if len(vv) else float(np.median(vals))
            else:
                pred = float(np.median(vals))
        elif method == "val_mae_weighted":
            if "val_mae" in g.columns:
                mae_all = pd.to_numeric(g["val_mae"], errors="coerce").to_numpy(dtype=float)
                mae = mae_all[finite]
                w = 1.0 / np.square(np.clip(mae, 1e-6, np.nanmax([np.nanmax(mae) if np.isfinite(mae).any() else 1.0, 1e-6])))
                w[~np.isfinite(w)] = 0.0
                if np.sum(w) > 0:
                    # prevent one model from dominating too much
                    cap = np.quantile(w[w > 0], 0.95) if np.any(w > 0) else 0.0
                    if cap > 0:
                        w = np.minimum(w, cap)
                    pred = float(np.average(vals, weights=w)) if np.sum(w) > 0 else float(np.median(vals))
                else:
                    pred = float(np.median(vals))
            else:
                pred = float(np.median(vals))
        elif method == "weighted" and weight_col and weight_col in g.columns:
            w_all = pd.to_numeric(g[weight_col], errors="coerce").to_numpy(dtype=float)
            w = w_all[finite]
            m = np.isfinite(w) & (w > 0)
            pred = float(np.average(vals[m], weights=w[m])) if m.sum() else float(np.median(vals))
        else:
            pred = float(np.median(vals))
        first = g.iloc[0]
        row = {"id": gid, "pred": pred, "pred_mean": float(np.mean(vals)) if len(vals) else np.nan,
               "pred_std": float(np.std(vals)) if len(vals) > 1 else 0.0,
               "pred_min": float(np.min(vals)) if len(vals) else np.nan,
               "pred_max": float(np.max(vals)) if len(vals) else np.nan,
               "n_runs": int(len(vals))}
        for c in meta_cols:
            if c not in row and c in g.columns:
                row[c] = first[c]
        if "target" in g.columns:
            yy = pd.to_numeric(g["target"], errors="coerce").dropna()
            row["target"] = yy.iloc[0] if len(yy) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-root", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--models", default="ridge,huber,elasticnet,extratrees,rf,hgb,gbrt,mlp,catboost")
    ap.add_argument("--seeds", default="42,52,62")
    ap.add_argument("--feature-mode", default="full", choices=["full", "addh", "bare", "delta", "addh_delta", "bare_delta", "addh_bare"])
    ap.add_argument("--pca-dim", type=int, default=0, help="0 disables PCA. Useful values: 32,64,128.")
    ap.add_argument("--target-abs-max", type=float, default=10.0)
    ap.add_argument("--drop-outlier-flags", action="store_true")
    ap.add_argument("--calibration", default="bias", choices=["none", "bias", "linear"])
    ap.add_argument("--aggregate-method", default="median", choices=["median", "mean", "trimmed_mean", "val_mae_weighted"])
    ap.add_argument("--max-val-mae", type=float, default=3.0)
    ap.add_argument("--max-test-rmse", type=float, default=4.0)
    ap.add_argument("--max-abs-pred", type=float, default=10.0)
    ap.add_argument("--clip-pred-abs", type=float, default=10.0)
    ap.add_argument("--enable-stacking", action="store_true")
    ap.add_argument("--addhout-master-csv", default=None)
    ap.add_argument("--addhout-eq-emb-pkl", default=None)
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--target-col", default="target")
    args = ap.parse_args()

    cv_root = Path(args.cv_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_seeds(args.seeds)
    model_names = available_model_names([x.strip() for x in args.models.split(",")])
    if not model_names:
        raise SystemExit("No available models.")

    print(f"[INFO] cv_root = {cv_root}")
    print(f"[INFO] work_dir = {work_dir}")
    print(f"[INFO] models = {model_names}")
    print(f"[INFO] seeds = {seeds}")
    print(f"[INFO] feature_mode = {args.feature_mode}")
    print(f"[INFO] pca_dim = {args.pca_dim}")

    addhout_emb_map = None
    addhout_master = None
    if args.addhout_eq_emb_pkl:
        addhout_emb_map = load_emb_pkl(Path(args.addhout_eq_emb_pkl))
    if args.addhout_master_csv:
        addhout_master = pd.read_csv(args.addhout_master_csv)
        if addhout_emb_map is not None:
            addhout_master = ensure_eq_emb(addhout_master, addhout_emb_map)

    all_run_rows = []
    test_pred_rows = []
    out_pred_rows = []
    stack_test_rows = []
    stack_out_rows = []

    fold_dirs = list_folds(cv_root)
    if not fold_dirs:
        raise SystemExit(f"No fold dirs found under {cv_root}")

    for fold_idx, fold_dir in enumerate(fold_dirs):
        m = re.search(r"fold_(\d+)", fold_dir.name)
        fold = int(m.group(1)) if m else fold_idx
        train_p = fold_dir / "regress_train.pkl"
        val_p = fold_dir / "regress_val.pkl"
        test_p = fold_dir / "regress_test.pkl"
        if not train_p.exists() or not val_p.exists():
            print(f"[WARN] skip {fold_dir}: missing train/val pkl")
            continue
        train_df = maybe_sanitize(read_pickle(train_p), args.target_abs_max, args.drop_outlier_flags)
        val_df = maybe_sanitize(read_pickle(val_p), args.target_abs_max, args.drop_outlier_flags)
        test_df = maybe_sanitize(read_pickle(test_p), args.target_abs_max, args.drop_outlier_flags) if test_p.exists() else pd.DataFrame()
        out_p = find_out_pkl(fold_dir)
        if out_p and out_p.exists():
            out_df = read_pickle(out_p)
        elif addhout_master is not None:
            out_df = addhout_master.copy()
        else:
            out_df = pd.DataFrame()

        print(f"[FOLD {fold}] train={len(train_df)} val={len(val_df)} test={len(test_df)} out={len(out_df)}")
        if len(train_df) == 0 or len(val_df) == 0:
            continue

        X_train, ok_train = matrix_from_df(train_df, args.feature_mode)
        X_val, ok_val = matrix_from_df(val_df, args.feature_mode)
        X_test, ok_test = matrix_from_df(test_df, args.feature_mode) if len(test_df) else (np.empty((0, X_train.shape[1])), np.zeros(0, dtype=bool))
        X_out, ok_out = matrix_from_df(out_df, args.feature_mode) if len(out_df) else (np.empty((0, X_train.shape[1])), np.zeros(0, dtype=bool))

        y_train = pd.to_numeric(train_df[args.target_col], errors="coerce").to_numpy(dtype=float)
        y_val = pd.to_numeric(val_df[args.target_col], errors="coerce").to_numpy(dtype=float)
        y_test = pd.to_numeric(test_df[args.target_col], errors="coerce").to_numpy(dtype=float) if len(test_df) and args.target_col in test_df.columns else np.full(len(test_df), np.nan)

        good_train = ok_train & np.isfinite(y_train)
        good_val = ok_val & np.isfinite(y_val)
        if good_train.sum() < 10 or good_val.sum() < 3:
            print(f"[WARN] fold {fold}: too few good samples")
            continue

        for seed in seeds:
            base_val_preds = {}
            base_test_preds = {}
            base_out_preds = {}
            for name in model_names:
                run_tag = f"fold{fold}_seed{seed}_{name}"
                try:
                    prep = build_preprocessor(args.pca_dim if args.pca_dim > 0 else None)
                    model = make_model(name, seed)
                    pipe = Pipeline([("prep", prep), ("model", model)])
                    pipe.fit(X_train[good_train], y_train[good_train])
                    pv = pipe.predict(X_val)
                    pt = pipe.predict(X_test) if len(test_df) else np.array([])
                    po = pipe.predict(X_out) if len(out_df) else np.array([])

                    # Calibrate using validation. Apply same transform to val/test/out.
                    pv_cal, calib = calibrate_pred(y_val, pv, pv, args.calibration)
                    pt_cal, _ = calibrate_pred(y_val, pv, pt, args.calibration) if len(pt) else (pt, calib)
                    po_cal, _ = calibrate_pred(y_val, pv, po, args.calibration) if len(po) else (po, calib)

                    if args.clip_pred_abs and args.clip_pred_abs > 0:
                        pv_cal = np.clip(pv_cal, -args.clip_pred_abs, args.clip_pred_abs)
                        if len(pt_cal): pt_cal = np.clip(pt_cal, -args.clip_pred_abs, args.clip_pred_abs)
                        if len(po_cal): po_cal = np.clip(po_cal, -args.clip_pred_abs, args.clip_pred_abs)

                    vm = metrics(y_val, pv_cal)
                    tm = metrics(y_test, pt_cal) if len(test_df) else {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan}
                    max_abs_pred = float(np.nanmax(np.abs(pt_cal))) if len(pt_cal) else np.nan
                    used = True
                    if np.isfinite(vm.get("mae", np.nan)) and vm["mae"] > args.max_val_mae:
                        used = False
                    if np.isfinite(tm.get("rmse", np.nan)) and tm["rmse"] > args.max_test_rmse:
                        used = False
                    if np.isfinite(max_abs_pred) and max_abs_pred > args.max_abs_pred:
                        used = False

                    all_run_rows.append({
                        "run_tag": run_tag, "fold": fold, "seed": seed, "model_name": name,
                        "feature_mode": args.feature_mode, "pca_dim": args.pca_dim,
                        "val_mae": vm.get("mae"), "val_rmse": vm.get("rmse"), "val_r2": vm.get("r2"),
                        "test_mae": tm.get("mae"), "test_rmse": tm.get("rmse"), "test_r2": tm.get("r2"),
                        "max_abs_pred": max_abs_pred, "used": used,
                        **calib,
                    })
                    base_val_preds[name] = pv_cal
                    if len(pt_cal): base_test_preds[name] = pt_cal
                    if len(po_cal): base_out_preds[name] = po_cal

                    if len(test_df):
                        tmp = test_df.copy()
                        tmp["pred"] = pt_cal
                        tmp["run_tag"] = run_tag
                        tmp["fold"] = fold
                        tmp["seed"] = seed
                        tmp["model_name"] = name
                        tmp["val_mae"] = vm.get("mae")
                        tmp["test_rmse"] = tm.get("rmse")
                        tmp["used"] = used
                        test_pred_rows.append(tmp)
                    if len(out_df):
                        tmpo = out_df.copy()
                        tmpo["pred"] = po_cal
                        tmpo["run_tag"] = run_tag
                        tmpo["fold"] = fold
                        tmpo["seed"] = seed
                        tmpo["model_name"] = name
                        tmpo["val_mae"] = vm.get("mae")
                        tmpo["used"] = used
                        out_pred_rows.append(tmpo)
                except Exception as e:
                    print(f"[WARN] {run_tag} failed: {e}")
                    all_run_rows.append({"run_tag": run_tag, "fold": fold, "seed": seed, "model_name": name, "used": False, "error": repr(e)})

            # Simple stacking per fold/seed using validation predictions as meta-training.
            if args.enable_stacking and len(base_val_preds) >= 2:
                common_names = [n for n in model_names if n in base_val_preds]
                Z_val = np.vstack([base_val_preds[n] for n in common_names]).T
                good_meta = np.isfinite(Z_val).all(axis=1) & np.isfinite(y_val)
                if good_meta.sum() >= max(5, len(common_names) + 2):
                    try:
                        meta = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Z_val[good_meta], y_val[good_meta])
                        run_tag = f"fold{fold}_seed{seed}_stack_ridge"
                        pv = meta.predict(Z_val)
                        vm = metrics(y_val, pv)
                        pt = np.array([])
                        po = np.array([])
                        if len(test_df) and all(n in base_test_preds for n in common_names):
                            Zt = np.vstack([base_test_preds[n] for n in common_names]).T
                            pt = meta.predict(Zt)
                            if args.clip_pred_abs and args.clip_pred_abs > 0:
                                pt = np.clip(pt, -args.clip_pred_abs, args.clip_pred_abs)
                            tm = metrics(y_test, pt)
                        else:
                            tm = {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan}
                        if len(out_df) and all(n in base_out_preds for n in common_names):
                            Zo = np.vstack([base_out_preds[n] for n in common_names]).T
                            po = meta.predict(Zo)
                            if args.clip_pred_abs and args.clip_pred_abs > 0:
                                po = np.clip(po, -args.clip_pred_abs, args.clip_pred_abs)
                        max_abs_pred = float(np.nanmax(np.abs(pt))) if len(pt) else np.nan
                        used = True
                        if np.isfinite(vm.get("mae", np.nan)) and vm["mae"] > args.max_val_mae:
                            used = False
                        if np.isfinite(tm.get("rmse", np.nan)) and tm["rmse"] > args.max_test_rmse:
                            used = False
                        if np.isfinite(max_abs_pred) and max_abs_pred > args.max_abs_pred:
                            used = False
                        all_run_rows.append({
                            "run_tag": run_tag, "fold": fold, "seed": seed, "model_name": "stack_ridge",
                            "feature_mode": args.feature_mode, "pca_dim": args.pca_dim,
                            "val_mae": vm.get("mae"), "val_rmse": vm.get("rmse"), "val_r2": vm.get("r2"),
                            "test_mae": tm.get("mae"), "test_rmse": tm.get("rmse"), "test_r2": tm.get("r2"),
                            "max_abs_pred": max_abs_pred, "used": used,
                        })
                        if len(pt):
                            tmp = test_df.copy(); tmp["pred"] = pt; tmp["run_tag"] = run_tag; tmp["fold"] = fold; tmp["seed"] = seed; tmp["model_name"] = "stack_ridge"; tmp["used"] = used
                            test_pred_rows.append(tmp)
                        if len(po):
                            tmpo = out_df.copy(); tmpo["pred"] = po; tmpo["run_tag"] = run_tag; tmpo["fold"] = fold; tmpo["seed"] = seed; tmpo["model_name"] = "stack_ridge"; tmpo["used"] = used
                            out_pred_rows.append(tmpo)
                    except Exception as e:
                        print(f"[WARN] stacking fold={fold} seed={seed} failed: {e}")

    run_df = pd.DataFrame(all_run_rows)
    run_df.to_csv(work_dir / "graph_ensemble_run_summary.csv", index=False)
    print(f"[OK] run summary -> {work_dir / 'graph_ensemble_run_summary.csv'}")

    if test_pred_rows:
        pred_df = pd.concat(test_pred_rows, ignore_index=True)
        pred_df.to_csv(work_dir / "test_pred_all_graph_runs.csv", index=False)
        used_df = pred_df[pred_df.get("used", True).astype(bool)].copy() if "used" in pred_df.columns else pred_df.copy()
        oof = aggregate_predictions(used_df, "pred", id_col=args.id_col, method=args.aggregate_method)
        if args.target_col in oof.columns:
            y = pd.to_numeric(oof[args.target_col], errors="coerce").to_numpy(dtype=float)
            p = pd.to_numeric(oof["pred"], errors="coerce").to_numpy(dtype=float)
            m = metrics(y, p)
            m["pearson"] = pearson_safe(y, p)
            m["spearman"] = spearman_safe(y, p)
            m["bias"] = float(np.nanmean(p - y)) if np.isfinite(p).any() and np.isfinite(y).any() else np.nan
            write_json(m, work_dir / "test_oof_graph_ensemble_metrics.json")
            print(f"[OOF] {m}")
        oof.to_csv(work_dir / "test_oof_graph_ensemble_by_id.csv", index=False)
        print(f"[OK] OOF by id -> {work_dir / 'test_oof_graph_ensemble_by_id.csv'}")

    if out_pred_rows:
        out_all = pd.concat(out_pred_rows, ignore_index=True)
        out_all.to_csv(work_dir / "addH_out_pred_all_graph_runs.csv", index=False)
        used_out = out_all[out_all.get("used", True).astype(bool)].copy() if "used" in out_all.columns else out_all.copy()
        byid = aggregate_predictions(used_out, "pred", id_col=args.id_col, method=args.aggregate_method)
        # add posterior metrics if addH-out has labels
        for true_col in ["h_ads_excel", "target", "target_computed"]:
            if true_col in byid.columns:
                byid[true_col] = pd.to_numeric(byid[true_col], errors="coerce")
                byid[f"abs_err_vs_{true_col}"] = (pd.to_numeric(byid["pred"], errors="coerce") - byid[true_col]).abs()
        if "h_ads_excel" in byid.columns:
            y = pd.to_numeric(byid["h_ads_excel"], errors="coerce").to_numpy(dtype=float)
            p = pd.to_numeric(byid["pred"], errors="coerce").to_numpy(dtype=float)
            post = metrics(y, p)
            post["pearson"] = pearson_safe(y, p)
            post["spearman"] = spearman_safe(y, p)
            post["bias"] = float(np.nanmean(p - y)) if np.isfinite(p).any() and np.isfinite(y).any() else np.nan
            write_json(post, work_dir / "addH_out_graph_ensemble_metrics_vs_excel.json")
            print(f"[ADDH-OUT vs Excel] {post}")
        byid = byid.sort_values("pred", ascending=True)
        byid.to_csv(work_dir / "addH_out_graph_ensemble_by_id.csv", index=False)
        print(f"[OK] addH-out by id -> {work_dir / 'addH_out_graph_ensemble_by_id.csv'}")

    print("[DONE] graph embedding ensemble")


if __name__ == "__main__":
    main()
