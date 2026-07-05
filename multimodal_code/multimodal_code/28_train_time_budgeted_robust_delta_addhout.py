#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-budgeted robust retraining for AddH/AddH-2 -> AddH-out.

This is the retraining step to use when full FAIR-Chem/Equiformer fine-tuning is
too expensive. It reuses cached dual embeddings and tabular/LLM/element features,
then trains many small, grouped-CV-guarded heads.

Strict default:
  - train labels come only from addH/addH-2;
  - addH-out labels, if supplied, are used only for post-hoc audit files;
  - final output is conservatively blended with the existing target-domain
    anchor prediction.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Time-budgeted robust AddH-out retraining.")
    ap.add_argument("--bundle-dir", default="outputs_addh_pretrained_delta_features")
    ap.add_argument("--bundle-npz", default="")
    ap.add_argument("--out-dir", default="outputs_addh_robust_retrain_delta")
    ap.add_argument("--existing-pred-csv", default="outputs_addh_target_calibrated_fast/target_calibrated_addhout_predictions.csv")
    ap.add_argument("--existing-pred-col", default="pred_fast_target_calibrated")
    ap.add_argument("--audit-labels-csv", default="auto")
    ap.add_argument("--audit-target-col", default="h_ads_excel")
    ap.add_argument("--profile", choices=["fast", "medium", "thorough"], default="fast")
    ap.add_argument("--n-splits", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=0, help="0 means use the profile default.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=0, help="0 means use the profile default.")
    ap.add_argument("--min-oof-improvement", type=float, default=0.03)
    ap.add_argument("--max-pred-mean-shift", type=float, default=2.20)
    ap.add_argument("--max-pred-std-ratio", type=float, default=2.50)
    ap.add_argument("--max-delta-blend-weight", type=float, default=0.22)
    ap.add_argument("--force-delta-weight", type=float, default=-1.0)
    ap.add_argument("--recenter-mode", choices=["none", "global_median", "material_median"], default="material_median")
    ap.add_argument("--clip-final-to-source-range", action="store_true")
    return ap.parse_args()


PROFILE_CONFIG = {
    "fast": {
        "repeats": 2,
        "top_k": 8,
        "feature_modes": ["tab", "delta", "addh_delta", "full"],
        "pca_dims": [0, 32, 128],
        "models": ["ridge", "huber", "elastic", "hgb", "knn"],
        "target_modes": ["residual", "absolute"],
    },
    "medium": {
        "repeats": 3,
        "top_k": 12,
        "feature_modes": ["tab", "delta", "addh_delta", "bare_delta", "full"],
        "pca_dims": [0, 16, 32, 64, 128],
        "models": ["ridge", "huber", "elastic", "hgb", "gbr", "extratrees", "knn"],
        "target_modes": ["residual", "absolute"],
    },
    "thorough": {
        "repeats": 5,
        "top_k": 16,
        "feature_modes": ["tab", "addh", "bare", "delta", "addh_delta", "bare_delta", "full"],
        "pca_dims": [0, 16, 32, 64, 96, 128],
        "models": ["ridge", "huber", "elastic", "hgb", "gbr", "extratrees", "rf", "knn"],
        "target_modes": ["residual", "absolute"],
    },
}


def finite_corr(a: Sequence[float], b: Sequence[float], spearman: bool = False) -> float:
    x = pd.to_numeric(pd.Series(a), errors="coerce")
    y = pd.to_numeric(pd.Series(b), errors="coerce")
    m = x.notna() & y.notna()
    if int(m.sum()) < 3:
        return float("nan")
    xv = x[m].to_numpy(float)
    yv = y[m].to_numpy(float)
    if spearman:
        xv = pd.Series(xv).rank(method="average").to_numpy(float)
        yv = pd.Series(yv).rank(method="average").to_numpy(float)
    if np.nanstd(xv) <= 1e-12 or np.nanstd(yv) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y = pd.to_numeric(pd.Series(y_true), errors="coerce")
    p = pd.to_numeric(pd.Series(y_pred), errors="coerce")
    m = y.notna() & p.notna()
    if int(m.sum()) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "pearson": np.nan, "spearman": np.nan}
    yy = y[m].to_numpy(float)
    pp = p[m].to_numpy(float)
    e = pp - yy
    return {
        "n": int(len(yy)),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(np.mean(e)),
        "pearson": finite_corr(yy, pp, False),
        "spearman": finite_corr(yy, pp, True),
    }


def metric_rows(df: pd.DataFrame, pred_cols: Iterable[str], target_col: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    materials: List[Optional[str]] = [None]
    if "material" in df.columns:
        materials.extend(sorted([str(x) for x in df["material"].dropna().unique()]))
    for c in pred_cols:
        if c not in df.columns:
            continue
        for mat in materials:
            d = df if mat is None else df[df["material"].astype(str) == mat]
            row: Dict[str, object] = {"pred_col": c, "target_col": target_col, "material": mat}
            row.update(metrics(d[target_col], d[c]))
            rows.append(row)
    return pd.DataFrame(rows)


def load_bundle(bundle_dir: Path, npz_arg: str):
    npz_path = Path(npz_arg) if npz_arg else bundle_dir / "pretrained_delta_feature_bundle.npz"
    if not npz_path.exists():
        raise SystemExit(f"[ERROR] missing feature bundle: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    train_meta = pd.DataFrame(
        {
            "id": data["train_ids"].astype(str),
            "family_base": data["train_groups"].astype(str),
            "material": data["train_material"].astype(str),
            "dopant": data["train_dopant"].astype(str),
            "target": data["y_train"].astype(float),
            "has_embedding": data["train_has_embedding"].astype(bool),
        }
    )
    addhout_meta = pd.DataFrame(
        {
            "id": data["addhout_ids"].astype(str),
            "material": data["addhout_material"].astype(str),
            "dopant": data["addhout_dopant"].astype(str),
            "has_embedding": data["addhout_has_embedding"].astype(bool),
        }
    )
    return npz_path, data, train_meta, addhout_meta


def repeated_group_folds(groups: Sequence[str], n_splits: int, repeats: int, seed: int) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    groups = np.asarray([str(g) for g in groups])
    uniq = np.asarray(pd.unique(groups))
    n = max(2, min(int(n_splits), len(uniq)))
    out: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for r in range(int(repeats)):
        rng = np.random.default_rng(seed + 1009 * r)
        shuffled = rng.permutation(uniq)
        fold_groups = [set(shuffled[i::n]) for i in range(n)]
        for k, gset in enumerate(fold_groups):
            va = np.where(np.isin(groups, list(gset)))[0]
            tr = np.where(~np.isin(groups, list(gset)))[0]
            if len(tr) and len(va):
                out.append((r, tr.astype(int), va.astype(int)))
    return out


def dopant_prior(train_df: pd.DataFrame, apply_df: pd.DataFrame) -> np.ndarray:
    agg = train_df.groupby("dopant")["target"].mean()
    fallback = float(train_df["target"].mean())
    return apply_df["dopant"].map(agg).fillna(fallback).to_numpy(float)


def fold_prior_oof(
    train_meta: pd.DataFrame,
    addhout_meta: pd.DataFrame,
    folds: List[Tuple[int, np.ndarray, np.ndarray]],
    repeats: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    n = len(train_meta)
    sums = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=float)
    out_preds: List[np.ndarray] = []
    repeat_mae: List[float] = []
    for r in range(repeats):
        pred_r = np.full(n, np.nan, dtype=float)
        for rr, tr, va in folds:
            if rr != r:
                continue
            p = dopant_prior(train_meta.iloc[tr], train_meta.iloc[va])
            pred_r[va] = p
            sums[va] += p
            counts[va] += 1.0
            out_preds.append(dopant_prior(train_meta.iloc[tr], addhout_meta))
        repeat_mae.append(metrics(train_meta["target"], pred_r)["mae"])
    oof = np.full(n, np.nan, dtype=float)
    m = counts > 0
    oof[m] = sums[m] / counts[m]
    out_full = dopant_prior(train_meta, addhout_meta)
    if out_preds:
        out_cv = np.nanmean(np.vstack(out_preds), axis=0)
        out_full = 0.85 * out_full + 0.15 * out_cv
    return oof, out_full, {
        "prior_mae": float(metrics(train_meta["target"], oof)["mae"]),
        "prior_spearman": float(metrics(train_meta["target"], oof)["spearman"]),
        "prior_repeat_mae_mean": float(np.nanmean(repeat_mae)),
        "prior_repeat_mae_std": float(np.nanstd(repeat_mae)),
    }


def select_graph_mode(X: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode in {"full", "all", "dual"}:
        return X
    if X.shape[1] % 3 != 0:
        return X
    d = X.shape[1] // 3
    addh = X[:, :d]
    bare = X[:, d : 2 * d]
    delta = X[:, 2 * d : 3 * d]
    if mode == "addh":
        return addh
    if mode == "bare":
        return bare
    if mode == "delta":
        return delta
    if mode == "addh_delta":
        return np.hstack([addh, delta])
    if mode == "bare_delta":
        return np.hstack([bare, delta])
    if mode == "addh_bare":
        return np.hstack([addh, bare])
    raise ValueError(f"unknown graph feature mode: {mode}")


def make_feature_block(
    mode: str,
    Xg_train: np.ndarray,
    Xg_out: np.ndarray,
    Xt_train: np.ndarray,
    Xt_out: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "tab":
        if Xt_train.shape[1] == 0:
            raise ValueError("tab mode requested but tabular feature dimension is zero")
        return Xt_train, Xt_out
    G = select_graph_mode(Xg_train, mode)
    Go = select_graph_mode(Xg_out, mode)
    if Xt_train.shape[1]:
        return np.hstack([G, Xt_train]), np.hstack([Go, Xt_out])
    return G, Go


def model_factory(family: str, seed: int):
    family = family.lower()
    if family == "ridge":
        return [(f"ridge_a{a:g}", Ridge(alpha=a)) for a in [3.0, 10.0, 30.0, 100.0, 300.0]]
    if family == "huber":
        return [
            (f"huber_a{a:g}_e{eps:g}", HuberRegressor(alpha=a, epsilon=eps, max_iter=800))
            for a in [0.0005, 0.002, 0.01]
            for eps in [1.15, 1.35]
        ]
    if family == "elastic":
        return [
            (f"elastic_a{a:g}", ElasticNet(alpha=a, l1_ratio=0.20, max_iter=6000, random_state=seed))
            for a in [0.003, 0.01, 0.03, 0.1]
        ]
    if family == "hgb":
        return [("hgb", HistGradientBoostingRegressor(max_iter=220, learning_rate=0.025, l2_regularization=1.0, max_leaf_nodes=8, random_state=seed))]
    if family == "gbr":
        return [("gbr", GradientBoostingRegressor(n_estimators=180, learning_rate=0.025, max_depth=2, random_state=seed))]
    if family == "extratrees":
        return [("extratrees", ExtraTreesRegressor(n_estimators=260, min_samples_leaf=3, max_features=0.65, random_state=seed, n_jobs=-1))]
    if family == "rf":
        return [("rf", RandomForestRegressor(n_estimators=320, min_samples_leaf=3, max_features=0.65, random_state=seed, n_jobs=-1))]
    if family == "knn":
        return [(f"knn_k{k}", KNeighborsRegressor(n_neighbors=k, weights="distance")) for k in [7, 11, 17, 25]]
    return []


def make_pipeline(estimator, pca_dim: int, n_train: int, n_features: int) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if pca_dim > 0:
        n_comp = min(int(pca_dim), max(1, n_train - 1), n_features)
        steps.append(("pca", PCA(n_components=n_comp, random_state=42)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def train_candidate(
    candidate_name: str,
    estimator,
    pca_dim: int,
    target_mode: str,
    X: np.ndarray,
    Xout: np.ndarray,
    train_meta: pd.DataFrame,
    addhout_meta: pd.DataFrame,
    folds: List[Tuple[int, np.ndarray, np.ndarray]],
    repeats: int,
    prior_oof: np.ndarray,
    prior_out: np.ndarray,
    seed: int,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    y = train_meta["target"].to_numpy(float)
    n_train, n_feat = X.shape
    pred_sums = np.zeros(n_train, dtype=float)
    pred_counts = np.zeros(n_train, dtype=float)
    repeat_preds: Dict[int, np.ndarray] = {r: np.full(n_train, np.nan, dtype=float) for r in range(repeats)}
    out_cv_preds: List[np.ndarray] = []

    for r, tr, va in folds:
        pipe = make_pipeline(clone(estimator), pca_dim, len(tr), n_feat)
        if target_mode == "residual":
            tr_prior = dopant_prior(train_meta.iloc[tr], train_meta.iloc[tr])
            y_fit = y[tr] - tr_prior
        else:
            y_fit = y[tr]
        pipe.fit(X[tr], y_fit)
        va_pred = pipe.predict(X[va])
        out_pred = pipe.predict(Xout)
        if target_mode == "residual":
            va_pred = prior_oof[va] + va_pred
            out_pred = prior_out + out_pred
        pred_sums[va] += va_pred
        pred_counts[va] += 1.0
        repeat_preds[r][va] = va_pred
        out_cv_preds.append(out_pred)

    pred_oof = np.full(n_train, np.nan, dtype=float)
    m = pred_counts > 0
    pred_oof[m] = pred_sums[m] / pred_counts[m]

    pipe = make_pipeline(clone(estimator), pca_dim, n_train, n_feat)
    if target_mode == "residual":
        y_fit_full = y - dopant_prior(train_meta, train_meta)
    else:
        y_fit_full = y
    pipe.fit(X, y_fit_full)
    out_full = pipe.predict(Xout)
    if target_mode == "residual":
        out_full = prior_out + out_full
    out_cv = np.nanmean(np.vstack(out_cv_preds), axis=0) if out_cv_preds else out_full
    out_pred = 0.65 * out_full + 0.35 * out_cv

    met = metrics(y, pred_oof)
    repeat_mae = [metrics(y, repeat_preds[r])["mae"] for r in range(repeats)]
    row: Dict[str, object] = {
        "candidate": candidate_name,
        "target_mode": target_mode,
        "pca_dim": int(pca_dim),
        "n_train": int(n_train),
        "n_features": int(n_feat),
        "repeat_mae_mean": float(np.nanmean(repeat_mae)),
        "repeat_mae_std": float(np.nanstd(repeat_mae)),
        "pred_mean_addhout": float(np.nanmean(out_pred)),
        "pred_std_addhout": float(np.nanstd(out_pred)),
        "pred_min_addhout": float(np.nanmin(out_pred)),
        "pred_max_addhout": float(np.nanmax(out_pred)),
    }
    row.update(met)
    return row, pred_oof, out_pred


def merge_existing_predictions(out: pd.DataFrame, existing_csv: Path, existing_col: str) -> pd.DataFrame:
    out = out.copy()
    out["pred_existing_anchor"] = np.nan
    if existing_csv.exists():
        existing = pd.read_csv(existing_csv)
        if "id" in existing.columns and existing_col in existing.columns:
            small = existing[["id", existing_col]].drop_duplicates("id").rename(columns={existing_col: "pred_existing_anchor"})
            out = out.drop(columns=["pred_existing_anchor"], errors="ignore").merge(small, on="id", how="left")
    return out


def weighted_nan_average(preds: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    if not preds:
        return np.array([], dtype=float)
    M = np.vstack([np.asarray(p, dtype=float) for p in preds])
    w = np.asarray(weights, dtype=float).reshape(-1, 1)
    ok = np.isfinite(M)
    num = np.nansum(np.where(ok, M, 0.0) * w, axis=0)
    den = np.sum(np.where(ok, w, 0.0), axis=0)
    out = np.full(M.shape[1], np.nan, dtype=float)
    m = den > 0
    out[m] = num[m] / den[m]
    return out


def recenter_prediction(df: pd.DataFrame, pred_col: str, anchor_col: str, mode: str) -> pd.Series:
    pred = pd.to_numeric(df[pred_col], errors="coerce").copy()
    anchor = pd.to_numeric(df[anchor_col], errors="coerce")
    if mode == "none":
        return pred
    out = pred.copy()
    if mode == "global_median":
        m = pred.notna() & anchor.notna()
        if int(m.sum()) >= 3:
            out.loc[m] = pred.loc[m] + (float(anchor.loc[m].median()) - float(pred.loc[m].median()))
        return out
    if mode == "material_median" and "material" in df.columns:
        for _, g in df.groupby("material", dropna=False):
            idx = g.index
            m = pred.loc[idx].notna() & anchor.loc[idx].notna()
            if int(m.sum()) >= 3:
                sel = idx[m.to_numpy()]
                out.loc[sel] = pred.loc[sel] + (float(anchor.loc[sel].median()) - float(pred.loc[sel].median()))
        return out
    return pred


def posthoc_audit(pred: pd.DataFrame, labels_path: Path, target_col: str, out_dir: Path) -> pd.DataFrame:
    if not labels_path.exists():
        return pd.DataFrame()
    labels = pd.read_csv(labels_path)
    if "id" not in labels.columns or target_col not in labels.columns:
        return pd.DataFrame()
    keep = [c for c in ["id", target_col, "material", "dopant"] if c in labels.columns]
    detail = pred.merge(labels[keep].drop_duplicates("id"), on="id", how="left", suffixes=("", "_audit"))
    pred_cols = [
        c
        for c in [
            "pred_robust_retrain_final",
            "pred_robust_retrain_ensemble",
            "pred_robust_retrain_recentered",
            "pred_existing_anchor",
            "pred_dopant_prior_robust",
        ]
        if c in detail.columns
    ]
    audit = metric_rows(detail, pred_cols, target_col)
    audit.to_csv(out_dir / "robust_retrain_posthoc_audit.csv", index=False)
    detail["err_robust_retrain_final"] = (
        pd.to_numeric(detail["pred_robust_retrain_final"], errors="coerce")
        - pd.to_numeric(detail[target_col], errors="coerce")
    )
    detail["abs_err_robust_retrain_final"] = detail["err_robust_retrain_final"].abs()
    detail.sort_values("abs_err_robust_retrain_final", ascending=False).to_csv(
        out_dir / "robust_retrain_posthoc_audit_detail.csv", index=False
    )
    try:
        audit.to_excel(out_dir / "robust_retrain_posthoc_audit.xlsx", index=False)
    except Exception:
        pass
    return audit


def main() -> None:
    args = parse_args()
    cfg = PROFILE_CONFIG[args.profile]
    repeats = int(args.repeats or cfg["repeats"])
    top_k = int(args.top_k or cfg["top_k"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path, data, train_meta_all, addhout_meta_all = load_bundle(Path(args.bundle_dir), args.bundle_npz)
    y_all = train_meta_all["target"].to_numpy(float)
    source_mean = float(np.nanmean(y_all))
    source_std = float(np.nanstd(y_all))
    source_min = float(np.nanmin(y_all))
    source_max = float(np.nanmax(y_all))

    Xg_all = data["X_graph_train"].astype(np.float32)
    Xgo_all = data["X_graph_addhout"].astype(np.float32)
    Xt_all = data["X_tab_train"].astype(np.float32)
    Xto_all = data["X_tab_addhout"].astype(np.float32)
    train_has = train_meta_all["has_embedding"].to_numpy(bool)
    out_has = addhout_meta_all["has_embedding"].to_numpy(bool)

    global_folds = repeated_group_folds(train_meta_all["family_base"].to_numpy(), args.n_splits, repeats, args.seed)
    prior_oof_all, prior_out_all, prior_stats = fold_prior_oof(train_meta_all, addhout_meta_all, global_folds, repeats)

    print(f"[INFO] bundle={npz_path}")
    print(f"[INFO] profile={args.profile} repeats={repeats} n_splits={args.n_splits} top_k={top_k}")
    print(f"[INFO] train={len(train_meta_all)} addH-out={len(addhout_meta_all)}")
    print(f"[INFO] train embedding coverage={train_has.mean():.3f} addH-out embedding coverage={out_has.mean():.3f}")
    print(f"[BASE] dopant prior OOF MAE={prior_stats['prior_mae']:.4f} Spearman={prior_stats['prior_spearman']:.4f}")

    model_specs = []
    for family in cfg["models"]:
        model_specs.extend(model_factory(family, args.seed))

    rows: List[Dict[str, object]] = []
    oof_preds: Dict[str, np.ndarray] = {}
    out_preds: Dict[str, np.ndarray] = {}

    for mode in cfg["feature_modes"]:
        if mode == "tab":
            tr_mask = np.ones(len(train_meta_all), dtype=bool)
            out_mask = np.ones(len(addhout_meta_all), dtype=bool)
        else:
            tr_mask = train_has.copy()
            out_mask = out_has.copy()
        if int(tr_mask.sum()) < max(20, args.n_splits * 3) or int(out_mask.sum()) == 0:
            print(f"[WARN] skip mode={mode}: insufficient coverage")
            continue
        train_meta = train_meta_all.loc[tr_mask].reset_index(drop=True).copy()
        addhout_meta = addhout_meta_all.loc[out_mask].reset_index(drop=True).copy()
        Xg = Xg_all[tr_mask]
        Xgo = Xgo_all[out_mask]
        Xt = Xt_all[tr_mask]
        Xto = Xto_all[out_mask]
        try:
            X, Xout = make_feature_block(mode, Xg, Xgo, Xt, Xto)
        except Exception as exc:
            print(f"[WARN] skip mode={mode}: {exc}")
            continue
        folds = repeated_group_folds(train_meta["family_base"].to_numpy(), args.n_splits, repeats, args.seed)
        prior_oof, prior_out, local_prior_stats = fold_prior_oof(train_meta, addhout_meta, folds, repeats)
        prior_mae = float(local_prior_stats["prior_mae"])
        prior_spearman = float(local_prior_stats["prior_spearman"])

        for target_mode in cfg["target_modes"]:
            for pca_dim in cfg["pca_dims"]:
                if pca_dim > 0 and pca_dim >= min(X.shape[0], X.shape[1]):
                    continue
                for model_name, estimator in model_specs:
                    key = f"{target_mode}__{mode}__pca{pca_dim}__{model_name}"
                    try:
                        row, oof_local, out_local = train_candidate(
                            candidate_name=key,
                            estimator=estimator,
                            pca_dim=int(pca_dim),
                            target_mode=target_mode,
                            X=X,
                            Xout=Xout,
                            train_meta=train_meta,
                            addhout_meta=addhout_meta,
                            folds=folds,
                            repeats=repeats,
                            prior_oof=prior_oof,
                            prior_out=prior_out,
                            seed=args.seed,
                        )
                    except Exception as exc:
                        print(f"[WARN] candidate failed {key}: {exc}")
                        continue
                    row["feature_mode"] = mode
                    row["model"] = model_name
                    row["prior_mae_same_coverage"] = prior_mae
                    row["prior_spearman_same_coverage"] = prior_spearman
                    row["oof_improvement_vs_prior"] = prior_mae - float(row["mae"])
                    row["spearman_improvement_vs_prior"] = (
                        float(row["spearman"]) - prior_spearman
                        if np.isfinite(row["spearman"]) and np.isfinite(prior_spearman)
                        else 0.0
                    )
                    row["abs_mean_shift_vs_source"] = abs(float(row["pred_mean_addhout"]) - source_mean)
                    row["std_ratio_vs_source"] = float(row["pred_std_addhout"]) / max(source_std, 1e-6)
                    stability_penalty = 0.35 * float(row["repeat_mae_std"])
                    shift_penalty = 0.04 * float(row["abs_mean_shift_vs_source"])
                    score = (
                        float(row["oof_improvement_vs_prior"])
                        + 0.10 * max(0.0, float(row["spearman_improvement_vs_prior"]))
                        - stability_penalty
                        - shift_penalty
                    )
                    row["selection_score"] = float(score)

                    oof_full = np.full(len(train_meta_all), np.nan, dtype=float)
                    out_full = np.full(len(addhout_meta_all), np.nan, dtype=float)
                    oof_full[tr_mask] = oof_local
                    out_full[out_mask] = out_local
                    rows.append(row)
                    oof_preds[key] = oof_full
                    out_preds[key] = out_full
                    print(
                        f"[CAND] {key:44s} OOF={row['mae']:.4f} "
                        f"imp={row['oof_improvement_vs_prior']:.4f} "
                        f"std={row['repeat_mae_std']:.4f} score={row['selection_score']:.4f}"
                    )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise SystemExit("[ERROR] no candidate model succeeded.")
    summary = summary.sort_values(["selection_score", "mae"], ascending=[False, True]).reset_index(drop=True)
    summary.to_csv(out_dir / "robust_retrain_oof_metrics.csv", index=False)

    selected = summary[
        (summary["oof_improvement_vs_prior"] >= args.min_oof_improvement)
        & (summary["selection_score"] > 0)
        & (summary["abs_mean_shift_vs_source"] <= args.max_pred_mean_shift)
        & (summary["std_ratio_vs_source"] <= args.max_pred_std_ratio)
    ].head(top_k).copy()
    if selected.empty:
        print("[WARN] no candidate passed robust gates; using anchor only.")
        weights = np.array([], dtype=float)
    else:
        raw = selected["selection_score"].to_numpy(float) / np.maximum(selected["mae"].to_numpy(float), 1e-6) ** 2
        weights = raw / raw.sum()
        selected["ensemble_weight"] = weights
    selected.to_csv(out_dir / "robust_retrain_selected_models.csv", index=False)

    oof_table = train_meta_all[["id", "family_base", "material", "dopant", "target"]].copy()
    oof_table["oof_dopant_prior_robust"] = prior_oof_all
    for key in selected.get("candidate", pd.Series(dtype=str)).astype(str).tolist():
        oof_table[f"oof__{key}"] = oof_preds[key]
    if len(selected):
        oof_table["oof_robust_retrain_ensemble"] = weighted_nan_average(
            [oof_preds[str(k)] for k in selected["candidate"].astype(str).tolist()], weights
        )
    else:
        oof_table["oof_robust_retrain_ensemble"] = np.nan
    oof_table.to_csv(out_dir / "robust_retrain_oof_predictions.csv", index=False)

    pred = addhout_meta_all[["id", "material", "dopant", "has_embedding"]].copy()
    pred["pred_dopant_prior_robust"] = prior_out_all
    pred = merge_existing_predictions(pred, Path(args.existing_pred_csv), args.existing_pred_col)
    if pred["pred_existing_anchor"].notna().sum() == 0:
        pred["pred_existing_anchor"] = pred["pred_dopant_prior_robust"]
    if len(selected):
        pred["pred_robust_retrain_ensemble"] = weighted_nan_average(
            [out_preds[str(k)] for k in selected["candidate"].astype(str).tolist()], weights
        )
    else:
        pred["pred_robust_retrain_ensemble"] = np.nan
    pred["pred_robust_retrain_recentered"] = recenter_prediction(
        pred, "pred_robust_retrain_ensemble", "pred_existing_anchor", args.recenter_mode
    )

    if args.force_delta_weight >= 0:
        delta_weight = min(max(float(args.force_delta_weight), 0.0), 1.0)
    elif len(selected):
        best_imp = float(selected["oof_improvement_vs_prior"].max())
        delta_weight = min(float(args.max_delta_blend_weight), max(0.0, 0.80 * best_imp / max(prior_stats["prior_mae"], 1e-6)))
    else:
        delta_weight = 0.0

    anchor = pd.to_numeric(pred["pred_existing_anchor"], errors="coerce")
    delta = pd.to_numeric(pred["pred_robust_retrain_recentered"], errors="coerce")
    final = anchor.copy()
    m = anchor.notna() & delta.notna() & (delta_weight > 0)
    final.loc[m] = (1.0 - delta_weight) * anchor.loc[m] + delta_weight * delta.loc[m]
    final = final.where(final.notna(), pred["pred_dopant_prior_robust"])
    if args.clip_final_to_source_range:
        final = final.clip(source_min, source_max)
    pred["pred_robust_retrain_final"] = final
    pred["robust_retrain_delta_weight"] = float(delta_weight)
    pred["robust_retrain_profile"] = args.profile

    # Compatibility aliases for downstream rank-trend/superblend scripts.
    pred["pred_delta_head_ensemble"] = pred["pred_robust_retrain_ensemble"]
    pred["pred_delta_head_recentered"] = pred["pred_robust_retrain_recentered"]
    pred["pred_pretrained_delta_final"] = pred["pred_robust_retrain_final"]
    pred["pretrained_delta_blend_weight"] = float(delta_weight)
    pred["robust_retrain_rank"] = pred["pred_robust_retrain_final"].rank(method="average", ascending=True)
    pred = pred.sort_values(["robust_retrain_rank", "id"], na_position="last").reset_index(drop=True)
    pred.to_csv(out_dir / "robust_retrain_addhout_predictions.csv", index=False)
    try:
        pred.to_excel(out_dir / "robust_retrain_addhout_predictions.xlsx", index=False)
    except Exception:
        pass

    audit_labels: Optional[Path] = None
    if args.audit_labels_csv == "auto":
        for candidate in [Path(args.bundle_dir) / "addhout_audit_labels.csv", Path(args.bundle_dir).parent / "outputs_addh_llm_element_priors" / "addhout_audit_labels.csv"]:
            if candidate.exists():
                audit_labels = candidate
                break
    elif args.audit_labels_csv:
        audit_labels = Path(args.audit_labels_csv)
    audit = pd.DataFrame()
    if audit_labels is not None and audit_labels.exists():
        audit = posthoc_audit(pred, audit_labels, args.audit_target_col, out_dir)
        if len(audit):
            print("[POSTHOC AUDIT ONLY]")
            print(audit.to_string(index=False))

    manifest = {
        "bundle_npz": str(npz_path),
        "out_dir": str(out_dir),
        "profile": args.profile,
        "repeats": repeats,
        "n_splits": args.n_splits,
        "top_k": top_k,
        "n_train": int(len(train_meta_all)),
        "n_addhout": int(len(addhout_meta_all)),
        "n_selected": int(len(selected)),
        "delta_weight": float(delta_weight),
        "labels_used_for_training_or_selection": False,
        "audit_labels_csv": str(audit_labels) if audit_labels else "",
        "source_stats": {"mean": source_mean, "std": source_std, "min": source_min, "max": source_max},
        "prior_stats": prior_stats,
        "profile_config": cfg,
        "outputs": {
            "predictions_csv": str(out_dir / "robust_retrain_addhout_predictions.csv"),
            "audit_csv": str(out_dir / "robust_retrain_posthoc_audit.csv"),
        },
    }
    with (out_dir / "robust_retrain_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] wrote {out_dir / 'robust_retrain_addhout_predictions.csv'}")
    print(f"[OK] selected_models={len(selected)} delta_weight={delta_weight:.4f}")


if __name__ == "__main__":
    main()
