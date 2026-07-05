#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Domain-adapted AddH/AddH-2 -> AddH-out retraining.

This is the next step after a normal grouped-CV model fails under AddH-out
domain shift. It still trains only on addH/addH-2 labels, but uses unlabeled
AddH-out feature distribution in three strict-safe ways:

  1. domain-classifier importance weights for source training rows;
  2. target-like OOF validation subset for model selection;
  3. addH-out prediction distribution gates before blending with the anchor.

AddH-out labels, if supplied, are used only for post-hoc audit files.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Domain-adapted AddH-out retraining.")
    ap.add_argument("--bundle-dir", default="outputs_addh_pretrained_delta_features")
    ap.add_argument("--bundle-npz", default="")
    ap.add_argument("--out-dir", default="outputs_addh_domain_adapted_delta")
    ap.add_argument("--existing-pred-csv", default="outputs_addh_target_calibrated_fast/target_calibrated_addhout_predictions.csv")
    ap.add_argument("--existing-pred-col", default="pred_fast_target_calibrated")
    ap.add_argument("--audit-labels-csv", default="auto")
    ap.add_argument("--audit-target-col", default="h_ads_excel")
    ap.add_argument("--profile", choices=["fast", "medium", "thorough"], default="fast")
    ap.add_argument("--n-splits", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=0, help="0 means profile default.")
    ap.add_argument("--top-k", type=int, default=0, help="0 means profile default.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-like-frac", type=float, default=0.35)
    ap.add_argument("--domain-weight-strength", type=float, default=0.70)
    ap.add_argument("--domain-weight-min", type=float, default=0.25)
    ap.add_argument("--domain-weight-max", type=float, default=4.00)
    ap.add_argument("--min-source-improvement", type=float, default=-0.02)
    ap.add_argument("--min-targetlike-improvement", type=float, default=0.03)
    ap.add_argument("--min-anchor-spearman", type=float, default=0.10)
    ap.add_argument("--max-anchor-median-abs-delta", type=float, default=2.00)
    ap.add_argument("--max-mean-shift-vs-anchor", type=float, default=1.20)
    ap.add_argument("--max-std-ratio-vs-anchor", type=float, default=2.25)
    ap.add_argument("--max-blend-weight", type=float, default=0.14)
    ap.add_argument("--force-blend-weight", type=float, default=-1.0)
    ap.add_argument("--recenter-mode", choices=["none", "global_median", "material_median"], default="material_median")
    ap.add_argument("--clip-final-to-source-range", action="store_true")
    return ap.parse_args()


PROFILE_CONFIG = {
    "fast": {
        "repeats": 2,
        "top_k": 8,
        "feature_modes": ["tab", "delta", "addh_delta", "bare_delta", "full"],
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
        "repeats": 4,
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
        "pearson": finite_corr(yy, pp),
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
        raise SystemExit(f"[ERROR] missing bundle: {npz_path}")
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
    folds: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for r in range(int(repeats)):
        rng = np.random.default_rng(seed + 9176 * r)
        shuffled = rng.permutation(uniq)
        fold_groups = [set(shuffled[i::n]) for i in range(n)]
        for gset in fold_groups:
            va = np.where(np.isin(groups, list(gset)))[0]
            tr = np.where(~np.isin(groups, list(gset)))[0]
            if len(tr) and len(va):
                folds.append((r, tr.astype(int), va.astype(int)))
    return folds


def weighted_mean(values: np.ndarray, weights: Optional[np.ndarray]) -> float:
    v = np.asarray(values, dtype=float)
    m = np.isfinite(v)
    if weights is None:
        return float(np.nanmean(v[m]))
    w = np.asarray(weights, dtype=float)[m]
    if w.sum() <= 1e-12:
        return float(np.nanmean(v[m]))
    return float(np.sum(v[m] * w) / np.sum(w))


def weighted_dopant_prior(train_df: pd.DataFrame, apply_df: pd.DataFrame, sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
    tmp = train_df[["dopant", "target"]].copy()
    tmp["_w"] = 1.0 if sample_weight is None else np.asarray(sample_weight, dtype=float)
    means: Dict[str, float] = {}
    for dop, g in tmp.groupby("dopant"):
        means[str(dop)] = weighted_mean(g["target"].to_numpy(float), g["_w"].to_numpy(float))
    fallback = weighted_mean(tmp["target"].to_numpy(float), tmp["_w"].to_numpy(float))
    return apply_df["dopant"].astype(str).map(means).fillna(fallback).to_numpy(float)


def fold_prior_oof(
    train_meta: pd.DataFrame,
    addhout_meta: pd.DataFrame,
    folds: List[Tuple[int, np.ndarray, np.ndarray]],
    repeats: int,
    sample_weight: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(train_meta)
    sums = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=float)
    out_preds: List[np.ndarray] = []
    for r, tr, va in folds:
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)[tr]
        p = weighted_dopant_prior(train_meta.iloc[tr], train_meta.iloc[va], sw)
        sums[va] += p
        counts[va] += 1.0
        out_preds.append(weighted_dopant_prior(train_meta.iloc[tr], addhout_meta, sw))
    oof = np.full(n, np.nan, dtype=float)
    m = counts > 0
    oof[m] = sums[m] / counts[m]
    sw_full = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    out_full = weighted_dopant_prior(train_meta, addhout_meta, sw_full)
    if out_preds:
        out_cv = np.nanmean(np.vstack(out_preds), axis=0)
        out_full = 0.85 * out_full + 0.15 * out_cv
    return oof, out_full


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
    raise ValueError(f"unknown feature mode: {mode}")


def make_feature_block(mode: str, Xg: np.ndarray, Xgo: np.ndarray, Xt: np.ndarray, Xto: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "tab":
        if Xt.shape[1] == 0:
            raise ValueError("tab mode requested but tabular dimension is zero")
        return Xt, Xto
    G = select_graph_mode(Xg, mode)
    Go = select_graph_mode(Xgo, mode)
    if Xt.shape[1]:
        return np.hstack([G, Xt]), np.hstack([Go, Xto])
    return G, Go


def preprocessor(n_rows: int, n_features: int, max_components: int = 64) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if n_features > max_components and n_rows > 4:
        steps.append(("pca", PCA(n_components=min(max_components, n_features, n_rows - 1), random_state=42)))
    return Pipeline(steps)


def domain_weights_and_targetlike(
    X: np.ndarray,
    Xout: np.ndarray,
    target_like_frac: float,
    strength: float,
    wmin: float,
    wmax: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    X = np.asarray(X, dtype=float)
    Xout = np.asarray(Xout, dtype=float)
    n_src = X.shape[0]
    n_tar = Xout.shape[0]
    weights = np.ones(n_src, dtype=float)
    info: Dict[str, float] = {"domain_auc": np.nan, "target_like_n": 0, "weight_min": 1.0, "weight_max": 1.0}

    prep = preprocessor(n_src + n_tar, X.shape[1], 64)
    Z = prep.fit_transform(np.vstack([X, Xout]))
    Zs = Z[:n_src]
    Zt = Z[n_src:]

    centroid_dist = np.linalg.norm(Zs - np.nanmean(Zt, axis=0), axis=1)
    nearest_dist = euclidean_distances(Zs, Zt).min(axis=1)
    sim_score = -0.50 * centroid_dist - 0.50 * nearest_dist
    cutoff = np.nanquantile(sim_score, max(0.0, min(1.0, 1.0 - float(target_like_frac))))
    target_like = sim_score >= cutoff
    info["target_like_n"] = int(target_like.sum())
    info["target_like_frac_actual"] = float(target_like.mean())
    info["target_similarity_min"] = float(np.nanmin(sim_score))
    info["target_similarity_max"] = float(np.nanmax(sim_score))

    try:
        clf_steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        if X.shape[1] > 64 and (n_src + n_tar) > 5:
            clf_steps.append(("pca", PCA(n_components=min(64, X.shape[1], n_src + n_tar - 1), random_state=seed)))
        clf_steps.append(
            (
                "clf",
                LogisticRegression(
                    C=0.5,
                    max_iter=1200,
                    class_weight="balanced",
                    random_state=seed,
                ),
            )
        )
        clf = Pipeline(clf_steps)
        Xd = np.vstack([X, Xout])
        yd = np.r_[np.zeros(n_src, dtype=int), np.ones(n_tar, dtype=int)]
        clf.fit(Xd, yd)
        proba_all = clf.predict_proba(Xd)[:, 1]
        auc = float(roc_auc_score(yd, proba_all))
        p_src = np.clip(proba_all[:n_src], 1e-4, 1 - 1e-4)
        odds = p_src / (1.0 - p_src)
        odds = odds / max(float(np.nanmean(odds)), 1e-12)
        weights = (1.0 - strength) + strength * odds
        weights = np.clip(weights, wmin, wmax)
        weights = weights / max(float(np.nanmean(weights)), 1e-12)
        info["domain_auc"] = auc
        info["weight_min"] = float(np.nanmin(weights))
        info["weight_max"] = float(np.nanmax(weights))
        info["weight_std"] = float(np.nanstd(weights))
    except Exception as exc:
        print(f"[WARN] domain classifier failed; using uniform weights: {exc}")
    return weights.astype(float), target_like.astype(bool), info


def model_factory(family: str, seed: int):
    family = family.lower()
    if family == "ridge":
        return [(f"ridge_a{a:g}", Ridge(alpha=a)) for a in [3.0, 10.0, 30.0, 100.0, 300.0]]
    if family == "huber":
        return [
            (f"huber_a{a:g}_e{eps:g}", HuberRegressor(alpha=a, epsilon=eps, max_iter=900))
            for a in [0.0005, 0.002, 0.01]
            for eps in [1.15, 1.35]
        ]
    if family == "elastic":
        return [
            (f"elastic_a{a:g}", ElasticNet(alpha=a, l1_ratio=0.20, max_iter=7000, random_state=seed))
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


def make_regression_pipeline(estimator, pca_dim: int, n_train: int, n_features: int) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if pca_dim > 0:
        steps.append(("pca", PCA(n_components=min(int(pca_dim), max(1, n_train - 1), n_features), random_state=42)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def fit_pipeline(pipe: Pipeline, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray]) -> Pipeline:
    if sample_weight is None:
        pipe.fit(X, y)
        return pipe
    try:
        pipe.fit(X, y, model__sample_weight=np.asarray(sample_weight, dtype=float))
    except Exception:
        pipe.fit(X, y)
    return pipe


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
    domain_weight: np.ndarray,
    prior_oof: np.ndarray,
    prior_out: np.ndarray,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    y = train_meta["target"].to_numpy(float)
    n_train, n_feat = X.shape
    pred_sums = np.zeros(n_train, dtype=float)
    pred_counts = np.zeros(n_train, dtype=float)
    repeat_preds: Dict[int, np.ndarray] = {r: np.full(n_train, np.nan, dtype=float) for r in range(repeats)}
    out_fold_preds: List[np.ndarray] = []

    for r, tr, va in folds:
        pipe = make_regression_pipeline(clone(estimator), pca_dim, len(tr), n_feat)
        sw_tr = domain_weight[tr]
        if target_mode == "residual":
            tr_prior = weighted_dopant_prior(train_meta.iloc[tr], train_meta.iloc[tr], sw_tr)
            y_fit = y[tr] - tr_prior
        else:
            y_fit = y[tr]
        fit_pipeline(pipe, X[tr], y_fit, sw_tr)
        va_pred = pipe.predict(X[va])
        out_pred = pipe.predict(Xout)
        if target_mode == "residual":
            va_pred = prior_oof[va] + va_pred
            out_pred = prior_out + out_pred
        pred_sums[va] += va_pred
        pred_counts[va] += 1.0
        repeat_preds[r][va] = va_pred
        out_fold_preds.append(out_pred)

    pred_oof = np.full(n_train, np.nan, dtype=float)
    m = pred_counts > 0
    pred_oof[m] = pred_sums[m] / pred_counts[m]

    pipe = make_regression_pipeline(clone(estimator), pca_dim, n_train, n_feat)
    if target_mode == "residual":
        full_prior_train = weighted_dopant_prior(train_meta, train_meta, domain_weight)
        y_fit_full = y - full_prior_train
    else:
        y_fit_full = y
    fit_pipeline(pipe, X, y_fit_full, domain_weight)
    out_full = pipe.predict(Xout)
    if target_mode == "residual":
        out_full = prior_out + out_full
    out_cv = np.nanmean(np.vstack(out_fold_preds), axis=0) if out_fold_preds else out_full
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
            "pred_domain_adapted_final",
            "pred_domain_adapted_ensemble",
            "pred_domain_adapted_recentered",
            "pred_existing_anchor",
            "pred_domain_dopant_prior",
        ]
        if c in detail.columns
    ]
    audit = metric_rows(detail, pred_cols, target_col)
    audit.to_csv(out_dir / "domain_adapted_posthoc_audit.csv", index=False)
    detail["err_domain_adapted_final"] = (
        pd.to_numeric(detail["pred_domain_adapted_final"], errors="coerce")
        - pd.to_numeric(detail[target_col], errors="coerce")
    )
    detail["abs_err_domain_adapted_final"] = detail["err_domain_adapted_final"].abs()
    detail.sort_values("abs_err_domain_adapted_final", ascending=False).to_csv(
        out_dir / "domain_adapted_posthoc_audit_detail.csv", index=False
    )
    try:
        audit.to_excel(out_dir / "domain_adapted_posthoc_audit.xlsx", index=False)
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
    Xg_all = data["X_graph_train"].astype(np.float32)
    Xgo_all = data["X_graph_addhout"].astype(np.float32)
    Xt_all = data["X_tab_train"].astype(np.float32)
    Xto_all = data["X_tab_addhout"].astype(np.float32)
    train_has = train_meta_all["has_embedding"].to_numpy(bool)
    out_has = addhout_meta_all["has_embedding"].to_numpy(bool)

    source_min = float(train_meta_all["target"].min())
    source_max = float(train_meta_all["target"].max())

    anchor_pred = merge_existing_predictions(
        addhout_meta_all[["id", "material", "dopant", "has_embedding"]].copy(),
        Path(args.existing_pred_csv),
        args.existing_pred_col,
    )
    anchor = pd.to_numeric(anchor_pred["pred_existing_anchor"], errors="coerce")
    anchor_mean = float(anchor.mean()) if anchor.notna().any() else float(train_meta_all["target"].mean())
    anchor_std = float(anchor.std()) if anchor.notna().sum() >= 3 else float(train_meta_all["target"].std())

    print(f"[INFO] bundle={npz_path}")
    print(f"[INFO] profile={args.profile} repeats={repeats} n_splits={args.n_splits} top_k={top_k}")
    print(f"[INFO] train={len(train_meta_all)} addH-out={len(addhout_meta_all)}")
    print(f"[INFO] embedding coverage train={train_has.mean():.3f} addH-out={out_has.mean():.3f}")
    print(f"[INFO] anchor mean={anchor_mean:.4f} std={anchor_std:.4f}")

    model_specs = []
    for family in cfg["models"]:
        model_specs.extend(model_factory(family, args.seed))

    rows: List[Dict[str, object]] = []
    oof_preds: Dict[str, np.ndarray] = {}
    out_preds: Dict[str, np.ndarray] = {}
    domain_summaries: List[Dict[str, object]] = []

    for mode in cfg["feature_modes"]:
        if mode == "tab":
            tr_mask = np.ones(len(train_meta_all), dtype=bool)
            out_mask = np.ones(len(addhout_meta_all), dtype=bool)
        else:
            tr_mask = train_has.copy()
            out_mask = out_has.copy()
        if int(tr_mask.sum()) < max(24, args.n_splits * 4) or int(out_mask.sum()) < 3:
            print(f"[WARN] skip mode={mode}: insufficient coverage")
            continue
        train_meta = train_meta_all.loc[tr_mask].reset_index(drop=True).copy()
        addhout_meta = addhout_meta_all.loc[out_mask].reset_index(drop=True).copy()
        try:
            X, Xout = make_feature_block(
                mode,
                Xg_all[tr_mask],
                Xgo_all[out_mask],
                Xt_all[tr_mask],
                Xto_all[out_mask],
            )
        except Exception as exc:
            print(f"[WARN] skip mode={mode}: {exc}")
            continue

        domain_weight, target_like, domain_info = domain_weights_and_targetlike(
            X=X,
            Xout=Xout,
            target_like_frac=args.target_like_frac,
            strength=args.domain_weight_strength,
            wmin=args.domain_weight_min,
            wmax=args.domain_weight_max,
            seed=args.seed,
        )
        domain_info["feature_mode"] = mode
        domain_info["n_train"] = int(len(train_meta))
        domain_info["n_addhout"] = int(len(addhout_meta))
        domain_summaries.append(domain_info)
        folds = repeated_group_folds(train_meta["family_base"].to_numpy(), args.n_splits, repeats, args.seed)
        prior_oof, prior_out = fold_prior_oof(train_meta, addhout_meta, folds, repeats, domain_weight)
        prior_all = metrics(train_meta["target"], prior_oof)
        prior_tl = metrics(train_meta.loc[target_like, "target"], pd.Series(prior_oof)[target_like])
        print(
            f"[MODE] {mode} domain_auc={domain_info['domain_auc']:.3f} "
            f"target_like={int(target_like.sum())}/{len(target_like)} "
            f"prior_mae={prior_all['mae']:.4f} prior_tl_mae={prior_tl['mae']:.4f}"
        )

        anchor_local = anchor_pred.loc[out_mask, "pred_existing_anchor"].reset_index(drop=True)
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
                            domain_weight=domain_weight,
                            prior_oof=prior_oof,
                            prior_out=prior_out,
                        )
                    except Exception as exc:
                        print(f"[WARN] candidate failed {key}: {exc}")
                        continue

                    targetlike_met = metrics(train_meta.loc[target_like, "target"], pd.Series(oof_local)[target_like])
                    source_improve = float(prior_all["mae"]) - float(row["mae"])
                    targetlike_improve = float(prior_tl["mae"]) - float(targetlike_met["mae"])
                    targetlike_spearman_improve = (
                        float(targetlike_met["spearman"]) - float(prior_tl["spearman"])
                        if np.isfinite(targetlike_met["spearman"]) and np.isfinite(prior_tl["spearman"])
                        else 0.0
                    )
                    out_ser = pd.Series(out_local)
                    anchor_ser = pd.to_numeric(anchor_local, errors="coerce")
                    m_anchor = out_ser.notna() & anchor_ser.notna()
                    anchor_spearman = finite_corr(anchor_ser[m_anchor], out_ser[m_anchor], True) if int(m_anchor.sum()) >= 3 else np.nan
                    anchor_mad = float((out_ser[m_anchor] - anchor_ser[m_anchor]).abs().median()) if int(m_anchor.sum()) else np.nan
                    mean_shift_anchor = abs(float(np.nanmean(out_local)) - anchor_mean)
                    std_ratio_anchor = float(np.nanstd(out_local)) / max(anchor_std, 1e-6)

                    score = (
                        0.62 * targetlike_improve
                        + 0.30 * source_improve
                        + 0.08 * max(0.0, targetlike_spearman_improve)
                        + 0.04 * max(0.0, anchor_spearman if np.isfinite(anchor_spearman) else 0.0)
                        - 0.35 * float(row["repeat_mae_std"])
                        - 0.05 * mean_shift_anchor
                        - 0.03 * (anchor_mad if np.isfinite(anchor_mad) else 2.0)
                    )

                    row["feature_mode"] = mode
                    row["model"] = model_name
                    row["domain_auc"] = domain_info["domain_auc"]
                    row["domain_weight_min"] = domain_info["weight_min"]
                    row["domain_weight_max"] = domain_info["weight_max"]
                    row["target_like_n"] = int(target_like.sum())
                    row["prior_mae"] = prior_all["mae"]
                    row["prior_targetlike_mae"] = prior_tl["mae"]
                    row["targetlike_mae"] = targetlike_met["mae"]
                    row["targetlike_rmse"] = targetlike_met["rmse"]
                    row["targetlike_spearman"] = targetlike_met["spearman"]
                    row["source_improvement_vs_prior"] = source_improve
                    row["targetlike_improvement_vs_prior"] = targetlike_improve
                    row["targetlike_spearman_improvement_vs_prior"] = targetlike_spearman_improve
                    row["anchor_spearman_addhout"] = anchor_spearman
                    row["anchor_median_abs_delta_addhout"] = anchor_mad
                    row["mean_shift_vs_anchor"] = mean_shift_anchor
                    row["std_ratio_vs_anchor"] = std_ratio_anchor
                    row["selection_score"] = float(score)

                    oof_full = np.full(len(train_meta_all), np.nan, dtype=float)
                    out_full = np.full(len(addhout_meta_all), np.nan, dtype=float)
                    oof_full[tr_mask] = oof_local
                    out_full[out_mask] = out_local
                    rows.append(row)
                    oof_preds[key] = oof_full
                    out_preds[key] = out_full
                    print(
                        f"[CAND] {key:44s} src={row['mae']:.4f} tl={targetlike_met['mae']:.4f} "
                        f"tl_imp={targetlike_improve:.4f} score={score:.4f} "
                        f"anchor_sp={anchor_spearman if np.isfinite(anchor_spearman) else np.nan:.3f}"
                    )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise SystemExit("[ERROR] no candidate succeeded.")
    summary = summary.sort_values(["selection_score", "targetlike_mae", "mae"], ascending=[False, True, True]).reset_index(drop=True)
    summary.to_csv(out_dir / "domain_adapted_oof_metrics.csv", index=False)
    pd.DataFrame(domain_summaries).to_csv(out_dir / "domain_adaptation_summary.csv", index=False)

    selected = summary[
        (summary["source_improvement_vs_prior"] >= args.min_source_improvement)
        & (summary["targetlike_improvement_vs_prior"] >= args.min_targetlike_improvement)
        & (summary["selection_score"] > 0)
        & (summary["mean_shift_vs_anchor"] <= args.max_mean_shift_vs_anchor)
        & (summary["std_ratio_vs_anchor"] <= args.max_std_ratio_vs_anchor)
        & (summary["anchor_median_abs_delta_addhout"] <= args.max_anchor_median_abs_delta)
        & (summary["anchor_spearman_addhout"].fillna(-1.0) >= args.min_anchor_spearman)
    ].head(top_k).copy()

    if selected.empty:
        print("[WARN] no model passed domain-adaptation gates; final will keep anchor.")
        weights = np.array([], dtype=float)
    else:
        raw = selected["selection_score"].to_numpy(float) / np.maximum(selected["targetlike_mae"].to_numpy(float), 1e-6) ** 2
        weights = raw / raw.sum()
        selected["ensemble_weight"] = weights
    selected.to_csv(out_dir / "domain_adapted_selected_models.csv", index=False)

    oof_table = train_meta_all[["id", "family_base", "material", "dopant", "target"]].copy()
    for key in selected.get("candidate", pd.Series(dtype=str)).astype(str).tolist():
        oof_table[f"oof__{key}"] = oof_preds[key]
    if len(selected):
        oof_table["oof_domain_adapted_ensemble"] = weighted_nan_average(
            [oof_preds[str(k)] for k in selected["candidate"].astype(str).tolist()], weights
        )
    else:
        oof_table["oof_domain_adapted_ensemble"] = np.nan
    oof_table.to_csv(out_dir / "domain_adapted_oof_predictions.csv", index=False)

    pred = anchor_pred.copy()
    if len(selected):
        pred["pred_domain_adapted_ensemble"] = weighted_nan_average(
            [out_preds[str(k)] for k in selected["candidate"].astype(str).tolist()], weights
        )
    else:
        pred["pred_domain_adapted_ensemble"] = np.nan
    global_folds = repeated_group_folds(train_meta_all["family_base"].to_numpy(), args.n_splits, repeats, args.seed)
    prior_oof_all, prior_out_all = fold_prior_oof(train_meta_all, addhout_meta_all, global_folds, repeats, None)
    pred["pred_domain_dopant_prior"] = prior_out_all
    pred["pred_domain_adapted_recentered"] = recenter_prediction(
        pred, "pred_domain_adapted_ensemble", "pred_existing_anchor", args.recenter_mode
    )

    if args.force_blend_weight >= 0:
        blend_weight = min(max(float(args.force_blend_weight), 0.0), 1.0)
    elif len(selected):
        best_tl = float(selected["targetlike_improvement_vs_prior"].max())
        denom = max(float(summary["prior_targetlike_mae"].median()), 1e-6)
        blend_weight = min(float(args.max_blend_weight), max(0.0, 0.65 * best_tl / denom))
    else:
        blend_weight = 0.0

    anchor_final = pd.to_numeric(pred["pred_existing_anchor"], errors="coerce")
    adapted = pd.to_numeric(pred["pred_domain_adapted_recentered"], errors="coerce")
    final = anchor_final.copy()
    m = anchor_final.notna() & adapted.notna() & (blend_weight > 0)
    final.loc[m] = (1.0 - blend_weight) * anchor_final.loc[m] + blend_weight * adapted.loc[m]
    final = final.where(final.notna(), pred["pred_domain_dopant_prior"])
    if args.clip_final_to_source_range:
        final = final.clip(source_min, source_max)
    pred["pred_domain_adapted_final"] = final
    pred["domain_adapted_blend_weight"] = float(blend_weight)
    pred["domain_adapted_profile"] = args.profile

    # Compatibility aliases for rank-trend and superblend scripts.
    pred["pred_delta_head_ensemble"] = pred["pred_domain_adapted_ensemble"]
    pred["pred_delta_head_recentered"] = pred["pred_domain_adapted_recentered"]
    pred["pred_pretrained_delta_final"] = pred["pred_domain_adapted_final"]
    pred["pretrained_delta_blend_weight"] = float(blend_weight)
    pred["domain_adapted_rank"] = pred["pred_domain_adapted_final"].rank(method="average", ascending=True)
    pred = pred.sort_values(["domain_adapted_rank", "id"], na_position="last").reset_index(drop=True)
    pred.to_csv(out_dir / "domain_adapted_addhout_predictions.csv", index=False)
    try:
        pred.to_excel(out_dir / "domain_adapted_addhout_predictions.xlsx", index=False)
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
        "blend_weight": float(blend_weight),
        "labels_used_for_training_or_selection": False,
        "audit_labels_csv": str(audit_labels) if audit_labels else "",
        "domain_weight_strength": args.domain_weight_strength,
        "target_like_frac": args.target_like_frac,
        "selection_gates": {
            "min_source_improvement": args.min_source_improvement,
            "min_targetlike_improvement": args.min_targetlike_improvement,
            "min_anchor_spearman": args.min_anchor_spearman,
            "max_anchor_median_abs_delta": args.max_anchor_median_abs_delta,
            "max_mean_shift_vs_anchor": args.max_mean_shift_vs_anchor,
            "max_std_ratio_vs_anchor": args.max_std_ratio_vs_anchor,
        },
        "outputs": {
            "predictions_csv": str(out_dir / "domain_adapted_addhout_predictions.csv"),
            "audit_csv": str(out_dir / "domain_adapted_posthoc_audit.csv"),
        },
    }
    with (out_dir / "domain_adapted_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] wrote {out_dir / 'domain_adapted_addhout_predictions.csv'}")
    print(f"[OK] selected_models={len(selected)} blend_weight={blend_weight:.4f}")


if __name__ == "__main__":
    main()
