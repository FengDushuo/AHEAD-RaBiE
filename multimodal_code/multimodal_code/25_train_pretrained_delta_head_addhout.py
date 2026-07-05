#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train frozen-pretrained-embedding delta heads for AddH-out prediction.

This is a small-data fine-tuning layer, not a full backbone fine-tune:
  - input graph features come from pretrained FAIR-Chem/EquiformerV2 embeddings;
  - labels are addH/addH-2 adsorption energies only;
  - grouped OOF selection protects against overfitting;
  - final prediction is conservatively blended with the current best calibrated
    target-domain prediction when available.

AddH-out labels, if supplied, are used only for post-hoc audit files.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train pretrained delta embedding heads for AddH-out.")
    ap.add_argument("--bundle-dir", default="outputs_addh_pretrained_delta_features")
    ap.add_argument("--bundle-npz", default="")
    ap.add_argument("--out-dir", default="outputs_addh_pretrained_delta_head")
    ap.add_argument("--existing-pred-csv", default="outputs_addh_target_calibrated_fast/target_calibrated_addhout_predictions.csv")
    ap.add_argument("--existing-pred-col", default="pred_fast_target_calibrated")
    ap.add_argument("--llm-pred-csv", default="")
    ap.add_argument("--strict-pred-csv", default="")
    ap.add_argument("--audit-labels-csv", default="auto")
    ap.add_argument("--audit-target-col", default="h_ads_excel")
    ap.add_argument("--n-splits", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feature-modes", default="delta,addh_delta,bare_delta,full")
    ap.add_argument("--pca-dims", default="0,16,32,64,128")
    ap.add_argument("--model-names", default="ridge,huber,elastic,extratrees,rf,hgb,gbr,knn,mlp")
    ap.add_argument("--target-modes", default="residual,absolute")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--min-oof-improvement", type=float, default=0.05)
    ap.add_argument("--max-pred-mean-shift", type=float, default=2.0)
    ap.add_argument("--max-delta-blend-weight", type=float, default=0.20)
    ap.add_argument("--force-delta-weight", type=float, default=-1.0)
    ap.add_argument("--recenter-delta-to-anchor", action="store_true", default=True)
    ap.add_argument("--no-recenter-delta-to-anchor", action="store_false", dest="recenter_delta_to_anchor")
    ap.add_argument("--clip-final-to-source-range", action="store_true")
    ap.add_argument("--oracle-diagnostic-tune", action="store_true")
    return ap.parse_args()


def parse_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def parse_int_list(raw: str) -> List[int]:
    vals = []
    for x in parse_list(raw):
        vals.append(int(float(x)))
    return sorted(set(vals))


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
    if np.std(xv) <= 1e-12 or np.std(yv) <= 1e-12:
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
            row = {"pred_col": c, "target_col": target_col, "material": mat}
            row.update(metrics(d[target_col], d[c]))
            rows.append(row)
    return pd.DataFrame(rows)


def select_graph_mode(X: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.strip().lower()
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


def make_group_folds(groups: np.ndarray, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray([str(g) for g in groups])
    n = max(2, min(int(n_splits), len(pd.unique(groups))))
    return [(np.asarray(tr, dtype=int), np.asarray(va, dtype=int)) for tr, va in GroupKFold(n_splits=n).split(np.zeros(len(groups)), groups=groups)]


def dopant_prior(train_df: pd.DataFrame, apply_df: pd.DataFrame) -> np.ndarray:
    agg = train_df.groupby("dopant")["target"].mean()
    fallback = float(train_df["target"].mean())
    return apply_df["dopant"].map(agg).fillna(fallback).to_numpy(float)


def fold_dopant_oof(train_meta: pd.DataFrame, folds: List[Tuple[np.ndarray, np.ndarray]], addhout_meta: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    y = train_meta["target"].to_numpy(float)
    oof = np.full(len(train_meta), np.nan, dtype=float)
    out_folds = []
    for tr, va in folds:
        tr_df = train_meta.iloc[tr]
        oof[va] = dopant_prior(tr_df, train_meta.iloc[va])
        out_folds.append(dopant_prior(tr_df, addhout_meta))
    out_full = dopant_prior(train_meta, addhout_meta)
    if out_folds:
        out_full = 0.90 * out_full + 0.10 * np.nanmean(np.vstack(out_folds), axis=0)
    return oof, out_full


def model_factory(name: str, seed: int):
    name = name.strip().lower()
    if name == "ridge":
        return [
            (f"ridge_a{a:g}", Ridge(alpha=a))
            for a in [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
        ]
    if name == "huber":
        return [
            (f"huber_a{a:g}", HuberRegressor(alpha=a, epsilon=eps, max_iter=800))
            for a in [0.0001, 0.001, 0.01]
            for eps in [1.15, 1.35]
        ]
    if name == "elastic":
        return [
            (f"elastic_a{a:g}", ElasticNet(alpha=a, l1_ratio=0.2, max_iter=5000, random_state=seed))
            for a in [0.001, 0.003, 0.01, 0.03, 0.1]
        ]
    if name == "extratrees":
        return [
            ("extratrees", ExtraTreesRegressor(n_estimators=320, min_samples_leaf=3, max_features=0.65, random_state=seed, n_jobs=-1))
        ]
    if name == "rf":
        return [
            ("rf", RandomForestRegressor(n_estimators=360, min_samples_leaf=3, max_features=0.65, random_state=seed, n_jobs=-1))
        ]
    if name == "hgb":
        return [
            ("hgb", HistGradientBoostingRegressor(max_iter=260, learning_rate=0.025, l2_regularization=1.0, max_leaf_nodes=8, random_state=seed))
        ]
    if name == "gbr":
        return [
            ("gbr", GradientBoostingRegressor(n_estimators=220, learning_rate=0.025, max_depth=2, random_state=seed))
        ]
    if name == "knn":
        return [
            (f"knn_k{k}", KNeighborsRegressor(n_neighbors=k, weights="distance"))
            for k in [5, 9, 15, 25, 35]
        ]
    if name == "mlp":
        return [
            (
                "mlp_small",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    alpha=0.02,
                    learning_rate_init=0.001,
                    max_iter=800,
                    early_stopping=True,
                    validation_fraction=0.2,
                    random_state=seed,
                ),
            )
        ]
    return []


def make_pipeline_model(model, pca_dim: int, n_train: int, n_features: int) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if pca_dim > 0:
        n_comp = min(int(pca_dim), max(1, n_train - 1), n_features)
        steps.append(("pca", PCA(n_components=n_comp, random_state=42)))
    steps.append(("model", model))
    return Pipeline(steps)


def train_candidate(
    model_name: str,
    estimator,
    feature_mode: str,
    pca_dim: int,
    target_mode: str,
    Xg: np.ndarray,
    Xg_out: np.ndarray,
    Xt: np.ndarray,
    Xt_out: np.ndarray,
    train_meta: pd.DataFrame,
    addhout_meta: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    prior_oof: np.ndarray,
    prior_out: np.ndarray,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    y = train_meta["target"].to_numpy(float)
    G = select_graph_mode(Xg, feature_mode)
    Go = select_graph_mode(Xg_out, feature_mode)
    X = np.hstack([G, Xt]) if Xt.shape[1] else G
    Xout = np.hstack([Go, Xt_out]) if Xt_out.shape[1] else Go
    n_train, n_feat = X.shape
    pred_oof = np.full(n_train, np.nan, dtype=float)
    out_folds = []
    for tr, va in folds:
        pipe = make_pipeline_model(clone(estimator), pca_dim, len(tr), n_feat)
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
        pred_oof[va] = va_pred
        out_folds.append(out_pred)

    pipe = make_pipeline_model(clone(estimator), pca_dim, n_train, n_feat)
    if target_mode == "residual":
        y_fit_full = y - dopant_prior(train_meta, train_meta)
    else:
        y_fit_full = y
    pipe.fit(X, y_fit_full)
    out_full = pipe.predict(Xout)
    if target_mode == "residual":
        out_full = prior_out + out_full
    out_cv = np.nanmean(np.vstack(out_folds), axis=0)
    out_pred = 0.70 * out_full + 0.30 * out_cv

    met = metrics(y, pred_oof)
    row: Dict[str, object] = {
        "candidate": f"{target_mode}__{feature_mode}__pca{pca_dim}__{model_name}",
        "target_mode": target_mode,
        "feature_mode": feature_mode,
        "pca_dim": int(pca_dim),
        "model": model_name,
        "pred_mean_addhout": float(np.nanmean(out_pred)),
        "pred_std_addhout": float(np.nanstd(out_pred)),
        "pred_min_addhout": float(np.nanmin(out_pred)),
        "pred_max_addhout": float(np.nanmax(out_pred)),
    }
    row.update(met)
    return row, pred_oof, out_pred


def load_bundle(bundle_dir: Path, npz_arg: str):
    npz_path = Path(npz_arg) if npz_arg else bundle_dir / "pretrained_delta_feature_bundle.npz"
    data = np.load(npz_path, allow_pickle=True)
    train_meta = pd.DataFrame({
        "id": data["train_ids"].astype(str),
        "family_base": data["train_groups"].astype(str),
        "material": data["train_material"].astype(str),
        "dopant": data["train_dopant"].astype(str),
        "target": data["y_train"].astype(float),
        "has_embedding": data["train_has_embedding"].astype(bool),
    })
    addhout_meta = pd.DataFrame({
        "id": data["addhout_ids"].astype(str),
        "material": data["addhout_material"].astype(str),
        "dopant": data["addhout_dopant"].astype(str),
        "has_embedding": data["addhout_has_embedding"].astype(bool),
    })
    return npz_path, data, train_meta, addhout_meta


def merge_existing_predictions(out: pd.DataFrame, existing_csv: Path, existing_col: str) -> pd.DataFrame:
    out = out.copy()
    out["pred_existing_anchor"] = np.nan
    if existing_csv.exists():
        existing = pd.read_csv(existing_csv)
        if "id" in existing.columns and existing_col in existing.columns:
            small = existing[["id", existing_col]].drop_duplicates("id").rename(columns={existing_col: "pred_existing_anchor"})
            out = out.drop(columns=["pred_existing_anchor"], errors="ignore").merge(small, on="id", how="left")
    return out


def posthoc_audit(pred: pd.DataFrame, labels_path: Path, target_col: str, out_dir: Path, oracle: bool) -> pd.DataFrame:
    if not labels_path.exists():
        return pd.DataFrame()
    labels = pd.read_csv(labels_path)
    if "id" not in labels.columns or target_col not in labels.columns:
        return pd.DataFrame()
    keep = [c for c in ["id", target_col, "material", "dopant"] if c in labels.columns]
    detail = pred.merge(labels[keep].drop_duplicates("id"), on="id", how="left", suffixes=("", "_audit"))
    if oracle and "pred_existing_anchor" in detail.columns and "pred_delta_head_ensemble" in detail.columns:
        best = (float("inf"), 0.0)
        anchor = pd.to_numeric(detail["pred_existing_anchor"], errors="coerce")
        delta = pd.to_numeric(detail["pred_delta_head_ensemble"], errors="coerce")
        for w in np.linspace(0, 1, 21):
            p = anchor.copy()
            m = delta.notna()
            p.loc[m] = (1 - w) * anchor.loc[m] + w * delta.loc[m]
            val = metrics(detail[target_col], p)["mae"]
            if np.isfinite(val) and val < best[0]:
                best = (float(val), float(w))
        detail["pred_oracle_delta_blend_diagnostic"] = anchor
        m = delta.notna()
        detail.loc[m, "pred_oracle_delta_blend_diagnostic"] = (
            (1 - best[1]) * anchor.loc[m] + best[1] * delta.loc[m]
        )
        with (out_dir / "oracle_delta_blend_diagnostic.json").open("w", encoding="utf-8") as f:
            json.dump({"mae": best[0], "delta_weight": best[1]}, f, indent=2)

    pred_cols = [
        c
        for c in [
            "pred_pretrained_delta_final",
            "pred_delta_head_ensemble",
            "pred_delta_head_recentered",
            "pred_existing_anchor",
            "pred_dopant_prior_bundle",
            "pred_oracle_delta_blend_diagnostic",
        ]
        if c in detail.columns
    ]
    audit = metric_rows(detail, pred_cols, target_col)
    audit.to_csv(out_dir / "pretrained_delta_head_posthoc_audit.csv", index=False)
    if "pred_pretrained_delta_final" in detail.columns:
        detail["err_pretrained_delta_final"] = (
            pd.to_numeric(detail["pred_pretrained_delta_final"], errors="coerce")
            - pd.to_numeric(detail[target_col], errors="coerce")
        )
        detail["abs_err_pretrained_delta_final"] = detail["err_pretrained_delta_final"].abs()
        detail = detail.sort_values("abs_err_pretrained_delta_final", ascending=False)
    detail.to_csv(out_dir / "pretrained_delta_head_posthoc_audit_detail.csv", index=False)
    try:
        audit.to_excel(out_dir / "pretrained_delta_head_posthoc_audit.xlsx", index=False)
    except Exception:
        pass
    return audit


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = Path(args.bundle_dir)
    npz_path, data, train_meta_all, addhout_meta_all = load_bundle(bundle_dir, args.bundle_npz)

    train_ok = train_meta_all["has_embedding"].to_numpy(bool)
    out_ok = addhout_meta_all["has_embedding"].to_numpy(bool)
    train_meta = train_meta_all.loc[train_ok].reset_index(drop=True).copy()
    addhout_meta_emb = addhout_meta_all.loc[out_ok].reset_index(drop=True).copy()
    Xg = data["X_graph_train"][train_ok].astype(np.float32)
    Xgo = data["X_graph_addhout"][out_ok].astype(np.float32)
    Xt = data["X_tab_train"][train_ok].astype(np.float32)
    Xto = data["X_tab_addhout"][out_ok].astype(np.float32)

    folds = make_group_folds(train_meta["family_base"].to_numpy(), args.n_splits)
    prior_oof, prior_out = fold_dopant_oof(train_meta, folds, addhout_meta_emb)
    prior_met = metrics(train_meta["target"], prior_oof)
    prior_mae = float(prior_met["mae"])

    print(f"[INFO] bundle={npz_path}")
    print(f"[INFO] train with embeddings={len(train_meta)} / {len(train_meta_all)}")
    print(f"[INFO] addH-out with embeddings={len(addhout_meta_emb)} / {len(addhout_meta_all)}")
    print(f"[INFO] graph_dim={Xg.shape[1]} tab_dim={Xt.shape[1]} folds={len(folds)}")
    print(f"[BASE] dopant prior OOF MAE={prior_mae:.4f}")

    feature_modes = parse_list(args.feature_modes)
    pca_dims = parse_int_list(args.pca_dims)
    target_modes = parse_list(args.target_modes)
    model_specs = []
    for family in parse_list(args.model_names):
        model_specs.extend(model_factory(family, args.seed))

    rows: List[Dict[str, object]] = []
    oof_pred_cols: Dict[str, np.ndarray] = {}
    out_pred_cols: Dict[str, np.ndarray] = {}
    source_mean = float(np.mean(train_meta["target"]))
    for target_mode in target_modes:
        for feature_mode in feature_modes:
            try:
                n_feat_base = select_graph_mode(Xg, feature_mode).shape[1] + Xt.shape[1]
            except Exception as exc:
                print(f"[WARN] skip feature mode {feature_mode}: {exc}")
                continue
            for pca_dim in pca_dims:
                if pca_dim > 0 and pca_dim >= min(len(train_meta), n_feat_base):
                    continue
                for model_name, estimator in model_specs:
                    try:
                        row, oof, out_pred = train_candidate(
                            model_name=model_name,
                            estimator=estimator,
                            feature_mode=feature_mode,
                            pca_dim=pca_dim,
                            target_mode=target_mode,
                            Xg=Xg,
                            Xg_out=Xgo,
                            Xt=Xt,
                            Xt_out=Xto,
                            train_meta=train_meta,
                            addhout_meta=addhout_meta_emb,
                            folds=folds,
                            prior_oof=prior_oof,
                            prior_out=prior_out,
                        )
                    except Exception as exc:
                        print(f"[WARN] candidate failed {target_mode}/{feature_mode}/pca{pca_dim}/{model_name}: {exc}")
                        continue
                    row["oof_improvement_vs_dopant"] = prior_mae - float(row["mae"])
                    row["abs_mean_shift_vs_source"] = abs(float(row["pred_mean_addhout"]) - source_mean)
                    rows.append(row)
                    key = str(row["candidate"])
                    oof_pred_cols[key] = oof
                    out_pred_cols[key] = out_pred
                    print(
                        f"[CAND] {key:42s} OOF MAE={row['mae']:.4f} "
                        f"improve={row['oof_improvement_vs_dopant']:.4f} "
                        f"shift={row['abs_mean_shift_vs_source']:.3f}"
                    )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise SystemExit("[ERROR] no candidate model succeeded.")
    summary = summary.sort_values(["mae", "rmse"], ascending=[True, True]).reset_index(drop=True)
    summary.to_csv(out_dir / "pretrained_delta_head_oof_metrics.csv", index=False)

    selected = summary[
        (summary["oof_improvement_vs_dopant"] >= args.min_oof_improvement)
        & (summary["abs_mean_shift_vs_source"] <= args.max_pred_mean_shift)
    ].head(args.top_k).copy()
    if selected.empty:
        print("[WARN] no delta-head candidate passed OOF/sanity gates; final will keep anchor predictions.")
    weights = np.array([], dtype=float)
    if not selected.empty:
        raw = selected["oof_improvement_vs_dopant"].to_numpy(float) / np.maximum(selected["mae"].to_numpy(float), 1e-6) ** 2
        weights = raw / raw.sum()
        selected["ensemble_weight"] = weights
    else:
        selected["ensemble_weight"] = []
    selected.to_csv(out_dir / "pretrained_delta_head_selected_models.csv", index=False)

    oof_table = train_meta[["id", "family_base", "material", "dopant", "target"]].copy()
    oof_table["oof_dopant_prior"] = prior_oof
    for key in selected.get("candidate", pd.Series(dtype=str)).astype(str).tolist():
        oof_table[f"oof__{key}"] = oof_pred_cols[key]
    oof_table.to_csv(out_dir / "pretrained_delta_head_oof_predictions.csv", index=False)

    addhout_pred = addhout_meta_all[["id", "material", "dopant", "has_embedding"]].copy()
    addhout_pred["pred_dopant_prior_bundle"] = np.nan
    addhout_pred.loc[out_ok, "pred_dopant_prior_bundle"] = prior_out
    addhout_pred = merge_existing_predictions(addhout_pred, Path(args.existing_pred_csv), args.existing_pred_col)
    if addhout_pred["pred_existing_anchor"].notna().sum() == 0:
        addhout_pred["pred_existing_anchor"] = addhout_pred["pred_dopant_prior_bundle"]

    delta_ensemble_emb = np.full(len(addhout_meta_emb), np.nan, dtype=float)
    if not selected.empty:
        V = np.vstack([out_pred_cols[str(k)] for k in selected["candidate"].astype(str).tolist()])
        delta_ensemble_emb = np.average(V, axis=0, weights=weights)
    addhout_pred["pred_delta_head_ensemble"] = np.nan
    addhout_pred.loc[out_ok, "pred_delta_head_ensemble"] = delta_ensemble_emb

    if args.recenter_delta_to_anchor:
        anchor_emb = pd.to_numeric(addhout_pred.loc[out_ok, "pred_existing_anchor"], errors="coerce")
        delta_ser = pd.to_numeric(addhout_pred.loc[out_ok, "pred_delta_head_ensemble"], errors="coerce")
        if delta_ser.notna().any() and anchor_emb.notna().any():
            shift = float(anchor_emb.median()) - float(delta_ser.median())
            addhout_pred["pred_delta_head_recentered"] = addhout_pred["pred_delta_head_ensemble"]
            addhout_pred.loc[out_ok, "pred_delta_head_recentered"] = delta_ser + shift
        else:
            addhout_pred["pred_delta_head_recentered"] = addhout_pred["pred_delta_head_ensemble"]
    else:
        addhout_pred["pred_delta_head_recentered"] = addhout_pred["pred_delta_head_ensemble"]

    if args.force_delta_weight >= 0:
        delta_weight = min(max(float(args.force_delta_weight), 0.0), 1.0)
    elif not selected.empty:
        best_improve = float(selected["oof_improvement_vs_dopant"].max())
        delta_weight = min(args.max_delta_blend_weight, max(0.0, best_improve / max(prior_mae, 1e-6)))
    else:
        delta_weight = 0.0
    anchor = pd.to_numeric(addhout_pred["pred_existing_anchor"], errors="coerce")
    delta = pd.to_numeric(addhout_pred["pred_delta_head_recentered"], errors="coerce")
    final = anchor.copy()
    m = delta.notna() & anchor.notna() & (delta_weight > 0)
    final.loc[m] = (1.0 - delta_weight) * anchor.loc[m] + delta_weight * delta.loc[m]
    if args.clip_final_to_source_range:
        final = final.clip(float(train_meta["target"].min()), float(train_meta["target"].max()))
    addhout_pred["pred_pretrained_delta_final"] = final
    addhout_pred["pretrained_delta_blend_weight"] = delta_weight
    addhout_pred["pretrained_delta_rank"] = pd.Series(final).rank(method="average", ascending=True).to_numpy()
    addhout_pred = addhout_pred.sort_values(["pretrained_delta_rank", "id"], na_position="last").reset_index(drop=True)
    addhout_pred.to_csv(out_dir / "pretrained_delta_head_addhout_predictions.csv", index=False)
    try:
        addhout_pred.to_excel(out_dir / "pretrained_delta_head_addhout_predictions.xlsx", index=False)
    except Exception:
        pass

    audit_labels: Optional[Path] = None
    if args.audit_labels_csv == "auto":
        # Prefer bundle-adjacent audit labels; otherwise skip quietly.
        for candidate in [bundle_dir / "addhout_audit_labels.csv", bundle_dir.parent / "outputs_addh_llm_element_priors" / "addhout_audit_labels.csv"]:
            if candidate.exists():
                audit_labels = candidate
                break
    elif args.audit_labels_csv:
        audit_labels = Path(args.audit_labels_csv)
    audit = pd.DataFrame()
    if audit_labels is not None and audit_labels.exists():
        audit = posthoc_audit(addhout_pred, audit_labels, args.audit_target_col, out_dir, args.oracle_diagnostic_tune)
        if len(audit):
            print("[POSTHOC AUDIT ONLY]")
            print(audit.to_string(index=False))

    manifest = {
        "bundle_npz": str(npz_path),
        "existing_pred_csv": str(args.existing_pred_csv),
        "existing_pred_col": args.existing_pred_col,
        "labels_used_for_training_or_selection": False,
        "n_train_with_embeddings": int(len(train_meta)),
        "n_addhout_with_embeddings": int(out_ok.sum()),
        "dopant_prior_oof": prior_met,
        "min_oof_improvement": args.min_oof_improvement,
        "max_pred_mean_shift": args.max_pred_mean_shift,
        "selected_models": selected.to_dict(orient="records"),
        "delta_blend_weight": float(delta_weight),
        "outputs": {
            "predictions_csv": str(out_dir / "pretrained_delta_head_addhout_predictions.csv"),
            "oof_metrics_csv": str(out_dir / "pretrained_delta_head_oof_metrics.csv"),
            "audit_csv": str(out_dir / "pretrained_delta_head_posthoc_audit.csv"),
        },
    }
    with (out_dir / "pretrained_delta_head_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] wrote {out_dir / 'pretrained_delta_head_addhout_predictions.csv'}")
    print(f"[INFO] selected_models={len(selected)} delta_blend_weight={delta_weight:.4f}")


if __name__ == "__main__":
    main()
