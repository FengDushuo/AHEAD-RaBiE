#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Target-domain-aware training for AddH/AddH-2 -> AddH-out.

Strict-blind default:
  - addH-out covariates are used for domain analysis and training weights;
  - addH-out labels are used only for post-hoc audit if available;
  - no addH-out label is used for model fitting, model selection, or blending.

Main idea:
  1) analyze how addH-out differs from addH/addH-2 in feature space;
  2) estimate target-likeness weights for training rows with a domain classifier
     and target-nearest-neighbor distances;
  3) train absolute and residual models with grouped OOF validation;
  4) select models by source OOF + target-weighted OOF + prediction-distribution
     guards;
  5) conservatively blend the new target-aware ensemble with the current best
     strict prediction.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


TARGET_OR_LEAKY_COLS = {
    "target",
    "energy_bare",
    "energy_addH",
    "energy_total_excel",
    "energy_slab_excel",
    "h_ads_excel",
    "target_computed",
    "fewshot_residual",
}

TEXT_OR_ID_COLS = {
    "split_role",
    "data_source",
    "id",
    "family_base",
    "material",
    "miller",
    "miller_text",
    "dopant",
    "status_bare",
    "status_addH",
    "contcar_path",
    "bare_contcar_path",
    "poscar_formula",
    "non_h_elements",
    "llm_rationale_short",
    "llm_sources",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Target-domain-aware AddH-out training.")
    ap.add_argument("--train-features", default="outputs_addh_llm_element_priors/knowledge_features_train.csv")
    ap.add_argument("--addhout-features", default="outputs_addh_llm_element_priors/knowledge_features_addhout.csv")
    ap.add_argument("--graph-bundle", default="outputs_addh_pretrained_delta_features/pretrained_delta_feature_bundle.npz")
    ap.add_argument("--graph-train-meta", default="outputs_addh_pretrained_delta_features/pretrained_delta_train_meta.csv")
    ap.add_argument("--graph-addhout-meta", default="outputs_addh_pretrained_delta_features/pretrained_delta_addhout_meta.csv")
    ap.add_argument("--anchor-csv", default="outputs_addh_superblend_precision/superblend_precision_addhout_predictions.csv")
    ap.add_argument("--fallback-anchor-csv", default="outputs_addh_target_calibrated_fast/target_calibrated_addhout_predictions.csv")
    ap.add_argument("--out-dir", default="outputs_addh_target_domain_aware_supertrainer")
    ap.add_argument("--audit-labels-csv", default="auto")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--target-col", default="target")
    ap.add_argument("--group-col", default="family_base")
    ap.add_argument("--material-col", default="material")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-abs-max", type=float, default=10.0)
    ap.add_argument("--include-source-stat-features", action="store_true")
    ap.add_argument("--include-pred-features", action="store_true")
    ap.add_argument("--graph-pca-dims", default="0,32,64,128")
    ap.add_argument("--feature-sets", default="tabular,tabular_graph64,graph64,tabular_graph128")
    ap.add_argument("--models", default="ridge,huber,elastic,extratrees,rf,hgb,gbr")
    ap.add_argument("--target-modes", default="absolute,residual_dopant,residual_llm")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--domain-weight-min", type=float, default=0.25)
    ap.add_argument("--domain-weight-max", type=float, default=8.0)
    ap.add_argument("--target-like-top-frac", type=float, default=0.30)
    ap.add_argument("--max-anchor-blend-weight", type=float, default=0.25)
    ap.add_argument("--max-deviation-from-anchor", type=float, default=1.60)
    ap.add_argument("--anchor-mae-col", default="auto")
    ap.add_argument("--anchor-trend-col", default="pred_superblend_trend")
    ap.add_argument("--final-mode", choices=["mae_guarded", "balanced", "trend"], default="mae_guarded")
    ap.add_argument("--write-xlsx", action="store_true")
    return ap.parse_args()


def parse_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def parse_ints(raw: str) -> List[int]:
    vals: List[int] = []
    for x in parse_list(raw):
        vals.append(int(float(x)))
    return sorted(set(vals))


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def parse_addhout_wide_excel(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    starts: List[Tuple[str, int]] = []
    for r in range(min(3, len(raw))):
        for c in range(raw.shape[1]):
            tok = str(raw.iat[r, c]).strip()
            if tok in {"CeO2", "ZnO"}:
                starts.append((tok, c))
    starts = sorted(set(starts), key=lambda x: x[1])
    if not starts:
        raise ValueError(f"Could not parse wide AddH-out Excel file: {path}")
    rows: List[Dict[str, object]] = []
    for _, row in raw.iterrows():
        idx = pd.to_numeric(pd.Series([row.iloc[0] if len(row) else np.nan]), errors="coerce").iloc[0]
        dop = str(row.iloc[1]).strip() if len(row) > 1 else ""
        if pd.isna(idx) or not dop or dop.lower() == "nan":
            continue
        i = int(idx)
        for mat, start in starts:
            if start + 2 >= raw.shape[1]:
                continue
            etot = pd.to_numeric(pd.Series([row.iloc[start]]), errors="coerce").iloc[0]
            eslab = pd.to_numeric(pd.Series([row.iloc[start + 1]]), errors="coerce").iloc[0]
            h = pd.to_numeric(pd.Series([row.iloc[start + 2]]), errors="coerce").iloc[0]
            if pd.isna(h):
                continue
            rows.append(
                {
                    "id": f"{mat}-{i}-{dop}",
                    "material": mat,
                    "dopant": dop,
                    "idx": i,
                    "h_ads_excel": float(h),
                    "target_computed": float(h),
                    "energy_total_excel": float(etot) if pd.notna(etot) else np.nan,
                    "energy_slab_excel": float(eslab) if pd.notna(eslab) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def find_audit_labels(raw: str) -> Optional[Path]:
    if raw and raw != "auto":
        p = Path(raw)
        return p if p.exists() else None
    candidates = [
        Path("outputs_addh_llm_element_priors/addhout_audit_labels.csv"),
        Path("addH-out/addhout_audit_labels.csv"),
        Path("addH-out/氢吸附能.xlsx"),
        Path("addH-out/hydrogen_adsorption_energy.xlsx"),
        Path("addH-out/energy.xlsx"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_audit_labels(raw: str, id_col: str) -> Tuple[pd.DataFrame, Optional[Path]]:
    p = find_audit_labels(raw)
    if p is None:
        return pd.DataFrame(columns=[id_col, "h_ads_excel"]), None
    if p.suffix.lower() in {".xlsx", ".xls"}:
        try:
            lab = read_table(p)
            if id_col not in lab.columns or "h_ads_excel" not in lab.columns:
                lab = parse_addhout_wide_excel(p)
        except Exception:
            lab = parse_addhout_wide_excel(p)
    else:
        lab = read_table(p)
    lab = lab.copy()
    if id_col not in lab.columns:
        if {"material", "idx", "dopant"}.issubset(lab.columns):
            lab[id_col] = (
                lab["material"].astype(str)
                + "-"
                + pd.to_numeric(lab["idx"], errors="coerce").fillna(-1).astype(int).astype(str)
                + "-"
                + lab["dopant"].astype(str)
            )
    if "h_ads_excel" not in lab.columns:
        for c in ["target", "target_computed", "h_ads", "H_ads"]:
            if c in lab.columns:
                lab["h_ads_excel"] = pd.to_numeric(lab[c], errors="coerce")
                break
    if id_col not in lab.columns or "h_ads_excel" not in lab.columns:
        return pd.DataFrame(columns=[id_col, "h_ads_excel"]), p
    keep = [id_col, "h_ads_excel"]
    for c in ["material", "dopant", "idx", "target_computed", "energy_total_excel", "energy_slab_excel"]:
        if c in lab.columns and c not in keep:
            keep.append(c)
    lab["h_ads_excel"] = pd.to_numeric(lab["h_ads_excel"], errors="coerce")
    return lab[keep].dropna(subset=[id_col, "h_ads_excel"]).drop_duplicates(id_col), p


def numeric_feature_cols(train: pd.DataFrame, target: pd.DataFrame, include_source: bool, include_pred: bool) -> List[str]:
    cols: List[str] = []
    for c in train.columns:
        if c not in target.columns:
            continue
        if c in TARGET_OR_LEAKY_COLS or c in TEXT_OR_ID_COLS:
            continue
        sc = str(c)
        if sc.startswith("source_") and not include_source:
            continue
        if sc.startswith("pred_") and not include_pred:
            continue
        s1 = pd.to_numeric(train[c], errors="coerce")
        s2 = pd.to_numeric(target[c], errors="coerce")
        if s1.notna().sum() >= 5 and s2.notna().sum() >= 2:
            cols.append(c)
    return sorted(cols)


def to_matrix(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    return df[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)


def align_by_id(meta: pd.DataFrame, arr: np.ndarray, ids: Sequence[str], id_col: str) -> np.ndarray:
    out = np.full((len(ids), arr.shape[1]), np.nan, dtype=np.float32)
    pos = {str(x): i for i, x in enumerate(meta[id_col].astype(str))}
    for i, sid in enumerate([str(x) for x in ids]):
        j = pos.get(sid)
        if j is not None and j < len(arr):
            out[i] = arr[j]
    return out


def load_graph_pcs(
    bundle_path: Path,
    train_meta_path: Path,
    out_meta_path: Path,
    train_ids: Sequence[str],
    out_ids: Sequence[str],
    id_col: str,
    dims: Sequence[int],
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    if not bundle_path.exists() or not train_meta_path.exists() or not out_meta_path.exists():
        return {}
    data = np.load(bundle_path)
    if "X_graph_train" not in data or "X_graph_addhout" not in data:
        return {}
    train_meta = pd.read_csv(train_meta_path)
    out_meta = pd.read_csv(out_meta_path)
    Xtr_raw = align_by_id(train_meta, data["X_graph_train"], train_ids, id_col)
    Xout_raw = align_by_id(out_meta, data["X_graph_addhout"], out_ids, id_col)
    both = np.vstack([Xtr_raw, Xout_raw])
    both = SimpleImputer(strategy="median").fit_transform(both)
    both = StandardScaler().fit_transform(both)
    Xtr_imp = both[: len(Xtr_raw)]
    Xout_imp = both[len(Xtr_raw) :]
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {0: (Xtr_imp, Xout_imp)}
    max_dim = min(Xtr_imp.shape[1], len(both) - 1)
    for d in dims:
        if d <= 0 or d > max_dim:
            continue
        pca = PCA(n_components=d, random_state=42)
        pcs = pca.fit_transform(both)
        out[d] = (pcs[: len(Xtr_raw)].astype(np.float32), pcs[len(Xtr_raw) :].astype(np.float32))
    return out


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


def metrics(y_true: Sequence[float], y_pred: Sequence[float], sample_weight: Optional[Sequence[float]] = None) -> Dict[str, float]:
    y = pd.to_numeric(pd.Series(y_true), errors="coerce")
    p = pd.to_numeric(pd.Series(y_pred), errors="coerce")
    m = y.notna() & p.notna()
    if int(m.sum()) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "pearson": np.nan, "spearman": np.nan}
    yy = y[m].to_numpy(float)
    pp = p[m].to_numpy(float)
    e = pp - yy
    if sample_weight is None:
        w = np.ones_like(yy)
    else:
        ww = np.asarray(sample_weight, dtype=float)
        w = ww[m.to_numpy()] if len(ww) == len(m) else np.ones_like(yy)
        w = np.where(np.isfinite(w), w, 1.0)
        if np.sum(w) <= 0:
            w = np.ones_like(yy)
    return {
        "n": int(len(yy)),
        "mae": float(np.average(np.abs(e), weights=w)),
        "rmse": float(np.sqrt(np.average(e * e, weights=w))),
        "bias": float(np.average(e, weights=w)),
        "pearson": finite_corr(yy, pp, False),
        "spearman": finite_corr(yy, pp, True),
    }


def metric_rows(df: pd.DataFrame, pred_cols: Iterable[str], target_col: str, group_col: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    groups: List[Tuple[Optional[str], pd.DataFrame]] = [(None, df)]
    if group_col in df.columns:
        for g, sub in df.groupby(group_col, dropna=False):
            groups.append((str(g), sub))
    for c in pred_cols:
        if c not in df.columns:
            continue
        for g, sub in groups:
            row: Dict[str, object] = {"pred_col": c, "target_col": target_col, group_col: g}
            row.update(metrics(sub[target_col], sub[c]))
            rows.append(row)
    return pd.DataFrame(rows)


def make_folds(groups: Sequence[str], n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups_arr = np.asarray([str(g) for g in groups])
    uniq = pd.unique(groups_arr)
    if len(uniq) >= 2:
        n = max(2, min(int(n_splits), len(uniq)))
        return [(np.asarray(tr), np.asarray(va)) for tr, va in GroupKFold(n_splits=n).split(np.zeros(len(groups_arr)), groups=groups_arr)]
    n = max(2, min(int(n_splits), len(groups_arr)))
    return [(np.asarray(tr), np.asarray(va)) for tr, va in KFold(n_splits=n, shuffle=True, random_state=seed).split(np.zeros(len(groups_arr)))]


def crossfit_dopant_prior(train: pd.DataFrame, out: pd.DataFrame, folds: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    y = train["target"].to_numpy(float)
    oof = np.full(len(train), np.nan, dtype=float)
    for tr, va in folds:
        tr_df = train.iloc[tr]
        global_mean = float(tr_df["target"].mean())
        by_dop = tr_df.groupby("dopant")["target"].mean() if "dopant" in tr_df.columns else pd.Series(dtype=float)
        by_mat = tr_df.groupby("material")["target"].mean() if "material" in tr_df.columns else pd.Series(dtype=float)
        vals = []
        for _, r in train.iloc[va].iterrows():
            v = np.nan
            if "dopant" in r and str(r["dopant"]) in by_dop.index:
                v = float(by_dop.loc[str(r["dopant"])])
            elif "material" in r and str(r["material"]) in by_mat.index:
                v = float(by_mat.loc[str(r["material"])])
            else:
                v = global_mean
            vals.append(v)
        oof[va] = vals
    full_global = float(train["target"].mean())
    full_by_dop = train.groupby("dopant")["target"].mean() if "dopant" in train.columns else pd.Series(dtype=float)
    full_by_mat = train.groupby("material")["target"].mean() if "material" in train.columns else pd.Series(dtype=float)
    out_vals = []
    for _, r in out.iterrows():
        v = np.nan
        if "source_dopant_target_mean_full" in r and pd.notna(r["source_dopant_target_mean_full"]):
            v = float(r["source_dopant_target_mean_full"])
        elif "dopant" in r and str(r["dopant"]) in full_by_dop.index:
            v = float(full_by_dop.loc[str(r["dopant"])])
        elif "material" in r and str(r["material"]) in full_by_mat.index:
            v = float(full_by_mat.loc[str(r["material"])])
        else:
            v = full_global
        out_vals.append(v)
    return oof, np.asarray(out_vals, dtype=float)


def llm_prior(train: pd.DataFrame, out: pd.DataFrame, fallback_train: np.ndarray, fallback_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tr = pd.to_numeric(train.get("llm_prior_h_ads_eV_guess", pd.Series(np.nan, index=train.index)), errors="coerce")
    ot = pd.to_numeric(out.get("llm_prior_h_ads_eV_guess", pd.Series(np.nan, index=out.index)), errors="coerce")
    trv = tr.to_numpy(float)
    outv = ot.to_numpy(float)
    return np.where(np.isfinite(trv), trv, fallback_train), np.where(np.isfinite(outv), outv, fallback_out)


def domain_weights(
    X_train: np.ndarray,
    X_out: np.ndarray,
    seed: int,
    w_min: float,
    w_max: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    X = np.vstack([X_train, X_out])
    y = np.r_[np.zeros(len(X_train), dtype=int), np.ones(len(X_out), dtype=int)]
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imp)
    probs = np.full(len(X), np.nan, dtype=float)
    n_splits = max(2, min(5, np.bincount(y).min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in cv.split(X_scaled, y):
        clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=seed)
        clf.fit(X_scaled[tr], y[tr])
        probs[va] = clf.predict_proba(X_scaled[va])[:, 1]
    try:
        auc = float(roc_auc_score(y, probs))
    except Exception:
        auc = float("nan")

    p_train = np.clip(probs[: len(X_train)], 1e-4, 1 - 1e-4)
    odds = p_train / (1.0 - p_train)
    prop_w = odds * (len(X_train) / max(len(X_out), 1))
    prop_w = np.clip(prop_w, w_min, w_max)

    nn = NearestNeighbors(n_neighbors=min(5, len(X_out))).fit(X_scaled[len(X_train) :])
    dist, _ = nn.kneighbors(X_scaled[: len(X_train)])
    d = np.mean(dist, axis=1)
    med = float(np.median(d))
    iqr = float(np.quantile(d, 0.75) - np.quantile(d, 0.25))
    scale = iqr if iqr > 1e-8 else max(float(np.std(d)), 1e-6)
    knn_w = np.exp(-(d - med) / scale)
    knn_w = np.clip(knn_w, w_min, w_max)
    w = np.sqrt(prop_w * knn_w)
    w = np.clip(w, w_min, w_max)
    w = w / max(float(np.mean(w)), 1e-12)
    info = {
        "domain_auc_oof": auc,
        "weight_min": float(np.min(w)),
        "weight_max": float(np.max(w)),
        "weight_mean": float(np.mean(w)),
        "weight_q10": float(np.quantile(w, 0.10)),
        "weight_q50": float(np.quantile(w, 0.50)),
        "weight_q90": float(np.quantile(w, 0.90)),
    }
    return w.astype(float), info


def shifted_feature_report(train: pd.DataFrame, out: pd.DataFrame, cols: Sequence[str], top_n: int = 40) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for c in cols:
        tr = pd.to_numeric(train[c], errors="coerce").dropna()
        ot = pd.to_numeric(out[c], errors="coerce").dropna()
        if len(tr) < 5 or len(ot) < 2:
            continue
        sd = float(tr.std())
        if sd <= 1e-12:
            continue
        rows.append(
            {
                "feature": c,
                "train_mean": float(tr.mean()),
                "addhout_mean": float(ot.mean()),
                "train_std": sd,
                "standardized_mean_diff": float((ot.mean() - tr.mean()) / sd),
                "train_q05": float(tr.quantile(0.05)),
                "addhout_q05": float(ot.quantile(0.05)),
                "train_q95": float(tr.quantile(0.95)),
                "addhout_q95": float(ot.quantile(0.95)),
            }
        )
    df = pd.DataFrame(rows)
    if len(df):
        df["_abs"] = df["standardized_mean_diff"].abs()
        df = df.sort_values("_abs", ascending=False).drop(columns=["_abs"]).head(top_n)
    return df


def model_factory(name: str, seed: int):
    name = name.lower()
    if name == "ridge":
        return [(f"ridge_a{a:g}", Ridge(alpha=a)) for a in [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]]
    if name == "huber":
        return [(f"huber_a{a:g}", HuberRegressor(alpha=a, epsilon=1.25, max_iter=800)) for a in [0.0001, 0.001, 0.01]]
    if name == "elastic":
        return [(f"elastic_a{a:g}", ElasticNet(alpha=a, l1_ratio=0.15, max_iter=6000, random_state=seed)) for a in [0.003, 0.01, 0.03, 0.1]]
    if name == "extratrees":
        return [
            (
                "extratrees",
                ExtraTreesRegressor(
                    n_estimators=350,
                    min_samples_leaf=2,
                    max_features=0.75,
                    random_state=seed,
                    n_jobs=-1,
                ),
            )
        ]
    if name == "rf":
        return [
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=350,
                    min_samples_leaf=3,
                    max_features=0.75,
                    random_state=seed,
                    n_jobs=-1,
                ),
            )
        ]
    if name == "hgb":
        return [
            (
                "hgb",
                HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=0.035,
                    l2_regularization=0.10,
                    min_samples_leaf=8,
                    random_state=seed,
                ),
            )
        ]
    if name == "gbr":
        return [
            (
                "gbr",
                GradientBoostingRegressor(
                    n_estimators=260,
                    learning_rate=0.035,
                    max_depth=2,
                    min_samples_leaf=4,
                    random_state=seed,
                ),
            )
        ]
    return []


def make_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", model),
        ]
    )


def estimator_supports_weight(pipe: Pipeline) -> bool:
    try:
        return "sample_weight" in inspect.signature(pipe.named_steps["model"].fit).parameters
    except Exception:
        return False


def fit_with_optional_weight(pipe: Pipeline, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray]) -> Pipeline:
    if sample_weight is not None and estimator_supports_weight(pipe):
        pipe.fit(X, y, model__sample_weight=sample_weight)
    else:
        pipe.fit(X, y)
    return pipe


def build_feature_sets(
    X_tab_train: np.ndarray,
    X_tab_out: np.ndarray,
    graph_pcs: Dict[int, Tuple[np.ndarray, np.ndarray]],
    requested: Sequence[str],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    sets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {"tabular": (X_tab_train, X_tab_out)}
    if 64 in graph_pcs:
        gtr, gout = graph_pcs[64]
        sets["graph64"] = (gtr, gout)
        sets["tabular_graph64"] = (np.hstack([X_tab_train, gtr]), np.hstack([X_tab_out, gout]))
    if 128 in graph_pcs:
        gtr, gout = graph_pcs[128]
        sets["graph128"] = (gtr, gout)
        sets["tabular_graph128"] = (np.hstack([X_tab_train, gtr]), np.hstack([X_tab_out, gout]))
    if 32 in graph_pcs:
        gtr, gout = graph_pcs[32]
        sets["graph32"] = (gtr, gout)
        sets["tabular_graph32"] = (np.hstack([X_tab_train, gtr]), np.hstack([X_tab_out, gout]))
    return {k: v for k, v in sets.items() if k in set(requested)}


def candidate_train_predict(
    X_train: np.ndarray,
    X_out: np.ndarray,
    y_abs: np.ndarray,
    target: np.ndarray,
    out_base: np.ndarray,
    train_weights: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    model,
) -> Tuple[np.ndarray, np.ndarray]:
    oof = np.full(len(X_train), np.nan, dtype=float)
    out_folds: List[np.ndarray] = []
    for tr, va in folds:
        pipe = make_pipeline(clone(model))
        fit_with_optional_weight(pipe, X_train[tr], target[tr], train_weights[tr])
        oof[va] = pipe.predict(X_train[va])
        out_folds.append(pipe.predict(X_out))
    pipe_full = make_pipeline(clone(model))
    fit_with_optional_weight(pipe_full, X_train, target, train_weights)
    out_pred = pipe_full.predict(X_out)
    if out_folds:
        out_pred = 0.85 * out_pred + 0.15 * np.nanmean(np.vstack(out_folds), axis=0)
    return oof, out_pred


def safe_norm_weights(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    s = np.where(np.isfinite(s), s, np.nanmax(s[np.isfinite(s)]) if np.isfinite(s).any() else 1.0)
    inv = 1.0 / np.maximum(s - np.nanmin(s) + 0.05, 0.05)
    inv = np.clip(inv, 0.0, np.nanquantile(inv, 0.90) if len(inv) > 2 else np.max(inv))
    if np.sum(inv) <= 0:
        return np.ones_like(inv) / max(len(inv), 1)
    return inv / np.sum(inv)


def first_existing(df: pd.DataFrame, cols: Sequence[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def load_anchor(path: str, fallback: str, id_col: str) -> Tuple[pd.DataFrame, Optional[Path]]:
    for raw in [path, fallback]:
        p = Path(raw)
        if p.exists():
            df = pd.read_csv(p)
            if id_col in df.columns:
                return df, p
    return pd.DataFrame(columns=[id_col]), None


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_all = pd.read_csv(args.train_features)
    addhout_all = pd.read_csv(args.addhout_features)
    if args.id_col not in train_all.columns or args.id_col not in addhout_all.columns:
        raise SystemExit("[ERROR] id column missing from feature tables.")
    if args.target_col not in train_all.columns:
        raise SystemExit(f"[ERROR] target column {args.target_col!r} missing from train features.")

    train_all = train_all.copy()
    train_all["target"] = pd.to_numeric(train_all[args.target_col], errors="coerce")
    usable = train_all["target"].notna() & (train_all["target"].abs() <= args.target_abs_max)
    train = train_all.loc[usable].reset_index(drop=True).copy()
    addhout = addhout_all.reset_index(drop=True).copy()
    y = train["target"].to_numpy(float)

    folds = make_folds(train[args.group_col] if args.group_col in train.columns else train[args.id_col], args.n_splits, args.seed)
    dop_oof, dop_out = crossfit_dopant_prior(train, addhout, folds)
    llm_oof, llm_out = llm_prior(train, addhout, dop_oof, dop_out)

    num_cols = numeric_feature_cols(train, addhout, args.include_source_stat_features, args.include_pred_features)
    X_tab_train = to_matrix(train, num_cols)
    X_tab_out = to_matrix(addhout, num_cols)
    graph_pcs = load_graph_pcs(
        Path(args.graph_bundle),
        Path(args.graph_train_meta),
        Path(args.graph_addhout_meta),
        train[args.id_col].astype(str).tolist(),
        addhout[args.id_col].astype(str).tolist(),
        args.id_col,
        parse_ints(args.graph_pca_dims),
    )
    feature_sets = build_feature_sets(X_tab_train, X_tab_out, graph_pcs, parse_list(args.feature_sets))
    if not feature_sets:
        feature_sets = {"tabular": (X_tab_train, X_tab_out)}

    domain_X_train = feature_sets.get("tabular_graph64", feature_sets.get("tabular", next(iter(feature_sets.values()))))[0]
    domain_X_out = feature_sets.get("tabular_graph64", feature_sets.get("tabular", next(iter(feature_sets.values()))))[1]
    weights, domain_info = domain_weights(domain_X_train, domain_X_out, args.seed, args.domain_weight_min, args.domain_weight_max)
    train["target_domain_weight"] = weights
    train["dopant_prior_oof"] = dop_oof
    train["llm_prior_oof"] = llm_oof
    addhout["dopant_prior_full"] = dop_out
    addhout["llm_prior_full"] = llm_out

    target_like_cut = np.quantile(weights, max(0.0, min(1.0, 1.0 - args.target_like_top_frac)))
    target_like_mask = weights >= target_like_cut

    candidates: List[Dict[str, object]] = []
    oof_cols: Dict[str, np.ndarray] = {}
    out_cols: Dict[str, np.ndarray] = {}
    source_q01, source_q99 = float(np.quantile(y, 0.01)), float(np.quantile(y, 0.99))
    source_mean, source_std = float(np.mean(y)), float(np.std(y))

    for fs_name, (Xtr, Xout) in feature_sets.items():
        for tm in parse_list(args.target_modes):
            tm = tm.lower()
            if tm == "absolute":
                target_train = y
                base_out = np.zeros(len(addhout), dtype=float)
                base_oof = np.zeros(len(train), dtype=float)
            elif tm == "residual_dopant":
                target_train = y - dop_oof
                base_oof = dop_oof
                base_out = dop_out
            elif tm == "residual_llm":
                target_train = y - llm_oof
                base_oof = llm_oof
                base_out = llm_out
            else:
                continue
            for model_group in parse_list(args.models):
                for model_name, model in model_factory(model_group, args.seed):
                    cand_name = f"{tm}__{fs_name}__{model_name}"
                    try:
                        raw_oof, raw_out = candidate_train_predict(Xtr, Xout, y, target_train, base_out, weights, folds, model)
                        pred_oof = raw_oof + base_oof
                        pred_out = raw_out + base_out
                        if not np.isfinite(pred_oof).any() or not np.isfinite(pred_out).any():
                            continue
                        m_all = metrics(y, pred_oof)
                        m_w = metrics(y, pred_oof, weights)
                        m_top = metrics(y[target_like_mask], pred_oof[target_like_mask])
                        pred_mean = float(np.nanmean(pred_out))
                        pred_std = float(np.nanstd(pred_out))
                        pred_min = float(np.nanmin(pred_out))
                        pred_max = float(np.nanmax(pred_out))
                        out_range_frac = float(np.mean((pred_out < source_q01 - 2.0) | (pred_out > source_q99 + 2.0)))
                        mean_shift = abs(pred_mean - source_mean)
                        std_ratio = pred_std / max(source_std, 1e-9)
                        dist_penalty = 0.10 * max(0.0, mean_shift - 2.0) + 0.25 * out_range_frac + 0.05 * max(0.0, std_ratio - 1.8)
                        composite = 0.40 * m_all["mae"] + 0.40 * m_w["mae"] + 0.20 * m_top["mae"] + dist_penalty
                        candidates.append(
                            {
                                "candidate": cand_name,
                                "target_mode": tm,
                                "feature_set": fs_name,
                                "model": model_name,
                                "n": m_all["n"],
                                "mae": m_all["mae"],
                                "weighted_mae": m_w["mae"],
                                "target_like_mae": m_top["mae"],
                                "rmse": m_all["rmse"],
                                "bias": m_all["bias"],
                                "pearson": m_all["pearson"],
                                "spearman": m_all["spearman"],
                                "pred_mean_addhout": pred_mean,
                                "pred_std_addhout": pred_std,
                                "pred_min_addhout": pred_min,
                                "pred_max_addhout": pred_max,
                                "out_range_frac": out_range_frac,
                                "mean_shift_vs_source": mean_shift,
                                "std_ratio_vs_source": std_ratio,
                                "distribution_penalty": dist_penalty,
                                "selection_score": composite,
                            }
                        )
                        oof_cols[cand_name] = pred_oof
                        out_cols[cand_name] = pred_out
                        print(f"[CAND] {cand_name:48s} score={composite:.4f} mae={m_all['mae']:.4f} wmae={m_w['mae']:.4f}")
                    except Exception as e:
                        print(f"[WARN] candidate failed: {cand_name}: {e}")
                        continue

    cand_df = pd.DataFrame(candidates)
    if len(cand_df) == 0:
        raise SystemExit("[ERROR] no candidate models succeeded.")
    cand_df = cand_df.sort_values(["selection_score", "weighted_mae", "mae"], kind="mergesort")
    selected = cand_df.head(max(1, args.top_k)).copy()
    ens_w = safe_norm_weights(selected["selection_score"].to_numpy(float))
    selected["ensemble_weight"] = ens_w
    selected_names = selected["candidate"].astype(str).tolist()
    oof_mat = np.vstack([oof_cols[n] for n in selected_names])
    out_mat = np.vstack([out_cols[n] for n in selected_names])
    pred_oof_ens = np.average(oof_mat, axis=0, weights=ens_w)
    pred_out_ens = np.average(out_mat, axis=0, weights=ens_w)

    anchor, anchor_path = load_anchor(args.anchor_csv, args.fallback_anchor_csv, args.id_col)
    result = addhout[[c for c in [args.id_col, args.material_col, "dopant", "idx"] if c in addhout.columns]].copy()
    result["pred_domain_aware_model"] = pred_out_ens
    if len(anchor):
        keep = [args.id_col] + [c for c in anchor.columns if c != args.id_col and (c.startswith("pred_") or c in {args.material_col, "dopant"})]
        keep = list(dict.fromkeys([c for c in keep if c in anchor.columns]))
        result = result.merge(anchor[keep].drop_duplicates(args.id_col), on=args.id_col, how="left", suffixes=("", "__anchor"))

    anchor_mae_col = args.anchor_mae_col
    if anchor_mae_col == "auto":
        anchor_mae_col = first_existing(
            result,
            ["pred_superblend_final", "pred_superblend_mae_guarded", "pred_fast_target_calibrated", "pred_existing_anchor"],
        ) or "pred_domain_aware_model"
    anchor_trend_col = args.anchor_trend_col if args.anchor_trend_col in result.columns else first_existing(
        result, ["pred_superblend_trend", "pred_rank_trend_calibrated", "pred_domain_aware_model"]
    )
    anchor_base = pd.to_numeric(result[anchor_mae_col], errors="coerce") if anchor_mae_col in result.columns else pd.Series(pred_out_ens, index=result.index)
    model_s = pd.Series(pred_out_ens, index=result.index)
    raw_delta = model_s - anchor_base
    guarded_delta = raw_delta.clip(lower=-args.max_deviation_from_anchor, upper=args.max_deviation_from_anchor)
    best_score = float(selected["selection_score"].iloc[0])
    median_score = float(cand_df["selection_score"].median())
    quality = max(0.0, min(1.0, (median_score - best_score) / max(median_score, 1e-9)))
    blend_weight = min(args.max_anchor_blend_weight, 0.08 + 0.35 * quality)
    result["pred_domain_aware_guarded"] = anchor_base + blend_weight * guarded_delta

    if anchor_trend_col and anchor_trend_col in result.columns:
        trend_s = pd.to_numeric(result[anchor_trend_col], errors="coerce")
        result["pred_domain_aware_balanced"] = 0.65 * result["pred_domain_aware_guarded"] + 0.35 * trend_s
        result["pred_domain_aware_trend"] = 0.45 * result["pred_domain_aware_guarded"] + 0.55 * trend_s
    else:
        result["pred_domain_aware_balanced"] = result["pred_domain_aware_guarded"]
        result["pred_domain_aware_trend"] = result["pred_domain_aware_guarded"]

    if args.final_mode == "trend":
        result["pred_domain_aware_final"] = result["pred_domain_aware_trend"]
    elif args.final_mode == "balanced":
        result["pred_domain_aware_final"] = result["pred_domain_aware_balanced"]
    else:
        result["pred_domain_aware_final"] = result["pred_domain_aware_guarded"]
    result["domain_aware_anchor_col"] = anchor_mae_col
    result["domain_aware_trend_col"] = anchor_trend_col or ""
    result["domain_aware_model_blend_weight"] = blend_weight

    oof = train[[c for c in [args.id_col, "family_base", args.material_col, "dopant"] if c in train.columns]].copy()
    oof["target"] = y
    oof["target_domain_weight"] = weights
    oof["pred_domain_aware_model_oof"] = pred_oof_ens
    oof["dopant_prior_oof"] = dop_oof
    oof["llm_prior_oof"] = llm_oof
    for n in selected_names:
        oof[f"oof__{n}"] = oof_cols[n]

    audit_labels, audit_path = load_audit_labels(args.audit_labels_csv, args.id_col)
    audit = pd.DataFrame()
    if len(audit_labels):
        detail = result.merge(audit_labels, on=args.id_col, how="left", suffixes=("", "__label"))
        if args.material_col not in detail.columns and f"{args.material_col}__label" in detail.columns:
            detail[args.material_col] = detail[f"{args.material_col}__label"]
        pred_cols = [
            c
            for c in [
                "pred_domain_aware_final",
                "pred_domain_aware_guarded",
                "pred_domain_aware_balanced",
                "pred_domain_aware_trend",
                "pred_domain_aware_model",
                anchor_mae_col,
                anchor_trend_col or "",
                "pred_rank_trend_calibrated",
                "pred_pretrained_delta_final",
                "pred_source_dopant_mean_prior",
            ]
            if c and c in detail.columns
        ]
        audit = metric_rows(detail, list(dict.fromkeys(pred_cols)), "h_ads_excel", args.material_col)
        detail.to_csv(out_dir / "domain_aware_posthoc_audit_detail.csv", index=False)
        audit.to_csv(out_dir / "domain_aware_posthoc_audit.csv", index=False)
        if args.write_xlsx:
            detail.to_excel(out_dir / "domain_aware_posthoc_audit_detail.xlsx", index=False)

    cand_df.to_csv(out_dir / "domain_aware_candidate_metrics.csv", index=False)
    selected.to_csv(out_dir / "domain_aware_selected_models.csv", index=False)
    result.to_csv(out_dir / "domain_aware_addhout_predictions.csv", index=False)
    oof.to_csv(out_dir / "domain_aware_oof_predictions.csv", index=False)
    shifted = shifted_feature_report(train, addhout, num_cols)
    shifted.to_csv(out_dir / "domain_shift_feature_report.csv", index=False)
    train[[c for c in [args.id_col, "family_base", args.material_col, "dopant", "target", "target_domain_weight"] if c in train.columns]].sort_values(
        "target_domain_weight", ascending=False
    ).head(80).to_csv(out_dir / "target_like_training_rows.csv", index=False)
    if args.write_xlsx:
        result.to_excel(out_dir / "domain_aware_addhout_predictions.xlsx", index=False)
        with pd.ExcelWriter(out_dir / "domain_aware_report.xlsx") as w:
            cand_df.to_excel(w, "candidates", index=False)
            selected.to_excel(w, "selected", index=False)
            shifted.to_excel(w, "domain_shift", index=False)
            if len(audit):
                audit.to_excel(w, "audit", index=False)

    manifest = {
        "script": Path(__file__).name,
        "strict_blind": True,
        "labels_used_for_training_or_selection": False,
        "audit_label_file": str(audit_path) if audit_path else None,
        "train_features": args.train_features,
        "addhout_features": args.addhout_features,
        "graph_bundle": args.graph_bundle if Path(args.graph_bundle).exists() else None,
        "anchor_file": str(anchor_path) if anchor_path else None,
        "n_train": int(len(train)),
        "n_addhout": int(len(addhout)),
        "n_numeric_features": int(len(num_cols)),
        "feature_sets": list(feature_sets.keys()),
        "domain_info": domain_info,
        "selected_models": selected.to_dict(orient="records"),
        "anchor_mae_col": anchor_mae_col,
        "anchor_trend_col": anchor_trend_col,
        "domain_aware_model_blend_weight": blend_weight,
        "final_mode": args.final_mode,
        "outputs": {
            "predictions": str(out_dir / "domain_aware_addhout_predictions.csv"),
            "candidate_metrics": str(out_dir / "domain_aware_candidate_metrics.csv"),
            "selected_models": str(out_dir / "domain_aware_selected_models.csv"),
            "oof_predictions": str(out_dir / "domain_aware_oof_predictions.csv"),
            "domain_shift_report": str(out_dir / "domain_shift_feature_report.csv"),
            "audit": str(out_dir / "domain_aware_posthoc_audit.csv"),
        },
    }
    (out_dir / "domain_aware_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] wrote {out_dir}")
    print("[DOMAIN]", json.dumps(domain_info, ensure_ascii=False))
    print("[SELECTED]")
    print(selected[["candidate", "selection_score", "mae", "weighted_mae", "target_like_mae", "ensemble_weight"]].to_string(index=False))
    if len(audit):
        print("[POSTHOC AUDIT ONLY]")
        print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
