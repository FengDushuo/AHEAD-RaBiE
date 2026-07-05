#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a strict-blind LLM/element-prior model and blend it with existing AddH-out
predictions.

Default behavior:
  - train only on addH/addH-2 labels;
  - evaluate source-domain OOF by family_base grouped CV;
  - predict AddH-out without using AddH-out labels;
  - use AddH-out labels only if --audit-labels-csv is provided.

The blend is deliberately conservative for target-domain shift:
  knowledge_model + dopant_stat_prior + distribution-sane base predictions.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ID_COLS = {
    "id", "split_role", "data_source", "family_base", "material", "idx", "miller", "miller_text",
    "dopant", "status_bare", "status_addH", "contcar_path", "bare_contcar_path", "poscar_formula",
    "non_h_elements", "llm_rationale_short", "llm_sources",
}
TARGET_DERIVED_COLS = {
    "target", "energy_bare", "energy_addH", "energy_total_excel", "energy_slab_excel",
    "h_ads_excel", "target_computed",
    "source_dopant_target_count_full", "source_dopant_target_mean_full",
    "source_dopant_target_median_full", "source_dopant_target_std_full",
    "source_dopant_target_min_full", "source_dopant_target_max_full",
}
PRED_COL_CANDIDATES = [
    "pred_strict_blind_final",
    "pred_strict_blind",
    "pred_strict_blind_weighted",
    "pred_strict_blind_median",
    "pred_strict_blind_trimmed_mean",
    "pred_median",
    "pred_mean",
    "pred_ensemble",
    "prediction",
    "y_pred",
    "pred",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", default="outputs_addh_llm_element_priors")
    ap.add_argument("--train-features", default=None)
    ap.add_argument("--addhout-features", default=None)
    ap.add_argument("--out-dir", default="outputs_addh_llm_element_knowledge_blend")
    ap.add_argument("--target-col", default="target")
    ap.add_argument("--group-col", default="family_base")
    ap.add_argument("--target-abs-max", type=float, default=10.0)
    ap.add_argument("--n-splits", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-knowledge-models", type=int, default=5)
    ap.add_argument("--scan-pred-root", action="append", default=["logs"])
    ap.add_argument("--base-pred-file", action="append", default=[])
    ap.add_argument("--min-base-coverage", type=float, default=0.90)
    ap.add_argument("--max-blind-mean-shift", type=float, default=2.0)
    ap.add_argument("--allow-biased-base-preds", action="store_true")
    ap.add_argument("--blend-knowledge", type=float, default=0.05)
    ap.add_argument("--blend-dopant", type=float, default=0.80)
    ap.add_argument("--blend-base-recenter", type=float, default=0.10)
    ap.add_argument("--blend-base-raw", type=float, default=0.05)
    ap.add_argument("--audit-labels-csv", default=None, help="Optional addhout_audit_labels.csv. Labels are post-hoc only.")
    return ap.parse_args()


def finite_corr(a: Sequence[float], b: Sequence[float], method: str = "pearson") -> float:
    s1 = pd.Series(a, dtype=float)
    s2 = pd.Series(b, dtype=float)
    m = s1.notna() & s2.notna()
    if m.sum() < 3:
        return np.nan
    x = s1[m].to_numpy(float)
    y = s2[m].to_numpy(float)
    if method == "spearman":
        x = pd.Series(x).rank(method="average").to_numpy(float)
        y = pd.Series(y).rank(method="average").to_numpy(float)
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if den <= 1e-12:
        return np.nan
    return float(np.sum(x * y) / den)


def metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if not m.any():
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "pearson": np.nan, "spearman": np.nan}
    e = p[m] - y[m]
    return {
        "n": int(m.sum()),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "bias": float(np.mean(e)),
        "pearson": finite_corr(y[m], p[m], "pearson"),
        "spearman": finite_corr(y[m], p[m], "spearman"),
    }


def make_group_folds(groups: Sequence[object], n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray([str(g) for g in groups])
    unique = np.array(sorted(pd.unique(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    buckets = [[] for _ in range(max(2, min(n_splits, len(unique))))]
    for i, g in enumerate(unique):
        buckets[i % len(buckets)].append(g)
    folds = []
    all_idx = np.arange(len(groups))
    for bucket in buckets:
        val_mask = np.isin(groups, bucket)
        val_idx = all_idx[val_mask]
        train_idx = all_idx[~val_mask]
        if len(val_idx) and len(train_idx):
            folds.append((train_idx, val_idx))
    return folds


def numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c in ID_COLS or c in TARGET_DERIVED_COLS:
            continue
        if c.startswith("source_clean_target_"):
            # Distribution anchors are allowed as scalar features; they are source-only constants.
            pass
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            cols.append(c)
    return cols


@dataclass
class Standardizer:
    med: Optional[np.ndarray] = None
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=float)
        med = np.nanmedian(X, axis=0)
        med[~np.isfinite(med)] = 0.0
        Xi = np.where(np.isfinite(X), X, med.reshape(1, -1))
        mean = Xi.mean(axis=0)
        std = Xi.std(axis=0)
        std[std < 1e-12] = 1.0
        self.med = med
        self.mean = mean
        self.std = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.med is not None and self.mean is not None and self.std is not None
        X = np.asarray(X, dtype=float)
        Xi = np.where(np.isfinite(X), X, self.med.reshape(1, -1))
        return (Xi - self.mean.reshape(1, -1)) / self.std.reshape(1, -1)


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X])


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    X1 = add_intercept(X)
    reg = np.eye(X1.shape[1]) * alpha
    reg[0, 0] = 0.0
    return np.linalg.pinv(X1.T @ X1 + reg) @ X1.T @ y


def ridge_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return add_intercept(X) @ beta


def pairwise_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A2 = np.sum(A * A, axis=1).reshape(-1, 1)
    B2 = np.sum(B * B, axis=1).reshape(1, -1)
    return np.maximum(A2 + B2 - 2.0 * (A @ B.T), 0.0)


def rbf_fit(X: np.ndarray, y: np.ndarray, alpha: float, gamma_scale: float):
    D = pairwise_sqdist(X, X)
    med = np.median(D[D > 1e-12]) if np.any(D > 1e-12) else 1.0
    gamma = gamma_scale / max(med, 1e-12)
    K = np.exp(-gamma * D)
    dual = np.linalg.pinv(K + alpha * np.eye(len(X))) @ y
    return X.copy(), dual, gamma


def rbf_predict(X: np.ndarray, fit_obj) -> np.ndarray:
    Xtr, dual, gamma = fit_obj
    return np.exp(-gamma * pairwise_sqdist(X, Xtr)) @ dual


def knn_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, k: int) -> np.ndarray:
    D = np.sqrt(pairwise_sqdist(Xte, Xtr))
    k = max(1, min(k, len(ytr)))
    idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    preds = []
    for i in range(Xte.shape[0]):
        dd = D[i, idx[i]]
        yy = ytr[idx[i]]
        w = 1.0 / np.maximum(dd, 1e-6)
        preds.append(float(np.sum(w * yy) / np.sum(w)))
    return np.asarray(preds, dtype=float)


def fold_safe_dopant_pred(train_df: pd.DataFrame, val_df: pd.DataFrame, mode: str) -> np.ndarray:
    agg = train_df.groupby("dopant")["target"].agg(mode)
    fallback = float(getattr(train_df["target"], mode)())
    return val_df["dopant"].map(agg).fillna(fallback).to_numpy(float)


def final_dopant_pred(train_df: pd.DataFrame, out_df: pd.DataFrame, mode: str = "mean") -> np.ndarray:
    agg = train_df.groupby("dopant")["target"].agg(mode)
    fallback = float(getattr(train_df["target"], mode)())
    return out_df["dopant"].map(agg).fillna(fallback).to_numpy(float)


def evaluate_model(
    name: str,
    kind: str,
    params: Dict[str, float],
    X: np.ndarray,
    y: np.ndarray,
    Xout: np.ndarray,
    train_df: pd.DataFrame,
    out_df: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    oof = np.full(len(y), np.nan, dtype=float)
    out_fold_preds = []
    for tr_idx, va_idx in folds:
        if kind.startswith("dopant_"):
            mode = "mean" if kind == "dopant_mean" else "median"
            pred_va = fold_safe_dopant_pred(train_df.iloc[tr_idx], train_df.iloc[va_idx], mode)
            pred_out = final_dopant_pred(train_df.iloc[tr_idx], out_df, mode)
        else:
            scaler = Standardizer().fit(X[tr_idx])
            Xtr = scaler.transform(X[tr_idx])
            Xva = scaler.transform(X[va_idx])
            Xo = scaler.transform(Xout)
            if kind == "ridge":
                beta = ridge_fit(Xtr, y[tr_idx], params["alpha"])
                pred_va = ridge_predict(Xva, beta)
                pred_out = ridge_predict(Xo, beta)
            elif kind == "rbf":
                fit_obj = rbf_fit(Xtr, y[tr_idx], params["alpha"], params["gamma_scale"])
                pred_va = rbf_predict(Xva, fit_obj)
                pred_out = rbf_predict(Xo, fit_obj)
            elif kind == "knn":
                pred_va = knn_predict(Xtr, y[tr_idx], Xva, int(params["k"]))
                pred_out = knn_predict(Xtr, y[tr_idx], Xo, int(params["k"]))
            else:
                raise ValueError(kind)
        oof[va_idx] = pred_va
        out_fold_preds.append(pred_out)

    met = metrics(y, oof)
    met.update({"model": name, "kind": kind})
    met.update(params)

    if kind.startswith("dopant_"):
        mode = "mean" if kind == "dopant_mean" else "median"
        out_full = final_dopant_pred(train_df, out_df, mode)
    else:
        scaler = Standardizer().fit(X)
        Xs = scaler.transform(X)
        Xo = scaler.transform(Xout)
        if kind == "ridge":
            out_full = ridge_predict(Xo, ridge_fit(Xs, y, params["alpha"]))
        elif kind == "rbf":
            out_full = rbf_predict(Xo, rbf_fit(Xs, y, params["alpha"], params["gamma_scale"]))
        else:
            out_full = knn_predict(Xs, y, Xo, int(params["k"]))

    out_cv_mean = np.nanmean(np.vstack(out_fold_preds), axis=0)
    # Blend full fit with fold-mean for a bit of stability on small data.
    out_pred = 0.70 * out_full + 0.30 * out_cv_mean
    return met, oof, out_pred


def model_grid() -> List[Tuple[str, str, Dict[str, float]]]:
    grid: List[Tuple[str, str, Dict[str, float]]] = []
    for a in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        grid.append((f"ridge_a{a:g}", "ridge", {"alpha": a}))
    for a in [0.1, 1.0, 10.0]:
        for gs in [0.25, 0.5, 1.0, 2.0]:
            grid.append((f"rbf_a{a:g}_g{gs:g}", "rbf", {"alpha": a, "gamma_scale": gs}))
    for k in [3, 5, 9, 15, 25]:
        grid.append((f"knn_k{k}", "knn", {"k": float(k)}))
    grid.append(("dopant_mean_foldsafe", "dopant_mean", {}))
    grid.append(("dopant_median_foldsafe", "dopant_median", {}))
    return grid


def detect_pred_col(df: pd.DataFrame) -> Optional[str]:
    for c in PRED_COL_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = str(c).lower()
        if "pred" in lc and not any(x in lc for x in ["std", "min", "max", "err", "rank", "uncertainty", "n_"]):
            return c
    return None


def candidate_prediction_paths(scan_roots: List[str], explicit: List[str]) -> List[Path]:
    paths = [Path(p) for p in explicit]
    for root in scan_roots:
        r = Path(root)
        if not r.exists():
            continue
        for p in r.rglob("*.csv"):
            name = p.name.lower()
            if any(bad in name for bad in ["selection_summary", "selected_models", "weights", "audit"]):
                continue
            if ("pred" in name or "prediction" in name) and ("addh" in name or "strict" in name):
                paths.append(p)
    seen = set()
    out = []
    for p in paths:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def load_base_predictions(
    addhout: pd.DataFrame,
    paths: List[Path],
    source_anchor_mean: float,
    min_coverage: float,
    max_mean_shift: float,
    allow_biased: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    base = addhout[["id"]].drop_duplicates().copy()
    summary_rows = []
    pred_cols: List[str] = []
    for p in paths:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "id" not in df.columns:
            continue
        pc = detect_pred_col(df)
        if not pc:
            continue
        tmp = df[["id", pc]].copy()
        tmp["id"] = tmp["id"].astype(str)
        tmp[pc] = pd.to_numeric(tmp[pc], errors="coerce")
        tag = re.sub(r"[^A-Za-z0-9]+", "_", str(p.with_suffix("")))[-80:].strip("_")
        col = f"base_pred__{tag}"
        tmp = tmp.rename(columns={pc: col})
        aligned = addhout[["id"]].merge(tmp, on="id", how="left")
        vals = pd.to_numeric(aligned[col], errors="coerce")
        coverage = float(vals.notna().mean())
        mean_pred = float(vals.mean(skipna=True)) if vals.notna().any() else np.nan
        median_pred = float(vals.median(skipna=True)) if vals.notna().any() else np.nan
        std_pred = float(vals.std(skipna=True)) if vals.notna().any() else np.nan
        mean_shift = abs(mean_pred - source_anchor_mean) if np.isfinite(mean_pred) else np.inf
        keep = coverage >= min_coverage and (allow_biased or mean_shift <= max_mean_shift)
        summary_rows.append({
            "path": str(p),
            "pred_col": pc,
            "aligned_col": col,
            "coverage": coverage,
            "mean_pred": mean_pred,
            "median_pred": median_pred,
            "std_pred": std_pred,
            "source_anchor_mean": source_anchor_mean,
            "abs_mean_shift_vs_source": mean_shift,
            "kept_for_base_pool": bool(keep),
        })
        if keep:
            base = base.merge(aligned[["id", col]], on="id", how="left")
            pred_cols.append(col)
    return base, pd.DataFrame(summary_rows), pred_cols


def rowwise_weighted_blend(parts: Dict[str, Tuple[np.ndarray, float]]) -> np.ndarray:
    names = list(parts)
    V = np.vstack([np.asarray(parts[n][0], dtype=float) for n in names]).T
    W0 = np.asarray([parts[n][1] for n in names], dtype=float)
    W = np.broadcast_to(W0.reshape(1, -1), V.shape).copy()
    W[~np.isfinite(V)] = 0.0
    wsum = W.sum(axis=1)
    out = np.full(V.shape[0], np.nan, dtype=float)
    ok = wsum > 0
    out[ok] = np.nansum(np.where(np.isfinite(V), V, 0.0) * W, axis=1)[ok] / wsum[ok]
    return out


def trimmed_mean_matrix(M: np.ndarray, trim: float = 0.1) -> np.ndarray:
    vals = []
    for row in M:
        x = row[np.isfinite(row)]
        if len(x) == 0:
            vals.append(np.nan)
            continue
        x = np.sort(x)
        k = int(math.floor(len(x) * trim))
        if k > 0 and 2 * k < len(x):
            x = x[k:-k]
        vals.append(float(np.mean(x)))
    return np.asarray(vals)


def audit_predictions(pred: pd.DataFrame, labels: pd.DataFrame, pred_cols: List[str], out_dir: Path) -> pd.DataFrame:
    lab_col = "h_ads_excel" if "h_ads_excel" in labels.columns else ("target" if "target" in labels.columns else None)
    if lab_col is None:
        return pd.DataFrame()
    df = pred.merge(labels[["id", lab_col, "material", "dopant"]].drop_duplicates("id"), on="id", how="left", suffixes=("", "_label"))
    rows = []
    for c in pred_cols:
        if c not in df.columns:
            continue
        met = metrics(df[lab_col], df[c])
        met["pred_col"] = c
        met["target_col"] = lab_col
        rows.append(met)
        for mat, g in df.groupby("material"):
            mm = metrics(g[lab_col], g[c])
            mm["pred_col"] = c
            mm["target_col"] = lab_col
            mm["material"] = mat
            rows.append(mm)
    audit = pd.DataFrame(rows)
    audit.to_csv(out_dir / "knowledge_blend_posthoc_audit.csv", index=False)
    detail = df.copy()
    if "pred_llm_element_knowledge_blend" in detail.columns:
        detail["err_final"] = pd.to_numeric(detail["pred_llm_element_knowledge_blend"], errors="coerce") - pd.to_numeric(detail[lab_col], errors="coerce")
        detail["abs_err_final"] = detail["err_final"].abs()
        detail.sort_values("abs_err_final", ascending=False).to_csv(out_dir / "knowledge_blend_posthoc_audit_detail.csv", index=False)
    return audit


def main() -> None:
    args = parse_args()
    feature_dir = Path(args.feature_dir)
    train_path = Path(args.train_features) if args.train_features else feature_dir / "knowledge_features_train.csv"
    addhout_path = Path(args.addhout_features) if args.addhout_features else feature_dir / "knowledge_features_addhout.csv"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_all = pd.read_csv(train_path)
    addhout = pd.read_csv(addhout_path)
    train_all["target"] = pd.to_numeric(train_all[args.target_col], errors="coerce")
    usable = train_all["target"].notna() & (train_all["target"].abs() <= args.target_abs_max)
    train = train_all.loc[usable].reset_index(drop=True).copy()
    addhout = addhout.reset_index(drop=True).copy()

    feature_cols = sorted(set(numeric_feature_columns(train)).intersection(numeric_feature_columns(addhout)))
    if not feature_cols:
        raise SystemExit("[ERROR] no numeric feature columns found.")
    X = train[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    Xout = addhout[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = train["target"].to_numpy(float)
    groups = train[args.group_col].fillna(train["id"]).astype(str).to_numpy()
    folds = make_group_folds(groups, args.n_splits, args.seed)
    if not folds:
        raise SystemExit("[ERROR] no valid grouped CV folds.")

    print(f"[INFO] train rows usable={len(train)} / raw={len(train_all)}")
    print(f"[INFO] addH-out rows={len(addhout)}")
    print(f"[INFO] numeric feature columns={len(feature_cols)}")
    print(f"[INFO] grouped folds={len(folds)} groups={pd.Series(groups).nunique()}")

    model_rows = []
    oof_table = train[["id", "family_base", "dopant", "target"]].copy()
    out_model_preds = addhout[["id", "material", "dopant"]].copy()
    for name, kind, params in model_grid():
        try:
            met, oof, out_pred = evaluate_model(name, kind, params, X, y, Xout, train, addhout, folds)
        except Exception as e:
            print(f"[WARN] model failed {name}: {e}")
            continue
        model_rows.append(met)
        oof_table[f"oof__{name}"] = oof
        out_model_preds[f"pred__{name}"] = out_pred
        print(f"[MODEL] {name:24s} OOF MAE={met['mae']:.4f} RMSE={met['rmse']:.4f} bias={met['bias']:.4f}")

    model_summary = pd.DataFrame(model_rows).sort_values(["mae", "rmse"], ascending=[True, True])
    if model_summary.empty:
        raise SystemExit("[ERROR] every model failed.")
    model_summary.to_csv(out_dir / "knowledge_model_oof_metrics.csv", index=False)
    oof_table.to_csv(out_dir / "knowledge_model_oof_predictions.csv", index=False)

    selected = model_summary.head(args.top_knowledge_models).copy()
    weights = 1.0 / np.maximum(selected["mae"].to_numpy(float), 1e-6) ** 2
    weights = weights / weights.sum()
    selected["knowledge_weight"] = weights
    selected.to_csv(out_dir / "knowledge_selected_models.csv", index=False)

    pred_parts: Dict[str, Tuple[np.ndarray, float]] = {}
    for (_, row), w in zip(selected.iterrows(), weights):
        col = f"pred__{row['model']}"
        pred_parts[str(row["model"])] = (out_model_preds[col].to_numpy(float), float(w))
    knowledge_pred = rowwise_weighted_blend(pred_parts)
    dopant_pred = final_dopant_pred(train, addhout, "mean")

    source_anchor_mean = float(train["target"].mean())
    pred_paths = candidate_prediction_paths(args.scan_pred_root, args.base_pred_file)
    base_df, base_summary, base_cols = load_base_predictions(
        addhout,
        pred_paths,
        source_anchor_mean,
        args.min_base_coverage,
        args.max_blind_mean_shift,
        args.allow_biased_base_preds,
    )
    base_summary.to_csv(out_dir / "base_prediction_file_selection.csv", index=False)
    if base_cols:
        B = base_df[base_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        base_median = np.full(B.shape[0], np.nan, dtype=float)
        base_std = np.full(B.shape[0], np.nan, dtype=float)
        finite_rows = np.isfinite(B).any(axis=1)
        if finite_rows.any():
            base_median[finite_rows] = np.nanmedian(B[finite_rows], axis=1)
            base_std[finite_rows] = np.nanstd(B[finite_rows], axis=1)
        base_trimmed = trimmed_mean_matrix(B, 0.1)
        base_raw = np.where(np.isfinite(base_trimmed), base_trimmed, base_median)
        base_recenter = base_raw - np.nanmedian(base_raw) + np.nanmedian(knowledge_pred)
        base_n = np.isfinite(B).sum(axis=1)
    else:
        base_raw = np.full(len(addhout), np.nan)
        base_recenter = np.full(len(addhout), np.nan)
        base_n = np.zeros(len(addhout), dtype=int)
        base_std = np.full(len(addhout), np.nan)

    final_pred = rowwise_weighted_blend({
        "knowledge": (knowledge_pred, args.blend_knowledge),
        "dopant_stat": (dopant_pred, args.blend_dopant),
        "base_recenter": (base_recenter, args.blend_base_recenter),
        "base_raw": (base_raw, args.blend_base_raw),
    })

    out = addhout.copy()
    out["pred_knowledge_model"] = knowledge_pred
    out["pred_source_dopant_mean_prior"] = dopant_pred
    out["pred_base_pool_raw"] = base_raw
    out["pred_base_pool_recenter_to_knowledge"] = base_recenter
    out["pred_base_pool_std"] = base_std
    out["pred_base_pool_n"] = base_n
    out["pred_llm_element_knowledge_blend"] = final_pred
    out["knowledge_blend_rank"] = pd.Series(final_pred).rank(method="average", ascending=True).to_numpy()
    out = out.sort_values(["knowledge_blend_rank", "pred_base_pool_std"], na_position="last")

    out_csv = out_dir / "knowledge_enhanced_addhout_predictions.csv"
    out_xlsx = out_dir / "knowledge_enhanced_addhout_predictions.xlsx"
    out.to_csv(out_csv, index=False)
    try:
        out.to_excel(out_xlsx, index=False)
    except Exception:
        pass

    manifest = {
        "train_features": str(train_path),
        "addhout_features": str(addhout_path),
        "usable_train_rows": int(len(train)),
        "raw_train_rows": int(len(train_all)),
        "addhout_rows": int(len(addhout)),
        "n_features": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "selected_knowledge_models": selected[["model", "mae", "rmse", "bias", "knowledge_weight"]].to_dict(orient="records"),
        "base_prediction_files_kept": base_summary[base_summary.get("kept_for_base_pool", False) == True]["path"].tolist() if len(base_summary) else [],
        "blend_weights": {
            "knowledge": args.blend_knowledge,
            "dopant": args.blend_dopant,
            "base_recenter": args.blend_base_recenter,
            "base_raw": args.blend_base_raw,
        },
        "strict_blind_prediction": True,
        "audit_labels_used_for_training_or_selection": False,
    }
    with (out_dir / "knowledge_blend_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if args.audit_labels_csv:
        labels = pd.read_csv(args.audit_labels_csv)
        audit_cols = [
            "pred_llm_element_knowledge_blend",
            "pred_knowledge_model",
            "pred_source_dopant_mean_prior",
            "pred_base_pool_raw",
            "pred_base_pool_recenter_to_knowledge",
        ]
        audit = audit_predictions(out, labels, audit_cols, out_dir)
        if len(audit):
            print("[POSTHOC AUDIT ONLY]")
            print(audit.to_string(index=False))

    print("[OK] wrote", out_csv)
    print("[OK] wrote", out_xlsx)
    print("[OK] wrote", out_dir / "knowledge_model_oof_metrics.csv")
    print("[OK] wrote", out_dir / "base_prediction_file_selection.csv")
    print("[INFO] selected knowledge models:")
    print(selected[["model", "mae", "rmse", "bias", "knowledge_weight"]].to_string(index=False))
    if len(base_summary):
        kept = base_summary[base_summary["kept_for_base_pool"]]
        print(f"[INFO] kept base prediction files: {len(kept)} / {len(base_summary)}")
        if len(kept):
            print(kept[["path", "pred_col", "coverage", "mean_pred", "abs_mean_shift_vs_source"]].to_string(index=False))


if __name__ == "__main__":
    main()
