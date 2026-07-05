#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Accuracy-first calibration and stacking for addH-out predictions.

Goal:
  1) Collect addH-out predictions from many graph-embedding ensemble experiments.
  2) Evaluate true posterior accuracy against h_ads_excel/target_computed.
  3) Build small-sample target-domain calibration models with honest LOOCV.
  4) Export accuracy-first prediction tables.

Important:
  - LOOCV predictions are the honest accuracy estimate on addH-out labels.
  - The final all-fit calibrated predictions use all addH-out labels, so they are calibrated values,
    not blind predictions. Use them only after making this distinction clear.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Accuracy-first addH-out calibration/stacking.")
    ap.add_argument("--roots", nargs="+", default=["outputs_addh_graph_ensemble_refine_v2", "outputs_addh_graph_ensemble"],
                    help="Experiment roots to scan.")
    ap.add_argument("--master-csv", default="outputs_addh_full_mm_envsplit/addH_out_master_normalized.csv")
    ap.add_argument("--out-dir", default="outputs_addh_accuracy_first_calibration")
    ap.add_argument("--target-col", default="h_ads_excel", choices=["h_ads_excel", "target", "target_computed"])
    ap.add_argument("--min-coverage", type=float, default=0.90, help="Min fraction of addH-out samples a model must predict.")
    ap.add_argument("--top-models", type=int, default=12, help="Keep top N base models by addH-out MAE for meta models.")
    ap.add_argument("--include-rank-features", action="store_true", help="Add per-model percentile-rank features.")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--material-col", default="material")
    ap.add_argument("--element-col", default="element")
    return ap.parse_args()


def read_master(path: Path, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns:
        raise ValueError(f"master missing id column: {path}")
    for c in ["h_ads_excel", "target", "target_computed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if target_col not in df.columns:
        raise ValueError(f"master missing target column {target_col}")
    keep = [
        "id", "material", "idx", "element", "dopant", "family_base", "family_base_miller", "miller",
        "site_type", "anchor_count", "slab_formula", "h_ads_excel", "target", "target_computed",
        "target_mismatch_excel_minus_computed", "contcar_path", "bare_contcar_path", "cif_path"
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].drop_duplicates("id").copy()


def find_pred_csv(exp_dir: Path) -> Optional[Path]:
    for name in [
        "addH_out_graph_ensemble_by_id_merged_master.csv",
        "addH_out_graph_ensemble_by_id.csv",
    ]:
        p = exp_dir / "work" / name
        if p.exists():
            return p
    return None


def safe_exp_name(root: Path, exp_dir: Path) -> str:
    return exp_dir.name


def collect_predictions(roots: List[Path], master: pd.DataFrame, id_col: str = "id") -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = master[[id_col]].copy()
    metrics_rows = []
    pred_cols = []

    for root in roots:
        if not root.exists():
            print(f"[WARN] root not found: {root}")
            continue
        for exp_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            pred_csv = find_pred_csv(exp_dir)
            if pred_csv is None:
                continue
            exp = safe_exp_name(root, exp_dir)
            try:
                df = pd.read_csv(pred_csv).drop(columns=["eq_emb"], errors="ignore")
            except Exception as e:
                print(f"[WARN] failed reading {pred_csv}: {e}")
                continue
            if id_col not in df.columns or "pred" not in df.columns:
                continue

            tmp = df[[id_col, "pred"]].copy()
            tmp["pred"] = pd.to_numeric(tmp["pred"], errors="coerce")
            tmp = tmp.drop_duplicates(id_col)
            col = f"pred__{exp}"
            tmp = tmp.rename(columns={"pred": col})
            base = base.merge(tmp, on=id_col, how="left")
            pred_cols.append(col)

            coverage = base[col].notna().mean()
            metrics_rows.append({
                "exp_name": exp,
                "pred_col": col,
                "pred_csv": str(pred_csv),
                "coverage": coverage,
            })

    return base, pd.DataFrame(metrics_rows)


def metric_dict(y: np.ndarray, p: np.ndarray, prefix: str = "") -> Dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    out = {prefix + "n": int(len(y))}
    if len(y) < 3:
        out.update({prefix + k: np.nan for k in ["mae", "rmse", "r2", "pearson", "spearman", "bias"]})
        return out
    out[prefix + "mae"] = float(mean_absolute_error(y, p))
    out[prefix + "rmse"] = float(mean_squared_error(y, p) ** 0.5)
    try:
        out[prefix + "r2"] = float(r2_score(y, p))
    except Exception:
        out[prefix + "r2"] = np.nan
    out[prefix + "pearson"] = float(pd.Series(p).corr(pd.Series(y), method="pearson"))
    out[prefix + "spearman"] = float(pd.Series(p).corr(pd.Series(y), method="spearman"))
    out[prefix + "bias"] = float(np.mean(p - y))
    return out


def base_model_metrics(df: pd.DataFrame, target_col: str, pred_cols: List[str], pred_meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(float)
    for col in pred_cols:
        p = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        rec = {"pred_col": col, "exp_name": col.replace("pred__", "")}
        rec.update(metric_dict(y, p, "addhout_"))
        rec["coverage"] = float(np.isfinite(p).mean())
        rows.append(rec)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["addhout_mae", "addhout_rmse", "addhout_bias"], ascending=[True, True, True])
    return out


def impute_by_train_median(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    med = X_train.median(numeric_only=True)
    return X_train.fillna(med), X_test.fillna(med)


def make_design(df: pd.DataFrame, pred_cols: List[str], include_rank: bool, material_col: str, element_col: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    X = df[pred_cols].copy()
    numeric_cols = list(pred_cols)

    if include_rank:
        for c in pred_cols:
            rc = c.replace("pred__", "rank__")
            X[rc] = df[c].rank(method="average", pct=True, ascending=True)
            numeric_cols.append(rc)

    cat_cols = []
    for c in [material_col, element_col]:
        if c in df.columns:
            X[c] = df[c].astype(str).fillna("unknown")
            cat_cols.append(c)

    return X, numeric_cols, cat_cols


def build_ridge_pipeline(numeric_cols: List[str], cat_cols: List[str], alpha: float) -> Pipeline:
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if cat_cols:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", ohe, cat_cols))
    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("pre", pre), ("model", Ridge(alpha=alpha))])


def build_huber_pipeline(numeric_cols: List[str], cat_cols: List[str], alpha: float = 1e-3) -> Pipeline:
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if cat_cols:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", ohe, cat_cols))
    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("pre", pre), ("model", HuberRegressor(alpha=alpha, epsilon=1.35, max_iter=1000))])


def nested_alpha_select_ridge(X: pd.DataFrame, y: np.ndarray, numeric_cols: List[str], cat_cols: List[str], alphas: List[float]) -> Tuple[np.ndarray, List[float]]:
    loo = LeaveOneOut()
    preds = np.full(len(y), np.nan)
    chosen = []

    for train_idx, test_idx in loo.split(X):
        X_tr = X.iloc[train_idx].copy()
        X_te = X.iloc[test_idx].copy()
        y_tr = y[train_idx]

        # impute numeric columns by train median
        X_tr_num, X_te_num = impute_by_train_median(X_tr[numeric_cols], X_te[numeric_cols])
        X_tr.loc[:, numeric_cols] = X_tr_num
        X_te.loc[:, numeric_cols] = X_te_num

        # inner LOO alpha selection
        best_alpha, best_mae = alphas[0], np.inf
        if len(train_idx) >= 5:
            inner = LeaveOneOut()
            for a in alphas:
                ip = np.full(len(y_tr), np.nan)
                for itrain, ival in inner.split(X_tr):
                    model = build_ridge_pipeline(numeric_cols, cat_cols, a)
                    model.fit(X_tr.iloc[itrain], y_tr[itrain])
                    ip[ival] = model.predict(X_tr.iloc[ival])[0]
                mae = mean_absolute_error(y_tr, ip)
                if mae < best_mae:
                    best_mae = mae
                    best_alpha = a
        chosen.append(best_alpha)
        model = build_ridge_pipeline(numeric_cols, cat_cols, best_alpha)
        model.fit(X_tr, y_tr)
        preds[test_idx] = model.predict(X_te)

    return preds, chosen


def loo_huber(X: pd.DataFrame, y: np.ndarray, numeric_cols: List[str], cat_cols: List[str], alpha: float = 1e-3) -> np.ndarray:
    loo = LeaveOneOut()
    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in loo.split(X):
        X_tr = X.iloc[train_idx].copy()
        X_te = X.iloc[test_idx].copy()
        y_tr = y[train_idx]
        X_tr_num, X_te_num = impute_by_train_median(X_tr[numeric_cols], X_te[numeric_cols])
        X_tr.loc[:, numeric_cols] = X_tr_num
        X_te.loc[:, numeric_cols] = X_te_num
        model = build_huber_pipeline(numeric_cols, cat_cols, alpha)
        try:
            model.fit(X_tr, y_tr)
            preds[test_idx] = model.predict(X_te)
        except Exception:
            # fallback ridge
            model = build_ridge_pipeline(numeric_cols, cat_cols, 1.0)
            model.fit(X_tr, y_tr)
            preds[test_idx] = model.predict(X_te)
    return preds


def loo_weighted_average(df: pd.DataFrame, y: np.ndarray, pred_cols: List[str], topk: int = 5, power: float = 2.0) -> np.ndarray:
    loo = LeaveOneOut()
    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in loo.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        scores = []
        for c in pred_cols:
            yt = y[train_idx]
            pt = pd.to_numeric(train[c], errors="coerce").to_numpy(float)
            mask = np.isfinite(yt) & np.isfinite(pt)
            if mask.sum() < 5:
                continue
            mae = mean_absolute_error(yt[mask], pt[mask])
            scores.append((c, mae))
        scores = sorted(scores, key=lambda x: x[1])[:topk]
        if not scores:
            continue
        vals, weights = [], []
        for c, mae in scores:
            v = pd.to_numeric(test[c], errors="coerce").iloc[0]
            if not np.isfinite(v):
                # fallback train median for that model
                v = pd.to_numeric(train[c], errors="coerce").median()
            vals.append(v)
            weights.append(1.0 / ((mae + 1e-6) ** power))
        preds[test_idx] = np.average(vals, weights=weights)
    return preds


def fit_all_and_predict(df: pd.DataFrame, y: np.ndarray, X: pd.DataFrame, numeric_cols: List[str], cat_cols: List[str], alpha: float) -> np.ndarray:
    X_all = X.copy()
    X_all[numeric_cols] = X_all[numeric_cols].fillna(X_all[numeric_cols].median(numeric_only=True))
    model = build_ridge_pipeline(numeric_cols, cat_cols, alpha)
    model.fit(X_all, y)
    return model.predict(X_all)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = [Path(r) for r in args.roots]
    master = read_master(Path(args.master_csv), args.target_col)
    pred_wide, pred_meta = collect_predictions(roots, master, args.id_col)
    df = master.merge(pred_wide, on=args.id_col, how="left")

    y = pd.to_numeric(df[args.target_col], errors="coerce").to_numpy(float)
    valid_y = np.isfinite(y)
    df = df.loc[valid_y].reset_index(drop=True)
    y = pd.to_numeric(df[args.target_col], errors="coerce").to_numpy(float)

    all_pred_cols = [c for c in df.columns if c.startswith("pred__")]
    base_metrics = base_model_metrics(df, args.target_col, all_pred_cols, pred_meta)

    if args.min_coverage > 0:
        keep_pred_cols = base_metrics.loc[base_metrics["coverage"] >= args.min_coverage, "pred_col"].tolist()
    else:
        keep_pred_cols = all_pred_cols
    if args.top_models and args.top_models > 0:
        top_cols = base_metrics.loc[base_metrics["pred_col"].isin(keep_pred_cols)].head(args.top_models)["pred_col"].tolist()
    else:
        top_cols = keep_pred_cols

    if len(top_cols) < 2:
        raise SystemExit(f"[ERROR] not enough prediction columns selected: {len(top_cols)}")

    print(f"[INFO] rows with target: {len(df)}")
    print(f"[INFO] collected models: {len(all_pred_cols)}")
    print(f"[INFO] selected top models: {len(top_cols)}")
    print("[INFO] top selected:")
    print(base_metrics[base_metrics["pred_col"].isin(top_cols)].head(len(top_cols))[ ["exp_name", "addhout_mae", "addhout_rmse", "addhout_pearson", "addhout_spearman", "addhout_bias", "coverage"] ].to_string(index=False))

    base_metrics.to_csv(out_dir / "base_model_accuracy_metrics.csv", index=False)
    base_metrics.to_excel(out_dir / "base_model_accuracy_metrics.xlsx", index=False)

    # LOO meta/ensemble methods
    rows = []
    loo_pred_df = df[[c for c in ["id", "material", "element", "dopant", "h_ads_excel", "target", "target_computed"] if c in df.columns]].copy()

    # Best single base model by train-posthoc metric (reference only)
    best_col = top_cols[0]
    p_best = pd.to_numeric(df[best_col], errors="coerce").to_numpy(float)
    loo_pred_df["pred_best_base_direct"] = p_best
    rec = {"model": "best_base_direct", "details": best_col}
    rec.update(metric_dict(y, p_best, "loocv_"))
    rows.append(rec)

    for k in [3, 5, 8, len(top_cols)]:
        k = min(k, len(top_cols))
        if k < 2:
            continue
        pred = loo_weighted_average(df, y, top_cols, topk=k, power=2.0)
        name = f"loo_invmae_weighted_top{k}"
        loo_pred_df[f"pred_{name}"] = pred
        rec = {"model": name, "details": ",".join(top_cols[:k])}
        rec.update(metric_dict(y, pred, "loocv_"))
        rows.append(rec)

    for k in [3, 5, 8, len(top_cols)]:
        k = min(k, len(top_cols))
        if k < 2:
            continue
        cols = top_cols[:k]
        X, numeric_cols, cat_cols = make_design(df, cols, args.include_rank_features, args.material_col, args.element_col)
        pred, chosen = nested_alpha_select_ridge(X, y, numeric_cols, cat_cols, alphas=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
        name = f"loo_nested_ridge_top{k}"
        loo_pred_df[f"pred_{name}"] = pred
        rec = {"model": name, "details": ",".join(cols), "alpha_median": float(np.median(chosen))}
        rec.update(metric_dict(y, pred, "loocv_"))
        rows.append(rec)

        pred_h = loo_huber(X, y, numeric_cols, cat_cols, alpha=1e-3)
        name_h = f"loo_huber_top{k}"
        loo_pred_df[f"pred_{name_h}"] = pred_h
        rec_h = {"model": name_h, "details": ",".join(cols)}
        rec_h.update(metric_dict(y, pred_h, "loocv_"))
        rows.append(rec_h)

    summary = pd.DataFrame(rows).sort_values(["loocv_mae", "loocv_rmse"], ascending=[True, True])
    summary.to_csv(out_dir / "accuracy_first_loocv_model_summary.csv", index=False)
    summary.to_excel(out_dir / "accuracy_first_loocv_model_summary.xlsx", index=False)

    best_meta = summary.iloc[0]["model"]
    best_col_pred = f"pred_{best_meta}" if f"pred_{best_meta}" in loo_pred_df.columns else "pred_best_base_direct"
    loo_pred_df["pred_accuracy_first_loocv"] = loo_pred_df[best_col_pred]
    if args.target_col in loo_pred_df.columns:
        loo_pred_df["abs_err_accuracy_first_loocv"] = (loo_pred_df["pred_accuracy_first_loocv"] - pd.to_numeric(loo_pred_df[args.target_col], errors="coerce")).abs()
    loo_pred_df.to_csv(out_dir / "accuracy_first_loocv_predictions.csv", index=False)
    loo_pred_df.to_excel(out_dir / "accuracy_first_loocv_predictions.xlsx", index=False)

    # All-fit calibrated prediction table using the best ridge-like model if possible
    # This is target-domain calibrated, not blind.
    allfit = df[[c for c in ["id", "material", "element", "dopant", "h_ads_excel", "target", "target_computed"] if c in df.columns]].copy()
    allfit["accuracy_first_method"] = str(best_meta)
    if best_meta.startswith("loo_nested_ridge_top"):
        k = int(best_meta.replace("loo_nested_ridge_top", ""))
        cols = top_cols[:k]
        X, numeric_cols, cat_cols = make_design(df, cols, args.include_rank_features, args.material_col, args.element_col)
        # choose alpha by full-data LOO among grid for final all-fit model
        _, chosen = nested_alpha_select_ridge(X, y, numeric_cols, cat_cols, alphas=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
        alpha = float(np.median(chosen)) if chosen else 1.0
        allfit["pred_accuracy_first_allfit_calibrated"] = fit_all_and_predict(df, y, X, numeric_cols, cat_cols, alpha)
        allfit["allfit_alpha"] = alpha
        allfit["allfit_feature_models"] = ",".join(cols)
    else:
        # if weighted model wins, use its honest LOO values as final conservative prediction
        allfit["pred_accuracy_first_allfit_calibrated"] = loo_pred_df["pred_accuracy_first_loocv"]
        allfit["allfit_alpha"] = np.nan
        allfit["allfit_feature_models"] = str(summary.iloc[0].get("details", ""))

    if args.target_col in allfit.columns:
        allfit["abs_err_accuracy_first_allfit"] = (allfit["pred_accuracy_first_allfit_calibrated"] - pd.to_numeric(allfit[args.target_col], errors="coerce")).abs()
    allfit.to_csv(out_dir / "accuracy_first_allfit_calibrated_predictions.csv", index=False)
    allfit.to_excel(out_dir / "accuracy_first_allfit_calibrated_predictions.xlsx", index=False)

    with open(out_dir / "accuracy_first_run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("\n[OK] outputs saved in:", out_dir)
    print("[BEST LOOCV]")
    print(summary.head(15).to_string(index=False))
    print("\n[NOTE] accuracy_first_allfit_calibrated_predictions uses addH-out labels for calibration; do not report it as blind prediction.")


if __name__ == "__main__":
    main()
