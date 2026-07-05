#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_num(x):
    return pd.to_numeric(x, errors='coerce')


def clean_cat(s: pd.Series) -> pd.Series:
    s = s.fillna('').astype(str).str.strip()
    s = s.replace({'nan': '', 'None': '', 'null': '', 'unknown': ''})
    return s.apply(lambda x: x if x else 'unknown')


def drop_empty_feature_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that have no usable information.

    This avoids repeated sklearn warnings such as:
    'Skipping features without any observed values'.
    """
    keep_cols = []
    for c in X.columns:
        ser = X[c]
        if pd.api.types.is_numeric_dtype(ser):
            if pd.to_numeric(ser, errors='coerce').notna().any():
                keep_cols.append(c)
        else:
            vals = clean_cat(ser)
            # Keep categorical columns if at least one non-unknown category exists,
            # or if the column has more than one category after cleaning.
            if (vals != 'unknown').any() or vals.nunique(dropna=False) > 1:
                keep_cols.append(c)
    return X[keep_cols].copy()


TARGETS_REG = ['H_adsorption_energy_value', 'H_adsorption_free_energy_value', 'H_adsorption_energy_value_eV', 'H_adsorption_free_energy_value_eV', 'OH_adsorption_energy_value_eV', 'OH_adsorption_free_energy_value_eV', 'H2O_adsorption_energy_value_eV', 'deprotonation_energy_value_eV', 'proton_transfer_barrier_eV', 'Volmer_barrier_eV', 'water_dissociation_barrier_eV', 'joint_hads_score']

DROP_COLS = {
    'paper_id', 'record_id', 'system_local_id', 'site_local_id', 'relpath', 'bucket', 'reaction',
    'evidence_json', 'evidence_all', 'system_signature', 'site_signature', 'adsorption_signature',
    'system_confidence_score_raw', 'site_confidence_score_raw', 'adsorption_confidence_score_raw',
    'system_confidence_score_status', 'site_confidence_score_status', 'adsorption_confidence_score_status',
}


def build_feature_matrix(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    use = df.copy()
    for c in use.columns:
        if use[c].dtype == object:
            use[c] = clean_cat(use[c])
    y = to_num(use[target_col])
    keep = y.notna()
    use = use.loc[keep].copy()
    y = y.loc[keep]
    feat_cols = [c for c in use.columns if c not in DROP_COLS and c != target_col and c not in TARGETS_REG]
    X = use[feat_cols].copy()
    X = drop_empty_feature_columns(X)
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return pd.concat([X, y.rename(target_col)], axis=1), num_cols, cat_cols


def make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', ohe),
    ])
    return ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols),
    ], remainder='drop')


def get_feature_names(preprocessor: ColumnTransformer, X_fit: pd.DataFrame, fallback_num: List[str], fallback_cat: List[str]) -> List[str]:
    try:
        names = preprocessor.get_feature_names_out().tolist()
        if names:
            return [str(x) for x in names]
    except Exception:
        pass
    names: List[str] = []
    names.extend(fallback_num)
    try:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        names.extend(ohe.get_feature_names_out(fallback_cat).tolist())
    except Exception:
        names.extend(fallback_cat)
    return names


def make_group_split(groups: pd.Series, n_splits: int, seed: int):
    n_groups = int(groups.astype(str).nunique())
    if n_groups >= 2:
        k = min(max(2, n_splits), n_groups)
        return GroupKFold(n_splits=k)
    return KFold(n_splits=min(max(2, n_splits), 5), shuffle=True, random_state=seed)


def run_regression(df: pd.DataFrame, target_col: str, group_col: str, outdir: Path, n_splits: int, seed: int) -> None:
    work, num_cols, cat_cols = build_feature_matrix(df, target_col)
    if work.empty or len(work) < 8:
        return
    X = work.drop(columns=[target_col])
    y = to_num(work[target_col])
    groups = clean_cat(df.loc[work.index, group_col]) if group_col in df.columns else pd.Series(['all'] * len(work), index=work.index)

    splitter = make_group_split(groups, n_splits, seed)
    models = {
        'rf': RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        'elasticnet': ElasticNet(alpha=0.02, l1_ratio=0.2, random_state=seed, max_iter=5000),
    }

    metric_rows = []
    perm_rows = []
    pred_rows = []
    best_name = None
    best_score = -1e9

    for model_name, estimator in models.items():
        fold_metrics = []
        fold_perm = []
        fold_preds = []
        for fold, (tr, te) in enumerate(splitter.split(X, y, groups if isinstance(splitter, GroupKFold) else None), start=1):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y.iloc[tr], y.iloc[te]
            pre = make_preprocessor(num_cols, cat_cols)
            pipe = Pipeline([('pre', pre), ('model', estimator)])
            pipe.fit(Xtr, ytr)
            yhat = pipe.predict(Xte)
            rmse = float(np.sqrt(mean_squared_error(yte, yhat)))
            mae = float(mean_absolute_error(yte, yhat))
            r2 = float(r2_score(yte, yhat)) if len(np.unique(yte)) > 1 else float('nan')
            fold_metrics.append({'model': model_name, 'fold': fold, 'target_col': target_col, 'rmse': rmse, 'mae': mae, 'r2': r2})
            fold_preds.append(pd.DataFrame({'model': model_name, 'fold': fold, 'target_col': target_col, 'y_true': yte.values, 'y_pred': yhat}))

            # permutation importance on transformed matrix to avoid column drift issues
            pre_fitted = pipe.named_steps['pre']
            mdl = pipe.named_steps['model']
            Xte_enc = pre_fitted.transform(Xte)
            feat_names = get_feature_names(pre_fitted, Xtr, num_cols, cat_cols)
            n_enc = Xte_enc.shape[1]
            if len(feat_names) < n_enc:
                feat_names = feat_names + [f'feature_{i:04d}' for i in range(len(feat_names), n_enc)]
            elif len(feat_names) > n_enc:
                feat_names = feat_names[:n_enc]
            try:
                perm = permutation_importance(mdl, Xte_enc, yte, n_repeats=10, random_state=seed, scoring='neg_mean_squared_error')
                n_imp = len(perm.importances_mean)
                feats = feat_names[:n_imp] if len(feat_names) >= n_imp else feat_names + [f'feature_{i:04d}' for i in range(len(feat_names), n_imp)]
                fold_perm.append(pd.DataFrame({'model': model_name, 'fold': fold, 'target_col': target_col, 'feature': feats, 'perm_importance': perm.importances_mean[:len(feats)]}))
            except Exception:
                pass
        if not fold_metrics:
            continue
        mdf = pd.DataFrame(fold_metrics)
        metric_rows.append(mdf)
        pred_rows.append(pd.concat(fold_preds, ignore_index=True))
        if fold_perm:
            perm_rows.append(pd.concat(fold_perm, ignore_index=True))
        mean_r2 = pd.to_numeric(mdf['r2'], errors='coerce').mean()
        mean_rmse = pd.to_numeric(mdf['rmse'], errors='coerce').mean()
        score = (mean_r2 if pd.notna(mean_r2) else -1e6) - 0.05 * (mean_rmse if pd.notna(mean_rmse) else 1e6)
        if score > best_score:
            best_score = score
            best_name = model_name

    if not metric_rows:
        return
    metrics = pd.concat(metric_rows, ignore_index=True)
    metrics.to_csv(outdir / f'regression_metrics_{target_col}.csv', index=False)
    preds = pd.concat(pred_rows, ignore_index=True)
    preds.to_csv(outdir / f'regression_predictions_{target_col}.csv', index=False)

    if perm_rows:
        perm_df = pd.concat(perm_rows, ignore_index=True)
        perm_df.to_csv(outdir / f'permutation_importance_raw_{target_col}.csv', index=False)
        top = perm_df.groupby('feature', as_index=False)['perm_importance'].mean().sort_values('perm_importance', ascending=False).head(20)
        top.to_csv(outdir / f'permutation_importance_top_{target_col}.csv', index=False)
        plt.figure(figsize=(8, max(5, 0.35 * len(top))))
        plt.barh(top['feature'][::-1], top['perm_importance'][::-1])
        plt.xlabel('Mean permutation importance'); plt.title(f'Top features for {target_col}')
        plt.tight_layout(); plt.savefig(outdir / f'permutation_importance_top_{target_col}.png', dpi=220); plt.close()

    # parity plot for best model
    if best_name is not None:
        sub = preds[preds['model'] == best_name].copy()
        if not sub.empty:
            plt.figure(figsize=(6, 6))
            plt.scatter(sub['y_true'], sub['y_pred'], s=18)
            lo = min(sub['y_true'].min(), sub['y_pred'].min())
            hi = max(sub['y_true'].max(), sub['y_pred'].max())
            plt.plot([lo, hi], [lo, hi])
            plt.xlabel('Observed'); plt.ylabel('Predicted'); plt.title(f'{target_col} parity ({best_name})')
            plt.tight_layout(); plt.savefig(outdir / f'parity_{target_col}_{best_name}.png', dpi=220); plt.close()


def run_classification(df: pd.DataFrame, target_name: str, y: pd.Series, group_col: str, outdir: Path, n_splits: int, seed: int) -> None:
    y = y.astype(int)
    if y.nunique() < 2 or len(y) < 12:
        return
    X = df.loc[y.index].copy()
    feat_cols = [c for c in X.columns if c not in DROP_COLS and c not in TARGETS_REG and not c.startswith('qc_') and c != target_name]
    X = X[feat_cols].copy()
    X = drop_empty_feature_columns(X)
    if X.empty:
        return
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    groups = clean_cat(df.loc[y.index, group_col]) if group_col in df.columns else pd.Series(['all'] * len(y), index=y.index)
    splitter = make_group_split(groups, n_splits, seed)

    models = {
        'rf_clf': RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        'logreg': LogisticRegression(max_iter=5000),
    }
    rows = []
    for name, estimator in models.items():
        for fold, (tr, te) in enumerate(splitter.split(X, y, groups if isinstance(splitter, GroupKFold) else None), start=1):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y.iloc[tr], y.iloc[te]
            if ytr.nunique() < 2:
                continue
            pipe = Pipeline([('pre', make_preprocessor(num_cols, cat_cols)), ('model', estimator)])
            pipe.fit(Xtr, ytr)
            yhat = pipe.predict(Xte)
            rec = {'model': name, 'fold': fold, 'target_col': target_name, 'accuracy': float(accuracy_score(yte, yhat)), 'f1': float(f1_score(yte, yhat))}
            try:
                proba = pipe.predict_proba(Xte)[:, 1]
                rec['roc_auc'] = float(roc_auc_score(yte, proba))
            except Exception:
                rec['roc_auc'] = np.nan
            rows.append(rec)
    if rows:
        pd.DataFrame(rows).to_csv(outdir / f'classification_metrics_{target_name}.csv', index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--group-col', default='paper_id')
    ap.add_argument('--n-splits', type=int, default=5)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--qc-filter', action='store_true')
    args = ap.parse_args()

    outdir = Path(args.outdir); ensure_dir(outdir)
    df = pd.read_csv(args.csv)
    if args.qc_filter and 'qc_keep_for_model' in df.columns:
        df = df[to_num(df['qc_keep_for_model']).fillna(0).astype(int) == 1].copy()

    summary: Dict[str, Dict[str, float]] = {}
    for t in TARGETS_REG:
        if t in df.columns:
            run_regression(df, t, args.group_col, outdir, args.n_splits, args.seed)
    # derived classifiers
    dgh_col = 'H_adsorption_free_energy_value_eV' if 'H_adsorption_free_energy_value_eV' in df.columns else 'H_adsorption_free_energy_value'
    if dgh_col in df.columns:
        vals = to_num(df[dgh_col])
        y = (vals.abs() <= 0.2)
        run_classification(df, 'near_thermoneutral_dG_H', y[vals.notna()], args.group_col, outdir, args.n_splits, args.seed)
    deh_col = 'H_adsorption_energy_value_eV' if 'H_adsorption_energy_value_eV' in df.columns else 'H_adsorption_energy_value'
    if deh_col in df.columns:
        vals2 = to_num(df[deh_col])
        y2 = (vals2.abs() <= 0.3)
        run_classification(df, 'near_balanced_dE_H', y2[vals2.notna()], args.group_col, outdir, args.n_splits, args.seed)

    summary['done'] = {'n_rows': int(len(df))}
    (outdir / 'summary_predictive.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[DONE] H-ads predictive modeling ->', outdir)


if __name__ == '__main__':
    main()
