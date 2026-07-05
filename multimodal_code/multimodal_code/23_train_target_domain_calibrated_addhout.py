#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast target-domain calibrated AddH-out prediction.

Purpose:
  - train only on addH/addH-2 labels;
  - reuse the element/LLM feature tables from stage 3;
  - avoid slow FAIR-Chem/multiview retraining;
  - produce conservative AddH-out predictions plus auditable alternatives.

Default final prediction is strict-blind with respect to AddH-out labels:
  - labels are never used for training, residual-model selection, or final
    material weights;
  - --audit-labels-csv only writes post-hoc audit files.

The strongest baseline in this project is the source dopant mean prior. This
script therefore trains residual corrections around that prior, and only lets
OOF-improving residual models change the prediction. It can also blend the
source prior with a recentered strict graph/multiview prediction using fixed
material weights. Those weights are explicit CLI parameters, not tuned from
AddH-out labels.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


NON_FEATURE_COLS = {
    "split_role",
    "data_source",
    "id",
    "family_base",
    "material",
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
    "target",
    "energy_bare",
    "energy_addH",
    "energy_total_excel",
    "energy_slab_excel",
    "h_ads_excel",
    "target_computed",
}

STRICT_PRED_CANDIDATES = [
    "pred_strict_blind_final",
    "pred_strict_blind_strategy_weighted",
    "pred_strict_blind_strategy_median",
    "pred_strict_blind",
    "pred_strict_blind_weighted",
    "pred_strict_blind_median",
    "pred",
    "pred_median",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fast target-domain calibrated AddH/AddH-2 -> AddH-out predictor."
    )
    ap.add_argument("--feature-dir", default="outputs_addh_llm_element_priors")
    ap.add_argument("--train-features", default="")
    ap.add_argument("--addhout-features", default="")
    ap.add_argument(
        "--llm-pred-csv",
        default="outputs_addh_llm_element_knowledge_blend_scnet_deepseek_v4_pro/knowledge_enhanced_addhout_predictions.csv",
    )
    ap.add_argument(
        "--strict-pred-csv",
        default="outputs_addh_strict_blind_final/strict_blind_strategy_ensemble_predictions.csv",
    )
    ap.add_argument("--out-dir", default="outputs_addh_target_calibrated_fast")
    ap.add_argument("--target-col", default="target")
    ap.add_argument("--group-col", default="family_base")
    ap.add_argument("--target-abs-max", type=float, default=10.0)
    ap.add_argument("--n-splits", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-residual-correction", type=float, default=1.25)
    ap.add_argument(
        "--min-residual-oof-improvement",
        type=float,
        default=0.03,
        help="Residual models must beat dopant-prior OOF MAE by at least this many eV.",
    )
    ap.add_argument(
        "--residual-shrink-grid",
        default="0,0.05,0.10,0.15,0.20,0.30,0.50,0.75,1.00",
        help="OOF-selected shrink factors for residual corrections.",
    )
    ap.add_argument(
        "--material-strict-weights",
        default="CeO2=0.20,ZnO=0.90",
        help="Fixed label-free blend weights for strict recentered prediction by material.",
    )
    ap.add_argument("--default-strict-weight", type=float, default=0.0)
    ap.add_argument(
        "--strict-recenter-anchor",
        choices=["prior_median", "prior_mean", "llm_median", "source_mean"],
        default="prior_median",
    )
    ap.add_argument(
        "--final-mode",
        choices=["auto", "prior", "residual_guarded", "material_strict"],
        default="auto",
        help="auto uses material_strict when strict predictions exist, otherwise residual_guarded.",
    )
    ap.add_argument("--clip-final-to-source-range", action="store_true")
    ap.add_argument("--audit-labels-csv", default="auto")
    ap.add_argument("--audit-target-col", default="h_ads_excel")
    ap.add_argument(
        "--oracle-diagnostic-tune",
        action="store_true",
        help="Post-hoc diagnostic only: tune material strict weights using audit labels.",
    )
    return ap.parse_args()


def parse_float_list(raw: str) -> List[float]:
    vals: List[float] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return sorted(set(vals))


def parse_weight_map(raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def finite_corr(x: Sequence[float], y: Sequence[float], spearman: bool = False) -> float:
    a = pd.to_numeric(pd.Series(x), errors="coerce")
    b = pd.to_numeric(pd.Series(y), errors="coerce")
    m = a.notna() & b.notna()
    if int(m.sum()) < 3:
        return float("nan")
    av = a[m].to_numpy(float)
    bv = b[m].to_numpy(float)
    if spearman:
        av = pd.Series(av).rank(method="average").to_numpy(float)
        bv = pd.Series(bv).rank(method="average").to_numpy(float)
    if np.nanstd(av) <= 1e-12 or np.nanstd(bv) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y = pd.to_numeric(pd.Series(y_true), errors="coerce")
    p = pd.to_numeric(pd.Series(y_pred), errors="coerce")
    m = y.notna() & p.notna()
    if int(m.sum()) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "pearson": np.nan, "spearman": np.nan}
    yy = y[m].to_numpy(float)
    pp = p[m].to_numpy(float)
    err = pp - yy
    return {
        "n": int(len(yy)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "bias": float(np.mean(err)),
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
            row: Dict[str, object] = {
                "pred_col": c,
                "target_col": target_col,
                "material": mat,
            }
            row.update(metrics(d[target_col], d[c]))
            rows.append(row)
    return pd.DataFrame(rows)


def numeric_feature_columns(train: pd.DataFrame, addhout: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in train.columns:
        if c in NON_FEATURE_COLS:
            continue
        if c not in addhout.columns:
            continue
        if c.startswith("source_"):
            # These are full-source statistics; using them as residual features
            # makes OOF checks less honest. They are used separately as priors.
            continue
        if c.startswith("pred_"):
            continue
        s1 = pd.to_numeric(train[c], errors="coerce")
        s2 = pd.to_numeric(addhout[c], errors="coerce")
        if s1.notna().any() and s2.notna().any():
            cols.append(c)
    return sorted(cols)


def make_group_folds(groups: Sequence[object], n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups_arr = np.asarray([str(g) for g in groups])
    unique = np.array(sorted(pd.unique(groups_arr)))
    n = max(2, min(int(n_splits), len(unique)))
    try:
        from sklearn.model_selection import GroupKFold

        return [
            (np.asarray(tr, dtype=int), np.asarray(va, dtype=int))
            for tr, va in GroupKFold(n_splits=n).split(np.zeros(len(groups_arr)), groups=groups_arr)
        ]
    except Exception:
        rng = np.random.default_rng(seed)
        rng.shuffle(unique)
        buckets = [[] for _ in range(n)]
        for i, g in enumerate(unique):
            buckets[i % n].append(g)
        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        all_idx = np.arange(len(groups_arr))
        for bucket in buckets:
            val_mask = np.isin(groups_arr, bucket)
            if val_mask.any() and (~val_mask).any():
                folds.append((all_idx[~val_mask], all_idx[val_mask]))
        return folds


def dopant_prior_from(train_df: pd.DataFrame, apply_df: pd.DataFrame, mode: str = "mean") -> np.ndarray:
    y = pd.to_numeric(train_df["target"], errors="coerce")
    base = train_df.copy()
    base["target"] = y
    base = base[base["target"].notna()]
    if mode == "median":
        agg = base.groupby("dopant")["target"].median()
        fallback = float(base["target"].median())
    else:
        agg = base.groupby("dopant")["target"].mean()
        fallback = float(base["target"].mean())
    return apply_df["dopant"].map(agg).fillna(fallback).to_numpy(float)


def build_models(seed: int):
    try:
        from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet, Ridge
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise SystemExit(
            "[ERROR] scikit-learn is required for 23_train_target_domain_calibrated_addhout.py "
            f"in this fast route: {exc}"
        )

    models = []
    for alpha in [3.0, 10.0, 30.0, 100.0, 300.0]:
        models.append((
            f"ridge_a{alpha:g}",
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha)),
        ))
    for alpha in [0.003, 0.01, 0.03, 0.10]:
        models.append((
            f"elastic_a{alpha:g}",
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                ElasticNet(alpha=alpha, l1_ratio=0.20, max_iter=5000, random_state=seed),
            ),
        ))
    for k in [9, 15, 25, 35]:
        models.append((
            f"knn_k{k}",
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                KNeighborsRegressor(n_neighbors=k, weights="distance"),
            ),
        ))
    models.extend([
        (
            "extratrees",
            ExtraTreesRegressor(
                n_estimators=220,
                random_state=seed,
                min_samples_leaf=3,
                max_features=0.70,
                n_jobs=-1,
            ),
        ),
        (
            "randomforest",
            RandomForestRegressor(
                n_estimators=260,
                random_state=seed,
                min_samples_leaf=3,
                max_features=0.70,
                n_jobs=-1,
            ),
        ),
        (
            "hist_gbdt",
            make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingRegressor(
                    max_iter=220,
                    learning_rate=0.03,
                    l2_regularization=1.0,
                    max_leaf_nodes=8,
                    random_state=seed,
                ),
            ),
        ),
        (
            "gradboost",
            make_pipeline(
                SimpleImputer(strategy="median"),
                GradientBoostingRegressor(
                    n_estimators=180,
                    learning_rate=0.03,
                    max_depth=2,
                    random_state=seed,
                ),
            ),
        ),
    ])
    return models


def residual_prior_oof(train: pd.DataFrame, addhout: pd.DataFrame, folds: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    prior_oof = np.full(len(train), np.nan, dtype=float)
    out_fold = []
    for tr_idx, va_idx in folds:
        tr = train.iloc[tr_idx]
        va = train.iloc[va_idx]
        prior_oof[va_idx] = dopant_prior_from(tr, va, "mean")
        out_fold.append(dopant_prior_from(tr, addhout, "mean"))
    prior_out = dopant_prior_from(train, addhout, "mean")
    if out_fold:
        # Keep full-source prior as the main anchor; fold-mean only checks stability.
        fold_mean = np.nanmean(np.vstack(out_fold), axis=0)
        prior_out = 0.90 * prior_out + 0.10 * fold_mean
    return prior_oof, prior_out


def train_residual_models(
    train: pd.DataFrame,
    addhout: pd.DataFrame,
    feature_cols: List[str],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    max_correction: float,
    shrink_grid: List[float],
    min_improvement: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    X = train[feature_cols].apply(pd.to_numeric, errors="coerce")
    Xout = addhout[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = train["target"].to_numpy(float)

    prior_oof, prior_out = residual_prior_oof(train, addhout, folds)
    prior_met = metrics(y, prior_oof)
    prior_mae = float(prior_met["mae"])

    oof_table = train[["id", "family_base", "material", "dopant", "target"]].copy()
    oof_table["oof_dopant_prior"] = prior_oof
    model_rows: List[Dict[str, object]] = []
    selected_parts: List[Tuple[str, float, np.ndarray, float]] = []

    for name, model in build_models(seed):
        raw_resid_oof = np.full(len(train), np.nan, dtype=float)
        out_resid_folds = []
        failed = False
        for tr_idx, va_idx in folds:
            try:
                tr = train.iloc[tr_idx]
                va = train.iloc[va_idx]
                prior_tr = dopant_prior_from(tr, tr, "mean")
                prior_va = dopant_prior_from(tr, va, "mean")
                resid_tr = y[tr_idx] - prior_tr
                model.fit(X.iloc[tr_idx], resid_tr)
                pred_va = np.clip(model.predict(X.iloc[va_idx]), -max_correction, max_correction)
                pred_out = np.clip(model.predict(Xout), -max_correction, max_correction)
                raw_resid_oof[va_idx] = pred_va
                out_resid_folds.append(pred_out)
                # Consistency check: prior_va and prior_oof should match.
                if not np.allclose(prior_va, prior_oof[va_idx], equal_nan=True):
                    warnings.warn("fold prior mismatch; continuing with cached OOF prior")
            except Exception as exc:
                print(f"[WARN] residual model failed {name}: {exc}")
                failed = True
                break
        if failed or not np.isfinite(raw_resid_oof).any():
            continue

        shrink_scores = []
        for shrink in shrink_grid:
            pred = prior_oof + shrink * raw_resid_oof
            met = metrics(y, pred)
            shrink_scores.append((shrink, float(met["mae"]), met))
        best_shrink, best_mae, best_met = sorted(shrink_scores, key=lambda x: (x[1], x[0]))[0]

        try:
            full_prior_train = dopant_prior_from(train, train, "mean")
            full_resid = y - full_prior_train
            model.fit(X, full_resid)
            full_out_resid = np.clip(model.predict(Xout), -max_correction, max_correction)
            cv_out_resid = np.nanmean(np.vstack(out_resid_folds), axis=0)
            out_resid = 0.70 * full_out_resid + 0.30 * cv_out_resid
        except Exception as exc:
            print(f"[WARN] final fit failed {name}: {exc}")
            continue

        selected = bool(best_mae <= prior_mae - min_improvement and best_shrink > 0)
        row: Dict[str, object] = {
            "model": name,
            "selected": selected,
            "best_shrink": best_shrink,
            "prior_oof_mae": prior_mae,
            "oof_mae": best_mae,
            "oof_improvement_vs_prior": prior_mae - best_mae,
        }
        row.update({f"oof_{k}": v for k, v in best_met.items()})
        model_rows.append(row)
        oof_table[f"oof_raw_resid__{name}"] = raw_resid_oof
        oof_table[f"oof_pred__{name}"] = prior_oof + best_shrink * raw_resid_oof
        if selected:
            weight = max(prior_mae - best_mae, 1e-6) / max(best_mae, 1e-6) ** 2
            selected_parts.append((name, weight, best_shrink * out_resid, best_shrink))
        print(
            f"[MODEL] {name:14s} selected={int(selected)} "
            f"OOF MAE={best_mae:.4f} prior={prior_mae:.4f} shrink={best_shrink:.2f}"
        )

    metrics_df = pd.DataFrame(model_rows).sort_values(["selected", "oof_mae"], ascending=[False, True])
    if selected_parts:
        w = np.asarray([p[1] for p in selected_parts], dtype=float)
        w = w / np.sum(w)
        resid = np.zeros(len(addhout), dtype=float)
        for wi, (_, _, out_resid, _) in zip(w, selected_parts):
            resid += wi * out_resid
        selected_df = pd.DataFrame(
            [
                {
                    "model": name,
                    "ensemble_weight": float(wi),
                    "best_shrink": float(shrink),
                }
                for wi, (name, _, _, shrink) in zip(w, selected_parts)
            ]
        )
    else:
        resid = np.zeros(len(addhout), dtype=float)
        selected_df = pd.DataFrame(columns=["model", "ensemble_weight", "best_shrink"])
    residual_guarded = prior_out + resid
    return metrics_df, selected_df, prior_oof, prior_out, oof_table.assign(oof_residual_guarded=prior_oof)


def detect_strict_pred_col(df: pd.DataFrame) -> Optional[str]:
    for c in STRICT_PRED_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = str(c).lower()
        if "pred" in lc and not any(tok in lc for tok in ["std", "rank", "err", "min", "max", "n_"]):
            if pd.to_numeric(df[c], errors="coerce").notna().any():
                return c
    return None


def merge_optional_predictions(
    out: pd.DataFrame,
    llm_path: Path,
    strict_path: Path,
    prior_out: np.ndarray,
    source_mean: float,
    strict_anchor: str,
) -> pd.DataFrame:
    pred = out.copy()
    pred["pred_dopant_prior"] = prior_out

    if llm_path.exists():
        llm = pd.read_csv(llm_path)
        if "id" in llm.columns:
            keep = [
                c
                for c in [
                    "id",
                    "pred_llm_element_knowledge_blend",
                    "pred_knowledge_model",
                    "pred_source_dopant_mean_prior",
                    "pred_base_pool_raw",
                    "pred_base_pool_recenter_to_knowledge",
                ]
                if c in llm.columns
            ]
            pred = pred.merge(llm[keep].drop_duplicates("id"), on="id", how="left")
    if "pred_llm_element_knowledge_blend" not in pred.columns:
        pred["pred_llm_element_knowledge_blend"] = np.nan

    pred["pred_strict_raw_fast"] = np.nan
    pred["pred_strict_recenter_fast"] = np.nan
    pred["strict_fast_pred_col"] = ""
    pred["strict_fast_raw_mean_shift_vs_source"] = np.nan
    if strict_path.exists():
        strict = pd.read_csv(strict_path)
        if "id" in strict.columns:
            pc = detect_strict_pred_col(strict)
            if pc:
                small = strict[["id", pc]].copy().rename(columns={pc: "pred_strict_raw_fast"})
                small["pred_strict_raw_fast"] = pd.to_numeric(small["pred_strict_raw_fast"], errors="coerce")
                pred = pred.drop(columns=["pred_strict_raw_fast"], errors="ignore").merge(
                    small.drop_duplicates("id"), on="id", how="left"
                )
                raw = pd.to_numeric(pred["pred_strict_raw_fast"], errors="coerce")
                prior = pd.to_numeric(pred["pred_dopant_prior"], errors="coerce")
                llm = pd.to_numeric(pred["pred_llm_element_knowledge_blend"], errors="coerce")
                if strict_anchor == "prior_mean":
                    anchor = float(prior.mean())
                elif strict_anchor == "llm_median" and llm.notna().any():
                    anchor = float(llm.median())
                elif strict_anchor == "source_mean":
                    anchor = float(source_mean)
                else:
                    anchor = float(prior.median())
                raw_med = float(raw.median()) if raw.notna().any() else float("nan")
                if np.isfinite(raw_med) and np.isfinite(anchor):
                    pred["pred_strict_recenter_fast"] = raw - raw_med + anchor
                pred["strict_fast_pred_col"] = pc
                mean_shift = abs(float(raw.mean()) - source_mean) if raw.notna().any() else float("nan")
                pred["strict_fast_raw_mean_shift_vs_source"] = mean_shift
    return pred


def material_strict_prediction(
    pred: pd.DataFrame,
    weights: Dict[str, float],
    default_weight: float,
) -> np.ndarray:
    prior = pd.to_numeric(pred["pred_dopant_prior"], errors="coerce").to_numpy(float)
    strict = pd.to_numeric(pred["pred_strict_recenter_fast"], errors="coerce").to_numpy(float)
    mats = pred["material"].astype(str).to_numpy() if "material" in pred.columns else np.array([""] * len(pred))
    out = prior.copy()
    for i, mat in enumerate(mats):
        if not np.isfinite(strict[i]):
            continue
        w = weights.get(mat, default_weight)
        w = min(max(float(w), 0.0), 1.0)
        out[i] = (1.0 - w) * prior[i] + w * strict[i]
    return out


def audit_and_optional_oracle(
    pred: pd.DataFrame,
    labels_path: Path,
    target_col: str,
    out_dir: Path,
    oracle: bool,
) -> pd.DataFrame:
    if not labels_path.exists():
        return pd.DataFrame()
    labels = pd.read_csv(labels_path)
    if target_col not in labels.columns or "id" not in labels.columns:
        print(f"[WARN] audit labels missing id/{target_col}: {labels_path}")
        return pd.DataFrame()
    keep = [c for c in ["id", target_col, "material", "dopant"] if c in labels.columns]
    detail = pred.merge(labels[keep].drop_duplicates("id"), on="id", how="left", suffixes=("", "_audit"))

    if oracle:
        best = (float("inf"), 0.0, 0.0)
        ce_mask = detail["material"].astype(str).eq("CeO2")
        zn_mask = detail["material"].astype(str).eq("ZnO")
        prior = pd.to_numeric(detail["pred_dopant_prior"], errors="coerce")
        strict = pd.to_numeric(detail["pred_strict_recenter_fast"], errors="coerce")
        for w_ce in np.linspace(0, 1, 21):
            for w_zn in np.linspace(0, 1, 21):
                p = prior.copy()
                m = ce_mask & strict.notna()
                p.loc[m] = (1 - w_ce) * prior.loc[m] + w_ce * strict.loc[m]
                m = zn_mask & strict.notna()
                p.loc[m] = (1 - w_zn) * prior.loc[m] + w_zn * strict.loc[m]
                val = metrics(detail[target_col], p)["mae"]
                if np.isfinite(val) and val < best[0]:
                    best = (float(val), float(w_ce), float(w_zn))
        p = prior.copy()
        m = ce_mask & strict.notna()
        p.loc[m] = (1 - best[1]) * prior.loc[m] + best[1] * strict.loc[m]
        m = zn_mask & strict.notna()
        p.loc[m] = (1 - best[2]) * prior.loc[m] + best[2] * strict.loc[m]
        detail["pred_oracle_material_strict_diagnostic"] = p
        with (out_dir / "oracle_material_strict_diagnostic.json").open("w", encoding="utf-8") as f:
            json.dump({"mae": best[0], "CeO2_weight": best[1], "ZnO_weight": best[2]}, f, indent=2)

    pred_cols = [
        c
        for c in [
            "pred_fast_target_calibrated",
            "pred_material_strict_heuristic",
            "pred_residual_guarded",
            "pred_dopant_prior",
            "pred_llm_element_knowledge_blend",
            "pred_strict_recenter_fast",
            "pred_strict_raw_fast",
            "pred_oracle_material_strict_diagnostic",
        ]
        if c in detail.columns
    ]
    audit = metric_rows(detail, pred_cols, target_col)
    audit.to_csv(out_dir / "target_calibrated_posthoc_audit.csv", index=False)
    if "pred_fast_target_calibrated" in detail.columns:
        detail["err_fast_target_calibrated"] = (
            pd.to_numeric(detail["pred_fast_target_calibrated"], errors="coerce")
            - pd.to_numeric(detail[target_col], errors="coerce")
        )
        detail["abs_err_fast_target_calibrated"] = detail["err_fast_target_calibrated"].abs()
        detail = detail.sort_values("abs_err_fast_target_calibrated", ascending=False)
    detail.to_csv(out_dir / "target_calibrated_posthoc_audit_detail.csv", index=False)
    try:
        audit.to_excel(out_dir / "target_calibrated_posthoc_audit.xlsx", index=False)
    except Exception:
        pass
    return audit


def main() -> None:
    args = parse_args()
    feature_dir = Path(args.feature_dir)
    train_path = Path(args.train_features) if args.train_features else feature_dir / "knowledge_features_train.csv"
    addhout_path = Path(args.addhout_features) if args.addhout_features else feature_dir / "knowledge_features_addhout.csv"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_all = pd.read_csv(train_path)
    addhout = pd.read_csv(addhout_path).reset_index(drop=True)
    if args.target_col not in train_all.columns:
        raise SystemExit(f"[ERROR] target column not found in train features: {args.target_col}")
    train_all["target"] = pd.to_numeric(train_all[args.target_col], errors="coerce")
    usable = train_all["target"].notna() & (train_all["target"].abs() <= args.target_abs_max)
    train = train_all.loc[usable].reset_index(drop=True).copy()
    if len(train) < 20:
        raise SystemExit("[ERROR] too few usable training rows after filtering.")

    feature_cols = numeric_feature_columns(train, addhout)
    if not feature_cols:
        raise SystemExit("[ERROR] no shared numeric feature columns found.")
    groups = train[args.group_col].fillna(train["id"]).astype(str).to_numpy()
    folds = make_group_folds(groups, args.n_splits, args.seed)
    if not folds:
        raise SystemExit("[ERROR] no valid grouped folds.")

    print(f"[INFO] train usable rows={len(train)} raw={len(train_all)}")
    print(f"[INFO] addH-out rows={len(addhout)}")
    print(f"[INFO] features={len(feature_cols)} folds={len(folds)} groups={pd.Series(groups).nunique()}")

    shrink_grid = parse_float_list(args.residual_shrink_grid)
    residual_metrics, selected_residual, prior_oof, prior_out, oof_table = train_residual_models(
        train=train,
        addhout=addhout,
        feature_cols=feature_cols,
        folds=folds,
        seed=args.seed,
        max_correction=args.max_residual_correction,
        shrink_grid=shrink_grid,
        min_improvement=args.min_residual_oof_improvement,
    )
    residual_metrics.to_csv(out_dir / "target_calibrated_residual_oof_metrics.csv", index=False)
    selected_residual.to_csv(out_dir / "target_calibrated_selected_residual_models.csv", index=False)
    oof_table.to_csv(out_dir / "target_calibrated_oof_predictions.csv", index=False)

    pred = merge_optional_predictions(
        addhout,
        Path(args.llm_pred_csv),
        Path(args.strict_pred_csv),
        prior_out,
        float(train["target"].mean()),
        args.strict_recenter_anchor,
    )
    # Recompute guarded residual output from selected model metadata. If no
    # residual model passed OOF protection, this intentionally equals the prior.
    if len(selected_residual):
        # Rebuild a stable residual ensemble by using the already selected model
        # names through another training pass. This avoids serializing sklearn
        # estimators and keeps the script single-file.
        selected_names = set(selected_residual["model"].astype(str))
        X = train[feature_cols].apply(pd.to_numeric, errors="coerce")
        Xout = addhout[feature_cols].apply(pd.to_numeric, errors="coerce")
        y = train["target"].to_numpy(float)
        full_prior_train = dopant_prior_from(train, train, "mean")
        residual_parts = []
        for name, model in build_models(args.seed):
            if name not in selected_names:
                continue
            shrink = float(selected_residual.loc[selected_residual["model"].astype(str) == name, "best_shrink"].iloc[0])
            weight = float(selected_residual.loc[selected_residual["model"].astype(str) == name, "ensemble_weight"].iloc[0])
            model.fit(X, y - full_prior_train)
            resid = np.clip(model.predict(Xout), -args.max_residual_correction, args.max_residual_correction)
            residual_parts.append((weight, shrink * resid))
        if residual_parts:
            total = sum(w for w, _ in residual_parts)
            resid = sum((w / total) * r for w, r in residual_parts)
            pred["pred_residual_guarded"] = pred["pred_dopant_prior"] + resid
        else:
            pred["pred_residual_guarded"] = pred["pred_dopant_prior"]
    else:
        pred["pred_residual_guarded"] = pred["pred_dopant_prior"]

    material_weights = parse_weight_map(args.material_strict_weights)
    pred["pred_material_strict_heuristic"] = material_strict_prediction(
        pred, material_weights, args.default_strict_weight
    )

    final_mode = args.final_mode
    strict_has_values = pd.to_numeric(pred["pred_strict_recenter_fast"], errors="coerce").notna().any()
    if final_mode == "auto":
        final_mode = "material_strict" if strict_has_values else "residual_guarded"
    if final_mode == "prior":
        final = pd.to_numeric(pred["pred_dopant_prior"], errors="coerce").to_numpy(float)
    elif final_mode == "residual_guarded":
        final = pd.to_numeric(pred["pred_residual_guarded"], errors="coerce").to_numpy(float)
    else:
        final = pd.to_numeric(pred["pred_material_strict_heuristic"], errors="coerce").to_numpy(float)

    if args.clip_final_to_source_range:
        final = np.clip(final, float(train["target"].min()), float(train["target"].max()))
    pred["pred_fast_target_calibrated"] = final
    pred["fast_target_calibrated_rank"] = pd.Series(final).rank(method="average", ascending=True).to_numpy()
    pred["fast_final_mode"] = final_mode
    pred["fast_material_strict_weights"] = json.dumps(material_weights, sort_keys=True)
    pred = pred.sort_values(["fast_target_calibrated_rank", "id"], na_position="last").reset_index(drop=True)

    out_csv = out_dir / "target_calibrated_addhout_predictions.csv"
    pred.to_csv(out_csv, index=False)
    try:
        pred.to_excel(out_dir / "target_calibrated_addhout_predictions.xlsx", index=False)
    except Exception:
        pass

    audit_labels: Optional[Path] = None
    if args.audit_labels_csv == "auto":
        candidate = feature_dir / "addhout_audit_labels.csv"
        if candidate.exists():
            audit_labels = candidate
    elif args.audit_labels_csv:
        audit_labels = Path(args.audit_labels_csv)
    audit = pd.DataFrame()
    if audit_labels is not None and audit_labels.exists():
        audit = audit_and_optional_oracle(
            pred=pred,
            labels_path=audit_labels,
            target_col=args.audit_target_col,
            out_dir=out_dir,
            oracle=args.oracle_diagnostic_tune,
        )
        if len(audit):
            print("[POSTHOC AUDIT ONLY]")
            print(audit.to_string(index=False))

    manifest = {
        "train_features": str(train_path),
        "addhout_features": str(addhout_path),
        "llm_pred_csv": str(args.llm_pred_csv),
        "strict_pred_csv": str(args.strict_pred_csv),
        "out_dir": str(out_dir),
        "labels_used_for_training_or_selection": False,
        "final_mode": final_mode,
        "material_strict_weights": material_weights,
        "default_strict_weight": args.default_strict_weight,
        "strict_recenter_anchor": args.strict_recenter_anchor,
        "n_train_usable": int(len(train)),
        "n_addhout": int(len(addhout)),
        "n_features": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "selected_residual_models": selected_residual.to_dict(orient="records"),
        "outputs": {
            "predictions_csv": str(out_csv),
            "audit_csv": str(out_dir / "target_calibrated_posthoc_audit.csv"),
        },
    }
    with (out_dir / "target_calibrated_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] wrote {out_csv}")
    print(f"[INFO] final_mode={final_mode}")
    print(f"[INFO] selected_residual_models={len(selected_residual)}")


if __name__ == "__main__":
    main()
