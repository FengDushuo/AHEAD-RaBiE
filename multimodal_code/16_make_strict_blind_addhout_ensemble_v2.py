#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strict blind AddH-out ensemble builder v2.

Compared with v1, this version adds:
  1) family-diverse / conservative model selection so full/delta models cannot dominate;
  2) robust prediction-column detection: pred, pred_median, pred_mean, pred_ensemble, prediction, y_pred;
  3) support for multi-view outputs in outputs_addh_modelgrid_v2 / outputs_addh_modelgrid_v2_full;
  4) optional model-family caps and required feature-mode quotas;
  5) separate selected_models_all and selected_models_diverse outputs.

Strict-blind rule:
  AddH-out labels are NEVER used for selection, weights, calibration, or final prediction.
  Labels are only included if --audit-with-labels is set, and then only for post-hoc audit metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except Exception:  # pragma: no cover
    mean_absolute_error = mean_squared_error = r2_score = None


PRED_FILENAMES = [
    "addH_out_graph_ensemble_by_id_merged_master.csv",
    "addH_out_graph_ensemble_by_id.csv",
    "addH_out_pred_ensemble_robust_final_by_id.csv",
    "addH_out_pred_ensemble_robust_final_by_id_clip10train.csv",
    "addH_out_pred_ensemble_robust.csv",
    "addH_out_pred_ensemble.csv",
    "addH_out_final_pred.csv",
    "addhout_pred.csv",
]

OOF_FILENAMES = [
    "test_oof_graph_ensemble_metrics.json",
    "oof_graph_ensemble_metrics.json",
    "metrics.json",
    "cv_metrics.json",
]

PRED_COL_CANDIDATES = [
    "pred", "pred_median", "pred_mean", "pred_ensemble", "prediction", "y_pred", "pred_cal", "pred_raw"
]

LABEL_COLS = {
    "h_ads_excel", "target", "target_computed",
    "abs_err_vs_target", "abs_err_vs_h_ads_excel", "abs_err_vs_target_computed",
}


def parse_key_value_map(s: str, value_type=float) -> Dict[str, float]:
    out = {}
    if not s:
        return out
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            continue
        k, v = part.split('=', 1)
        k = k.strip()
        try:
            out[k] = value_type(v.strip())
        except Exception:
            pass
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Make strict blind AddH-out ensemble from existing prediction runs, with family-diverse selection.")
    ap.add_argument("--roots", nargs="+", default=[
        "outputs_addh_graph_ensemble_refine_v2",
        "outputs_addh_graph_ensemble",
        "outputs_addh_modelgrid_v2",
        "outputs_addh_modelgrid_v2_full",
    ])
    ap.add_argument("--addhout-master-csv", default="outputs_addh_full_mm_envsplit/addH_out_master_normalized.csv")
    ap.add_argument("--train-master-csv", default="outputs_addh_full_mm_envsplit/addH_master_target_weighted_mild.csv")
    ap.add_argument("--out-dir", default="outputs_addh_strict_blind_diverse")
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--min-coverage", type=float, default=0.90)
    ap.add_argument("--min-n-addhout", type=int, default=10)
    ap.add_argument("--include-exp-regex", default="")
    ap.add_argument("--exclude-exp-regex", default="")
    ap.add_argument("--score-mode", default="conservative", choices=["balanced", "oof_first", "stability_first", "conservative"])
    ap.add_argument("--weight-mode", default="soft_inverse_rmse", choices=["soft_inverse_rmse", "rank", "uniform"])
    ap.add_argument("--target-col", default="target")
    ap.add_argument("--clip-to-source-range", action="store_true")
    ap.add_argument("--audit-with-labels", action="store_true")

    # family-diverse / conservative options
    ap.add_argument("--family-diverse", action="store_true", help="Use feature-mode and model-family caps instead of naive top-k.")
    ap.add_argument("--feature-mode-cap", type=int, default=3, help="Default max models per feature_mode when --family-diverse is used.")
    ap.add_argument("--feature-mode-cap-map", default="full=3,bare_delta=2,addh_delta=2,delta=1,addh=3,addh_bare=3,bare=1,graph_only=2,concat_interact=2,gated_sum=1,residual_graph=1,text_only=0",
                    help="Comma key=value caps by feature_mode/model_variant.")
    ap.add_argument("--model-family-cap-map", default="graph_ensemble=12,multiview=4,unknown=2",
                    help="Comma key=value caps by model_family.")
    ap.add_argument("--require-feature-modes", default="addh_bare=1,addh=1,full=1",
                    help="Comma key=value minimum quotas by feature_mode if eligible models exist. Set empty to disable.")
    ap.add_argument("--conservative-mode-penalty", default="full=0.08,bare_delta=0.06,addh_delta=0.05,delta=0.10,text_only=0.25,addh=0.00,addh_bare=0.00,bare=0.03,graph_only=0.03,concat_interact=0.05,gated_sum=0.06,residual_graph=0.06",
                    help="Soft score penalty by feature_mode/model_variant; does not use addH-out labels.")
    ap.add_argument("--prefer-feature-modes", default="addh_bare,addh,graph_only,full",
                    help="Comma list that gets a tiny preference bonus; not a hard filter.")
    ap.add_argument("--strict-output-no-labels", action="store_true", help="Force output to exclude labels even if master has them; default true unless audit is set.")
    return ap.parse_args()


def safe_json_load(p: Path) -> Dict:
    try:
        with p.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def find_pred_path(exp_dir: Path) -> Optional[Path]:
    for sub in [exp_dir / "work", exp_dir / "cv_work", exp_dir / "final_work", exp_dir / "final_pred", exp_dir]:
        for name in PRED_FILENAMES:
            p = sub / name
            if p.exists():
                return p
    # broader fallback: avoid reading giant unrelated files; only exact-ish prediction names
    for p in exp_dir.rglob("*.csv"):
        n = p.name.lower()
        if "addh" in n and "pred" in n and ("out" in n or "addhout" in n):
            return p
    return None


def find_oof_path(exp_dir: Path) -> Optional[Path]:
    for sub in [exp_dir / "work", exp_dir / "cv_work", exp_dir]:
        for name in OOF_FILENAMES:
            p = sub / name
            if p.exists():
                return p
    return None


def detect_pred_col(df: pd.DataFrame) -> Optional[str]:
    for c in PRED_COL_CANDIDATES:
        if c in df.columns:
            return c
    # fallback: any column containing pred, excluding std/min/max/err
    for c in df.columns:
        lc = str(c).lower()
        if "pred" in lc and not any(x in lc for x in ["std", "min", "max", "err", "rank", "n_"]):
            return c
    return None


def infer_model_family(root: Path, exp: str, cfg: Dict) -> str:
    r = str(root).lower()
    if "modelgrid" in r or "multiview" in r:
        return "multiview"
    if "graph_ensemble" in r:
        return "graph_ensemble"
    mv = str(cfg.get("model_variant", "")).lower()
    if mv in {"graph_only", "concat_interact", "gated_sum", "residual_graph", "text_only"}:
        return "multiview"
    return "unknown"


def normalize_feature_mode(exp: str, cfg: Dict, root: Path) -> str:
    for k in ["feature_mode", "model_variant"]:
        v = cfg.get(k)
        if v is not None and str(v) and str(v) != "nan":
            return str(v)
    # infer from experiment name
    lower = exp.lower()
    for m in ["addh_bare", "addh_delta", "bare_delta", "graph_only", "concat_interact", "residual_graph", "gated_sum", "text_only", "full", "delta", "addh", "bare"]:
        if m in lower:
            return m
    if "modelgrid" in str(root).lower():
        return "multiview_unknown"
    return "unknown"


def normalize_metrics_dict(m: Dict) -> Dict[str, float]:
    out = {}
    aliases = {
        "mae": "oof_mae", "rmse": "oof_rmse", "r2": "oof_r2",
        "test_mae": "oof_mae", "test_rmse": "oof_rmse", "test_r2": "oof_r2",
        "oof_mae": "oof_mae", "oof_rmse": "oof_rmse", "oof_r2": "oof_r2",
    }
    for k, v in m.items():
        kk = aliases.get(str(k), str(k))
        if kk in {"oof_mae", "oof_rmse", "oof_r2"}:
            try:
                out[kk] = float(v)
            except Exception:
                pass
    return out


def load_summary_oof_maps(roots: List[Path]) -> Dict[str, Dict[str, float]]:
    maps: Dict[str, Dict[str, float]] = {}
    candidates = [
        "graph_ensemble_comparison_summary.csv",
        "experiment_comparison_summary.csv",
        "graph_ensemble_addhout_posterior_summary.csv",  # only OOF columns used
    ]
    for root in roots:
        for name in candidates:
            p = root / name
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            if "exp_name" not in df.columns:
                continue
            for _, r in df.iterrows():
                exp = str(r.get("exp_name", ""))
                if not exp:
                    continue
                rec = maps.setdefault(exp, {})
                for c in ["oof_mae", "oof_rmse", "oof_r2"]:
                    if c in df.columns and pd.notna(r.get(c)):
                        try:
                            rec[c] = float(r.get(c))
                        except Exception:
                            pass
    return maps


def scan_experiments(roots: List[Path], master_ids: List[str], args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    include_re = re.compile(args.include_exp_regex) if args.include_exp_regex else None
    exclude_re = re.compile(args.exclude_exp_regex) if args.exclude_exp_regex else None
    fallback_oof = load_summary_oof_maps(roots)

    rows = []
    pred_dfs: Dict[str, pd.DataFrame] = {}
    seen = set()

    for root in roots:
        if not root.exists():
            continue
        for exp_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            exp = exp_dir.name
            if exp in seen:
                continue
            if include_re and not include_re.search(exp):
                continue
            if exclude_re and exclude_re.search(exp):
                continue
            pred_path = find_pred_path(exp_dir)
            if pred_path is None:
                continue
            try:
                pred = pd.read_csv(pred_path).drop(columns=["eq_emb"], errors="ignore")
            except Exception:
                continue
            if "id" not in pred.columns:
                continue
            pred_col = detect_pred_col(pred)
            if pred_col is None:
                continue
            pred = pred.copy()
            pred["id"] = pred["id"].astype(str)
            pred["pred"] = pd.to_numeric(pred[pred_col], errors="coerce")
            # preserve uncertainty columns where available
            for c in ["pred_mean", "pred_std", "pred_min", "pred_max", "n_runs"]:
                if c in pred.columns:
                    pred[c] = pd.to_numeric(pred[c], errors="coerce")
            if "pred_std" not in pred.columns and "std" in pred.columns:
                pred["pred_std"] = pd.to_numeric(pred["std"], errors="coerce")

            pred_for_ensemble = pred[[c for c in pred.columns if c not in LABEL_COLS]].copy()

            cfg = safe_json_load(exp_dir / "experiment_config.json")
            oof = {}
            oof_path = find_oof_path(exp_dir)
            if oof_path:
                oof = normalize_metrics_dict(safe_json_load(oof_path))
            if not oof:
                oof = fallback_oof.get(exp, {})

            n_pred = int(pred_for_ensemble["pred"].notna().sum())
            ids_pred = set(pred_for_ensemble.loc[pred_for_ensemble["pred"].notna(), "id"].astype(str))
            coverage = len(ids_pred.intersection(master_ids)) / max(len(master_ids), 1)
            feature_mode = normalize_feature_mode(exp, cfg, root)
            model_family = infer_model_family(root, exp, cfg)

            rec = {
                "exp_name": exp,
                "root": str(root),
                "pred_path": str(pred_path),
                "pred_col_used": pred_col,
                "model_family": model_family,
                "feature_mode": feature_mode,
                "n_pred": n_pred,
                "n_master": len(master_ids),
                "coverage": coverage,
                "target_abs_max": cfg.get("target_abs_max", np.nan),
                "pca_dim": cfg.get("pca_dim", np.nan),
                "calibration": cfg.get("calibration", np.nan),
                "aggregate_method": cfg.get("aggregate_method", cfg.get("ensemble", np.nan)),
                "models": cfg.get("models", np.nan),
            }
            rec.update(oof)
            if "pred_std" in pred_for_ensemble.columns:
                rec["pred_std_mean_addhout_unlabeled"] = float(pred_for_ensemble["pred_std"].mean(skipna=True))
                rec["pred_std_median_addhout_unlabeled"] = float(pred_for_ensemble["pred_std"].median(skipna=True))
            else:
                # if no within-run uncertainty, use rough cross-run unavailable marker; later family selection can still include it
                rec["pred_std_mean_addhout_unlabeled"] = np.nan
                rec["pred_std_median_addhout_unlabeled"] = np.nan
            rec["pred_mean_addhout_unlabeled"] = float(pred_for_ensemble["pred"].mean(skipna=True))
            rec["pred_min_addhout_unlabeled"] = float(pred_for_ensemble["pred"].min(skipna=True))
            rec["pred_max_addhout_unlabeled"] = float(pred_for_ensemble["pred"].max(skipna=True))
            rec["n_runs_mean"] = float(pred_for_ensemble["n_runs"].mean(skipna=True)) if "n_runs" in pred_for_ensemble.columns else np.nan

            rows.append(rec)
            pred_dfs[exp] = pred_for_ensemble
            seen.add(exp)

    return pd.DataFrame(rows), pred_dfs


def add_source_range_diagnostics(summary: pd.DataFrame, train_master: Path, target_col: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    diag = {}
    if not train_master.exists():
        summary["source_range_penalty"] = 0.0
        return summary, diag
    try:
        cols = pd.read_csv(train_master, nrows=1).columns
        if target_col not in cols:
            summary["source_range_penalty"] = 0.0
            return summary, diag
        train = pd.read_csv(train_master)
        y = pd.to_numeric(train[target_col], errors="coerce").dropna()
        if len(y) < 5:
            raise ValueError("not enough source targets")
        lo, hi = float(y.quantile(0.01)), float(y.quantile(0.99))
        q05, q95 = float(y.quantile(0.05)), float(y.quantile(0.95))
        diag = {"source_q01": lo, "source_q99": hi, "source_q05": q05, "source_q95": q95}
        span = max(hi - lo, 1e-9)
        low_excess = ((summary["pred_min_addhout_unlabeled"] - lo).clip(upper=0).abs() / span).fillna(0)
        high_excess = ((summary["pred_max_addhout_unlabeled"] - hi).clip(lower=0).abs() / span).fillna(0)
        summary["source_range_penalty"] = low_excess + high_excess
    except Exception:
        summary["source_range_penalty"] = 0.0
    return summary, diag


def rank_series(s: pd.Series, ascending: bool) -> pd.Series:
    return s.rank(ascending=ascending, method="average", na_option="bottom", pct=True)


def compute_blind_score(summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = summary.copy()
    for c in ["oof_mae", "oof_rmse", "oof_r2", "coverage", "pred_std_mean_addhout_unlabeled", "source_range_penalty"]:
        if c not in df.columns:
            df[c] = np.nan
    df["eligible"] = True
    df.loc[df["coverage"] < args.min_coverage, "eligible"] = False
    df.loc[df["n_pred"] < args.min_n_addhout, "eligible"] = False
    df.loc[df["oof_rmse"].isna(), "eligible"] = False

    prefer_modes = [x.strip() for x in args.prefer_feature_modes.split(',') if x.strip()]
    df["feature_preference_bonus"] = 0.0
    if prefer_modes:
        df["feature_preference_bonus"] = df["feature_mode"].astype(str).apply(lambda x: -0.02 if x in prefer_modes else 0.0)

    penalties = parse_key_value_map(args.conservative_mode_penalty, float)
    df["conservative_feature_penalty"] = df["feature_mode"].astype(str).map(penalties).fillna(0.05)

    r_rmse = rank_series(df["oof_rmse"], ascending=True)
    r_mae = rank_series(df["oof_mae"], ascending=True)
    r_r2 = rank_series(df["oof_r2"], ascending=False)
    r_cov = rank_series(df["coverage"], ascending=False)
    # NaN uncertainty should not be best; fill for rank via rank bottom
    r_std = rank_series(df["pred_std_mean_addhout_unlabeled"], ascending=True)
    r_range = rank_series(df["source_range_penalty"], ascending=True)

    if args.score_mode == "oof_first":
        score = 0.45*r_rmse + 0.25*r_mae + 0.15*r_r2 + 0.05*r_cov + 0.05*r_std + 0.05*r_range
    elif args.score_mode == "stability_first":
        score = 0.25*r_rmse + 0.15*r_mae + 0.10*r_r2 + 0.15*r_cov + 0.25*r_std + 0.10*r_range
    elif args.score_mode == "conservative":
        # Still source-domain based, but less dominated by OOF and adds conservative feature prior.
        score = 0.24*r_rmse + 0.16*r_mae + 0.10*r_r2 + 0.14*r_cov + 0.22*r_std + 0.09*r_range
        score = score + df["conservative_feature_penalty"]
    else:
        score = 0.35*r_rmse + 0.20*r_mae + 0.15*r_r2 + 0.10*r_cov + 0.15*r_std + 0.05*r_range

    df["blind_selection_score"] = score + df["feature_preference_bonus"]
    df.loc[~df["eligible"], "blind_selection_score"] = np.inf
    return df.sort_values(["blind_selection_score", "oof_rmse", "coverage"], ascending=[True, True, False])


def select_family_diverse(summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    eligible = summary[summary["eligible"]].copy().sort_values(["blind_selection_score", "oof_rmse"])
    if eligible.empty:
        return eligible
    if not args.family_diverse:
        return eligible.head(args.top_k).copy()

    fcap_map = {k:int(v) for k,v in parse_key_value_map(args.feature_mode_cap_map, float).items()}
    famcap_map = {k:int(v) for k,v in parse_key_value_map(args.model_family_cap_map, float).items()}
    req_map = {k:int(v) for k,v in parse_key_value_map(args.require_feature_modes, float).items()}

    selected_idx = []
    feature_counts: Dict[str, int] = {}
    family_counts: Dict[str, int] = {}

    def can_add(row) -> bool:
        fm = str(row.get("feature_mode", "unknown"))
        fam = str(row.get("model_family", "unknown"))
        cap = fcap_map.get(fm, args.feature_mode_cap)
        famcap = famcap_map.get(fam, args.top_k)
        if cap <= 0:
            return False
        if feature_counts.get(fm, 0) >= cap:
            return False
        if family_counts.get(fam, 0) >= famcap:
            return False
        return True

    def add_row(idx, row):
        fm = str(row.get("feature_mode", "unknown"))
        fam = str(row.get("model_family", "unknown"))
        selected_idx.append(idx)
        feature_counts[fm] = feature_counts.get(fm, 0) + 1
        family_counts[fam] = family_counts.get(fam, 0) + 1

    # Phase 1: satisfy minimum quotas where possible
    for fm, min_n in req_map.items():
        cand = eligible[eligible["feature_mode"].astype(str) == fm]
        for idx, row in cand.iterrows():
            if len(selected_idx) >= args.top_k:
                break
            if idx in selected_idx:
                continue
            if feature_counts.get(fm, 0) >= min_n:
                break
            if can_add(row):
                add_row(idx, row)

    # Phase 2: fill by score with caps
    for idx, row in eligible.iterrows():
        if len(selected_idx) >= args.top_k:
            break
        if idx in selected_idx:
            continue
        if can_add(row):
            add_row(idx, row)

    # If caps too strict, relax only feature cap but keep model family cap, to avoid empty output
    if len(selected_idx) < min(args.top_k, len(eligible)):
        for idx, row in eligible.iterrows():
            if len(selected_idx) >= args.top_k:
                break
            if idx in selected_idx:
                continue
            fam = str(row.get("model_family", "unknown"))
            famcap = famcap_map.get(fam, args.top_k)
            if family_counts.get(fam, 0) >= famcap:
                continue
            add_row(idx, row)

    out = eligible.loc[selected_idx].copy()
    out["selection_policy"] = "family_diverse_conservative"
    return out


def selected_weights(sel: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "uniform":
        w = pd.Series(1.0, index=sel["exp_name"].values)
    elif mode == "rank":
        vals = np.arange(len(sel), 0, -1, dtype=float)
        w = pd.Series(vals, index=sel["exp_name"].values)
    else:
        rmse = pd.to_numeric(sel["oof_rmse"], errors="coerce").replace(0, np.nan)
        cov = pd.to_numeric(sel["coverage"], errors="coerce").fillna(0.0)
        std = pd.to_numeric(sel["pred_std_mean_addhout_unlabeled"], errors="coerce")
        std_default = std.median(skipna=True) if std.notna().any() else 1.0
        std_factor = 1.0 / (1.0 + std.fillna(std_default))
        vals = (1.0 / (rmse ** 2)).fillna(0.0) * cov * std_factor
        w = pd.Series(vals.values, index=sel["exp_name"].values)
        if not np.isfinite(w).any() or w.sum() <= 0:
            w = pd.Series(1.0, index=sel["exp_name"].values)
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if w.sum() <= 0:
        w[:] = 1.0
    return w / w.sum()


def trimmed_mean(vals: np.ndarray, trim: float = 0.1) -> float:
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan
    if len(vals) < 5:
        return float(np.mean(vals))
    vals = np.sort(vals)
    k = int(math.floor(len(vals) * trim))
    if k > 0 and 2*k < len(vals):
        vals = vals[k:-k]
    return float(np.mean(vals))


def make_ensemble_predictions(master: pd.DataFrame, selected: pd.DataFrame, weights: pd.Series, pred_dfs: Dict[str, pd.DataFrame], args: argparse.Namespace, source_diag: Dict[str, float]) -> pd.DataFrame:
    out = master.copy()
    pred_cols = []
    for exp in selected["exp_name"]:
        df = pred_dfs[exp][[c for c in ["id", "pred", "pred_std", "n_runs"] if c in pred_dfs[exp].columns]].copy()
        df = df.rename(columns={"pred": f"pred__{exp}", "pred_std": f"predstd__{exp}", "n_runs": f"nruns__{exp}"})
        out = out.merge(df, on="id", how="left")
        pred_cols.append(f"pred__{exp}")

    W = weights.reindex(selected["exp_name"].values).fillna(0.0)
    pred_matrix = out[pred_cols].apply(pd.to_numeric, errors="coerce")
    pred_values = pred_matrix.to_numpy(dtype=float)
    weight_values = np.array([W[c.replace("pred__", "")] for c in pred_cols], dtype=float)

    weighted_preds, med_preds, trim_preds, std_across, n_models = [], [], [], [], []
    for row in pred_values:
        mask = np.isfinite(row)
        n_models.append(int(mask.sum()))
        if mask.sum() == 0:
            weighted_preds.append(np.nan); med_preds.append(np.nan); trim_preds.append(np.nan); std_across.append(np.nan); continue
        ww = weight_values[mask]
        ww = np.ones_like(ww) / len(ww) if ww.sum() <= 0 else ww / ww.sum()
        vv = row[mask]
        weighted_preds.append(float(np.sum(ww * vv)))
        med_preds.append(float(np.median(vv)))
        trim_preds.append(trimmed_mean(vv, 0.1))
        std_across.append(float(np.std(vv, ddof=1)) if len(vv) > 1 else 0.0)

    out["pred_strict_blind_weighted"] = weighted_preds
    out["pred_strict_blind_median"] = med_preds
    out["pred_strict_blind_trimmed_mean"] = trim_preds
    out["pred_strict_blind_std_across_models"] = std_across
    out["pred_strict_blind_n_models"] = n_models
    # Conservative blend: median has larger weight to reduce dominance from overconfident OOF models.
    out["pred_strict_blind"] = 0.50*out["pred_strict_blind_weighted"] + 0.30*out["pred_strict_blind_median"] + 0.20*out["pred_strict_blind_trimmed_mean"]

    if args.clip_to_source_range and source_diag:
        lo, hi = source_diag.get("source_q01"), source_diag.get("source_q99")
        if lo is not None and hi is not None:
            out["pred_strict_blind_unclipped"] = out["pred_strict_blind"]
            out["pred_strict_blind"] = out["pred_strict_blind"].clip(lo, hi)

    out["selected_models"] = ",".join(selected["exp_name"].tolist())
    out["strict_blind_weight_mode"] = args.weight_mode
    out["strict_blind_score_mode"] = args.score_mode
    out["strict_blind_family_diverse"] = bool(args.family_diverse)
    out["strict_blind_prediction_rank"] = out["pred_strict_blind"].rank(method="average", ascending=True)
    out["strict_blind_uncertainty_rank"] = out["pred_strict_blind_std_across_models"].rank(method="average", ascending=True)
    return out


def audit_metrics(out: pd.DataFrame, target_cols: Iterable[str], pred_col: str = "pred_strict_blind") -> pd.DataFrame:
    rows = []
    if mean_absolute_error is None:
        return pd.DataFrame(rows)
    for c in target_cols:
        if c not in out.columns:
            continue
        tmp = out[[pred_col, c]].copy()
        tmp[pred_col] = pd.to_numeric(tmp[pred_col], errors="coerce")
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        tmp = tmp.dropna()
        if len(tmp) < 3:
            continue
        rows.append({
            "target_col": c,
            "n": len(tmp),
            "mae": mean_absolute_error(tmp[c], tmp[pred_col]),
            "rmse": mean_squared_error(tmp[c], tmp[pred_col]) ** 0.5,
            "r2": r2_score(tmp[c], tmp[pred_col]),
            "pearson": tmp[pred_col].corr(tmp[c], method="pearson"),
            "spearman": tmp[pred_col].corr(tmp[c], method="spearman"),
            "bias": (tmp[pred_col] - tmp[c]).mean(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    roots = [Path(x) for x in args.roots]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    master_full = pd.read_csv(args.addhout_master_csv)
    if "id" not in master_full.columns:
        raise SystemExit("[ERROR] addH-out master must contain id column.")
    master_full["id"] = master_full["id"].astype(str)
    master_ids = master_full["id"].dropna().astype(str).tolist()

    meta_cols = [
        "id", "material", "idx", "element", "dopant", "family_base", "family_base_miller", "miller",
        "site_type", "anchor_count", "slab_formula", "data_source", "contcar_path", "bare_contcar_path", "cif_path",
    ]
    if args.audit_with_labels and not args.strict_output_no_labels:
        meta_cols += ["h_ads_excel", "target", "target_computed", "target_mismatch_excel_minus_computed"]
    meta_cols = [c for c in meta_cols if c in master_full.columns]
    master = master_full[meta_cols].drop_duplicates("id").copy()

    print("[INFO] scanning experiments...")
    summary, pred_dfs = scan_experiments(roots, master_ids, args)
    if summary.empty:
        raise SystemExit("[ERROR] no usable prediction experiments found.")

    summary, source_diag = add_source_range_diagnostics(summary, Path(args.train_master_csv), args.target_col)
    summary = compute_blind_score(summary, args)

    summary_path = out_dir / "strict_blind_model_selection_summary.csv"
    summary_xlsx = out_dir / "strict_blind_model_selection_summary.xlsx"
    summary.to_csv(summary_path, index=False)
    try:
        summary.to_excel(summary_xlsx, index=False)
    except Exception:
        pass

    selected_all = summary[summary["eligible"]].head(args.top_k).copy()
    selected = select_family_diverse(summary, args)
    if selected.empty:
        raise SystemExit("[ERROR] no eligible models after strict blind filtering.")
    weights = selected_weights(selected, args.weight_mode)
    selected["strict_blind_weight"] = selected["exp_name"].map(weights)

    selected_all_path = out_dir / "strict_blind_selected_models_naive_topk.csv"
    selected_all.to_csv(selected_all_path, index=False)
    selected_path = out_dir / "strict_blind_selected_models.csv"
    selected.to_csv(selected_path, index=False)
    with (out_dir / "strict_blind_selected_models.txt").open("w", encoding="utf-8") as f:
        for _, r in selected.iterrows():
            f.write(
                f"{r['exp_name']}\tfeature_mode={r.get('feature_mode')}\tmodel_family={r.get('model_family')}"
                f"\tweight={r.get('strict_blind_weight', np.nan):.6f}\toof_rmse={r.get('oof_rmse', np.nan)}\tcoverage={r.get('coverage', np.nan)}\n"
            )

    print("[INFO] selected models:")
    show_sel = ["exp_name", "model_family", "feature_mode", "oof_rmse", "oof_mae", "oof_r2", "coverage", "pred_std_mean_addhout_unlabeled", "blind_selection_score", "strict_blind_weight"]
    show_sel = [c for c in show_sel if c in selected.columns]
    print(selected[show_sel].to_string(index=False))

    pred_out = make_ensemble_predictions(master, selected, weights, pred_dfs, args, source_diag)
    pred_out = pred_out.sort_values(["pred_strict_blind", "pred_strict_blind_std_across_models"], ascending=[True, True])

    pred_csv = out_dir / "strict_blind_addhout_predictions.csv"
    pred_xlsx = out_dir / "strict_blind_addhout_predictions.xlsx"
    pred_out.to_csv(pred_csv, index=False)
    try:
        pred_out.to_excel(pred_xlsx, index=False)
    except Exception:
        pass

    if args.audit_with_labels:
        # Merge labels only for posthoc metrics, not for prediction file unless user disabled strict labels.
        audit_df = pred_out.merge(master_full[[c for c in ["id", "h_ads_excel", "target", "target_computed"] if c in master_full.columns]], on="id", how="left")
        audit = audit_metrics(audit_df, ["h_ads_excel", "target", "target_computed"])
        audit.to_csv(out_dir / "strict_blind_posthoc_audit_metrics.csv", index=False)
        print("[POSTHOC AUDIT ONLY - NOT USED FOR SELECTION]")
        print(audit.to_string(index=False))

    print("[OK] saved:", summary_path)
    print("[OK] saved:", selected_path)
    print("[OK] saved:", pred_csv)
    print("[INFO] top predictions:")
    show_cols = [
        "id", "material", "element", "dopant", "pred_strict_blind", "pred_strict_blind_weighted",
        "pred_strict_blind_median", "pred_strict_blind_trimmed_mean", "pred_strict_blind_std_across_models", "pred_strict_blind_n_models",
    ]
    show_cols = [c for c in show_cols if c in pred_out.columns]
    print(pred_out[show_cols].head(50).to_string(index=False))


if __name__ == "__main__":
    main()
