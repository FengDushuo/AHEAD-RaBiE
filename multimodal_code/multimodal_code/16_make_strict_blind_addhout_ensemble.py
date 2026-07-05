#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strict blind AddH-out ensemble builder.

Goal
----
Build the best possible *label-blind* AddH-out prediction by using only:
  1) source-domain validation metrics from addH/addH-2 training data (OOF/CV),
  2) addH-out structures/features/predictions as unlabeled inference targets,
  3) prediction agreement/uncertainty on addH-out.

It deliberately does NOT use addH-out labels (h_ads_excel/target/target_computed)
for model selection, weighting, calibration, or ranking.

Outputs
-------
- strict_blind_model_selection_summary.csv/xlsx
- strict_blind_selected_models.txt
- strict_blind_addhout_predictions.csv/xlsx
- optional audit metrics if --audit-with-labels is supplied; this is post-hoc only.
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
    "addH_out_pred_ensemble_robust.csv",
    "addH_out_pred_ensemble.csv",
]

OOF_FILENAMES = [
    "test_oof_graph_ensemble_metrics.json",
    "oof_graph_ensemble_metrics.json",
    "metrics.json",
]

LABEL_COLS = {"h_ads_excel", "target", "target_computed", "abs_err_vs_target", "abs_err_vs_h_ads_excel", "abs_err_vs_target_computed"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Make strict blind AddH-out ensemble from existing prediction runs.")
    ap.add_argument("--roots", nargs="+", default=["outputs_addh_graph_ensemble_refine_v2", "outputs_addh_graph_ensemble"],
                    help="Experiment roots to scan. Newer roots should be listed first.")
    ap.add_argument("--addhout-master-csv", default="outputs_addh_full_mm_envsplit/addH_out_master_normalized.csv")
    ap.add_argument("--train-master-csv", default="outputs_addh_full_mm_envsplit/addH_master_target_weighted_mild.csv",
                    help="Optional source training table, only used for source-range plausibility diagnostics.")
    ap.add_argument("--out-dir", default="outputs_addh_strict_blind")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--min-coverage", type=float, default=0.90)
    ap.add_argument("--min-n-addhout", type=int, default=10)
    ap.add_argument("--prefer-feature-modes", default="",
                    help="Optional comma list, e.g. addh_bare,addh,full. Only affects tie/soft preference, not hard filtering.")
    ap.add_argument("--include-exp-regex", default="", help="Only include exp_name matching regex if provided.")
    ap.add_argument("--exclude-exp-regex", default="", help="Exclude exp_name matching regex if provided.")
    ap.add_argument("--score-mode", default="balanced", choices=["balanced", "oof_first", "stability_first"],
                    help="How to rank source-trained models without using addH-out labels.")
    ap.add_argument("--weight-mode", default="soft_inverse_rmse", choices=["soft_inverse_rmse", "rank", "uniform"],
                    help="How to weight selected model predictions.")
    ap.add_argument("--clip-to-source-range", action="store_true",
                    help="Clip final predictions to source training target 1%-99% range. Disabled by default to preserve blind extrapolation.")
    ap.add_argument("--target-col", default="target", help="Source target column in train master for range diagnostics.")
    ap.add_argument("--audit-with-labels", action="store_true",
                    help="Post-hoc only: include addH-out labels and compute metrics. Never used for selection/weights.")
    return ap.parse_args()


def safe_json_load(p: Path) -> Dict:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def find_pred_path(exp_dir: Path) -> Optional[Path]:
    for sub in [exp_dir / "work", exp_dir / "cv_work", exp_dir]:
        for name in PRED_FILENAMES:
            p = sub / name
            if p.exists():
                return p
    return None


def find_oof_path(exp_dir: Path) -> Optional[Path]:
    for sub in [exp_dir / "work", exp_dir / "cv_work", exp_dir]:
        for name in OOF_FILENAMES:
            p = sub / name
            if p.exists():
                return p
    return None


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
    """Fallback: read only OOF columns from existing summary CSV files.
    This intentionally ignores all addH-out label metrics.
    """
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
            # keep first occurrence from earlier root priority
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
            if "id" not in pred.columns or "pred" not in pred.columns:
                continue
            pred = pred.copy()
            pred["id"] = pred["id"].astype(str)
            pred["pred"] = pd.to_numeric(pred["pred"], errors="coerce")
            for c in ["pred_mean", "pred_std", "pred_min", "pred_max", "n_runs"]:
                if c in pred.columns:
                    pred[c] = pd.to_numeric(pred[c], errors="coerce")
            # Do not use label columns in selection. Drop from saved pred copy for safety.
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

            rec = {
                "exp_name": exp,
                "root": str(root),
                "pred_path": str(pred_path),
                "n_pred": n_pred,
                "n_master": len(master_ids),
                "coverage": coverage,
                "feature_mode": cfg.get("feature_mode", cfg.get("model_variant", np.nan)),
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
    if not train_master.exists() or target_col not in pd.read_csv(train_master, nrows=1).columns:
        summary["source_range_penalty"] = 0.0
        return summary, diag
    try:
        train = pd.read_csv(train_master)
        y = pd.to_numeric(train[target_col], errors="coerce").dropna()
        if len(y) < 5:
            raise ValueError("not enough source targets")
        lo, hi = float(y.quantile(0.01)), float(y.quantile(0.99))
        q05, q95 = float(y.quantile(0.05)), float(y.quantile(0.95))
        diag = {"source_q01": lo, "source_q99": hi, "source_q05": q05, "source_q95": q95}
        # Model-level diagnostic only; not a strong penalty. Penalize if model predicts far beyond source central range.
        span = max(hi - lo, 1e-9)
        low_excess = ((summary["pred_min_addhout_unlabeled"] - lo).clip(upper=0).abs() / span).fillna(0)
        high_excess = ((summary["pred_max_addhout_unlabeled"] - hi).clip(lower=0).abs() / span).fillna(0)
        summary["source_range_penalty"] = low_excess + high_excess
    except Exception:
        summary["source_range_penalty"] = 0.0
    return summary, diag


def rank_series(s: pd.Series, ascending: bool, na_value: float = 1e9) -> pd.Series:
    x = s.copy()
    if not ascending:
        # High is good, rank descending.
        return x.rank(ascending=False, method="average", na_option="bottom", pct=True)
    return x.rank(ascending=True, method="average", na_option="bottom", pct=True)


def compute_blind_score(summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = summary.copy()
    # Hard filters first
    df["eligible"] = True
    df.loc[df["coverage"] < args.min_coverage, "eligible"] = False
    df.loc[df["n_pred"] < args.min_n_addhout, "eligible"] = False
    df.loc[df["oof_rmse"].isna(), "eligible"] = False

    prefer_modes = [x.strip() for x in args.prefer_feature_modes.split(",") if x.strip()]
    if prefer_modes:
        df["feature_preference_bonus"] = df["feature_mode"].astype(str).apply(lambda x: 0.0 if x in prefer_modes else 0.10)
    else:
        df["feature_preference_bonus"] = 0.0

    r_rmse = rank_series(df["oof_rmse"], ascending=True)
    r_mae = rank_series(df["oof_mae"] if "oof_mae" in df.columns else df["oof_rmse"], ascending=True)
    r_r2 = rank_series(df["oof_r2"] if "oof_r2" in df.columns else -df["oof_rmse"], ascending=False)
    r_cov = rank_series(df["coverage"], ascending=False)
    r_std = rank_series(df["pred_std_mean_addhout_unlabeled"], ascending=True)
    r_range = rank_series(df["source_range_penalty"], ascending=True)

    if args.score_mode == "oof_first":
        score = 0.45*r_rmse + 0.25*r_mae + 0.15*r_r2 + 0.05*r_cov + 0.05*r_std + 0.05*r_range
    elif args.score_mode == "stability_first":
        score = 0.25*r_rmse + 0.15*r_mae + 0.10*r_r2 + 0.15*r_cov + 0.25*r_std + 0.10*r_range
    else:
        score = 0.35*r_rmse + 0.20*r_mae + 0.15*r_r2 + 0.10*r_cov + 0.15*r_std + 0.05*r_range

    df["blind_selection_score"] = score + df["feature_preference_bonus"]
    df.loc[~df["eligible"], "blind_selection_score"] = np.inf
    return df.sort_values(["blind_selection_score", "oof_rmse", "coverage"], ascending=[True, True, False])


def selected_weights(sel: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "uniform":
        w = pd.Series(1.0, index=sel["exp_name"].values)
    elif mode == "rank":
        n = len(sel)
        vals = np.arange(n, 0, -1, dtype=float)
        w = pd.Series(vals, index=sel["exp_name"].values)
    else:
        rmse = pd.to_numeric(sel["oof_rmse"], errors="coerce").replace(0, np.nan)
        cov = pd.to_numeric(sel["coverage"], errors="coerce").fillna(0.0)
        std = pd.to_numeric(sel["pred_std_mean_addhout_unlabeled"], errors="coerce")
        std_factor = 1.0 / (1.0 + std.fillna(std.median(skipna=True) if std.notna().any() else 1.0))
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
    std_cols = []

    for exp in selected["exp_name"]:
        df = pred_dfs[exp][[c for c in ["id", "pred", "pred_std", "n_runs"] if c in pred_dfs[exp].columns]].copy()
        df = df.rename(columns={"pred": f"pred__{exp}", "pred_std": f"predstd__{exp}", "n_runs": f"nruns__{exp}"})
        out = out.merge(df, on="id", how="left")
        pred_cols.append(f"pred__{exp}")
        if f"predstd__{exp}" in out.columns:
            std_cols.append(f"predstd__{exp}")

    W = weights.reindex(selected["exp_name"].values).fillna(0.0)
    # row-wise weighted mean with available predictions only
    pred_matrix = out[pred_cols].apply(pd.to_numeric, errors="coerce")
    pred_values = pred_matrix.to_numpy(dtype=float)
    weight_values = np.array([W[c.replace("pred__", "")] for c in pred_cols], dtype=float)

    weighted_preds = []
    med_preds = []
    trim_preds = []
    std_across = []
    n_models = []
    for row in pred_values:
        mask = np.isfinite(row)
        n_models.append(int(mask.sum()))
        if mask.sum() == 0:
            weighted_preds.append(np.nan)
            med_preds.append(np.nan)
            trim_preds.append(np.nan)
            std_across.append(np.nan)
            continue
        ww = weight_values[mask]
        if ww.sum() <= 0:
            ww = np.ones_like(ww) / len(ww)
        else:
            ww = ww / ww.sum()
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

    # Final strict-blind prediction: robust blend of weighted and median, no labels.
    out["pred_strict_blind"] = 0.65*out["pred_strict_blind_weighted"] + 0.35*out["pred_strict_blind_median"]

    if args.clip_to_source_range and source_diag:
        lo, hi = source_diag.get("source_q01"), source_diag.get("source_q99")
        if lo is not None and hi is not None:
            out["pred_strict_blind_unclipped"] = out["pred_strict_blind"]
            out["pred_strict_blind"] = out["pred_strict_blind"].clip(lo, hi)

    out["selected_models"] = ",".join(selected["exp_name"].tolist())
    out["strict_blind_weight_mode"] = args.weight_mode
    out["strict_blind_score_mode"] = args.score_mode

    # Candidate order if user wants screening, but this is not used as accuracy metric.
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

    # For strict blind output, keep metadata but drop labels unless audit requested.
    meta_cols = [
        "id", "material", "idx", "element", "dopant", "family_base", "family_base_miller", "miller",
        "site_type", "anchor_count", "slab_formula", "data_source", "contcar_path", "bare_contcar_path", "cif_path",
    ]
    if args.audit_with_labels:
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

    selected = summary[summary["eligible"]].head(args.top_k).copy()
    if selected.empty:
        raise SystemExit("[ERROR] no eligible models after strict blind filtering.")
    weights = selected_weights(selected, args.weight_mode)
    selected["strict_blind_weight"] = selected["exp_name"].map(weights)

    selected_path = out_dir / "strict_blind_selected_models.csv"
    selected.to_csv(selected_path, index=False)
    with (out_dir / "strict_blind_selected_models.txt").open("w", encoding="utf-8") as f:
        for _, r in selected.iterrows():
            f.write(f"{r['exp_name']}\tweight={r['strict_blind_weight']:.6f}\toof_rmse={r.get('oof_rmse', np.nan)}\tcoverage={r.get('coverage', np.nan)}\n")

    print("[INFO] selected models:")
    print(selected[["exp_name", "feature_mode", "oof_rmse", "oof_mae", "oof_r2", "coverage", "pred_std_mean_addhout_unlabeled", "blind_selection_score", "strict_blind_weight"]].to_string(index=False))

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
        audit = audit_metrics(pred_out, ["h_ads_excel", "target", "target_computed"])
        audit.to_csv(out_dir / "strict_blind_posthoc_audit_metrics.csv", index=False)
        print("[POSTHOC AUDIT ONLY - NOT USED FOR SELECTION]")
        print(audit.to_string(index=False))

    print("[OK] saved:", summary_path)
    print("[OK] saved:", selected_path)
    print("[OK] saved:", pred_csv)
    print("[INFO] top predictions:")
    show_cols = [
        "id", "material", "element", "dopant", "pred_strict_blind", "pred_strict_blind_weighted",
        "pred_strict_blind_median", "pred_strict_blind_std_across_models", "pred_strict_blind_n_models",
    ]
    if args.audit_with_labels:
        show_cols += ["h_ads_excel", "target_computed"]
    show_cols = [c for c in show_cols if c in pred_out.columns]
    print(pred_out[show_cols].head(50).to_string(index=False))


if __name__ == "__main__":
    main()
