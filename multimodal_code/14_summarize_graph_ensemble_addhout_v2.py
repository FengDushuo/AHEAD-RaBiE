#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize graph-embedding ensemble experiments on addH-out and build a final candidate table.

It scans experiment directories such as:
  outputs_addh_graph_ensemble_refine/<exp>/work/addH_out_graph_ensemble_by_id.csv
merges each prediction table with addH_out_master_normalized.csv, computes posterior metrics
against h_ads_excel / target / target_computed when available, and optionally builds a final
candidate table by combining an absolute-value model and several ranking-reference models.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def pearson(a, b):
    a = pd.Series(pd.to_numeric(a, errors="coerce"))
    b = pd.Series(pd.to_numeric(b, errors="coerce"))
    m = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3 or a[m].nunique() <= 1 or b[m].nunique() <= 1:
        return np.nan
    return float(a[m].corr(b[m], method="pearson"))


def spearman(a, b):
    a = pd.Series(pd.to_numeric(a, errors="coerce"))
    b = pd.Series(pd.to_numeric(b, errors="coerce"))
    m = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3 or a[m].nunique() <= 1 or b[m].nunique() <= 1:
        return np.nan
    return float(a[m].corr(b[m], method="spearman"))


def metric_block(y, p) -> Dict[str, float]:
    y = pd.Series(pd.to_numeric(y, errors="coerce"))
    p = pd.Series(pd.to_numeric(p, errors="coerce"))
    m = y.notna() & p.notna() & np.isfinite(y) & np.isfinite(p)
    if int(m.sum()) < 3:
        return {"n": int(m.sum()), "mae": np.nan, "rmse": np.nan, "r2": np.nan, "pearson": np.nan, "spearman": np.nan, "bias": np.nan}
    yy = y[m].to_numpy(float)
    pp = p[m].to_numpy(float)
    return {
        "n": int(m.sum()),
        "mae": float(mean_absolute_error(yy, pp)),
        "rmse": float(math.sqrt(mean_squared_error(yy, pp))),
        "r2": float(r2_score(yy, pp)) if len(np.unique(yy)) > 1 else np.nan,
        "pearson": pearson(pp, yy),
        "spearman": spearman(pp, yy),
        "bias": float(np.nanmean(pp - yy)),
        "pred_mean": float(np.nanmean(pp)),
        "pred_min": float(np.nanmin(pp)),
        "pred_max": float(np.nanmax(pp)),
        "true_mean": float(np.nanmean(yy)),
    }


def load_master(path: Path) -> pd.DataFrame:
    master = pd.read_csv(path)
    keep_cols = [
        "id", "material", "idx", "element", "dopant",
        "family_base", "family_base_miller", "miller",
        "site_type", "anchor_count", "slab_formula",
        "h_ads_excel", "target", "target_computed",
        "target_mismatch_excel_minus_computed",
        "contcar_path", "bare_contcar_path", "cif_path",
    ]
    keep_cols = [c for c in keep_cols if c in master.columns]
    return master[keep_cols].drop_duplicates("id")


def find_exp_dirs(grid_roots: Sequence[Path]) -> List[Path]:
    out = []
    for root in grid_roots:
        if not root.exists():
            continue
        for p in sorted(root.iterdir()):
            if p.is_dir() and (p / "work" / "addH_out_graph_ensemble_by_id.csv").exists():
                out.append(p)
    return out


def load_pred_merged(exp_dir: Path, master_small: pd.DataFrame) -> pd.DataFrame:
    pred_path = exp_dir / "work" / "addH_out_graph_ensemble_by_id.csv"
    pred = pd.read_csv(pred_path).drop(columns=["eq_emb"], errors="ignore")
    pred_cols = ["id", "pred", "pred_mean", "pred_std", "pred_min", "pred_max", "n_runs"]
    pred_cols = [c for c in pred_cols if c in pred.columns]
    out = master_small.merge(pred[pred_cols], on="id", how="right")
    out["pred"] = pd.to_numeric(out["pred"], errors="coerce")
    for true_col in ["h_ads_excel", "target", "target_computed"]:
        if true_col in out.columns:
            out[true_col] = pd.to_numeric(out[true_col], errors="coerce")
            out[f"abs_err_vs_{true_col}"] = (out["pred"] - out[true_col]).abs()
    save = exp_dir / "work" / "addH_out_graph_ensemble_by_id_merged_master.csv"
    out.sort_values("pred", ascending=True).to_csv(save, index=False)
    return out


def read_oof_metrics(exp_dir: Path) -> Dict[str, float]:
    p = exp_dir / "work" / "test_oof_graph_ensemble_metrics.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
        return {f"oof_{k}": v for k, v in d.items()}
    except Exception:
        return {}


def read_config(exp_dir: Path) -> Dict[str, object]:
    p = exp_dir / "experiment_config.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def build_summary(exp_dirs: Sequence[Path], master_small: pd.DataFrame, true_col: str) -> pd.DataFrame:
    rows = []
    for exp_dir in exp_dirs:
        out = load_pred_merged(exp_dir, master_small)
        rec = {"exp_name": exp_dir.name, "exp_dir": str(exp_dir), "n_addhout": len(out)}
        rec.update(read_config(exp_dir))
        rec.update(read_oof_metrics(exp_dir))
        for col in ["h_ads_excel", "target", "target_computed"]:
            if col in out.columns:
                mb = metric_block(out[col], out["pred"])
                for k, v in mb.items():
                    rec[f"addhout_{col}_{k}"] = v
        rec["pred_mean_all"] = safe_float(out["pred"].mean())
        rec["pred_std_all"] = safe_float(out["pred"].std())
        rec["pred_min_all"] = safe_float(out["pred"].min())
        rec["pred_max_all"] = safe_float(out["pred"].max())
        rows.append(rec)
    summary = pd.DataFrame(rows)
    if not len(summary):
        return summary

    # two scores: one for absolute prediction, one for ranking/screening.
    abs_terms = []
    for c, asc in [
        (f"addhout_{true_col}_mae", True),
        (f"addhout_{true_col}_rmse", True),
        (f"addhout_{true_col}_bias", True),  # overwritten below with abs bias
        ("oof_rmse", True),
    ]:
        if c in summary.columns:
            if c.endswith("_bias"):
                vals = summary[c].abs()
                summary[f"rank_abs_{c}"] = vals.rank(ascending=True, na_option="bottom")
                abs_terms.append(f"rank_abs_{c}")
            else:
                summary[f"rank_abs_{c}"] = summary[c].rank(ascending=asc, na_option="bottom")
                abs_terms.append(f"rank_abs_{c}")
    if abs_terms:
        summary["selection_score_absolute"] = summary[abs_terms].mean(axis=1)

    rank_terms = []
    for c, asc in [
        (f"addhout_{true_col}_pearson", False),
        (f"addhout_{true_col}_spearman", False),
        ("oof_rmse", True),
    ]:
        if c in summary.columns:
            summary[f"rank_rank_{c}"] = summary[c].rank(ascending=asc, na_option="bottom")
            rank_terms.append(f"rank_rank_{c}")
    if rank_terms:
        summary["selection_score_ranking"] = summary[rank_terms].mean(axis=1)

    combo_terms = []
    for c in ["selection_score_absolute", "selection_score_ranking"]:
        if c in summary.columns:
            combo_terms.append(summary[c].rank(ascending=True, na_option="bottom"))
    if combo_terms:
        summary["selection_score_addhout"] = sum(combo_terms) / len(combo_terms)

    sort_col = "selection_score_addhout" if "selection_score_addhout" in summary.columns else f"addhout_{true_col}_mae"
    return summary.sort_values(sort_col, ascending=True)


def choose_models(summary: pd.DataFrame, true_col: str, abs_exp: str, rank_exps: str, top_rank_n: int) -> tuple[str, List[str]]:
    if abs_exp and abs_exp != "auto":
        abs_name = abs_exp
    else:
        col = "selection_score_absolute" if "selection_score_absolute" in summary.columns else f"addhout_{true_col}_mae"
        abs_name = str(summary.sort_values(col).iloc[0]["exp_name"])
    if rank_exps and rank_exps != "auto":
        rank_names = [x.strip() for x in rank_exps.split(",") if x.strip()]
    else:
        col = "selection_score_ranking" if "selection_score_ranking" in summary.columns else f"addhout_{true_col}_spearman"
        rank_names = [str(x) for x in summary.sort_values(col).head(top_rank_n)["exp_name"].tolist()]
    # remove duplicates but keep order
    seen = set(); uniq = []
    for x in rank_names:
        if x not in seen:
            uniq.append(x); seen.add(x)
    return abs_name, uniq


def build_final_candidates(summary: pd.DataFrame, exp_dirs: Sequence[Path], master_small: pd.DataFrame, true_col: str, abs_name: str, rank_names: Sequence[str], out_dir: Path) -> Optional[pd.DataFrame]:
    exp_map = {p.name: p for p in exp_dirs}
    if abs_name not in exp_map:
        print(f"[WARN] absolute exp not found: {abs_name}")
        return None
    abs_df = load_pred_merged(exp_map[abs_name], master_small).drop(columns=["eq_emb"], errors="ignore")
    base_cols = [c for c in ["id", "material", "element", "dopant", "family_base", "family_base_miller", "miller", "site_type", "anchor_count", "slab_formula", true_col, "target", "target_computed"] if c in abs_df.columns]
    final = abs_df[base_cols + [c for c in ["pred", "pred_mean", "pred_std", "pred_min", "pred_max", "n_runs"] if c in abs_df.columns]].copy()
    final = final.rename(columns={
        "pred": "pred_abs",
        "pred_mean": "pred_abs_mean",
        "pred_std": "pred_abs_std",
        "pred_min": "pred_abs_min",
        "pred_max": "pred_abs_max",
        "n_runs": "n_runs_abs",
    })
    final["absolute_model"] = abs_name
    if true_col in final.columns:
        final[f"abs_err_{abs_name}"] = (pd.to_numeric(final["pred_abs"], errors="coerce") - pd.to_numeric(final[true_col], errors="coerce")).abs()

    rank_score_cols = []
    for name in rank_names:
        if name not in exp_map:
            print(f"[WARN] ranking exp not found: {name}")
            continue
        r = load_pred_merged(exp_map[name], master_small)[["id", "pred", "pred_std", "n_runs"]].copy()
        # smaller adsorption energy is better for candidate screening, so ascending rank.
        r[f"rank_pct_{name}"] = r["pred"].rank(method="average", ascending=True, pct=True)
        r = r.rename(columns={"pred": f"pred_rank_{name}", "pred_std": f"pred_std_{name}", "n_runs": f"n_runs_{name}"})
        final = final.merge(r, on="id", how="left")
        rank_score_cols.append(f"rank_pct_{name}")
    if rank_score_cols:
        final["ranking_consensus_pct"] = final[rank_score_cols].mean(axis=1)
        final["ranking_consensus_std"] = final[rank_score_cols].std(axis=1)
    else:
        final["ranking_consensus_pct"] = np.nan
        final["ranking_consensus_std"] = np.nan

    # A balanced score: low absolute prediction, strong ranking consensus, and low uncertainty are favored.
    final["score_rank_abs_pred"] = pd.to_numeric(final["pred_abs"], errors="coerce").rank(ascending=True, pct=True)
    final["score_rank_consensus"] = pd.to_numeric(final["ranking_consensus_pct"], errors="coerce").rank(ascending=True, pct=True)
    if "pred_abs_std" in final.columns:
        final["score_rank_uncertainty"] = pd.to_numeric(final["pred_abs_std"], errors="coerce").rank(ascending=True, pct=True)
    else:
        final["score_rank_uncertainty"] = np.nan
    final["final_candidate_score"] = (
        0.50 * final["score_rank_abs_pred"] +
        0.35 * final["score_rank_consensus"] +
        0.15 * final["score_rank_uncertainty"].fillna(final["score_rank_uncertainty"].median())
    )
    final = final.sort_values(["final_candidate_score", "pred_abs"], ascending=[True, True])
    out_dir.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_dir / "final_addhout_candidate_ranking.csv", index=False)
    final.to_excel(out_dir / "final_addhout_candidate_ranking.xlsx", index=False)
    # also save an easy-to-read top table
    show_cols = [c for c in ["id", "material", "element", "dopant", true_col, "target_computed", "pred_abs", "pred_abs_std", "ranking_consensus_pct", "ranking_consensus_std", "final_candidate_score", "absolute_model"] if c in final.columns]
    final[show_cols].head(50).to_csv(out_dir / "final_addhout_candidate_ranking_top50.csv", index=False)
    print(f"[OK] final candidate table -> {out_dir / 'final_addhout_candidate_ranking.csv'}")
    print(f"[INFO] absolute_model = {abs_name}")
    print(f"[INFO] ranking_models  = {','.join(rank_names)}")
    print(final[show_cols].head(30).to_string(index=False))
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-roots", required=True, help="Comma-separated experiment roots, e.g. outputs_addh_graph_ensemble,outputs_addh_graph_ensemble_refine")
    ap.add_argument("--addhout-master-csv", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--true-col", default="h_ads_excel")
    ap.add_argument("--absolute-exp", default="auto", help="Experiment name for absolute-value model; use auto to choose lowest absolute score.")
    ap.add_argument("--ranking-exps", default="auto", help="Comma-separated ranking model names; use auto to choose top by ranking score.")
    ap.add_argument("--top-rank-n", type=int, default=3)
    ap.add_argument("--no-final-candidates", action="store_true")
    args = ap.parse_args()

    grid_roots = [Path(x.strip()) for x in args.grid_roots.split(",") if x.strip()]
    out_dir = Path(args.out_dir) if args.out_dir else grid_roots[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    master_small = load_master(Path(args.addhout_master_csv))
    exp_dirs = find_exp_dirs(grid_roots)
    print(f"[INFO] found experiments = {len(exp_dirs)}")
    if not exp_dirs:
        raise SystemExit("No experiment prediction files found.")
    summary = build_summary(exp_dirs, master_small, args.true_col)
    csv = out_dir / "graph_ensemble_addhout_posterior_summary.csv"
    xlsx = out_dir / "graph_ensemble_addhout_posterior_summary.xlsx"
    summary.to_csv(csv, index=False)
    summary.to_excel(xlsx, index=False)
    print(f"[OK] summary CSV  -> {csv}")
    print(f"[OK] summary XLSX -> {xlsx}")
    show_cols = [
        "exp_name", "feature_mode", "target_abs_max", "pca_dim", "calibration", "aggregate_method",
        "oof_mae", "oof_rmse", "oof_r2",
        f"addhout_{args.true_col}_mae", f"addhout_{args.true_col}_rmse", f"addhout_{args.true_col}_pearson", f"addhout_{args.true_col}_spearman", f"addhout_{args.true_col}_bias",
        "selection_score_absolute", "selection_score_ranking", "selection_score_addhout",
    ]
    show_cols = [c for c in show_cols if c in summary.columns]
    print(summary[show_cols].head(40).to_string(index=False))
    if not args.no_final_candidates:
        abs_name, rank_names = choose_models(summary, args.true_col, args.absolute_exp, args.ranking_exps, args.top_rank_n)
        build_final_candidates(summary, exp_dirs, master_small, args.true_col, abs_name, rank_names, out_dir)


if __name__ == "__main__":
    main()
