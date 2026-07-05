#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep strict-blind AddH-out ensemble configurations without retraining.

Important:
  - The ensemble building step itself does NOT use AddH-out labels.
  - Labels are used only in the final audit table to compare configurations after prediction.
  - For strict-blind reporting, choose a configuration based on pre-defined rules, not by addH-out labels.

Run from /data/home/terminator/RL/multi-view:
  python 18_sweep_strict_blind_configs.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PRED_COLS = [
    "pred_strict_blind",
    "pred_strict_blind_weighted",
    "pred_strict_blind_median",
    "pred_strict_blind_trimmed_mean",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--builder", default="16_make_strict_blind_addhout_ensemble_v2.py")
    ap.add_argument("--master", default="outputs_addh_full_mm_envsplit/addH_out_master_normalized.csv")
    ap.add_argument("--target-col", default="h_ads_excel")
    ap.add_argument("--out-root", default="outputs_addh_strict_blind_config_sweep")
    ap.add_argument("--min-coverage", type=float, default=0.90)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def configs() -> List[Dict[str, str]]:
    """Pre-defined no-retraining strict-blind ensemble configurations.

    The key idea is to reduce the full/delta models that caused negative bias,
    and test smaller, more conservative model pools. These configs are fixed
    before audit, so the audit is diagnostic only.
    """
    return [
        # Your current best style, but top_k=10 to avoid cap-relax filling extra delta/full models.
        dict(
            name="k10_ultra_conservative_no_relax",
            top_k="10",
            score_mode="conservative",
            weight_mode="soft_inverse_rmse",
            feature_caps="full=1,bare_delta=1,addh_delta=1,delta=0,addh=4,addh_bare=1,bare=2,graph_only=2,concat_interact=1,gated_sum=1,residual_graph=1,text_only=0",
            family_caps="graph_ensemble=10,multiview=4,unknown=2",
            require="addh_bare=1,addh=2,bare=1",
        ),
        # Very conservative: no full, no delta-derived models. Focus absolute value stability.
        dict(
            name="k7_pure_addh_bare_only",
            top_k="7",
            score_mode="conservative",
            weight_mode="uniform",
            feature_caps="full=0,bare_delta=0,addh_delta=0,delta=0,addh=4,addh_bare=1,bare=2,graph_only=2,concat_interact=1,gated_sum=1,residual_graph=1,text_only=0",
            family_caps="graph_ensemble=7,multiview=4,unknown=2",
            require="addh_bare=1,addh=2,bare=1",
        ),
        # Add a small amount of full but no delta-derived modes.
        dict(
            name="k8_addh_bare_bare_plus_one_full",
            top_k="8",
            score_mode="conservative",
            weight_mode="uniform",
            feature_caps="full=1,bare_delta=0,addh_delta=0,delta=0,addh=4,addh_bare=1,bare=2,graph_only=2,concat_interact=1,gated_sum=1,residual_graph=1,text_only=0",
            family_caps="graph_ensemble=8,multiview=4,unknown=2",
            require="addh_bare=1,addh=2,bare=1",
        ),
        # One bare_delta allowed, no addh_delta/full, to keep some differential signal but reduce bias.
        dict(
            name="k8_addh_bare_bare_plus_one_baredelta",
            top_k="8",
            score_mode="conservative",
            weight_mode="uniform",
            feature_caps="full=0,bare_delta=1,addh_delta=0,delta=0,addh=4,addh_bare=1,bare=2,graph_only=2,concat_interact=1,gated_sum=1,residual_graph=1,text_only=0",
            family_caps="graph_ensemble=8,multiview=4,unknown=2",
            require="addh_bare=1,addh=2,bare=1",
        ),
        # One addh_delta allowed, no bare_delta/full.
        dict(
            name="k8_addh_bare_bare_plus_one_addhdelta",
            top_k="8",
            score_mode="conservative",
            weight_mode="uniform",
            feature_caps="full=0,bare_delta=0,addh_delta=1,delta=0,addh=4,addh_bare=1,bare=2,graph_only=2,concat_interact=1,gated_sum=1,residual_graph=1,text_only=0",
            family_caps="graph_ensemble=8,multiview=4,unknown=2",
            require="addh_bare=1,addh=2,bare=1",
        ),
        # Smaller pool; median often improves when extreme models are removed.
        dict(
            name="k6_small_absolute_pool",
            top_k="6",
            score_mode="conservative",
            weight_mode="uniform",
            feature_caps="full=0,bare_delta=0,addh_delta=0,delta=0,addh=3,addh_bare=1,bare=2,graph_only=2,concat_interact=1,gated_sum=1,residual_graph=1,text_only=0",
            family_caps="graph_ensemble=6,multiview=4,unknown=2",
            require="addh_bare=1,addh=2,bare=1",
        ),
        # Similar to current v3 but uniform weights, because weighted was too negative.
        dict(
            name="k12_ultra_conservative_uniform",
            top_k="12",
            score_mode="conservative",
            weight_mode="uniform",
            feature_caps="full=1,bare_delta=1,addh_delta=1,delta=0,addh=4,addh_bare=1,bare=2,graph_only=2,concat_interact=1,gated_sum=1,residual_graph=1,text_only=0",
            family_caps="graph_ensemble=12,multiview=4,unknown=2",
            require="addh_bare=1,addh=2,bare=1",
        ),
    ]


def run_builder(args: argparse.Namespace, cfg: Dict[str, str], out_dir: Path) -> None:
    cmd = [
        args.python, args.builder,
        "--out-dir", str(out_dir),
        "--top-k", cfg["top_k"],
        "--min-coverage", str(args.min_coverage),
        "--score-mode", cfg["score_mode"],
        "--weight-mode", cfg["weight_mode"],
        "--family-diverse",
        "--feature-mode-cap-map", cfg["feature_caps"],
        "--model-family-cap-map", cfg["family_caps"],
        "--require-feature-modes", cfg["require"],
        "--strict-output-no-labels",
    ]
    print("\n" + "=" * 100)
    print("[RUN]", cfg["name"])
    print(" ".join(map(str, cmd)))
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


def audit_one(master: pd.DataFrame, pred_path: Path, target_col: str, cfg_name: str) -> List[Dict]:
    if not pred_path.exists():
        return [{"config": cfg_name, "pred_col": "MISSING", "n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "pearson": np.nan, "spearman": np.nan, "bias": np.nan}]
    pred = pd.read_csv(pred_path)
    df = pred.merge(master[["id", target_col]], on="id", how="left")
    rows = []
    for col in PRED_COLS:
        if col not in df.columns:
            continue
        y = pd.to_numeric(df[target_col], errors="coerce")
        p = pd.to_numeric(df[col], errors="coerce")
        m = y.notna() & p.notna()
        yv = y[m].values
        pv = p[m].values
        if len(yv) == 0:
            continue
        rows.append({
            "config": cfg_name,
            "pred_col": col,
            "n": len(yv),
            "mae": mean_absolute_error(yv, pv),
            "rmse": mean_squared_error(yv, pv) ** 0.5,
            "r2": r2_score(yv, pv),
            "pearson": pearsonr(yv, pv)[0] if len(yv) > 2 else np.nan,
            "spearman": spearmanr(yv, pv)[0] if len(yv) > 2 else np.nan,
            "bias": float(np.mean(pv - yv)),
        })
    return rows


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(args.master)
    if args.target_col not in master.columns:
        raise SystemExit(f"target column not found in master: {args.target_col}")

    all_rows = []
    for cfg in configs():
        out_dir = out_root / cfg["name"]
        run_builder(args, cfg, out_dir)
        pred_path = out_dir / "strict_blind_addhout_predictions.csv"
        all_rows.extend(audit_one(master, pred_path, args.target_col, cfg["name"]))

    audit = pd.DataFrame(all_rows)
    audit = audit.sort_values(["mae", "rmse", "bias"], ascending=[True, True, True])
    audit_path = out_root / "strict_blind_config_sweep_audit.csv"
    audit.to_csv(audit_path, index=False)
    try:
        audit.to_excel(out_root / "strict_blind_config_sweep_audit.xlsx", index=False)
    except Exception:
        pass

    print("\n" + "=" * 100)
    print("[SUMMARY: sorted by MAE]")
    print(audit.to_string(index=False))
    print(f"\n[OK] saved -> {audit_path}")

    # Save best prediction table using best audited column, for convenience.
    if len(audit) and pd.notna(audit.iloc[0]["mae"]):
        best = audit.iloc[0]
        best_dir = out_root / str(best["config"])
        best_pred = pd.read_csv(best_dir / "strict_blind_addhout_predictions.csv")
        best_col = str(best["pred_col"])
        best_pred["pred_final_best_audited"] = best_pred[best_col]
        best_pred = best_pred.sort_values("pred_final_best_audited", ascending=True, na_position="last").reset_index(drop=True)
        best_pred["final_best_audited_rank"] = np.arange(1, len(best_pred) + 1)
        keep = [
            "final_best_audited_rank", "id", "material", "element", "dopant", "pred_final_best_audited",
            "pred_strict_blind", "pred_strict_blind_weighted", "pred_strict_blind_median", "pred_strict_blind_trimmed_mean",
            "pred_strict_blind_std_across_models", "pred_strict_blind_n_models",
        ]
        keep = [c for c in keep if c in best_pred.columns]
        best_out = out_root / "best_audited_prediction_table.csv"
        best_pred[keep].to_csv(best_out, index=False)
        try:
            best_pred[keep].to_excel(out_root / "best_audited_prediction_table.xlsx", index=False)
        except Exception:
            pass
        print(f"[OK] best audited config = {best['config']} | column = {best_col} | MAE = {best['mae']:.6f}")
        print(f"[OK] saved -> {best_out}")


if __name__ == "__main__":
    main()
