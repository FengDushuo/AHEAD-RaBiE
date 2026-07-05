#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step5d: Interpretability report for H adsorption / deprotonation literature-mined database.

Robustness fixes:
- Does not require `record_id` in model_feature_table.csv.
- Uses groupby.size() for counts, so any input table with valid grouping columns works.
- Gracefully skips optional score columns when they are absent.
- Handles missing/empty target columns without raising KeyError.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_top_permutation(predictive_dir: Path, target: str, topn: int = 15) -> pd.DataFrame:
    fp = predictive_dir / f"permutation_importance_top_{target}.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()
    return df.head(topn).copy()


def collect_group_effects(descriptive_dir: Path, target: str, topn: int = 10) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for fp in descriptive_dir.glob(f"*_effect_{target}.csv"):
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if {"group_col", "target_col", "n", "median"}.issubset(df.columns):
            rows.append(df.head(topn).copy())
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["n", "median"], ascending=[False, False]).reset_index(drop=True)
    return out


def collect_interaction_effects(interaction_dir: Path, target: str, topn: int = 10) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for fp in interaction_dir.glob(f"interaction_*__{target}.csv"):
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if {"n", "mean", "median"}.issubset(df.columns):
            df["source_file"] = fp.name
            rows.append(df.sort_values(["n", "median"], ascending=[False, False]).head(topn))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _first_existing(columns: List[str], candidates: List[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def _numeric_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() == 0:
        return np.nan
    return float(vals.mean())


def make_priority_groups(model_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mechanism-relevant groups for DFT design prioritization.

    The previous version used `n=('record_id', 'count')`, which failed when
    model_feature_table.csv did not contain record_id. Here the count is computed
    with groupby.size(), and optional columns are attached only if present.
    """
    if model_df.empty:
        return pd.DataFrame()

    use = model_df.copy()
    columns = list(use.columns)

    dgh_col = _first_existing(columns, ["H_adsorption_free_energy_value_eV", "H_adsorption_free_energy_value"])
    deh_col = _first_existing(columns, ["H_adsorption_energy_value_eV", "H_adsorption_energy_value"])

    if dgh_col:
        use["abs_dG_H"] = pd.to_numeric(use[dgh_col], errors="coerce").abs()
    if deh_col:
        use["abs_dE_H"] = pd.to_numeric(use[deh_col], errors="coerce").abs()

    grp_cols = [
        c for c in [
            "material_class", "surface_facet", "dopant", "defect_type",
            "bridge_structure", "M_O_X_configuration", "ads_site_type",
            "active_atom", "oxidation_state",
        ]
        if c in use.columns
    ]
    if not grp_cols:
        return pd.DataFrame()

    # Avoid over-fragmenting if too many high-cardinality grouping columns are present.
    # Keep all available columns by default, but this still works with missing fields.
    gb = use.groupby(grp_cols, dropna=False)
    out = gb.size().rename("n").reset_index()

    optional_mean_cols = [
        "confidence_score", "joint_hads_score", "DFT_priority_score",
        "abs_dG_H", "abs_dE_H",
        "OH_adsorption_energy_value_eV", "OH_adsorption_free_energy_value_eV",
        "H2O_adsorption_energy_value_eV", "deprotonation_energy_value_eV",
        "Volmer_barrier_eV", "proton_transfer_barrier_eV", "water_dissociation_barrier_eV",
    ]
    for col in optional_mean_cols:
        if col in use.columns:
            tmp = gb[col].apply(_numeric_mean).rename(f"mean_{col}").reset_index()
            out = out.merge(tmp, on=grp_cols, how="left")

    # User-friendly aliases for the most commonly used sort columns.
    if "mean_confidence_score" in out.columns:
        out["mean_confidence"] = out["mean_confidence_score"]
    if "mean_joint_hads_score" in out.columns:
        out["joint_hads_score"] = out["mean_joint_hads_score"]
    if "mean_abs_dG_H" in out.columns:
        out["abs_dG_H"] = out["mean_abs_dG_H"]
    if "mean_abs_dE_H" in out.columns:
        out["abs_dE_H"] = out["mean_abs_dE_H"]

    sort_cols: List[str] = []
    asc: List[bool] = []
    for c in ["mean_DFT_priority_score", "joint_hads_score", "mean_confidence", "n"]:
        if c in out.columns:
            sort_cols.append(c)
            asc.append(False)
    for c in ["abs_dG_H", "abs_dE_H"]:
        if c in out.columns:
            sort_cols.append(c)
            asc.append(True)
            break
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=asc).reset_index(drop=True)
    else:
        out = out.sort_values("n", ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-csv", required=True)
    ap.add_argument("--descriptive-dir", required=True)
    ap.add_argument("--predictive-dir", required=True)
    ap.add_argument("--interaction-dir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    descriptive_dir = Path(args.descriptive_dir)
    predictive_dir = Path(args.predictive_dir)
    interaction_dir = Path(args.interaction_dir)
    model_df = pd.read_csv(args.model_csv)

    targets = [
        "H_adsorption_energy_value_eV",
        "H_adsorption_free_energy_value_eV",
        "OH_adsorption_energy_value_eV",
        "OH_adsorption_free_energy_value_eV",
        "H2O_adsorption_energy_value_eV",
        "deprotonation_energy_value_eV",
        "Volmer_barrier_eV",
        "proton_transfer_barrier_eV",
        "water_dissociation_barrier_eV",
        "joint_hads_score",
    ]
    summary: Dict[str, dict] = {}

    for target in targets:
        perm = load_top_permutation(predictive_dir, target)
        group = collect_group_effects(descriptive_dir, target)
        inter = collect_interaction_effects(interaction_dir, target)
        if not perm.empty:
            perm.to_csv(outdir / f"top_permutation_{target}.csv", index=False)
        if not group.empty:
            group.to_csv(outdir / f"top_group_effects_{target}.csv", index=False)
        if not inter.empty:
            inter.to_csv(outdir / f"top_interactions_{target}.csv", index=False)
        summary[target] = {
            "n_top_permutation": int(len(perm)),
            "n_top_group_rows": int(len(group)),
            "n_top_interaction_rows": int(len(inter)),
            "top_feature": str(perm.iloc[0]["feature"]) if (not perm.empty and "feature" in perm.columns) else None,
        }

    priority = make_priority_groups(model_df)
    if not priority.empty:
        priority.to_csv(outdir / "hads_design_priority_groups.csv", index=False)
        summary["priority_groups"] = {"n_rows": int(len(priority))}
    else:
        summary["priority_groups"] = {"n_rows": 0, "note": "No valid grouping columns or no rows available."}

    md_lines = ["# H-adsorption/deprotonation interpretability report", ""]
    for target, info in summary.items():
        md_lines.append(f"## {target}")
        for k, v in info.items():
            md_lines.append(f"- {k}: {v}")
        md_lines.append("")
    (outdir / "report_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "report_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("[DONE] H-ads interpretability report ->", outdir)


if __name__ == "__main__":
    main()
