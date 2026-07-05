#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_num(x):
    return pd.to_numeric(x, errors='coerce')


def clean_cat(s: pd.Series) -> pd.Series:
    s = s.fillna('').astype(str).str.strip()
    s = s.replace({'nan': '', 'None': '', 'null': '', 'unknown': ''})
    return s.apply(lambda x: x if x else 'unknown')


TARGETS = [
    'H_adsorption_energy_value_eV',
    'H_adsorption_free_energy_value_eV',
    'OH_adsorption_energy_value_eV',
    'OH_adsorption_free_energy_value_eV',
    'H2O_adsorption_energy_value_eV',
    'deprotonation_energy_value_eV',
    'proton_transfer_barrier_eV',
    'Volmer_barrier_eV',
    'water_dissociation_barrier_eV',
    'H_adsorption_energy_value',
    'H_adsorption_free_energy_value',
    'joint_hads_score',
]

GROUP_COLS = [
    'material_class', 'material_name', 'composition', 'surface_facet', 'support', 'dopant', 'interface_type',
    'defect_type', 'model_type', 'ads_site_type', 'active_atom', 'local_geometry', 'vacancy_nearby',
    'hydroxylated_surface', 'oxidation_state', 'charge_transfer_direction', 'DFT_functional',
    'calculation_model', 'adsorption_strength_trend', 'reference_surface',
]


def save_hist_and_box(df: pd.DataFrame, target_col: str, outdir: Path) -> None:
    vals = to_num(df[target_col]).dropna()
    if vals.empty:
        return
    plt.figure(figsize=(7, 5))
    plt.hist(vals, bins=min(30, max(8, int(np.sqrt(len(vals))))))
    plt.xlabel(target_col); plt.ylabel('Count'); plt.title(f'Distribution of {target_col}')
    plt.tight_layout(); plt.savefig(outdir / f'{target_col}_hist.png', dpi=220); plt.close()

    plt.figure(figsize=(4.5, 5))
    plt.boxplot(vals.values, tick_labels=[target_col], showfliers=False)
    plt.ylabel(target_col); plt.title(f'Boxplot of {target_col}')
    plt.tight_layout(); plt.savefig(outdir / f'{target_col}_box.png', dpi=220); plt.close()


def summarize_group_metric(df: pd.DataFrame, group_col: str, target_col: str, min_n: int = 3) -> pd.DataFrame:
    work = df[[group_col, target_col]].copy()
    work[group_col] = clean_cat(work[group_col])
    work[target_col] = to_num(work[target_col])
    work = work[work[target_col].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    stat = work.groupby(group_col, dropna=False)[target_col].agg(
        n='count', mean='mean', median='median', std='std',
        q1=lambda x: np.nanquantile(x, 0.25), q3=lambda x: np.nanquantile(x, 0.75),
        vmin='min', vmax='max'
    ).reset_index()
    global_median = float(work[target_col].median())
    stat['delta_vs_global_median'] = stat['median'] - global_median
    stat['group_col'] = group_col
    stat['target_col'] = target_col
    stat = stat.sort_values(['n', 'median'], ascending=[False, False]).reset_index(drop=True)
    return stat[stat['n'] >= min_n].reset_index(drop=True)


def save_boxplot(df: pd.DataFrame, group_col: str, target_col: str, out_png: Path, topn: int = 12):
    work = df[[group_col, target_col]].copy()
    work[group_col] = clean_cat(work[group_col])
    work[target_col] = to_num(work[target_col])
    work = work[work[target_col].notna()].copy()
    if work.empty:
        return
    top_levels = work[group_col].value_counts().head(topn).index.tolist()
    sub = work[work[group_col].isin(top_levels)].copy()
    if sub.empty:
        return
    ordered = sub.groupby(group_col)[target_col].median().sort_values(ascending=False).index.tolist()
    data = [sub.loc[sub[group_col] == k, target_col].values for k in ordered]
    plt.figure(figsize=(max(8, 0.75 * len(ordered)), 6))
    plt.boxplot(data, tick_labels=ordered, showfliers=False)
    plt.xticks(rotation=45, ha='right'); plt.ylabel(target_col); plt.title(f'{target_col} by {group_col}')
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--min-n', type=int, default=3)
    ap.add_argument('--qc-filter', action='store_true')
    args = ap.parse_args()

    outdir = Path(args.outdir); ensure_dir(outdir)
    df = pd.read_csv(args.csv)
    if args.qc_filter and 'qc_keep_for_model' in df.columns:
        df = df[to_num(df['qc_keep_for_model']).fillna(0).astype(int) == 1].copy()

    summary: List[dict] = []
    for target in TARGETS:
        if target not in df.columns:
            continue
        save_hist_and_box(df, target, outdir)
        vals = to_num(df[target])
        summary.append({
            'target_col': target,
            'n_nonnull': int(vals.notna().sum()),
            'mean': float(vals.mean()) if vals.notna().any() else None,
            'median': float(vals.median()) if vals.notna().any() else None,
            'std': float(vals.std()) if vals.notna().sum() > 1 else None,
        })
        for group_col in GROUP_COLS:
            if group_col not in df.columns:
                continue
            stat = summarize_group_metric(df, group_col, target, min_n=args.min_n)
            if stat.empty:
                continue
            stat.to_csv(outdir / f'{group_col}_effect_{target}.csv', index=False)
            save_boxplot(df, group_col, target, outdir / f'group_box_{group_col}__{target}.png')
            summary.append({
                'group_col': group_col,
                'target_col': target,
                'n_groups': int(len(stat)),
                'top_group': stat.iloc[0][group_col],
                'top_group_median': float(stat.iloc[0]['median']),
            })

    (outdir / 'summary_descriptive.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[DONE] H-ads descriptive stats ->', outdir)


if __name__ == '__main__':
    main()
