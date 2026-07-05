#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import List, Tuple

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


TARGETS = ['H_adsorption_energy_value_eV', 'H_adsorption_free_energy_value_eV', 'OH_adsorption_energy_value_eV', 'OH_adsorption_free_energy_value_eV', 'H2O_adsorption_energy_value_eV', 'deprotonation_energy_value_eV', 'proton_transfer_barrier_eV', 'Volmer_barrier_eV', 'water_dissociation_barrier_eV', 'joint_hads_score']
INTERACTIONS: List[Tuple[str, str]] = [
    ('ads_site_type', 'active_atom'),
    ('ads_site_type', 'vacancy_nearby'),
    ('surface_facet', 'dopant'),
    ('material_class', 'defect_type'),
    ('active_atom', 'oxidation_state'),
    ('ads_site_type', 'hydroxylated_surface'),
    ('material_class', 'adsorption_strength_trend'),
    ('DFT_functional', 'calculation_model'),
]


def save_heatmap(pivot: pd.DataFrame, title: str, out_png: Path) -> None:
    if pivot.empty:
        return
    arr = pivot.values.astype(float)
    plt.figure(figsize=(max(6, 0.6 * arr.shape[1]), max(5, 0.4 * arr.shape[0])))
    im = plt.imshow(arr, aspect='auto')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=45, ha='right')
    plt.yticks(range(pivot.shape[0]), pivot.index)
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()


def interaction_table(df: pd.DataFrame, c1: str, c2: str, target: str, min_n: int = 2) -> pd.DataFrame:
    work = df[[c1, c2, target]].copy()
    work[c1] = clean_cat(work[c1]); work[c2] = clean_cat(work[c2]); work[target] = to_num(work[target])
    work = work[work[target].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    stat = work.groupby([c1, c2], dropna=False)[target].agg(n='count', mean='mean', median='median').reset_index()
    stat = stat[stat['n'] >= min_n].copy()
    return stat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--min-n', type=int, default=2)
    ap.add_argument('--qc-filter', action='store_true')
    args = ap.parse_args()

    outdir = Path(args.outdir); ensure_dir(outdir)
    df = pd.read_csv(args.csv)
    if args.qc_filter and 'qc_keep_for_model' in df.columns:
        df = df[to_num(df['qc_keep_for_model']).fillna(0).astype(int) == 1].copy()

    summary = []
    for target in TARGETS:
        if target not in df.columns:
            continue
        for c1, c2 in INTERACTIONS:
            if c1 not in df.columns or c2 not in df.columns:
                continue
            stat = interaction_table(df, c1, c2, target, min_n=args.min_n)
            if stat.empty:
                continue
            base = f'{c1}__{c2}__{target}'
            stat.to_csv(outdir / f'interaction_{base}.csv', index=False)
            pivot_mean = stat.pivot(index=c1, columns=c2, values='mean').fillna(np.nan)
            pivot_median = stat.pivot(index=c1, columns=c2, values='median').fillna(np.nan)
            pivot_n = stat.pivot(index=c1, columns=c2, values='n').fillna(0)
            save_heatmap(pivot_mean, f'Mean {target}: {c1} × {c2}', outdir / f'heatmap_mean_{base}.png')
            save_heatmap(pivot_median, f'Median {target}: {c1} × {c2}', outdir / f'heatmap_median_{base}.png')
            save_heatmap(pivot_n, f'Count: {c1} × {c2}', outdir / f'heatmap_count_{base}.png')
            summary.append({'target_col': target, 'col1': c1, 'col2': c2, 'n_pairs': int(len(stat))})

    (outdir / 'summary_interaction.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[DONE] H-ads interaction effects ->', outdir)


if __name__ == '__main__':
    main()
