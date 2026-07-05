#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create material-separated filtered Fig. 2 parity plots."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_CSV = Path("outputs_addh_science_figures_latest/tables/science_addhout_plotting_data.csv")
OUT_DIR = Path("outputs_addh_fig2_material_filtered")
TARGET_COL = "h_ads_excel"

METHODS = [
    ("pred_superblend_final", "Strict superblend", "#6F6F6F"),
    ("pred_chem_spike_final", "Chemistry prior", "#D55E00"),
    ("fewshot_prediction_median", "Few-shot calibrated", "#0072B2"),
]

MATERIAL_CONFIGS = [
    {
        "material": "CeO2",
        "display": r"CeO$_2$",
        "exclude": {"Hg"},
        "color": "#009E73",
        "stem": "fig2_ceo2_without_hg",
        "title": r"CeO$_2$ AddH-out prediction (Hg excluded)",
    },
    {
        "material": "ZnO",
        "display": "ZnO",
        "exclude": {"Ce", "Hg"},
        "color": "#CC79A7",
        "stem": "fig2_zno_without_ce_hg",
        "title": "ZnO AddH-out prediction (Ce and Hg excluded)",
    },
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.4,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def metric(y: Sequence[float], p: Sequence[float]) -> Dict[str, float]:
    df = pd.DataFrame({"y": y, "p": p}).apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) == 0:
        return {"mae": np.nan, "rmse": np.nan, "bias": np.nan, "pearson": np.nan, "spearman": np.nan}
    e = df["p"].to_numpy(float) - df["y"].to_numpy(float)
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(np.mean(e)),
        "pearson": float(df["y"].corr(df["p"], method="pearson")) if len(df) >= 3 else np.nan,
        "spearman": float(df["y"].corr(df["p"], method="spearman")) if len(df) >= 3 else np.nan,
    }


def common_limits(df: pd.DataFrame) -> Tuple[float, float]:
    vals: List[float] = []
    for pred_col, _, _ in METHODS:
        if pred_col in df.columns:
            sub = df[[TARGET_COL, pred_col]].apply(pd.to_numeric, errors="coerce").dropna()
            vals.extend(sub[TARGET_COL].tolist())
            vals.extend(sub[pred_col].tolist())
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    pad = max(0.25, 0.055 * (hi - lo))
    return lo - pad, hi + pad


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def parity_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    pred_col: str,
    method_name: str,
    point_color: str,
    limits: Tuple[float, float],
) -> Dict[str, float]:
    sub = df[[TARGET_COL, pred_col]].apply(pd.to_numeric, errors="coerce").dropna()
    m = metric(sub[TARGET_COL], sub[pred_col])
    lo, hi = limits
    ax.plot([lo, hi], [lo, hi], color="#303030", lw=0.8, zorder=1)
    ax.scatter(
        sub[TARGET_COL],
        sub[pred_col],
        s=22,
        color=point_color,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.92,
        zorder=2,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(method_name, pad=2.5)
    ax.set_xlabel("DFT H adsorption energy (eV)")
    ax.set_ylabel("Predicted energy (eV)")
    ax.text(
        0.05,
        0.95,
        f"MAE={m['mae']:.2f}\nRMSE={m['rmse']:.2f}\nSpearman={m['spearman']:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.3,
        color="#111111",
    )
    clean_axis(ax)
    return m


def make_material_figure(df_all: pd.DataFrame, cfg: Dict[str, object]) -> pd.DataFrame:
    material = str(cfg["material"])
    exclude = {str(x) for x in cfg["exclude"]}
    sub = df_all[(df_all["material"].astype(str) == material) & (~df_all["dopant"].astype(str).isin(exclude))].copy()
    sub = sub.sort_values("dopant")
    sub.to_csv(OUT_DIR / f"{cfg['stem']}_plotting_data.csv", index=False)

    limits = common_limits(sub)
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.48), sharex=True, sharey=True)
    rows: List[Dict[str, object]] = []
    for i, (pred_col, method_name, _) in enumerate(METHODS):
        ax = axes[i]
        ax.text(-0.13, 1.08, chr(ord("A") + i), transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="top", ha="left")
        m = parity_panel(ax, sub, pred_col, method_name, str(cfg["color"]), limits)
        m.update(
            {
                "material": material,
                "excluded_dopants": ",".join(sorted(exclude)),
                "pred_col": pred_col,
                "method": method_name,
                "n_points_after_filter": int(len(sub)),
            }
        )
        rows.append(m)
        if i > 0:
            ax.set_ylabel("")

    fig.suptitle(str(cfg["title"]), fontsize=8.6, fontweight="bold", y=1.02)
    fig.subplots_adjust(wspace=0.20, left=0.07, right=0.99, bottom=0.20, top=0.83)
    for ext in ["pdf", "svg", "png"]:
        path = OUT_DIR / f"{cfg['stem']}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.035)
        else:
            fig.savefig(path, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    return pd.DataFrame(rows)


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    all_metrics = []
    for cfg in MATERIAL_CONFIGS:
        all_metrics.append(make_material_figure(df, cfg))
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(OUT_DIR / "fig2_material_filtered_metrics.csv", index=False)
    print(f"[OK] wrote {OUT_DIR}")
    print(metrics[["material", "method", "n_points_after_filter", "mae", "rmse", "spearman"]].to_string(index=False))


if __name__ == "__main__":
    main()
