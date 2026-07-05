#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Science-style AddH-out manuscript and supplement figures.

The script consumes the existing AddH-out held-out validation outputs and writes
publication-oriented vector/raster figures. It does not retrain models and it
does not overwrite the earlier paper-artifact figures.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


METHODS = [
    ("pred_superblend_final", "Strict\nsuperblend", "#6f6f6f"),
    ("pred_chem_spike_final", "Chemistry\nprior", "#D55E00"),
    ("fewshot_calibrated", "Few-shot\ncalibrated", "#0072B2"),
]

PRED_METHODS = [
    ("pred_superblend_final", "Strict\nsuperblend", "#6f6f6f"),
    ("pred_chem_spike_final", "Chemistry\nprior", "#D55E00"),
    ("fewshot_prediction_median", "Few-shot\ncalibrated", "#0072B2"),
]

MATERIAL_COLORS = {
    "CeO2": "#009E73",
    "CeO": "#009E73",
    "ZnO": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Make Science-style AddH-out figures.")
    ap.add_argument("--holdout-dir", default="outputs_addh_fewshot_holdout_validation")
    ap.add_argument("--sensitivity-dir", default="outputs_addh_calibration_fraction_sensitivity")
    ap.add_argument("--pred-csv", default="outputs_addh_bidirectional_chemistry_prior/bidirectional_chemistry_addhout_predictions.csv")
    ap.add_argument("--labels-csv", default="outputs_addh_llm_element_priors/addhout_audit_labels.csv")
    ap.add_argument("--out-dir", default="outputs_addh_science_figures")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--target-col", default="h_ads_excel")
    ap.add_argument("--fewshot-pred-col", default="fewshot_prediction")
    ap.add_argument("--training-log-csv", default="", help="Optional CSV with epoch/iteration and train/validation metrics.")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--formats", default="pdf,svg,png", help="Comma-separated output formats.")
    return ap.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def maybe_read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_table(path)


def as_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def finite_corr(y: pd.Series, p: pd.Series, method: str) -> float:
    df = pd.DataFrame({"y": as_num(y), "p": as_num(p)}).dropna()
    if len(df) < 3:
        return np.nan
    if df["y"].nunique() < 2 or df["p"].nunique() < 2:
        return np.nan
    return float(df["y"].corr(df["p"], method=method))


def metrics(y: Sequence[float], p: Sequence[float]) -> Dict[str, float]:
    df = pd.DataFrame({"y": y, "p": p}).apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "pearson": np.nan, "spearman": np.nan, "r2": np.nan}
    err = df["p"].to_numpy(float) - df["y"].to_numpy(float)
    denom = float(np.sum((df["y"].to_numpy(float) - df["y"].mean()) ** 2))
    r2 = np.nan if denom <= 0 else 1.0 - float(np.sum(err**2)) / denom
    return {
        "n": int(len(df)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "pearson": finite_corr(df["y"], df["p"], "pearson"),
        "spearman": finite_corr(df["y"], df["p"], "spearman"),
        "r2": r2,
    }


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.0,
            "patch.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 150,
            "savefig.facecolor": "white",
        }
    )


def save_all(fig: plt.Figure, out_dir: Path, stem: str, formats: Sequence[str], dpi: int) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for fmt in formats:
        fmt = fmt.strip().lower().lstrip(".")
        if not fmt:
            continue
        path = out_dir / f"{stem}.{fmt}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.035}
        if fmt in {"png", "tif", "tiff", "jpg", "jpeg"}:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        written.append(str(path))
    plt.close(fig)
    return written


def panel(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.13, 1.08, letter, transform=ax.transAxes, fontsize=8.0, fontweight="bold", va="top", ha="left")


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def collect_fewshot_median(holdout_predictions: pd.DataFrame, id_col: str, target_col: str, fewshot_col: str) -> pd.DataFrame:
    if len(holdout_predictions) == 0 or fewshot_col not in holdout_predictions.columns:
        return pd.DataFrame()
    hp = holdout_predictions[holdout_predictions["split_name"].astype(str) == "material_stratified_random"].copy()
    if len(hp) == 0:
        hp = holdout_predictions.copy()
    group_cols = [id_col]
    for c in ["material", "dopant"]:
        if c in hp.columns and c not in group_cols:
            group_cols.append(c)
    agg = {target_col: "first", fewshot_col: "median"}
    out = hp.groupby(group_cols, dropna=False).agg(agg).reset_index()
    return out.rename(columns={fewshot_col: "fewshot_prediction_median"})


def merge_prediction_data(
    pred: pd.DataFrame,
    labels: pd.DataFrame,
    fewshot_median: pd.DataFrame,
    id_col: str,
    target_col: str,
) -> pd.DataFrame:
    data = pred.copy()
    if target_col not in data.columns:
        cols = [id_col, target_col]
        for c in ["material", "dopant"]:
            if c in labels.columns and c not in cols:
                cols.append(c)
        data = data.merge(labels[cols].drop_duplicates(id_col), on=id_col, how="left", suffixes=("", "_label"))
    if len(fewshot_median):
        keep = [id_col, "fewshot_prediction_median"]
        data = data.merge(fewshot_median[keep], on=id_col, how="left")
    for c in ["material", "dopant"]:
        alt = f"{c}_label"
        if c not in data.columns and alt in data.columns:
            data[c] = data[alt]
        elif alt in data.columns:
            data[c] = data[c].fillna(data[alt])
    return data


def common_limits(frames: Sequence[Tuple[pd.DataFrame, str, str]]) -> Tuple[float, float]:
    vals: List[float] = []
    for df, target_col, pred_col in frames:
        if len(df) and target_col in df.columns and pred_col in df.columns:
            sub = df[[target_col, pred_col]].apply(pd.to_numeric, errors="coerce").dropna()
            vals.extend(sub[target_col].tolist())
            vals.extend(sub[pred_col].tolist())
    if not vals:
        return -1.0, 1.0
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    pad = max(0.25, 0.045 * (hi - lo))
    return lo - pad, hi + pad


def parity_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    target_col: str,
    pred_col: str,
    label: str,
    fallback_color: str,
    limits: Tuple[float, float],
    show_material_legend: bool = False,
) -> Dict[str, float]:
    cols = [target_col, pred_col]
    if "material" in df.columns:
        cols.append("material")
    sub = df[cols].copy()
    sub[target_col] = as_num(sub[target_col])
    sub[pred_col] = as_num(sub[pred_col])
    sub = sub.dropna(subset=[target_col, pred_col])
    m = metrics(sub[target_col], sub[pred_col]) if len(sub) else metrics([], [])
    lo, hi = limits
    ax.plot([lo, hi], [lo, hi], color="#303030", lw=0.8, zorder=1)
    if len(sub):
        if "material" in sub.columns:
            for material, grp in sub.groupby("material", dropna=False):
                mat = str(material)
                ax.scatter(
                    grp[target_col],
                    grp[pred_col],
                    s=14,
                    color=MATERIAL_COLORS.get(mat, fallback_color),
                    edgecolor="white",
                    linewidth=0.25,
                    alpha=0.90,
                    zorder=2,
                    label=mat,
                )
        else:
            ax.scatter(
                sub[target_col],
                sub[pred_col],
                s=14,
                color=fallback_color,
                edgecolor="white",
                linewidth=0.25,
                alpha=0.90,
                zorder=2,
            )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("DFT H adsorption energy (eV)")
    ax.set_ylabel("Predicted energy (eV)")
    ax.set_title(label.replace("\n", " "), pad=2.0)
    txt = f"MAE={m['mae']:.2f}\nRMSE={m['rmse']:.2f}\nSpearman={m['spearman']:.2f}"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", ha="left", fontsize=6.0)
    if show_material_legend and "material" in sub.columns:
        handles, labels = ax.get_legend_handles_labels()
        unique: Dict[str, object] = {}
        for h, l in zip(handles, labels):
            unique.setdefault(l, h)
        ax.legend(
            unique.values(),
            unique.keys(),
            title="Material",
            title_fontsize=5.8,
            frameon=False,
            loc="lower right",
            handletextpad=0.2,
            borderaxespad=0.2,
        )
    return m


def figure_workflow(out_dir: Path, formats: Sequence[str], dpi: int) -> List[str]:
    fig, ax = plt.subplots(figsize=(7.3, 2.55))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)

    boxes = [
        (0.25, 1.65, 1.45, 0.72, "Source labels\naddH + addH-2", "#E8E8E8"),
        (2.00, 1.65, 1.62, 0.72, "Multimodal\nfeatures", "#DCEAF7"),
        (3.95, 1.65, 1.72, 0.72, "Source-domain\nmodels", "#DCEAF7"),
        (6.00, 1.65, 1.72, 0.72, "Chemistry-guided\nanchor", "#F7E1D5"),
        (7.92, 1.65, 1.78, 0.72, "AddH-out\nheld-out prediction", "#D7EAD8"),
        (6.00, 0.48, 1.72, 0.62, "Few-shot\ncalibration labels", "#F9F0CC"),
        (7.92, 0.48, 1.78, 0.62, "Repeated splits\nMAE, RMSE, ranks", "#F9F0CC"),
    ]
    for x, y, w, h, text, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=color,
            edgecolor="#4a4a4a",
            linewidth=0.7,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.0)

    arrows = [
        ((1.72, 2.01), (1.96, 2.01)),
        ((3.64, 2.01), (3.91, 2.01)),
        ((5.69, 2.01), (5.97, 2.01)),
        ((7.74, 2.01), (7.89, 2.01)),
        ((6.87, 1.63), (6.87, 1.12)),
        ((7.74, 0.79), (7.89, 0.79)),
    ]
    for a, b in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=0.75, color="#333333"))

    ax.text(2.81, 1.25, "composition, local geometry,\ngraph embeddings, LLM priors", ha="center", va="top", fontsize=6.1, color="#555555")
    ax.text(4.82, 1.25, "strict blind baseline\nand ensemble anchors", ha="center", va="top", fontsize=6.1, color="#555555")
    return save_all(fig, out_dir, "fig1_workflow_schematic", formats, dpi)


def figure_parity(data: pd.DataFrame, target_col: str, out_dir: Path, formats: Sequence[str], dpi: int) -> Tuple[List[str], pd.DataFrame]:
    frames = [(data, target_col, c) for c, _, _ in PRED_METHODS if c in data.columns]
    limits = common_limits(frames)
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.45), sharex=True, sharey=True)
    rows: List[Dict[str, object]] = []
    for i, (pred_col, label, color) in enumerate(PRED_METHODS):
        ax = axes[i]
        panel(ax, chr(ord("A") + i))
        if pred_col in data.columns:
            m = parity_panel(ax, data, target_col, pred_col, label, color, limits, show_material_legend=(i == 2))
            m.update({"figure": "fig2_parity_comparison", "pred_col": pred_col, "method": label.replace("\n", " ")})
            rows.append(m)
        clean_axis(ax)
        if i > 0:
            ax.set_ylabel("")
    fig.subplots_adjust(wspace=0.20, left=0.07, right=0.99, bottom=0.20, top=0.88)
    return save_all(fig, out_dir, "fig2_addhout_parity_comparison", formats, dpi), pd.DataFrame(rows)


def plot_curve_band(ax: plt.Axes, sens: pd.DataFrame, y_col: str, q05_col: str, q95_col: str, ylabel: str) -> None:
    for pred_col, label, color in METHODS:
        sub = sens[sens["pred_col"].astype(str) == pred_col].copy()
        if len(sub) == 0:
            continue
        sub["calibration_fraction"] = as_num(sub["calibration_fraction"])
        sub = sub.sort_values("calibration_fraction")
        x = as_num(sub["calibration_fraction"]).to_numpy(float)
        y = as_num(sub[y_col]).to_numpy(float)
        ax.plot(x, y, marker="o", ms=3.0, color=color, label=label.replace("\n", " "))
        if q05_col in sub.columns and q95_col in sub.columns:
            q05 = as_num(sub[q05_col]).to_numpy(float)
            q95 = as_num(sub[q95_col]).to_numpy(float)
            ax.fill_between(x, q05, q95, color=color, alpha=0.12, linewidth=0)
    ax.set_xlabel("AddH-out labels used for calibration")
    ax.set_ylabel(ylabel)
    clean_axis(ax)


def figure_calibration(
    sens: pd.DataFrame,
    paired: pd.DataFrame,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[str]:
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.35))
    for ax, letter in zip(axes, "ABC"):
        panel(ax, letter)

    plot_curve_band(axes[0], sens, "mae_mean", "mae_q05", "mae_q95", "Held-out MAE (eV)")
    axes[0].set_ylim(bottom=0)

    plot_curve_band(axes[1], sens, "spearman_mean", "spearman_q05", "spearman_q95", "Spearman rank corr.")
    axes[1].set_ylim(0.35, 1.03)

    if len(paired) and "baseline_col" in paired.columns:
        sub = paired[paired["baseline_col"].astype(str) == "pred_chem_spike_final"].copy()
        if len(sub):
            sub["calibration_fraction"] = as_num(sub["calibration_fraction"])
            sub = sub.sort_values("calibration_fraction")
            x = as_num(sub["calibration_fraction"]).to_numpy(float)
            y = as_num(sub["mean_mae_reduction"]).to_numpy(float)
            axes[2].axhline(0, color="#333333", lw=0.65)
            axes[2].plot(x, y, marker="o", ms=3.0, color="#009E73")
            if "q05_mae_reduction" in sub.columns and "q95_mae_reduction" in sub.columns:
                q05 = as_num(sub["q05_mae_reduction"]).to_numpy(float)
                q95 = as_num(sub["q95_mae_reduction"]).to_numpy(float)
                axes[2].fill_between(x, q05, q95, color="#009E73", alpha=0.14, linewidth=0)
            if "win_rate" in sub.columns:
                for xx, yy, wr in zip(x, y, as_num(sub["win_rate"])):
                    key_fraction = np.isclose(xx, [0.10, 0.20, 0.30, 0.50, 0.80]).any()
                    if np.isfinite(wr) and key_fraction:
                        axes[2].text(xx, yy + 0.035, f"{wr:.0%}", ha="center", va="bottom", fontsize=5.4)
    axes[2].set_xlabel("AddH-out labels used for calibration")
    axes[2].set_ylabel("MAE reduction vs\nchemistry prior (eV)")
    clean_axis(axes[2])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.50, 1.03))
    fig.subplots_adjust(wspace=0.48, left=0.08, right=0.99, bottom=0.23, top=0.78)
    return save_all(fig, out_dir, "fig3_calibration_sensitivity", formats, dpi)


def split_metric_long(holdout_predictions: pd.DataFrame, id_col: str, target_col: str, fewshot_col: str) -> pd.DataFrame:
    if len(holdout_predictions) == 0:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    pred_cols = [
        ("pred_superblend_final", "Strict superblend"),
        ("pred_chem_spike_final", "Chemistry prior"),
        (fewshot_col, "Few-shot calibrated"),
    ]
    for (split_name, split_id), group in holdout_predictions.groupby(["split_name", "split_id"], dropna=False):
        for pred_col, label in pred_cols:
            if pred_col not in group.columns:
                continue
            m = metrics(group[target_col], group[pred_col])
            m.update({"split_name": split_name, "split_id": split_id, "pred_col": pred_col, "method": label})
            rows.append(m)
    return pd.DataFrame(rows)


def dopant_error_table(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if "dopant" not in data.columns:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for pred_col, label, _ in PRED_METHODS:
        if pred_col not in data.columns:
            continue
        tmp = data[["dopant", target_col, pred_col]].copy()
        tmp[target_col] = as_num(tmp[target_col])
        tmp[pred_col] = as_num(tmp[pred_col])
        tmp = tmp.dropna()
        if len(tmp) == 0:
            continue
        tmp["abs_error"] = (tmp[pred_col] - tmp[target_col]).abs()
        for dopant, sub in tmp.groupby("dopant", dropna=False):
            rows.append({"dopant": str(dopant), "pred_col": pred_col, "method": label.replace("\n", " "), "mae": float(sub["abs_error"].mean()), "n": int(len(sub))})
    return pd.DataFrame(rows)


def figure_error_diagnostics(
    split_metrics: pd.DataFrame,
    dopant_errors: pd.DataFrame,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[str]:
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.65), gridspec_kw={"width_ratios": [0.95, 1.45]})
    for ax, letter in zip(axes, "AB"):
        panel(ax, letter)

    sub = split_metrics[split_metrics["split_name"].astype(str) == "material_stratified_random"].copy()
    positions = np.arange(1, len(METHODS) + 1)
    vals: List[np.ndarray] = []
    labels: List[str] = []
    colors: List[str] = []
    for pred_col, label, color in METHODS:
        v = as_num(sub.loc[sub["pred_col"].astype(str) == pred_col, "mae"]).dropna().to_numpy(float)
        if len(v):
            vals.append(v)
            labels.append(label.replace("\n", " "))
            colors.append(color)
    if vals:
        parts = axes[0].violinplot(vals, positions=positions[: len(vals)], widths=0.75, showmeans=False, showextrema=False, showmedians=False)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.18)
            body.set_edgecolor(color)
            body.set_linewidth(0.5)
        bp = axes[0].boxplot(vals, positions=positions[: len(vals)], widths=0.36, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor("white")
            patch.set_edgecolor(color)
        for k in ["whiskers", "caps", "medians"]:
            for item in bp[k]:
                item.set_color("#333333")
                item.set_linewidth(0.65)
        axes[0].set_xticks(positions[: len(vals)], labels, rotation=25, ha="right")
    axes[0].set_ylabel("Held-out split MAE (eV)")
    axes[0].set_ylim(bottom=0)
    clean_axis(axes[0])

    if len(dopant_errors):
        order = (
            dopant_errors[dopant_errors["pred_col"] == "fewshot_prediction_median"]
            .sort_values("mae", ascending=True)["dopant"]
            .tolist()
        )
        if not order:
            order = dopant_errors.groupby("dopant")["mae"].mean().sort_values().index.tolist()
        ybase = np.arange(len(order))
        offsets = {"pred_superblend_final": -0.18, "pred_chem_spike_final": 0.0, "fewshot_prediction_median": 0.18}
        for pred_col, label, color in PRED_METHODS:
            subm = dopant_errors[dopant_errors["pred_col"] == pred_col].set_index("dopant").reindex(order)
            if len(subm) == 0:
                continue
            axes[1].scatter(subm["mae"], ybase + offsets[pred_col], s=13, color=color, label=label.replace("\n", " "), alpha=0.86)
        axes[1].set_yticks(ybase, order)
        axes[1].set_xlabel("Dopant-wise MAE (eV)")
        axes[1].set_ylim(-0.6, len(order) - 0.4)
        axes[1].legend(frameon=False, loc="lower right", ncol=1)
    clean_axis(axes[1])

    fig.subplots_adjust(wspace=0.43, left=0.08, right=0.99, bottom=0.28, top=0.91)
    return save_all(fig, out_dir, "fig4_error_robustness", formats, dpi)


def infer_metric_columns(df: pd.DataFrame) -> Tuple[Optional[str], List[Tuple[str, str]]]:
    lower = {c.lower(): c for c in df.columns}
    x_col = None
    for cand in ["epoch", "epochs", "iter", "iteration", "step", "round"]:
        if cand in lower:
            x_col = lower[cand]
            break
    metrics_found: List[Tuple[str, str]] = []
    candidates = [
        ("train_loss", "train loss"),
        ("training_loss", "train loss"),
        ("loss_train", "train loss"),
        ("val_loss", "validation loss"),
        ("valid_loss", "validation loss"),
        ("validation_loss", "validation loss"),
        ("eval_loss", "validation loss"),
        ("train_mae", "train MAE"),
        ("val_mae", "validation MAE"),
        ("valid_mae", "validation MAE"),
        ("train_rmse", "train RMSE"),
        ("val_rmse", "validation RMSE"),
        ("valid_rmse", "validation RMSE"),
        ("r2", "R2"),
        ("val_r2", "validation R2"),
        ("valid_r2", "validation R2"),
    ]
    for key, label in candidates:
        if key in lower:
            metrics_found.append((lower[key], label))
    return x_col, metrics_found


def figure_training_curves(training_log_csv: str, out_dir: Path, formats: Sequence[str], dpi: int) -> Tuple[List[str], str]:
    if not training_log_csv:
        return [], "skipped: no --training-log-csv provided"
    path = Path(training_log_csv)
    if not path.exists():
        return [], f"skipped: training log not found: {path}"
    df = read_table(path)
    x_col, metric_cols = infer_metric_columns(df)
    if not metric_cols:
        return [], f"skipped: no recognized metric columns in {path}"
    if x_col is None:
        df = df.copy()
        x_col = "_row"
        df[x_col] = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    for col, label in metric_cols[:6]:
        x = as_num(df[x_col])
        y = as_num(df[col])
        good = x.notna() & y.notna()
        if good.sum() == 0:
            continue
        ax.plot(x[good], y[good], marker="", lw=0.9, label=label)
    ax.set_xlabel(x_col)
    ax.set_ylabel("Metric value")
    ax.legend(frameon=False, loc="best")
    clean_axis(ax)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.20, top=0.92)
    return save_all(fig, out_dir, "figS1_training_validation_curves", formats, dpi), f"used: {path}"


def figure_r2_distribution(metric_long: pd.DataFrame, out_dir: Path, formats: Sequence[str], dpi: int) -> List[str]:
    if len(metric_long) == 0 or "r2" not in metric_long.columns:
        return []
    sub = metric_long[metric_long["split_name"].astype(str) == "material_stratified_random"].copy()
    if len(sub) == 0:
        return []
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    vals: List[np.ndarray] = []
    labels: List[str] = []
    colors: List[str] = []
    for pred_col, label, color in METHODS:
        key = "fewshot_prediction" if pred_col == "fewshot_calibrated" else pred_col
        v = as_num(sub.loc[sub["pred_col"].astype(str) == key, "r2"]).dropna().to_numpy(float)
        if len(v):
            vals.append(v)
            labels.append(label.replace("\n", " "))
            colors.append(color)
    if not vals:
        plt.close(fig)
        return []
    bp = ax.boxplot(vals, patch_artist=True, showfliers=False, widths=0.55)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.20)
        patch.set_edgecolor(color)
    for k in ["whiskers", "caps", "medians"]:
        for item in bp[k]:
            item.set_color("#333333")
            item.set_linewidth(0.65)
    ax.axhline(0, color="#333333", lw=0.6, ls=":")
    ax.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=25, ha="right")
    ax.set_ylabel("Held-out split R2")
    clean_axis(ax)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.31, top=0.94)
    return save_all(fig, out_dir, "figS2_r2_distribution", formats, dpi)


def summarize_metric_long(metric_long: pd.DataFrame) -> pd.DataFrame:
    if len(metric_long) == 0:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for keys, group in metric_long.groupby(["split_name", "pred_col", "method"], dropna=False):
        split_name, pred_col, method = keys
        row: Dict[str, object] = {"split_name": split_name, "pred_col": pred_col, "method": method, "n_splits": int(group["split_id"].nunique())}
        for c in ["mae", "rmse", "pearson", "spearman", "r2"]:
            vals = as_num(group[c]).dropna()
            row[f"{c}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{c}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            row[f"{c}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{c}_q05"] = float(vals.quantile(0.05)) if len(vals) else np.nan
            row[f"{c}_q95"] = float(vals.quantile(0.95)) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    setup_style()
    formats = [x.strip().lower().lstrip(".") for x in args.formats.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    fig_main_dir = out_dir / "main_figures"
    fig_supp_dir = out_dir / "supplementary_figures"
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    holdout_dir = Path(args.holdout_dir)
    sensitivity_dir = Path(args.sensitivity_dir)
    pred_csv = Path(args.pred_csv)
    labels_csv = Path(args.labels_csv)

    holdout_summary = read_table(holdout_dir / "fewshot_holdout_summary.csv")
    split_metrics = read_table(holdout_dir / "fewshot_holdout_split_metrics.csv")
    holdout_predictions = read_table(holdout_dir / "fewshot_holdout_predictions.csv")
    sens = maybe_read_table(sensitivity_dir / "calibration_fraction_summary.csv")
    paired = maybe_read_table(sensitivity_dir / "calibration_fraction_paired_improvement.csv")
    pred = read_table(pred_csv)
    labels = read_table(labels_csv)

    fewshot_median = collect_fewshot_median(holdout_predictions, args.id_col, args.target_col, args.fewshot_pred_col)
    if len(fewshot_median):
        fewshot_median.to_csv(table_dir / "science_fewshot_holdout_median_predictions.csv", index=False)
    data = merge_prediction_data(pred, labels, fewshot_median, args.id_col, args.target_col)
    data.to_csv(table_dir / "science_addhout_plotting_data.csv", index=False)

    written: Dict[str, List[str]] = {}
    written["fig1_workflow_schematic"] = figure_workflow(fig_main_dir, formats, args.dpi)

    files, parity_metrics = figure_parity(data, args.target_col, fig_main_dir, formats, args.dpi)
    written["fig2_addhout_parity_comparison"] = files
    if len(parity_metrics):
        parity_metrics.to_csv(table_dir / "science_parity_metrics.csv", index=False)

    if len(sens):
        written["fig3_calibration_sensitivity"] = figure_calibration(sens, paired, fig_main_dir, formats, args.dpi)

    dopant_errors = dopant_error_table(data, args.target_col)
    if len(dopant_errors):
        dopant_errors.to_csv(table_dir / "science_dopant_error_summary.csv", index=False)
    written["fig4_error_robustness"] = figure_error_diagnostics(split_metrics, dopant_errors, fig_main_dir, formats, args.dpi)

    metric_long = split_metric_long(holdout_predictions, args.id_col, args.target_col, args.fewshot_pred_col)
    if len(metric_long):
        metric_long.to_csv(table_dir / "science_repeated_holdout_metrics_with_r2.csv", index=False)
        metric_summary = summarize_metric_long(metric_long)
        metric_summary.to_csv(table_dir / "science_repeated_holdout_metric_summary_with_r2.csv", index=False)
        written["figS2_r2_distribution"] = figure_r2_distribution(metric_long, fig_supp_dir, formats, args.dpi)

    supp_files, training_note = figure_training_curves(args.training_log_csv, fig_supp_dir, formats, args.dpi)
    if supp_files:
        written["figS1_training_validation_curves"] = supp_files

    manifest = {
        "script": Path(__file__).name,
        "style": {
            "target_journal": "Science-family manuscript figures",
            "figure_widths_in": {
                "single_column": 3.5,
                "two_column": 7.3,
            },
            "export_formats": formats,
            "png_dpi": args.dpi,
            "vector_formats": [fmt for fmt in formats if fmt in {"pdf", "svg", "eps"}],
            "note": "Use held-out few-shot figures as main claims. Treat post-hoc bidirectional chemistry results as supplementary/reference only.",
        },
        "inputs": {
            "holdout_dir": str(holdout_dir),
            "sensitivity_dir": str(sensitivity_dir),
            "pred_csv": str(pred_csv),
            "labels_csv": str(labels_csv),
            "training_log_csv": args.training_log_csv,
            "training_curve_status": training_note,
        },
        "outputs": written,
    }
    (out_dir / "science_figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Science-style figures written to {out_dir}")
    print(json.dumps(written, indent=2, ensure_ascii=False))
    if len(parity_metrics):
        cols = ["method", "n", "mae", "rmse", "spearman", "r2"]
        print("[PARITY METRICS]")
        print(parity_metrics[cols].to_string(index=False))
    print(f"[TRAINING CURVES] {training_note}")


if __name__ == "__main__":
    main()
