#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build paper-ready tables and figures for AddH-out results.

Inputs are the held-out few-shot validation outputs, optional calibration-size
sensitivity outputs, and the prediction/label CSVs. Outputs include compact CSV
tables, a Markdown summary, and PNG figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Make AddH-out paper tables and figures.")
    ap.add_argument("--holdout-dir", default="outputs_addh_fewshot_holdout_validation")
    ap.add_argument("--sensitivity-dir", default="outputs_addh_calibration_fraction_sensitivity")
    ap.add_argument("--pred-csv", default="outputs_addh_bidirectional_chemistry_prior/bidirectional_chemistry_addhout_predictions.csv")
    ap.add_argument("--labels-csv", default="outputs_addh_llm_element_priors/addhout_audit_labels.csv")
    ap.add_argument("--out-dir", default="outputs_addh_paper_artifacts")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--target-col", default="h_ads_excel")
    ap.add_argument("--fewshot-pred-col", default="fewshot_prediction")
    ap.add_argument("--dpi", type=int, default=220)
    return ap.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def metric(y: Sequence[float], p: Sequence[float]) -> Dict[str, float]:
    df = pd.DataFrame({"y": y, "p": p}).dropna()
    if len(df) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "pearson": np.nan, "spearman": np.nan}
    e = df["p"].to_numpy(float) - df["y"].to_numpy(float)
    return {
        "n": int(len(df)),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(np.mean(e)),
        "pearson": float(df["y"].corr(df["p"], method="pearson")) if len(df) >= 3 else np.nan,
        "spearman": float(df["y"].corr(df["p"], method="spearman")) if len(df) >= 3 else np.nan,
    }


def find_row(df: pd.DataFrame, split_name: str, pred_col: str) -> Optional[pd.Series]:
    sub = df[(df["split_name"].astype(str) == split_name) & (df["pred_col"].astype(str) == pred_col)]
    if len(sub) == 0:
        return None
    return sub.iloc[0]


def result_row_from_summary(label: str, summary: pd.DataFrame, split_name: str, pred_col: str, claim: str) -> Dict[str, object]:
    row = find_row(summary, split_name, pred_col)
    if row is None:
        return {"method": label, "claim": claim, "available": False}
    keys = [
        "n_splits",
        "mae_mean",
        "mae_std",
        "mae_median",
        "mae_q05",
        "mae_q95",
        "rmse_mean",
        "rmse_std",
        "pearson_mean",
        "pearson_std",
        "spearman_mean",
        "spearman_std",
    ]
    out: Dict[str, object] = {"method": label, "claim": claim, "available": True}
    for k in keys:
        out[k] = row.get(k, np.nan)
    return out


def result_row_from_reference(label: str, ref: pd.DataFrame, pred_col: str, claim: str) -> Dict[str, object]:
    sub = ref[ref["pred_col"].astype(str) == pred_col]
    if len(sub) == 0:
        return {"method": label, "claim": claim, "available": False}
    row = sub.iloc[0]
    return {
        "method": label,
        "claim": claim,
        "available": True,
        "n_splits": 1,
        "mae_mean": row.get("mae", np.nan),
        "mae_std": np.nan,
        "mae_median": row.get("mae", np.nan),
        "mae_q05": np.nan,
        "mae_q95": np.nan,
        "rmse_mean": row.get("rmse", np.nan),
        "rmse_std": np.nan,
        "pearson_mean": row.get("pearson", np.nan),
        "pearson_std": np.nan,
        "spearman_mean": row.get("spearman", np.nan),
        "spearman_std": np.nan,
    }


def format_pm(mean: object, std: object, ndigits: int = 3) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{float(mean):.{ndigits}f}"
    return f"{float(mean):.{ndigits}f} +/- {float(std):.{ndigits}f}"


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    rows = ["| Method | Claim | MAE | RMSE | Pearson | Spearman |", "|---|---|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        rows.append(
            "| {method} | {claim} | {mae} | {rmse} | {pearson} | {spearman} |".format(
                method=r["method"],
                claim=r["claim"],
                mae=format_pm(r.get("mae_mean"), r.get("mae_std")),
                rmse=format_pm(r.get("rmse_mean"), r.get("rmse_std")),
                pearson=format_pm(r.get("pearson_mean"), r.get("pearson_std")),
                spearman=format_pm(r.get("spearman_mean"), r.get("spearman_std")),
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def plot_parity(df: pd.DataFrame, target_col: str, pred_col: str, out_path: Path, title: str, dpi: int) -> None:
    sub = df[[target_col, pred_col]].dropna()
    if len(sub) == 0:
        return
    m = metric(sub[target_col], sub[pred_col])
    lo = float(np.nanmin([sub[target_col].min(), sub[pred_col].min()]))
    hi = float(np.nanmax([sub[target_col].max(), sub[pred_col].max()]))
    pad = max(0.2, 0.05 * (hi - lo))
    lo -= pad
    hi += pad
    plt.figure(figsize=(5.2, 4.6))
    plt.scatter(sub[target_col], sub[pred_col], s=34, alpha=0.82, edgecolor="white", linewidth=0.5, color="#2d6cdf")
    plt.plot([lo, hi], [lo, hi], color="#333333", linewidth=1.2)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("DFT H adsorption energy (eV)")
    plt.ylabel("Predicted H adsorption energy (eV)")
    plt.title(title)
    txt = f"n={m['n']}\nMAE={m['mae']:.3f}\nRMSE={m['rmse']:.3f}\nSpearman={m['spearman']:.3f}"
    plt.text(0.04, 0.96, txt, transform=plt.gca().transAxes, va="top", ha="left", fontsize=9, bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_mae_distribution(split_metrics: pd.DataFrame, out_path: Path, dpi: int) -> None:
    sub = split_metrics[split_metrics["split_name"].astype(str) == "material_stratified_random"].copy()
    if len(sub) == 0:
        return
    order = [
        ("pred_superblend_final", "Strict superblend", "#777777"),
        ("pred_chem_spike_final", "Chemistry-spike", "#d7822b"),
        ("fewshot_calibrated", "Few-shot calibrated", "#2d6cdf"),
    ]
    plt.figure(figsize=(6.6, 4.5))
    bins = np.linspace(0, max(2.6, sub["mae"].max() * 1.05), 32)
    for col, label, color in order:
        vals = sub.loc[sub["pred_col"] == col, "mae"].dropna()
        if len(vals):
            plt.hist(vals, bins=bins, alpha=0.45, label=label, color=color)
    plt.xlabel("Held-out split MAE (eV)")
    plt.ylabel("Number of splits")
    plt.title("MAE distribution across repeated AddH-out holdouts")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_calibration_curve(sens_summary: pd.DataFrame, out_path: Path, dpi: int) -> None:
    if len(sens_summary) == 0 or "calibration_fraction" not in sens_summary.columns:
        return
    order = [
        ("pred_superblend_final", "Strict superblend", "#777777"),
        ("pred_chem_spike_final", "Chemistry-spike", "#d7822b"),
        ("fewshot_calibrated", "Few-shot calibrated", "#2d6cdf"),
    ]
    plt.figure(figsize=(6.2, 4.4))
    for col, label, color in order:
        sub = sens_summary[sens_summary["pred_col"].astype(str) == col].sort_values("calibration_fraction")
        if len(sub) == 0:
            continue
        x = pd.to_numeric(sub["calibration_fraction"], errors="coerce")
        y = pd.to_numeric(sub["mae_mean"], errors="coerce")
        yerr = pd.to_numeric(sub["mae_std"], errors="coerce")
        plt.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.7, capsize=3, label=label, color=color)
    plt.xlabel("Fraction of AddH-out labels used for calibration")
    plt.ylabel("Held-out MAE (eV)")
    plt.title("Calibration-size sensitivity")
    plt.ylim(bottom=0)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_dopant_error(df: pd.DataFrame, target_col: str, pred_cols: Sequence[Tuple[str, str]], out_path: Path, dpi: int) -> None:
    rows: List[Dict[str, object]] = []
    for col, label in pred_cols:
        if col not in df.columns:
            continue
        tmp = df.dropna(subset=[target_col, col]).copy()
        tmp["abs_err"] = (tmp[col] - tmp[target_col]).abs()
        for dop, sub in tmp.groupby("dopant", dropna=False):
            rows.append({"dopant": str(dop), "method": label, "mae": float(sub["abs_err"].mean())})
    tab = pd.DataFrame(rows)
    if len(tab) == 0:
        return
    order = tab.groupby("dopant")["mae"].mean().sort_values(ascending=False).index.tolist()
    methods = tab["method"].drop_duplicates().tolist()
    x = np.arange(len(order))
    width = 0.8 / max(1, len(methods))
    plt.figure(figsize=(max(7.0, 0.36 * len(order) + 2.0), 4.4))
    colors = ["#777777", "#d7822b", "#2d6cdf"]
    for i, method in enumerate(methods):
        vals = tab[tab["method"] == method].set_index("dopant").reindex(order)["mae"]
        plt.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width=width, label=method, color=colors[i % len(colors)], alpha=0.86)
    plt.xticks(x, order, rotation=45, ha="right")
    plt.ylabel("Mean absolute error (eV)")
    plt.title("Dopant-wise error")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    holdout_dir = Path(args.holdout_dir)
    sensitivity_dir = Path(args.sensitivity_dir)
    summary = read_table(holdout_dir / "fewshot_holdout_summary.csv")
    split_metrics = read_table(holdout_dir / "fewshot_holdout_split_metrics.csv")
    holdout_predictions = read_table(holdout_dir / "fewshot_holdout_predictions.csv")
    full_ref = read_table(holdout_dir / "full_data_reference_metrics.csv")

    main_rows = [
        result_row_from_summary(
            "Strict superblend",
            summary,
            "material_stratified_random",
            "pred_superblend_final",
            "held-out baseline",
        ),
        result_row_from_summary(
            "Chemistry-spike prior",
            summary,
            "material_stratified_random",
            "pred_chem_spike_final",
            "held-out baseline",
        ),
        result_row_from_summary(
            "Few-shot calibrated chemistry prior",
            summary,
            "material_stratified_random",
            "fewshot_calibrated",
            "main held-out result",
        ),
        result_row_from_summary(
            "Few-shot calibrated chemistry prior",
            summary,
            "leave_one_dopant_out",
            "fewshot_calibrated",
            "leave-one-dopant-out",
        ),
        result_row_from_reference(
            "Bidirectional chemistry prior",
            full_ref,
            "pred_bidir_chem_conservative",
            "full-data reference, not held-out",
        ),
        result_row_from_reference(
            "Bidirectional chemistry prior",
            full_ref,
            "pred_bidir_chem_final",
            "post-hoc upper bound, not held-out",
        ),
    ]
    main_table = pd.DataFrame(main_rows)
    main_table.to_csv(table_dir / "addhout_main_results.csv", index=False)
    write_markdown_table(main_table, table_dir / "addhout_main_results.md")

    if (sensitivity_dir / "calibration_fraction_summary.csv").exists():
        sens_summary = read_table(sensitivity_dir / "calibration_fraction_summary.csv")
        sens_summary.to_csv(table_dir / "addhout_calibration_fraction_summary.csv", index=False)
    else:
        sens_summary = pd.DataFrame()

    if (sensitivity_dir / "calibration_fraction_paired_improvement.csv").exists():
        paired = read_table(sensitivity_dir / "calibration_fraction_paired_improvement.csv")
        paired.to_csv(table_dir / "addhout_calibration_fraction_paired_improvement.csv", index=False)

    pred = read_table(Path(args.pred_csv))
    labels = read_table(Path(args.labels_csv))
    data = pred.merge(labels[[args.id_col, args.target_col]], on=args.id_col, how="left")
    plot_parity(data, args.target_col, "pred_superblend_final", fig_dir / "parity_strict_superblend.png", "Strict superblend", args.dpi)
    plot_parity(data, args.target_col, "pred_chem_spike_final", fig_dir / "parity_chemistry_spike.png", "Chemistry-spike prior", args.dpi)

    hp = holdout_predictions[holdout_predictions["split_name"].astype(str) == "material_stratified_random"].copy()
    if len(hp):
        agg = (
            hp.groupby([args.id_col, "material", "dopant"], dropna=False)
            .agg({args.target_col: "first", args.fewshot_pred_col: "median"})
            .reset_index()
            .rename(columns={args.fewshot_pred_col: "fewshot_prediction_median"})
        )
        agg.to_csv(table_dir / "addhout_fewshot_holdout_median_predictions.csv", index=False)
        plot_parity(
            agg,
            args.target_col,
            "fewshot_prediction_median",
            fig_dir / "parity_fewshot_holdout_median.png",
            "Few-shot calibrated holdout median",
            args.dpi,
        )
        data_for_dopant = data.merge(agg[[args.id_col, "fewshot_prediction_median"]], on=args.id_col, how="left")
    else:
        data_for_dopant = data

    plot_mae_distribution(split_metrics, fig_dir / "mae_distribution_repeated_holdout.png", args.dpi)
    plot_calibration_curve(sens_summary, fig_dir / "calibration_fraction_vs_mae.png", args.dpi)
    plot_dopant_error(
        data_for_dopant,
        args.target_col,
        [
            ("pred_superblend_final", "Strict"),
            ("pred_chem_spike_final", "Spike"),
            ("fewshot_prediction_median", "Few-shot"),
        ],
        fig_dir / "dopant_wise_error.png",
        args.dpi,
    )

    manifest = {
        "script": Path(__file__).name,
        "holdout_dir": str(holdout_dir),
        "sensitivity_dir": str(sensitivity_dir),
        "pred_csv": args.pred_csv,
        "labels_csv": args.labels_csv,
        "paper_claim": "main result should use held-out few-shot validation; full-data bidirectional result is post-hoc reference",
        "outputs": {
            "main_results_csv": str(table_dir / "addhout_main_results.csv"),
            "main_results_md": str(table_dir / "addhout_main_results.md"),
            "figures": str(fig_dir),
        },
    }
    (out_dir / "paper_artifacts_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] wrote {out_dir}")
    print(main_table.to_string(index=False))


if __name__ == "__main__":
    main()
