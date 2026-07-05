#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rank/trend-focused post-processing for AddH-out predictions.

This script is label-free by default. It is meant for the case where the best
MAE prediction has a useful value distribution, but another strict-blind
prediction has better within-material trend/order.

Default:
  - value distribution: pred_pretrained_delta_final
  - trend/order score: pred_existing_anchor
  - group: material

The quantile method preserves the sorted values from the value column inside
each material, and assigns them to rows ordered by the score column. This can
improve trend consistency without using addH-out labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Trend/rank calibration for AddH-out predictions.")
    ap.add_argument(
        "--pred-csv",
        default="outputs_addh_pretrained_delta_head/pretrained_delta_head_addhout_predictions.csv",
    )
    ap.add_argument("--out-dir", default="outputs_addh_rank_trend_calibrated")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--group-col", default="material")
    ap.add_argument("--value-col", default="pred_pretrained_delta_final")
    ap.add_argument("--score-col", default="pred_existing_anchor")
    ap.add_argument("--fallback-col", default="pred_pretrained_delta_final")
    ap.add_argument(
        "--final-method",
        choices=["quantile", "rankz", "blend"],
        default="quantile",
        help="quantile is the best balanced default; rankz prioritizes monotonic trend.",
    )
    ap.add_argument("--blend-quantile", type=float, default=0.70)
    ap.add_argument("--blend-rankz", type=float, default=0.30)
    ap.add_argument("--audit-labels-csv", default="")
    ap.add_argument("--audit-target-col", default="h_ads_excel")
    return ap.parse_args()


def finite_corr(x: Sequence[float], y: Sequence[float], spearman: bool = False) -> float:
    a = pd.to_numeric(pd.Series(x), errors="coerce")
    b = pd.to_numeric(pd.Series(y), errors="coerce")
    m = a.notna() & b.notna()
    if int(m.sum()) < 3:
        return float("nan")
    av = a[m].to_numpy(float)
    bv = b[m].to_numpy(float)
    if spearman:
        av = pd.Series(av).rank(method="average").to_numpy(float)
        bv = pd.Series(bv).rank(method="average").to_numpy(float)
    if np.nanstd(av) <= 1e-12 or np.nanstd(bv) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y = pd.to_numeric(pd.Series(y_true), errors="coerce")
    p = pd.to_numeric(pd.Series(y_pred), errors="coerce")
    m = y.notna() & p.notna()
    if int(m.sum()) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "pearson": np.nan, "spearman": np.nan}
    yy = y[m].to_numpy(float)
    pp = p[m].to_numpy(float)
    e = pp - yy
    return {
        "n": int(len(yy)),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(np.mean(e)),
        "pearson": finite_corr(yy, pp, False),
        "spearman": finite_corr(yy, pp, True),
    }


def metric_rows(df: pd.DataFrame, pred_cols: Iterable[str], target_col: str, group_col: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    groups: List[Optional[str]] = [None]
    if group_col in df.columns:
        groups.extend(sorted([str(x) for x in df[group_col].dropna().unique()]))
    for c in pred_cols:
        if c not in df.columns:
            continue
        for g in groups:
            d = df if g is None else df[df[group_col].astype(str) == g]
            row = {"pred_col": c, "target_col": target_col, group_col: g}
            row.update(metrics(d[target_col], d[c]))
            rows.append(row)
    return pd.DataFrame(rows)


def quantile_by_group(df: pd.DataFrame, value_col: str, score_col: str, group_col: str, fallback_col: str) -> pd.Series:
    out = pd.to_numeric(df[fallback_col], errors="coerce").copy()
    for _, g in df.groupby(group_col, dropna=False):
        idx = g.index
        values = pd.to_numeric(g[value_col], errors="coerce")
        scores = pd.to_numeric(g[score_col], errors="coerce")
        ok = values.notna() & scores.notna()
        if int(ok.sum()) < 3:
            continue
        sorted_values = np.sort(values[ok].to_numpy(float))
        ordered_idx = scores[ok].sort_values(kind="mergesort").index
        out.loc[ordered_idx] = sorted_values
    return out


def rankz_by_group(df: pd.DataFrame, value_col: str, score_col: str, group_col: str, fallback_col: str) -> pd.Series:
    out = pd.to_numeric(df[fallback_col], errors="coerce").copy()
    for _, g in df.groupby(group_col, dropna=False):
        idx = g.index
        values = pd.to_numeric(g[value_col], errors="coerce")
        scores = pd.to_numeric(g[score_col], errors="coerce")
        ok = values.notna() & scores.notna()
        if int(ok.sum()) < 3:
            continue
        ranks = scores[ok].rank(method="average").to_numpy(float)
        z = (ranks - ranks.mean()) / (ranks.std() if ranks.std() > 1e-12 else 1.0)
        v = values[ok].to_numpy(float)
        out.loc[scores[ok].index] = float(np.mean(v)) + float(np.std(v)) * z
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(args.pred_csv)
    for c in [args.id_col, args.group_col, args.value_col, args.score_col, args.fallback_col]:
        if c not in pred.columns:
            raise SystemExit(f"[ERROR] missing required column {c!r} in {args.pred_csv}")

    pred["pred_trend_quantile_by_group"] = quantile_by_group(
        pred, args.value_col, args.score_col, args.group_col, args.fallback_col
    )
    pred["pred_trend_rankz_by_group"] = rankz_by_group(
        pred, args.value_col, args.score_col, args.group_col, args.fallback_col
    )
    if args.final_method == "rankz":
        final = pred["pred_trend_rankz_by_group"]
    elif args.final_method == "blend":
        q = pd.to_numeric(pred["pred_trend_quantile_by_group"], errors="coerce")
        r = pd.to_numeric(pred["pred_trend_rankz_by_group"], errors="coerce")
        den = args.blend_quantile + args.blend_rankz
        final = (args.blend_quantile * q + args.blend_rankz * r) / max(den, 1e-12)
    else:
        final = pred["pred_trend_quantile_by_group"]
    pred["pred_rank_trend_calibrated"] = final
    pred["rank_trend_calibrated_rank"] = pd.Series(final).rank(method="average", ascending=True).to_numpy()
    pred["rank_trend_score_col"] = args.score_col
    pred["rank_trend_value_col"] = args.value_col
    pred["rank_trend_final_method"] = args.final_method
    pred = pred.sort_values(["rank_trend_calibrated_rank", args.id_col], na_position="last").reset_index(drop=True)

    pred_csv = out_dir / "rank_trend_calibrated_addhout_predictions.csv"
    pred.to_csv(pred_csv, index=False)
    try:
        pred.to_excel(out_dir / "rank_trend_calibrated_addhout_predictions.xlsx", index=False)
    except Exception:
        pass

    audit = pd.DataFrame()
    if args.audit_labels_csv:
        labels_path = Path(args.audit_labels_csv)
        if labels_path.exists():
            labels = pd.read_csv(labels_path)
            if args.id_col in labels.columns and args.audit_target_col in labels.columns:
                keep = [c for c in [args.id_col, args.audit_target_col, args.group_col, "dopant"] if c in labels.columns]
                detail = pred.merge(labels[keep].drop_duplicates(args.id_col), on=args.id_col, how="left", suffixes=("", "_audit"))
                audit_cols = [
                    "pred_rank_trend_calibrated",
                    "pred_trend_quantile_by_group",
                    "pred_trend_rankz_by_group",
                    args.value_col,
                    args.score_col,
                ]
                audit = metric_rows(detail, audit_cols, args.audit_target_col, args.group_col)
                audit.to_csv(out_dir / "rank_trend_posthoc_audit.csv", index=False)
                if "pred_rank_trend_calibrated" in detail.columns:
                    detail["err_rank_trend_calibrated"] = (
                        pd.to_numeric(detail["pred_rank_trend_calibrated"], errors="coerce")
                        - pd.to_numeric(detail[args.audit_target_col], errors="coerce")
                    )
                    detail["abs_err_rank_trend_calibrated"] = detail["err_rank_trend_calibrated"].abs()
                    detail = detail.sort_values("abs_err_rank_trend_calibrated", ascending=False)
                detail.to_csv(out_dir / "rank_trend_posthoc_audit_detail.csv", index=False)
                try:
                    audit.to_excel(out_dir / "rank_trend_posthoc_audit.xlsx", index=False)
                except Exception:
                    pass
                print("[POSTHOC AUDIT ONLY]")
                print(audit.to_string(index=False))

    manifest = {
        "pred_csv": str(args.pred_csv),
        "out_dir": str(out_dir),
        "labels_used_for_calibration": False,
        "group_col": args.group_col,
        "value_col": args.value_col,
        "score_col": args.score_col,
        "fallback_col": args.fallback_col,
        "final_method": args.final_method,
        "outputs": {
            "predictions_csv": str(pred_csv),
            "audit_csv": str(out_dir / "rank_trend_posthoc_audit.csv"),
        },
    }
    with (out_dir / "rank_trend_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] wrote {pred_csv}")


if __name__ == "__main__":
    main()
