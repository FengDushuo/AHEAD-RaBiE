#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final conservative super-blend for AddH-out prediction.

This script does not train a new backbone. It sits after the current fast
target-domain calibration, pretrained-delta head, and rank/trend calibration
steps, then creates a small set of guarded final prediction columns.

Default columns are label-free at runtime:
  - pred_superblend_balanced: best default when both MAE and trend matter.
  - pred_superblend_trend: more rank/trend oriented.
  - pred_superblend_mae_guarded: MAE-oriented diagnostic blend; more aggressive.

Optional audit-label calibration is explicit and marked in the manifest. Use it
only when addH-out labels are intentionally allowed as calibration data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Super-blend final AddH-out predictions.")
    ap.add_argument("--rank-trend-csv", default="outputs_addh_rank_trend_calibrated/rank_trend_calibrated_addhout_predictions.csv")
    ap.add_argument("--delta-csv", default="outputs_addh_pretrained_delta_head/pretrained_delta_head_addhout_predictions.csv")
    ap.add_argument("--target-csv", default="outputs_addh_target_calibrated_fast/target_calibrated_addhout_predictions.csv")
    ap.add_argument("--knowledge-csv", default="outputs_addh_llm_element_knowledge_blend_scnet_deepseek_v4_pro/knowledge_enhanced_addhout_predictions.csv")
    ap.add_argument("--out-dir", default="outputs_addh_superblend_precision")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--group-col", default="material")
    ap.add_argument("--audit-labels-csv", default="auto")
    ap.add_argument("--audit-target-col", default="h_ads_excel")
    ap.add_argument(
        "--final-method",
        choices=["balanced", "trend", "mae_guarded", "rank_trend", "auto_audit"],
        default="balanced",
        help="auto_audit requires --allow-audit-label-selection and uses labels only for current-batch selection.",
    )
    ap.add_argument("--balanced-rankz-weights", default="CeO2=0.66,ZnO=0.00")
    ap.add_argument("--trend-rankz-weights", default="CeO2=0.87,ZnO=0.00")
    ap.add_argument("--mae-delta-weights", default="CeO2=0.24,ZnO=0.48")
    ap.add_argument(
        "--allow-audit-label-selection",
        action="store_true",
        help="Allow selecting the final built-in candidate using addH-out audit labels.",
    )
    ap.add_argument(
        "--allow-audit-offset-calibration",
        action="store_true",
        help="Allow per-material offset calibration using addH-out audit labels. This is not strict-blind.",
    )
    ap.add_argument(
        "--offset-base-col",
        default="auto",
        help="Base column for explicit audit-offset calibration. auto tries built-in candidates and keeps the lowest audit MAE.",
    )
    ap.add_argument("--offset-max-abs", type=float, default=0.80)
    ap.add_argument("--offset-step", type=float, default=0.01)
    ap.add_argument("--clip-to-candidate-envelope", action="store_true")
    return ap.parse_args()


def parse_weight_map(raw: str, default: float = 0.0) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = float(v.strip())
    out["__default__"] = default
    return out


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def merge_optional(base: pd.DataFrame, csv_path: str, id_col: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        print(f"[WARN] optional file missing: {path}")
        return base
    other = pd.read_csv(path)
    if id_col not in other.columns:
        print(f"[WARN] optional file has no id column {id_col!r}: {path}")
        return base
    keep = [id_col]
    for c in other.columns:
        if c == id_col:
            continue
        if c.startswith("pred_") or c.startswith("llm_") or c.startswith("source_"):
            keep.append(c)
        elif c in {"material", "dopant", "has_embedding"} and c not in base.columns:
            keep.append(c)
    other = other[keep].drop_duplicates(id_col)
    rename: Dict[str, str] = {}
    for c in other.columns:
        if c == id_col:
            continue
        if c in base.columns:
            rename[c] = f"{c}__from_{path.parent.name}"
    other = other.rename(columns=rename)
    return base.merge(other, on=id_col, how="left")


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
            row: Dict[str, object] = {"pred_col": c, "target_col": target_col, group_col: g}
            row.update(metrics(d[target_col], d[c]))
            rows.append(row)
    return pd.DataFrame(rows)


def filled_numeric(df: pd.DataFrame, col: str, fallback: pd.Series) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
    return s.where(s.notna(), fallback)


def material_blend(
    df: pd.DataFrame,
    a: pd.Series,
    b: pd.Series,
    weights: Dict[str, float],
    group_col: str,
) -> pd.Series:
    group = df[group_col].astype(str) if group_col in df.columns else pd.Series("__default__", index=df.index)
    w = group.map(weights).fillna(weights.get("__default__", 0.0)).astype(float).clip(0.0, 1.0)
    return (1.0 - w) * a + w * b


def candidate_envelope_clip(df: pd.DataFrame, pred: pd.Series, cols: Sequence[str]) -> pd.Series:
    vals = []
    for c in cols:
        if c in df.columns:
            vals.append(pd.to_numeric(df[c], errors="coerce"))
    if not vals:
        return pred
    arr = pd.concat(vals, axis=1)
    lo = arr.min(axis=1, skipna=True)
    hi = arr.max(axis=1, skipna=True)
    return pred.clip(lower=lo, upper=hi)


def find_audit_labels(args: argparse.Namespace, rank_path: Path) -> Optional[Path]:
    if args.audit_labels_csv and args.audit_labels_csv != "auto":
        p = Path(args.audit_labels_csv)
        return p if p.exists() else None
    candidates = [
        rank_path.parent / "addhout_audit_labels.csv",
        rank_path.parent.parent / "outputs_addh_llm_element_priors" / "addhout_audit_labels.csv",
        Path("outputs_addh_llm_element_priors") / "addhout_audit_labels.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def tune_material_offsets(
    detail: pd.DataFrame,
    base_col: str,
    target_col: str,
    group_col: str,
    max_abs: float,
    step: float,
) -> Tuple[pd.Series, Dict[str, float], Dict[str, float]]:
    base = pd.to_numeric(detail[base_col], errors="coerce")
    y = pd.to_numeric(detail[target_col], errors="coerce")
    groups = sorted([str(x) for x in detail[group_col].dropna().unique()]) if group_col in detail.columns else ["__all__"]
    offsets: Dict[str, float] = {}
    group_mae: Dict[str, float] = {}
    grid = np.arange(-abs(max_abs), abs(max_abs) + 0.5 * step, abs(step))
    out = base.copy()
    for g in groups:
        idx = detail.index if g == "__all__" else detail.index[detail[group_col].astype(str) == g]
        m = y.loc[idx].notna() & base.loc[idx].notna()
        if int(m.sum()) < 3:
            offsets[g] = 0.0
            continue
        best = (float("inf"), 0.0)
        yy = y.loc[idx][m].to_numpy(float)
        bb = base.loc[idx][m].to_numpy(float)
        for off in grid:
            mae = float(np.mean(np.abs((bb + float(off)) - yy)))
            if mae < best[0]:
                best = (mae, float(off))
        offsets[g] = best[1]
        group_mae[g] = best[0]
        out.loc[idx] = base.loc[idx] + best[1]
    return out, offsets, group_mae


def choose_by_audit(detail: pd.DataFrame, pred_cols: Sequence[str], target_col: str) -> str:
    scored: List[Tuple[float, float, str]] = []
    for c in pred_cols:
        if c not in detail.columns:
            continue
        m = metrics(detail[target_col], detail[c])
        mae = float(m.get("mae", np.nan))
        sp = float(m.get("spearman", np.nan))
        if np.isfinite(mae):
            scored.append((mae, -sp if np.isfinite(sp) else 0.0, c))
    if not scored:
        raise SystemExit("[ERROR] auto_audit requested but no candidate could be audited.")
    scored.sort()
    return scored[0][2]


def main() -> None:
    args = parse_args()
    rank_path = Path(args.rank_trend_csv)
    if not rank_path.exists():
        raise SystemExit(f"[ERROR] missing rank trend predictions: {rank_path}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(rank_path)
    if args.id_col not in pred.columns:
        raise SystemExit(f"[ERROR] missing id column {args.id_col!r} in {rank_path}")
    for p in [args.delta_csv, args.target_csv, args.knowledge_csv]:
        pred = merge_optional(pred, p, args.id_col)

    if args.group_col not in pred.columns:
        raise SystemExit(f"[ERROR] missing group column {args.group_col!r} after merge.")

    q_col = first_existing(pred, ["pred_rank_trend_calibrated", "pred_trend_quantile_by_group", "pred_pretrained_delta_final", "pred_existing_anchor"])
    rz_col = first_existing(pred, ["pred_trend_rankz_by_group", "pred_rank_trend_calibrated", "pred_pretrained_delta_final", "pred_existing_anchor"])
    delta_col = first_existing(pred, ["pred_delta_head_ensemble", "pred_delta_head_recentered", "pred_pretrained_delta_final"])
    if not q_col or not rz_col or not delta_col:
        raise SystemExit("[ERROR] required prediction columns are missing; run delta and rank-trend steps first.")

    q = pd.to_numeric(pred[q_col], errors="coerce")
    rz = filled_numeric(pred, rz_col, q)
    delta = filled_numeric(pred, delta_col, q)
    q = q.where(q.notna(), rz)

    balanced_w = parse_weight_map(args.balanced_rankz_weights)
    trend_w = parse_weight_map(args.trend_rankz_weights)
    mae_w = parse_weight_map(args.mae_delta_weights)

    pred["pred_superblend_balanced"] = material_blend(pred, q, rz, balanced_w, args.group_col)
    pred["pred_superblend_trend"] = material_blend(pred, q, rz, trend_w, args.group_col)
    pred["pred_superblend_mae_guarded"] = material_blend(pred, rz, delta, mae_w, args.group_col)

    envelope_cols = [
        q_col,
        rz_col,
        delta_col,
        "pred_superblend_balanced",
        "pred_superblend_trend",
        "pred_superblend_mae_guarded",
    ]
    if args.clip_to_candidate_envelope:
        pred["pred_superblend_mae_guarded"] = candidate_envelope_clip(
            pred, pd.to_numeric(pred["pred_superblend_mae_guarded"], errors="coerce"), envelope_cols
        )

    labels_path = find_audit_labels(args, rank_path)
    labels_used_for_selection = False
    labels_used_for_offset = False
    detail = pd.DataFrame()
    audit = pd.DataFrame()
    candidate_cols = [
        "pred_superblend_balanced",
        "pred_superblend_trend",
        "pred_superblend_mae_guarded",
        "pred_rank_trend_calibrated",
        "pred_trend_rankz_by_group",
        "pred_pretrained_delta_final",
        "pred_existing_anchor",
    ]
    candidate_cols = [c for c in candidate_cols if c in pred.columns]

    selected_final = {
        "balanced": "pred_superblend_balanced",
        "trend": "pred_superblend_trend",
        "mae_guarded": "pred_superblend_mae_guarded",
        "rank_trend": "pred_rank_trend_calibrated" if "pred_rank_trend_calibrated" in pred.columns else "pred_superblend_balanced",
    }.get(args.final_method, "pred_superblend_balanced")

    if labels_path and labels_path.exists():
        labels = pd.read_csv(labels_path)
        if args.id_col in labels.columns and args.audit_target_col in labels.columns:
            keep = [c for c in [args.id_col, args.audit_target_col, args.group_col, "dopant"] if c in labels.columns]
            detail = pred.merge(labels[keep].drop_duplicates(args.id_col), on=args.id_col, how="left", suffixes=("", "_audit"))
            if args.allow_audit_offset_calibration:
                offset_base_col = args.offset_base_col
                offset_pred = None
                offsets: Dict[str, float] = {}
                offset_group_mae: Dict[str, float] = {}
                if offset_base_col == "auto":
                    best_offset = (float("inf"), "", pd.Series(dtype=float), {}, {})
                    for base_col in candidate_cols:
                        if base_col not in detail.columns:
                            continue
                        cand_pred, cand_offsets, cand_group_mae = tune_material_offsets(
                            detail=detail,
                            base_col=base_col,
                            target_col=args.audit_target_col,
                            group_col=args.group_col,
                            max_abs=args.offset_max_abs,
                            step=args.offset_step,
                        )
                        cand_mae = float(metrics(detail[args.audit_target_col], cand_pred)["mae"])
                        if np.isfinite(cand_mae) and cand_mae < best_offset[0]:
                            best_offset = (cand_mae, base_col, cand_pred, cand_offsets, cand_group_mae)
                    if not best_offset[1]:
                        raise SystemExit("[ERROR] offset-base-col auto found no auditable candidate.")
                    offset_base_col = best_offset[1]
                    offset_pred = best_offset[2]
                    offsets = best_offset[3]
                    offset_group_mae = best_offset[4]
                else:
                    if offset_base_col not in detail.columns:
                        raise SystemExit(f"[ERROR] offset base column not found: {offset_base_col}")
                    offset_pred, offsets, offset_group_mae = tune_material_offsets(
                        detail=detail,
                        base_col=offset_base_col,
                        target_col=args.audit_target_col,
                        group_col=args.group_col,
                        max_abs=args.offset_max_abs,
                        step=args.offset_step,
                    )
                detail["pred_superblend_audit_offset_calibrated"] = offset_pred
                pred = pred.merge(
                    detail[[args.id_col, "pred_superblend_audit_offset_calibrated"]],
                    on=args.id_col,
                    how="left",
                )
                candidate_cols.append("pred_superblend_audit_offset_calibrated")
                labels_used_for_offset = True
                with (out_dir / "audit_offset_calibration.json").open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "labels_used_for_calibration": True,
                            "base_col": offset_base_col,
                            "offsets": offsets,
                            "group_mae_after_offset": offset_group_mae,
                            "warning": "Uses addH-out labels; not strict-blind.",
                        },
                        f,
                        indent=2,
                    )
            if args.final_method == "auto_audit":
                if not args.allow_audit_label_selection:
                    raise SystemExit("[ERROR] auto_audit requires --allow-audit-label-selection.")
                selected_final = choose_by_audit(detail if "pred_superblend_audit_offset_calibrated" not in pred.columns else pred.merge(labels[keep].drop_duplicates(args.id_col), on=args.id_col, how="left", suffixes=("", "_audit2")), candidate_cols, args.audit_target_col)
                labels_used_for_selection = True

    if args.final_method == "auto_audit" and not labels_used_for_selection:
        raise SystemExit("[ERROR] auto_audit requested, but no usable audit labels were found.")
    if args.allow_audit_offset_calibration and not labels_used_for_offset:
        raise SystemExit("[ERROR] audit offset calibration requested, but no usable audit labels were found.")

    if selected_final not in pred.columns:
        raise SystemExit(f"[ERROR] selected final column missing: {selected_final}")
    pred["pred_superblend_final"] = pd.to_numeric(pred[selected_final], errors="coerce")
    pred["superblend_final_source_col"] = selected_final
    pred["superblend_final_rank"] = pred["pred_superblend_final"].rank(method="average", ascending=True)
    pred = pred.sort_values(["superblend_final_rank", args.id_col], na_position="last").reset_index(drop=True)

    pred_csv = out_dir / "superblend_precision_addhout_predictions.csv"
    pred.to_csv(pred_csv, index=False)
    try:
        pred.to_excel(out_dir / "superblend_precision_addhout_predictions.xlsx", index=False)
    except Exception:
        pass

    if labels_path and labels_path.exists():
        labels = pd.read_csv(labels_path)
        if args.id_col in labels.columns and args.audit_target_col in labels.columns:
            keep = [c for c in [args.id_col, args.audit_target_col, args.group_col, "dopant"] if c in labels.columns]
            detail = pred.merge(labels[keep].drop_duplicates(args.id_col), on=args.id_col, how="left", suffixes=("", "_audit"))
            audit_cols = list(dict.fromkeys(["pred_superblend_final"] + candidate_cols))
            audit = metric_rows(detail, audit_cols, args.audit_target_col, args.group_col)
            audit.to_csv(out_dir / "superblend_precision_posthoc_audit.csv", index=False)
            detail["err_superblend_final"] = (
                pd.to_numeric(detail["pred_superblend_final"], errors="coerce")
                - pd.to_numeric(detail[args.audit_target_col], errors="coerce")
            )
            detail["abs_err_superblend_final"] = detail["err_superblend_final"].abs()
            detail.sort_values("abs_err_superblend_final", ascending=False).to_csv(
                out_dir / "superblend_precision_posthoc_audit_detail.csv", index=False
            )
            try:
                audit.to_excel(out_dir / "superblend_precision_posthoc_audit.xlsx", index=False)
            except Exception:
                pass
            print("[POSTHOC AUDIT]")
            print(audit.to_string(index=False))

    manifest = {
        "rank_trend_csv": str(rank_path),
        "delta_csv": args.delta_csv,
        "target_csv": args.target_csv,
        "knowledge_csv": args.knowledge_csv,
        "out_dir": str(out_dir),
        "final_method": args.final_method,
        "selected_final_col": selected_final,
        "labels_used_for_selection": labels_used_for_selection,
        "labels_used_for_offset_calibration": labels_used_for_offset,
        "balanced_rankz_weights": balanced_w,
        "trend_rankz_weights": trend_w,
        "mae_delta_weights": mae_w,
        "source_columns": {"quantile": q_col, "rankz": rz_col, "delta": delta_col},
        "outputs": {
            "predictions_csv": str(pred_csv),
            "audit_csv": str(out_dir / "superblend_precision_posthoc_audit.csv"),
        },
    }
    with (out_dir / "superblend_precision_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] wrote {pred_csv}")
    print(f"[FINAL] pred_superblend_final source={selected_final}")
    if labels_used_for_selection or labels_used_for_offset:
        print("[WARN] addH-out labels were used for selection/calibration; do not treat that column as strict-blind.")


if __name__ == "__main__":
    main()
