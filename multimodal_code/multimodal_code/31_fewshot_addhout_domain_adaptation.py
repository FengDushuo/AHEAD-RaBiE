#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-shot target-domain calibration for AddH-out.

This script is intentionally conservative:
  - strict-blind predictions are read as fixed inputs;
  - only the selected AddH-out calibration rows are allowed to fit offsets/slopes;
  - all reported few-shot scores are evaluated on held-out AddH-out rows;
  - label usage is written into the manifest.

Use this route for a paper section such as:
  "few-shot target-domain adaptation with limited AddH-out labels".
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_BASE_COLS = [
    "pred_superblend_mae_guarded",
    "pred_superblend_balanced",
    "pred_superblend_trend",
    "pred_rank_trend_calibrated",
    "pred_pretrained_delta_final",
    "pred_fast_target_calibrated",
    "pred_llm_element_knowledge_blend",
    "pred_source_dopant_mean_prior",
    "pred_existing_anchor",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Few-shot AddH-out calibration and held-out evaluation.")
    ap.add_argument("--pred-csv", default="outputs_addh_superblend_precision/superblend_precision_addhout_predictions.csv")
    ap.add_argument("--labels", default="auto", help="CSV/XLSX labels. auto searches common AddH-out label files.")
    ap.add_argument("--out-dir", default="outputs_addh_fewshot_domain_calibration")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--group-col", default="material")
    ap.add_argument("--target-col", default="h_ads_excel")
    ap.add_argument("--base-cols", default="auto", help="Comma list, or auto.")
    ap.add_argument("--shots-per-material", default="0,1,2,3,4,5,6,8,10")
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--calibrators",
        default="none,global_offset,material_offset,global_affine,global_affine_material_offset,guarded_auto",
        help="Comma list of calibration methods.",
    )
    ap.add_argument("--split-strategy", choices=["random", "diverse_uncertainty"], default="random")
    ap.add_argument("--operational-shots-per-material", type=int, default=-1)
    ap.add_argument("--operational-strategy", choices=["diverse_uncertainty", "random"], default="diverse_uncertainty")
    ap.add_argument("--operational-base-col", default="pred_superblend_balanced")
    ap.add_argument("--operational-calibrator", default="material_offset")
    ap.add_argument("--calibration-ids", default="", help="Comma-separated IDs or a text/CSV file with an id column.")
    ap.add_argument("--max-abs-offset", type=float, default=1.20)
    ap.add_argument("--offset-shrink-k", type=float, default=4.0)
    ap.add_argument("--guard-min-improvement", type=float, default=0.03)
    ap.add_argument("--slope-min", type=float, default=0.55)
    ap.add_argument("--slope-max", type=float, default=1.45)
    ap.add_argument("--write-xlsx", action="store_true")
    return ap.parse_args()


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for x in parse_csv_list(raw):
        out.append(int(float(x)))
    return sorted(set(out))


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def parse_addhout_wide_excel(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    material_starts: List[Tuple[str, int]] = []
    for r in range(min(3, len(raw))):
        for c in range(raw.shape[1]):
            token = str(raw.iat[r, c]).strip()
            if token in {"CeO2", "ZnO"}:
                material_starts.append((token, c))
    material_starts = sorted(set(material_starts), key=lambda x: x[1])
    if not material_starts:
        raise ValueError(f"Could not locate CeO2/ZnO blocks in {path}")

    rows: List[Dict[str, object]] = []
    for _, row in raw.iterrows():
        idx_raw = row.iloc[0] if len(row) > 0 else np.nan
        dopant = row.iloc[1] if len(row) > 1 else ""
        idx_num = pd.to_numeric(pd.Series([idx_raw]), errors="coerce").iloc[0]
        if pd.isna(idx_num):
            continue
        dopant_s = str(dopant).strip()
        if not dopant_s or dopant_s.lower() == "nan":
            continue
        idx = int(idx_num)
        for material, start in material_starts:
            if start + 2 >= raw.shape[1]:
                continue
            etotal = pd.to_numeric(pd.Series([row.iloc[start]]), errors="coerce").iloc[0]
            eslab = pd.to_numeric(pd.Series([row.iloc[start + 1]]), errors="coerce").iloc[0]
            h_ads = pd.to_numeric(pd.Series([row.iloc[start + 2]]), errors="coerce").iloc[0]
            if pd.isna(h_ads):
                continue
            rows.append(
                {
                    "id": f"{material}-{idx}-{dopant_s}",
                    "material": material,
                    "idx": idx,
                    "dopant": dopant_s,
                    "h_ads_excel": float(h_ads),
                    "target_computed": float(h_ads),
                    "energy_total_excel": float(etotal) if pd.notna(etotal) else np.nan,
                    "energy_slab_excel": float(eslab) if pd.notna(eslab) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def find_labels(raw: str) -> Optional[Path]:
    if raw and raw != "auto":
        p = Path(raw)
        return p if p.exists() else None
    candidates = [
        Path("outputs_addh_llm_element_priors/addhout_audit_labels.csv"),
        Path("addH-out/addhout_audit_labels.csv"),
        Path("addH-out/氢吸附能.xlsx"),
        Path("addH-out/hydrogen_adsorption_energy.xlsx"),
        Path("addH-out/energy.xlsx"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_labels(raw: str, id_col: str, target_col: str) -> Tuple[pd.DataFrame, Optional[Path]]:
    path = find_labels(raw)
    if path is None:
        return pd.DataFrame(columns=[id_col, target_col]), None
    if path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            labels = read_table(path)
            if id_col not in labels.columns or target_col not in labels.columns:
                labels = parse_addhout_wide_excel(path)
        except Exception:
            labels = parse_addhout_wide_excel(path)
    else:
        labels = read_table(path)

    labels = labels.copy()
    if id_col not in labels.columns:
        cols = set(labels.columns)
        if {"material", "idx", "dopant"}.issubset(cols):
            labels[id_col] = (
                labels["material"].astype(str)
                + "-"
                + pd.to_numeric(labels["idx"], errors="coerce").fillna(-1).astype(int).astype(str)
                + "-"
                + labels["dopant"].astype(str)
            )
        else:
            raise ValueError(f"Label file {path} has no {id_col!r} column.")
    if target_col not in labels.columns:
        for c in ["h_ads_excel", "target_computed", "target", "H_ads", "h_ads"]:
            if c in labels.columns:
                labels[target_col] = pd.to_numeric(labels[c], errors="coerce")
                break
    if target_col not in labels.columns:
        raise ValueError(f"Label file {path} has no usable target column.")
    labels[target_col] = pd.to_numeric(labels[target_col], errors="coerce")
    keep = [id_col, target_col]
    for c in ["material", "dopant", "idx", "energy_total_excel", "energy_slab_excel", "target_computed"]:
        if c in labels.columns and c not in keep:
            keep.append(c)
    labels = labels[keep].dropna(subset=[id_col, target_col]).drop_duplicates(id_col)
    return labels, path


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


def metric_rows(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str,
    group_col: str,
    role: str,
    extra: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    groups: List[Tuple[str, pd.DataFrame]] = [("ALL", df)]
    if group_col in df.columns:
        for g, sub in df.groupby(group_col, dropna=False):
            groups.append((str(g), sub))
    for group, sub in groups:
        row: Dict[str, object] = {"role": role, "material": group, "pred_col": pred_col}
        if extra:
            row.update(extra)
        row.update(metrics(sub[target_col], sub[pred_col]))
        rows.append(row)
    return rows


def available_base_cols(pred: pd.DataFrame, raw: str) -> List[str]:
    if raw != "auto":
        return [c for c in parse_csv_list(raw) if c in pred.columns]
    cols = [c for c in DEFAULT_BASE_COLS if c in pred.columns]
    if not cols:
        cols = [c for c in pred.columns if c.startswith("pred_")]
    return cols


def prediction_disagreement(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    vals = []
    for c in cols:
        if c in df.columns:
            vals.append(pd.to_numeric(df[c], errors="coerce"))
    if len(vals) < 2:
        return pd.Series(0.0, index=df.index)
    arr = pd.concat(vals, axis=1)
    return arr.std(axis=1, skipna=True).fillna(0.0)


def diverse_uncertainty_select(
    df: pd.DataFrame,
    k: int,
    group_col: str,
    base_col: str,
    pred_cols: Sequence[str],
) -> List[str]:
    if k <= 0:
        return []
    selected: List[str] = []
    disagree = prediction_disagreement(df, pred_cols)
    for _, sub in df.groupby(group_col, dropna=False):
        sub = sub.copy()
        sub["_base"] = pd.to_numeric(sub[base_col], errors="coerce")
        sub["_disagree"] = disagree.loc[sub.index]
        sub = sub.sort_values(["_base", "_disagree"], ascending=[True, False], kind="mergesort")
        n = len(sub)
        kk = min(k, max(0, n - 1))
        if kk <= 0:
            continue
        positions = np.linspace(0, n - 1, kk).round().astype(int).tolist()
        ids: List[str] = []
        for pos in positions:
            lo = max(0, pos - 1)
            hi = min(n, pos + 2)
            window = sub.iloc[lo:hi].sort_values("_disagree", ascending=False, kind="mergesort")
            for sid in window["id"].astype(str):
                if sid not in ids:
                    ids.append(sid)
                    break
        if len(ids) < kk:
            for sid in sub.sort_values("_disagree", ascending=False, kind="mergesort")["id"].astype(str):
                if sid not in ids:
                    ids.append(sid)
                if len(ids) >= kk:
                    break
        selected.extend(ids[:kk])
    return selected


def random_select(df: pd.DataFrame, k: int, group_col: str, rng: np.random.Generator) -> List[str]:
    selected: List[str] = []
    if k <= 0:
        return selected
    for _, sub in df.groupby(group_col, dropna=False):
        ids = sub["id"].astype(str).to_numpy()
        kk = min(k, max(0, len(ids) - 1))
        if kk <= 0:
            continue
        selected.extend(rng.choice(ids, size=kk, replace=False).tolist())
    return selected


def choose_calibration_ids(
    df: pd.DataFrame,
    k: int,
    group_col: str,
    base_col: str,
    pred_cols: Sequence[str],
    strategy: str,
    rng: np.random.Generator,
) -> List[str]:
    if strategy == "diverse_uncertainty":
        return diverse_uncertainty_select(df, k, group_col, base_col, pred_cols)
    return random_select(df, k, group_col, rng)


def robust_median(values: Sequence[float]) -> float:
    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(s) == 0:
        return 0.0
    return float(np.median(s.to_numpy(float)))


def clip_float(x: float, lo: float, hi: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(min(max(x, lo), hi))


def fit_calibrator(
    calib: pd.DataFrame,
    base_col: str,
    target_col: str,
    group_col: str,
    method: str,
    max_abs_offset: float,
    offset_shrink_k: float,
    slope_min: float,
    slope_max: float,
    guard_min_improvement: float = 0.03,
) -> Dict[str, object]:
    p = pd.to_numeric(calib[base_col], errors="coerce")
    y = pd.to_numeric(calib[target_col], errors="coerce")
    ok = p.notna() & y.notna()
    c = calib.loc[ok].copy()
    p = p.loc[ok].to_numpy(float)
    y = y.loc[ok].to_numpy(float)
    n = int(len(c))
    params: Dict[str, object] = {"method": method, "base_col": base_col, "n_calibration": n}
    if method == "none" or n == 0:
        params.update({"global_offset": 0.0, "slope": 1.0, "intercept": 0.0, "group_offsets": {}})
        return params

    if method == "guarded_auto":
        candidates = ["none", "global_offset", "material_offset"]
        if n >= 6:
            candidates.extend(["global_affine", "global_affine_material_offset"])
        scores: Dict[str, float] = {}
        for cand in candidates:
            loo_pred = np.full(n, np.nan, dtype=float)
            for i in range(n):
                train_i = c.drop(c.index[i]).copy()
                test_i = c.iloc[[i]].copy()
                cand_params = fit_calibrator(
                    train_i,
                    base_col,
                    target_col,
                    group_col,
                    cand,
                    max_abs_offset,
                    offset_shrink_k,
                    slope_min,
                    slope_max,
                    guard_min_improvement,
                )
                loo_pred[i] = float(apply_calibrator(test_i, cand_params, group_col).iloc[0])
            scores[cand] = metrics(y, loo_pred)["mae"]
        none_score = scores.get("none", float("inf"))
        ranked = sorted(scores.items(), key=lambda kv: (math.inf if pd.isna(kv[1]) else kv[1], kv[0] != "none"))
        selected = ranked[0][0] if ranked else "none"
        if selected != "none" and scores[selected] > none_score - guard_min_improvement:
            selected = "none"
        selected_params = fit_calibrator(
            c,
            base_col,
            target_col,
            group_col,
            selected,
            max_abs_offset,
            offset_shrink_k,
            slope_min,
            slope_max,
            guard_min_improvement,
        )
        return {
            "method": "guarded_auto",
            "base_col": base_col,
            "n_calibration": n,
            "selected_method": selected,
            "loo_mae": scores,
            "guard_min_improvement": guard_min_improvement,
            "selected_params": selected_params,
        }

    resid = y - p
    shrink = n / max(n + offset_shrink_k, 1e-12)
    global_offset = clip_float(shrink * robust_median(resid), -max_abs_offset, max_abs_offset)
    group_offsets: Dict[str, float] = {}
    if method in {"material_offset", "global_affine_material_offset"} and group_col in c.columns:
        for g, sub in c.groupby(group_col, dropna=False):
            pp = pd.to_numeric(sub[base_col], errors="coerce")
            yy = pd.to_numeric(sub[target_col], errors="coerce")
            mm = pp.notna() & yy.notna()
            ng = int(mm.sum())
            if ng == 0:
                continue
            raw = robust_median(yy.loc[mm].to_numpy(float) - pp.loc[mm].to_numpy(float))
            gs = ng / max(ng + offset_shrink_k, 1e-12)
            off = (1.0 - gs) * global_offset + gs * raw
            group_offsets[str(g)] = clip_float(off, -max_abs_offset, max_abs_offset)

    slope = 1.0
    intercept = 0.0
    affine_blend = 0.0
    if method in {"global_affine", "global_affine_material_offset"} and n >= 4 and float(np.std(p)) > 1e-8:
        raw_slope, raw_intercept = np.polyfit(p, y, deg=1)
        slope = clip_float(float(raw_slope), slope_min, slope_max)
        intercept = clip_float(float(raw_intercept), -2.5 * max_abs_offset, 2.5 * max_abs_offset)
        affine_blend = n / max(n + 8.0, 1e-12)

    params.update(
        {
            "global_offset": global_offset,
            "slope": slope,
            "intercept": intercept,
            "affine_blend": affine_blend,
            "group_offsets": group_offsets,
            "max_abs_offset": max_abs_offset,
        }
    )
    return params


def apply_calibrator(df: pd.DataFrame, params: Dict[str, object], group_col: str) -> pd.Series:
    base_col = str(params["base_col"])
    method = str(params["method"])
    p = pd.to_numeric(df[base_col], errors="coerce")
    if method == "guarded_auto":
        selected = params.get("selected_params")
        if isinstance(selected, dict):
            return apply_calibrator(df, selected, group_col)
        return p
    if method == "none":
        return p

    global_offset = float(params.get("global_offset", 0.0))
    out = p + global_offset
    if method in {"global_affine", "global_affine_material_offset"}:
        slope = float(params.get("slope", 1.0))
        intercept = float(params.get("intercept", 0.0))
        blend = float(params.get("affine_blend", 0.0))
        affine = slope * p + intercept
        out = (1.0 - blend) * out + blend * affine

    if method in {"material_offset", "global_affine_material_offset"} and group_col in df.columns:
        offsets = {str(k): float(v) for k, v in dict(params.get("group_offsets", {})).items()}
        mapped = df[group_col].astype(str).map(offsets)
        material_out = p + mapped.fillna(global_offset)
        if method == "global_affine_material_offset":
            out = 0.50 * out + 0.50 * material_out
        else:
            out = material_out
    return out


def parse_calibration_ids(raw: str) -> List[str]:
    raw = str(raw or "").strip()
    if not raw:
        return []
    p = Path(raw)
    if p.exists():
        if p.suffix.lower() in {".csv", ".xlsx", ".xls"}:
            df = read_table(p)
            col = "id" if "id" in df.columns else df.columns[0]
            return [str(x) for x in df[col].dropna().astype(str).tolist()]
        return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return parse_csv_list(raw)


def aggregate_detail(detail: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    group_cols = ["shots_per_material", "base_col", "calibrator", "material"]
    for keys, sub in detail.groupby(group_cols, dropna=False):
        row: Dict[str, object] = dict(zip(group_cols, keys))
        for metric in ["n", "mae", "rmse", "bias", "pearson", "spearman"]:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
            if len(vals) == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_median"] = np.nan
                continue
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_median"] = float(vals.median())
            if metric in {"mae", "rmse", "pearson", "spearman"}:
                row[f"{metric}_q10"] = float(vals.quantile(0.10))
                row[f"{metric}_q90"] = float(vals.quantile(0.90))
        rows.append(row)
    return pd.DataFrame(rows)


def best_by_shot(agg: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.Series] = []
    overall = agg[agg["material"].astype(str) == "ALL"].copy()
    for shot, sub in overall.groupby("shots_per_material", dropna=False):
        s = sub.copy()
        s["_mae"] = pd.to_numeric(s["mae_mean"], errors="coerce")
        s["_spearman"] = pd.to_numeric(s["spearman_mean"], errors="coerce").fillna(-999.0)
        s = s.sort_values(["_mae", "_spearman"], ascending=[True, False], kind="mergesort")
        if len(s):
            rows.append(s.iloc[0].drop(labels=["_mae", "_spearman"]))
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = Path(args.pred_csv)
    if not pred_path.exists():
        raise SystemExit(f"[ERROR] prediction file not found: {pred_path}")
    pred = pd.read_csv(pred_path)
    if args.id_col != "id":
        pred = pred.rename(columns={args.id_col: "id"})
    if "id" not in pred.columns:
        raise SystemExit(f"[ERROR] {pred_path} has no id column.")

    labels, label_path = load_labels(args.labels, "id", args.target_col)
    labels = labels.rename(columns={args.target_col: "target"})
    merged = pred.merge(labels, on="id", how="left", suffixes=("", "__label"))
    if "material" not in merged.columns and "material__label" in merged.columns:
        merged["material"] = merged["material__label"]
    if "dopant" not in merged.columns and "dopant__label" in merged.columns:
        merged["dopant"] = merged["dopant__label"]
    merged["target"] = pd.to_numeric(merged["target"], errors="coerce")

    base_cols = available_base_cols(merged, args.base_cols)
    if not base_cols:
        raise SystemExit("[ERROR] no usable prediction columns found.")
    for c in base_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
    if args.operational_base_col not in merged.columns:
        args.operational_base_col = base_cols[0]

    labeled = merged[merged["target"].notna()].copy().reset_index(drop=True)
    if len(labeled) == 0:
        print("[WARN] no AddH-out labels found. Only calibration candidate selection will be written.")

    shots = parse_int_list(args.shots_per_material)
    calibrators = parse_csv_list(args.calibrators)
    detail_rows: List[Dict[str, object]] = []

    if len(labeled) > 0:
        for shot in shots:
            n_rep = 1 if shot <= 0 else max(1, args.repeats)
            for rep in range(n_rep):
                rng = np.random.default_rng(args.seed + 1009 * rep + 17 * shot)
                strategy = "random" if args.split_strategy == "random" else "diverse_uncertainty"
                split_base = args.operational_base_col if args.operational_base_col in labeled.columns else base_cols[0]
                calib_ids = choose_calibration_ids(labeled, shot, "material", split_base, base_cols, strategy, rng)
                calib_mask = labeled["id"].astype(str).isin(set(calib_ids))
                calib_df = labeled.loc[calib_mask].copy()
                hold_df = labeled.loc[~calib_mask].copy()
                for base_col in base_cols:
                    methods = ["none"] if shot <= 0 else calibrators
                    for method in methods:
                        params = fit_calibrator(
                            calib_df,
                            base_col,
                            "target",
                            "material",
                            method,
                            args.max_abs_offset,
                            args.offset_shrink_k,
                            args.slope_min,
                            args.slope_max,
                            args.guard_min_improvement,
                        )
                        eval_df = hold_df.copy()
                        eval_df["pred_fewshot_eval"] = apply_calibrator(eval_df, params, "material")
                        extra = {
                            "shots_per_material": shot,
                            "repeat": rep,
                            "n_calibration": int(len(calib_df)),
                            "n_holdout": int(len(hold_df)),
                            "base_col": base_col,
                            "calibrator": method,
                            "split_strategy": strategy,
                        }
                        detail_rows.extend(metric_rows(eval_df, "pred_fewshot_eval", "target", "material", "heldout", extra))

    detail = pd.DataFrame(detail_rows)
    if len(detail):
        detail.to_csv(out_dir / "fewshot_holdout_detail.csv", index=False)
        agg = aggregate_detail(detail)
        agg.to_csv(out_dir / "fewshot_holdout_summary.csv", index=False)
        best = best_by_shot(agg)
        best.to_csv(out_dir / "fewshot_recommended_by_holdout.csv", index=False)
        if args.write_xlsx:
            with pd.ExcelWriter(out_dir / "fewshot_holdout_report.xlsx") as w:
                detail.to_excel(w, sheet_name="detail", index=False)
                agg.to_excel(w, sheet_name="summary", index=False)
                best.to_excel(w, sheet_name="recommended", index=False)
    else:
        agg = pd.DataFrame()
        best = pd.DataFrame()

    # Operational calibration: either explicit ids, or a deterministic small
    # diverse/uncertain subset. If labels are absent, this is just a proposal
    # for which AddH-out rows should be labeled next.
    explicit_ids = parse_calibration_ids(args.calibration_ids)
    selected_ids: List[str] = []
    if explicit_ids:
        selected_ids = explicit_ids
    elif args.operational_shots_per_material >= 0:
        rng = np.random.default_rng(args.seed + 777)
        selected_ids = choose_calibration_ids(
            merged,
            args.operational_shots_per_material,
            "material",
            args.operational_base_col,
            base_cols,
            args.operational_strategy,
            rng,
        )

    selection = merged[merged["id"].astype(str).isin(set(selected_ids))].copy()
    selection_cols = [
        c
        for c in ["id", "material", "dopant", args.operational_base_col, "target"]
        if c in selection.columns
    ]
    if selection_cols:
        selection[selection_cols].to_csv(out_dir / "fewshot_calibration_selection.csv", index=False)

    op_pred = merged.copy()
    op_pred["fewshot_role"] = "unlabeled"
    if selected_ids:
        op_pred.loc[op_pred["id"].astype(str).isin(set(selected_ids)), "fewshot_role"] = "calibration"
        op_pred.loc[op_pred["target"].notna() & ~op_pred["id"].astype(str).isin(set(selected_ids)), "fewshot_role"] = "heldout"
    op_params = fit_calibrator(
        op_pred[op_pred["fewshot_role"] == "calibration"].copy(),
        args.operational_base_col,
        "target",
        "material",
        args.operational_calibrator,
        args.max_abs_offset,
        args.offset_shrink_k,
        args.slope_min,
        args.slope_max,
        args.guard_min_improvement,
    )
    op_pred["pred_fewshot_calibrated"] = apply_calibrator(op_pred, op_params, "material")
    op_pred["fewshot_base_col"] = args.operational_base_col
    op_pred["fewshot_calibrator"] = args.operational_calibrator
    if "target" in op_pred.columns:
        op_pred["fewshot_residual"] = op_pred["pred_fewshot_calibrated"] - op_pred["target"]
    op_pred.to_csv(out_dir / "fewshot_operational_predictions.csv", index=False)
    if args.write_xlsx:
        op_pred.to_excel(out_dir / "fewshot_operational_predictions.xlsx", index=False)

    audit_rows: List[Dict[str, object]] = []
    if op_pred["target"].notna().any():
        for role, sub in [
            ("all_labeled", op_pred[op_pred["target"].notna()].copy()),
            ("heldout_only", op_pred[op_pred["fewshot_role"] == "heldout"].copy()),
            ("calibration_only", op_pred[op_pred["fewshot_role"] == "calibration"].copy()),
        ]:
            if len(sub):
                audit_rows.extend(
                    metric_rows(
                        sub,
                        "pred_fewshot_calibrated",
                        "target",
                        "material",
                        role,
                        {
                            "shots_per_material": args.operational_shots_per_material,
                            "base_col": args.operational_base_col,
                            "calibrator": args.operational_calibrator,
                            "n_calibration": int((op_pred["fewshot_role"] == "calibration").sum()),
                        },
                    )
                )
        pd.DataFrame(audit_rows).to_csv(out_dir / "fewshot_operational_audit.csv", index=False)

    manifest = {
        "script": Path(__file__).name,
        "prediction_file": str(pred_path),
        "label_file": str(label_path) if label_path else None,
        "n_predictions": int(len(pred)),
        "n_labeled": int(merged["target"].notna().sum()),
        "base_cols": base_cols,
        "shots_per_material_grid": shots,
        "repeats": int(args.repeats),
        "calibrators": calibrators,
        "strict_blind": False,
        "label_usage": "AddH-out labels are used only for selected calibration rows; held-out rows are used for evaluation.",
        "operational_base_col": args.operational_base_col,
        "operational_calibrator": args.operational_calibrator,
        "operational_selected_ids": selected_ids,
        "operational_params": op_params,
        "outputs": {
            "detail": str(out_dir / "fewshot_holdout_detail.csv"),
            "summary": str(out_dir / "fewshot_holdout_summary.csv"),
            "recommended": str(out_dir / "fewshot_recommended_by_holdout.csv"),
            "selection": str(out_dir / "fewshot_calibration_selection.csv"),
            "predictions": str(out_dir / "fewshot_operational_predictions.csv"),
            "audit": str(out_dir / "fewshot_operational_audit.csv"),
        },
    }
    (out_dir / "fewshot_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] wrote {out_dir}")
    if len(best):
        show = best[best["material"].astype(str) == "ALL"] if "material" in best.columns else best
        cols = [c for c in ["shots_per_material", "base_col", "calibrator", "mae_mean", "mae_std", "spearman_mean"] if c in show.columns]
        print(show[cols].to_string(index=False))


if __name__ == "__main__":
    main()
