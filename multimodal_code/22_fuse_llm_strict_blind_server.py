#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fuse the strict-blind LLM/element-knowledge prediction with the heavy
graph/multiview strict-blind prediction.

This script is intentionally conservative:
  - It never needs addH-out labels to build predictions.
  - If the heavy strict-blind prediction has a large mean shift relative to the
    source training target distribution, it is either down-weighted or ignored.
  - Optional labels are used only for a post-hoc audit CSV.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


STRICT_PRED_CANDIDATES = [
    "pred_strict_blind_final",
    "pred_strict_blind_strategy_weighted",
    "pred_strict_blind_strategy_median",
    "pred_strict_blind",
    "pred_strict_blind_weighted",
    "pred_strict_blind_median",
    "pred",
    "pred_median",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Server final blend for AddH-out strict-blind predictions.")
    ap.add_argument(
        "--llm-pred-csv",
        default="outputs_addh_llm_element_knowledge_blend_scnet_deepseek_v4_pro/knowledge_enhanced_addhout_predictions.csv",
    )
    ap.add_argument(
        "--strict-pred-csv",
        default="outputs_addh_strict_blind_final/strict_blind_strategy_ensemble_predictions.csv",
    )
    ap.add_argument("--out-dir", default="outputs_addh_server_final_blend")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--llm-col", default="pred_llm_element_knowledge_blend")
    ap.add_argument("--dopant-col", default="pred_source_dopant_mean_prior")
    ap.add_argument("--strict-col", default="auto")
    ap.add_argument("--train-features-csv", default="outputs_addh_llm_element_priors/knowledge_features_train.csv")
    ap.add_argument("--train-target-col", default="target")
    ap.add_argument("--target-abs-max", type=float, default=10.0)
    ap.add_argument("--audit-labels-csv", default="")
    ap.add_argument("--audit-target-col", default="h_ads_excel")
    ap.add_argument("--blend-llm", type=float, default=0.85)
    ap.add_argument("--blend-strict", type=float, default=0.15)
    ap.add_argument("--blend-dopant", type=float, default=0.0)
    ap.add_argument(
        "--max-strict-mean-shift",
        type=float,
        default=1.50,
        help="Ignore strict-blind prediction if abs(mean(strict)-mean(source clean target)) exceeds this value.",
    )
    ap.add_argument("--no-recenter-strict", action="store_true")
    ap.add_argument("--clip-to-source-range", action="store_true")
    return ap.parse_args()


def detect_pred_col(df: pd.DataFrame, requested: str) -> str:
    if requested and requested != "auto":
        if requested not in df.columns:
            raise SystemExit(f"[ERROR] requested strict prediction column not found: {requested}")
        return requested
    for c in STRICT_PRED_CANDIDATES:
        if c in df.columns:
            return c
    numeric = [
        c
        for c in df.columns
        if c != "id" and (c.startswith("pred") or c.endswith("_pred")) and pd.api.types.is_numeric_dtype(df[c])
    ]
    if numeric:
        return numeric[0]
    raise SystemExit("[ERROR] could not auto-detect a strict-blind prediction column.")


def clean_source_target(train_path: Path, target_col: str, target_abs_max: float) -> pd.Series:
    if not train_path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(train_path)
    if target_col not in df.columns:
        return pd.Series(dtype=float)
    y = pd.to_numeric(df[target_col], errors="coerce")
    return y[y.notna() & (y.abs() <= target_abs_max)].reset_index(drop=True)


def rowwise_blend(parts: Dict[str, Tuple[np.ndarray, float]]) -> np.ndarray:
    n = len(next(iter(parts.values()))[0])
    num = np.zeros(n, dtype=float)
    den = np.zeros(n, dtype=float)
    for values, weight in parts.values():
        if weight <= 0:
            continue
        v = np.asarray(values, dtype=float)
        m = np.isfinite(v)
        num[m] += v[m] * weight
        den[m] += weight
    out = np.full(n, np.nan, dtype=float)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def metrics_for(df: pd.DataFrame, pred_col: str, target_col: str, material: Optional[str] = None) -> Dict[str, object]:
    d = df.copy()
    if material is not None and "material" in d.columns:
        d = d[d["material"].astype(str) == material]
    y = pd.to_numeric(d[target_col], errors="coerce")
    p = pd.to_numeric(d[pred_col], errors="coerce")
    m = y.notna() & p.notna()
    yv = y[m].to_numpy(float)
    pv = p[m].to_numpy(float)
    if len(yv) == 0:
        return {
            "pred_col": pred_col,
            "target_col": target_col,
            "material": material,
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "pearson": np.nan,
            "spearman": np.nan,
        }
    return {
        "pred_col": pred_col,
        "target_col": target_col,
        "material": material,
        "n": int(len(yv)),
        "mae": float(np.mean(np.abs(pv - yv))),
        "rmse": float(np.mean((pv - yv) ** 2) ** 0.5),
        "bias": float(np.mean(pv - yv)),
        "pearson": corr(yv, pv),
        "spearman": corr(pd.Series(yv).rank().to_numpy(float), pd.Series(pv).rank().to_numpy(float)),
    }


def audit_predictions(pred: pd.DataFrame, labels_path: Path, target_col: str, out_dir: Path) -> pd.DataFrame:
    if not labels_path.exists():
        return pd.DataFrame()
    labels = pd.read_csv(labels_path)
    if target_col not in labels.columns:
        return pd.DataFrame()
    keep = [c for c in ["id", target_col] if c in labels.columns]
    merged = pred.merge(labels[keep], on="id", how="left")
    pred_cols = [
        c
        for c in [
            "pred_addh_server_final_blend",
            "pred_llm_element_knowledge_blend",
            "pred_strict_blind_server_recenter",
            "pred_strict_blind_server_raw",
            "pred_source_dopant_mean_prior",
        ]
        if c in merged.columns
    ]
    rows: List[Dict[str, object]] = []
    materials = [None]
    if "material" in merged.columns:
        materials.extend(sorted([m for m in merged["material"].dropna().astype(str).unique()]))
    for c in pred_cols:
        for mat in materials:
            rows.append(metrics_for(merged, c, target_col, mat))
    audit = pd.DataFrame(rows)
    audit.to_csv(out_dir / "server_final_blend_posthoc_audit.csv", index=False)
    merged.to_csv(out_dir / "server_final_blend_posthoc_audit_detail.csv", index=False)
    try:
        audit.to_excel(out_dir / "server_final_blend_posthoc_audit.xlsx", index=False)
    except Exception:
        pass
    return audit


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    llm_path = Path(args.llm_pred_csv)
    if not llm_path.exists():
        raise SystemExit(f"[ERROR] missing LLM prediction CSV: {llm_path}")
    llm = pd.read_csv(llm_path)
    if args.id_col not in llm.columns:
        raise SystemExit(f"[ERROR] id column not found in LLM prediction CSV: {args.id_col}")
    if args.llm_col not in llm.columns:
        raise SystemExit(f"[ERROR] LLM prediction column not found: {args.llm_col}")

    out = llm.copy()
    out["pred_llm_element_knowledge_blend"] = pd.to_numeric(out[args.llm_col], errors="coerce")

    y_source = clean_source_target(Path(args.train_features_csv), args.train_target_col, args.target_abs_max)
    source_mean = float(y_source.mean()) if len(y_source) else float("nan")
    source_min = float(y_source.min()) if len(y_source) else float("nan")
    source_max = float(y_source.max()) if len(y_source) else float("nan")

    strict_used = False
    strict_col = ""
    strict_mean_shift = float("nan")
    strict_path = Path(args.strict_pred_csv)
    if strict_path.exists():
        strict = pd.read_csv(strict_path)
        if args.id_col not in strict.columns:
            print(f"[WARN] strict prediction CSV has no id column; ignoring: {strict_path}")
        else:
            strict_col = detect_pred_col(strict, args.strict_col)
            strict_small = strict[[args.id_col, strict_col]].copy()
            strict_small = strict_small.rename(columns={strict_col: "pred_strict_blind_server_raw"})
            out = out.merge(strict_small, on=args.id_col, how="left")
            raw = pd.to_numeric(out["pred_strict_blind_server_raw"], errors="coerce")
            strict_mean = float(raw.mean()) if raw.notna().any() else float("nan")
            strict_mean_shift = abs(strict_mean - source_mean) if np.isfinite(strict_mean) and np.isfinite(source_mean) else float("nan")
            if not args.no_recenter_strict:
                llm_med = float(pd.to_numeric(out["pred_llm_element_knowledge_blend"], errors="coerce").median())
                raw_med = float(raw.median()) if raw.notna().any() else float("nan")
                if np.isfinite(raw_med) and np.isfinite(llm_med):
                    out["pred_strict_blind_server_recenter"] = raw - raw_med + llm_med
                else:
                    out["pred_strict_blind_server_recenter"] = raw
            else:
                out["pred_strict_blind_server_recenter"] = raw
            strict_used = bool(np.isfinite(strict_mean_shift) and strict_mean_shift <= args.max_strict_mean_shift)
            if not strict_used:
                print(
                    f"[WARN] strict prediction ignored: abs(mean shift)={strict_mean_shift:.4f} "
                    f"> max={args.max_strict_mean_shift:.4f}"
                )
    else:
        out["pred_strict_blind_server_raw"] = np.nan
        out["pred_strict_blind_server_recenter"] = np.nan
        print(f"[WARN] strict prediction CSV not found; final blend will use LLM/dopant only: {strict_path}")

    dopant_values = (
        pd.to_numeric(out[args.dopant_col], errors="coerce").to_numpy(float)
        if args.dopant_col in out.columns
        else np.full(len(out), np.nan)
    )
    strict_weight = args.blend_strict if strict_used else 0.0
    final = rowwise_blend(
        {
            "llm": (pd.to_numeric(out["pred_llm_element_knowledge_blend"], errors="coerce").to_numpy(float), args.blend_llm),
            "strict": (pd.to_numeric(out["pred_strict_blind_server_recenter"], errors="coerce").to_numpy(float), strict_weight),
            "dopant": (dopant_values, args.blend_dopant),
        }
    )
    if args.clip_to_source_range and np.isfinite(source_min) and np.isfinite(source_max):
        final = np.clip(final, source_min, source_max)

    out["pred_addh_server_final_blend"] = final
    out["server_final_rank"] = pd.Series(final).rank(method="average", ascending=True).to_numpy()
    out["server_final_blend_strict_used"] = int(strict_used)
    out["server_final_blend_strict_mean_shift_vs_source"] = strict_mean_shift
    out["server_final_blend_strict_col"] = strict_col
    out = out.sort_values(["server_final_rank", "id"], na_position="last").reset_index(drop=True)

    out_csv = out_dir / "server_final_addhout_predictions.csv"
    out_xlsx = out_dir / "server_final_addhout_predictions.xlsx"
    out.to_csv(out_csv, index=False)
    try:
        out.to_excel(out_xlsx, index=False)
    except Exception:
        pass

    manifest = {
        "llm_pred_csv": str(llm_path),
        "strict_pred_csv": str(strict_path),
        "strict_pred_col": strict_col,
        "strict_used": strict_used,
        "strict_mean_shift_vs_source": strict_mean_shift,
        "source_clean_target_mean": source_mean,
        "source_clean_target_min": source_min,
        "source_clean_target_max": source_max,
        "blend_weights_requested": {
            "llm": args.blend_llm,
            "strict": args.blend_strict,
            "dopant": args.blend_dopant,
        },
        "blend_weights_effective": {
            "llm": args.blend_llm,
            "strict": strict_weight,
            "dopant": args.blend_dopant,
        },
        "labels_used_for_prediction": False,
        "outputs": {
            "csv": str(out_csv),
            "xlsx": str(out_xlsx),
        },
    }
    with (out_dir / "server_final_blend_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    audit = pd.DataFrame()
    if args.audit_labels_csv:
        audit = audit_predictions(out, Path(args.audit_labels_csv), args.audit_target_col, out_dir)
        if len(audit):
            print("[POSTHOC AUDIT ONLY]")
            print(audit.to_string(index=False))

    print("[OK] wrote", out_csv)
    print("[OK] wrote", out_xlsx)
    print("[INFO] strict_used=", strict_used)
    print("[INFO] strict_mean_shift_vs_source=", strict_mean_shift)


if __name__ == "__main__":
    main()
