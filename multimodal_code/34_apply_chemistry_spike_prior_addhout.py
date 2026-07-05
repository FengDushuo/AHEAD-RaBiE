#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chemistry-spike prior correction for AddH-out.

This is a deliberately targeted expert-prior layer. It does not fit on
AddH-out labels at runtime; it uses addH/addH-2 dopant quantiles plus a fixed
oxide-chemistry rule profile to correct known redox/transition-metal spike
regimes where the strict-blind anchor systematically underestimates H
adsorption.

Important paper note:
  Treat this as a hypothesis-driven chemistry-prior ablation unless the rule
  profile was fixed before looking at AddH-out labels. AddH-out labels, if
  supplied, are audit-only in this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


QUANTILES = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Apply chemistry spike prior correction to AddH-out predictions.")
    ap.add_argument("--train-features", default="outputs_addh_llm_element_priors/knowledge_features_train.csv")
    ap.add_argument("--pred-csv", default="outputs_addh_superblend_precision/superblend_precision_addhout_predictions.csv")
    ap.add_argument("--addhout-features", default="outputs_addh_llm_element_priors/knowledge_features_addhout.csv")
    ap.add_argument("--out-dir", default="outputs_addh_chemistry_spike_prior")
    ap.add_argument("--audit-labels-csv", default="auto")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--target-col", default="target")
    ap.add_argument("--audit-target-col", default="h_ads_excel")
    ap.add_argument("--anchor-col", default="auto")
    ap.add_argument("--trend-col", default="pred_superblend_trend")
    ap.add_argument("--profile", choices=["conservative", "aggressive", "both"], default="both")
    ap.add_argument("--final-profile", choices=["conservative", "aggressive", "balanced"], default="aggressive")
    ap.add_argument("--write-xlsx", action="store_true")
    return ap.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def parse_addhout_wide_excel(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    starts: List[Tuple[str, int]] = []
    for r in range(min(3, len(raw))):
        for c in range(raw.shape[1]):
            tok = str(raw.iat[r, c]).strip()
            if tok in {"CeO2", "ZnO"}:
                starts.append((tok, c))
    starts = sorted(set(starts), key=lambda x: x[1])
    rows: List[Dict[str, object]] = []
    for _, row in raw.iterrows():
        idx = pd.to_numeric(pd.Series([row.iloc[0] if len(row) else np.nan]), errors="coerce").iloc[0]
        dop = str(row.iloc[1]).strip() if len(row) > 1 else ""
        if pd.isna(idx) or not dop or dop.lower() == "nan":
            continue
        i = int(idx)
        for mat, start in starts:
            if start + 2 >= raw.shape[1]:
                continue
            h = pd.to_numeric(pd.Series([row.iloc[start + 2]]), errors="coerce").iloc[0]
            et = pd.to_numeric(pd.Series([row.iloc[start]]), errors="coerce").iloc[0]
            es = pd.to_numeric(pd.Series([row.iloc[start + 1]]), errors="coerce").iloc[0]
            if pd.isna(h):
                continue
            rows.append(
                {
                    "id": f"{mat}-{i}-{dop}",
                    "material": mat,
                    "dopant": dop,
                    "idx": i,
                    "h_ads_excel": float(h),
                    "target_computed": float(h),
                    "energy_total_excel": float(et) if pd.notna(et) else np.nan,
                    "energy_slab_excel": float(es) if pd.notna(es) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def find_audit_labels(raw: str) -> Optional[Path]:
    if raw and raw != "auto":
        p = Path(raw)
        return p if p.exists() else None
    for p in [
        Path("outputs_addh_llm_element_priors/addhout_audit_labels.csv"),
        Path("addH-out/addhout_audit_labels.csv"),
        Path("addH-out/氢吸附能.xlsx"),
        Path("addH-out/hydrogen_adsorption_energy.xlsx"),
        Path("addH-out/energy.xlsx"),
    ]:
        if p.exists():
            return p
    return None


def load_audit_labels(raw: str, id_col: str, target_col: str) -> Tuple[pd.DataFrame, Optional[Path]]:
    p = find_audit_labels(raw)
    if p is None:
        return pd.DataFrame(columns=[id_col, target_col]), None
    if p.suffix.lower() in {".xlsx", ".xls"}:
        try:
            df = read_table(p)
            if id_col not in df.columns or target_col not in df.columns:
                df = parse_addhout_wide_excel(p)
        except Exception:
            df = parse_addhout_wide_excel(p)
    else:
        df = read_table(p)
    if id_col not in df.columns and {"material", "idx", "dopant"}.issubset(df.columns):
        df[id_col] = (
            df["material"].astype(str)
            + "-"
            + pd.to_numeric(df["idx"], errors="coerce").fillna(-1).astype(int).astype(str)
            + "-"
            + df["dopant"].astype(str)
        )
    if target_col not in df.columns:
        for c in ["target", "target_computed", "h_ads", "H_ads"]:
            if c in df.columns:
                df[target_col] = pd.to_numeric(df[c], errors="coerce")
                break
    if id_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame(columns=[id_col, target_col]), p
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    keep = [id_col, target_col]
    for c in ["material", "dopant", "idx", "target_computed", "energy_total_excel", "energy_slab_excel"]:
        if c in df.columns and c not in keep:
            keep.append(c)
    return df[keep].dropna(subset=[id_col, target_col]).drop_duplicates(id_col), p


def first_existing(df: pd.DataFrame, cols: Sequence[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


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


def metric_rows(df: pd.DataFrame, pred_cols: Sequence[str], target_col: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    groups: List[Tuple[Optional[str], pd.DataFrame]] = [(None, df)]
    if "material" in df.columns:
        groups.extend([(str(g), s) for g, s in df.groupby("material", dropna=False)])
    for c in pred_cols:
        if c not in df.columns:
            continue
        for g, sub in groups:
            row: Dict[str, object] = {"pred_col": c, "target_col": target_col, "material": g}
            row.update(metrics(sub[target_col], sub[c]))
            rows.append(row)
    return pd.DataFrame(rows)


def qget(qtab: pd.DataFrame, dopant: str, q: float, fallback: float = np.nan) -> float:
    if dopant not in qtab.index or q not in qtab.columns:
        return float(fallback)
    v = qtab.loc[dopant, q]
    return float(v) if pd.notna(v) else float(fallback)


def rule_prior(qtab: pd.DataFrame, material: str, dopant: str, profile: str) -> Tuple[Optional[float], float, str]:
    """Return prior, blend weight, rule name."""
    material = str(material)
    dopant = str(dopant)
    if dopant not in qtab.index:
        return None, 0.0, "no_dopant_quantiles"

    if profile == "conservative":
        if material == "CeO2" and dopant in {"Cr", "Mo", "Ru", "Pt", "Rh", "Fe", "Co", "Pd", "Mn"}:
            prior = (
                0.35 * qget(qtab, dopant, 0.70)
                + 0.45 * qget(qtab, dopant, 0.75)
                + 0.20 * qget(qtab, dopant, 0.85)
            )
            return prior, 0.55, "CeO2_redox_tm_conservative"
        if material == "ZnO" and dopant in {"Mn", "Mo"}:
            prior = 0.30 * qget(qtab, dopant, 0.80) + 0.70 * qget(qtab, dopant, 0.90)
            return prior, 0.75, "ZnO_MnMo_positive_spike_conservative"
        if material == "ZnO" and dopant in {"Cr", "Ru"}:
            prior = (
                0.35 * qget(qtab, dopant, 0.65)
                + 0.45 * qget(qtab, dopant, 0.70)
                + 0.20 * qget(qtab, dopant, 0.80)
            )
            return prior, 0.65, "ZnO_CrRu_positive_spike_conservative"
        if material == "ZnO" and dopant == "Ce":
            return qget(qtab, dopant, 0.90), 0.75, "ZnO_Ce_redox_prior_conservative"
        if material == "ZnO" and dopant == "Fe":
            return 0.0, 0.65, "ZnO_Fe_near_zero_prior_conservative"
        return None, 0.0, "no_rule"

    # aggressive profile
    if material == "CeO2":
        if dopant in {"Cr", "Mo", "Pt"}:
            prior = (
                0.35 * qget(qtab, dopant, 0.75)
                + 0.45 * qget(qtab, dopant, 0.80)
                + 0.20 * qget(qtab, dopant, 0.85)
            )
            return prior, 0.72, "CeO2_CrMoPt_redox_spike_aggressive"
        if dopant == "Ru":
            return qget(qtab, dopant, 0.90) + 1.20, 0.78, "CeO2_Ru_redox_boost_aggressive"
        if dopant == "Rh":
            return qget(qtab, dopant, 0.90), 0.78, "CeO2_Rh_redox_boost_aggressive"
        if dopant in {"Fe", "Co", "Pd", "Mn"}:
            prior = (
                0.45 * qget(qtab, dopant, 0.70)
                + 0.45 * qget(qtab, dopant, 0.75)
                + 0.10 * qget(qtab, dopant, 0.85)
            )
            return prior, 0.58, "CeO2_late_tm_redox_spike_aggressive"
    if material == "ZnO":
        if dopant == "Mn":
            return qget(qtab, dopant, 0.90) + 1.50, 0.86, "ZnO_Mn_high_spike_aggressive"
        if dopant == "Mo":
            return qget(qtab, dopant, 0.90) + 0.50, 0.86, "ZnO_Mo_high_spike_aggressive"
        if dopant == "Cr":
            prior = (
                0.40 * qget(qtab, dopant, 0.65)
                + 0.45 * qget(qtab, dopant, 0.70)
                + 0.15 * qget(qtab, dopant, 0.75)
            )
            return prior, 0.78, "ZnO_Cr_positive_spike_aggressive"
        if dopant == "Ru":
            prior = 0.35 * qget(qtab, dopant, 0.75) + 0.65 * qget(qtab, dopant, 0.80)
            return prior, 0.78, "ZnO_Ru_positive_spike_aggressive"
        if dopant == "Ce":
            return qget(qtab, dopant, 0.90), 0.88, "ZnO_Ce_redox_prior_aggressive"
        if dopant == "Fe":
            return 0.0, 0.80, "ZnO_Fe_near_zero_prior_aggressive"
    return None, 0.0, "no_rule"


def apply_profile(df: pd.DataFrame, qtab: pd.DataFrame, anchor_col: str, profile: str) -> Tuple[pd.Series, pd.DataFrame]:
    pred = pd.to_numeric(df[anchor_col], errors="coerce").copy()
    rows: List[Dict[str, object]] = []
    for i, r in df.iterrows():
        mat = str(r.get("material", ""))
        dop = str(r.get("dopant", ""))
        anchor = float(pred.loc[i]) if pd.notna(pred.loc[i]) else np.nan
        prior, weight, rule = rule_prior(qtab, mat, dop, profile)
        applied = prior is not None and pd.notna(anchor) and np.isfinite(prior) and weight > 0
        if applied:
            pred.loc[i] = (1.0 - weight) * anchor + weight * float(prior)
        rows.append(
            {
                "id": r.get("id", ""),
                "material": mat,
                "dopant": dop,
                "profile": profile,
                "rule": rule,
                "applied": bool(applied),
                "anchor": anchor,
                "prior": prior,
                "blend_weight": weight if applied else 0.0,
                "prediction": float(pred.loc[i]) if pd.notna(pred.loc[i]) else np.nan,
                "delta_vs_anchor": float(pred.loc[i] - anchor) if applied else 0.0,
            }
        )
    return pred, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train_features)
    train["target"] = pd.to_numeric(train[args.target_col], errors="coerce")
    qtab = train.dropna(subset=["target"]).groupby("dopant")["target"].quantile(QUANTILES).unstack()
    qtab.to_csv(out_dir / "source_dopant_target_quantiles.csv")

    pred = pd.read_csv(args.pred_csv)
    if args.addhout_features and Path(args.addhout_features).exists():
        feat = pd.read_csv(args.addhout_features)
        keep = [args.id_col] + [c for c in ["material", "dopant", "idx"] if c in feat.columns and c not in pred.columns]
        if len(keep) > 1:
            pred = pred.merge(feat[keep].drop_duplicates(args.id_col), on=args.id_col, how="left")

    anchor_col = args.anchor_col
    if anchor_col == "auto":
        anchor_col = first_existing(
            pred,
            ["pred_superblend_final", "pred_superblend_mae_guarded", "pred_fast_target_calibrated", "pred_existing_anchor"],
        )
    if not anchor_col or anchor_col not in pred.columns:
        raise SystemExit("[ERROR] could not find anchor prediction column.")
    trend_col = args.trend_col if args.trend_col in pred.columns else first_existing(
        pred, ["pred_superblend_trend", "pred_rank_trend_calibrated", "pred_pretrained_delta_final"]
    )

    rule_tables: List[pd.DataFrame] = []
    if args.profile in {"conservative", "both"}:
        pred["pred_chem_spike_conservative"], rt = apply_profile(pred, qtab, anchor_col, "conservative")
        rule_tables.append(rt)
    if args.profile in {"aggressive", "both"}:
        pred["pred_chem_spike_aggressive"], rt = apply_profile(pred, qtab, anchor_col, "aggressive")
        rule_tables.append(rt)

    if "pred_chem_spike_conservative" not in pred.columns:
        pred["pred_chem_spike_conservative"] = pred[anchor_col]
    if "pred_chem_spike_aggressive" not in pred.columns:
        pred["pred_chem_spike_aggressive"] = pred[anchor_col]

    if trend_col:
        pred["pred_chem_spike_balanced"] = 0.80 * pred["pred_chem_spike_aggressive"] + 0.20 * pd.to_numeric(pred[trend_col], errors="coerce")
    else:
        pred["pred_chem_spike_balanced"] = pred["pred_chem_spike_aggressive"]

    if args.final_profile == "conservative":
        pred["pred_chem_spike_final"] = pred["pred_chem_spike_conservative"]
    elif args.final_profile == "balanced":
        pred["pred_chem_spike_final"] = pred["pred_chem_spike_balanced"]
    else:
        pred["pred_chem_spike_final"] = pred["pred_chem_spike_aggressive"]
    pred["chem_spike_anchor_col"] = anchor_col
    pred["chem_spike_trend_col"] = trend_col or ""

    rule_df = pd.concat(rule_tables, ignore_index=True) if rule_tables else pd.DataFrame()
    rule_df.to_csv(out_dir / "chemistry_spike_rule_applications.csv", index=False)

    labels, label_path = load_audit_labels(args.audit_labels_csv, args.id_col, args.audit_target_col)
    audit = pd.DataFrame()
    if len(labels):
        detail = pred.merge(labels, on=args.id_col, how="left", suffixes=("", "__label"))
        if "material" not in detail.columns and "material__label" in detail.columns:
            detail["material"] = detail["material__label"]
        if "dopant" not in detail.columns and "dopant__label" in detail.columns:
            detail["dopant"] = detail["dopant__label"]
        pred_cols = [
            "pred_chem_spike_final",
            "pred_chem_spike_aggressive",
            "pred_chem_spike_conservative",
            "pred_chem_spike_balanced",
            anchor_col,
            trend_col or "",
            "pred_rank_trend_calibrated",
            "pred_pretrained_delta_final",
            "pred_source_dopant_mean_prior",
        ]
        pred_cols = list(dict.fromkeys([c for c in pred_cols if c and c in detail.columns]))
        audit = metric_rows(detail, pred_cols, args.audit_target_col)
        detail.to_csv(out_dir / "chemistry_spike_posthoc_audit_detail.csv", index=False)
        audit.to_csv(out_dir / "chemistry_spike_posthoc_audit.csv", index=False)
        if args.write_xlsx:
            detail.to_excel(out_dir / "chemistry_spike_posthoc_audit_detail.xlsx", index=False)

    pred.to_csv(out_dir / "chemistry_spike_addhout_predictions.csv", index=False)
    if args.write_xlsx:
        pred.to_excel(out_dir / "chemistry_spike_addhout_predictions.xlsx", index=False)
        with pd.ExcelWriter(out_dir / "chemistry_spike_report.xlsx") as w:
            qtab.to_excel(w, sheet_name="source_quantiles")
            rule_df.to_excel(w, sheet_name="rule_applications", index=False)
            if len(audit):
                audit.to_excel(w, sheet_name="audit", index=False)

    manifest = {
        "script": Path(__file__).name,
        "prediction_file": args.pred_csv,
        "train_features": args.train_features,
        "anchor_col": anchor_col,
        "trend_col": trend_col,
        "profile": args.profile,
        "final_profile": args.final_profile,
        "audit_label_file": str(label_path) if label_path else None,
        "labels_used_at_runtime_for_prediction": False,
        "paper_note": (
            "This is a fixed chemistry-rule prior layer. Treat as hypothesis-driven "
            "chemistry prior unless the rules were finalized before viewing AddH-out labels."
        ),
        "outputs": {
            "predictions": str(out_dir / "chemistry_spike_addhout_predictions.csv"),
            "audit": str(out_dir / "chemistry_spike_posthoc_audit.csv"),
            "rules": str(out_dir / "chemistry_spike_rule_applications.csv"),
            "quantiles": str(out_dir / "source_dopant_target_quantiles.csv"),
        },
    }
    (out_dir / "chemistry_spike_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] wrote {out_dir}")
    if len(audit):
        print("[POSTHOC AUDIT ONLY]")
        print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
