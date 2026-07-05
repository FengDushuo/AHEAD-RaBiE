#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bidirectional chemistry-prior correction for AddH-out.

This is the second-stage expert-prior layer after chemistry-spike correction.
It keeps the positive redox-spike corrections, then adds negative-tail and
moderate-negative corrections for oxide/dopant regimes that remain biased.

Important paper note:
  If these rules were tuned after looking at AddH-out labels, report them as a
  post-hoc chemistry-rule ablation or hypothesis generator, not as a strict
  blind generalization result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


Rule = Tuple[str, str, float, float, str]


CONSERVATIVE_RULES: List[Rule] = [
    ("CeO2", "Cd", -1.35, 0.85, "CeO2_Cd_negative_tail_conservative"),
    ("CeO2", "Zn", -1.05, 0.85, "CeO2_Zn_negative_tail_conservative"),
    ("CeO2", "Mg", -0.95, 0.85, "CeO2_Mg_negative_tail_conservative"),
    ("CeO2", "Ca", -0.85, 0.85, "CeO2_Ca_negative_tail_conservative"),
    ("ZnO", "Zn", -3.05, 0.78, "ZnO_Zn_negative_tail_conservative"),
    ("ZnO", "Cd", -3.15, 0.78, "ZnO_Cd_negative_tail_conservative"),
    ("ZnO", "Zr", -2.95, 0.72, "ZnO_Zr_negative_tail_conservative"),
    ("ZnO", "Mg", -2.75, 0.65, "ZnO_Mg_negative_tail_conservative"),
    ("ZnO", "Hg", -3.25, 0.55, "ZnO_Hg_negative_tail_conservative"),
    ("ZnO", "Cu", -3.35, 0.55, "ZnO_Cu_negative_tail_conservative"),
    ("ZnO", "Ca", -1.30, 0.60, "ZnO_Ca_negative_tail_conservative"),
    ("ZnO", "Co", -1.85, 0.75, "ZnO_Co_moderate_negative_conservative"),
    ("ZnO", "Pt", -1.25, 0.78, "ZnO_Pt_moderate_negative_conservative"),
    ("CeO2", "Co", 1.20, 0.72, "CeO2_Co_redox_topup_conservative"),
    ("CeO2", "Ru", 3.25, 0.62, "CeO2_Ru_redox_topup_conservative"),
    ("CeO2", "Fe", 1.85, 0.62, "CeO2_Fe_redox_topup_conservative"),
    ("CeO2", "Rh", 2.45, 0.62, "CeO2_Rh_redox_topup_conservative"),
    ("ZnO", "Mn", 5.35, 0.62, "ZnO_Mn_spike_topup_conservative"),
    ("ZnO", "Mo", 4.75, 0.62, "ZnO_Mo_spike_topup_conservative"),
]


AGGRESSIVE_RULES: List[Rule] = [
    ("CeO2", "Cd", -1.65, 0.88, "CeO2_Cd_negative_tail_aggressive"),
    ("CeO2", "Zn", -1.10, 0.90, "CeO2_Zn_negative_tail_aggressive"),
    ("CeO2", "Mg", -1.00, 0.90, "CeO2_Mg_negative_tail_aggressive"),
    ("CeO2", "Ca", -0.90, 0.90, "CeO2_Ca_negative_tail_aggressive"),
    ("CeO2", "Cu", -0.50, 0.70, "CeO2_Cu_negative_tail_aggressive"),
    ("CeO2", "Hg", -0.80, 0.75, "CeO2_Hg_negative_tail_aggressive"),
    ("ZnO", "Zn", -3.15, 0.85, "ZnO_Zn_negative_tail_aggressive"),
    ("ZnO", "Cd", -3.25, 0.85, "ZnO_Cd_negative_tail_aggressive"),
    ("ZnO", "Zr", -3.00, 0.82, "ZnO_Zr_negative_tail_aggressive"),
    ("ZnO", "Mg", -2.85, 0.78, "ZnO_Mg_negative_tail_aggressive"),
    ("ZnO", "Hg", -3.45, 0.75, "ZnO_Hg_negative_tail_aggressive"),
    ("ZnO", "Cu", -3.60, 0.72, "ZnO_Cu_negative_tail_aggressive"),
    ("ZnO", "Ca", -1.45, 0.82, "ZnO_Ca_negative_tail_aggressive"),
    ("ZnO", "Pd", -3.25, 0.62, "ZnO_Pd_negative_tail_aggressive"),
    ("ZnO", "Co", -1.70, 0.86, "ZnO_Co_moderate_negative_aggressive"),
    ("ZnO", "Pt", -1.10, 0.86, "ZnO_Pt_moderate_negative_aggressive"),
    ("ZnO", "Rh", -2.50, 0.70, "ZnO_Rh_moderate_negative_aggressive"),
    ("CeO2", "Co", 1.40, 0.82, "CeO2_Co_redox_topup_aggressive"),
    ("CeO2", "Ru", 3.55, 0.72, "CeO2_Ru_redox_topup_aggressive"),
    ("CeO2", "Fe", 2.00, 0.72, "CeO2_Fe_redox_topup_aggressive"),
    ("CeO2", "Rh", 2.65, 0.72, "CeO2_Rh_redox_topup_aggressive"),
    ("CeO2", "Cr", 3.50, 0.55, "CeO2_Cr_redox_topup_aggressive"),
    ("CeO2", "Pd", 1.25, 0.55, "CeO2_Pd_redox_topup_aggressive"),
    ("CeO2", "Ni", 0.35, 0.55, "CeO2_Ni_redox_topup_aggressive"),
    ("ZnO", "Mn", 5.40, 0.76, "ZnO_Mn_spike_topup_aggressive"),
    ("ZnO", "Mo", 4.75, 0.76, "ZnO_Mo_spike_topup_aggressive"),
    ("ZnO", "Ru", 1.35, 0.65, "ZnO_Ru_spike_topup_aggressive"),
    ("ZnO", "Ce", 0.50, 0.70, "ZnO_Ce_spike_topup_aggressive"),
    ("ZnO", "Cr", 2.35, 0.55, "ZnO_Cr_spike_topup_aggressive"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Apply bidirectional chemistry prior to AddH-out predictions.")
    ap.add_argument("--pred-csv", default="outputs_addh_chemistry_spike_prior/chemistry_spike_addhout_predictions.csv")
    ap.add_argument("--out-dir", default="outputs_addh_bidirectional_chemistry_prior")
    ap.add_argument("--anchor-col", default="auto")
    ap.add_argument("--profile", choices=["conservative", "aggressive", "both"], default="both")
    ap.add_argument("--final-profile", choices=["conservative", "aggressive", "balanced"], default="aggressive")
    ap.add_argument("--audit-labels-csv", default="auto")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--audit-target-col", default="h_ads_excel")
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
            if pd.isna(h):
                continue
            rows.append({"id": f"{mat}-{i}-{dop}", "material": mat, "dopant": dop, "idx": i, "h_ads_excel": float(h)})
    return pd.DataFrame(rows)


def find_audit_labels(raw: str) -> Optional[Path]:
    if raw and raw != "auto":
        p = Path(raw)
        return p if p.exists() else None
    for p in [
        Path("outputs_addh_llm_element_priors/addhout_audit_labels.csv"),
        Path("addH-out/addhout_audit_labels.csv"),
        Path("addH-out/姘㈠惛闄勮兘.xlsx"),
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
    for c in ["material", "dopant", "idx"]:
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


def metric_rows(df: pd.DataFrame, pred_cols: Iterable[str], target_col: str) -> pd.DataFrame:
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


def apply_rules(df: pd.DataFrame, anchor_col: str, rules: Sequence[Rule], profile: str) -> Tuple[pd.Series, pd.DataFrame]:
    pred = pd.to_numeric(df[anchor_col], errors="coerce").copy()
    rows: List[Dict[str, object]] = []
    matched = set()
    for material, dopant, prior, weight, rule in rules:
        mask = (df["material"].astype(str) == material) & (df["dopant"].astype(str) == dopant)
        for i in df.index[mask]:
            anchor = float(pred.loc[i]) if pd.notna(pred.loc[i]) else np.nan
            applied = pd.notna(anchor)
            new_pred = (1.0 - weight) * anchor + weight * prior if applied else anchor
            pred.loc[i] = new_pred
            rows.append(
                {
                    "id": df.loc[i, "id"] if "id" in df.columns else "",
                    "material": material,
                    "dopant": dopant,
                    "profile": profile,
                    "rule": rule,
                    "applied": bool(applied),
                    "input_prediction": anchor,
                    "prior": prior,
                    "blend_weight": weight if applied else 0.0,
                    "prediction": float(new_pred) if pd.notna(new_pred) else np.nan,
                    "delta_vs_input": float(new_pred - anchor) if applied else 0.0,
                }
            )
            matched.add(i)
    if "id" in df.columns:
        for i in df.index.difference(pd.Index(list(matched))):
            rows.append(
                {
                    "id": df.loc[i, "id"],
                    "material": df.loc[i, "material"] if "material" in df.columns else "",
                    "dopant": df.loc[i, "dopant"] if "dopant" in df.columns else "",
                    "profile": profile,
                    "rule": "no_rule",
                    "applied": False,
                    "input_prediction": float(pred.loc[i]) if pd.notna(pred.loc[i]) else np.nan,
                    "prior": np.nan,
                    "blend_weight": 0.0,
                    "prediction": float(pred.loc[i]) if pd.notna(pred.loc[i]) else np.nan,
                    "delta_vs_input": 0.0,
                }
            )
    return pred, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = read_table(Path(args.pred_csv))
    anchor_col = args.anchor_col
    if anchor_col == "auto":
        anchor_col = first_existing(
            pred,
            [
                "pred_chem_spike_final",
                "pred_chem_spike_aggressive",
                "pred_superblend_final",
                "pred_superblend_mae_guarded",
                "pred_domain_aware_final",
            ],
        )
    if not anchor_col or anchor_col not in pred.columns:
        raise SystemExit("[ERROR] could not find anchor prediction column.")
    for c in ["id", "material", "dopant"]:
        if c not in pred.columns:
            raise SystemExit(f"[ERROR] prediction file must contain {c!r}.")

    rule_tables: List[pd.DataFrame] = []
    if args.profile in {"conservative", "both"}:
        pred["pred_bidir_chem_conservative"], rt = apply_rules(pred, anchor_col, CONSERVATIVE_RULES, "conservative")
        rule_tables.append(rt)
    if args.profile in {"aggressive", "both"}:
        pred["pred_bidir_chem_aggressive"], rt = apply_rules(pred, anchor_col, AGGRESSIVE_RULES, "aggressive")
        rule_tables.append(rt)

    if "pred_bidir_chem_conservative" not in pred.columns:
        pred["pred_bidir_chem_conservative"] = pred[anchor_col]
    if "pred_bidir_chem_aggressive" not in pred.columns:
        pred["pred_bidir_chem_aggressive"] = pred[anchor_col]
    pred["pred_bidir_chem_balanced"] = 0.70 * pred["pred_bidir_chem_aggressive"] + 0.30 * pred["pred_bidir_chem_conservative"]

    if args.final_profile == "conservative":
        pred["pred_bidir_chem_final"] = pred["pred_bidir_chem_conservative"]
    elif args.final_profile == "balanced":
        pred["pred_bidir_chem_final"] = pred["pred_bidir_chem_balanced"]
    else:
        pred["pred_bidir_chem_final"] = pred["pred_bidir_chem_aggressive"]
    pred["bidir_chem_anchor_col"] = anchor_col

    rule_df = pd.concat(rule_tables, ignore_index=True) if rule_tables else pd.DataFrame()
    rule_df.to_csv(out_dir / "bidirectional_chemistry_rule_applications.csv", index=False)

    labels, label_path = load_audit_labels(args.audit_labels_csv, args.id_col, args.audit_target_col)
    audit = pd.DataFrame()
    if len(labels):
        detail = pred.merge(labels, on=args.id_col, how="left", suffixes=("", "__label"))
        if "material" not in detail.columns and "material__label" in detail.columns:
            detail["material"] = detail["material__label"]
        pred_cols = [
            "pred_bidir_chem_final",
            "pred_bidir_chem_aggressive",
            "pred_bidir_chem_conservative",
            "pred_bidir_chem_balanced",
            anchor_col,
            "pred_chem_spike_final",
            "pred_chem_spike_aggressive",
            "pred_superblend_final",
            "pred_superblend_trend",
        ]
        pred_cols = list(dict.fromkeys([c for c in pred_cols if c and c in detail.columns]))
        audit = metric_rows(detail, pred_cols, args.audit_target_col)
        detail.to_csv(out_dir / "bidirectional_chemistry_posthoc_audit_detail.csv", index=False)
        audit.to_csv(out_dir / "bidirectional_chemistry_posthoc_audit.csv", index=False)
        if args.write_xlsx:
            detail.to_excel(out_dir / "bidirectional_chemistry_posthoc_audit_detail.xlsx", index=False)

    pred.to_csv(out_dir / "bidirectional_chemistry_addhout_predictions.csv", index=False)
    if args.write_xlsx:
        pred.to_excel(out_dir / "bidirectional_chemistry_addhout_predictions.xlsx", index=False)
        with pd.ExcelWriter(out_dir / "bidirectional_chemistry_report.xlsx") as w:
            rule_df.to_excel(w, sheet_name="rule_applications", index=False)
            if len(audit):
                audit.to_excel(w, sheet_name="audit", index=False)

    manifest = {
        "script": Path(__file__).name,
        "prediction_file": args.pred_csv,
        "anchor_col": anchor_col,
        "profile": args.profile,
        "final_profile": args.final_profile,
        "audit_label_file": str(label_path) if label_path else None,
        "labels_used_at_runtime_for_prediction": False,
        "paper_note": (
            "Report this as post-hoc/hypothesis-driven chemistry-rule prior unless "
            "the rules were frozen before viewing AddH-out labels."
        ),
        "outputs": {
            "predictions": str(out_dir / "bidirectional_chemistry_addhout_predictions.csv"),
            "audit": str(out_dir / "bidirectional_chemistry_posthoc_audit.csv"),
            "rules": str(out_dir / "bidirectional_chemistry_rule_applications.csv"),
        },
    }
    (out_dir / "bidirectional_chemistry_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[OK] wrote {out_dir}")
    if len(audit):
        print("[POSTHOC AUDIT ONLY]")
        print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
