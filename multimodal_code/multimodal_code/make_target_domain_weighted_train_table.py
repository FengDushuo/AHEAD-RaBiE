#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_target_domain_weighted_train_table.py

Build a target-domain weighted training table.

Purpose
-------
Given:
1) a source training table (e.g. addH_master_merged_robust.csv)
2) a target-domain table (e.g. addH_out_master_normalized.csv or addH_out_master.csv)

the script computes a target-domain similarity weight for each training row,
so rows that are more similar to the target domain receive larger weights.

Typical use cases
-----------------
- Train CatBoost with Pool(..., weight=train_df["w_domain"])
- Train NN with weighted sampler / weighted loss
- Inspect which source samples look most target-like

Main ideas
----------
The total weight is built from several components:

    w_domain = w_material * w_miller * w_site * w_anchor *
               w_dopant * w_formula * w_emb(optional)

Then:
- clip to [min_weight, max_weight]
- normalize to mean 1.0

Output files
------------
1) weighted train csv:
   addH_master_target_weighted.csv
2) debug csv:
   target_domain_weight_debug.csv
3) target profile json:
   target_domain_profile.json

Notes
-----
- The script prioritizes normalized standard fields:
    target, dopant, family_base, family_base_miller, miller, site_type, anchor_count, slab_formula, text
- It also supports legacy target-domain fields through fallbacks:
    element -> dopant
    material -> family_base
    h_ads_excel / target_computed -> target
- Optional embedding weighting:
    if both train and target embedding dicts are provided (id -> vector),
    the script adds an embedding-similarity component.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from pymatgen.core import Element
except Exception:
    Element = None


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train-csv", required=True, help="Source training table, e.g. addH_master_merged_robust.csv")
    ap.add_argument("--target-csv", required=True, help="Target-domain table, e.g. addH_out_master_normalized.csv")

    ap.add_argument("--train-eq-emb-pkl", default=None, help="Optional dict[id] -> embedding for training samples")
    ap.add_argument("--target-eq-emb-pkl", default=None, help="Optional dict[id] -> embedding for target-domain samples")

    ap.add_argument("--output-csv", default="addH_master_target_weighted.csv")
    ap.add_argument("--debug-csv", default="target_domain_weight_debug.csv")
    ap.add_argument("--profile-json", default="target_domain_profile.json")

    # weight ranges
    ap.add_argument("--min-weight", type=float, default=0.50)
    ap.add_argument("--max-weight", type=float, default=4.00)

    # component toggles
    ap.add_argument("--use-material", action="store_true", default=True)
    ap.add_argument("--use-miller", action="store_true", default=True)
    ap.add_argument("--use-site", action="store_true", default=True)
    ap.add_argument("--use-anchor", action="store_true", default=True)
    ap.add_argument("--use-dopant", action="store_true", default=True)
    ap.add_argument("--use-formula", action="store_true", default=True)
    ap.add_argument("--use-emb", action="store_true", default=False)

    # multipliers / strengths
    ap.add_argument("--material-exact", type=float, default=2.50)
    ap.add_argument("--material-family", type=float, default=1.80)
    ap.add_argument("--material-oxide", type=float, default=1.20)
    ap.add_argument("--material-other", type=float, default=0.70)

    ap.add_argument("--miller-exact", type=float, default=1.50)
    ap.add_argument("--miller-known-mismatch", type=float, default=1.00)
    ap.add_argument("--miller-unknown", type=float, default=0.80)

    ap.add_argument("--site-min", type=float, default=0.80)
    ap.add_argument("--site-max", type=float, default=1.40)

    ap.add_argument("--anchor-min", type=float, default=0.80)
    ap.add_argument("--anchor-max", type=float, default=1.30)

    ap.add_argument("--dopant-alpha", type=float, default=0.35)
    ap.add_argument("--dopant-min", type=float, default=0.70)
    ap.add_argument("--dopant-max", type=float, default=1.60)

    ap.add_argument("--formula-oxide", type=float, default=1.20)
    ap.add_argument("--formula-other", type=float, default=0.90)

    ap.add_argument("--emb-min", type=float, default=0.80)
    ap.add_argument("--emb-max", type=float, default=1.50)

    ap.add_argument("--normalize-to-mean1", action="store_true", default=True)
    ap.add_argument("--strict", action="store_true")
    return ap.parse_args()


_FORMULA_TOK_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def _load_emb(path: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Embedding file must be dict[id] -> vector, got {type(obj)}")
    out: Dict[str, np.ndarray] = {}
    for k, v in obj.items():
        out[str(k)] = np.asarray(v, dtype=np.float32).reshape(-1)
    return out


def _normalize_miller_text(x: Any) -> str:
    if pd.isna(x):
        return "(? ? ?)"
    s = str(x).strip()
    if not s:
        return "(? ? ?)"
    s = s.replace("[", "(").replace("]", ")")
    s = re.sub(r"\s+", " ", s)
    if re.fullmatch(r"\d{3}", s):
        return f"({s[0]} {s[1]} {s[2]})"
    m = re.fullmatch(r"\(?\s*(\d)\s+(\d)\s+(\d)\s*\)?", s)
    if m:
        return f"({m.group(1)} {m.group(2)} {m.group(3)})"
    return s


def _miller_key(miller_text: str) -> str:
    s = _normalize_miller_text(miller_text)
    m = re.fullmatch(r"\((\d)\s+(\d)\s+(\d)\)", s)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    return "unknown"


def _parse_formula_counts(formula: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not isinstance(formula, str) or not formula:
        return counts
    for el, n in _FORMULA_TOK_RE.findall(formula):
        counts[el] = counts.get(el, 0) + (int(n) if n else 1)
    return counts


def _is_oxide_formula(formula: str) -> bool:
    cnt = _parse_formula_counts(formula)
    return ("O" in cnt) and (len(cnt) >= 2)


def _standardize_schema(df: pd.DataFrame, is_target: bool) -> pd.DataFrame:
    out = df.copy()

    if "id" not in out.columns:
        raise ValueError("Expected 'id' column")

    if "target" not in out.columns:
        if "h_ads_excel" in out.columns:
            out["target"] = pd.to_numeric(out["h_ads_excel"], errors="coerce")
        elif "target_computed" in out.columns:
            out["target"] = pd.to_numeric(out["target_computed"], errors="coerce")
        else:
            out["target"] = np.nan
    else:
        out["target"] = pd.to_numeric(out["target"], errors="coerce")

    if "dopant" not in out.columns:
        if "element" in out.columns:
            out["dopant"] = out["element"]
        else:
            out["dopant"] = "unknown"
    out["dopant"] = out["dopant"].fillna("unknown").astype(str)

    if "family_base" not in out.columns:
        if "material" in out.columns:
            out["family_base"] = out["material"]
        else:
            out["family_base"] = "unknown"
    out["family_base"] = out["family_base"].fillna("unknown").astype(str)

    if "miller" not in out.columns:
        out["miller"] = "(? ? ?)"
    out["miller"] = out["miller"].map(_normalize_miller_text)

    if "family_base_miller" not in out.columns:
        out["family_base_miller"] = out.apply(
            lambda r: f"{r['family_base']}-{_miller_key(r['miller'])}" if _miller_key(r["miller"]) != "unknown" else str(r["family_base"]),
            axis=1,
        )
    else:
        miss = out["family_base_miller"].isna() | (out["family_base_miller"].astype(str).str.strip() == "")
        if miss.any():
            out.loc[miss, "family_base_miller"] = out.loc[miss].apply(
                lambda r: f"{r['family_base']}-{_miller_key(r['miller'])}" if _miller_key(r["miller"]) != "unknown" else str(r["family_base"]),
                axis=1,
            )
    out["family_base_miller"] = out["family_base_miller"].fillna("unknown").astype(str)

    if "site_type" not in out.columns:
        out["site_type"] = "unknown"
    out["site_type"] = out["site_type"].fillna("unknown").astype(str)

    if "anchor_count" not in out.columns:
        out["anchor_count"] = np.nan
    out["anchor_count"] = pd.to_numeric(out["anchor_count"], errors="coerce")

    if "slab_formula" not in out.columns:
        out["slab_formula"] = "unknown"
    out["slab_formula"] = out["slab_formula"].fillna("unknown").astype(str)

    if "text" not in out.columns:
        if "text_raw" in out.columns:
            out["text"] = out["text_raw"]
        else:
            out["text"] = ""
    out["text"] = out["text"].fillna("").astype(str)

    if "data_source" not in out.columns:
        out["data_source"] = "target_domain" if is_target else "source_train"

    return out


def _add_dopant_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    z_list, row_list, group_list, block_list = [], [], [], []
    is_tm, is_post, is_met, is_lanth, is_act = [], [], [], [], []

    for dop in out["dopant"].fillna("").astype(str):
        if not dop or Element is None:
            z_list.append(np.nan); row_list.append(np.nan); group_list.append(np.nan); block_list.append("unknown")
            is_tm.append(0); is_post.append(0); is_met.append(0); is_lanth.append(0); is_act.append(0)
            continue
        try:
            el = Element(dop)
            z_list.append(float(el.Z))
            row_list.append(float(el.row) if el.row is not None else np.nan)
            group_list.append(float(el.group) if el.group is not None else np.nan)
            block_list.append(str(el.block) if el.block is not None else "unknown")
            is_tm.append(int(bool(getattr(el, "is_transition_metal", False))))
            is_post.append(int(bool(getattr(el, "is_post_transition_metal", False))))
            is_met.append(int(bool(getattr(el, "is_metalloid", False))))
            is_lanth.append(int(bool(getattr(el, "is_lanthanoid", False))))
            is_act.append(int(bool(getattr(el, "is_actinoid", False))))
        except Exception:
            z_list.append(np.nan); row_list.append(np.nan); group_list.append(np.nan); block_list.append("unknown")
            is_tm.append(0); is_post.append(0); is_met.append(0); is_lanth.append(0); is_act.append(0)

    out["dopant_Z"] = z_list
    out["dopant_row"] = row_list
    out["dopant_group_num"] = group_list
    out["dopant_block"] = block_list
    out["dopant_is_transition"] = is_tm
    out["dopant_is_post_transition"] = is_post
    out["dopant_is_metalloid"] = is_met
    out["dopant_is_lanthanoid"] = is_lanth
    out["dopant_is_actinoid"] = is_act
    return out


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (da * db))


def _clip_scale_from_similarity(sim: float, out_min: float, out_max: float) -> float:
    s = (sim + 1.0) / 2.0
    return out_min + s * (out_max - out_min)


def build_target_profile(target_df: pd.DataFrame, target_emb: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
    prof: Dict[str, Any] = {}

    prof["n_rows"] = int(len(target_df))
    prof["target_materials"] = sorted(target_df["family_base"].dropna().astype(str).unique().tolist())
    prof["target_family_base_miller"] = sorted(target_df["family_base_miller"].dropna().astype(str).unique().tolist())
    prof["target_millers"] = sorted(target_df["miller"].dropna().astype(str).unique().tolist())

    site_freq = target_df["site_type"].fillna("unknown").astype(str).value_counts(normalize=True).to_dict()
    prof["site_freq"] = {str(k): float(v) for k, v in site_freq.items()}

    anchor = pd.to_numeric(target_df["anchor_count"], errors="coerce")
    prof["anchor_mean"] = float(anchor.mean()) if anchor.notna().any() else None
    prof["anchor_std"] = float(anchor.std()) if anchor.notna().any() else None

    prof["oxide_fraction"] = float(np.mean(target_df["slab_formula"].astype(str).map(_is_oxide_formula))) if len(target_df) > 0 else 0.0

    dop_num_cols = [
        "dopant_Z", "dopant_row", "dopant_group_num",
        "dopant_is_transition", "dopant_is_post_transition",
        "dopant_is_metalloid", "dopant_is_lanthanoid", "dopant_is_actinoid",
    ]
    dop_center = {}
    for c in dop_num_cols:
        s = pd.to_numeric(target_df[c], errors="coerce")
        dop_center[c] = float(s.mean()) if s.notna().any() else None
    prof["dopant_numeric_center"] = dop_center

    block_freq = target_df["dopant_block"].fillna("unknown").astype(str).value_counts(normalize=True).to_dict()
    prof["dopant_block_freq"] = {str(k): float(v) for k, v in block_freq.items()}

    if target_emb is not None and len(target_emb) > 0:
        ids = target_df["id"].astype(str).tolist()
        vecs = [target_emb[i] for i in ids if i in target_emb]
        if vecs:
            dims = sorted(set(v.shape[0] for v in vecs))
            if len(dims) == 1:
                center = np.mean(np.stack(vecs, axis=0), axis=0)
                prof["emb_center_dim"] = int(center.shape[0])
                prof["emb_center"] = center.tolist()
            else:
                prof["emb_center_dim"] = None
                prof["emb_center"] = None
        else:
            prof["emb_center_dim"] = None
            prof["emb_center"] = None
    else:
        prof["emb_center_dim"] = None
        prof["emb_center"] = None

    return prof


def score_material(row: pd.Series, target_profile: Dict[str, Any], args) -> float:
    fbm = str(row["family_base_miller"])
    fb = str(row["family_base"])
    target_fbm = set(target_profile["target_family_base_miller"])
    target_fb = set(target_profile["target_materials"])

    if fbm in target_fbm:
        return float(args.material_exact)
    if fb in target_fb:
        return float(args.material_family)
    if _is_oxide_formula(str(row.get("slab_formula", ""))):
        return float(args.material_oxide)
    return float(args.material_other)


def score_miller(row: pd.Series, target_profile: Dict[str, Any], args) -> float:
    m = str(row["miller"])
    target_millers = set(target_profile["target_millers"])
    if "?" in m:
        return float(args.miller_unknown)
    if m in target_millers:
        return float(args.miller_exact)
    return float(args.miller_known_mismatch)


def score_site(row: pd.Series, target_profile: Dict[str, Any], args) -> float:
    s = str(row["site_type"])
    freq = target_profile.get("site_freq", {})
    p = float(freq.get(s, 0.0))
    if p <= 0:
        return float(args.site_min)
    maxp = max(freq.values()) if len(freq) > 0 else 1.0
    t = p / max(maxp, 1e-12)
    return float(args.site_min + t * (args.site_max - args.site_min))


def score_anchor(row: pd.Series, target_profile: Dict[str, Any], args) -> float:
    a = pd.to_numeric(row.get("anchor_count"), errors="coerce")
    mu = target_profile.get("anchor_mean", None)
    sd = target_profile.get("anchor_std", None)

    if pd.isna(a) or mu is None or sd is None or not np.isfinite(mu):
        return 1.0
    sd = max(float(sd) if (sd is not None and np.isfinite(sd)) else 0.0, 0.5)
    sim = math.exp(-abs(float(a) - float(mu)) / sd)
    return float(args.anchor_min + sim * (args.anchor_max - args.anchor_min))


def score_dopant(row: pd.Series, target_profile: Dict[str, Any], args) -> float:
    cols = [
        "dopant_Z", "dopant_row", "dopant_group_num",
        "dopant_is_transition", "dopant_is_post_transition",
        "dopant_is_metalloid", "dopant_is_lanthanoid", "dopant_is_actinoid",
    ]
    center = target_profile.get("dopant_numeric_center", {})

    x = []
    c = []
    for col in cols:
        xv = pd.to_numeric(row.get(col), errors="coerce")
        cv = center.get(col, None)
        if pd.notna(xv) and cv is not None and np.isfinite(cv):
            x.append(float(xv))
            c.append(float(cv))
    if len(x) == 0:
        return 1.0

    x = np.asarray(x, dtype=float)
    c = np.asarray(c, dtype=float)
    d = float(np.linalg.norm(x - c) / max(len(x), 1))
    sim = math.exp(-float(args.dopant_alpha) * d)
    score = float(args.dopant_min + sim * (args.dopant_max - args.dopant_min))

    bf = target_profile.get("dopant_block_freq", {})
    block = str(row.get("dopant_block", "unknown"))
    if block in bf and len(bf) > 0 and bf[block] > 0:
        score *= 1.0 + 0.1 * min(1.0, float(bf[block]) / max(bf.values()))
    return float(score)


def score_formula(row: pd.Series, target_profile: Dict[str, Any], args) -> float:
    formula = str(row.get("slab_formula", ""))
    return float(args.formula_oxide if _is_oxide_formula(formula) else args.formula_other)


def score_emb(row: pd.Series, target_profile: Dict[str, Any], train_emb: Optional[Dict[str, np.ndarray]], args) -> float:
    if train_emb is None or len(train_emb) == 0:
        return 1.0
    center = target_profile.get("emb_center", None)
    if center is None:
        return 1.0

    sid = str(row["id"])
    if sid not in train_emb:
        return 1.0

    v = train_emb[sid]
    c = np.asarray(center, dtype=np.float32).reshape(-1)
    if v.shape[0] != c.shape[0]:
        return 1.0

    sim = _cosine_similarity(v, c)
    return float(_clip_scale_from_similarity(sim, float(args.emb_min), float(args.emb_max)))


def save_json(path: Path, obj: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()

    train_df = pd.read_csv(args.train_csv)
    target_df = pd.read_csv(args.target_csv)

    train_df = _standardize_schema(train_df, is_target=False)
    target_df = _standardize_schema(target_df, is_target=True)

    train_df = _add_dopant_descriptors(train_df)
    target_df = _add_dopant_descriptors(target_df)

    train_emb = _load_emb(args.train_eq_emb_pkl)
    target_emb = _load_emb(args.target_eq_emb_pkl)

    target_profile = build_target_profile(target_df, target_emb=target_emb)

    rows = []
    for _, row in train_df.iterrows():
        wm = score_material(row, target_profile, args) if args.use_material else 1.0
        wmi = score_miller(row, target_profile, args) if args.use_miller else 1.0
        ws = score_site(row, target_profile, args) if args.use_site else 1.0
        wa = score_anchor(row, target_profile, args) if args.use_anchor else 1.0
        wd = score_dopant(row, target_profile, args) if args.use_dopant else 1.0
        wf = score_formula(row, target_profile, args) if args.use_formula else 1.0
        we = score_emb(row, target_profile, train_emb, args) if args.use_emb else 1.0

        w_domain = wm * wmi * ws * wa * wd * wf * we
        rows.append((wm, wmi, ws, wa, wd, wf, we, w_domain))

    score_df = pd.DataFrame(rows, columns=[
        "w_material", "w_miller", "w_site", "w_anchor", "w_dopant", "w_formula", "w_emb", "w_domain_raw"
    ])

    out_df = pd.concat([train_df.reset_index(drop=True), score_df], axis=1)

    out_df["w_domain"] = out_df["w_domain_raw"].clip(lower=float(args.min_weight), upper=float(args.max_weight))
    if args.normalize_to_mean1 and len(out_df) > 0:
        mean_w = float(out_df["w_domain"].mean())
        if mean_w > 1e-12:
            out_df["w_domain"] = out_df["w_domain"] / mean_w

    target_fbm = set(target_profile["target_family_base_miller"])
    target_fb = set(target_profile["target_materials"])

    def _target_like_group(row):
        fbm = str(row["family_base_miller"])
        fb = str(row["family_base"])
        if fbm in target_fbm:
            return f"{fbm}_like"
        if fb in target_fb:
            return f"{fb}_family_like"
        if _is_oxide_formula(str(row.get('slab_formula', ''))):
            return "oxide_like"
        return "other"

    out_df["is_target_like"] = out_df["family_base"].astype(str).isin(target_fb) | out_df["family_base_miller"].astype(str).isin(target_fbm)
    out_df["target_domain_group"] = out_df.apply(_target_like_group, axis=1)

    out_df.to_csv(args.output_csv, index=False)

    debug_cols = [
        "id", "target", "family_base", "family_base_miller", "dopant", "miller", "site_type", "anchor_count",
        "slab_formula", "is_target_like", "target_domain_group",
        "w_material", "w_miller", "w_site", "w_anchor", "w_dopant", "w_formula", "w_emb", "w_domain_raw", "w_domain",
    ]
    debug_cols = [c for c in debug_cols if c in out_df.columns]
    debug_df = out_df[debug_cols].sort_values("w_domain", ascending=False)
    debug_df.to_csv(args.debug_csv, index=False)

    profile_out = dict(target_profile)
    profile_out["settings"] = {
        "min_weight": float(args.min_weight),
        "max_weight": float(args.max_weight),
        "normalize_to_mean1": bool(args.normalize_to_mean1),
        "use_emb": bool(args.use_emb),
    }
    save_json(Path(args.profile_json), profile_out)

    print(f"[OK] weighted train csv -> {args.output_csv}")
    print(f"[OK] debug csv          -> {args.debug_csv}")
    print(f"[OK] profile json       -> {args.profile_json}")
    print(f"[INFO] train rows       = {len(out_df)}")
    print(f"[INFO] target rows      = {len(target_df)}")
    print(f"[INFO] w_domain mean    = {out_df['w_domain'].mean():.6f}")
    print(f"[INFO] w_domain min/max = {out_df['w_domain'].min():.6f} / {out_df['w_domain'].max():.6f}")
    print(f"[INFO] target materials = {target_profile['target_materials']}")
    print(f"[INFO] target fbm       = {target_profile['target_family_base_miller']}")


if __name__ == "__main__":
    main()
