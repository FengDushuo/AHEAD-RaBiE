#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_make_singleview_data_cv_meta.py

Updated version:
- prioritizes normalized standard fields:
    * target
    * dopant
    * family_base_miller
- also supports legacy addH_out_master.csv style fields through fallbacks:
    * element -> dopant
    * material -> family_base
    * h_ads_excel / target_computed -> target
    * text -> text_raw

Output per fold
---------------
fold_k/
  split_assignments.csv
  nn_train.pkl
  nn_val.pkl
  nn_test.pkl
  cat_train.pkl
  cat_val.pkl
  cat_test.pkl
  addH_out_nn_pred_input.pkl
  addH_out_cat_pred_input.pkl
  addH_out_pred_manifest.csv
  fold_meta.json

Root output
-----------
cv_summary.csv
dataset_info.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
except Exception as e:
    raise SystemExit(f"scikit-learn is required: {e}")

try:
    from pymatgen.core import Element
except Exception:
    Element = None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addh-master-csv", required=True)
    ap.add_argument("--eq-emb-pkl", required=True, help="dict[id] -> addH structure embedding")
    ap.add_argument("--addhout-master-csv", default=None)
    ap.add_argument("--addhout-eq-emb-pkl", default=None)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--group-col", default="family_base_miller")
    ap.add_argument("--cv-mode", default="gkfold", choices=["single", "gkfold", "logo"])
    ap.add_argument("--n-splits", type=int, default=4)
    ap.add_argument("--val-mode", default="group_holdout", choices=["group_holdout", "group_kfold_inner"])
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--text-col", default="text")
    ap.add_argument("--strict", action="store_true")
    return ap.parse_args()


def _load_emb(path: Path) -> Dict[str, object]:
    import pickle
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Embedding file must be dict[id] -> emb, got {type(obj)}")
    return obj


def _json_safe(x: Any):
    import numpy as _np
    if isinstance(x, (_np.integer,)):
        return int(x)
    if isinstance(x, (_np.floating,)):
        return float(x)
    if isinstance(x, (_np.bool_,)):
        return bool(x)
    return x


def _json_safe_list(xs: Sequence[Any]):
    return [_json_safe(x) for x in xs]


def _assign_groups_single(groups: Sequence[object], seed: int, val_frac: float):
    rng = np.random.default_rng(seed)
    groups = np.array(sorted(list(set(groups))))
    rng.shuffle(groups)
    n = len(groups)
    if n < 3:
        raise ValueError("Need at least 3 unique groups for single split")
    n_test = 1 if n <= 4 else max(1, round(n * 0.25))
    n_val = 1 if n <= 4 else max(1, round((n - n_test) * val_frac))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("Not enough groups for train split")
    train_g = set(groups[:n_train])
    val_g = set(groups[n_train:n_train + n_val])
    test_g = set(groups[n_train + n_val:])
    return train_g, val_g, test_g


def _make_outer_splits(groups: Sequence[object], cv_mode: str, n_splits: int, seed: int):
    groups = np.asarray(groups)
    dummy_X = np.zeros((len(groups), 1), dtype=np.float32)
    dummy_y = np.zeros(len(groups), dtype=np.float32)

    if cv_mode == "single":
        uniq = np.array(sorted(list(set(groups))))
        train_g, val_g, test_g = _assign_groups_single(uniq, seed=seed, val_frac=0.25)
        trainval_idx = np.where(np.isin(groups, list(train_g | val_g)))[0]
        test_idx = np.where(np.isin(groups, list(test_g)))[0]
        return [(trainval_idx, test_idx)]

    uniq_n = len(set(groups))
    if cv_mode == "gkfold":
        if uniq_n < 2:
            raise ValueError("Need at least 2 unique groups for GroupKFold")
        n_splits_eff = min(n_splits, uniq_n)
        splitter = GroupKFold(n_splits=n_splits_eff)
        return list(splitter.split(dummy_X, dummy_y, groups=groups))

    if cv_mode == "logo":
        splitter = LeaveOneGroupOut()
        return list(splitter.split(dummy_X, dummy_y, groups=groups))

    raise ValueError(f"Unsupported cv_mode: {cv_mode}")


def _choose_val_groups_from_train(
    outer_train_df: pd.DataFrame,
    group_col: str,
    val_mode: str,
    val_frac: float,
    seed: int,
    outer_fold_idx: int,
):
    train_groups = np.array(sorted(outer_train_df[group_col].dropna().unique()))
    n_groups = len(train_groups)
    if n_groups < 2:
        raise ValueError(f"Outer fold {outer_fold_idx}: need at least 2 train groups to split train/val")

    if val_mode == "group_holdout":
        rng = np.random.default_rng(seed + outer_fold_idx)
        groups = train_groups.copy()
        rng.shuffle(groups)
        n_val = max(1, round(n_groups * val_frac))
        if n_val >= n_groups:
            n_val = n_groups - 1
        val_g = set(groups[:n_val])
        train_g = set(groups[n_val:])
        return train_g, val_g

    if val_mode == "group_kfold_inner":
        n_inner = max(2, min(4, n_groups))
        inner = GroupKFold(n_splits=n_inner)
        dummy_X = np.zeros((len(outer_train_df), 1), dtype=np.float32)
        dummy_y = np.zeros(len(outer_train_df), dtype=np.float32)
        inner_splits = list(inner.split(dummy_X, dummy_y, groups=outer_train_df[group_col].to_numpy()))
        inner_train_idx, inner_val_idx = inner_splits[0]
        train_g = set(outer_train_df.iloc[inner_train_idx][group_col].unique())
        val_g = set(outer_train_df.iloc[inner_val_idx][group_col].unique())
        return train_g, val_g

    raise ValueError(f"Unsupported val_mode: {val_mode}")


_FORMULA_TOK_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def _parse_formula_counts(formula: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not isinstance(formula, str) or not formula:
        return counts
    for el, n in _FORMULA_TOK_RE.findall(formula):
        counts[el] = counts.get(el, 0) + (int(n) if n else 1)
    return counts


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


def _text_structured(row: pd.Series, text_col: str) -> str:
    parts = []
    if pd.notna(row.get("dopant")):
        parts.append(f"dopant={row['dopant']}")
    if pd.notna(row.get("miller")):
        parts.append(f"surface={row['miller']}")
    if pd.notna(row.get("site_type")):
        parts.append(f"site={row['site_type']}")
    if pd.notna(row.get("anchor_count")):
        parts.append(f"anchors={row['anchor_count']}")
    if pd.notna(row.get("family_base_miller")):
        parts.append(f"group={row['family_base_miller']}")
    raw = row.get(text_col, "")
    if pd.notna(raw) and str(raw).strip():
        parts.append(f"raw={raw}")
    return " ; ".join(parts)


def _target_bin_from_train(train_targets: pd.Series, series: pd.Series, n_bins: int = 5) -> pd.Series:
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(train_targets.astype(float), qs))
    if len(edges) <= 2:
        return pd.Series(np.zeros(len(series), dtype=int), index=series.index)
    b = pd.cut(series.astype(float), bins=edges, include_lowest=True, labels=False, duplicates="drop")
    return b.fillna(-1).astype(int)


def _derive_dopant_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = df.copy()
    cols_num = []
    cols_cat = []
    if "dopant" not in df.columns:
        return df, cols_num, cols_cat

    z_list, row_list, group_list = [], [], []
    block_list = []
    is_tm, is_post, is_met, is_lanth, is_act = [], [], [], [], []

    for dop in df["dopant"].fillna("").astype(str):
        if not dop or Element is None:
            z_list.append(np.nan); row_list.append(np.nan); group_list.append(np.nan)
            block_list.append("unknown")
            is_tm.append(0); is_post.append(0); is_met.append(0); is_lanth.append(0); is_act.append(0)
            continue
        try:
            el = Element(dop)
            z_list.append(int(el.Z))
            row_list.append(int(el.row) if el.row is not None else np.nan)
            group_list.append(int(el.group) if el.group is not None else np.nan)
            block_list.append(str(el.block) if el.block is not None else "unknown")
            is_tm.append(int(bool(getattr(el, "is_transition_metal", False))))
            is_post.append(int(bool(getattr(el, "is_post_transition_metal", False))))
            is_met.append(int(bool(getattr(el, "is_metalloid", False))))
            is_lanth.append(int(bool(getattr(el, "is_lanthanoid", False))))
            is_act.append(int(bool(getattr(el, "is_actinoid", False))))
        except Exception:
            z_list.append(np.nan); row_list.append(np.nan); group_list.append(np.nan)
            block_list.append("unknown")
            is_tm.append(0); is_post.append(0); is_met.append(0); is_lanth.append(0); is_act.append(0)

    df["dopant_Z"] = z_list
    df["dopant_row"] = row_list
    df["dopant_group_num"] = group_list
    df["dopant_block"] = block_list
    df["dopant_is_transition"] = is_tm
    df["dopant_is_post_transition"] = is_post
    df["dopant_is_metalloid"] = is_met
    df["dopant_is_lanthanoid"] = is_lanth
    df["dopant_is_actinoid"] = is_act

    cols_num += ["dopant_Z", "dopant_row", "dopant_group_num", "dopant_is_transition", "dopant_is_post_transition", "dopant_is_metalloid", "dopant_is_lanthanoid", "dopant_is_actinoid"]
    cols_cat += ["dopant_block"]
    return df, cols_num, cols_cat


def _derive_formula_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = df.copy()
    counts_total = []
    n_species = []
    host_heavy = []

    for formula in df.get("slab_formula", pd.Series([""] * len(df))).fillna("").astype(str):
        cnt = _parse_formula_counts(formula)
        counts_total.append(int(sum(cnt.values())) if cnt else np.nan)
        n_species.append(int(len(cnt)) if cnt else np.nan)
        if cnt:
            host_heavy.append(max(cnt.items(), key=lambda kv: kv[1])[0])
        else:
            host_heavy.append("unknown")

    df["slab_atom_count"] = counts_total
    df["slab_species_count"] = n_species
    df["slab_major_species"] = host_heavy
    return df, ["slab_atom_count", "slab_species_count"], ["slab_major_species"]


def _normalize_standard_schema(df: pd.DataFrame, text_col: str, is_out: bool = False) -> pd.DataFrame:
    """
    Prioritize standard fields:
      target, dopant, family_base_miller
    but gracefully support legacy addH_out_master.csv fields.
    """
    df = df.copy()

    # id
    if "id" not in df.columns:
        raise ValueError("Expected 'id' column")

    # text -> text_raw fallback
    if "text_raw" not in df.columns:
        if text_col in df.columns:
            df["text_raw"] = df[text_col]
        elif "text" in df.columns:
            df["text_raw"] = df["text"]
        else:
            df["text_raw"] = ""

    # prioritize target, then fallback to legacy addH_out fields
    if "target" not in df.columns:
        if "h_ads_excel" in df.columns:
            df["target"] = pd.to_numeric(df["h_ads_excel"], errors="coerce")
        elif "target_computed" in df.columns:
            df["target"] = pd.to_numeric(df["target_computed"], errors="coerce")
        else:
            df["target"] = np.nan
    else:
        df["target"] = pd.to_numeric(df["target"], errors="coerce")

    # prioritize dopant, fallback to element
    if "dopant" not in df.columns:
        if "element" in df.columns:
            df["dopant"] = df["element"]
        else:
            df["dopant"] = "unknown"
    df["dopant"] = df["dopant"].fillna("unknown").astype(str)

    # prioritize family_base, fallback to material
    if "family_base" not in df.columns:
        if "material" in df.columns:
            df["family_base"] = df["material"]
        else:
            df["family_base"] = "unknown"
    df["family_base"] = df["family_base"].fillna("unknown").astype(str)

    # normalize miller string if present
    if "miller" in df.columns:
        df["miller"] = df["miller"].map(_normalize_miller_text)
    else:
        df["miller"] = "(? ? ?)"

    # prioritize family_base_miller, otherwise derive it
    if "family_base_miller" not in df.columns:
        df["family_base_miller"] = df.apply(
            lambda r: f"{r['family_base']}-{_miller_key(r['miller'])}" if _miller_key(r["miller"]) != "unknown" else str(r["family_base"]),
            axis=1,
        )
    else:
        # keep existing non-null values, fill missing from family_base + miller
        miss = df["family_base_miller"].isna() | (df["family_base_miller"].astype(str).str.strip() == "")
        df.loc[miss, "family_base_miller"] = df.loc[miss].apply(
            lambda r: f"{r['family_base']}-{_miller_key(r['miller'])}" if _miller_key(r["miller"]) != "unknown" else str(r["family_base"]),
            axis=1,
        )
    df["family_base_miller"] = df["family_base_miller"].fillna("unknown").astype(str)

    # data source
    if "data_source" not in df.columns:
        df["data_source"] = "addH_out" if is_out else "addH_train"

    # text_structured will use normalized standard fields
    df["text_structured"] = df.apply(lambda r: _text_structured(r, text_col="text_raw"), axis=1)
    df["text_len"] = df["text_raw"].fillna("").astype(str).map(len)
    df["has_known_miller"] = df["miller"].fillna("").astype(str).map(lambda x: 0 if "?" in x or not x.strip() else 1)
    df["converged_int"] = df.get("converged", pd.Series([False] * len(df))).fillna(False).astype(int)

    return df


def _prepare_master(df: pd.DataFrame, emb: Dict[str, object], text_col: str, strict: bool):
    df = _normalize_standard_schema(df, text_col=text_col, is_out=False)
    df = df.copy()

    required = ["id", "target", "text_raw", "dopant", "family_base_miller"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required standard columns in master CSV: {miss}")

    df["eq_emb"] = df["id"].astype(str).map(emb)

    df, dop_num_cols, dop_cat_cols = _derive_dopant_features(df)
    df, form_num_cols, form_cat_cols = _derive_formula_features(df)

    # prioritize standard fields first
    cat_cols = []
    for c in ["family_base_miller", "dopant", "family_base", "site_type", "miller", "data_source", "slab_formula"]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("unknown")
            cat_cols.append(c)
    cat_cols += dop_cat_cols + form_cat_cols
    cat_cols = [c for c in dict.fromkeys(cat_cols) if c in df.columns]

    num_cols = []
    for c in ["anchor_count", "text_len", "has_known_miller", "converged_int"]:
        if c in df.columns:
            num_cols.append(c)
    num_cols += dop_num_cols + form_num_cols
    num_cols = [c for c in dict.fromkeys(num_cols) if c in df.columns]

    # auxiliary labels
    site_map = {}
    if "site_type" in df.columns:
        site_map = {k: i for i, k in enumerate(sorted(df["site_type"].astype(str).unique()))}
        df["site_type_label"] = df["site_type"].map(site_map).astype(int)

    fam_map = {k: i for i, k in enumerate(sorted(df["family_base_miller"].astype(str).unique()))}
    df["family_label"] = df["family_base_miller"].map(fam_map).astype(int)

    usable = df[
        df["target"].notna()
        & df["text_raw"].notna()
        & df["eq_emb"].notna()
        & df.get("parse_ok", True)
    ].copy()

    if strict and usable.empty:
        raise ValueError("No usable rows remained after filtering target/text/eq_emb/parse_ok")

    meta = {
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "aux_label_cols": [c for c in ["site_type_label", "family_label", "target_bin"] if c in usable.columns],
        "site_label_map": site_map,
        "family_label_map": fam_map,
    }
    return usable, meta


def _prepare_addhout(df: pd.DataFrame, emb: Dict[str, object], text_col: str):
    df = _normalize_standard_schema(df, text_col=text_col, is_out=True)
    df = df.copy()

    required = ["id", "text_raw", "dopant", "family_base_miller"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required normalized columns in addH-out CSV: {miss}")

    df["eq_emb"] = df["id"].astype(str).map(emb)

    df, _, _ = _derive_dopant_features(df)
    df, _, _ = _derive_formula_features(df)

    usable = df[df["text_raw"].notna() & df["eq_emb"].notna() & df.get("parse_ok", True)].copy()
    return usable


def _save_fold(
    fold_dir: Path,
    split_df: pd.DataFrame,
    addhout_df: pd.DataFrame | None,
    group_col: str,
    train_groups: set,
    val_groups: set,
    test_groups: set,
    outer_fold_idx: int,
    cv_mode: str,
    meta_cols: Dict[str, list],
):
    fold_dir.mkdir(parents=True, exist_ok=True)

    def _split_name(g):
        if g in train_groups:
            return "train"
        if g in val_groups:
            return "val"
        if g in test_groups:
            return "test"
        return "drop"

    split_df = split_df.copy()
    split_df["split"] = split_df[group_col].map(_split_name)
    split_df["outer_fold"] = int(outer_fold_idx)
    split_df["cv_mode"] = cv_mode

    train_mask = split_df["split"] == "train"
    split_df["target_bin"] = -1
    if train_mask.sum() >= 5:
        split_df["target_bin"] = _target_bin_from_train(split_df.loc[train_mask, "target"], split_df["target"], n_bins=5)

    split_df.to_csv(fold_dir / "split_assignments.csv", index=False)

    base_cols = [
        "id", "target", "text_raw", "text_structured", "eq_emb",
        "site_type_label", "family_label", "target_bin",
    ] + [c for c in meta_cols["cat_cols"] if c in split_df.columns] + [c for c in meta_cols["num_cols"] if c in split_df.columns]

    base_cols = [c for c in dict.fromkeys(base_cols) if c in split_df.columns]

    nn_train = split_df.loc[split_df["split"] == "train", base_cols].copy()
    nn_val = split_df.loc[split_df["split"] == "val", base_cols].copy()
    nn_test = split_df.loc[split_df["split"] == "test", base_cols].copy()

    cat_train = nn_train.copy()
    cat_val = nn_val.copy()
    cat_test = nn_test.copy()

    if nn_train.empty or nn_val.empty or nn_test.empty:
        raise ValueError(f"Fold {outer_fold_idx}: empty split detected train={len(nn_train)} val={len(nn_val)} test={len(nn_test)}")

    nn_train.to_pickle(fold_dir / "nn_train.pkl")
    nn_val.to_pickle(fold_dir / "nn_val.pkl")
    nn_test.to_pickle(fold_dir / "nn_test.pkl")

    cat_train.to_pickle(fold_dir / "cat_train.pkl")
    cat_val.to_pickle(fold_dir / "cat_val.pkl")
    cat_test.to_pickle(fold_dir / "cat_test.pkl")

    if addhout_df is not None and len(addhout_df) > 0:
        out_cols = ["id", "target", "text_raw", "text_structured", "eq_emb"] + [c for c in meta_cols["cat_cols"] if c in addhout_df.columns] + [c for c in meta_cols["num_cols"] if c in addhout_df.columns]
        out_cols = [c for c in dict.fromkeys(out_cols) if c in addhout_df.columns]
        out_nn = addhout_df[out_cols].copy()
        out_cat = out_nn.copy()
        out_nn.to_pickle(fold_dir / "addH_out_nn_pred_input.pkl")
        out_cat.to_pickle(fold_dir / "addH_out_cat_pred_input.pkl")
        addhout_df.to_csv(fold_dir / "addH_out_pred_manifest.csv", index=False)

    meta = {
        "outer_fold": int(outer_fold_idx),
        "cv_mode": cv_mode,
        "group_col": group_col,
        "train_groups": _json_safe_list(sorted(list(train_groups))),
        "val_groups": _json_safe_list(sorted(list(val_groups))),
        "test_groups": _json_safe_list(sorted(list(test_groups))),
        "n_train": int(len(nn_train)),
        "n_val": int(len(nn_val)),
        "n_test": int(len(nn_test)),
        "n_addhout": int(len(addhout_df)) if addhout_df is not None else 0,
        "cat_cols": meta_cols["cat_cols"],
        "num_cols": meta_cols["num_cols"],
    }
    with (fold_dir / "fold_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    addh = pd.read_csv(args.addh_master_csv)
    emb = _load_emb(Path(args.eq_emb_pkl))
    usable_addh, meta = _prepare_master(addh, emb, text_col=args.text_col, strict=args.strict)

    addhout_usable = None
    if args.addhout_master_csv and args.addhout_eq_emb_pkl:
        addhout = pd.read_csv(args.addhout_master_csv)
        addhout_emb = _load_emb(Path(args.addhout_eq_emb_pkl))
        addhout_usable = _prepare_addhout(addhout, addhout_emb, text_col=args.text_col)

    if args.group_col not in usable_addh.columns:
        raise ValueError(f"group_col {args.group_col!r} not found in usable addH dataframe")

    outer_splits = _make_outer_splits(
        usable_addh[args.group_col].to_numpy(),
        cv_mode=args.cv_mode,
        n_splits=args.n_splits,
        seed=args.seed,
    )

    summary_rows = []
    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        outer_train_df = usable_addh.iloc[outer_train_idx].copy()
        outer_test_df = usable_addh.iloc[outer_test_idx].copy()

        test_groups = set(outer_test_df[args.group_col].unique())
        train_groups, val_groups = _choose_val_groups_from_train(
            outer_train_df=outer_train_df,
            group_col=args.group_col,
            val_mode=args.val_mode,
            val_frac=args.val_frac,
            seed=args.seed,
            outer_fold_idx=fold_idx,
        )

        allowed = train_groups | val_groups | test_groups
        fold_df = usable_addh[usable_addh[args.group_col].isin(allowed)].copy()

        fold_dir = out_dir / f"fold_{fold_idx}"
        _save_fold(
            fold_dir=fold_dir,
            split_df=fold_df,
            addhout_df=addhout_usable,
            group_col=args.group_col,
            train_groups=train_groups,
            val_groups=val_groups,
            test_groups=test_groups,
            outer_fold_idx=fold_idx,
            cv_mode=args.cv_mode,
            meta_cols=meta,
        )

        summary_rows.append({
            "fold": int(fold_idx),
            "train_groups": ",".join(map(str, sorted(train_groups))),
            "val_groups": ",".join(map(str, sorted(val_groups))),
            "test_groups": ",".join(map(str, sorted(test_groups))),
            "n_train": int((fold_df[args.group_col].isin(train_groups)).sum()),
            "n_val": int((fold_df[args.group_col].isin(val_groups)).sum()),
            "n_test": int((fold_df[args.group_col].isin(test_groups)).sum()),
        })

    pd.DataFrame(summary_rows).to_csv(out_dir / "cv_summary.csv", index=False)

    eq_dims = sorted({int(np.asarray(x).reshape(-1).shape[0]) for x in usable_addh["eq_emb"]})
    info = {
        "rows": int(len(usable_addh)),
        "group_col": args.group_col,
        "eq_emb_dims": eq_dims,
        "cat_cols": meta["cat_cols"],
        "num_cols": meta["num_cols"],
        "aux_label_cols": meta["aux_label_cols"],
        "text_cols": ["text_raw", "text_structured"],
        "schema_priority": ["target", "dopant", "family_base_miller"],
    }
    with (out_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("[OK] saved singleview CV dirs under", out_dir)
    print("[INFO] usable addH rows =", len(usable_addh))
    print("[INFO] usable addH-out rows =", 0 if addhout_usable is None else len(addhout_usable))
    print("[INFO] eq_emb_dims =", eq_dims)
    print("[INFO] cat_cols =", meta["cat_cols"])
    print("[INFO] num_cols =", meta["num_cols"])
    print("[INFO] summary csv ->", out_dir / "cv_summary.csv")
    print("[INFO] dataset info ->", out_dir / "dataset_info.json")


if __name__ == "__main__":
    main()
