#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Sequence, Tuple, Any

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except Exception:
        StratifiedGroupKFold = None
except Exception as e:
    raise SystemExit(
        "This script requires scikit-learn.\n"
        f"Original import error: {e}"
    )


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build cross-validation multi-view data splits; optionally keep eq_emb in regression datasets for multimodal regression."
    )
    ap.add_argument("--addh-master-csv", required=True)
    ap.add_argument("--eq-emb-pkl", required=True, help="dict[id] -> eq_emb for addH")
    ap.add_argument("--addhout-master-csv", required=True)
    ap.add_argument(
        "--addhout-eq-emb-pkl",
        default=None,
        help="Optional dict[id] -> eq_emb for addH-out. Required if --include-regress-eq-emb is set and you want multimodal inference on addH-out.",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--group-col", default="family_base", help="Leakage-avoid grouping column.")
    ap.add_argument(
        "--cv-mode",
        default="gkfold",
        choices=["single", "gkfold", "stratified_gkfold", "logo"],
        help="single reproduces one split; gkfold creates K outer folds; stratified_gkfold keeps group leakage control while balancing target bins across folds; logo creates leave-one-group-out folds.",
    )
    ap.add_argument("--n-splits", type=int, default=4, help="Number of outer folds for gkfold/stratified_gkfold.")
    ap.add_argument("--stratify-bins", type=int, default=5, help="Number of target bins used by stratified_gkfold and stratified_group_holdout validation splitting.")
    ap.add_argument("--stratify-target-col", default="target", help="Numeric target column used for stratified group splitting.")
    ap.add_argument(
        "--val-mode",
        default="group_holdout",
        choices=["group_holdout", "stratified_group_holdout", "group_kfold_inner"],
        help="How to create validation groups from the outer-train groups.",
    )
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.25,
        help="Fraction of outer-train groups assigned to val when val-mode=group_holdout.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--include-regress-eq-emb",
        action="store_true",
        help="If set, regress_train/val/test.pkl and addH_out_pred_input.pkl will also contain eq_emb.",
    )
    ap.add_argument(
        "--include-addhout-in-clip",
        action="store_true",
        help="Add unlabeled addH-out rows to clip_train.pkl for target-domain graph-text self-supervised alignment. Labels are not used and target is overwritten to 0.0.",
    )
    ap.add_argument(
        "--exclude-target-outliers",
        action="store_true",
        help="Exclude rows with outlier_flag_target=True from regression/CV construction. Default keeps flagged rows so they can be down-weighted instead.",
    )
    ap.add_argument(
        "--target-abs-max",
        type=float,
        default=None,
        help="Optional hard safety filter: exclude addH rows with abs(target) greater than this value before CV splitting. Useful for removing parse/energy-alignment failures such as |E_ads| >> physically plausible range.",
    )
    ap.add_argument(
        "--extra-regress-cols",
        default="text_structured,text_raw,family_base,family_base_miller,dopant,miller,site_type,anchor_count,slab_formula,data_source,w_domain,outlier_flag_target,outlier_reason_target",
        help="Comma-separated optional columns to keep in regress_train/val/test/addH_out_pred_input pkl files.",
    )
    ap.add_argument(
        "--extra-clip-cols",
        default="text_structured,text_raw,family_base,family_base_miller,dopant,miller,site_type,anchor_count,slab_formula,data_source",
        help="Comma-separated optional columns to keep in clip_train/clip_val pkl files.",
    )
    return ap.parse_args()


def _load_eq_emb(eq_emb_pkl: Path) -> Dict[str, object]:
    with open(eq_emb_pkl, "rb") as f:
        emb = pickle.load(f)
    if not isinstance(emb, dict):
        raise TypeError(f"eq_emb pickle must be a dict[id] -> embedding, got {type(emb)}")
    return emb




def _parse_csv_list(raw: str) -> list[str]:
    if raw is None:
        return []
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _unique_preserve_order(xs):
    out = []
    seen = set()
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _existing_cols(df: pd.DataFrame, cols) -> list[str]:
    return [c for c in _unique_preserve_order(cols) if c in df.columns]


def _select_with_optional(df: pd.DataFrame, mandatory, optional) -> pd.DataFrame:
    cols = _existing_cols(df, list(mandatory) + list(optional))
    miss = [c for c in mandatory if c not in df.columns]
    if miss:
        raise ValueError(f"Missing mandatory columns while building split file: {miss}")
    return df[cols].copy()


def _prepare_clip_addhout(addhout_usable: pd.DataFrame, optional_cols) -> pd.DataFrame:
    # Use addH-out only for graph-text SSL/domain adaptation. Any target column from Excel
    # is intentionally overwritten so regression labels are never leaked.
    mandatory = ["id", "text", "eq_emb"]
    clip_out = _select_with_optional(addhout_usable, mandatory=mandatory, optional=optional_cols)
    clip_out["target"] = 0.0
    ordered = _existing_cols(clip_out, ["id", "text", "target", "eq_emb"] + list(optional_cols))
    return clip_out[ordered].copy()


def _json_safe_scalar(x: Any):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _json_safe_list(xs):
    return [_json_safe_scalar(x) for x in xs]


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    """Robust bool parser for CSV-loaded flags.

    pandas normally reads True/False as bool, but mixed values can become strings;
    bool("False") would incorrectly evaluate to True, so parse explicitly.
    """
    if s.dtype == bool:
        return s.fillna(False)
    return (
        s.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes", "y"})
    )


def _filter_addh(
    addh: pd.DataFrame,
    emb: Dict[str, object],
    group_col: str,
    exclude_target_outliers: bool = False,
    target_abs_max: float | None = None,
) -> pd.DataFrame:
    addh = addh.copy()
    addh["eq_emb"] = addh["id"].map(emb)

    required = ["id", "text", "target", group_col]
    miss = [c for c in required if c not in addh.columns]
    if miss:
        raise ValueError(f"Missing required columns in addH master: {miss}")

    mask = (
        addh["parse_ok"].fillna(False)
        & addh["text"].notna()
        & addh["target"].notna()
        & addh["eq_emb"].notna()
        & addh[group_col].notna()
    )
    if exclude_target_outliers and "outlier_flag_target" in addh.columns:
        mask = mask & (~_coerce_bool_series(addh["outlier_flag_target"]))

    if target_abs_max is not None:
        target_abs_max = float(target_abs_max)
        if target_abs_max <= 0:
            raise ValueError("--target-abs-max must be positive when provided")
        mask = mask & (addh["target"].astype(float).abs() <= target_abs_max)

    usable = addh[mask].copy()

    if usable.empty:
        raise ValueError("No usable addH rows remained after filtering parse_ok/text/target/eq_emb/group.")
    return usable


def _filter_addhout(addhout: pd.DataFrame, addhout_emb: Dict[str, object] | None, include_regress_eq_emb: bool, include_addhout_in_clip: bool = False) -> pd.DataFrame:
    miss = [c for c in ["id", "text"] if c not in addhout.columns]
    if miss:
        raise ValueError(f"Missing required columns in addH-out master: {miss}")

    addhout_usable = addhout[addhout["parse_ok"].fillna(False) & addhout["text"].notna()].copy()

    need_addhout_emb = bool(include_regress_eq_emb or include_addhout_in_clip)
    if need_addhout_emb:
        if addhout_emb is None:
            raise ValueError(
                "addH-out eq_emb is required because --include-regress-eq-emb and/or --include-addhout-in-clip was set, "
                "but --addhout-eq-emb-pkl was not provided."
            )
        addhout_usable["eq_emb"] = addhout_usable["id"].map(addhout_emb)
        addhout_usable = addhout_usable[addhout_usable["eq_emb"].notna()].copy()

    if addhout_usable.empty:
        raise ValueError("No usable addH-out rows remained after filtering parse_ok/text/(eq_emb).")
    return addhout_usable




def _safe_qcut_bins(values: pd.Series, n_bins: int) -> pd.Series:
    """Return robust integer bins for a numeric series.

    qcut can fail when many groups share the same median target. This helper
    reduces the number of bins automatically and falls back to rank-based bins.
    """
    vals = pd.to_numeric(values, errors="coerce")
    if vals.notna().sum() == 0:
        return pd.Series(np.zeros(len(vals), dtype=int), index=values.index)
    n_bins = int(max(2, n_bins))
    uniq = int(vals.dropna().nunique())
    bins = int(max(1, min(n_bins, uniq)))
    if bins <= 1:
        return pd.Series(np.zeros(len(vals), dtype=int), index=values.index)
    try:
        out = pd.qcut(vals, q=bins, labels=False, duplicates="drop")
        return out.fillna(0).astype(int)
    except Exception:
        ranks = vals.rank(method="average", na_option="keep")
        out = pd.qcut(ranks, q=bins, labels=False, duplicates="drop")
        return out.fillna(0).astype(int)


def _group_target_bins(df: pd.DataFrame, group_col: str, target_col: str, n_bins: int) -> tuple[pd.Series, dict]:
    if target_col not in df.columns:
        raise ValueError(f"Missing --stratify-target-col in data: {target_col}")
    gmed = df.groupby(group_col)[target_col].median()
    gbins = _safe_qcut_bins(gmed, n_bins=n_bins)
    mapping = {str(k): int(v) for k, v in gbins.items()}
    row_bins = df[group_col].astype(str).map(mapping).fillna(0).astype(int)
    return row_bins, mapping


def _split_target_stats(df: pd.DataFrame, group_col: str, groups: set, target_col: str = "target") -> dict:
    part = df[df[group_col].isin(groups)].copy()
    if part.empty or target_col not in part.columns:
        return {"n": int(len(part)), "target_min": np.nan, "target_max": np.nan, "target_mean": np.nan, "target_median": np.nan, "target_std": np.nan}
    y = pd.to_numeric(part[target_col], errors="coerce")
    return {
        "n": int(len(part)),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "target_mean": float(y.mean()),
        "target_median": float(y.median()),
        "target_std": float(y.std(ddof=1)) if y.notna().sum() > 1 else 0.0,
    }

def _assign_groups_single(groups: Sequence[object], seed: int, val_frac: float) -> Tuple[set, set, set]:
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


def _make_outer_splits(df: pd.DataFrame, group_col: str, cv_mode: str, n_splits: int, seed: int, stratify_bins: int, stratify_target_col: str):
    groups = df[group_col].to_numpy()
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

    if cv_mode == "stratified_gkfold":
        if uniq_n < 2:
            raise ValueError("Need at least 2 unique groups for StratifiedGroupKFold")
        if StratifiedGroupKFold is None:
            raise ImportError("StratifiedGroupKFold is unavailable in this scikit-learn version. Upgrade scikit-learn or use --cv-mode gkfold.")
        n_splits_eff = min(n_splits, uniq_n)
        y_bins, bin_map = _group_target_bins(df, group_col=group_col, target_col=stratify_target_col, n_bins=stratify_bins)
        print("[INFO] stratified_gkfold group target bins =", json.dumps(bin_map, ensure_ascii=False))
        splitter = StratifiedGroupKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)
        return list(splitter.split(dummy_X, y_bins.to_numpy(), groups=groups))

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
    stratify_bins: int = 5,
    stratify_target_col: str = "target",
) -> Tuple[set, set]:
    train_groups = np.array(sorted(outer_train_df[group_col].dropna().unique()))
    n_groups = len(train_groups)
    if n_groups < 2:
        raise ValueError(
            f"Outer fold {outer_fold_idx}: need at least 2 train groups to split train/val, got {n_groups}"
        )

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


    if val_mode == "stratified_group_holdout":
        rng = np.random.default_rng(seed + outer_fold_idx)
        gmed = outer_train_df.groupby(group_col)[stratify_target_col].median()
        gbins = _safe_qcut_bins(gmed, n_bins=stratify_bins)
        val_g = set()
        for b in sorted(gbins.unique()):
            cand = np.array(sorted(gbins[gbins == b].index.astype(str).tolist()))
            rng.shuffle(cand)
            if len(cand) <= 1:
                # Keep singleton bins in training; otherwise the bin disappears from train.
                continue
            n_take = max(1, round(len(cand) * val_frac))
            if n_take >= len(cand):
                n_take = len(cand) - 1
            val_g.update(cand[:n_take])
        if not val_g:
            groups = train_groups.astype(str).copy()
            rng.shuffle(groups)
            n_val = max(1, round(len(groups) * val_frac))
            if n_val >= len(groups):
                n_val = len(groups) - 1
            val_g = set(groups[:n_val])
        train_g = set(map(str, train_groups)) - val_g
        if not train_g:
            raise ValueError(f"Outer fold {outer_fold_idx}: stratified validation consumed all train groups")
        # Preserve original group dtypes where possible.
        return train_g, val_g

    if val_mode == "group_kfold_inner":
        n_inner = min(4, n_groups)
        n_inner = max(2, n_inner)
        inner = GroupKFold(n_splits=n_inner)
        dummy_X = np.zeros((len(outer_train_df), 1), dtype=np.float32)
        dummy_y = np.zeros(len(outer_train_df), dtype=np.float32)
        inner_splits = list(inner.split(dummy_X, dummy_y, groups=outer_train_df[group_col].to_numpy()))
        inner_train_idx, inner_val_idx = inner_splits[0]
        train_g = set(outer_train_df.iloc[inner_train_idx][group_col].unique())
        val_g = set(outer_train_df.iloc[inner_val_idx][group_col].unique())
        return train_g, val_g

    raise ValueError(f"Unsupported val_mode: {val_mode}")


def _save_fold(
    fold_dir: Path,
    usable_addh: pd.DataFrame,
    addhout_usable: pd.DataFrame,
    group_col: str,
    train_groups: set,
    val_groups: set,
    test_groups: set,
    outer_fold_idx: int,
    cv_mode: str,
    include_regress_eq_emb: bool,
    include_addhout_in_clip: bool,
    extra_regress_cols: list[str],
    extra_clip_cols: list[str],
):
    fold_dir.mkdir(parents=True, exist_ok=True)

    split_df = usable_addh.copy()

    def _split_name(g):
        if g in train_groups:
            return "train"
        if g in val_groups:
            return "val"
        if g in test_groups:
            return "test"
        return "drop"

    split_df["split"] = split_df[group_col].map(_split_name)
    split_df["outer_fold"] = outer_fold_idx
    split_df["cv_mode"] = cv_mode
    split_df.to_csv(fold_dir / "addH_split_assignments.csv", index=False)

    clip_mandatory = ["id", "text", "target", "eq_emb"]
    clip_train = _select_with_optional(
        split_df[split_df["split"] == "train"],
        mandatory=clip_mandatory,
        optional=extra_clip_cols,
    )
    clip_val = _select_with_optional(
        split_df[split_df["split"] == "val"],
        mandatory=clip_mandatory,
        optional=extra_clip_cols,
    )

    n_addhout_clip_train = 0
    if include_addhout_in_clip:
        addhout_clip = _prepare_clip_addhout(addhout_usable, optional_cols=extra_clip_cols)
        common_cols = _unique_preserve_order(list(clip_train.columns) + list(addhout_clip.columns))
        for c in common_cols:
            if c not in clip_train.columns:
                clip_train[c] = np.nan
            if c not in addhout_clip.columns:
                addhout_clip[c] = np.nan
        clip_train = pd.concat([clip_train[common_cols], addhout_clip[common_cols]], ignore_index=True)
        n_addhout_clip_train = int(len(addhout_clip))

    regress_mandatory = ["id", "text", "target"] + (["eq_emb"] if include_regress_eq_emb else [])
    regress_train = _select_with_optional(
        split_df[split_df["split"] == "train"],
        mandatory=regress_mandatory,
        optional=extra_regress_cols,
    )
    regress_val = _select_with_optional(
        split_df[split_df["split"] == "val"],
        mandatory=regress_mandatory,
        optional=extra_regress_cols,
    )
    regress_test = _select_with_optional(
        split_df[split_df["split"] == "test"],
        mandatory=regress_mandatory,
        optional=extra_regress_cols,
    )

    if clip_train.empty or clip_val.empty or regress_test.empty:
        raise ValueError(
            f"Fold {outer_fold_idx}: empty split detected "
            f"(clip_train={len(clip_train)}, clip_val={len(clip_val)}, regress_test={len(regress_test)})"
        )

    clip_train.to_pickle(fold_dir / "clip_train.pkl")
    clip_val.to_pickle(fold_dir / "clip_val.pkl")
    regress_train.to_pickle(fold_dir / "regress_train.pkl")
    regress_val.to_pickle(fold_dir / "regress_val.pkl")
    regress_test.to_pickle(fold_dir / "regress_test.pkl")

    addhout_pred_mandatory = ["id", "text"] + (["eq_emb"] if include_regress_eq_emb else [])
    addhout_pred = _select_with_optional(addhout_usable, mandatory=addhout_pred_mandatory, optional=extra_regress_cols)
    # Prediction script accepts target if present; overwrite to avoid accidental label leakage.
    addhout_pred["target"] = 0.0
    ordered_cols = _existing_cols(addhout_pred, ["id", "text"] + (["eq_emb"] if include_regress_eq_emb else []) + ["target"] + extra_regress_cols)
    addhout_pred = addhout_pred[ordered_cols]
    addhout_pred.to_pickle(fold_dir / "addH_out_pred_input.pkl")
    addhout_usable.to_csv(fold_dir / "addH_out_pred_manifest.csv", index=False)

    meta = {
        "outer_fold": int(outer_fold_idx),
        "cv_mode": cv_mode,
        "group_col": group_col,
        "include_regress_eq_emb": bool(include_regress_eq_emb),
        "include_addhout_in_clip": bool(include_addhout_in_clip),
        "n_addhout_clip_train": int(n_addhout_clip_train),
        "extra_regress_cols_kept": _existing_cols(regress_train, extra_regress_cols),
        "extra_clip_cols_kept": _existing_cols(clip_train, extra_clip_cols),
        "train_groups": _json_safe_list(sorted(list(train_groups))),
        "val_groups": _json_safe_list(sorted(list(val_groups))),
        "test_groups": _json_safe_list(sorted(list(test_groups))),
        "n_clip_train": int(len(clip_train)),
        "n_clip_val": int(len(clip_val)),
        "n_regress_train": int(len(regress_train)),
        "n_regress_val": int(len(regress_val)),
        "n_regress_test": int(len(regress_test)),
        "n_addhout_pred": int(len(addhout_pred)),
    }
    with open(fold_dir / "fold_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    addh = pd.read_csv(args.addh_master_csv)
    emb = _load_eq_emb(Path(args.eq_emb_pkl))
    usable_addh = _filter_addh(
        addh,
        emb,
        args.group_col,
        exclude_target_outliers=args.exclude_target_outliers,
        target_abs_max=args.target_abs_max,
    )

    # Keep grouping stable across CSV type inference, especially when groups are numeric-like.
    usable_addh[args.group_col] = usable_addh[args.group_col].astype(str)

    addhout = pd.read_csv(args.addhout_master_csv)
    addhout_emb = _load_eq_emb(Path(args.addhout_eq_emb_pkl)) if args.addhout_eq_emb_pkl else None
    addhout_usable = _filter_addhout(addhout, addhout_emb, args.include_regress_eq_emb, include_addhout_in_clip=args.include_addhout_in_clip)

    extra_regress_cols = _parse_csv_list(args.extra_regress_cols)
    extra_clip_cols = _parse_csv_list(args.extra_clip_cols)

    outer_splits = _make_outer_splits(
        usable_addh,
        group_col=args.group_col,
        cv_mode=args.cv_mode,
        n_splits=args.n_splits,
        seed=args.seed,
        stratify_bins=args.stratify_bins,
        stratify_target_col=args.stratify_target_col,
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
            stratify_bins=args.stratify_bins,
            stratify_target_col=args.stratify_target_col,
        )

        allowed = train_groups | val_groups | test_groups
        fold_df = usable_addh[usable_addh[args.group_col].isin(allowed)].copy()

        fold_dir = out_dir / f"fold_{fold_idx}"
        _save_fold(
            fold_dir=fold_dir,
            usable_addh=fold_df,
            addhout_usable=addhout_usable,
            group_col=args.group_col,
            train_groups=train_groups,
            val_groups=val_groups,
            test_groups=test_groups,
            outer_fold_idx=fold_idx,
            cv_mode=args.cv_mode,
            include_regress_eq_emb=args.include_regress_eq_emb,
            include_addhout_in_clip=args.include_addhout_in_clip,
            extra_regress_cols=extra_regress_cols,
            extra_clip_cols=extra_clip_cols,
        )

        train_stats = _split_target_stats(fold_df, args.group_col, train_groups, target_col=args.stratify_target_col)
        val_stats = _split_target_stats(fold_df, args.group_col, val_groups, target_col=args.stratify_target_col)
        test_stats = _split_target_stats(fold_df, args.group_col, test_groups, target_col=args.stratify_target_col)
        summary_rows.append({
            "fold": int(fold_idx),
            "train_groups": ",".join(map(str, sorted(train_groups))),
            "val_groups": ",".join(map(str, sorted(val_groups))),
            "test_groups": ",".join(map(str, sorted(test_groups))),
            "n_train": int((fold_df[args.group_col].isin(train_groups)).sum()),
            "n_val": int((fold_df[args.group_col].isin(val_groups)).sum()),
            "n_test": int((fold_df[args.group_col].isin(test_groups)).sum()),
            "train_target_min": train_stats["target_min"],
            "train_target_max": train_stats["target_max"],
            "train_target_median": train_stats["target_median"],
            "val_target_min": val_stats["target_min"],
            "val_target_max": val_stats["target_max"],
            "val_target_median": val_stats["target_median"],
            "test_target_min": test_stats["target_min"],
            "test_target_max": test_stats["target_max"],
            "test_target_median": test_stats["target_median"],
            "include_regress_eq_emb": bool(args.include_regress_eq_emb),
            "include_addhout_in_clip": bool(args.include_addhout_in_clip),
            "exclude_target_outliers": bool(args.exclude_target_outliers),
            "target_abs_max": None if args.target_abs_max is None else float(args.target_abs_max),
            "cv_mode": args.cv_mode,
            "val_mode": args.val_mode,
            "stratify_bins": int(args.stratify_bins),
        })

    pd.DataFrame(summary_rows).to_csv(out_dir / "cv_summary.csv", index=False)

    print("[OK] saved CV fold dirs under", out_dir)
    print("[INFO] usable addH rows =", len(usable_addh))
    print("[INFO] usable addH-out rows =", len(addhout_usable))
    print("[INFO] num folds =", len(summary_rows))
    print("[INFO] include_regress_eq_emb =", bool(args.include_regress_eq_emb))
    print("[INFO] include_addhout_in_clip =", bool(args.include_addhout_in_clip))
    print("[INFO] exclude_target_outliers =", bool(args.exclude_target_outliers))
    print("[INFO] target_abs_max =", args.target_abs_max)
    print("[INFO] summary csv ->", out_dir / "cv_summary.csv")


if __name__ == "__main__":
    main()
