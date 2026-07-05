#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
except Exception as e:
    raise SystemExit(
        "This script requires scikit-learn.\n"
        f"Original import error: {e}"
    )


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build cross-validation multi-view data splits compatible with multi-view."
    )
    ap.add_argument("--addh-master-csv", required=True)
    ap.add_argument("--eq-emb-pkl", required=True)
    ap.add_argument("--addhout-master-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--group-col", default="family_base", help="Leakage-avoid grouping column.")
    ap.add_argument(
        "--cv-mode",
        default="gkfold",
        choices=["single", "gkfold", "logo"],
        help="single reproduces one split; gkfold creates K outer folds; logo creates leave-one-group-out folds.",
    )
    ap.add_argument("--n-splits", type=int, default=4, help="Number of outer folds for gkfold.")
    ap.add_argument(
        "--val-mode",
        default="group_holdout",
        choices=["group_holdout", "group_kfold_inner"],
        help="How to create validation groups from the outer-train groups.",
    )
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.25,
        help="Fraction of outer-train groups assigned to val when val-mode=group_holdout.",
    )
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def _json_safe_scalar(x):
    """Convert numpy/pandas scalar types to plain Python JSON-serializable scalars."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _json_safe_list(xs):
    return [_json_safe_scalar(x) for x in xs]


def _load_eq_emb(eq_emb_pkl: Path) -> Dict[str, object]:
    with open(eq_emb_pkl, "rb") as f:
        emb = pickle.load(f)
    if not isinstance(emb, dict):
        raise TypeError(f"eq_emb pickle must be a dict[id] -> embedding, got {type(emb)}")
    return emb


def _filter_addh(addh: pd.DataFrame, emb: Dict[str, object], group_col: str) -> pd.DataFrame:
    addh = addh.copy()
    addh["eq_emb"] = addh["id"].map(emb)

    required = ["id", "text", "target", group_col]
    miss = [c for c in required if c not in addh.columns]
    if miss:
        raise ValueError(f"Missing required columns in addH master: {miss}")

    usable = addh[
        addh["parse_ok"].fillna(False)
        & addh["text"].notna()
        & addh["target"].notna()
        & addh["eq_emb"].notna()
        & addh[group_col].notna()
    ].copy()

    if usable.empty:
        raise ValueError("No usable addH rows remained after filtering parse_ok/text/target/eq_emb/group.")
    return usable


def _filter_addhout(addhout: pd.DataFrame) -> pd.DataFrame:
    miss = [c for c in ["id", "text"] if c not in addhout.columns]
    if miss:
        raise ValueError(f"Missing required columns in addH-out master: {miss}")
    addhout_usable = addhout[addhout["parse_ok"].fillna(False) & addhout["text"].notna()].copy()
    if addhout_usable.empty:
        raise ValueError("No usable addH-out rows remained after filtering parse_ok/text.")
    return addhout_usable


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
    split_df["outer_fold"] = int(outer_fold_idx)
    split_df["cv_mode"] = cv_mode
    split_df.to_csv(fold_dir / "addH_split_assignments.csv", index=False)

    clip_train = split_df[split_df["split"] == "train"][["id", "text", "target", "eq_emb"]].copy()
    clip_val = split_df[split_df["split"] == "val"][["id", "text", "target", "eq_emb"]].copy()
    regress_train = split_df[split_df["split"] == "train"][["id", "text", "target"]].copy()
    regress_val = split_df[split_df["split"] == "val"][["id", "text", "target"]].copy()
    regress_test = split_df[split_df["split"] == "test"][["id", "text", "target"]].copy()

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

    addhout_pred = addhout_usable[["id", "text"]].copy()
    addhout_pred["target"] = 0.0
    addhout_pred.to_pickle(fold_dir / "addH_out_pred_input.pkl")
    addhout_usable.to_csv(fold_dir / "addH_out_pred_manifest.csv", index=False)

    meta = {
        "outer_fold": int(outer_fold_idx),
        "cv_mode": str(cv_mode),
        "group_col": str(group_col),
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
    usable_addh = _filter_addh(addh, emb, args.group_col)

    addhout = pd.read_csv(args.addhout_master_csv)
    addhout_usable = _filter_addhout(addhout)

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
            usable_addh=fold_df,
            addhout_usable=addhout_usable,
            group_col=args.group_col,
            train_groups=train_groups,
            val_groups=val_groups,
            test_groups=test_groups,
            outer_fold_idx=fold_idx,
            cv_mode=args.cv_mode,
        )

        summary_rows.append({
            "fold": int(fold_idx),
            "train_groups": ",".join(map(str, _json_safe_list(sorted(train_groups)))),
            "val_groups": ",".join(map(str, _json_safe_list(sorted(val_groups)))),
            "test_groups": ",".join(map(str, _json_safe_list(sorted(test_groups)))),
            "n_train": int((fold_df[args.group_col].isin(train_groups)).sum()),
            "n_val": int((fold_df[args.group_col].isin(val_groups)).sum()),
            "n_test": int((fold_df[args.group_col].isin(test_groups)).sum()),
        })

    pd.DataFrame(summary_rows).to_csv(out_dir / "cv_summary.csv", index=False)

    print("[OK] saved CV fold dirs under", out_dir)
    print("[INFO] usable addH rows =", len(usable_addh))
    print("[INFO] usable addH-out rows =", len(addhout_usable))
    print("[INFO] num folds =", len(summary_rows))
    print("[INFO] summary csv ->", out_dir / "cv_summary.csv")


if __name__ == "__main__":
    main()
