#!/usr/bin/env python3
from __future__ import annotations
import argparse, pickle
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addh-master-csv", required=True)
    ap.add_argument("--eq-emb-pkl", required=True)
    ap.add_argument("--addhout-master-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--group-col", default="family_base", help="Leakage-avoid split group column; default family_base")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def assign_group_split(groups, seed=42):
    rng = np.random.default_rng(seed)
    groups = np.array(sorted(list(groups)))
    rng.shuffle(groups)
    n = len(groups)
    if n < 3:
        raise ValueError("Need at least 3 unique groups for train/val/test split")
    n_test = 1 if n <= 4 else max(1, round(n * 0.25))
    n_val = 1 if n <= 4 else max(1, round(n * 0.25))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("Not enough groups for train split")
    train_g = set(groups[:n_train])
    val_g = set(groups[n_train:n_train+n_val])
    test_g = set(groups[n_train+n_val:])
    return train_g, val_g, test_g


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    addh = pd.read_csv(args.addh_master_csv)
    with open(args.eq_emb_pkl, "rb") as f:
        emb = pickle.load(f)
    addh["eq_emb"] = addh["id"].map(emb)

    required = ["id", "text", "target", args.group_col]
    miss = [c for c in required if c not in addh.columns]
    if miss:
        raise ValueError(f"Missing required columns in addH master: {miss}")

    usable = addh[
        addh["parse_ok"].fillna(False)
        & addh["text"].notna()
        & addh["target"].notna()
        & addh["eq_emb"].notna()
        & addh[args.group_col].notna()
    ].copy()

    train_g, val_g, test_g = assign_group_split(set(usable[args.group_col].unique()), seed=args.seed)

    def _split_name(g):
        if g in train_g:
            return "train"
        if g in val_g:
            return "val"
        return "test"

    usable["split"] = usable[args.group_col].map(_split_name)
    usable.to_csv(out_dir / "addH_split_assignments.csv", index=False)

    clip_train = usable[usable["split"] == "train"][["id", "text", "target", "eq_emb"]].copy()
    clip_val = usable[usable["split"] == "val"][["id", "text", "target", "eq_emb"]].copy()
    regress_train = usable[usable["split"] == "train"][["id", "text", "target"]].copy()
    regress_val = usable[usable["split"] == "val"][["id", "text", "target"]].copy()
    regress_test = usable[usable["split"] == "test"][["id", "text", "target"]].copy()

    clip_train.to_pickle(out_dir / "clip_train.pkl")
    clip_val.to_pickle(out_dir / "clip_val.pkl")
    regress_train.to_pickle(out_dir / "regress_train.pkl")
    regress_val.to_pickle(out_dir / "regress_val.pkl")
    regress_test.to_pickle(out_dir / "regress_test.pkl")

    addhout = pd.read_csv(args.addhout_master_csv)
    addhout_usable = addhout[addhout["parse_ok"].fillna(False) & addhout["text"].notna()].copy()
    addhout_pred = addhout_usable[["id", "text"]].copy()
    addhout_pred["target"] = 0.0
    addhout_pred.to_pickle(out_dir / "addH_out_pred_input.pkl")
    addhout_usable.to_csv(out_dir / "addH_out_pred_manifest.csv", index=False)

    print("[OK] saved pkl files under", out_dir)
    print("[INFO] clip_train", clip_train.shape, "clip_val", clip_val.shape)
    print("[INFO] regress_train", regress_train.shape, "regress_val", regress_val.shape, "regress_test", regress_test.shape)
    print("[INFO] addH-out pred", addhout_pred.shape)
    print("[INFO] group split train/val/test =", sorted(train_g), sorted(val_g), sorted(test_g))

if __name__ == "__main__":
    main()
