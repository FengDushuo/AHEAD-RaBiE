#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a reusable feature bundle for pretrained dual-embedding delta-head models.

Inputs are the stage-2/3 outputs that already exist in the server run:
  - addH/addH-2 knowledge feature table;
  - addH-out knowledge feature table;
  - dual FAIR-Chem embeddings:
      concat(addH_embedding, bare_embedding, addH_embedding - bare_embedding)

The script does not train a model and never uses addH-out labels except copying
an optional audit-label path into the manifest. The produced NPZ/CSV files are
consumed by 25_train_pretrained_delta_head_addhout.py.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


TARGET_OR_LEAKY_COLS = {
    "target",
    "energy_bare",
    "energy_addH",
    "energy_total_excel",
    "energy_slab_excel",
    "h_ads_excel",
    "target_computed",
}

ID_TEXT_COLS = {
    "split_role",
    "data_source",
    "id",
    "family_base",
    "material",
    "miller_text",
    "dopant",
    "status_bare",
    "status_addH",
    "contcar_path",
    "bare_contcar_path",
    "poscar_formula",
    "non_h_elements",
    "llm_rationale_short",
    "llm_sources",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build pretrained delta-head feature bundle.")
    ap.add_argument("--train-features", required=True, help="knowledge_features_train.csv")
    ap.add_argument("--addhout-features", required=True, help="knowledge_features_addhout.csv")
    ap.add_argument("--train-dual-emb-pkl", required=True, help="addH_dual_eq_emb.pkl")
    ap.add_argument("--addhout-dual-emb-pkl", required=True, help="addH_out_dual_eq_emb.pkl")
    ap.add_argument("--out-dir", default="outputs_addh_pretrained_delta_features")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--target-col", default="target")
    ap.add_argument("--target-abs-max", type=float, default=10.0)
    ap.add_argument(
        "--include-source-stat-features",
        action="store_true",
        help="Include source_* statistics as tabular features. Default excludes them from X and uses priors separately.",
    )
    ap.add_argument(
        "--min-train-coverage",
        type=float,
        default=0.75,
        help="Fail if fewer than this fraction of usable train rows have embeddings.",
    )
    ap.add_argument(
        "--min-addhout-coverage",
        type=float,
        default=0.75,
        help="Fail if fewer than this fraction of addH-out rows have embeddings.",
    )
    ap.add_argument("--audit-labels-csv", default="", help="Optional audit label CSV path for manifest only.")
    return ap.parse_args()


def load_embedding_dict(path: Path) -> Dict[str, np.ndarray]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    out: Dict[str, np.ndarray] = {}
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, pd.DataFrame):
        id_col = "id" if "id" in obj.columns else obj.columns[0]
        emb_col = "eq_emb" if "eq_emb" in obj.columns else None
        if emb_col is None:
            for c in obj.columns:
                if "emb" in str(c).lower():
                    emb_col = c
                    break
        if emb_col is None:
            raise ValueError(f"Could not find embedding column in {path}")
        items = ((r[id_col], r[emb_col]) for _, r in obj.iterrows())
    else:
        raise TypeError(f"Unsupported embedding pickle type {type(obj)} in {path}")
    for k, v in items:
        try:
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
            if arr.size > 0 and np.isfinite(arr).any():
                out[str(k)] = arr
        except Exception:
            continue
    return out


def common_dim(emb: Dict[str, np.ndarray], ids: Iterable[str]) -> int:
    dims: List[int] = []
    for sid in ids:
        v = emb.get(str(sid))
        if v is not None:
            dims.append(int(v.shape[0]))
    if not dims:
        raise ValueError("No embedding dimensions found for requested IDs.")
    return max(set(dims), key=dims.count)


def make_embedding_matrix(df: pd.DataFrame, emb: Dict[str, np.ndarray], id_col: str, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.full((len(df), dim), np.nan, dtype=np.float32)
    ok = np.zeros(len(df), dtype=bool)
    for i, sid in enumerate(df[id_col].astype(str)):
        v = emb.get(sid)
        if v is None:
            continue
        if len(v) >= dim:
            X[i] = v[:dim]
        else:
            X[i, : len(v)] = v
        ok[i] = True
    return X, ok


def tabular_feature_cols(train: pd.DataFrame, out: pd.DataFrame, include_source: bool) -> List[str]:
    cols: List[str] = []
    for c in train.columns:
        if c in TARGET_OR_LEAKY_COLS or c in ID_TEXT_COLS:
            continue
        if c not in out.columns:
            continue
        if str(c).startswith("pred_"):
            continue
        if str(c).startswith("source_") and not include_source:
            continue
        s1 = pd.to_numeric(train[c], errors="coerce")
        s2 = pd.to_numeric(out[c], errors="coerce")
        if s1.notna().any() and s2.notna().any():
            cols.append(c)
    return sorted(cols)


def to_numeric_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)


def safe_series(df: pd.DataFrame, col: str, default: str = "") -> np.ndarray:
    if col in df.columns:
        return df[col].fillna(default).astype(str).to_numpy()
    return np.array([default] * len(df), dtype=object)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_all = pd.read_csv(args.train_features)
    addhout_all = pd.read_csv(args.addhout_features)
    if args.id_col not in train_all.columns or args.id_col not in addhout_all.columns:
        raise SystemExit(f"[ERROR] id column {args.id_col!r} missing from input feature tables.")
    if args.target_col not in train_all.columns:
        raise SystemExit(f"[ERROR] target column {args.target_col!r} missing from train features.")

    train_all["target"] = pd.to_numeric(train_all[args.target_col], errors="coerce")
    usable = train_all["target"].notna() & (train_all["target"].abs() <= args.target_abs_max)
    train = train_all.loc[usable].reset_index(drop=True).copy()
    addhout = addhout_all.reset_index(drop=True).copy()

    train_emb = load_embedding_dict(Path(args.train_dual_emb_pkl))
    out_emb = load_embedding_dict(Path(args.addhout_dual_emb_pkl))
    dim_train = common_dim(train_emb, train[args.id_col].astype(str))
    dim_out = common_dim(out_emb, addhout[args.id_col].astype(str))
    dim = min(dim_train, dim_out)
    if dim % 3 != 0:
        print(f"[WARN] dual embedding dimension {dim} is not divisible by 3; feature modes will fall back to full.")

    Xg_train, ok_train = make_embedding_matrix(train, train_emb, args.id_col, dim)
    Xg_out, ok_out = make_embedding_matrix(addhout, out_emb, args.id_col, dim)
    train_cov = float(ok_train.mean()) if len(ok_train) else 0.0
    out_cov = float(ok_out.mean()) if len(ok_out) else 0.0
    if train_cov < args.min_train_coverage:
        raise SystemExit(f"[ERROR] train embedding coverage too low: {train_cov:.3f}")
    if out_cov < args.min_addhout_coverage:
        raise SystemExit(f"[ERROR] addH-out embedding coverage too low: {out_cov:.3f}")

    tab_cols = tabular_feature_cols(train, addhout, args.include_source_stat_features)
    Xt_train = to_numeric_matrix(train, tab_cols)
    Xt_out = to_numeric_matrix(addhout, tab_cols)

    npz_path = out_dir / "pretrained_delta_feature_bundle.npz"
    np.savez_compressed(
        npz_path,
        X_graph_train=Xg_train,
        X_graph_addhout=Xg_out,
        X_tab_train=Xt_train,
        X_tab_addhout=Xt_out,
        train_has_embedding=ok_train,
        addhout_has_embedding=ok_out,
        y_train=train["target"].to_numpy(np.float32),
        train_ids=train[args.id_col].astype(str).to_numpy(),
        addhout_ids=addhout[args.id_col].astype(str).to_numpy(),
        train_groups=safe_series(train, "family_base"),
        train_material=safe_series(train, "material"),
        addhout_material=safe_series(addhout, "material"),
        train_dopant=safe_series(train, "dopant"),
        addhout_dopant=safe_series(addhout, "dopant"),
        tabular_feature_cols=np.asarray(tab_cols, dtype=object),
        graph_dim=np.asarray([dim], dtype=np.int64),
    )

    train_meta = train[
        [c for c in [args.id_col, "family_base", "material", "dopant", "miller", "target"] if c in train.columns]
    ].copy()
    train_meta["has_dual_embedding"] = ok_train
    train_meta.to_csv(out_dir / "pretrained_delta_train_meta.csv", index=False)

    addhout_meta = addhout[
        [c for c in [args.id_col, "family_base", "material", "dopant", "miller"] if c in addhout.columns]
    ].copy()
    addhout_meta["has_dual_embedding"] = ok_out
    addhout_meta.to_csv(out_dir / "pretrained_delta_addhout_meta.csv", index=False)

    manifest = {
        "train_features": str(args.train_features),
        "addhout_features": str(args.addhout_features),
        "train_dual_emb_pkl": str(args.train_dual_emb_pkl),
        "addhout_dual_emb_pkl": str(args.addhout_dual_emb_pkl),
        "audit_labels_csv": str(args.audit_labels_csv or ""),
        "target_col": args.target_col,
        "target_abs_max": args.target_abs_max,
        "n_train_raw": int(len(train_all)),
        "n_train_usable": int(len(train)),
        "n_addhout": int(len(addhout)),
        "graph_dim": int(dim),
        "train_embedding_coverage": train_cov,
        "addhout_embedding_coverage": out_cov,
        "n_tabular_features": int(len(tab_cols)),
        "tabular_feature_cols": tab_cols,
        "labels_used_for_training_or_selection": False,
        "outputs": {
            "npz": str(npz_path),
            "train_meta": str(out_dir / "pretrained_delta_train_meta.csv"),
            "addhout_meta": str(out_dir / "pretrained_delta_addhout_meta.csv"),
        },
    }
    with (out_dir / "pretrained_delta_feature_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] wrote {npz_path}")
    print(f"[INFO] train usable rows={len(train)} coverage={train_cov:.3f}")
    print(f"[INFO] addH-out rows={len(addhout)} coverage={out_cov:.3f}")
    print(f"[INFO] graph_dim={dim} tabular_features={len(tab_cols)}")


if __name__ == "__main__":
    main()
