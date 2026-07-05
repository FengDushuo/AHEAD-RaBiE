#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build dual graph embeddings for adsorption-energy prediction:

    eq_emb_dual = concat(addH_emb, bare_emb, addH_emb - bare_emb)

Why this matters
----------------
The target is usually E(addH) - E(slab) - E(H). A single addH graph embedding
only describes the final adsorbed structure, whereas the dual embedding makes
the input consistent with the energy-difference definition.

Typical use
-----------
# 1) extract addH embeddings from contcar_path
python 03_extract_eq_emb_fairchem.py --master-csv addH_master_merged.csv \
  --structure-col contcar_path --save-pkl addH_eq_emb.pkl ...

# 2) extract bare embeddings from bare_contcar_path
python 03_extract_eq_emb_fairchem.py --master-csv addH_master_merged.csv \
  --structure-col bare_contcar_path --save-pkl addH_bare_eq_emb.pkl ...

# 3) build dual embeddings
python 06_build_dual_graph_embeddings.py \
  --master-csv addH_master_merged.csv \
  --addh-emb-pkl addH_eq_emb.pkl \
  --bare-emb-pkl addH_bare_eq_emb.pkl \
  --save-pkl addH_dual_eq_emb.pkl \
  --meta-csv addH_dual_eq_emb.meta.csv
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Concatenate addH, bare slab, and delta graph embeddings.")
    ap.add_argument("--master-csv", required=True, help="CSV used as the ID source/order.")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--addh-emb-pkl", required=True, help="dict[id] -> embedding for addH structures")
    ap.add_argument("--bare-emb-pkl", required=True, help="dict[id] -> embedding for bare slab structures")
    ap.add_argument("--save-pkl", required=True, help="Output dict[id] -> concat(addH, bare, addH-bare)")
    ap.add_argument("--meta-csv", default=None, help="Optional metadata CSV; default <save-pkl>.meta.csv")
    ap.add_argument(
        "--missing-bare",
        default="skip",
        choices=["skip", "zeros"],
        help="How to handle IDs missing bare embeddings. 'skip' is safer; 'zeros' preserves rows but is less physical.",
    )
    ap.add_argument(
        "--missing-addh",
        default="skip",
        choices=["skip"],
        help="Currently addH embeddings are required; rows without addH embeddings are skipped.",
    )
    ap.add_argument("--strict-dim", action="store_true", help="Raise if addH/bare embedding dimensions differ.")
    ap.add_argument(
        "--require-success-min-frac",
        type=float,
        default=0.0,
        help="Fail with non-zero exit if output embedding fraction is below this threshold. Use 0 to disable.",
    )
    return ap.parse_args()


def _load_dict(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected pickle dict[id] -> embedding, got {type(obj)} from {path}")
    return {str(k): v for k, v in obj.items()}


def _as_vec(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)


def main() -> None:
    args = parse_args()
    master = pd.read_csv(args.master_csv)
    if args.id_col not in master.columns:
        raise ValueError(f"Missing id column {args.id_col!r} in {args.master_csv}")

    addh = _load_dict(Path(args.addh_emb_pkl))
    bare = _load_dict(Path(args.bare_emb_pkl))

    out: Dict[str, np.ndarray] = {}
    rows = []
    addh_dims = []
    bare_dims = []

    for sid in master[args.id_col].astype(str).tolist():
        status = "ok"
        note = ""
        if sid not in addh:
            status = "missing_addh"
            rows.append({"id": sid, "status": status, "note": note, "out_dim": None})
            continue
        a = _as_vec(addh[sid])
        addh_dims.append(int(a.shape[0]))

        if sid in bare:
            b = _as_vec(bare[sid])
            bare_dims.append(int(b.shape[0]))
        elif args.missing_bare == "zeros":
            b = np.zeros_like(a)
            status = "ok_missing_bare_zero_filled"
            note = "bare embedding missing; zero-filled"
        else:
            status = "missing_bare"
            rows.append({"id": sid, "status": status, "note": note, "addh_dim": int(a.shape[0]), "bare_dim": None, "out_dim": None})
            continue

        if a.shape[0] != b.shape[0]:
            msg = f"dimension mismatch: addH={a.shape[0]} bare={b.shape[0]}"
            if args.strict_dim:
                raise ValueError(f"{sid}: {msg}")
            # Align conservatively by truncating to the shared minimum dimension.
            d = min(a.shape[0], b.shape[0])
            a = a[:d]
            b = b[:d]
            status = "ok_dim_truncated"
            note = msg

        delta = a - b
        y = np.concatenate([a, b, delta], axis=0).astype(np.float32)
        out[sid] = y
        rows.append({
            "id": sid,
            "status": status,
            "note": note,
            "addh_dim": int(a.shape[0]),
            "bare_dim": int(b.shape[0]),
            "out_dim": int(y.shape[0]),
        })

    save_pkl = Path(args.save_pkl)
    save_pkl.parent.mkdir(parents=True, exist_ok=True)
    with save_pkl.open("wb") as f:
        pickle.dump(out, f)

    meta_csv = Path(args.meta_csv) if args.meta_csv else Path(str(save_pkl) + ".meta.csv")
    meta = pd.DataFrame(rows)
    meta.to_csv(meta_csv, index=False)

    print(f"[OK] saved dual embeddings -> {save_pkl}")
    print(f"[OK] saved metadata        -> {meta_csv}")
    print(f"[INFO] output embeddings   = {len(out)} / {len(master)}")
    if out:
        dims = sorted({int(v.shape[0]) for v in out.values()})
        print(f"[INFO] output dims         = {dims}")
    if not meta.empty:
        print("[INFO] status counts:")
        print(meta["status"].value_counts(dropna=False).to_string())

    ok_frac = (len(out) / len(master)) if len(master) else 0.0
    if float(args.require_success_min_frac) > 0 and ok_frac < float(args.require_success_min_frac):
        print(
            f"[ERROR] dual embedding output fraction too low: "
            f"{len(out)} / {len(master)} = {ok_frac:.3f} "
            f"< required {float(args.require_success_min_frac):.3f}"
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
