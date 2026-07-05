#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build local geometry features for AddH/AddH-2 training and AddH-out prediction.

The extractor intentionally has no pymatgen/ase dependency. It parses VASP
POSCAR/CONTCAR files directly, then writes:

  - local_geometry_features_train.csv
  - local_geometry_features_addhout.csv
  - knowledge_features_train_geom.csv
  - knowledge_features_addhout_geom.csv

Feature columns are identical for train and AddH-out. Labels are not used.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


Z_TABLE = {
    "H": 1, "O": 8, "Mg": 12, "Ca": 20, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Mo": 42, "Ru": 44,
    "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "Ce": 58, "Pt": 78,
    "Au": 79, "Hg": 80, "Zr": 40,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract local geometry features from CONTCAR files.")
    ap.add_argument("--train-features", default="outputs_addh_llm_element_priors/knowledge_features_train.csv")
    ap.add_argument("--addhout-features", default="outputs_addh_llm_element_priors/knowledge_features_addhout.csv")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out-dir", default="outputs_addh_local_geometry_features")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--contcar-col", default="contcar_path")
    ap.add_argument("--bare-contcar-col", default="bare_contcar_path")
    ap.add_argument("--pbc-axes", default="xy", choices=["xy", "xyz", "none"])
    return ap.parse_args()


def is_int_tokens(tokens: Sequence[str]) -> bool:
    if not tokens:
        return False
    try:
        [int(float(x)) for x in tokens]
        return True
    except Exception:
        return False


def resolve_path(root: Path, raw: object) -> Path:
    s = "" if pd.isna(raw) else str(raw).strip()
    if not s:
        return Path("__missing__")
    s = s.replace("\\", "/")
    p = Path(s)
    if p.is_absolute():
        return p
    return root / p


def parse_poscar(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None
    lines = [ln.strip() for ln in lines if ln.strip()]
    if len(lines) < 8:
        return None
    try:
        scale_tokens = lines[1].split()
        if len(scale_tokens) == 1:
            scale = float(scale_tokens[0])
            if scale == 0:
                scale = 1.0
            lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)], dtype=float)
            lattice = lattice * abs(scale)
        else:
            scale_vec = np.array([float(x) for x in scale_tokens[:3]], dtype=float)
            lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)], dtype=float)
            lattice = lattice * scale_vec.reshape(3, 1)
        line5 = lines[5].split()
        line6 = lines[6].split()
        if is_int_tokens(line5):
            elems = [f"X{i+1}" for i in range(len(line5))]
            counts = [int(float(x)) for x in line5]
            coord_idx = 6
        else:
            elems = line5
            counts = [int(float(x)) for x in line6]
            coord_idx = 7
        if lines[coord_idx].lower().startswith("s"):
            coord_idx += 1
        coord_type = lines[coord_idx].lower()
        pos_start = coord_idx + 1
        n_atoms = int(sum(counts))
        pos_lines = lines[pos_start : pos_start + n_atoms]
        if len(pos_lines) < n_atoms:
            return None
        coords = np.array([[float(x) for x in ln.split()[:3]] for ln in pos_lines], dtype=float)
        symbols: List[str] = []
        for e, c in zip(elems, counts):
            symbols.extend([e] * int(c))
        symbols = symbols[:n_atoms]
        if coord_type.startswith("d"):
            frac = coords.copy()
            cart = frac @ lattice
        else:
            cart = coords * abs(float(scale_tokens[0])) if len(scale_tokens) == 1 else coords
            frac = cart @ np.linalg.inv(lattice)
        return {"path": str(path), "lattice": lattice, "symbols": symbols, "frac": frac, "cart": cart}
    except Exception:
        return None


def host_element(material: str, symbols: Sequence[str], dopant: str) -> str:
    mat = str(material or "")
    if mat.startswith("CeO2"):
        return "Ce"
    if mat.startswith("ZnO"):
        return "Zn"
    counts: Dict[str, int] = {}
    for s in symbols:
        if s not in {"H", "O", dopant}:
            counts[s] = counts.get(s, 0) + 1
    if counts:
        return max(counts.items(), key=lambda kv: kv[1])[0]
    for s in symbols:
        if s not in {"H", "O"}:
            return s
    return ""


def pbc_flags(raw: str) -> Tuple[bool, bool, bool]:
    raw = str(raw).lower()
    if raw == "xyz":
        return True, True, True
    if raw == "xy":
        return True, True, False
    return False, False, False


def distance_matrix(struct: Dict[str, object], idx_a: Sequence[int], idx_b: Sequence[int], pbc: Tuple[bool, bool, bool]) -> np.ndarray:
    if not idx_a or not idx_b:
        return np.zeros((len(idx_a), len(idx_b)), dtype=float) * np.nan
    frac = np.asarray(struct["frac"], dtype=float)
    lattice = np.asarray(struct["lattice"], dtype=float)
    fa = frac[np.asarray(idx_a, dtype=int)]
    fb = frac[np.asarray(idx_b, dtype=int)]
    diff = fa[:, None, :] - fb[None, :, :]
    for ax, use_pbc in enumerate(pbc):
        if use_pbc:
            diff[:, :, ax] -= np.round(diff[:, :, ax])
    cart_diff = diff @ lattice
    return np.linalg.norm(cart_diff, axis=2)


def safe_stat(vals: Sequence[float], fn: str) -> float:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if fn == "min":
        return float(np.min(arr))
    if fn == "max":
        return float(np.max(arr))
    if fn == "mean":
        return float(np.mean(arr))
    if fn == "std":
        return float(np.std(arr))
    if fn == "median":
        return float(np.median(arr))
    raise ValueError(fn)


def sorted_dist_features(prefix: str, out: Dict[str, float], vals: Sequence[float], ks: Sequence[int]) -> None:
    arr = np.asarray(vals, dtype=float)
    arr = np.sort(arr[np.isfinite(arr)])
    for k in ks:
        out[f"{prefix}_d{k}"] = float(arr[k - 1]) if arr.size >= k else float("nan")
    out[f"{prefix}_min"] = float(arr[0]) if arr.size else float("nan")
    out[f"{prefix}_mean3"] = float(np.mean(arr[:3])) if arr.size else float("nan")
    out[f"{prefix}_mean5"] = float(np.mean(arr[:5])) if arr.size else float("nan")


def count_within(vals: Sequence[float], cutoff: float) -> float:
    arr = np.asarray(vals, dtype=float)
    return float(np.sum(np.isfinite(arr) & (arr <= cutoff)))


def extract_structure_features(
    struct: Optional[Dict[str, object]],
    dopant: str,
    material: str,
    prefix: str,
    pbc: Tuple[bool, bool, bool],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    keys = [
        "path_exists", "parse_ok", "n_atoms", "n_h", "n_o", "n_dopant", "n_metal", "n_elements",
        "h_frac_z", "h_cart_z", "h_height_above_nonh_top", "h_o_min", "h_o_d2", "h_o_d3",
        "h_o_mean3", "h_o_coord_1p2", "h_o_coord_1p5", "h_o_coord_2p0", "h_dopant_min",
        "h_host_min", "h_metal_min", "h_nearest_metal_Z", "h_nearest_o_frac_z",
        "h_minus_o_frac_z", "h_minus_o_cart_z", "h_nearest_o_metal_coord_2p6",
        "h_nearest_o_metal_coord_3p0", "h_nearest_o_dopant_min", "dopant_frac_z",
        "dopant_cart_z", "dopant_height_above_nonh_top", "dopant_o_min", "dopant_o_d2",
        "dopant_o_d3", "dopant_o_mean3", "dopant_o_mean5", "dopant_o_coord_2p2",
        "dopant_o_coord_2p5", "dopant_o_coord_2p8", "dopant_o_coord_3p2",
        "dopant_h_min", "dopant_host_min", "host_o_min", "host_o_mean3",
        "slab_thickness_z", "nonh_top_cart_z", "nonh_bottom_cart_z",
    ]
    for k in keys:
        out[f"{prefix}_{k}"] = float("nan")
    out[f"{prefix}_path_exists"] = 0.0
    out[f"{prefix}_parse_ok"] = 0.0
    if struct is None:
        return out

    symbols = list(struct["symbols"])
    cart = np.asarray(struct["cart"], dtype=float)
    frac = np.asarray(struct["frac"], dtype=float)
    n = len(symbols)
    idx_h = [i for i, s in enumerate(symbols) if s == "H"]
    idx_o = [i for i, s in enumerate(symbols) if s == "O"]
    dopant = str(dopant)
    idx_dop = [i for i, s in enumerate(symbols) if s == dopant]
    host = host_element(material, symbols, dopant)
    idx_host = [i for i, s in enumerate(symbols) if s == host]
    idx_metal = [i for i, s in enumerate(symbols) if s not in {"H", "O"}]
    idx_nonh = [i for i, s in enumerate(symbols) if s != "H"]
    nonh_z = cart[idx_nonh, 2] if idx_nonh else cart[:, 2]
    top_z = safe_stat(nonh_z, "max")
    bottom_z = safe_stat(nonh_z, "min")
    out[f"{prefix}_path_exists"] = 1.0
    out[f"{prefix}_parse_ok"] = 1.0
    out[f"{prefix}_n_atoms"] = float(n)
    out[f"{prefix}_n_h"] = float(len(idx_h))
    out[f"{prefix}_n_o"] = float(len(idx_o))
    out[f"{prefix}_n_dopant"] = float(len(idx_dop))
    out[f"{prefix}_n_metal"] = float(len(idx_metal))
    out[f"{prefix}_n_elements"] = float(len(set(symbols)))
    out[f"{prefix}_nonh_top_cart_z"] = top_z
    out[f"{prefix}_nonh_bottom_cart_z"] = bottom_z
    out[f"{prefix}_slab_thickness_z"] = top_z - bottom_z if np.isfinite(top_z) and np.isfinite(bottom_z) else float("nan")

    h_idx: Optional[int] = None
    if idx_h:
        h_idx = max(idx_h, key=lambda i: frac[i, 2])
        out[f"{prefix}_h_frac_z"] = float(frac[h_idx, 2])
        out[f"{prefix}_h_cart_z"] = float(cart[h_idx, 2])
        out[f"{prefix}_h_height_above_nonh_top"] = float(cart[h_idx, 2] - top_z) if np.isfinite(top_z) else float("nan")
        if idx_o:
            h_o = distance_matrix(struct, [h_idx], idx_o, pbc).reshape(-1)
            sorted_dist_features(f"{prefix}_h_o", out, h_o, [1, 2, 3])
            out[f"{prefix}_h_o_coord_1p2"] = count_within(h_o, 1.2)
            out[f"{prefix}_h_o_coord_1p5"] = count_within(h_o, 1.5)
            out[f"{prefix}_h_o_coord_2p0"] = count_within(h_o, 2.0)
            nearest_o = idx_o[int(np.nanargmin(h_o))]
            out[f"{prefix}_h_nearest_o_frac_z"] = float(frac[nearest_o, 2])
            out[f"{prefix}_h_minus_o_frac_z"] = float(frac[h_idx, 2] - frac[nearest_o, 2])
            out[f"{prefix}_h_minus_o_cart_z"] = float(cart[h_idx, 2] - cart[nearest_o, 2])
            if idx_metal:
                o_m = distance_matrix(struct, [nearest_o], idx_metal, pbc).reshape(-1)
                out[f"{prefix}_h_nearest_o_metal_coord_2p6"] = count_within(o_m, 2.6)
                out[f"{prefix}_h_nearest_o_metal_coord_3p0"] = count_within(o_m, 3.0)
            if idx_dop:
                o_d = distance_matrix(struct, [nearest_o], idx_dop, pbc).reshape(-1)
                out[f"{prefix}_h_nearest_o_dopant_min"] = safe_stat(o_d, "min")
        if idx_dop:
            h_d = distance_matrix(struct, [h_idx], idx_dop, pbc).reshape(-1)
            out[f"{prefix}_h_dopant_min"] = safe_stat(h_d, "min")
        if idx_host:
            h_host = distance_matrix(struct, [h_idx], idx_host, pbc).reshape(-1)
            out[f"{prefix}_h_host_min"] = safe_stat(h_host, "min")
        if idx_metal:
            h_m = distance_matrix(struct, [h_idx], idx_metal, pbc).reshape(-1)
            out[f"{prefix}_h_metal_min"] = safe_stat(h_m, "min")
            if np.isfinite(out[f"{prefix}_h_metal_min"]):
                nearest_m = idx_metal[int(np.nanargmin(h_m))]
                out[f"{prefix}_h_nearest_metal_Z"] = float(Z_TABLE.get(symbols[nearest_m], np.nan))

    dop_idx: Optional[int] = None
    if idx_dop:
        if h_idx is not None:
            d_h = distance_matrix(struct, idx_dop, [h_idx], pbc).reshape(-1)
            dop_idx = idx_dop[int(np.nanargmin(d_h))]
        else:
            dop_idx = max(idx_dop, key=lambda i: frac[i, 2])
        out[f"{prefix}_dopant_frac_z"] = float(frac[dop_idx, 2])
        out[f"{prefix}_dopant_cart_z"] = float(cart[dop_idx, 2])
        out[f"{prefix}_dopant_height_above_nonh_top"] = float(cart[dop_idx, 2] - top_z) if np.isfinite(top_z) else float("nan")
        if idx_o:
            d_o = distance_matrix(struct, [dop_idx], idx_o, pbc).reshape(-1)
            sorted_dist_features(f"{prefix}_dopant_o", out, d_o, [1, 2, 3])
            out[f"{prefix}_dopant_o_coord_2p2"] = count_within(d_o, 2.2)
            out[f"{prefix}_dopant_o_coord_2p5"] = count_within(d_o, 2.5)
            out[f"{prefix}_dopant_o_coord_2p8"] = count_within(d_o, 2.8)
            out[f"{prefix}_dopant_o_coord_3p2"] = count_within(d_o, 3.2)
        if idx_h:
            d_h = distance_matrix(struct, [dop_idx], idx_h, pbc).reshape(-1)
            out[f"{prefix}_dopant_h_min"] = safe_stat(d_h, "min")
        host_others = [i for i in idx_host if i != dop_idx]
        if host_others:
            d_host = distance_matrix(struct, [dop_idx], host_others, pbc).reshape(-1)
            out[f"{prefix}_dopant_host_min"] = safe_stat(d_host, "min")

    if idx_host and idx_o:
        # Use the uppermost host atom as a stable surface proxy.
        surf_host = max(idx_host, key=lambda i: frac[i, 2])
        h_o = distance_matrix(struct, [surf_host], idx_o, pbc).reshape(-1)
        out[f"{prefix}_host_o_min"] = safe_stat(h_o, "min")
        out[f"{prefix}_host_o_mean3"] = float(np.mean(np.sort(h_o[np.isfinite(h_o)])[:3])) if np.isfinite(h_o).any() else float("nan")

    return out


def add_delta_features(row: Dict[str, float]) -> None:
    pairs = [
        "dopant_o_min", "dopant_o_d2", "dopant_o_d3", "dopant_o_mean3", "dopant_o_mean5",
        "dopant_o_coord_2p2", "dopant_o_coord_2p5", "dopant_o_coord_2p8", "dopant_o_coord_3p2",
        "dopant_frac_z", "dopant_cart_z", "dopant_height_above_nonh_top",
        "host_o_min", "host_o_mean3", "slab_thickness_z",
    ]
    for k in pairs:
        a = row.get(f"geom_addh_{k}", np.nan)
        b = row.get(f"geom_bare_{k}", np.nan)
        row[f"geom_delta_{k}"] = float(a) - float(b) if np.isfinite(a) and np.isfinite(b) else float("nan")


def build_geometry_table(df: pd.DataFrame, root: Path, args: argparse.Namespace) -> pd.DataFrame:
    pbc = pbc_flags(args.pbc_axes)
    rows: List[Dict[str, float]] = []
    for i, rec in df.iterrows():
        sid = str(rec.get(args.id_col, i))
        dopant = str(rec.get("dopant", ""))
        material = str(rec.get("material", ""))
        addh_path = resolve_path(root, rec.get(args.contcar_col, ""))
        bare_path = resolve_path(root, rec.get(args.bare_contcar_col, ""))
        addh_struct = parse_poscar(addh_path)
        bare_struct = parse_poscar(bare_path)
        out: Dict[str, float] = {args.id_col: sid}
        out.update(extract_structure_features(addh_struct, dopant, material, "geom_addh", pbc))
        out.update(extract_structure_features(bare_struct, dopant, material, "geom_bare", pbc))
        add_delta_features(out)
        rows.append(out)
        if (i + 1) % 100 == 0:
            print(f"[INFO] parsed {i + 1}/{len(df)}")
    return pd.DataFrame(rows)


def merge_augmented(df: pd.DataFrame, geom: pd.DataFrame, id_col: str) -> pd.DataFrame:
    drop_cols = [c for c in df.columns if str(c).startswith("geom_")]
    base = df.drop(columns=drop_cols, errors="ignore")
    return base.merge(geom, on=id_col, how="left")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train_features)
    addhout = pd.read_csv(args.addhout_features)
    if args.id_col not in train.columns or args.id_col not in addhout.columns:
        raise SystemExit(f"[ERROR] id column {args.id_col!r} missing.")
    for c in [args.contcar_col, args.bare_contcar_col]:
        if c not in train.columns or c not in addhout.columns:
            raise SystemExit(f"[ERROR] path column {c!r} missing from train/addH-out features.")

    print(f"[INFO] root={root}")
    print(f"[INFO] train rows={len(train)} addH-out rows={len(addhout)}")
    geom_train = build_geometry_table(train, root, args)
    geom_out = build_geometry_table(addhout, root, args)

    train_aug = merge_augmented(train, geom_train, args.id_col)
    out_aug = merge_augmented(addhout, geom_out, args.id_col)

    geom_train.to_csv(out_dir / "local_geometry_features_train.csv", index=False)
    geom_out.to_csv(out_dir / "local_geometry_features_addhout.csv", index=False)
    train_aug.to_csv(out_dir / "knowledge_features_train_geom.csv", index=False)
    out_aug.to_csv(out_dir / "knowledge_features_addhout_geom.csv", index=False)

    manifest = {
        "root": str(root),
        "train_features": args.train_features,
        "addhout_features": args.addhout_features,
        "out_dir": str(out_dir),
        "pbc_axes": args.pbc_axes,
        "n_train": int(len(train)),
        "n_addhout": int(len(addhout)),
        "n_geometry_cols": int(len([c for c in geom_train.columns if c != args.id_col])),
        "train_parse_ok_addh": float(pd.to_numeric(geom_train.get("geom_addh_parse_ok"), errors="coerce").mean()),
        "train_parse_ok_bare": float(pd.to_numeric(geom_train.get("geom_bare_parse_ok"), errors="coerce").mean()),
        "addhout_parse_ok_addh": float(pd.to_numeric(geom_out.get("geom_addh_parse_ok"), errors="coerce").mean()),
        "addhout_parse_ok_bare": float(pd.to_numeric(geom_out.get("geom_bare_parse_ok"), errors="coerce").mean()),
        "outputs": {
            "geometry_train": str(out_dir / "local_geometry_features_train.csv"),
            "geometry_addhout": str(out_dir / "local_geometry_features_addhout.csv"),
            "augmented_train": str(out_dir / "knowledge_features_train_geom.csv"),
            "augmented_addhout": str(out_dir / "knowledge_features_addhout_geom.csv"),
        },
    }
    with (out_dir / "local_geometry_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    print(f"[OK] wrote geometry features to {out_dir}")


if __name__ == "__main__":
    main()
