#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
v4: Surface-energy oriented pipeline
- Fetch metal oxides from Materials Project
- 2D supercell only (ax, ay) with az=1 to reach target total atom count (~N_target)
- Remove the topmost O (final total atoms ≈ N_target-1)
- Add symmetric vacuum along z (copied from a reference CIF or manual)
- Identify TOP/BOTTOM metal surface layers (tol in Å)
- Do dopant:host ≈ 1:3 substitution on surface metal sites (both sides by default)
- Export CIFs + manifest

Requirements:
  pip install "pymatgen>=2023.0.0" mp-api numpy pandas
Env:
  export MAPI_KEY=YOUR_MP_KEY
"""

import os, re, argparse
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Structure, Element, Lattice
from pymatgen.transformations.standard_transformations import ConventionalCellTransformation

ALL_METALS = [
    "Li","Be","Na","Mg","K","Ca","Rb","Sr","Cs","Ba","Fr","Ra",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Al","Ga","In","Sn","Tl","Pb","Bi",
    "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
]

# ---------- helpers ----------
def is_metal_oxide(struct: Structure) -> bool:
    els = [sp.symbol for sp in struct.composition.elements]
    return ("O" in els) and any(e.is_metal for e in struct.composition.elements if e.symbol != "O")

def best_2d_supercell_for_target_N(struct: Structure, N_target: int) -> Tuple[int,int]:
    """
    Choose (ax, ay) (with az=1) so that total atoms after supercell and removing 1 O
    is as close as possible to N_target-1.
    Search small (ax, ay) up to 8x8 by default for practicality.
    """
    N0 = len(struct)
    target_after = max(1, N_target - 1)  # after removing one O
    best = (1,1); best_err = 10**9
    for ax in range(1, 9):
        for ay in range(1, 9):
            N = N0 * ax * ay - 1  # after removing one O
            err = abs(N - target_after)
            if err < best_err:
                best, best_err = (ax, ay), err
    return best

def make_2d_supercell(struct: Structure, ax:int, ay:int) -> Structure:
    s = struct.copy()
    s.make_supercell([ax, ay, 1])
    return s

def remove_topmost_oxygen(struct: Structure) -> Tuple[Structure, int, float, int]:
    coords = np.array([s.coords for s in struct])
    o_idx = [i for i,s in enumerate(struct.sites) if s.specie.symbol == "O"]
    if not o_idx: raise RuntimeError("no oxygen atom")
    top = max(o_idx, key=lambda i: coords[i,2])
    zmax = float(coords[top,2])
    s2 = struct.copy()
    s2.remove_sites([top])
    new_total = len(s2)
    return s2, top, zmax, new_total

def parse_ref_vac_and_tol(ref_cif: Optional[str]) -> Tuple[Optional[float], float]:
    """
    Return (total_vacuum_A or None, tol_A).
    Vacuum: from c_len - occupied extent (using Cartn_z if present else fract_z*c).
    tol: half the gap between top and next metal layers (clamped 0.6..2.5 Å).
    """
    default_tol = 1.2
    if (ref_cif is None) or (not os.path.exists(ref_cif)):
        return None, default_tol
    txt = open(ref_cif, "r", encoding="utf-8", errors="ignore").read()
    def fval(pat):
        m = re.search(pat, txt, re.IGNORECASE)
        if not m: return None
        sval = re.sub(r"\([^)]*\)", "", m.group(1))
        try: return float(sval)
        except: return None
    c_len = fval(r"_cell_length_c\s+([0-9\.\(\)Ee+\-]+)")
    # parse atom sites
    lines = txt.splitlines()
    start, headers, rows = None, [], []
    i=0
    while i < len(lines):
        if lines[i].strip().lower().startswith("loop_"):
            j = i+1; tmp=[]
            while j < len(lines) and lines[j].strip().startswith("_"):
                tmp.append(lines[j].strip()); j+=1
            if any(h.lower().startswith("_atom_site_") for h in tmp):
                start = i; headers = tmp[:]
                k = j
                while k < len(lines):
                    s = lines[k].strip()
                    if (not s) or s.lower().startswith("loop_") or s.lower().startswith("data_") or s.startswith("_"):
                        break
                    toks = [m.group(0).strip("'").strip('"') for m in re.finditer(r"(?:'[^']*'|\"[^\"]*\"|\S+)", s)]
                    if toks: rows.append(toks)
                    k+=1
                break
            else:
                i=j; continue
        i+=1
    if not headers or not rows:
        return None, default_tol
    headers = [h.split()[0] for h in headers]
    def col(*cands):
        for c in cands:
            if c in headers: return headers.index(c)
        return None
    c_sym = col("_atom_site_type_symbol") or col("_atom_site_label")
    c_z   = col("_atom_site_Cartn_z")
    c_fz  = col("_atom_site_fract_z")
    def fnum(s):
        try: return float(re.sub(r"\([^)]*\)", "", s))
        except: return None
    METALS = set(ALL_METALS)

    zs, z_metal = [], []
    for r in rows:
        if len(r) < len(headers):
            r = r + ["?"]*(len(headers)-len(r))
        sym = r[c_sym]
        z = fnum(r[c_z]) if c_z is not None else None
        fz = fnum(r[c_fz]) if c_fz is not None else None
        if z is None and (fz is not None) and (c_len is not None):
            z = fz * c_len
        if z is not None:
            zs.append(z)
            if sym.strip("0123456789") in METALS:
                z_metal.append(z)

    total_vac = None
    if (c_len is not None) and zs:
        extent = max(zs) - min(zs)
        total_vac = max(0.0, c_len - extent)

    tol = default_tol
    if len(z_metal) >= 2:
        z_metal.sort()
        top = z_metal[-1]
        nxt = None
        for val in reversed(z_metal[:-1]):
            if abs(top - val) > 1e-3:
                nxt = val; break
        if nxt is not None:
            gapA = top - nxt
            tol = max(0.6, min(2.5, 0.5 * gapA))
    return total_vac, tol

def add_symmetric_vacuum(struct: Structure, total_vacuum_A: float) -> Structure:
    if total_vacuum_A is None or total_vacuum_A <= 1e-6:
        return struct.copy()
    s = struct.copy()
    z = np.array([site.coords[2] for site in s])
    zmin, zmax = float(np.min(z)), float(np.max(z))
    slab_thick = zmax - zmin
    old_lat = s.lattice
    new_c = slab_thick + total_vacuum_A
    scale = new_c / old_lat.c
    new_cvec = old_lat.matrix[2] * scale
    new_lat = Lattice([old_lat.matrix[0], old_lat.matrix[1], new_cvec])
    z_center_old = 0.5*(zmin + zmax)
    shift = np.array([0,0, (new_c/2.0) - z_center_old])
    cart = np.array([site.coords for site in s])
    cart_shifted = cart + shift
    new_s = Structure(new_lat, [site.specie for site in s], cart_shifted, coords_are_cartesian=True, to_unit_cell=True)
    return new_s

def get_surface_metal_indices_both(struct: Structure, tol_A: float) -> Tuple[List[int], List[int]]:
    coords = np.array([s.coords for s in struct])
    z = coords[:,2]
    metals = [i for i,s in enumerate(struct.sites) if isinstance(s.specie, Element) and s.specie.is_metal and s.specie.symbol != "O"]
    if not metals: return [], []
    zmax = float(np.max(z[metals])); zmin = float(np.min(z[metals]))
    top = sorted([i for i in metals if (zmax - z[i]) <= tol_A])
    bot = sorted([i for i in metals if (z[i] - zmin) <= tol_A])
    return top, bot

def choose_sites(indices: List[int], dopant_to_host: Tuple[int,int], k_override: Optional[int], rng: np.random.RandomState) -> List[int]:
    if not indices: return []
    if k_override is not None:
        k = max(1, min(len(indices), int(k_override)))
    else:
        d,h = dopant_to_host
        frac = d / (d + h)  # e.g., 1:3 -> 0.25
        k = max(1, min(len(indices), int(round(len(indices)*frac))))
    return sorted(rng.choice(indices, size=k, replace=False).tolist())

def substitute_species(struct: Structure, idx_list: List[int], dopant: str) -> Tuple[Structure, List[int]]:
    s2 = struct.copy()
    applied = []
    for i in idx_list:
        if s2[i].specie.symbol != dopant:
            s2[i] = Element(dopant); applied.append(i)
    return s2, applied

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_cif", type=str, default=None, help="reference CIF to copy vacuum & infer tol")
    ap.add_argument("--outdir", type=str, default="mp_oxides_surface_energy_v4_out")
    ap.add_argument("--dopants", type=str, default="ALL", help="ALL or comma list")
    ap.add_argument("--dopant_to_host", type=str, default="1:3", help="ratio per side (1:3 ≈ 25%)")
    ap.add_argument("--k_top", type=int, default=None, help="override #doped sites on top surface")
    ap.add_argument("--k_bot", type=int, default=None, help="override #doped sites on bottom surface")
    ap.add_argument("--target_atoms", type=int, default=95, help="target total atoms after supercell & O removal (~95)")
    ap.add_argument("--surface_tol", type=float, default=None, help="manual tol (Å)")
    ap.add_argument("--max_docs", type=int, default=100, help="limit number of materials")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # dopants
    if args.dopants.strip().upper() == "ALL":
        dopant_list = ALL_METALS[:]
    else:
        dopant_list = [x.strip() for x in args.dopants.split(",") if x.strip()]
        dopant_list = [d for d in dopant_list if d in ALL_METALS]
        if not dopant_list:
            raise ValueError("No valid metallic dopants.")

    # ratios
    try:
        d,h = [int(x) for x in args.dopant_to_host.split(":")]
        assert d>0 and h>0
    except Exception:
        raise ValueError("--dopant_to_host must be 'd:h', e.g., 1:3")

    # vacuum & tol
    ref_vac, tol_infer = parse_ref_vac_and_tol(args.ref_cif)
    tol = args.surface_tol if args.surface_tol is not None else tol_infer
    print(f"[Info] tol = {tol:.3f} Å ; ref total vacuum = {ref_vac if ref_vac is not None else 'None'} Å")

    api_key = os.environ.get("MAPI_KEY")
    if not api_key:
        raise RuntimeError("Missing MAPI_KEY env var.")
    rng = np.random.RandomState(args.seed)

    rows = []

    with MPRester(api_key) as mpr:
        # fetch oxides; then local filter for metals
        docs = list(mpr.summary.search(elements=["O"], fields=["material_id","formula_pretty","structure"], chunk_size=200))
        if args.max_docs and len(docs) > args.max_docs:
            docs = docs[:args.max_docs]

        for ddoc in docs:
            try:
                struct: Structure = ddoc.structure
            except Exception:
                continue
            if not is_metal_oxide(struct):
                continue

            # choose 2D supercell factors for target atoms
            ax, ay = best_2d_supercell_for_target_N(struct, args.target_atoms)
            sup = make_2d_supercell(struct, ax, ay)

            # remove topmost O
            try:
                sup2, del_idx, zO, N_after = remove_topmost_oxygen(sup)
            except Exception as e:
                rows.append({"material_id": ddoc.material_id, "formula": getattr(ddoc,"formula_pretty",""),
                             "status": "skip_no_O", "reason": str(e)})
                continue

            # add vacuum (copy from ref)
            if ref_vac is not None and ref_vac > 1e-6:
                sup3 = add_symmetric_vacuum(sup2, total_vacuum_A=ref_vac)
            else:
                sup3 = sup2

            # surfaces
            top_idx, bot_idx = get_surface_metal_indices_both(sup3, tol_A=tol)
            if not top_idx and not bot_idx:
                rows.append({"material_id": ddoc.material_id, "formula": getattr(ddoc,"formula_pretty",""),
                             "status": "skip_no_surface_metals"})
                continue

            for dop in dopant_list:
                pick_top = choose_sites(top_idx, (d,h), args.k_top, rng) if top_idx else []
                pick_bot = choose_sites(bot_idx, (d,h), args.k_bot, rng) if bot_idx else []
                picks = sorted(set(pick_top + pick_bot))
                doped, applied = substitute_species(sup3, picks, dop)
                outname = f"{ddoc.material_id}_2D_{ax}x{ay}_N{N_after}_bisurf_{dop}_d{d}_h{h}.cif"
                outpath = os.path.join(args.outdir, outname)
                doped.to(fmt="cif", filename=outpath)
                rows.append({
                    "material_id": ddoc.material_id, "formula": getattr(ddoc,"formula_pretty",""),
                    "status": "ok", "outfile": outpath, "dopant": dop,
                    "supercell_2d": f"{ax}x{ay}x1",
                    "deleted_O_index": del_idx, "deleted_O_z": zO,
                    "final_total_atoms": N_after,
                    "surface_tol_A": tol,
                    "top_surface_sites": len(top_idx), "bot_surface_sites": len(bot_idx),
                    "picked_top": len(pick_top), "picked_bot": len(pick_bot),
                    "applied_total": len(applied), "applied_indices": ";".join(map(str, applied))
                })

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "manifest.csv"), index=False)
        print(f"[OK] Wrote {sum(r.get('status')=='ok' for r in rows)} doped samples; total records {len(rows)}.")
    else:
        print("[WARN] No outputs; check filters/params.]")

if __name__ == "__main__":
    main()
