#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v5: template-guided (host_ref & doped_ref) surface-energy pipeline

- Geometry template from host_ref (e.g., 0-out.cif):
  * Copy total vacuum thickness along c
  * Infer surface-layer thickness tol (Å) from metal interlayer gap
- Doping template from doped_ref (e.g., 2-out.cif):
  * If --dopants not given, infer dopant metals from doped_ref (metals present there but minority/new)
  * If --sides is 'auto', infer doping sidedness (top/bottom/both) by dopant z-distribution in doped_ref
  * If --host_species not given, try to infer likely host sublattice as the dominant metal(s) in host_ref
    (you can override with --host_species "Ce,Zr" etc.)

- Workflow:
  * Fetch metal oxides (MP Summary API)
  * XY-only supercell (az=1) to reach ~ target total atoms (after deleting one top O)
  * Delete one O at the top along c (fractional z with unwrap)
  * Add symmetric vacuum along c copied from host_ref; center slab in fractional coords
  * Identify TOP/BOTTOM metal surface layers using tol (Å)
  * Global dopant:host ≈ 1:3 on chosen sides, avoid no-op, backfill to meet target
  * Export CIFs + manifest.csv

Dependencies: pymatgen>=2023, mp-api, numpy, pandas
Env: export MAPI_KEY=...
"""

import os, re, argparse
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Structure, Element, Lattice
from pymatgen.transformations.standard_transformations import ConventionalCellTransformation
from collections import Counter

ALL_METALS = [
    "Li","Be","Na","Mg","K","Ca","Rb","Sr","Cs","Ba","Fr","Ra",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Al","Ga","In","Sn","Tl","Pb","Bi",
    "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
]

# -------------------- utilities --------------------
def is_metal_oxide(struct: Structure) -> bool:
    els = [sp.symbol for sp in struct.composition.elements]
    return ("O" in els) and any(e.is_metal for e in struct.composition.elements if e.symbol != "O")

def best_2d_supercell_for_target_N(struct: Structure, N_target: int, max_mul: int = 12) -> Tuple[int,int]:
    N0 = len(struct)
    target_after = max(1, N_target)  # we will delete one O after supercell
    best = (1,1); best_err = 10**9
    for ax in range(1, max_mul+1):
        for ay in range(1, max_mul+1):
            N_after = N0 * ax * ay - 1
            err = abs(N_after - target_after)
            if err < best_err:
                best, best_err = (ax, ay), err
    return best

def make_2d_supercell(struct: Structure, ax:int, ay:int) -> Structure:
    s = struct.copy()
    s.make_supercell([ax, ay, 1])
    return s

def _unwrap_frac_z(z: np.ndarray) -> np.ndarray:
    z_sorted = np.sort(z)
    gaps = np.diff(np.r_[z_sorted, z_sorted[0]+1.0])
    k = int(np.argmax(gaps))
    base = z_sorted[(k+1)%len(z_sorted)]
    zu = z - base
    zu[zu<0]+=1.0
    return zu

def unwrap_frac_along_c(struct: Structure) -> np.ndarray:
    fz = np.array([site.frac_coords[2] for site in struct.sites])
    return _unwrap_frac_z(fz)

def remove_topmost_oxygen_by_frac(struct: Structure) -> Tuple[Structure, int, float, int]:
    s = struct.copy()
    fz = np.array([site.frac_coords[2] for site in s])
    zu = _unwrap_frac_z(fz)
    o_idx = [i for i,site in enumerate(s.sites) if site.specie.symbol == "O"]
    if not o_idx: raise RuntimeError("no oxygen atom")
    top = max(o_idx, key=lambda i: zu[i])
    top_z = float(zu[top])
    s.remove_sites([top])
    return s, top, top_z, len(s)

def parse_cif_atoms(ref_cif: str) -> Dict[str, List[float]]:
    if (ref_cif is None) or (not os.path.exists(ref_cif)):
        return {}
    try:
        s = Structure.from_file(ref_cif)
        return {"_pz": [site.frac_coords[2] for site in s.sites],
                "_sym": [site.specie.symbol for site in s.sites],
                "_struct": s}
    except Exception:
        return {}

def infer_vac_and_tol_from_host(ref_cif: Optional[str]) -> Tuple[Optional[float], float, Optional[Structure]]:
    default_tol = 1.2
    if (ref_cif is None) or (not os.path.exists(ref_cif)):
        return None, default_tol, None
    try:
        s = Structure.from_file(ref_cif)
    except Exception:
        return None, default_tol, None
    c_len = s.lattice.c
    zf = unwrap_frac_along_c(s)
    extent_A = (zf.max() - zf.min()) * c_len
    vac = max(0.0, c_len - extent_A)

    metals = [i for i,site in enumerate(s.sites) if isinstance(site.specie, Element) and site.specie.is_metal and site.specie.symbol != "O"]
    tol = default_tol
    if len(metals) >= 2:
        z_m = zf[metals]
        z_m.sort()
        top = z_m[-1]
        nxt = None
        for val in z_m[-2::-1]:
            if abs(top-val) > 1e-4:
                nxt = val; break
        if nxt is not None:
            gapA = (top - nxt) * c_len
            tol = max(0.6, min(2.5, 0.5*gapA))
    return vac, tol, s

from collections import Counter
def infer_dopants_and_sides_from_doped(ref_cif: Optional[str], host_struct: Optional[Structure]) -> Tuple[List[str], str]:
    if (ref_cif is None) or (not os.path.exists(ref_cif)):
        return [], "both"
    try:
        ds = Structure.from_file(ref_cif)
    except Exception:
        return [], "both"

    metal_sites = [site.specie.symbol for site in ds.sites if isinstance(site.specie, Element) and site.specie.is_metal and site.specie.symbol != "O"]
    cnt = Counter(metal_sites)
    candidate = set()
    if host_struct is not None:
        host_metals = [site.specie.symbol for site in host_struct.sites if isinstance(site.specie, Element) and site.specie.is_metal and site.specie.symbol != "O"]
        host_top = {m for m,c in Counter(host_metals).most_common(3)}
    else:
        host_top = set()
    if cnt:
        med = np.median(list(cnt.values()))
        for m, c in cnt.items():
            if (m not in host_top and m in ALL_METALS) or (c <= 0.5*med):
                candidate.add(m)
    dopants = sorted(list(candidate)) if candidate else sorted(list({m for m in cnt.keys() if m in ALL_METALS}))

    zu = unwrap_frac_along_c(ds)
    metals_idx = [i for i,site in enumerate(ds.sites) if isinstance(site.specie, Element) and site.specie.is_metal and site.specie.symbol != "O"]
    if not metals_idx:
        return dopants, "both"
    z_m = zu[metals_idx]
    sym_m = [ds[i].specie.symbol for i in metals_idx]
    dop_idx = [i for i,sy in zip(metals_idx, sym_m) if sy in dopants]
    if not dop_idx:
        return dopants, "both"
    zu_d = zu[dop_idx]
    zmin, zmax = float(np.min(z_m)), float(np.max(z_m))
    thr = (zmax - zmin) * 0.25
    top_n = np.sum((zmax - zu_d) <= thr)
    bot_n = np.sum((zu_d - zmin) <= thr)
    sides = "both"
    if top_n > 0 and bot_n == 0:
        sides = "top"
    elif bot_n > 0 and top_n == 0:
        sides = "bottom"
    return dopants, sides

def add_symmetric_vacuum_along_c(struct: Structure, total_vac_A: float) -> Structure:
    if total_vac_A is None or total_vac_A <= 1e-6:
        return struct.copy()
    s = struct.copy()
    lat = s.lattice
    c_len_old = lat.c
    zu = unwrap_frac_along_c(s)
    span = zu.max() - zu.min()
    slab_thick_A = span * c_len_old
    new_c = slab_thick_A + total_vac_A
    scale = new_c / c_len_old
    new_cvec = lat.matrix[2] * scale
    new_lat = Lattice([lat.matrix[0], lat.matrix[1], new_cvec])
    center = 0.5*(zu.max()+zu.min())
    f = np.array([site.frac_coords for site in s])
    f[:,2] = f[:,2] - center + 0.5
    f[:,2] = f[:,2] - np.floor(f[:,2])
    return Structure(new_lat, [site.specie for site in s], f, coords_are_cartesian=False, to_unit_cell=True)

def get_surface_metal_indices_by_frac(struct: Structure, tol_A: float) -> Tuple[List[int], List[int]]:
    s = struct
    lat = s.lattice
    zu = unwrap_frac_along_c(s)
    metals = [i for i,site in enumerate(s.sites) if isinstance(site.specie, Element) and site.specie.is_metal and site.specie.symbol != "O"]
    if not metals: return [], []
    zmax = float(np.max(zu[metals])); zmin = float(np.min(zu[metals]))
    thr = tol_A / lat.c
    top = sorted([i for i in metals if (zmax - zu[i]) <= thr])
    bot = sorted([i for i in metals if (zu[i] - zmin) <= thr])
    return top, bot

def choose_sites_global_13_filtered(s: Structure, top_idx: List[int], bot_idx: List[int], dopant: str,
                                    d:int, h:int, rng: np.random.RandomState, sides: str, host_species: Optional[set]):
    idx_pool = []
    if sides in ("both","top"):
        idx_pool += top_idx
    if sides in ("both","bottom"):
        idx_pool += bot_idx
    if host_species:
        idx_pool = [i for i in idx_pool if s[i].specie.symbol in host_species]
    dopable = [i for i in idx_pool if s[i].specie.symbol != dopant]
    S = len(dopable)
    if S == 0:
        return [], 0, 0, 0
    target = int(round(S * d / (d + h)))
    k = max(0, min(S, target))
    picks = sorted(rng.choice(dopable, size=k, replace=False).tolist()) if k>0 else []
    return picks, S, target, k

def substitute_species(struct: Structure, indices: List[int], dopant: str):
    s2 = struct.copy()
    applied = []
    for i in indices:
        if s2[i].specie.symbol != dopant:
            s2[i] = Element(dopant); applied.append(i)
    return s2, applied

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host_ref", type=str, default="0-out.cif", help="host geometry template CIF (vacuum & tol)")
    ap.add_argument("--doped_ref", type=str, default="2-out.cif", help="doping-style template CIF (infer dopants/sides)")
    ap.add_argument("--outdir", type=str, default="out_v5")
    ap.add_argument("--dopants", type=str, default="", help="comma list; if empty, infer from doped_ref")
    ap.add_argument("--host_species", type=str, default="", help="comma list of host metals to substitute (e.g., 'Ce,Zr')")
    ap.add_argument("--sides", type=str, default="auto", choices=["auto","top","bottom","both"], help="which sides to dope")
    ap.add_argument("--dopant_to_host", type=str, default="1:3")
    ap.add_argument("--target_atoms", type=int, default=95)
    ap.add_argument("--surface_tol", type=float, default=None)
    ap.add_argument("--max_docs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ref_vac, tol_infer, host_struct = infer_vac_and_tol_from_host(args.host_ref)
    tol = args.surface_tol if args.surface_tol is not None else tol_infer

    user_dopants = [x.strip() for x in args.dopants.split(",") if x.strip()]
    if user_dopants:
        dopant_list = [d for d in user_dopants if d in ALL_METALS]
    else:
        dopant_list, sides_auto = infer_dopants_and_sides_from_doped(args.doped_ref, host_struct)
    sides = args.sides if args.sides != "auto" else (sides_auto if 'sides_auto' in locals() else "both")

    host_species = set([x.strip() for x in args.host_species.split(",") if x.strip()]) if args.host_species else None

    try:
        d, h = [int(x) for x in args.dopant_to_host.split(":")]
        assert d>0 and h>0
    except Exception:
        raise ValueError("--dopant_to_host must be 'd:h', e.g., 1:3")

    print(f"[Info] host_ref vacuum={ref_vac if ref_vac is not None else 'None'} Å ; tol={tol:.3f} Å ; sides={sides} ; dopants={dopant_list if dopant_list else '[]'}")

    api_key = os.environ.get("MAPI_KEY")
    if not api_key:
        raise RuntimeError("Missing MAPI_KEY")
    rng = np.random.RandomState(args.seed)

    rows = []

    with MPRester(api_key) as mpr:
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

            ax, ay = best_2d_supercell_for_target_N(struct, args.target_atoms)
            sup = make_2d_supercell(struct, ax, ay)

            try:
                sup2, del_idx, del_fz, N_after = remove_topmost_oxygen_by_frac(sup)
            except Exception as e:
                rows.append({"material_id": ddoc.material_id, "formula": getattr(ddoc,"formula_pretty",""),
                             "status": "skip_no_O", "reason": str(e)})
                continue

            sup3 = add_symmetric_vacuum_along_c(sup2, total_vac_A=ref_vac) if (ref_vac and ref_vac>1e-6) else sup2

            top_idx, bot_idx = get_surface_metal_indices_by_frac(sup3, tol_A=tol)
            if not top_idx and not bot_idx:
                rows.append({"material_id": ddoc.material_id, "formula": getattr(ddoc,"formula_pretty",""),
                             "status": "skip_no_surface_metals"})
                continue

            if not dopant_list:
                rows.append({"material_id": ddoc.material_id, "formula": getattr(ddoc,"formula_pretty",""),
                             "status": "skip_no_dopant_inferred"})
                continue

            for dop in dopant_list:
                picks, S, target, k = choose_sites_global_13_filtered(sup3, top_idx, bot_idx, dop, d, h, rng, sides=sides, host_species=host_species)
                doped, applied = substitute_species(sup3, picks, dop)
                if len(applied) < target:
                    pool = [i for i in (set(top_idx+bot_idx) if sides=='both' else (set(top_idx) if sides=='top' else set(bot_idx)))
                            if (not host_species or doped[i].specie.symbol in host_species) and doped[i].specie.symbol != dop]
                    remaining = [i for i in pool if i not in applied]
                    need = target - len(applied)
                    if need > 0 and remaining:
                        add_more = sorted(rng.choice(list(remaining), size=min(need, len(remaining)), replace=False).tolist())
                        doped, applied2 = substitute_species(doped, add_more, dop)
                        applied += applied2

                outname = f"{ddoc.material_id}_2D_{ax}x{ay}_N{N_after}_bisurf_{sides}_{dop}_d{d}_h{h}.cif"
                outpath = os.path.join(args.outdir, outname)
                doped.to(fmt="cif", filename=outpath)

                rows.append({
                    "material_id": ddoc.material_id, "formula": getattr(ddoc,"formula_pretty",""),
                    "status": "ok", "outfile": outpath, "dopant": dop, "sides": sides,
                    "supercell_2d": f"{ax}x{ay}x1",
                    "deleted_O_index": del_idx, "deleted_O_fz_unwrap": del_fz,
                    "final_total_atoms": N_after,
                    "surface_tol_A": tol,
                    "top_surface_sites": len(top_idx), "bot_surface_sites": len(bot_idx),
                    "dopable_sites": S, "target_replacements": target, "picked_initial": k,
                    "applied_total": len(applied), "applied_indices": ";".join(map(str, applied))
                })

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "manifest.csv"), index=False)
        print(f"[OK] Wrote {sum(r.get('status')=='ok' for r in rows)} doped samples; total records {len(rows)}.")
    else:
        print("[WARN] No outputs; check filters/params.]")

if __name__ == "__main__":
    main()
