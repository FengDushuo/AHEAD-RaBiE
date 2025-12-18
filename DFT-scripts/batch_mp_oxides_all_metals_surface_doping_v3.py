#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量获取 Materials Project 的金属氧化物，
扩胞到 O=64 → 删除 z 轴最大的 O → 留真空层 → 上下两侧表面金属位点 1:3 替位掺杂。
"""

import os, re, argparse
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Structure, Element, Lattice
from pymatgen.transformations.standard_transformations import ConventionalCellTransformation

# 全部金属元素
ALL_METALS = [
    "Li","Be","Na","Mg","K","Ca","Rb","Sr","Cs","Ba","Fr","Ra",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Al","Ga","In","Sn","Tl","Pb","Bi",
    "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
]

def is_metal_oxide(struct: Structure) -> bool:
    els = [sp.symbol for sp in struct.composition.elements]
    return ("O" in els) and any(e.is_metal for e in struct.composition.elements if e.symbol != "O")

def decompose_to_abc(factor: int) -> Tuple[int,int,int]:
    """把超胞因子分解成接近立方的 (a,b,c)。"""
    best = (1,1,factor); score = 1e9; root = round(factor ** (1/3))
    for a in range(1, factor+1):
        if factor % a: continue
        rem = factor // a
        for b in range(1, rem+1):
            if rem % b: continue
            c = rem // b
            s = abs(a-root) + abs(b-root) + abs(c-root)
            if s < score: best, score = (a,b,c), s
    return best

def find_supercell_for_O(struct: Structure, target_O: int = 64):
    """扩胞直到 O=64"""
    candidates = [struct, ConventionalCellTransformation().apply_transformation(struct), struct.get_primitive_structure()]
    for base in candidates:
        nO = sum(1 for sp in base.species if getattr(sp, "symbol", str(sp)) == "O")
        if nO == 0 or (target_O % nO != 0):
            continue
        factor = target_O // nO
        ax,ay,az = decompose_to_abc(factor)
        sup = base.copy()
        sup.make_supercell([ax,ay,az])
        nO2 = sum(1 for sp in sup.species if sp.symbol == "O")
        if nO2 == target_O:
            return sup, (ax,ay,az)
    return None

def remove_topmost_oxygen(struct: Structure):
    """删除 z 最大的 O"""
    coords = np.array([s.coords for s in struct])
    o_idx = [i for i,s in enumerate(struct.sites) if s.specie.symbol == "O"]
    top = max(o_idx, key=lambda i: coords[i,2])
    zmax = float(coords[top,2])
    s2 = struct.copy()
    s2.remove_sites([top])
    new_O = sum(1 for sp in s2.species if sp.symbol == "O")
    return s2, top, zmax, new_O

def parse_ref_vacuum_and_tol(ref_cif: Optional[str]):
    """解析参考结构中的真空层厚度和表面厚度 tol"""
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
    # 简化：这里直接取 c_len - slab 厚度 为真空
    zs = []
    for line in txt.splitlines():
        if "_atom_site_fract_z" in line or "_atom_site_Cartn_z" in line:
            continue
    ref_vac = None
    if c_len:
        # 粗略：直接返回 c_len*0.25 当真空
        ref_vac = c_len*0.25
    return ref_vac, default_tol

def add_symmetric_vacuum(struct: Structure, total_vacuum_A: float) -> Structure:
    """在 slab 上下加对称真空层"""
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

def get_surface_metal_indices_both(struct: Structure, tol_A: float):
    """找到上下两侧表面的金属原子索引"""
    coords = np.array([s.coords for s in struct])
    z = coords[:,2]
    metals = [i for i,s in enumerate(struct.sites) if isinstance(s.specie, Element) and s.specie.is_metal and s.specie.symbol != "O"]
    zmax, zmin = np.max(z[metals]), np.min(z[metals])
    top = [i for i in metals if (zmax - z[i]) <= tol_A]
    bot = [i for i in metals if (z[i] - zmin) <= tol_A]
    return top, bot

def choose_sites(indices: List[int], frac: float, rng):
    k = max(1, int(round(len(indices)*frac)))
    return sorted(rng.choice(indices, size=k, replace=False).tolist())

def substitute_species(struct: Structure, idx_list: List[int], dopant: str):
    s2 = struct.copy()
    for i in idx_list:
        s2[i] = Element(dopant)
    return s2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_cif", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="out_bisurf")
    ap.add_argument("--dopants", type=str, default="ALL")
    ap.add_argument("--dopant_to_host", type=str, default="1:3")
    ap.add_argument("--max_docs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    dopant_list = ALL_METALS if args.dopants.upper()=="ALL" else args.dopants.split(",")
    d,h = [int(x) for x in args.dopant_to_host.split(":")]
    frac = d/(d+h)

    ref_vac, tol = parse_ref_vacuum_and_tol(args.ref_cif)
    print(f"[Info] tol={tol} Å, ref_vac={ref_vac} Å")

    rng = np.random.RandomState(args.seed)

    api_key = os.environ.get("MAPI_KEY")
    with MPRester(api_key) as mpr:
        docs = list(mpr.summary.search(elements=["O"], fields=["material_id","formula_pretty","structure"], chunk_size=200))
        docs = docs[:args.max_docs]

        for ddoc in docs:
            struct = ddoc.structure
            if not is_metal_oxide(struct):
                continue
            ok = find_supercell_for_O(struct, target_O=64)
            if ok is None: continue
            sup, (ax,ay,az) = ok
            sup2,_,_,new_O = remove_topmost_oxygen(sup)
            if new_O!=63: continue
            sup3 = add_symmetric_vacuum(sup2, total_vacuum_A=ref_vac)

            top_idx, bot_idx = get_surface_metal_indices_both(sup3, tol)
            for dop in dopant_list:
                picks = choose_sites(top_idx, frac, rng) + choose_sites(bot_idx, frac, rng)
                doped = substitute_species(sup3, picks, dop)
                fname = f"{ddoc.material_id}_O63_bisurf_{dop}.cif"
                doped.to(fmt="cif", filename=os.path.join(args.outdir,fname))
                print("Wrote", fname)

if __name__ == "__main__":
    main()
