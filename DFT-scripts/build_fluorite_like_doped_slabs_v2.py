#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Materials Project 获取“萤石型 MO2 (Fm-3m, #225)”的非铈金属氧化物，
在金属子晶格上按 1:3 (=25%) 全局替位掺杂，
切 slab (默认 111)，orient Otop，加真空，仅在 a,b 扩胞，
最后删除最顶层 1 个 O。
"""

import os, argparse, math
from typing import List, Tuple, Optional, Dict, Any, Set
import numpy as np
import pandas as pd
from pymatgen.core import Structure, Element, Lattice, Composition
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester

ALLOWED_METALS: Set[str] = set([
    "Li","Na","Mg","K","Ca","Ba",
    "V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Zr","Mo","Ru","Rh","Pd","Ag","Cd",
    "Hf","Pt","Au","Hg",
    "Al","Pb",
    "Ce","Gd"
])

def is_fluorite_MO2(struct: Structure, allow_ce: bool=False) -> bool:
    try:
        sga = SpacegroupAnalyzer(struct, symprec=1e-3, angle_tolerance=5)
        if sga.get_space_group_number() != 225:
            return False
    except Exception:
        return False
    rf = Composition(struct.composition.reduced_formula)
    elsyms = sorted([el.symbol for el in rf.elements])
    if "O" not in elsyms or len(elsyms) != 2:
        return False
    m = [el for el in rf.elements if el.symbol != "O"][0]
    if abs(rf.get_el_amt_dict().get("O",0) / rf.get_el_amt_dict().get(m.symbol,1) - 2.0) > 1e-6:
        return False
    if (not allow_ce) and m.symbol=="Ce":
        return False
    if m.symbol not in ALLOWED_METALS:
        return False
    return True

def _unwrap_frac_z(z: np.ndarray) -> np.ndarray:
    if len(z)==0: return z
    zs=np.sort(z); gaps=np.diff(np.r_[zs, zs[0]+1.0])
    k=int(np.argmax(gaps)); base=zs[(k+1)%len(zs)]
    zu=z-base; zu[zu<0]+=1.0
    return zu

def orient_O_on_top(struct: Structure, probe_frac=0.10) -> Structure:
    s=struct.copy()
    f=np.array([site.frac_coords for site in s]); zu=_unwrap_frac_z(f[:,2])
    N=len(s); cut=max(1,int(np.ceil(N*probe_frac)))
    idx=np.argsort(zu); top=idx[-cut:]; bot=idx[:cut]
    nO_top=sum(1 for i in top if s[i].specie.symbol=="O")
    nO_bot=sum(1 for i in bot if s[i].specie.symbol=="O")
    if nO_top < nO_bot:
        f[:,2]=(1.0-zu)%1.0
        return Structure(s.lattice, [site.specie for site in s], f,
                         coords_are_cartesian=False, to_unit_cell=True)
    return s

def add_vacuum_asymmetric(struct: Structure, vac_top: float, vac_bot: float) -> Structure:
    s=struct.copy(); lat0=s.lattice; Z0=lat0.c
    f=np.array([site.frac_coords for site in s]); zu=_unwrap_frac_z(f[:,2])
    zmin,zmax=float(np.min(zu)), float(np.max(zu))
    span=max(1e-12, zmax-zmin); slab_A=span*Z0
    new_c=slab_A+vac_top+vac_bot; scale=new_c/Z0
    new_lat=Lattice([lat0.matrix[0], lat0.matrix[1], lat0.matrix[2]*scale])
    new_span=slab_A/new_c; new_base=vac_bot/new_c
    f[:,2]=(new_base + (zu-zmin)*(new_span/span))%1.0
    return Structure(new_lat, [site.specie for site in s], f,
                     coords_are_cartesian=False, to_unit_cell=True)

def remove_topmost_O(struct: Structure) -> Tuple[Structure,int,float]:
    s=struct.copy()
    fz=_unwrap_frac_z(np.array([site.frac_coords[2] for site in s]))
    o_ids=[i for i,site in enumerate(s.sites) if site.specie.symbol=="O"]
    if not o_ids: raise RuntimeError("No O found.")
    top=max(o_ids, key=lambda i: fz[i]); zval=float(fz[top])
    s.remove_sites([top]); return s, top, zval

def minimal_xy_supercell(struct: Structure, a_min: float, b_min: float) -> Tuple[int,int]:
    lat=struct.lattice
    ia=max(1, math.ceil(a_min/lat.a))
    ib=max(1, math.ceil(b_min/lat.b))
    return ia, ib

def make_supercell_xy(struct: Structure, ia:int, ib:int) -> Structure:
    s=struct.copy(); s.make_supercell([ia,ib,1]); return s

def pick_uc_metal_sites_strict(struct: Structure, dopant: str, frac: float,
                               rng: np.random.RandomState) -> List[int]:
    metal_ids=[i for i,s in enumerate(struct.sites)
               if isinstance(s.specie, Element) and s.specie.is_metal and s.specie.symbol!="O"]
    M_tot=len(metal_ids)
    if M_tot==0: return []
    dop_ids=[i for i in metal_ids if struct[i].specie.symbol==dopant]
    M_d0=len(dop_ids)
    K_target=int(round(frac*M_tot))
    k_needed=K_target-M_d0
    if k_needed<=0: return []
    cand=[i for i in metal_ids if struct[i].specie.symbol!=dopant]
    if len(cand)<k_needed: return []
    xy=np.array([struct[i].frac_coords[:2] for i in cand])
    if k_needed<=1 or len(cand)<=3:
        return sorted(rng.choice(cand, size=k_needed, replace=False).tolist())
    center=np.mean(xy,axis=0); d2=np.sum((xy-center)**2,axis=1)
    seed=int(np.argmax(d2)); chosen=[seed]
    dist=np.linalg.norm(xy-xy[seed],axis=1)
    while len(chosen)<k_needed:
        nxt=int(np.argmax(dist))
        if nxt in chosen:
            remain=[i for i in range(len(cand)) if i not in chosen]
            if not remain: break
            nxt=int(rng.choice(remain))
        chosen.append(nxt)
        dist=np.minimum(dist,np.linalg.norm(xy-xy[nxt],axis=1))
    return sorted(cand[i] for i in chosen)

def substitute(struct: Structure, idx: List[int], dopant: str) -> Structure:
    s2=struct.copy()
    for i in idx:
        if s2[i].specie.symbol!=dopant:
            s2[i]=Element(dopant)
    return s2

def make_slab_from_bulk(bulk: Structure,
                        hkl: Tuple[int,int,int],
                        min_slab: float,
                        min_vac: float,
                        prefer_O_top=True) -> Optional[Structure]:
    gen=SlabGenerator(
        initial_structure=bulk,
        miller_index=hkl,
        min_slab_size=min_slab,
        min_vacuum_size=min_vac,
        center_slab=False,
        in_unit_planes=True,
        primitive=True,
        reorient_lattice=True,
    )
    slabs=gen.get_slabs(ftol=0.1)
    if not slabs: return None
    if not prefer_O_top: return slabs[0]
    # 选 O 顶的 slab
    def score(slab):
        f=np.array([site.frac_coords for site in slab])
        zu=_unwrap_frac_z(f[:,2])
        idx=np.argsort(zu); cut=max(1,int(np.ceil(len(slab)*0.1)))
        top=idx[-cut:]
        ofrac=sum(1 for i in top if slab[i].specie.symbol=="O")/cut
        return -ofrac
    slabs.sort(key=score)
    return slabs[0]

def parse_millers(millers_str:str) -> List[Tuple[int,int,int]]:
    items=[]
    for blk in millers_str.replace("(","").replace(")","").split(";"):
        blk=blk.strip()
        if not blk: continue
        h,k,l=[int(x) for x in blk.split(",")]
        items.append((h,k,l))
    return items

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outdir",type=str,default="out_fluorite_like_doped_slabs")
    ap.add_argument("--ids",type=str,default="")
    ap.add_argument("--mapi_key",type=str,default=None)
    ap.add_argument("--dopants",type=str,default="Zr,Hf,Ni,Cu,Fe,Co,Ag,Pt,Au")
    ap.add_argument("--dopant_to_host",type=str,default="1:3")
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--millers",type=str,default="1,1,1")
    ap.add_argument("--min_slab",type=float,default=12.0)
    ap.add_argument("--min_vac",type=float,default=12.0)
    ap.add_argument("--prefer_O_top",action="store_true")
    ap.add_argument("--vac_top",type=float,default=20.0)
    ap.add_argument("--vac_bot",type=float,default=3.0)
    ap.add_argument("--rm_top_O",action="store_true")
    ap.add_argument("--a_min",type=float,default=10.0)
    ap.add_argument("--b_min",type=float,default=10.0)
    ap.add_argument("--max_docs",type=int,default=50)
    args=ap.parse_args()

    os.makedirs(args.outdir,exist_ok=True)
    rng=np.random.RandomState(args.seed)

    d,h=[int(x) for x in args.dopant_to_host.split(":")]
    frac=d/(d+h)

    dopant_list=[x.strip() for x in args.dopants.split(",") if x.strip()]

    # 从 MP 读取
    api_key=args.mapi_key or os.environ.get("MAPI_KEY")
    if not api_key: raise RuntimeError("Missing MAPI_KEY")
    entries=[]
    with MPRester(api_key) as mpr:
        docs=list(mpr.summary.search(fields=["material_id","structure"],chunk_size=200))
    for ddoc in docs:
        s=ddoc.structure
        if is_fluorite_MO2(s, allow_ce=False):
            entries.append((ddoc.material_id,s))
            if len(entries)>=args.max_docs: break

    millers=parse_millers(args.millers)
    manifest=[]
    for bid, bulk in entries:
        for dop in dopant_list:
            picks=pick_uc_metal_sites_strict(bulk, dopant=dop, frac=frac, rng=rng)
            if not picks: continue
            bulk_doped=substitute(bulk,picks,dop)
            for hkl in millers:
                slab=make_slab_from_bulk(bulk_doped,hkl,min_slab=args.min_slab,min_vac=args.min_vac,
                                         prefer_O_top=args.prefer_O_top)
                if slab is None: continue
                slab=orient_O_on_top(slab,probe_frac=0.10)
                slab_v=add_vacuum_asymmetric(slab,vac_top=args.vac_top,vac_bot=args.vac_bot)
                ia,ib=minimal_xy_supercell(slab_v,a_min=args.a_min,b_min=args.b_min)
                slab_xy=make_supercell_xy(slab_v,ia,ib)
                deleted={}
                if args.rm_top_O:
                    try:
                        slab_xy,del_idx,del_fz=remove_topmost_O(slab_xy)
                        deleted={"deleted_O_index":del_idx,"deleted_O_fz":del_fz}
                    except Exception: pass
                tag=f"{bid}_fluorite_hkl{hkl[0]}{hkl[1]}{hkl[2]}_{dop}_a{ia}x_b{ib}x"
                if args.rm_top_O: tag+="_rmTopO"
                outpath=os.path.join(args.outdir,tag+".cif")
                slab_xy.to(fmt="cif",filename=outpath)
                rec={"id":bid,"dopant":dop,"outfile":outpath,"hkl":hkl,
                     "ia":ia,"ib":ib,"vac_top":args.vac_top,"vac_bot":args.vac_bot}
                rec.update(deleted)
                manifest.append(rec)
                print("[WROTE]",outpath)
    if manifest:
        pd.DataFrame(manifest).to_csv(os.path.join(args.outdir,"manifest.csv"),index=False)
        print(f"[OK] wrote {len(manifest)} structures")

if __name__=="__main__":
    main()
