#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Host–O(二元) → 元胞金属位点全局 1:3 掺杂(25%) → 切 slab(整层)
→ 尽量 O 在上 → 加真空(上/下) → 仅 a,b 扩胞 → 删除最顶层 1 个 O
→ 【可选】最后将“晶胞 ab 面”刚体旋到全局 XY 平面 (不要求 c//z)

宿主: Ca, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Mo, Ru, Rh, Pd, Ag, Cd, Pt, Au
掺杂: Mg, Ca, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Mo, Ru, Rh, Pd, Ag, Cd, Pt, Au
比例: 金属子晶格严格 1:3 (=25%)

依赖: pymatgen, mp-api, numpy, pandas
"""

import os, argparse, math, json, gzip
from typing import List, Tuple, Optional, Dict, Any, Set
import numpy as np
import pandas as pd

from pymatgen.core import Structure, Element, Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester

# ------------------ 白名单 ------------------ #
HOST_METALS: Set[str] = set([
    "Ca","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Zr","Mo","Ru","Rh","Pd","Ag","Cd","Pt","Au"
])
DOPANT_METALS: Set[str] = set([
    "Mg","Ca","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Zr","Mo","Ru","Rh","Pd","Ag","Cd","Pt","Au"
])

# ------------------ 旋转/拟合工具 ------------------ #
def _rotation_from_u_to_v(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u = u / np.linalg.norm(u); v = v / np.linalg.norm(v)
    c = np.clip(np.dot(u, v), -1.0, 1.0)
    if np.isclose(c, 1.0): return np.eye(3)
    if np.isclose(c, -1.0):
        axis = np.cross(u, np.array([1.0,0.0,0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(u, np.array([0.0,1.0,0.0]))
        axis /= np.linalg.norm(axis)
        K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        return -np.eye(3) + 2*np.outer(axis, axis)
    axis = np.cross(u, v); s = np.linalg.norm(axis); axis /= s
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s**2))

def _Rz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def align_ab_plane_to_xy_lattice(struct: Structure, align_a_to_x: bool = False) -> Structure:
    """
    刚体旋转(晶格+坐标)：把 (a×b) 的法向旋到全局 +z，使 ab 面 || XY。
    可选：再绕 z 轴旋转，使 a 的投影对齐 +x（美观/统一）。
    """
    s = struct.copy()
    M = s.lattice.matrix           # 行向量依次是 a,b,c
    a_vec, b_vec, c_vec = M[0], M[1], M[2]
    n = np.cross(a_vec, b_vec)
    if np.linalg.norm(n) < 1e-10:
        # a 与 b 共线的退化情况，跳过
        return s
    # 旋到 +z
    ez = np.array([0.,0.,1.])
    R1 = _rotation_from_u_to_v(n, ez)
    M1 = (R1 @ M.T).T
    cart = np.array([site.coords for site in s])
    C1 = (R1 @ cart.T).T

    if align_a_to_x:
        a1 = M1[0]
        # 只用其在 XY 的投影方向
        ax, ay = a1[0], a1[1]
        if abs(ax)+abs(ay) > 1e-12:
            phi = math.atan2(ay, ax)  # 需要再旋转 -phi 到 +x
            R2 = _Rz(-phi)
            M2 = (R2 @ M1.T).T
            C2 = (R2 @ C1.T).T
            M1, C1 = M2, C2

    # 数值清理：把极小的 z 分量抹零（只对 a,b 的 z）
    M1[0,2] = 0.0; M1[1,2] = 0.0

    lat_new = Lattice(M1)
    return Structure(lat_new, [site.specie for site in s], C1,
                     coords_are_cartesian=True, to_unit_cell=False)

# ------------------ 其余结构工具 ------------------ #
def _unwrap_frac_z(z: np.ndarray) -> np.ndarray:
    if len(z)==0: return z
    zs = np.sort(z); gaps = np.diff(np.r_[zs, zs[0]+1.0])
    k = int(np.argmax(gaps)); base = zs[(k+1)%len(zs)]
    zu = z - base; zu[zu<0] += 1.0
    return zu

def orient_O_on_top(struct: Structure, probe_frac: float = 0.10) -> Structure:
    s = struct.copy()
    f = np.array([site.frac_coords for site in s]); zu = _unwrap_frac_z(f[:,2])
    N = len(s); cut = max(1, int(np.ceil(N * probe_frac)))
    idx = np.argsort(zu); top = idx[-cut:]; bot = idx[:cut]
    nO_top = sum(1 for i in top if s[i].specie.symbol=="O")
    nO_bot = sum(1 for i in bot if s[i].specie.symbol=="O")
    if nO_top < nO_bot:
        f[:,2] = (1.0 - zu) % 1.0
        return Structure(s.lattice, [site.specie for site in s], f,
                         coords_are_cartesian=False, to_unit_cell=True)
    return s

def add_vacuum_asymmetric(struct: Structure, vac_top: float, vac_bot: float) -> Structure:
    s = struct.copy(); lat0 = s.lattice; Z0 = lat0.c
    f = np.array([site.frac_coords for site in s]); zu = _unwrap_frac_z(f[:,2])
    zmin, zmax = float(np.min(zu)), float(np.max(zu))
    span = max(1e-12, zmax - zmin); slab_A = span * Z0
    new_c = slab_A + float(vac_top) + float(vac_bot); scale = new_c / Z0
    new_lat = Lattice([lat0.matrix[0], lat0.matrix[1], lat0.matrix[2] * scale])
    new_span = slab_A / new_c; new_base = vac_bot / new_c
    f[:,2] = (new_base + (zu - zmin) * (new_span / span)) % 1.0
    return Structure(new_lat, [site.specie for site in s], f,
                     coords_are_cartesian=False, to_unit_cell=True)

def remove_topmost_O(struct: Structure) -> Tuple[Structure,int,float]:
    s = struct.copy()
    fz = _unwrap_frac_z(np.array([site.frac_coords[2] for site in s]))
    o_ids = [i for i,site in enumerate(s.sites) if site.specie.symbol=="O"]
    if not o_ids: raise RuntimeError("No O found.")
    top = max(o_ids, key=lambda i: fz[i]); zval = float(fz[top])
    s.remove_sites([top]); return s, top, zval

def minimal_xy_supercell(struct: Structure, a_min: float, b_min: float) -> Tuple[int,int]:
    lat = struct.lattice
    ia = max(1, math.ceil(a_min / lat.a))
    ib = max(1, math.ceil(b_min / lat.b))
    return ia, ib

def make_supercell_xy(struct: Structure, ia: int, ib: int) -> Structure:
    s = struct.copy(); s.make_supercell([ia, ib, 1]); return s

def is_binary_host_oxide(struct: Structure) -> bool:
    elems = sorted(set(sp.symbol for sp in struct.composition.elements))
    if len(elems) != 2 or "O" not in elems: return False
    host = [e for e in elems if e != "O"][0]
    return host in HOST_METALS

def reduce_cell(bulk: Structure, mode: str) -> Structure:
    mode = (mode or "none").lower()
    try:
        if mode=="primitive": return bulk.get_primitive_structure()
        if mode=="symm_primitive":
            sga=SpacegroupAnalyzer(bulk, symprec=1e-3, angle_tolerance=5)
            return sga.get_primitive_standard_structure()
        if mode=="conventional":
            sga=SpacegroupAnalyzer(bulk, symprec=1e-3, angle_tolerance=5)
            return sga.get_conventional_standard_structure()
    except Exception:
        pass
    return bulk

# ------------------ 掺杂（严格 1:3） ------------------ #
def pick_uc_metal_sites_strict(struct: Structure, dopant: str, frac: float,
                               rng: np.random.RandomState) -> List[int]:
    metal_ids = [i for i,s in enumerate(struct.sites)
                 if isinstance(s.specie, Element) and s.specie.is_metal and s.specie.symbol!="O"]
    M_tot = len(metal_ids)
    if M_tot == 0: return []
    dop_exist = [i for i in metal_ids if struct[i].specie.symbol==dopant]
    K_target = int(round(frac * M_tot))
    k_needed = K_target - len(dop_exist)
    if k_needed <= 0: return []
    cand = [i for i in metal_ids if struct[i].specie.symbol != dopant]
    if len(cand) < k_needed: return []
    xy = np.array([struct[i].frac_coords[:2] for i in cand])
    if k_needed <= 1 or len(cand) <= 3:
        return sorted(rng.choice(cand, size=k_needed, replace=False).tolist())
    center = np.mean(xy, axis=0); d2 = np.sum((xy-center)**2, axis=1)
    seed = int(np.argmax(d2)); chosen=[seed]
    dist = np.linalg.norm(xy-xy[seed], axis=1)
    while len(chosen) < k_needed:
        nxt = int(np.argmax(dist))
        if nxt in chosen:
            remain=[i for i in range(len(cand)) if i not in chosen]
            if not remain: break
            nxt = int(rng.choice(remain))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(xy-xy[nxt], axis=1))
    return sorted(cand[i] for i in chosen)

def substitute(struct: Structure, idx: List[int], dopant: str) -> Structure:
    s2 = struct.copy()
    for i in idx:
        if s2[i].specie.symbol != dopant:
            s2[i] = Element(dopant)
    return s2

# ------------------ 切 slab ------------------ #
def parse_millers(millers_str: str) -> List[Tuple[int,int,int]]:
    items=[]
    for blk in millers_str.replace("(","").replace(")","").split(";"):
        blk=blk.strip()
        if not blk: continue
        h,k,l=[int(x) for x in blk.split(",")]
        items.append((h,k,l))
    return items

def make_slab_from_bulk(bulk: Structure, hkl: Tuple[int,int,int],
                        min_slab: float, prefer_O_top: bool=True) -> Optional[Structure]:
    gen = SlabGenerator(
        initial_structure=bulk,
        miller_index=hkl,
        min_slab_size=min_slab,
        min_vacuum_size=0.1,
        center_slab=False,
        in_unit_planes=True,
        primitive=True,
        reorient_lattice=True,
    )
    slabs = gen.get_slabs(ftol=0.1)
    if not slabs: return None
    if not prefer_O_top: return slabs[0]
    def score(slab):
        f=np.array([site.frac_coords for site in slab]); zu=_unwrap_frac_z(f[:,2])
        idx=np.argsort(zu); cut=max(1,int(np.ceil(len(slab)*0.1)))
        top=idx[-cut:]; ofrac=sum(1 for i in top if slab[i].specie.symbol=="O")/cut
        return -ofrac
    slabs.sort(key=score)
    return slabs[0]

def align_ab_plane_to_xy_lattice(struct, align_a_to_x=False):
    import numpy as np, math
    from pymatgen.core import Lattice, Structure
    M = struct.lattice.matrix  # 行向量 a,b,c
    a_vec, b_vec = M[0], M[1]
    n = np.cross(a_vec, b_vec)
    n = n / np.linalg.norm(n)
    ez = np.array([0.,0.,1.])

    # 把 n 旋到 +z 的旋转矩阵 R
    def rot_u_to_v(u, v):
        u = u/np.linalg.norm(u); v = v/np.linalg.norm(v)
        c = np.clip(np.dot(u, v), -1.0, 1.0)
        if abs(c-1) < 1e-12: return np.eye(3)
        if abs(c+1) < 1e-12:
            axis = np.cross(u, np.array([1.,0.,0.]))
            if np.linalg.norm(axis) < 1e-8:
                axis = np.cross(u, np.array([0.,1.,0.]))
            axis /= np.linalg.norm(axis)
            return -np.eye(3) + 2*np.outer(axis, axis)
        axis = np.cross(u, v); s = np.linalg.norm(axis); axis /= s
        K = np.array([[0,-axis[2],axis[1]],
                      [axis[2],0,-axis[0]],
                      [-axis[1],axis[0],0]])
        return np.eye(3) + K + K @ K * ((1 - c) / (s**2))

    R1 = rot_u_to_v(n, ez)

    M1 = (R1 @ M.T).T
    C  = np.array([site.coords for site in struct])
    C1 = (R1 @ C.T).T

    if align_a_to_x:
        a1 = M1[0]
        ax, ay = a1[0], a1[1]
        if abs(ax)+abs(ay) > 1e-12:
            phi = math.atan2(ay, ax)
            Rz = np.array([[ math.cos(-phi), -math.sin(-phi), 0],
                           [ math.sin(-phi),  math.cos(-phi), 0],
                           [ 0,               0,              1]])
            M1 = (Rz @ M1.T).T
            C1 = (Rz @ C1.T).T

    # 数值清理：让 a、b 的 z 分量严格为 0（避免 1e-16 这种尾数）
    M1[0,2] = 0.0; M1[1,2] = 0.0

    lat_new = Lattice(M1)
    return Structure(lat_new, [site.specie for site in struct], C1,
                     coords_are_cartesian=True, to_unit_cell=False)


# ------------------ MP 缓存拉取 ------------------ #
def load_or_fetch_host_binaries(api_key: str, hosts: List[str],
                                per_host_max: int, cache_dir: str,
                                total_max: int) -> List[Dict[str, Any]]:
    os.makedirs(cache_dir, exist_ok=True)
    cif_dir = os.path.join(cache_dir, "cache_cifs")
    os.makedirs(cif_dir, exist_ok=True)
    out: List[Dict[str, Any]] = []
    with MPRester(api_key) as mpr:
        for host in hosts:
            cache_file = os.path.join(cache_dir, f"summary_{host}-O.json.gz")
            entries = None
            if os.path.exists(cache_file):
                print(f"[CACHE] 使用 {cache_file}")
                with gzip.open(cache_file, "rt", encoding="utf-8") as f:
                    entries = json.load(f)
            if entries is None:
                print(f"[FETCH] 从 MP 拉取 {host}-O …")
                docs = list(mpr.summary.search(
                    chemsys=[f"{host}-O"],
                    fields=["material_id","structure","formula_pretty"],
                    chunk_size=200
                ))
                entries=[]
                for ddoc in docs[:per_host_max]:
                    mid = ddoc.material_id
                    cif_path = os.path.join(cif_dir, f"{mid}.cif")
                    if not os.path.exists(cif_path):
                        try:
                            ddoc.structure.to(fmt="cif", filename=cif_path)
                        except Exception:
                            continue
                    entries.append({"material_id": mid,
                                    "cif": cif_path,
                                    "formula": getattr(ddoc,"formula_pretty",""),
                                    "host": host})
                with gzip.open(cache_file, "wt", encoding="utf-8") as f:
                    json.dump(entries, f)
            out.extend(entries)
            if len(out) >= total_max: break
    return out[:total_max]

# ------------------ 主程序 ------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="out_host_doped_slabs")

    # MP & 缓存
    ap.add_argument("--mapi_key", type=str, default=None)
    ap.add_argument("--cache_dir", type=str, default="mp_cache")
    ap.add_argument("--per_host_max", type=int, default=10)
    ap.add_argument("--total_max", type=int, default=60)

    # 掺杂
    ap.add_argument("--dopants", type=str,
        default="Mg,Ca,Cr,Mn,Fe,Co,Ni,Cu,Zn,Zr,Mo,Ru,Rh,Pd,Ag,Cd,Pt,Au")
    ap.add_argument("--dopant_to_host", type=str, default="1:3")
    ap.add_argument("--seed", type=int, default=0)

    # 减胞/切片
    ap.add_argument("--reduce", type=str, default="symm_primitive",
        choices=["none","primitive","symm_primitive","conventional"])
    ap.add_argument("--millers", type=str, default="1,1,0;1,0,1;0,1,1")
    ap.add_argument("--min_slab", type=float, default=12.0)
    ap.add_argument("--prefer_O_top", action="store_true")

    # 真空/扩胞/删O
    ap.add_argument("--vac_top", type=float, default=20.0)
    ap.add_argument("--vac_bot", type=float, default=3.0)
    ap.add_argument("--a_min", type=float, default=10.0)
    ap.add_argument("--b_min", type=float, default=10.0)
    ap.add_argument("--rm_top_O", action="store_true")

    # 最终对齐：ab 面 || XY（不要求 c//z）
    ap.add_argument("--final_align_ab", action="store_true",
                    help="最后把晶胞 ab 面刚体旋到全局 XY 平面")
    ap.add_argument("--align_a_to_x", action="store_true",
                    help="对齐后再把 a 的投影沿 +x（便于统一取向）")

    # 限量（可选）
    ap.add_argument("--sample_combos", type=int, default=0)
    ap.add_argument("--limit_outputs", type=int, default=0)
    ap.add_argument("--final_atoms_min", type=int, default=0)
    ap.add_argument("--final_atoms_max", type=int, default=0)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # 掺杂列表
    req_dops=[x.strip() for x in args.dopants.split(",") if x.strip()]
    dopant_list = sorted(list(DOPANT_METALS.intersection(req_dops)))
    if not dopant_list:
        raise RuntimeError("掺杂元素列表为空或不在白名单。")

    d, h = [int(x) for x in args.dopant_to_host.split(":")]
    assert d>0 and h>0
    frac = d/(d+h)

    api_key = args.mapi_key or os.environ.get("MAPI_KEY")
    if not api_key: raise RuntimeError("Missing MAPI_KEY")
    host_list = sorted(list(HOST_METALS))
    summary_entries = load_or_fetch_host_binaries(
        api_key, host_list, args.per_host_max, args.cache_dir, args.total_max
    )
    if not summary_entries:
        print("[WARN] 无候选，退出。"); return

    millers = parse_millers(args.millers)

    # 组合级抽样
    combos=[]
    for ent in summary_entries:
        for dop in dopant_list:
            for hkl in millers:
                combos.append((ent, dop, hkl))
    rng.shuffle(combos)
    if args.sample_combos and args.sample_combos>0:
        combos = combos[:args.sample_combos]

    manifest: List[Dict[str,Any]] = []
    ok_written = 0

    for (ent, dop, hkl) in combos:
        if args.limit_outputs and ok_written >= args.limit_outputs:
            print(f"[STOP] 达到上限 limit_outputs={args.limit_outputs}"); break

        mid = ent["material_id"]; cif_path = ent["cif"]; host = ent["host"]
        try:
            bulk0 = Structure.from_file(cif_path)
        except Exception as e:
            print(f"[SKIP] 读取失败 {mid}: {e}"); continue

        if not is_binary_host_oxide(bulk0):
            manifest.append({"id": mid, "status": "skip_not_binary_host_oxide"}); continue

        bulk = reduce_cell(bulk0, args.reduce)

        # 1) 元胞 1:3 掺杂
        picks = pick_uc_metal_sites_strict(bulk, dopant=dop, frac=frac, rng=rng)
        if not picks:
            manifest.append({"id": mid, "host": host, "dopant": dop,
                             "status": "skip_ratio_unreachable"}); continue
        bulk_doped = substitute(bulk, picks, dop)

        # 2) 切 slab（整层，不截原子）
        slab = make_slab_from_bulk(bulk_doped, hkl, min_slab=args.min_slab,
                                   prefer_O_top=args.prefer_O_top)
        if slab is None:
            manifest.append({"id": mid, "host": host, "dopant": dop,
                             "hkl": hkl, "status": "no_slab"}); continue

        # 3) 尽量 O 顶
        slab = orient_O_on_top(slab, probe_frac=0.10)

        # 4) 加真空
        slab_v = add_vacuum_asymmetric(slab, vac_top=args.vac_top, vac_bot=args.vac_bot)

        # 5) 仅 a,b 扩胞
        ia, ib = minimal_xy_supercell(slab_v, a_min=args.a_min, b_min=args.b_min)
        slab_xy = make_supercell_xy(slab_v, ia, ib)

        # 6) 删除最顶层 1 个 O
        deleted={}
        if args.rm_top_O:
            try:
                slab_xy, del_idx, del_fz = remove_topmost_O(slab_xy)
                deleted={"deleted_O_index": del_idx, "deleted_O_fz": del_fz}
            except Exception as e:
                manifest.append({"id": mid, "host": host, "dopant": dop, "hkl": hkl,
                                 "status": "skip_no_O_for_rm", "reason": str(e)})
                continue

        # 7) 【新】最后把 ab 面对齐到 XY（刚体，满足你的诉求）
        if args.final_align_ab:
            slab_xy = align_ab_plane_to_xy_lattice(slab_xy, align_a_to_x=True)

        # 原子数筛选（可选）
        if args.final_atoms_min or args.final_atoms_max:
            nat = len(slab_xy)
            if (args.final_atoms_min and nat < args.final_atoms_min) or \
               (args.final_atoms_max and nat > args.final_atoms_max):
                manifest.append({"id": mid, "host": host, "dopant": dop, "hkl": hkl,
                                 "status": "skip_atoms_out_of_range", "nat": nat})
                continue

        # 输出
        tag = (f"{mid}_host{host}_hkl{hkl[0]}{hkl[1]}{hkl[2]}_ucGLOBALdope_{dop}"
               f"_vacTop{int(args.vac_top)}_vacBot{int(args.vac_bot)}"
               f"_a{ia}x_b{ib}x"
               f"{'_rmTopO' if args.rm_top_O else ''}"
               f"{'_ab2XY' if args.final_align_ab else ''}"
               f"{'_a2x' if args.align_a_to_x else ''}")
        outpath = os.path.join(args.outdir, tag + ".cif")
        slab_xy.to(fmt="cif", filename=outpath)

        rec = {"id": mid, "status": "ok", "outfile": outpath,
               "host": host, "dopant": dop, "hkl": hkl,
               "ia": ia, "ib": ib, "vac_top": args.vac_top, "vac_bot": args.vac_bot,
               "picked_uc": len(picks), "ratio_global": f"{d}:{h}",
               "final_align_ab": args.final_align_ab, "align_a_to_x": args.align_a_to_x}
        rec.update(deleted)
        manifest.append(rec)
        ok_written += 1
        print("[WROTE]", outpath)

    if manifest:
        pd.DataFrame(manifest).to_csv(os.path.join(args.outdir,"manifest.csv"), index=False)
        print(f"[OK] wrote {ok_written} structures; total {len(manifest)}")
    else:
        print("[WARN] no outputs")

if __name__ == "__main__":
    main()
