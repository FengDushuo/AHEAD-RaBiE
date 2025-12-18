#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Host–O → 元胞金属子晶格全局 1:3 掺杂(25%) → 切 slab(整层; 不重定向晶格)
→ 尽量 O 在上 → 【刚体】ab→XY（改变笛卡尔坐标系；c 不强制 || z）
→ 保持 c_xy 倾斜，仅沿全局 z 加真空 → 仅 a,b 扩胞 → 可选删除最顶层 1 个 O
→ 可选把底面平移到 z=0（仅平移）

宿主白名单: Ca, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Mo, Ru, Rh, Pd, Ag, Cd, Pt, Au
掺杂白名单: Mg, Ca, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Mo, Ru, Rh, Pd, Ag, Cd, Pt, Au

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

# ------------------ 小工具 ------------------ #
def _rotation_from_u_to_v(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u = u/np.linalg.norm(u); v = v/np.linalg.norm(v)
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    if abs(c-1.0) < 1e-12:
        return np.eye(3)
    if abs(c+1.0) < 1e-12:
        axis = np.cross(u, np.array([1.,0.,0.]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(u, np.array([0.,1.,0.]))
        axis /= np.linalg.norm(axis)
        K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        return -np.eye(3) + 2*np.outer(axis, axis)
    axis = np.cross(u, v); s = np.linalg.norm(axis); axis /= s
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s**2))

def _Rz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def _unwrap_frac_z(z: np.ndarray) -> np.ndarray:
    if len(z)==0: return z
    zs = np.sort(z); gaps = np.diff(np.r_[zs, zs[0]+1.0])
    k = int(np.argmax(gaps)); base = zs[(k+1)%len(zs)]
    zu = z - base; zu[zu<0] += 1.0
    return zu

# ------------------ 关键：刚体 ab→XY（改变笛卡尔坐标系；c 不强制 || z） ------------------ #
def align_ab_plane_to_xy_lattice(struct: Structure, align_a_to_x: bool=False) -> Structure:
    """旋转晶格+坐标，使 (a×b) 的法向 → +z；可选再绕 z 使 a 的投影沿 +x。"""
    s = struct.copy()
    M = s.lattice.matrix
    a_vec, b_vec = M[0], M[1]
    n = np.cross(a_vec, b_vec)
    if np.linalg.norm(n) < 1e-10:
        return s  # 退化
    n = n/np.linalg.norm(n)
    ez = np.array([0.,0.,1.])
    R1 = _rotation_from_u_to_v(n, ez)
    M1 = (R1 @ M.T).T
    C1 = (R1 @ np.array([site.coords for site in s]).T).T

    if align_a_to_x:
        a1 = M1[0]; ax, ay = a1[0], a1[1]
        if abs(ax)+abs(ay) > 1e-12:
            phi = math.atan2(ay, ax)
            R2 = _Rz(-phi)
            M1 = (R2 @ M1.T).T
            C1 = (R2 @ C1.T).T

    # 数值清理：把 a、b 的 z 分量清零（避免 1e-16 尾数）
    M1[0,2] = 0.0; M1[1,2] = 0.0

    return Structure(Lattice(M1), [site.specie for site in s], C1,
                     coords_are_cartesian=True, to_unit_cell=False)

def nudge_c_xy_if_collinear(struct: Structure, eps: float = 1e-3) -> Structure:
    """如 c 与 z 几乎共线，则给 c 注入极小的 a/b 分量，避免 c ∥ z。"""
    s = struct.copy()
    M = s.lattice.matrix.copy()
    a, b, c = M[0], M[1], M[2]
    if abs(c[0]) + abs(c[1]) < eps:
        c = c + eps * (a/np.linalg.norm(a) + b/np.linalg.norm(b))
        M[2] = c
        return Structure(Lattice(M), [site.specie for site in s],
                         [site.coords for site in s],
                         coords_are_cartesian=True, to_unit_cell=False)
    return s

# ------------------ 真空：保持 c_xy，只调 c_z，并沿全局 z 平移坐标 ------------------ #
def add_vacuum_z_keep_cxy(struct: Structure, vac_top: float, vac_bot: float,
                          set_bottom_z0: bool=False) -> Structure:
    """
    在全局 z 上加真空：保持 a,b 不变；仅把 c 的 z 分量改为 (slab_thick + vac_top + vac_bot)。
    保留 c 的 x/y 倾斜分量。将原子整体沿 z 平移，使底部留出 vac_bot；可选把底面放到 z=0。
    """
    s = struct.copy()
    L = s.lattice
    M = L.matrix.copy()
    a, b, c = M[0], M[1], M[2]

    cart = np.array([site.coords for site in s])
    z = cart[:, 2]
    zmin, zmax = float(np.min(z)), float(np.max(z))
    slab_thick = max(1e-8, zmax - zmin)

    Z_new = slab_thick + float(vac_top) + float(vac_bot)
    c_new = np.array([c[0], c[1], Z_new], dtype=float)
    M_new = np.vstack([a, b, c_new])
    L_new = Lattice(M_new)

    shift_z = (0.0 if set_bottom_z0 else vac_bot) - zmin
    cart_new = cart + np.array([0.0, 0.0, shift_z], dtype=float)

    return Structure(L_new, [site.specie for site in s], cart_new,
                     coords_are_cartesian=True, to_unit_cell=False)

# ------------------ 其他结构操作 ------------------ #
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

# ------------------ 切 slab（不重定向晶格） ------------------ #
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
        reorient_lattice=False,   # 关键：不要把 c 强行对齐法向
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

    # 对齐/底面平移
    ap.add_argument("--align_a_to_x", action="store_true",
                    help="ab→XY 后再绕 z 使 a 的投影沿 +x（可视化更一致）")
    ap.add_argument("--set_bottom_z0", action="store_true",
                    help="加真空时把底面放到 z=0（仅平移）")
    ap.add_argument("--nudge_c_xy", action="store_true",
                    help="若 c 几乎 ∥ z，则为 c 注入极小 xy 分量，避免与 z 重合")
    ap.add_argument("--nudge_eps", type=float, default=1e-3)

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

        # 2) 切 slab（不重定向晶格）
        slab = make_slab_from_bulk(bulk_doped, hkl, min_slab=args.min_slab,
                                   prefer_O_top=args.prefer_O_top)
        if slab is None:
            manifest.append({"id": mid, "host": host, "dopant": dop,
                             "hkl": hkl, "status": "no_slab"}); continue

        # 3) O 顶（必要时翻转分数 z）
        slab = orient_O_on_top(slab, probe_frac=0.10)

        # 4) 刚体对齐：ab→XY（改变笛卡尔坐标系；c 不强制 || z）
        slab = align_ab_plane_to_xy_lattice(slab, align_a_to_x=args.align_a_to_x)

        # 5) （可选）若 c 太接近 z，注入极小 c_xy，避免 c∥z
        if args.nudge_c_xy:
            slab = nudge_c_xy_if_collinear(slab, eps=args.nudge_eps)

        # 6) 沿全局 z 加真空（保持 c_xy 倾斜分量）
        slab = add_vacuum_z_keep_cxy(slab, vac_top=args.vac_top,
                                     vac_bot=args.vac_bot,
                                     set_bottom_z0=args.set_bottom_z0)

        # 7) 仅 a,b 扩胞
        ia, ib = minimal_xy_supercell(slab, a_min=args.a_min, b_min=args.b_min)
        slab = make_supercell_xy(slab, ia, ib)

        # 8) （可选）删除最顶层 1 个 O（此时 vacancy 在顶表面）
        deleted={}
        if args.rm_top_O:
            try:
                slab, del_idx, del_fz = remove_topmost_O(slab)
                deleted={"deleted_O_index": del_idx, "deleted_O_fz": del_fz}
            except Exception as e:
                manifest.append({"id": mid, "host": host, "dopant": dop, "hkl": hkl,
                                 "status": "skip_no_O_for_rm", "reason": str(e)})
                continue

        # 原子数筛选（可选）
        if args.final_atoms_min or args.final_atoms_max:
            nat = len(slab)
            if (args.final_atoms_min and nat < args.final_atoms_min) or \
               (args.final_atoms_max and nat > args.final_atoms_max):
                manifest.append({"id": mid, "host": host, "dopant": dop, "hkl": hkl,
                                 "status": "skip_atoms_out_of_range", "nat": nat})
                continue

        # 输出
        tag = (f"{mid}_host{host}_hkl{hkl[0]}{hkl[1]}{hkl[2]}_ucGLOBALdope_{dop}"
               f"_ab2XY_keepc"
               f"_vacTop{int(args.vac_top)}_vacBot{int(args.vac_bot)}"
               f"_a{ia}x_b{ib}x"
               f"{'_rmTopO' if args.rm_top_O else ''}"
               f"{'_a2x' if args.align_a_to_x else ''}"
               f"{'_z0' if args.set_bottom_z0 else ''}"
               f"{'_nudge' if args.nudge_c_xy else ''}")
        outpath = os.path.join(args.outdir, tag + ".cif")
        slab.to(fmt="cif", filename=outpath)

        rec = {"id": mid, "status": "ok", "outfile": outpath,
               "host": host, "dopant": dop, "hkl": hkl,
               "ia": ia, "ib": ib, "vac_top": args.vac_top, "vac_bot": args.vac_bot,
               "picked_uc": len(picks), "ratio_global": f"{int(frac*4)}:{int((1-frac)*4)}",
               "align": "ab2XY_keep_cxy", "rmTopO": bool(args.rm_top_O)}
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
