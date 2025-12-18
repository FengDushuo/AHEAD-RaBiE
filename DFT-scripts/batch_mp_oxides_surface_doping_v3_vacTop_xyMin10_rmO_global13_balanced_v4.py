#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按指定 mp-id：
  1) 从 MP 拉结构（可先瘦身：primitive / symm_primitive / conventional）
  2) 生成给定 (hkl) 截面 slab（少量 shift 快速扫描）
     - 可指定目标层数 --n_layers（按 z 聚类计层，容差 --layer_tol）
     - 优先满足“顶层 O 终止”（可用 --min_o_frac 控制顶部窗口含氧比例）
     - 若无完全匹配层数，退而求其次选“层数差最小 + 顶部 O 最多”的
  3) 删除最顶层 1 个 O（造氧空位）
  4) z 向非对称真空（--vac_top / --vac_bot）
  5) XY 受控扩胞（a,b 最小长度 + 倍数上限 + 目标原子数）
  6) 顶/底严格 1:3（25%）金属掺杂（白名单，FPS 均匀）
  7) 输出 .cif 与 manifest.tsv

依赖：pymatgen, mp-api, numpy
"""

import os, argparse, math, inspect, random
from typing import List, Tuple, Optional, Set, Dict, Any
import numpy as np

from mp_api.client import MPRester
from pymatgen.core import Structure, Element, Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import itertools
from pymatgen.core import PeriodicSite
import numpy as np
from pymatgen.core import Lattice, Structure
import numpy as np
from pymatgen.core import Lattice, Structure

import itertools
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pymatgen.core import Structure

import itertools, numpy as np
from typing import Optional, Tuple, Dict, Any
from pymatgen.core import Structure, Lattice

# ====== 公共小工具 ======
def _lat_metrics(a_vec, b_vec, c_vec):
    la, lb, lc = np.linalg.norm(a_vec), np.linalg.norm(b_vec), np.linalg.norm(c_vec)
    def ang(u, v):
        cu = np.clip(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)), -1.0, 1.0)
        return float(np.degrees(np.arccos(cu)))
    alpha = ang(b_vec, c_vec); beta = ang(a_vec, c_vec); gamma = ang(a_vec, b_vec)
    return la, lb, lc, alpha, beta, gamma

def _score_cubic(a,b,c,alpha,beta,gamma):
    L = (a+b+c)/3.0
    len_dev = (abs(a-L)+abs(b-L)+abs(c-L))/(3.0*L)
    ang_dev = (abs(alpha-90)+abs(beta-90)+abs(gamma-90))/270.0
    return len_dev + ang_dev

# ====== 方案A：整数超胞“近立方”（无应变） ======
def search_3d_near_cubic_int_supercell(
    slab: Structure, max_n: int = 3, tol_len: float = 0.02, tol_ang: float = 1.0,
    max_atoms_after: Optional[int] = None
) -> Tuple[Optional[Structure], Dict[str, Any]]:
    """搜索 3x3 整数矩阵 H（det>0），复制/变基后尽量立方；不改结构。"""
    best = None
    for H in itertools.product(range(-max_n, max_n+1), repeat=9):
        H3 = np.array(H, dtype=int).reshape(3,3)
        det3 = int(round(np.linalg.det(H3)))
        if det3 <= 0:
            continue
        atoms_after = len(slab)*det3
        if max_atoms_after and atoms_after > max_atoms_after:
            continue
        s2 = slab.copy(); s2.make_supercell(H3)
        a_vec,b_vec,c_vec = s2.lattice.matrix
        a,b,c,alpha,beta,gamma = _lat_metrics(a_vec,b_vec,c_vec)
        ok_len = (abs(a-b)/((a+b)/2) <= tol_len) and (abs(a-c)/((a+c)/2) <= tol_len)
        ok_ang = (abs(alpha-90)<=tol_ang and abs(beta-90)<=tol_ang and abs(gamma-90)<=tol_ang)
        sc = _score_cubic(a,b,c,alpha,beta,gamma)
        cand = (sc, s2, H3, a,b,c,alpha,beta,gamma, atoms_after)
        if (best is None) or (cand[0] < best[0]): best = cand
        if ok_len and ok_ang:
            return s2, {"within_tol": True, "H3": H3.tolist(), "atoms": atoms_after,
                        "a":a,"b":b,"c":c,"alpha":alpha,"beta":beta,"gamma":gamma}
    if best is None:
        return None, {"within_tol": False, "reason":"no_candidate"}
    sc,s2,H3,a,b,c,alpha,beta,gamma,atoms_after = best
    return s2, {"within_tol": False, "best_score": sc, "H3": H3.tolist(), "atoms": atoms_after,
                "a":a,"b":b,"c":c,"alpha":alpha,"beta":beta,"gamma":gamma}

# ====== 方案B：仿射“严格立方”（有应变） ======
def _affine_deform(struct: Structure, F: np.ndarray) -> Structure:
    A = struct.lattice.matrix.T
    A_new = (F @ A).T
    cart = np.array([s.coords for s in struct.sites])
    cart_new = (F @ cart.T).T
    return Structure(Lattice(A_new), [s.specie for s in struct.sites],
                     cart_new, coords_are_cartesian=True, to_unit_cell=True)

def make_fully_cubic_affine(struct: Structure, edge: float = None) -> Structure:
    """把当前晶格仿射到严格立方（a=b=c, 90°）。"""
    A = struct.lattice.matrix  # 行向量
    a, b, c = A[0], A[1], A[2]
    e1 = a/np.linalg.norm(a)
    b_perp = b - np.dot(b, e1)*e1
    if np.linalg.norm(b_perp) < 1e-10:
        tmp = np.array([1.0,0.0,0.0])
        if np.linalg.norm(np.cross(e1,tmp)) < 1e-6: tmp = np.array([0.0,1.0,0.0])
        b_perp = np.cross(np.cross(e1,tmp), e1)
    e2 = b_perp / np.linalg.norm(b_perp)
    e3 = np.cross(e1,e2); e3 /= np.linalg.norm(e3)
    L = max(np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)) if edge is None else float(edge)
    A_t = np.vstack([L*e1, L*e2, L*e3])  # 目标立方基
    F = A_t.T @ np.linalg.inv(A.T)
    return _affine_deform(struct, F)


def lattice_metrics(a_vec, b_vec, c_vec):
    """返回边长与夹角（度）"""
    la, lb, lc = np.linalg.norm(a_vec), np.linalg.norm(b_vec), np.linalg.norm(c_vec)
    def ang(u, v):
        cu = np.clip(np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v)), -1.0, 1.0)
        return float(np.degrees(np.arccos(cu)))
    alpha = ang(b_vec, c_vec)
    beta  = ang(a_vec, c_vec)
    gamma = ang(a_vec, b_vec)
    return la, lb, lc, alpha, beta, gamma

def make_supercell_int(struct: Structure, H3: np.ndarray) -> Structure:
    """使用 3x3 整数矩阵 H3 做超胞/变基（det>0）。不引入应变。"""
    s2 = struct.copy()
    s2.make_supercell(H3)
    return s2

def score_cubic(a, b, c, alpha, beta, gamma):
    """对“立方度”打分：长度等边 + 角度近 90°；分数越小越好。"""
    L = (a + b + c) / 3.0
    len_dev = (abs(a-L) + abs(b-L) + abs(c-L)) / (3.0*L)  # 相对偏差
    ang_dev = (abs(alpha-90.0) + abs(beta-90.0) + abs(gamma-90.0)) / 270.0
    return len_dev + ang_dev

def search_2d_square_inplane(struct: Structure,
                             max_n: int = 4,
                             tol_len: float = 0.02,
                             tol_ang: float = 1.0,
                             max_atoms_after: Optional[int] = None
                             ) -> Tuple[Optional[Structure], Dict[str, Any]]:
    """
    仅在 ab 平面搜索 2x2 整数矩阵 [[i,j],[k,l]]（det>0），构造 3x3 H3=diag(2D,1)，
    使新的 a,b 近似正交且等长（c 不变）。无应变，仅复制/变基。
    满足容差则返回；否则返回最佳近似并在 info 指明 'within_tol': False。
    """
    lat = struct.lattice.matrix
    a0, b0, c0 = lat[0], lat[1], lat[2]
    best = None
    for i,j,k,l in itertools.product(range(-max_n, max_n+1), repeat=4):
        H2 = np.array([[i,j],[k,l]], dtype=int)
        det2 = int(round(np.linalg.det(H2)))
        if det2 <= 0:
            continue
        H3 = np.array([[i,j,0],[k,l,0],[0,0,1]], dtype=int)
        atoms_after = len(struct) * det2
        if (max_atoms_after is not None) and (atoms_after > max_atoms_after):
            continue
        s2 = make_supercell_int(struct, H3)
        a_vec, b_vec, c_vec = s2.lattice.matrix
        a, b, c, alpha, beta, gamma = lattice_metrics(a_vec, b_vec, c_vec)
        len_ok = abs(a-b) / ((a+b)/2.0) <= tol_len
        ang_ok = abs(gamma - 90.0) <= tol_ang  # 只需要面内角 ~90°
        sc = score_cubic(a, b, c, alpha, beta, gamma)  # 综合分
        cand = (sc, s2, a, b, c, alpha, beta, gamma, (i,j,k,l), atoms_after)
        if (best is None) or (cand[0] < best[0]):
            best = cand
        if len_ok and ang_ok:
            return s2, {"within_tol": True, "H2": (i,j,k,l), "atoms": atoms_after,
                        "a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma}
    if best is None:
        return None, {"within_tol": False, "reason": "no_candidate"}
    sc, s2, a, b, c, alpha, beta, gamma, H2, atoms_after = best
    return s2, {"within_tol": False, "best_score": sc, "H2": H2, "atoms": atoms_after,
                "a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma}

def search_3d_near_cubic(struct: Structure,
                         max_n: int = 3,
                         tol_len: float = 0.02,
                         tol_ang: float = 1.0,
                         max_atoms_after: Optional[int] = None
                         ) -> Tuple[Optional[Structure], Dict[str, Any]]:
    """
    在 3D 搜索 3x3 整数矩阵 H3（det>0），使 (a,b,c) 近等边且 (α,β,γ) 近 90°。
    无应变，仅复制/变基。注意：ic>1 意味着会复制 slab+真空的 c 方向周期。
    """
    best = None
    for H in itertools.product(range(-max_n, max_n+1), repeat=9):
        H3 = np.array(H, dtype=int).reshape(3,3)
        det3 = int(round(np.linalg.det(H3)))
        if det3 <= 0:
            continue
        atoms_after = len(struct) * det3
        if (max_atoms_after is not None) and (atoms_after > max_atoms_after):
            continue
        s2 = make_supercell_int(struct, H3)
        a_vec, b_vec, c_vec = s2.lattice.matrix
        a, b, c, alpha, beta, gamma = lattice_metrics(a_vec, b_vec, c_vec)
        len_ok = (abs(a-b)/((a+b)/2.0) <= tol_len) and (abs(a-c)/((a+c)/2.0) <= tol_len)
        ang_ok = (abs(alpha-90.0) <= tol_ang and abs(beta-90.0) <= tol_ang and abs(gamma-90.0) <= tol_ang)
        sc = score_cubic(a, b, c, alpha, beta, gamma)
        cand = (sc, s2, a, b, c, alpha, beta, gamma, H3, atoms_after)
        if (best is None) or (cand[0] < best[0]):
            best = cand
        if len_ok and ang_ok:
            return s2, {"within_tol": True, "H3": H3.tolist(), "atoms": atoms_after,
                        "a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma}
    if best is None:
        return None, {"within_tol": False, "reason": "no_candidate"}
    sc, s2, a, b, c, alpha, beta, gamma, H3, atoms_after = best
    return s2, {"within_tol": False, "best_score": sc, "H3": H3.tolist(), "atoms": atoms_after,
                "a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma}


def _affine_deform(struct: Structure, F: np.ndarray) -> Structure:
    """
    对结构施加仿射变形 x' = F x 。
    同时把晶格矩阵 A 变为 A' = F A ，并对所有原子笛卡尔坐标施加同样 F。
    """
    A = struct.lattice.matrix.T  # 3x3 列为 a,b,c
    A_new = (F @ A).T
    cart = np.array([s.coords for s in struct.sites])  # Nx3
    cart_new = (F @ cart.T).T
    lat_new = Lattice(A_new)
    return Structure(lat_new, [s.specie for s in struct.sites], cart_new,
                     coords_are_cartesian=True, to_unit_cell=True)

def make_ab_square(struct: Structure, target_len: float = None) -> Structure:
    """
    面内立方化：令 a' ⟂ b' 且 |a'|=|b'|=target_len（若未给，则取当前 |a|、|b| 的均值）。
    c' = c（不改法向与真空）。
    实现：在当前 a,b 张成的平面内做 Gram-Schmidt + 等长缩放，构造 F 使 A' = F A。
    """
    A = struct.lattice.matrix  # 行向量 [a;b;c]
    a, b, c = A[0], A[1], A[2]
    # Gram-Schmidt：a0 = a，b0 去投影
    a0 = a
    e1 = a0 / np.linalg.norm(a0)
    b_par = np.dot(b, e1) * e1
    b_perp = b - b_par
    if np.linalg.norm(b_perp) < 1e-10:
        # a 与 b 几乎共线，取与 a 正交的任意向量
        tmp = np.array([1.0, 0.0, 0.0])
        if np.linalg.norm(np.cross(e1, tmp)) < 1e-6:
            tmp = np.array([0.0, 1.0, 0.0])
        b_perp = np.cross(np.cross(e1, tmp), e1)
    e2 = b_perp / np.linalg.norm(b_perp)
    # 目标长度
    if target_len is None:
        L = 0.5 * (np.linalg.norm(a) + np.linalg.norm(b))
    else:
        L = float(target_len)
    a_target = L * e1
    b_target = L * e2
    c_target = c  # 保持法向/真空
    A_target = np.vstack([a_target, b_target, c_target])  # 行向量
    # F 满足 A_target = F A
    F = A_target.T @ np.linalg.inv(A.T)
    return _affine_deform(struct, F)

def make_fully_cubic(struct: Structure, edge: float = None) -> Structure:
    """
    全立方：令 a'=b'=c'，互相正交。默认边长为 max(|a|,|b|,|c|)，也可指定 edge。
    注意：这会改变真空厚度或大幅放大面内面积，不建议用于表面。
    """
    A = struct.lattice.matrix  # 行向量
    a, b, c = A[0], A[1], A[2]
    e1 = a / np.linalg.norm(a)
    # 让 e2 在 a-b 平面内与 e1 正交
    b_perp = b - np.dot(b, e1) * e1
    if np.linalg.norm(b_perp) < 1e-10:
        # 退化时构造一个与 e1 正交的向量
        tmp = np.array([1.0, 0.0, 0.0])
        if np.linalg.norm(np.cross(e1, tmp)) < 1e-6:
            tmp = np.array([0.0, 1.0, 0.0])
        b_perp = np.cross(np.cross(e1, tmp), e1)
    e2 = b_perp / np.linalg.norm(b_perp)
    e3 = np.cross(e1, e2); e3 /= np.linalg.norm(e3)
    L = max(np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)) if edge is None else float(edge)
    A_target = np.vstack([L*e1, L*e2, L*e3])
    F = A_target.T @ np.linalg.inv(A.T)
    return _affine_deform(struct, F)


def _unit(v): 
    n = np.linalg.norm(v); 
    return v / n if n > 0 else v

def _rot_mat_from_u_to_v(u, v):
    """Rodrigues 旋转：把向量 u 旋到 v（均为单位向量）"""
    u = _unit(u); v = _unit(v)
    c = np.dot(u, v)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        # u 与 v 反向：任选一条与 u 不平行的轴做 180°
        x = np.array([1.0,0.0,0.0])
        axis = _unit(np.cross(u, x)) if np.linalg.norm(np.cross(u, x)) > 1e-8 else _unit(np.cross(u, np.array([0.0,1.0,0.0])))
        K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        return np.eye(3) + 2*K@K   # 180°
    axis = _unit(np.cross(u, v))
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3) + K + K@K * (c/(1+c))

def rigid_reorient_c_perp_ab(struct: Structure) -> Structure:
    """
    刚体旋转结构：将第三晶格矢量 c 对齐到 z 轴，使 c ⟂ ab。
    不改变原子相对几何，仅变换坐标系。
    """
    lat = struct.lattice
    a, b, c = lat.matrix
    R1 = _rot_mat_from_u_to_v(c, np.array([0.0, 0.0, 1.0]))  # c -> z
    A1 = (R1 @ lat.matrix.T).T
    # 现在 c 基本沿 z，再绕 z 轴旋转，使 a 落在 xy 平面（清理数值残差）
    a1, b1, c1 = A1
    a1[2] = 0.0; b1[2] = 0.0
    # 可再把 c 的 x,y 残差置零
    c1[0] = 0.0; c1[1] = 0.0
    new_lat = Lattice([a1, b1, c1])

    # 旋转所有原子到新坐标系
    cart = np.array([site.coords for site in struct.sites])
    cart1 = (R1 @ cart.T).T
    # 去掉极小的 z 残差（与 a,b 平面正交）
    cart1[:,2] = cart1[:,2]  # 保留真实 z；a,b 面内坐标无需再改
    new_struct = Structure(new_lat, [site.specie for site in struct.sites], cart1, coords_are_cartesian=True, to_unit_cell=True)
    return new_struct


def try_2d_supercell_square(slab: Structure,
                            max_n: int = 3,
                            tol_ang_deg: float = 3.0,
                            tol_len_frac: float = 0.05,
                            max_atoms_after: Optional[int] = None):
    """
    在 ab 平面搜索 2x2 整数超胞 [[i,j],[k,l]]（|i|,|j|,|k|,|l| ≤ max_n，det>0），
    使新面内晶格尽量接近正方：a≈b 且 γ≈90°。仅复制原子，不拉伸。
    满足 tol 后返回超胞 slab；否则返回 None。
    """
    lat = slab.lattice
    a_vec, b_vec, c_vec = lat.matrix[0], lat.matrix[1], lat.matrix[2]

    best = None  # (score, H2, new_struct)
    for i,j,k,l in itertools.product(range(-max_n, max_n+1), repeat=4):
        H2 = np.array([[i,j],[k,l]], dtype=int)
        det = int(round(np.linalg.det(H2)))
        if det <= 0 or det == 0:  # 只要正定的 2D 超胞
            continue
        # 新面内晶格向量
        a2 = i*a_vec + j*b_vec
        b2 = k*a_vec + l*b_vec
        a_len = np.linalg.norm(a2)
        b_len = np.linalg.norm(b2)
        if a_len < 1e-6 or b_len < 1e-6:
            continue
        cosg = np.dot(a2, b2) / (a_len*b_len)
        cosg = max(-1.0, min(1.0, cosg))
        gamma = np.degrees(np.arccos(cosg))

        # 目标：a≈b、γ≈90°
        len_dev = abs(a_len - b_len) / ((a_len + b_len)/2.0)
        ang_dev = abs(gamma - 90.0)

        score = len_dev + (ang_dev/90.0)  # 简单无量纲打分
        # 原子数约束
        atoms_after = len(slab) * abs(det)
        if (max_atoms_after is not None) and (atoms_after > max_atoms_after):
            continue

        # 记最好
        if (best is None) or (score < best[0]):
            # 复制结构：在 3D 里用 [[i,j,0],[k,l,0],[0,0,1]]
            H3 = np.array([[i,j,0],
                           [k,l,0],
                           [0,0,1]], dtype=int)
            s2 = slab.copy()
            s2.make_supercell(H3)
            best = (score, (i,j,k,l), s2, a_len, b_len, gamma, atoms_after, det)

    if best is None:
        return None, None

    score, (i,j,k,l), s2, a_len, b_len, gamma, atoms_after, det = best
    # 校验是否达到“近方形”的阈值
    len_dev = abs(a_len - b_len) / ((a_len + b_len)/2.0)
    ang_dev = abs(gamma - 90.0)
    if (len_dev <= tol_len_frac) and (ang_dev <= tol_ang_deg):
        return s2, {"H2": (i,j,k,l), "a": a_len, "b": b_len, "gamma": gamma,
                    "atoms": atoms_after, "det": det}
    return None, {"best_score": score, "best_H2": (i,j,k,l), "a": a_len, "b": b_len,
                  "gamma": gamma, "atoms": atoms_after, "det": det}


# ====== 白名单金属 ======
ALLOWED_METALS: Set[str] = {
    "Li","Na","Mg","K","Ca","Ba",
    "V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Zr","Mo","Ru","Rh","Pd","Ag","Cd",
    "Hf","Pt","Au","Hg",
    "Al","Pb",
    "Ce","Gd"
}

# ---------- 工具 ----------
def _unwrap_frac_z(z: np.ndarray) -> np.ndarray:
    if len(z) == 0: return z
    z_sorted = np.sort(z)
    gaps = np.diff(np.r_[z_sorted, z_sorted[0] + 1.0])
    k = int(np.argmax(gaps))
    base = z_sorted[(k + 1) % len(z_sorted)]
    zu = z - base
    zu[zu < 0] += 1.0
    return zu

def unwrap_frac_along_c(struct: Structure) -> np.ndarray:
    return _unwrap_frac_z(np.array([s.frac_coords[2] for s in struct.sites]))

def is_metal_oxide(struct: Structure) -> bool:
    els = [sp.symbol for sp in struct.composition.elements]
    return ("O" in els) and any(e.is_metal and e.symbol != "O"
                                for e in struct.composition.elements)

def metal_species_in_struct(struct: Structure) -> Set[str]:
    return {e.symbol for e in struct.composition.elements if e.symbol!="O" and e.is_metal}

def orient_oxygen_to_top(struct: Structure, top_frac=0.10) -> Structure:
    s = struct.copy()
    f = np.array([site.frac_coords for site in s])
    zu = _unwrap_frac_z(f[:, 2])
    N = len(s)
    cut = max(1, int(np.ceil(N * top_frac)))
    idx = np.argsort(zu)
    top_ids = idx[-cut:]; bot_ids = idx[:cut]
    nO_top = sum(1 for i in top_ids if s[i].specie.symbol == "O")
    nO_bot = sum(1 for i in bot_ids if s[i].specie.symbol == "O")
    if nO_top < nO_bot:
        f_new = np.copy(f)
        f_new[:, 2] = (1.0 - zu) % 1.0
        return Structure(s.lattice, [site.specie for site in s], f_new,
                         coords_are_cartesian=False, to_unit_cell=True)
    return s

def add_vacuum_asymmetric(struct: Structure, vac_top_A: float, vac_bot_A: float) -> Structure:
    s = struct.copy()
    lat0 = s.lattice
    Z0 = lat0.c
    f = np.array([site.frac_coords for site in s])
    zu = _unwrap_frac_z(f[:, 2])
    zmin, zmax = float(np.min(zu)), float(np.max(zu))
    span_frac = max(1e-12, zmax - zmin)
    slab_A = span_frac * Z0
    new_c = slab_A + float(vac_top_A) + float(vac_bot_A)
    scale = new_c / Z0
    new_lat = Lattice([lat0.matrix[0], lat0.matrix[1], lat0.matrix[2] * scale])
    new_span = slab_A / new_c
    new_base = vac_bot_A / new_c
    new_f = np.copy(f)
    new_f[:, 2] = (new_base + (zu - zmin) * (new_span / span_frac)) % 1.0
    return Structure(new_lat, [site.specie for site in s], new_f,
                     coords_are_cartesian=False, to_unit_cell=True)

def call_with_supported(func, **kwargs):
    try:
        sig = inspect.signature(func)
        ok = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return func(**ok)
    except Exception:
        return func(*kwargs.values())

def top_window_ids(struct: Structure, win_A: float = 1.5):
    z = np.array([site.coords[2] for site in struct.sites])
    zmax = float(np.max(z))
    ids = [i for i in range(len(struct)) if (zmax - z[i]) <= win_A + 1e-8]
    return ids, z, zmax

def is_O_terminated(struct: Structure, win_A: float, min_o_frac: float) -> bool:
    ids, z, zmax = top_window_ids(struct, win_A)
    if not ids: return False
    tops = [i for i in ids if abs(z[i]-zmax) <= 1e-6]
    if not tops or not all(struct[i].specie.symbol == "O" for i in tops):
        return False
    o_cnt = sum(1 for i in ids if struct[i].specie.symbol == "O")
    return (o_cnt / max(1, len(ids))) >= float(min_o_frac)

def remove_topmost_oxygen_by_cart(struct: Structure) -> Structure:
    s = struct.copy()
    z = np.array([site.coords[2] for site in s.sites])
    o_idx = [i for i,site in enumerate(s.sites) if site.specie.symbol == "O"]
    if not o_idx:
        return s
    top = max(o_idx, key=lambda i: z[i])
    s.remove_sites([top])
    return s

# ---------- 按 z 聚类统计层数 ----------
def count_layers_by_cart_z(struct: Structure, tol_A: float) -> Tuple[int, List[float]]:
    z = sorted([site.coords[2] for site in struct.sites])
    if not z: return 0, []
    layers = [z[0]]
    for zz in z[1:]:
        if abs(zz - layers[-1]) > tol_A:
            layers.append(zz)
    return len(layers), layers

# ---------- 生成 slab（支持层数偏好与 O 终止） ----------
def quick_make_slab_with_layers(bulk: Structure,
                                miller: Tuple[int,int,int],
                                min_slab: float, min_vac: float,
                                shifts: List[float],
                                center_slab: bool,
                                top_window: float, min_o_frac: float,
                                primitive: bool,
                                n_layers: Optional[int],
                                layer_tol: float) -> Optional[Structure]:
    gen = call_with_supported(
        SlabGenerator,
        initial_structure=bulk,
        miller_index=tuple(miller),
        min_slab_size=min_slab,
        min_vacuum_size=min_vac,
        center_slab=center_slab,
        in_unit_planes=True,
        primitive=primitive,
        max_normal_search=1,
        reorient_lattice=True,
    )

    # 第一优先：层数满足 & 顶层 O 终止
    best_alt = None  # (layer_diff, -o_frac_top, slab)
    for sh in shifts:
        try:
            slab = gen.get_slab(shift=sh)
        except TypeError:
            slabs = gen.get_slabs(ftol=0.1) if hasattr(gen, "get_slabs") else []
            cands = slabs
        else:
            cands = [slab] if slab is not None else []

        for s in cands:
            L, _ = count_layers_by_cart_z(s, tol_A=layer_tol)
            o_term = is_O_terminated(s, top_window, min_o_frac)
            layer_diff = 0 if (n_layers is None) else abs(L - int(n_layers))
            # 完全匹配层数 + O 终止 → 立即返回
            if (n_layers is None or layer_diff == 0) and o_term:
                return s
            # 保存“次优”（层数差最小 + 顶层 O 最多）
            ids, z, zmax = top_window_ids(s, top_window)
            o_frac = 0.0 if not ids else sum(1 for i in ids if s[i].specie.symbol=="O")/len(ids)
            score = (layer_diff, -o_frac)
            if (best_alt is None) or (score < (best_alt[0], best_alt[1])):
                best_alt = (layer_diff, -o_frac, s)

    return None if best_alt is None else best_alt[2]

# ---------- 表面金属位点 ----------
def surface_metal_indices(struct: Structure, tol_A: float) -> Tuple[List[int], List[int]]:
    s = struct
    lat = s.lattice
    zu = unwrap_frac_along_c(s)
    metals = [i for i,site in enumerate(s.sites)
              if isinstance(site.specie, Element) and site.specie.is_metal and site.specie.symbol != "O"]
    if not metals:
        return [], []
    zmax, zmin = float(np.max(zu[metals])), float(np.min(zu[metals]))
    thr = tol_A / lat.c
    top = sorted([i for i in metals if (zmax - zu[i]) <= thr])
    bot = sorted([i for i in metals if (zu[i] - zmin) <= thr])
    return top, bot

# ---------- FPS 均匀采样 ----------
def farthest_point_sampling_xy(struct: Structure, candidates: List[int], k: int) -> List[int]:
    if k <= 0 or not candidates: return []
    k = min(k, len(candidates))
    xy = np.array([struct[i].coords[:2] for i in candidates])
    center = np.mean(xy, axis=0)
    d2 = np.sum((xy - center)**2, axis=1)
    seed = int(np.argmax(d2))
    chosen = [seed]
    dist = np.linalg.norm(xy - xy[seed], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(xy - xy[nxt], axis=1))
    return sorted(candidates[i] for i in chosen)

def pick_hosts_global_balanced_uniform(top_idx: List[int], bot_idx: List[int],
                                       struct: Structure, dopant: str, frac: float):
    def host_pool(idxs):
        return [i for i in idxs if (isinstance(struct[i].specie, Element)
                and struct[i].specie.is_metal and struct[i].specie.symbol != "O"
                and struct[i].specie.symbol != dopant)]
    host_top = host_pool(top_idx); host_bot = host_pool(bot_idx)
    S_top, S_bot = len(host_top), len(host_bot)
    S_total = S_top + S_bot
    if S_total == 0:
        return [], [], {"reason": "no_host", "S_top": S_top, "S_bot": S_bot, "S_total": 0}
    T_total = max(1, int(round(S_total * frac)))  # 25%
    if T_total % 2 == 1: T_total -= 1
    T_half = T_total // 2
    if T_half == 0 or S_top < T_half or S_bot < T_half:
        return [], [], {"reason": "insufficient", "S_top": S_top, "S_bot": S_bot,
                        "S_total": S_total, "T_total": T_total, "T_half": T_half}
    picks_top = farthest_point_sampling_xy(struct, host_top, T_half)
    picks_bot = farthest_point_sampling_xy(struct, host_bot, T_half)
    return picks_top, picks_bot, {
        "S_top": S_top, "S_bot": S_bot, "S_total": S_total,
        "T_total": T_total, "T_top": T_half, "T_bot": T_half
    }

def substitute(struct: Structure, idx: List[int], dopant: str) -> Structure:
    s2 = struct.copy()
    for i in idx:
        if s2[i].specie.symbol != dopant:
            s2[i] = Element(dopant)
    return s2

# ---------- 受控 XY 扩胞 ----------
def choose_xy_multipliers(struct: Structure,
                          a_min: float, b_min: float,
                          a_max_mult: int, b_max_mult: int,
                          target_atoms: Optional[int]) -> Tuple[int,int]:
    lat = struct.lattice
    base_atoms = len(struct)
    best = None
    for ia in range(1, max(1,a_max_mult)+1):
        for ib in range(1, max(1,b_max_mult)+1):
            if (lat.a * ia) < a_min or (lat.b * ib) < b_min:
                continue
            atoms = base_atoms * ia * ib
            cost = (ia*ib) if target_atoms is None else abs(atoms - target_atoms) + 1e-6*(ia*ib)
            cand = (cost, ia, ib)
            if (best is None) or (cand < best):
                best = cand
    if best is None:
        best = (float("inf"), 1, 1)
        for ia in range(1, max(1,a_max_mult)+1):
            for ib in range(1, max(1,b_max_mult)+1):
                da = max(0.0, a_min - lat.a*ia)
                db = max(0.0, b_min - lat.b*ib)
                cost = da + db + 1e-6*(ia*ib)
                cand = (cost, ia, ib)
                if cand < best:
                    best = cand
    _, ia, ib = best
    return ia, ib

def make_supercell_xy(struct: Structure, ia: int, ib: int) -> Structure:
    if ia==1 and ib==1: return struct.copy()
    s = struct.copy()
    s.make_supercell([ia, ib, 1])
    return s

# ---------- 读入后瘦身 ----------
def reduce_cell(bulk: Structure, mode: str) -> Structure:
    mode = mode.lower()
    if mode == "none": return bulk
    try:
        if mode == "primitive":
            return bulk.get_primitive_structure()
        elif mode == "symm_primitive":
            sga = SpacegroupAnalyzer(bulk, symprec=1e-3, angle_tolerance=5)
            return sga.get_primitive_standard_structure()
        elif mode == "conventional":
            sga = SpacegroupAnalyzer(bulk, symprec=1e-3, angle_tolerance=5)
            return sga.get_conventional_standard_structure()
        else:
            return bulk
    except Exception:
        return bulk

# ---------- 解析 miller 参数 ----------
def parse_miller(s: str) -> Tuple[int,int,int]:
    s = s.strip().replace("(", "").replace(")", "")
    parts = [p for p in s.replace(",", " ").split() if p]
    if len(parts) != 3:
        raise ValueError("--miller 需要 3 个整数，例如 '1,1,1' 或 '0 0 1'")
    return tuple(int(x) for x in parts)  # type: ignore

# ================= 主程序 =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="out_ids_layers")
    ap.add_argument("--ids", type=str, default="mp-1336,mp-1692,mp-7947,mp-753682,mp-755073,mp-776104,mp-1093993,mp-1147768,mp-1178232")
    ap.add_argument("--mapi_key", type=str, default=None)

    # 截面
    ap.add_argument("--miller", type=str, default="1,1,1", help="Miller 指数，如 '1,1,1' 或 '0 0 1'")
    ap.add_argument("--n_shifts", type=int, default=5)
    ap.add_argument("--top_window", type=float, default=1.5)
    ap.add_argument("--min_o_frac", type=float, default=0.5)
    ap.add_argument("--primitive_slab", action="store_true")

    # 指定层数（可选）
    ap.add_argument("--n_layers", type=int, default=None, help="若给出，将优先选择层数=该值的 slab")
    ap.add_argument("--layer_tol", type=float, default=0.5, help="按 z 聚类计层的容差(Å)")

    # 最小厚/真空（仍需提供，用于 SlabGenerator 构造候选）
    ap.add_argument("--min_slab", type=float, default=10.0)
    ap.add_argument("--min_vac",  type=float, default=10.0)

    # 真空
    ap.add_argument("--vac_top",  type=float, default=20.0)
    ap.add_argument("--vac_bot",  type=float, default=3.0)

    # 读入瘦身
    ap.add_argument("--reduce", type=str, default="symm_primitive",
                    choices=["none","primitive","symm_primitive","conventional"])

    # XY 扩胞控制
    ap.add_argument("--a_min", type=float, default=8.0)
    ap.add_argument("--b_min", type=float, default=8.0)
    ap.add_argument("--a_max_mult", type=int, default=2)
    ap.add_argument("--b_max_mult", type=int, default=2)
    ap.add_argument("--target_atoms", type=int, default=90)
    ap.add_argument("--max_atoms_after", type=int, default=120)

    # 掺杂
    ap.add_argument("--dopants", type=str, default="ALL")
    ap.add_argument("--dopant_to_host", type=str, default="1:3")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--square_inplane", action="store_true",
                help="在 ab 平面搜索接近正方的 2D 超胞（只复制，不拉伸）")
    ap.add_argument("--sq_max_n", type=int, default=3,
                    help="2D 超胞搜索的整数范围 |i|,|j|,|k|,|l| ≤ N")
    ap.add_argument("--sq_tol_ang", type=float, default=3.0,
                    help="近方形判定角度容差(°)，默认±3°")
    ap.add_argument("--sq_tol_len", type=float, default=0.05,
                    help="近方形判定边长相对差容差，默认5%")
    ap.add_argument("--enforce_c_perp", action="store_true",
                help="刚体旋转使 c 轴垂直于 ab 面（表面法向沿全局 z）")
    ap.add_argument("--cubic_inplane", action="store_true",
                help="将 ab 平面立方化（a ⟂ b 且 a=b），c 不变。用于 (0 1 1) 等表面时保持真空。")
    ap.add_argument("--cubic_full", action="store_true",
                    help="将晶胞完全立方 (a=b=c)。会影响真空/面内尺寸，表面计算一般不建议。")
    ap.add_argument("--cubic_len", type=float, default=0.0,
                    help="立方化目标边长；0 表示自动选择（面内取均值，全立方取 max）。")
    ap.add_argument("--cube_mode", type=str, default="off",
                choices=["off","inplane","full"],
                help="立方化模式：off=不启用；inplane=仅 ab 近立方（2D 整数超胞，无应变，c 不变）；full=3D 近立方（整数超胞）")
    ap.add_argument("--cube_max_n", type=int, default=3, help="立方化搜索的整数范围 |n|≤N")
    ap.add_argument("--cube_tol_len", type=float, default=0.02, help="长度相对容差（默认2%）")
    ap.add_argument("--cube_tol_ang", type=float, default=1.0, help="角度容差（默认±1°）")
    ap.add_argument("--cubic_mode", type=str, default="off",
                choices=["off","supercell","affine"],
                help="切面后立方化：off=不立方；supercell=整数超胞近立方（无应变）；affine=仿射严格立方（有应变）")

    args = ap.parse_args()
    random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # 解析 miller
    miller = parse_miller(args.miller)

    # 掺杂列表
    if args.dopants.upper()=="ALL":
        dopant_list = sorted(ALLOWED_METALS)
    else:
        _in = [x.strip() for x in args.dopants.split(",") if x.strip()]
        dopant_list = sorted(list(ALLOWED_METALS.intersection(_in)))
        if not dopant_list:
            raise RuntimeError("给定掺杂元素不在白名单中。")
    d,h = [int(x) for x in args.dopant_to_host.split(":")]
    assert d>0 and h>0
    frac = d/(d+h)  # 1:3 => 0.25

    # 拉 MP
    api_key = args.mapi_key or os.environ.get("MAPI_KEY")
    if not api_key:
        raise RuntimeError("Missing MAPI_KEY")
    ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    with MPRester(api_key) as mpr:
        docs = list(mpr.summary.search(material_ids=ids, fields=["material_id","formula_pretty","structure"], chunk_size=50))
    found: Dict[str, Structure] = {d.material_id: d.structure for d in docs}
    missing = [x for x in ids if x not in found]
    if missing:
        print("[WARN] 未找到：", ",".join(missing))

    # 处理
    records: List[Dict[str,Any]] = []
    for mid in ids:
        if mid not in found:
            continue
        struct = found[mid]
        if not is_metal_oxide(struct):
            records.append({"material_id": mid, "status": "skip_not_oxide"}); continue
        metals_in = metal_species_in_struct(struct)
        if (not metals_in) or (not metals_in.issubset(ALLOWED_METALS)):
            records.append({"material_id": mid, "status": "skip_disallowed_metals",
                            "metals_in_struct": ",".join(sorted(metals_in))})
            continue

        # 1) 瘦身 + O 在上
        bulk = reduce_cell(struct, args.reduce)
        bulk = orient_oxygen_to_top(bulk, top_frac=0.10)

        # 2) 快切 (hkl)，带层数偏好 + O 终止
        shifts = [(i+0.5)/args.n_shifts for i in range(args.n_shifts)]
        slab = quick_make_slab_with_layers(
            bulk, miller=miller,
            min_slab=args.min_slab, min_vac=args.min_vac,
            shifts=shifts, center_slab=False,
            top_window=args.top_window, min_o_frac=args.min_o_frac,
            primitive=args.primitive_slab,
            n_layers=args.n_layers, layer_tol=args.layer_tol
        )
        if slab is None:
            records.append({"material_id": mid, "status": "skip_no_slab"}); continue
        
        if args.enforce_c_perp:
            slab = rigid_reorient_c_perp_ab(slab)

        # 3)  z 真空
        slab = add_vacuum_asymmetric(slab, vac_top_A=args.vac_top, vac_bot_A=args.vac_bot)

        if args.cubic_mode == "supercell":   # 无应变近立方（推荐先试）
            slab2, info = search_3d_near_cubic_int_supercell(
                slab,
                max_n=args.cube_max_n,         # 建议 3~5
                tol_len=args.cube_tol_len,     # 例如 0.01 (=1%)
                tol_ang=args.cube_tol_ang,     # 例如 0.5 (度)
                max_atoms_after=args.max_atoms_after
            )
            if slab2 is not None: slab = slab2
        elif args.cubic_mode == "affine":     # 严格立方（有应变）
            slab = make_fully_cubic_affine(slab, edge=None)

        # (B) 面内立方化 / 全立方（任选其一；若两者同时给，则优先生效 full）
        if args.cubic_full:
            L = None if args.cubic_len <= 0 else args.cubic_len
            slab = make_fully_cubic(slab, edge=L)
        elif args.cubic_inplane:
            L = None if args.cubic_len <= 0 else args.cubic_len
            slab = make_ab_square(slab, target_len=L)

        if args.square_inplane:
            sq_slab, info = try_2d_supercell_square(
                slab,
                max_n=args.sq_max_n,
                tol_ang_deg=args.sq_tol_ang,
                tol_len_frac=args.sq_tol_len,
                max_atoms_after=args.max_atoms_after
            )
            if sq_slab is not None:
                slab = sq_slab

        if args.cube_mode == "inplane":
            slab2, info = search_2d_square_inplane(
                slab, max_n=args.cube_max_n,
                tol_len=args.cube_tol_len, tol_ang=args.cube_tol_ang,
                max_atoms_after=args.max_atoms_after
            )
            if slab2 is not None: slab = slab2
        elif args.cube_mode == "full":
            slab2, info = search_3d_near_cubic(
                slab, max_n=args.cube_max_n,
                tol_len=args.cube_tol_len, tol_ang=args.cube_tol_ang,
                max_atoms_after=args.max_atoms_after
            )
            if slab2 is not None: slab = slab2

        # 4) XY 受控扩胞
        target_atoms = None if args.target_atoms in (0,-1) else int(args.target_atoms)
        ia, ib = choose_xy_multipliers(slab,
                                       a_min=args.a_min, b_min=args.b_min,
                                       a_max_mult=args.a_max_mult, b_max_mult=args.b_max_mult,
                                       target_atoms=target_atoms)
        slab = make_supercell_xy(slab, ia, ib)
        if args.max_atoms_after and len(slab) > int(args.max_atoms_after):
            records.append({"material_id": mid, "status": "skip_too_many_atoms",
                            "atoms": len(slab)}); continue
        # 删除最顶层 1 个 O
        slab = remove_topmost_oxygen_by_cart(slab)
        # 5) 写基线
        base_path = os.path.join(
            args.outdir,
            f"{mid}_{miller[0]}{miller[1]}{miller[2]}_Otop_rmTopO_a{ia}x_b{ib}x_vacTop{int(args.vac_top)}_vacBot{int(args.vac_bot)}.cif"
        )
        slab.to(fmt="cif", filename=base_path)
        records.append({"material_id": mid, "status": "ok_base", "outfile": base_path,
                        "ia": ia, "ib": ib, "atoms": len(slab)})

        # 6) 表面索引 + 严格 1:3 掺杂
        top_idx, bot_idx = surface_metal_indices(slab, tol_A=1.2)
        if not top_idx or not bot_idx:
            records.append({"material_id": mid, "status": "no_surface_for_doping"}); continue

        for dop in dopant_list:
            picks_top, picks_bot, info = pick_hosts_global_balanced_uniform(top_idx, bot_idx, slab, dop, frac)
            if not picks_top and not picks_bot:
                records.append({"material_id": mid, "status":"skip_capacity", "dopant": dop, **info}); continue
            doped = substitute(slab, picks_top + picks_bot, dop)
            fpath = os.path.join(
                args.outdir,
                f"{mid}_{miller[0]}{miller[1]}{miller[2]}_Otop_rmTopO_a{ia}x_b{ib}x_vacTop{int(args.vac_top)}_vacBot{int(args.vac_bot)}_dope_{dop}.cif"
            )
            doped.to(fmt="cif", filename=fpath)
            records.append({"material_id": mid, "status":"ok_doped", "outfile": fpath,
                            "dopant": dop, **info})

    # manifest
    man_path = os.path.join(args.outdir, "manifest.tsv")
    keys = sorted({k for r in records for k in r.keys()})
    with open(man_path, "w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for r in records:
            f.write("\t".join(str(r.get(k,"")) for k in keys) + "\n")
    print(f"[OK] records={len(records)} -> {man_path}")

if __name__ == "__main__":
    main()
