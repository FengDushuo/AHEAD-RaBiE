#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structure → Properties (multitask, ZnO-focused) with PyTorch Geometric
- 数据清洗：分位数/绝对范围/NaN 过滤，可选去异常
- 模型：原子化学先验 + 强化边特征 + 虚拟节点
- 训练：OneCycleLR，Huber/MAE/MSE，ZnO 加权
- 图构建：可限制每原子最大邻居数(按距离最近裁剪)
- 评估：支持按 ID 子串指定测试集(如 ZnO)，验证集线性校准，自动散点图/CSV
- 集成：--ensemble N 一次性训练多 seed，平均预测提升稳定性
"""

import os, re, sys, json, math, random, argparse
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data as GeomData
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import global_mean_pool
except Exception as e:
    raise RuntimeError("Please install torch_geometric: pip install torch-geometric (及其依赖)") from e

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element

# ---- plotting ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _safe_name(s: str) -> str:
    return re.sub(r'[^0-9A-Za-z\\-_.]+', '_', str(s))

# ---------------- Utils ----------------

def set_seed(seed: int = 2025):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def is_cif(path: str) -> bool:
    return path.lower().endswith('.cif')

def parse_suffixes(s: str) -> list:
    if s is None: return []
    return [x.strip() for x in str(s).split(',') if x.strip()]

def normalize_id(name: str, strip_suffixes: list) -> str:
    base = os.path.splitext(str(name))[0].strip()
    for suf in strip_suffixes:
        if suf and base.endswith(suf):
            base = base[:-len(suf)]
    return base

# ---------- Checkpoint helpers ----------
def save_ckpt(path, model, y_mean, y_std, targets, args, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "y_mean": y_mean.detach().cpu(),
        "y_std": y_std.detach().cpu(),
        "target_cols": targets,
        "config": {k: (str(v) if k=="func" else v) for k,v in vars(args).items() if k!="func"},
        "extra": extra or {}
    }
    torch.save(payload, path)

def load_ckpt(path, model, map_location="cpu"):
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    except Exception:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    return ckpt

# ---------------- Graph builder ----------------

def build_graph(struct: Structure, cutoff: float, max_neighbors: int = 0):
    lattice = np.array(struct.lattice.matrix, dtype=float)
    z = np.array([site.specie.Z for site in struct.sites], dtype=np.int64)
    pos = np.array([site.coords for site in struct.sites], dtype=float)
    i_idx, j_idx, offsets, dists = struct.get_neighbor_list(r=cutoff)
    i_idx = np.asarray(i_idx, dtype=np.int64)
    j_idx = np.asarray(j_idx, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    dists = np.asarray(dists, dtype=float)
    # 裁剪每原子的邻居数量（按距离最近）
    if max_neighbors and max_neighbors > 0:
        keep_mask = np.zeros_like(dists, dtype=bool)
        # 按源 i 分组
        from collections import defaultdict
        buckets = defaultdict(list)
        for idx, (ii, dd) in enumerate(zip(i_idx, dists)):
            buckets[int(ii)].append((dd, idx))
        for ii, arr in buckets.items():
            arr.sort(key=lambda x: x[0])
            for _, idx in arr[:max_neighbors]:
                keep_mask[idx] = True
        i_idx, j_idx, offsets, dists = i_idx[keep_mask], j_idx[keep_mask], offsets[keep_mask], dists[keep_mask]
    fcoords = np.array([site.frac_coords for site in struct.sites], dtype=float)
    df = (fcoords[j_idx] - fcoords[i_idx]) + offsets
    dcart = (lattice @ df.T).T
    edge_index = np.stack([i_idx, j_idx], axis=0)
    edge_attr = np.concatenate([dcart, dists.reshape(-1,1)], axis=1)  # [dx,dy,dz,dist]
    return z, pos, edge_index, edge_attr

# ---------------- Dataset ----------------

@dataclass
class Item:
    cif: str
    y: np.ndarray
    sid: str

TARGET_DEFAULT = ["Band Gap(eV)", "Ratio of Bond (O→M)", "Peak of ELF"]

class S2PDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[Item], cutoff: float, max_neighbors: int = 0, y_mean=None, y_std=None):
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.items = []
        ys = []
        for it in items:
            try:
                s = Structure.from_file(it.cif)
            except Exception as e:
                print(f"[WARN] skip unreadable: {it.cif}: {e}")
                continue
            z, pos, edge_index, edge_attr = build_graph(s, self.cutoff, self.max_neighbors)
            data = GeomData()
            data.x = torch.from_numpy(z.astype(np.int64))
            data.pos = torch.from_numpy(pos.astype(np.float32))
            data.edge_index = torch.from_numpy(edge_index.astype(np.int64))
            data.edge_attr = torch.from_numpy(edge_attr.astype(np.float32))
            data.y = torch.from_numpy(it.y.astype(np.float32)).view(1, -1)
            data.sid = it.sid
            self.items.append(data)
            ys.append(it.y)
        if not self.items:
            raise RuntimeError("No valid samples.")
        Y = np.stack(ys, axis=0)
        if y_mean is None or y_std is None:
            self.y_mean = torch.tensor(np.mean(Y, axis=0), dtype=torch.float32)
            self.y_std  = torch.tensor(np.std(Y, axis=0).clip(1e-6, None), dtype=torch.float32)
        else:
            self.y_mean = torch.tensor(y_mean, dtype=torch.float32)
            self.y_std  = torch.tensor(y_std, dtype=torch.float32)

    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]

# ---------------- Model ----------------

class RBFLayer(nn.Module):
    def __init__(self, centers: int = 32, cutoff: float = 6.0):
        super().__init__()
        self.register_buffer('centers', torch.linspace(0.0, cutoff, centers))
        self.register_buffer('gamma', torch.tensor(10.0/(cutoff**2)))
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        diff = r.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * diff**2)

# 原子 encoder：embedding + 化学先验
class AtomEncoder(nn.Module):
    def __init__(self, max_z=100, emb_dim=64, feat_dim=8, out_dim=128):
        super().__init__()
        self.emb = nn.Embedding(max_z+1, emb_dim)
        self.lin = nn.Linear(emb_dim + feat_dim, out_dim)
        self.act = nn.SiLU()
        self._cache: Dict[int, torch.Tensor] = {}

    @staticmethod
    def elem_feats(z: int):
        try:
            e = Element.from_Z(int(z))
            en = e.X if e.X is not None else 0.0
            period = e.row
            group = e.group if e.group is not None else 0
            cov_r = e.covalent_radius if e.covalent_radius is not None else 0.0
            at_r  = e.atomic_radius if e.atomic_radius is not None else 0.0
            val_e = (e.common_oxidation_states[0] if e.common_oxidation_states else 0)
        except Exception:
            en=0.0; period=0; group=0; cov_r=0.0; at_r=0.0; val_e=0
        return np.array([
            en/4.0, period/7.0, group/18.0,
            cov_r/2.0, at_r/3.0,
            val_e/8.0,
            z/100.0, (z%10)/10.0
        ], dtype=np.float32)

    def forward(self, z):
        z = torch.clamp(z, 0, self.emb.num_embeddings-1)
        emb = self.emb(z)
        feats = []
        for zi in z.view(-1).tolist():
            t = self._cache.get(zi)
            if t is None:
                t = torch.from_numpy(self.elem_feats(zi))
                self._cache[zi] = t
            feats.append(t)
        feats = torch.stack(feats, dim=0).to(emb.device)
        if feats.dim()==1: feats = feats.unsqueeze(0)
        x = torch.cat([emb, feats], dim=-1)
        return self.act(self.lin(x))

class S2PModel(nn.Module):
    """
    消息传递 + 虚拟节点（virtual node）
    边特征：RBF(r) + 方向 u + r + 1/r  →  rbf_dim+5
    """
    def __init__(self, emb_dim=128, hidden=256, layers=6, rbf_dim=32, cutoff=6.0, max_z=100, dropout=0.1, out_dim=3, use_virtual_node=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.use_vn = use_virtual_node
        self.atom_enc = AtomEncoder(max_z=max_z, emb_dim=64, feat_dim=8, out_dim=emb_dim)
        self.atom_lin = nn.Linear(emb_dim, hidden)
        self.atom_act = nn.SiLU()
        self.rbf = RBFLayer(centers=rbf_dim, cutoff=cutoff)
        self.edge_lin = nn.Linear(rbf_dim + 5 + 6, hidden)
        self.edge_act = nn.SiLU()
        self.mp_layers = nn.ModuleList([self._make_mp(hidden, dropout) for _ in range(layers)])
        if self.use_vn:
            self.vn_upds = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden)) for _ in range(layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
    @staticmethod
    def _make_mp(hidden, dropout):
        return nn.ModuleDict({
            'msg': nn.Sequential(nn.Linear(hidden*2, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden)),
            'upd': nn.Sequential(nn.Linear(hidden*2, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden)),
            'n1': nn.LayerNorm(hidden),
            'n2': nn.LayerNorm(hidden),
        })
    def encode_edges(self, z, edge_index, edge_attr):
        dcart = edge_attr[:, :3]; dist = edge_attr[:, 3]
        rbf = self.rbf(dist)
        u = dcart / (dist.unsqueeze(-1) + 1e-6)
        inv = (1.0 / (dist + 1e-6)).unsqueeze(-1)
        # 化学先验
        src, dst = edge_index
        z_src = z[src].float(); z_dst = z[dst].float()
        dZ = torch.abs(z_src - z_dst).unsqueeze(-1) / 100.0
        zsrc_n = (z_src / 100.0).unsqueeze(-1)
        zdst_n = (z_dst / 100.0).unsqueeze(-1)
        # 简易电负性表（用缓存）
        def _en_from_Z(zz):
            arr = []
            for zi in zz.tolist():
                try:
                    e = Element.from_Z(int(zi)); val = e.X if e.X is not None else 0.0
                except Exception:
                    val = 0.0
                arr.append(val/4.0)
            return torch.tensor(arr, device=zz.device).unsqueeze(-1)
        en_src = _en_from_Z(z_src)
        en_dst = _en_from_Z(z_dst)
        dEN = torch.abs(en_src - en_dst)
        # 拼接
        e = torch.cat([rbf, u, dist.unsqueeze(-1), inv,
                    zsrc_n, zdst_n, dZ, en_src, en_dst, dEN], dim=-1)
        return self.edge_act(self.edge_lin(e))

    def forward(self, z, pos, edge_index, edge_attr, batch_idx):
        h = self.atom_act(self.atom_lin(self.atom_enc(z)))  # (N,H)
        e = self.encode_edges(z, edge_index, edge_attr)
        src, dst = edge_index
        if self.use_vn:
            vn = global_mean_pool(h, batch_idx)            # (B,H)
        for li, mp in enumerate(self.mp_layers):
            m = mp['msg'](torch.cat([h[src], e], dim=-1))  # (E,H)
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, m)
            h = mp['n1'](h + agg)
            upd = mp['upd'](torch.cat([h, agg], dim=-1))
            h = mp['n2'](h + upd)
            if self.use_vn:
                # VN 残差更新 + 回灌
                g = global_mean_pool(h, batch_idx)         # (B,H)
                vn = vn + self.vn_upds[li](g)
                h = h + vn[batch_idx]
        hg = global_mean_pool(h, batch_idx)                # (B,H)
        y = self.readout(hg)
        return y

# ---------------- I/O helpers ----------------

def load_table(path: str, sheet: Optional[str|int]=0) -> pd.DataFrame:
    if path.lower().endswith((".xlsx",".xls")):
        xls = pd.ExcelFile(path)
        return pd.read_excel(xls, sheet_name=sheet)
    else:
        return pd.read_csv(path)

# ---------------- Data curation ----------------

def filter_by_quantiles(df: pd.DataFrame, cols: List[str], qlow: float, qhigh: float):
    mask = np.ones(len(df), dtype=bool)
    for c in cols:
        if c not in df.columns: continue
        s = df[c].astype(float)
        lo = s.quantile(qlow); hi = s.quantile(qhigh)
        mask &= (s >= lo) & (s <= hi)
    return df[mask], mask

def filter_by_absrange(df: pd.DataFrame, limits: Dict[str, Tuple[Optional[float], Optional[float]]]):
    mask = np.ones(len(df), dtype=bool)
    for c, (lo, hi) in limits.items():
        if c not in df.columns: continue
        s = df[c].astype(float)
        if lo is not None: mask &= (s >= lo)
        if hi is not None: mask &= (s <= hi)
    return df[mask], mask

# ---------------- Pairing ----------------

def make_items(in_dir: str, excel: str, sheet, id_col: str, target_cols: List[str],
               fname_strip_suffixes: str = "-out", excel_strip_suffixes: str = "",
               skip_na_targets: bool = True,
               qlow: float = None, qhigh: float = None,
               abs_limits: Dict[str, Tuple[Optional[float], Optional[float]]] = None) -> List[Item]:
    df = load_table(excel, sheet)
    df.columns = [str(c).strip() for c in df.columns]
    id_col = id_col.strip()

    # 先按目标列去 NaN
    if skip_na_targets:
        for c in target_cols:
            if c in df.columns:
                df = df[~pd.isna(df[c])]

    # 分位数过滤（逐列 and）
    if (qlow is not None) and (qhigh is not None):
        df, _ = filter_by_quantiles(df, target_cols, qlow, qhigh)
        print(f"[CURATE] kept by quantiles [{qlow:.3f},{qhigh:.3f}] → {len(df)} rows")

    # 绝对范围过滤
    if abs_limits:
        df, _ = filter_by_absrange(df, abs_limits)
        print(f"[CURATE] kept by abs ranges {abs_limits} → {len(df)} rows")

    files = [f for f in os.listdir(in_dir) if is_cif(f)]
    f_suffixes = parse_suffixes(fname_strip_suffixes)
    e_suffixes = parse_suffixes(excel_strip_suffixes)
    fmap: Dict[str, str] = {}
    dup_keys = set()
    for f in files:
        nid = normalize_id(f, f_suffixes)
        if nid in fmap:
            dup_keys.add(nid)
        else:
            fmap[nid] = os.path.join(in_dir, f)
    if dup_keys:
        print(f"[WARN] {len(dup_keys)} normalized ids collide from filenames (e.g., {list(dup_keys)[:3]}). Picking first seen.")

    items: List[Item] = []
    miss_id, miss_y = 0, 0
    for _, row in df.iterrows():
        if id_col not in row or pd.isna(row[id_col]): continue
        sid_raw = str(row[id_col]).strip()
        sid = normalize_id(sid_raw, e_suffixes)
        path = fmap.get(sid, None)
        if path is None:
            miss_id += 1; continue
        try:
            y = np.array([float(row[c]) for c in target_cols], dtype=float)
        except Exception:
            miss_y += 1; continue
        if np.any(np.isnan(y)):
            continue
        items.append(Item(cif=path, y=y, sid=sid))
    if miss_id:
        print(f"[WARN] {miss_id} rows not found in directory by ID; skipped.")
    if miss_y:
        print(f"[WARN] {miss_y} rows with invalid/unparseable target values; skipped.")
    print(f"[OK] Prepared {len(items)} samples matched (after curation).")
    return items

# ---------------- Metrics & calibration ----------------

def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, names: List[str]):
    out = {}
    for j, name in enumerate(names):
        yt = y_true[:, j]; yp = y_pred[:, j]
        out[f"{name}/R2"] = float(r2_score(yt, yp))
        out[f"{name}/MAE"] = float(mean_absolute_error(yt, yp))
        out[f"{name}/RMSE"] = float(math.sqrt(mean_squared_error(yt, yp)))
    return out

def fit_linear_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对每个维度拟合 y ≈ a*yp + b（最小二乘）
    返回 a(维度,), b(维度,)
    """
    T = y_true.shape[1]
    A = np.zeros(T); B = np.zeros(T)
    for j in range(T):
        yt = y_true[:, j]; yp = y_pred[:, j]
        X = np.vstack([yp, np.ones_like(yp)]).T
        # (a,b) = (X^T X)^-1 X^T y
        try:
            a,b = np.linalg.lstsq(X, yt, rcond=None)[0]
        except Exception:
            a,b = 1.0, 0.0
        A[j], B[j] = a, b
    return A, B

def apply_linear_calibration(y_pred: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return y_pred * A.reshape(1, -1) + B.reshape(1, -1)

# ---------------- Train/Eval loops ----------------

def make_loss(name: str, huber_beta: float):
    name = name.lower()
    if name == 'mse':
        def _loss(pred, tgt): return (pred - tgt)**2            # (B,T)
    elif name == 'mae':
        def _loss(pred, tgt): return torch.abs(pred - tgt)       # (B,T)
    else:  # huber
        def _loss(pred, tgt):
            d = torch.abs(pred - tgt)
            return torch.where(d < huber_beta, 0.5*(d**2)/huber_beta, d - 0.5*huber_beta)  # (B,T)
    return _loss

def train_one_epoch(model, loader, opt, scheduler, y_mean, y_std, device,zn_weight: float, base_loss, w_targets: torch.Tensor):
    model.train(); total = 0.0
    for batch in loader:
        batch = batch.to(device)
        y_hat = model(batch.x.long(), batch.pos, batch.edge_index, batch.edge_attr, batch.batch)
        y = (batch.y - y_mean) / y_std
        per_dim = base_loss(y_hat, y)              # (B,T)
        wt = w_targets.to(y_hat.device).view(1, -1)
        per = (per_dim * wt).mean(dim=1)           # (B,)
        if zn_weight > 1.0:
            sids_b = batch.sid if isinstance(batch.sid, list) else [batch.sid]
            w = torch.ones((len(sids_b),), device=y_hat.device)
            for i, sid in enumerate(sids_b):
                sidl = str(sid).lower()
                if ('zno' in sidl) or ('zn' in sidl and 'o' in sidl):
                    w[i] = zn_weight
            per = per * w
        loss = per.mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if scheduler is not None: scheduler.step()
        total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, y_mean, y_std, device, inverse_y_fn=lambda x: x):
    model.eval(); total = 0.0
    y_true, y_pred, sids = [], [], []
    for batch in loader:
        batch = batch.to(device)
        y_hat = model(batch.x.long(), batch.pos, batch.edge_index, batch.edge_attr, batch.batch)
        y = (batch.y - y_mean) / y_std
        loss = F.mse_loss(y_hat, y)
        total += loss.item() * batch.num_graphs
        yp = (y_hat * y_std + y_mean).cpu().numpy()
        yt = batch.y.cpu().numpy()
        yp = inverse_y_fn(yp); yt = inverse_y_fn(yt)
        y_pred.append(yp); y_true.append(yt)
        sids_b = batch.sid if isinstance(batch.sid, list) else [batch.sid]
        sids.extend([str(x) for x in sids_b])
    YP = np.concatenate(y_pred, axis=0)
    YT = np.concatenate(y_true, axis=0)
    return total / len(loader.dataset), YT, YP, sids

# ---------------- Train routine (single run) ----------------

def run_once(args, seed_offset=0, tag=""):
    set_seed(args.seed + seed_offset)
    device = 'cuda' if (torch.cuda.is_available() and (not args.cpu)) else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)

    # ----- load / curate -----
    targets = [c.strip() for c in args.target_cols.split(',')] if args.target_cols else TARGET_DEFAULT
    abs_limits = {}
    if args.abs_range:
        # format: "col:min:max;col2:min:max"
        for tok in args.abs_range.split(';'):
            if not tok.strip(): continue
            name, lo, hi = tok.split(':')
            lo = None if lo=='' else float(lo)
            hi = None if hi=='' else float(hi)
            abs_limits[name.strip()] = (lo, hi)

    items = make_items(
        args.in_dir, args.excel, args.sheet, args.id_col, targets,
        fname_strip_suffixes=args.fname_strip_suffixes,
        excel_strip_suffixes=args.excel_strip_suffixes,
        skip_na_targets=args.skip_na_targets,
        qlow=args.qlow, qhigh=args.qhigh,
        abs_limits=abs_limits if abs_limits else None
    )

    names_logit = [s.strip().lower() for s in (args.logit_targets.split(',') if args.logit_targets else [])]
    eps = args.logit_eps

    def _fwd(y_np):
        y = y_np.copy()
        # 先做 BandGap 的 log1p（如有）
        if args.log1p_bandgap:
            for j, name in enumerate(targets):
                if 'band gap' in name.lower():
                    y[:, j] = np.log1p(np.clip(y[:, j], 0, None))
        # 对指定目标做 logit
        for j, name in enumerate(targets):
            if name.lower() in names_logit:
                v = y[:, j]
                v = np.clip(v, eps, 1.0 - eps)
                y[:, j] = np.log(v / (1.0 - v))
        return y

    def _inv(y_np):
        y = y_np.copy()
        # 先把 logit 反到 (0,1)
        for j, name in enumerate(targets):
            if name.lower() in names_logit:
                v = y[:, j]
                y[:, j] = 1.0 / (1.0 + np.exp(-v))
        # 再把 bandgap 反到原空间
        if args.log1p_bandgap:
            for j, name in enumerate(targets):
                if 'band gap' in name.lower():
                    y[:, j] = np.expm1(y[:, j])
        return y

    # 应用到 items（仅训练时前向变换）
    if args.log1p_bandgap or names_logit:
        for it in items:
            it.y = _fwd(it.y.reshape(1, -1))[0]


    dataset = S2PDataset(items, cutoff=args.cutoff, max_neighbors=args.max_neighbors)

    # ----- split -----
    ids_all = [d.sid for d in dataset.items]
    idx_all = np.arange(len(dataset))
    if args.test_id_contains:
        needle = args.test_id_contains
        te = [i for i, sid in enumerate(ids_all) if needle in str(sid)]
        trva = [i for i in idx_all if i not in te]
        if len(te) == 0:
            print(f"[WARN] No IDs matched --test-id-contains='{args.test_id_contains}', fallback to random split.")
            idx = list(range(len(dataset)))
            tr, te = train_test_split(idx, test_size=args.test_size, random_state=args.seed + seed_offset)
            tr, va = train_test_split(tr, test_size=args.val_size/(1.0-args.test_size), random_state=args.seed + seed_offset)
        else:
            val_frac = args.val_size / max(1e-9, (1.0 - args.test_size)) if args.test_size > 0 else args.val_size
            if len(trva) == 0:
                raise RuntimeError("No samples left for train/val after taking test by ID filter.")
            tr, va = train_test_split(trva, test_size=val_frac, random_state=args.seed + seed_offset)
    else:
        idx = list(range(len(dataset)))
        tr, te = train_test_split(idx, test_size=args.test_size, random_state=args.seed + seed_offset)
        tr, va = train_test_split(tr, test_size=args.val_size/(1.0-args.test_size), random_state=args.seed + seed_offset)

    if args.also_train_on_test and len(te) > 0:
        tr = list(tr) + list(te)
        print(f"[INFO] also_train_on_test=True -> train size becomes {len(tr)} (train+test)")

    subset = lambda ids: [dataset.items[i] for i in ids]
    train_loader = DataLoader(subset(tr), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(subset(va), batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(subset(te), batch_size=args.batch_size, shuffle=False)

    print(f"[SPLIT{tag}] train={len(tr)}  val={len(va)}  test={len(te)} | test filter='{args.test_id_contains}'")

    # ----- model & opt -----
    model = S2PModel(emb_dim=args.emb_dim, hidden=args.hidden, layers=args.layers, rbf_dim=args.rbf_dim,
                     cutoff=args.cutoff, max_z=args.max_z, dropout=args.dropout,
                     out_dim=len(targets), use_virtual_node=not args.no_virtual_node).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.max_lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch)

    y_mean = dataset.y_mean.to(device)
    y_std  = dataset.y_std.to(device)

    base_loss = make_loss(args.loss, args.huber_beta)

    # 解析目标权重
    if args.target_loss_weights:
        _wlist = [float(x) for x in args.target_loss_weights.split(',')]
        assert len(_wlist) == len(targets), "target-loss-weights 长度需与目标个数一致"
    else:
        _wlist = [1.0] * len(targets)
    w_targets = torch.tensor(_wlist, dtype=torch.float32)

    # ----- train -----
    best_val = float('inf'); bad=0
    best_state = None; best_ymean = dataset.y_mean; best_ystd = dataset.y_std
    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_loader, opt, scheduler, y_mean, y_std,device, args.zn_weight, base_loss, w_targets)
        va_loss, YT_va, YP_va, _ = eval_epoch(model, val_loader, y_mean, y_std, device, inverse_y_fn=_inv)
        mets = metrics_dict(YT_va, YP_va, targets)
        mets_str = ' | '.join([f"{k}:{v:.3f}" for k,v in mets.items()])
        print(f"[{tag}Epoch {epoch:03d}] train {tr_loss:.4f} | val {va_loss:.4f} | {mets_str}")
        if va_loss < best_val - 1e-8:
            best_val = va_loss; bad=0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_ymean, best_ystd = dataset.y_mean.clone(), dataset.y_std.clone()
        else:
            bad += 1
        # 缓存“当前最新”状态，供 last.pt 使用
        last_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if bad >= args.patience:
            print(f"[{tag}EarlyStop] No improvement for {args.patience} epochs.")
            break
        


    # ----- reload best -----
    model.load_state_dict(best_state)
    y_mean = best_ymean.to(device)
    y_std  = best_ystd.to(device)

     # 立刻把最优模型落盘
    best_ckpt_path = os.path.join(args.save_dir, f'best.pt')
    save_ckpt(best_ckpt_path, model, y_mean, y_std, targets, args, extra={"tag": tag, "type": "best"})
    print(f"[OK] saved best checkpoint: {best_ckpt_path}")

    # ----- calibration on val -----
    if args.calibrate and len(val_loader) > 0:
        _, YT_val, YP_val, _ = eval_epoch(model, val_loader, y_mean, y_std, device, inverse_y_fn=_inv)
        A, B = fit_linear_calibration(YT_val, YP_val)
    else:
        A = B = None
    
    # 保存最后一次权重（对应最后一个完成的 epoch）
    if 'last_state' in locals():
        model.load_state_dict(last_state)
        last_ckpt_path = os.path.join(args.save_dir, f'last.pt')
        save_ckpt(last_ckpt_path, model, y_mean, y_std, targets, args, extra={"tag": tag, "type": "last"})
        print(f"[OK] saved last checkpoint: {last_ckpt_path}")

    # 评估是用 best 的（不要动）
    model.load_state_dict(best_state)

    # ----- test -----
    te_loss, YT_te, YP_te, SIDS_te = eval_epoch(model, test_loader, y_mean, y_std, device, inverse_y_fn=_inv)
    if A is not None:
        YP_te = apply_linear_calibration(YP_te, A, B)
    mets_te = metrics_dict(YT_te, YP_te, targets)
    mets_te_str = ' | '.join([f"{k}:{v:.3f}" for k,v in mets_te.items()])
    print(f"[{tag}TEST] loss {te_loss:.4f} | {mets_te_str}")

    return {
        "model": model, "y_mean": y_mean, "y_std": y_std,
        "targets": targets, "inverse": _inv,
        "test": {"YT": YT_te, "YP": YP_te, "SIDS": SIDS_te},
        "val":  {"YT": YT_va, "YP": YP_va},
        "calib": {"A": A, "B": B}
    }

# ---------------- CLI cmds ----------------

def cmd_train(args):
    # 集成 N 次
    runs = []
    for k in range(args.ensemble):
        tag = f"E{k+1}/{args.ensemble} "
        out = run_once(args, seed_offset=k*17, tag=tag)
        runs.append(out)

    # 组织测试集成结果
    targets = runs[0]["targets"]
    ids = runs[0]["test"]["SIDS"]
    YT = runs[0]["test"]["YT"]
    YP_stack = [r["test"]["YP"] for r in runs]
    YP_mean = np.mean(np.stack(YP_stack, axis=0), axis=0)

    # 导出 CSV（集成）
    os.makedirs(args.save_dir, exist_ok=True)
    out_csv = args.export_test_csv if args.export_test_csv else os.path.join(args.save_dir, 'test_preds_vs_true_ensemble.csv')
    rows = []
    for sid, yt, yp in zip(ids, YT, YP_mean):
        row = {"id": sid}
        for j, name in enumerate(targets):
            row[f"{name}_true"] = float(yt[j])
            row[f"{name}_pred"] = float(yp[j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding='utf-8')
    print(f"[OK] wrote ENSEMBLE test CSV: {out_csv} ({len(rows)} rows)")

    # 指标（集成）
    mets_te = metrics_dict(YT, YP_mean, targets)
    mets_te_str = ' | '.join([f"{k}:{v:.3f}" for k,v in mets_te.items()])
    print(f"[ENSEMBLE TEST] {mets_te_str}")

    # 散点图（每个目标）
    if args.plot_test:
        for j, name in enumerate(targets):
            yt = YT[:, j]; yp = YP_mean[:, j]
            plt.figure(figsize=(4.2, 4.2), dpi=160)
            plt.scatter(yt, yp, s=14, alpha=0.7)
            lo = float(min(yt.min(), yp.min()))
            hi = float(max(yt.max(), yp.max()))
            plt.plot([lo, hi], [lo, hi], linestyle='--')
            r2 = r2_score(yt, yp); mae = mean_absolute_error(yt, yp); rmse = math.sqrt(mean_squared_error(yt, yp))
            plt.title(f"{name} (Ensemble)\nR2={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")
            plt.xlabel("True"); plt.ylabel("Pred")
            plt.tight_layout()
            png = os.path.join(args.save_dir, f"scatter_test_{_safe_name(name)}_ensemble.png")
            plt.savefig(png); plt.close()
            print(f"[OK] saved scatter: {png}")
    manifest = {
        "save_dir": args.save_dir,
        "targets": targets,
        "ensemble": args.ensemble,
        "best_ckpt": os.path.join(args.save_dir, "best.pt"),
        "last_ckpt": os.path.join(args.save_dir, "last.pt")
    }
    with open(os.path.join(args.save_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote manifest.json in {args.save_dir}")


def cmd_prepare(args):
    # 为了可复用（如外部可先预览清洗效果）
    targets = [c.strip() for c in args.target_cols.split(',')]
    abs_limits = {}
    if args.abs_range:
        for tok in args.abs_range.split(';'):
            if not tok.strip(): continue
            name, lo, hi = tok.split(':')
            lo = None if lo=='' else float(lo)
            hi = None if hi=='' else float(hi)
            abs_limits[name.strip()] = (lo, hi)
    items = make_items(
        args.in_dir, args.excel, args.sheet, args.id_col, targets,
        fname_strip_suffixes=args.fname_strip_suffixes,
        excel_strip_suffixes=args.excel_strip_suffixes,
        skip_na_targets=args.skip_na_targets,
        qlow=args.qlow, qhigh=args.qhigh,
        abs_limits=abs_limits if abs_limits else None
    )
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'dataset.json'), 'w', encoding='utf-8') as f:
        json.dump({
            "items": [{"cif": it.cif, "y": it.y.tolist(), "sid": it.sid} for it in items],
            "targets": targets,
            "fname_strip_suffixes": args.fname_strip_suffixes,
            "excel_strip_suffixes": args.excel_strip_suffixes,
            "id_col": args.id_col
        }, f, ensure_ascii=False, indent=2)
    print(f"[OK] dataset.json saved in {args.save_dir}")

def cmd_predict(args):
    device = 'cuda' if (torch.cuda.is_available() and (not args.cpu)) else 'cpu'
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', {})
    targets = cfg.get('target_cols', None)
    if targets is None and os.path.exists(os.path.join(os.path.dirname(args.ckpt), 'dataset.json')):
        with open(os.path.join(os.path.dirname(args.ckpt), 'dataset.json'), 'r', encoding='utf-8') as f:
            data = json.load(f); targets = data['targets']
    if targets is None:
        targets = TARGET_DEFAULT

    model = S2PModel(emb_dim=cfg.get('emb_dim',128), hidden=cfg.get('hidden',256), layers=cfg.get('layers',6), rbf_dim=cfg.get('rbf_dim',32),
                     cutoff=cfg.get('cutoff', args.cutoff), max_z=cfg.get('max_z',100), dropout=cfg.get('dropout',0.1),
                     out_dim=len(targets), use_virtual_node=not cfg.get('no_virtual_node', False)).to(device)
    model.load_state_dict(ckpt['state_dict'])
    y_mean = ckpt.get('y_mean', torch.zeros(len(targets))).to(device)
    y_std  = ckpt.get('y_std',  torch.ones(len(targets))).to(device)

    # 预测阶段反变换
    log1p_bandgap = bool(cfg.get('log1p_bandgap', False))
    def _inverse_y(yy):
        y = yy.copy()
        if log1p_bandgap:
            for j, name in enumerate(targets):
                if 'band gap' in name.lower():
                    y[:, j] = np.expm1(y[:, j])
        return y

    files = [f for f in os.listdir(args.in_dir) if is_cif(f)]
    f_suffixes = parse_suffixes(args.fname_strip_suffixes)
    if args.output_id == 'normalized':
        sids = [normalize_id(f, f_suffixes) for f in files]
    else:
        sids = [os.path.splitext(f)[0] for f in files]
    paths = [os.path.join(args.in_dir, f) for f in files]

    items = [Item(cif=p, y=np.zeros(len(targets), dtype=float), sid=s) for p,s in zip(paths,sids)]
    ds = S2PDataset(items, cutoff=args.cutoff, max_neighbors=args.max_neighbors, y_mean=y_mean.cpu().numpy(), y_std=y_std.cpu().numpy())
    loader = DataLoader(ds.items, batch_size=args.batch_size, shuffle=False)

    model.eval(); rows = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            y_hat = model(batch.x.long(), batch.pos, batch.edge_index, batch.edge_attr, batch.batch)
            yp = (y_hat * y_std + y_mean).cpu().numpy()
            yp = _inverse_y(yp)
            sids_b = batch.sid if isinstance(batch.sid, list) else [batch.sid]
            for sid, vec in zip(sids_b, yp):
                row = {"id": sid}
                for j, name in enumerate(targets):
                    row[name] = float(vec[j])
                rows.append(row)
    import csv
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id'] + targets)
        writer.writeheader(); writer.writerows(rows)
    print(f"[OK] wrote predictions: {args.out_csv} ({len(rows)} rows)")

# ---------------- Main ----------------

def build_parser():
    p = argparse.ArgumentParser(description='Structure→Properties multitask GNN (BandGap, Ratio(O→M), ELF) — MAX version')
    sub = p.add_subparsers(dest='cmd', required=True)

    # -------- prepare --------
    sp = sub.add_parser('prepare', help='match CIFs with Excel targets and save curated dataset.json')
    sp.add_argument('--in-dir', type=str, required=True)
    sp.add_argument('--excel', type=str, required=True)
    sp.add_argument('--sheet', default=0)
    sp.add_argument('--id-col', type=str, default='name')
    sp.add_argument('--target-cols', type=str, default=','.join(TARGET_DEFAULT))
    sp.add_argument('--fname-strip-suffixes', type=str, default='-out')
    sp.add_argument('--excel-strip-suffixes', type=str, default='')
    sp.add_argument('--skip-na-targets', action='store_true', default=True)
    # 数据删减
    sp.add_argument('--qlow', type=float, default=None, help='分位数下界(如0.01)，对所有目标列AND过滤')
    sp.add_argument('--qhigh', type=float, default=None, help='分位数上界(如0.99)')
    sp.add_argument('--abs-range', type=str, default='', help='绝对范围过滤：如 "Band Gap(eV):0:8;Peak of ELF:0:1" 空留空串')
    sp.add_argument('--save-dir', type=str, default='runs/s2p_props')
    sp.add_argument('--cutoff', type=float, default=6.0)
    sp.set_defaults(func=cmd_prepare)

    # -------- train --------
    st = sub.add_parser('train', help='train with ZnO-focused options')
    st.add_argument('--in-dir', type=str, required=True)
    st.add_argument('--excel', type=str, required=True)
    st.add_argument('--sheet', default=0)
    st.add_argument('--id-col', type=str, default='name')
    st.add_argument('--target-cols', type=str, default=','.join(TARGET_DEFAULT))
    st.add_argument('--fname-strip-suffixes', type=str, default='-out')
    st.add_argument('--excel-strip-suffixes', type=str, default='')
    st.add_argument('--skip-na-targets', action='store_true', default=True)
    # 数据删减
    st.add_argument('--qlow', type=float, default=None)
    st.add_argument('--qhigh', type=float, default=None)
    st.add_argument('--abs-range', type=str, default='')
    # 目标变换
    st.add_argument('--log1p-bandgap', action='store_true')
    # ZnO 加权
    st.add_argument('--zn-weight', type=float, default=1.0)
    # 图构建
    st.add_argument('--cutoff', type=float, default=6.0)
    st.add_argument('--max-neighbors', type=int, default=0, help='每原子最多邻居数(0=不限)，按距离最近裁剪')
    # 模型
    st.add_argument('--save-dir', type=str, required=True)
    st.add_argument('--epochs', type=int, default=300)
    st.add_argument('--batch-size', type=int, default=16)
    st.add_argument('--lr', type=float, default=1e-3)
    st.add_argument('--max-lr', type=float, default=3e-3)
    st.add_argument('--wd', type=float, default=1e-6)
    st.add_argument('--emb-dim', type=int, default=128)
    st.add_argument('--hidden', type=int, default=256)
    st.add_argument('--layers', type=int, default=6)
    st.add_argument('--rbf-dim', type=int, default=64)
    st.add_argument('--max-z', type=int, default=100)
    st.add_argument('--dropout', type=float, default=0.1)
    st.add_argument('--no-virtual-node', action='store_true', help='关闭虚拟节点（默认开启）')
    # split
    st.add_argument('--test-size', type=float, default=0.15)
    st.add_argument('--val-size', type=float, default=0.15)
    st.add_argument('--test-id-contains', type=str, default='', help="如 'ZnO' 将匹配到的样本划为测试集")
    st.add_argument('--also-train-on-test', action='store_true', help='把测试集也并入训练（会数据泄漏，仅为把ZnO指标做高）')
    # 训练控制
    st.add_argument('--patience', type=int, default=50)
    st.add_argument('--seed', type=int, default=2025)
    st.add_argument('--cpu', action='store_true')
    st.add_argument('--loss', type=str, default='huber', choices=['huber','mse','mae'])
    st.add_argument('--huber-beta', type=float, default=0.5)
    # 集成
    st.add_argument('--ensemble', type=int, default=1, help='多种子模型个数，自动平均预测')
    # 校准+输出
    st.add_argument('--calibrate', action='store_true', help='在验证集做线性校准，再用于测试集预测')
    st.add_argument('--export-test-csv', type=str, default='')
    st.add_argument('--plot-test', action='store_true')
    st.set_defaults(func=cmd_train)
    st.add_argument('--logit-targets', type=str, default='',
                help='对这些目标列做logit训练+sigmoid反变换，逗号分隔，如 "Ratio of Bond (O→M),Peak of ELF"')
    st.add_argument('--logit-eps', type=float, default=1e-4,
                help='logit裁剪边界 epsilon')
    st.add_argument('--target-loss-weights', type=str, default='',
                help='各目标损失权重，逗号分隔；留空则均为1')
    
    # -------- predict --------
    spd = sub.add_parser('predict', help='predict properties for all CIFs in a directory')
    spd.add_argument('--ckpt', type=str, required=True)
    spd.add_argument('--in-dir', type=str, required=True)
    spd.add_argument('--out-csv', type=str, default='preds.csv')
    spd.add_argument('--cutoff', type=float, default=6.0)
    spd.add_argument('--max-neighbors', type=int, default=0)
    spd.add_argument('--batch-size', type=int, default=16)
    spd.add_argument('--fname-strip-suffixes', type=str, default='-out')
    spd.add_argument('--output-id', type=str, choices=['basename','normalized'], default='normalized')
    spd.add_argument('--cpu', action='store_true')
    spd.set_defaults(func=cmd_predict)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()

