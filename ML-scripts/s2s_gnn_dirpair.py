#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structure-to-Structure GNN (vacancy-aware) — v2 with better matching & training stability

改进点：
  • 允许 out 原子数 < in（氧空位）
  • 用『匈牙利算法』做物种内一一最优匹配（较贪心更稳），并设置最大匹配距离阈值
  • 两头输出：位移 Δr（仅对保留原子） + 移除概率（默认仅 O 允许移除）
  • 训练稳定性：OneCycleLR 调度、梯度裁剪、Dropout、(可选) Δr 归一化按坐标维度加权
  • 评估拆分：分别报告 Δr RMSE(Å) 与 vacancy 分类 AUROC/AP
  • checkpoint 兼容 PyTorch>=2.6 安全反序列化

用法：
  # 1) 构建配对
  python s2s_gnn_dirpair.py prepare \
    --in-dir ML-vac-full-cif --out-dir ML-vac-out-cif \
    --save-dir runs/s2s_vac --cutoff 6.0

  # 2) 训练（推荐参数）
  python s2s_gnn_dirpair.py train \
    --save-dir runs/s2s_vac --cutoff 6.0 \
    --epochs 200 --batch-size 16 --lr 1e-3 --max-lr 3e-3 \
    --layers 6 --hidden 256 --dropout 0.15 \
    --lambda-bce 0.3 --norm-dr --match-max-dist 1.2 --remove-element O

  # 3) 预测（阈值或 Top-K）
  python s2s_gnn_dirpair.py predict \
    --in-cif ML-vac-full-cif/353-011-Au.cif \
    --ckpt runs/s2s_vac/best.pt \
    --out-cif pred-353-011-Au-out.cif \
    --cutoff 6.0 --remove-mode threshold --remove-threshold 0.5 --remove-element O

依赖：pymatgen, torch, torch_geometric, numpy, scikit-learn, scipy(用于匈牙利匹配；若无则回退贪心)
"""

import os, re, sys, json, math, random, argparse
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data as GeomData
    from torch_geometric.loader import DataLoader
except Exception as e:
    raise RuntimeError("Please install torch_geometric: pip install torch-geometric (and torch-scatter etc.)") from e

from pymatgen.core import Structure

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ---------------- Utils ----------------

def set_seed(seed: int = 2025):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def is_cif(path: str) -> bool:
    return path.lower().endswith('.cif')

# ---------------- Pairing by directories ----------------

def _basename_wo_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def find_pairs_by_dirs(in_dir: str, out_dir: str) -> List[Tuple[str, str]]:
    in_files = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if is_cif(f)]
    out_files = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if is_cif(f)]
    in_map = {_basename_wo_ext(p): p for p in in_files}
    out_map = {_basename_wo_ext(p): p for p in out_files}
    commons = sorted(set(in_map.keys()) & set(out_map.keys()))
    pairs = [(in_map[k], out_map[k]) for k in commons]
    miss_in = sorted(set(out_map.keys()) - set(in_map.keys()))
    miss_out = sorted(set(in_map.keys()) - set(out_map.keys()))
    if miss_in:
        print(f"[WARN] {len(miss_in)} names only in out_dir (first 3): {miss_in[:3]}")
    if miss_out:
        print(f"[WARN] {len(miss_out)} names only in in_dir (first 3): {miss_out[:3]}")
    return pairs

# ---------------- Geometry helpers ----------------

def frac_diff_min_image(frac_j: np.ndarray, frac_i: np.ndarray) -> np.ndarray:
    d = frac_j - frac_i
    return d - np.round(d)

def cart_min_image_delta(lattice: np.ndarray, frac_j: np.ndarray, frac_i: np.ndarray) -> np.ndarray:
    df = frac_diff_min_image(frac_j, frac_i)
    return lattice @ df

def min_image_dist(lattice: np.ndarray, f1: np.ndarray, f2: np.ndarray) -> float:
    return float(np.linalg.norm(cart_min_image_delta(lattice, f1, f2)))

def build_graph(struct: Structure, cutoff: float):
    lattice = np.array(struct.lattice.matrix, dtype=float)
    z = np.array([site.specie.Z for site in struct.sites], dtype=np.int64)
    pos = np.array([site.coords for site in struct.sites], dtype=float)
    i_idx, j_idx, offsets, dists = struct.get_neighbor_list(r=cutoff)
    i_idx = np.asarray(i_idx, dtype=np.int64)
    j_idx = np.asarray(j_idx, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    dists = np.asarray(dists, dtype=float)
    fcoords = np.array([site.frac_coords for site in struct.sites], dtype=float)
    df = (fcoords[j_idx] - fcoords[i_idx]) + offsets
    dcart = (lattice @ df.T).T
    edge_index = np.stack([i_idx, j_idx], axis=0)
    edge_attr = np.concatenate([dcart, dists.reshape(-1,1)], axis=1)
    return z, pos, edge_index, edge_attr

# ---------------- Optimal species-wise matching ----------------

def hungarian_species_match(in_s: Structure, out_s: Structure, max_dist: float = 1.5) -> Dict[int, int]:
    """Return mapping from in_index -> out_index for atoms that survive.
    物种内按最短镜像距离构造代价矩阵，Hungarian 一次性最优匹配；过滤距离>max_dist 的匹配。
    若无 SciPy，则回退到简单贪心。
    """
    latt = np.array(in_s.lattice.matrix, dtype=float)
    in_f = np.array([a.frac_coords for a in in_s.sites])
    out_f = np.array([a.frac_coords for a in out_s.sites])
    in_sp = [str(a.specie) for a in in_s.sites]
    out_sp = [str(a.specie) for a in out_s.sites]
    mapping: Dict[int,int] = {}

    species = sorted(set(out_sp))
    for sp in species:
        in_ids = [i for i,s in enumerate(in_sp) if s==sp]
        out_ids = [j for j,s in enumerate(out_sp) if s==sp]
        if not in_ids or not out_ids:
            continue
        n_in, n_out = len(in_ids), len(out_ids)
        # 构造代价矩阵 (n_in, n_out)
        C = np.zeros((n_in, n_out), dtype=float)
        for ii, i in enumerate(in_ids):
            for jj, j in enumerate(out_ids):
                C[ii, jj] = min_image_dist(latt, out_f[j], in_f[i])
        if _HAS_SCIPY:
            row_ind, col_ind = linear_sum_assignment(C)
            for r, c in zip(row_ind, col_ind):
                i = in_ids[r]; j = out_ids[c]
                if C[r, c] <= max_dist:
                    mapping[i] = j
        else:
            # 贪心回退
            pairs = []
            for ii, i in enumerate(in_ids):
                for jj, j in enumerate(out_ids):
                    pairs.append((C[ii, jj], i, j))
            pairs.sort(key=lambda x: x[0])
            used_i, used_j = set(), set()
            for d, i, j in pairs:
                if i in used_i or j in used_j:
                    continue
                if d <= max_dist:
                    mapping[i] = j
                    used_i.add(i); used_j.add(j)
    return mapping

# ---------------- Dataset ----------------

@dataclass
class PairItem:
    in_cif: str
    out_cif: str

class S2SDataset(torch.utils.data.Dataset):
    def __init__(self, pairs: List[PairItem], cutoff: float, remove_element: Optional[str] = 'O', match_max_dist: float = 1.5, norm_dr: bool = True):
        self.items: List[GeomData] = []
        self.cutoff = cutoff
        self.norm_dr = norm_dr
        self.dr_scale = None  # (3,) per-dim std for Δr

        total_keep = 0
        total_remove = 0
        all_dr_keep = []
        for in_p, out_p in [(p.in_cif, p.out_cif) for p in pairs]:
            try:
                in_s = Structure.from_file(in_p)
                out_s = Structure.from_file(out_p)
            except Exception as e:
                print(f"[WARN] skip unreadable: {in_p} <-> {out_p}: {e}")
                continue
            mapping = hungarian_species_match(in_s, out_s, max_dist=match_max_dist)
            N = len(in_s)
            latt_in = np.array(in_s.lattice.matrix, dtype=float)
            y_dr = np.zeros((N,3), dtype=float)
            y_rm = np.ones((N,1), dtype=float)  # 1=removed
            mask_keep = np.zeros((N,), dtype=bool)
            for i_in, j_out in mapping.items():
                fi = in_s[i_in].frac_coords
                fj = out_s[j_out].frac_coords
                y_dr[i_in] = cart_min_image_delta(latt_in, fj, fi)
                y_rm[i_in,0] = 0.0
                mask_keep[i_in] = True
            symbols = [str(site.specie) for site in in_s.sites]
            if remove_element:
                not_target = np.array([s!=remove_element for s in symbols])
                y_rm[not_target,0] = 0.0
                mask_keep[not_target] = True
            total_keep += int(mask_keep.sum())
            total_remove += int((~mask_keep).sum())
            if mask_keep.any():
                all_dr_keep.append(y_dr[mask_keep])
            z, pos, edge_index, edge_attr = build_graph(in_s, self.cutoff)
            data = GeomData()
            data.x = torch.from_numpy(z.astype(np.int64))
            data.pos = torch.from_numpy(pos.astype(np.float32))
            data.edge_index = torch.from_numpy(edge_index.astype(np.int64))
            data.edge_attr = torch.from_numpy(edge_attr.astype(np.float32))
            data.y_dr = torch.from_numpy(y_dr.astype(np.float32))
            data.y_rm = torch.from_numpy(y_rm.astype(np.float32))
            data.mask_keep = torch.from_numpy(mask_keep)
            data.sid = os.path.basename(in_p).replace('.cif','')
            self.items.append(data)
        if not self.items:
            raise RuntimeError("No valid pairs prepared. Check directory names and file contents.")
        # class balance
        pos = float(total_remove); neg = float(total_keep)
        self.pos_weight = torch.tensor([ (neg / max(pos,1.0)) ], dtype=torch.float32)
        # Δr scale (per-dim std; robust到极端)
        if self.norm_dr and all_dr_keep:
            dr_cat = np.concatenate(all_dr_keep, axis=0)
            std = dr_cat.std(axis=0)
            std = np.clip(std, 1e-3, None)
            self.dr_scale = torch.tensor(std, dtype=torch.float32)  # (3,)
        print(f"[INFO] Removal class balance: keep={int(neg)} | remove={int(pos)} | pos_weight={self.pos_weight.item():.3f}")
        if self.dr_scale is not None:
            print(f"[INFO] Δr per-dim std used for normalization: {self.dr_scale.tolist()}")

    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]

# ---------------- Model ----------------

class RBFLayer(nn.Module):
    def __init__(self, centers: int = 32, cutoff: float = 5.0):
        super().__init__()
        self.register_buffer('centers', torch.linspace(0.0, cutoff, centers))
        self.register_buffer('gamma', torch.tensor(10.0/(cutoff**2)))
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        diff = r.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * diff**2)

class AtomEncoder(nn.Module):
    def __init__(self, max_z=100, emb_dim=128):
        super().__init__()
        self.emb = nn.Embedding(max_z+1, emb_dim)
    def forward(self, z):
        z = torch.clamp(z, 0, self.emb.num_embeddings-1)
        return self.emb(z)

class S2SModel(nn.Module):
    def __init__(self, emb_dim=128, hidden=128, layers=4, rbf_dim=32, cutoff=5.0, max_z=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.atom_enc = AtomEncoder(max_z=max_z, emb_dim=emb_dim)
        # project atom embedding (emb_dim) → hidden to avoid shape mismatch when concat with edge features
        self.atom_lin = nn.Linear(emb_dim, hidden)
        self.atom_act = nn.SiLU()
        self.rbf = RBFLayer(centers=rbf_dim, cutoff=cutoff)
        self.edge_lin = nn.Linear(rbf_dim + 3 + 1, hidden)
        self.edge_act = nn.SiLU()
        self.mp_layers = nn.ModuleList([self._make_mp(hidden, dropout) for _ in range(layers)])
        self.head_dr = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 3))
        self.head_rm = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
    @staticmethod
    def _make_mp(hidden, dropout):
        return nn.ModuleDict({
            'msg': nn.Sequential(nn.Linear(hidden*2, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden)),
            'upd': nn.Sequential(nn.Linear(hidden*2, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden)),
            'n1': nn.LayerNorm(hidden),
            'n2': nn.LayerNorm(hidden),
        })
    def encode_edges(self, edge_attr):
        dcart = edge_attr[:, :3]; dist = edge_attr[:, 3]
        rbf = self.rbf(dist)
        e = torch.cat([rbf, dcart, dist.unsqueeze(-1)], dim=-1)
        return self.edge_act(self.edge_lin(e))
    def forward(self, z, pos, edge_index, edge_attr):
        h = self.atom_act(self.atom_lin(self.atom_enc(z)))
        e = self.encode_edges(edge_attr)
        src, dst = edge_index
        for mp in self.mp_layers:
            m = mp['msg'](torch.cat([h[src], e], dim=-1))
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, m)
            h = mp['n1'](h + agg)
            upd = mp['upd'](torch.cat([h, agg], dim=-1))
            h = mp['n2'](h + upd)
        dr = self.head_dr(h)
        rm_logits = self.head_rm(h)
        return dr, rm_logits

# ---------------- Train/Eval ----------------

def split_indices(n: int, test_size=0.15, val_size=0.15, seed=2025):
    idx = list(range(n))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed)
    tr, va = train_test_split(tr, test_size=val_size/(1.0-test_size), random_state=seed)
    return tr, va, te

class FocalWithLogitsLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.pos_weight = pos_weight
    def forward(self, logits, targets):
        # logits: (N,1), targets: (N,1)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=self.pos_weight)
        p = torch.sigmoid(logits)
        pt = torch.where(targets>0.5, p, 1-p)
        focal = (self.alpha * (1-pt)**self.gamma) * bce
        return focal.mean()

def _dr_scale_loss(dr_pred, y_dr, mask_keep, dr_scale: Optional[torch.Tensor]):
    if mask_keep.sum() == 0:
        return dr_pred.new_tensor(0.0)
    if dr_scale is None:
        return F.mse_loss(dr_pred[mask_keep], y_dr[mask_keep])
    s = dr_scale.to(dr_pred.device).view(1,3)
    return F.mse_loss(dr_pred[mask_keep]/s, y_dr[mask_keep]/s)

def train_one_epoch(model, loader, opt, scheduler, bce_crit, lambda_bce, dr_scale, clip_norm, device):
    model.train(); total = 0.0
    for batch in loader:
        batch = batch.to(device)
        dr_pred, rm_logit = model(batch.x.long(), batch.pos, batch.edge_index, batch.edge_attr)
        mse = _dr_scale_loss(dr_pred, batch.y_dr, batch.mask_keep, dr_scale)
        bce = bce_crit(rm_logit, batch.y_rm)
        loss = mse + lambda_bce * bce
        opt.zero_grad(); loss.backward()
        if clip_norm and clip_norm>0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        opt.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, bce_crit, lambda_bce, dr_scale, device):
    model.eval(); total = 0.0
    all_rm_t, all_rm_p = [], []
    se_sum, n_keep = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        dr_pred, rm_logit = model(batch.x.long(), batch.pos, batch.edge_index, batch.edge_attr)
        # loss
        mse = _dr_scale_loss(dr_pred, batch.y_dr, batch.mask_keep, dr_scale)
        bce = bce_crit(rm_logit, batch.y_rm)
        loss = mse + lambda_bce * bce
        total += loss.item() * batch.num_graphs
        # metrics
        keep = batch.mask_keep
        if keep.sum()>0:
            diff = (dr_pred[keep] - batch.y_dr[keep]).pow(2).sum(dim=1).sqrt().cpu().numpy()  # per-atom L2
            se_sum += float((diff**2).sum())
            n_keep += int(keep.sum())
        all_rm_t.append(batch.y_rm.detach().cpu().numpy().reshape(-1))
        all_rm_p.append(torch.sigmoid(rm_logit).detach().cpu().numpy().reshape(-1))
    # compute metrics
    if n_keep>0:
        rmse = math.sqrt(se_sum / (n_keep*1.0))
    else:
        rmse = float('nan')
    y_true = np.concatenate(all_rm_t); y_prob = np.concatenate(all_rm_p)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
    try:
        ap = average_precision_score(y_true, y_prob)
    except Exception:
        ap = float('nan')
    return total / len(loader.dataset), rmse, auc, ap

# ---------------- Checkpoint helpers ----------------

def _prune_cfg(cfg_dict):
    safe_types = (int, float, str, bool, type(None))
    out = {}
    for k, v in cfg_dict.items():
        if k == 'func':
            continue
        if isinstance(v, safe_types):
            out[k] = v
        else:
            try:
                out[k] = str(v)
            except Exception:
                pass
    return out

def save_checkpoint(path, model, epoch, best_val, cfg, dr_scale):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cfg_safe = _prune_cfg(cfg) if isinstance(cfg, dict) else None
    payload = {"epoch": epoch, "state_dict": model.state_dict(), "best_val": best_val}
    if cfg_safe is not None:
        payload["config"] = cfg_safe
    if dr_scale is not None:
        payload["dr_scale"] = dr_scale.cpu()
    torch.save(payload, path)

def load_checkpoint(path, model, map_location="cpu"):
    state = None; ckpt = None
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
        state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    except Exception:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    return ckpt

# ---------------- Prediction ----------------

def wrap01(frac):
    return frac - np.floor(frac)

@torch.no_grad()
def predict_structure(model: S2SModel, in_cif: str, cutoff: float, device: str,
                      remove_mode: str = 'threshold', remove_threshold: float = 0.5,
                      remove_topk: int = 0, remove_element: Optional[str] = 'O',
                      dr_scale: Optional[torch.Tensor] = None) -> Structure:
    s = Structure.from_file(in_cif)
    z, pos, edge_index, edge_attr = build_graph(s, cutoff)
    zt = torch.from_numpy(z.astype(np.int64)).to(device)
    post = torch.from_numpy(pos.astype(np.float32)).to(device)
    eit = torch.from_numpy(edge_index.astype(np.int64)).to(device)
    eattr = torch.from_numpy(edge_attr.astype(np.float32)).to(device)
    dr_pred, rm_logit = model(zt, post, eit, eattr)
    # 若训练用了 dr_scale，仅用于 loss，不影响推理值；这里无需反归一化
    rm_prob = torch.sigmoid(rm_logit).cpu().numpy().reshape(-1)
    dr = dr_pred.cpu().numpy()

    species = [str(site.specie) for site in s.sites]
    elig = np.array([True]*len(species))
    if remove_element:
        elig = np.array([sp==remove_element for sp in species])

    remove_mask = np.zeros((len(species),), dtype=bool)
    if remove_mode == 'threshold':
        remove_mask = (rm_prob >= remove_threshold) & elig
    elif remove_mode == 'topk' and remove_topk > 0:
        idx = np.where(elig)[0]
        if len(idx) > 0:
            top_idx = idx[np.argsort(-rm_prob[idx])[:min(remove_topk, len(idx))]]
            remove_mask[top_idx] = True
    else:
        remove_mask = (rm_prob >= remove_threshold) & elig

    latt = np.array(s.lattice.matrix, dtype=float)
    inv = np.linalg.inv(latt)
    cart = np.array([site.coords for site in s.sites], dtype=float)
    keep_mask = ~remove_mask
    new_cart = cart[keep_mask] + dr[keep_mask]
    new_frac = (inv @ new_cart.T).T
    new_frac = wrap01(new_frac)
    new_species = [s.sites[i].specie for i in range(len(s.sites)) if keep_mask[i]]
    from pymatgen.core import Structure as PMGStructure
    new_struct = PMGStructure(lattice=s.lattice, species=new_species, coords=new_frac,
                              to_unit_cell=True, coords_are_cartesian=False)
    return new_struct

# ---------------- CLI ----------------

def cmd_prepare(args):
    pairs = find_pairs_by_dirs(args.in_dir, args.out_dir)
    if not pairs:
        raise RuntimeError("No matched filenames across the two directories.")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'pairs.json'), 'w', encoding='utf-8') as f:
        json.dump([{'in_cif': a, 'out_cif': b} for a,b in pairs], f, ensure_ascii=False, indent=2)
    print(f"[OK] Prepared {len(pairs)} pairs → {os.path.join(args.save_dir, 'pairs.json')}")


def load_pairs(save_dir: str) -> List[PairItem]:
    with open(os.path.join(save_dir, 'pairs.json'), 'r', encoding='utf-8') as f:
        lst = json.load(f)
    return [PairItem(**d) for d in lst]


def cmd_train(args):
    device = 'cuda' if (torch.cuda.is_available() and (not args.cpu)) else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)
    if os.path.exists(os.path.join(args.save_dir, 'pairs.json')):
        pairs = load_pairs(args.save_dir)
    else:
        pairs = [PairItem(*ab) for ab in find_pairs_by_dirs(args.in_dir, args.out_dir)]
    if not pairs:
        raise RuntimeError("Empty pairs. Run 'prepare' first or check directories.")

    dataset = S2SDataset(pairs, cutoff=args.cutoff, remove_element=args.remove_element, match_max_dist=args.match_max_dist, norm_dr=args.norm_dr)
    tr_idx, va_idx, te_idx = split_indices(len(dataset))
    subset = lambda ids: [dataset.items[i] for i in ids]
    train_loader = DataLoader(subset(tr_idx), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(subset(va_idx), batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(subset(te_idx), batch_size=args.batch_size, shuffle=False)

    model = S2SModel(emb_dim=args.emb_dim, hidden=args.hidden, layers=args.layers,
                     rbf_dim=args.rbf_dim, cutoff=args.cutoff, max_z=args.max_z, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # scheduler: OneCycleLR (per-batch)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.max_lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch)

    # loss for removal: focal 或 BCE（默认用 BCE + pos_weight）
    if args.use_focal:
        bce_crit = FocalWithLogitsLoss(alpha=0.25, gamma=2.0, pos_weight=dataset.pos_weight.to(device))
    else:
        bce_crit = nn.BCEWithLogitsLoss(pos_weight=dataset.pos_weight.to(device))

    best_val = float('inf'); best_path = os.path.join(args.save_dir, 'best.pt')
    patience = args.patience; bad = 0
    cfg = vars(args)

    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_loader, opt, scheduler, bce_crit, args.lambda_bce, dataset.dr_scale, args.clip_norm, device)
        va_loss, va_rmse, va_auc, va_ap = eval_epoch(model, val_loader, bce_crit, args.lambda_bce, dataset.dr_scale, device)
        print(f"Epoch {epoch:03d} | train {tr_loss:.6f} | val {va_loss:.6f} | Δr_RMSE {va_rmse:.4f} Å | AUC {va_auc:.3f} | AP {va_ap:.3f}")
        if va_loss < best_val - 1e-8:
            best_val = va_loss; save_checkpoint(best_path, model, epoch, best_val, cfg, dataset.dr_scale); bad = 0
        else:
            bad += 1
        if bad >= patience:
            print(f"[EarlyStop] No val improvement for {patience} epochs.")
            break

    ckpt = load_checkpoint(best_path, model, map_location=device)
    dr_scale = ckpt.get('dr_scale', dataset.dr_scale)
    te_loss, te_rmse, te_auc, te_ap = eval_epoch(model, test_loader, bce_crit, args.lambda_bce, dr_scale, device)
    print(f"[TEST] loss {te_loss:.6f} | Δr_RMSE {te_rmse:.4f} Å | AUC {te_auc:.3f} | AP {te_ap:.3f}")


def cmd_predict(args):
    device = 'cuda' if (torch.cuda.is_available() and (not args.cpu)) else 'cpu'
    dummy = S2SModel(emb_dim=args.emb_dim, hidden=args.hidden, layers=args.layers,
                     rbf_dim=args.rbf_dim, cutoff=args.cutoff, max_z=args.max_z, dropout=args.dropout).to(device)
    ckpt = load_checkpoint(args.ckpt, dummy, map_location=device)
    cfg = ckpt.get('config', None)
    dr_scale = ckpt.get('dr_scale', None)
    if cfg:
        print('[INFO] Using architecture from checkpoint config.')
        dummy = S2SModel(emb_dim=cfg.get('emb_dim',128), hidden=cfg.get('hidden',128), layers=cfg.get('layers',4),
                         rbf_dim=cfg.get('rbf_dim',32), cutoff=cfg.get('cutoff', args.cutoff), max_z=cfg.get('max_z',100), dropout=cfg.get('dropout',0.1)).to(device)
        load_checkpoint(args.ckpt, dummy, map_location=device)
    pred_struct = predict_structure(dummy, args.in_cif, cutoff=args.cutoff, device=device,
                                    remove_mode=args.remove_mode, remove_threshold=args.remove_threshold,
                                    remove_topk=args.remove_topk, remove_element=args.remove_element,
                                    dr_scale=dr_scale)
    out_path = args.out_cif if args.out_cif else re.sub(r"\.cif$", "-pred-out.cif", args.in_cif, flags=re.I)
    pred_struct.to(fmt='cif', filename=out_path)
    print(f"[OK] wrote predicted structure: {out_path}")

# ---------------- Main ----------------

def build_parser():
    p = argparse.ArgumentParser(description='Structure→Structure GNN with vacancy handling (Hungarian matching, schedulers, metrics)')
    sub = p.add_subparsers(dest='cmd', required=True)

    sp = sub.add_parser('prepare', help='scan two directories and save pairs.json')
    sp.add_argument('--in-dir', type=str, required=True)
    sp.add_argument('--out-dir', type=str, required=True)
    sp.add_argument('--save-dir', type=str, default='runs/s2s_vac')
    sp.add_argument('--cutoff', type=float, default=6.0)
    sp.set_defaults(func=cmd_prepare)

    st = sub.add_parser('train', help='train the GNN (Δr + removal)')
    st.add_argument('--in-dir', type=str, default='ML-vac-full-cif')
    st.add_argument('--out-dir', type=str, default='ML-vac-out-cif')
    st.add_argument('--save-dir', type=str, default='runs/s2s_vac')
    st.add_argument('--cutoff', type=float, default=6.0)
    st.add_argument('--epochs', type=int, default=200)
    st.add_argument('--batch-size', type=int, default=16)
    st.add_argument('--lr', type=float, default=1e-3)
    st.add_argument('--max-lr', type=float, default=3e-3)
    st.add_argument('--wd', type=float, default=1e-6)
    st.add_argument('--emb-dim', type=int, default=128)
    st.add_argument('--hidden', type=int, default=256)
    st.add_argument('--layers', type=int, default=6)
    st.add_argument('--rbf-dim', type=int, default=32)
    st.add_argument('--max-z', type=int, default=100)
    st.add_argument('--dropout', type=float, default=0.15)
    st.add_argument('--lambda-bce', type=float, default=0.3)
    st.add_argument('--test-size', type=float, default=0.15)
    st.add_argument('--val-size', type=float, default=0.15)
    st.add_argument('--patience', type=int, default=20)
    st.add_argument('--seed', type=int, default=2025)
    st.add_argument('--remove-element', type=str, default='O')
    st.add_argument('--match-max-dist', type=float, default=1.2, help='max Å for matching in→out atoms')
    st.add_argument('--norm-dr', action='store_true', help='use per-dim std to normalize Δr loss weight')
    st.add_argument('--use-focal', action='store_true', help='use focal loss for removal head')
    st.add_argument('--clip-norm', type=float, default=1.0, help='gradient clip norm (0=disable)')
    st.add_argument('--cpu', action='store_true')
    st.set_defaults(func=cmd_train)

    spd = sub.add_parser('predict', help='predict evolved out.cif for a single in.cif')
    spd.add_argument('--in-cif', type=str, required=True)
    spd.add_argument('--ckpt', type=str, required=True)
    spd.add_argument('--out-cif', type=str, default=None)
    spd.add_argument('--cutoff', type=float, default=6.0)
    spd.add_argument('--emb-dim', type=int, default=128)
    spd.add_argument('--hidden', type=int, default=256)
    spd.add_argument('--layers', type=int, default=6)
    spd.add_argument('--rbf-dim', type=int, default=32)
    spd.add_argument('--max-z', type=int, default=100)
    spd.add_argument('--dropout', type=float, default=0.15)
    spd.add_argument('--remove-mode', type=str, choices=['threshold','topk'], default='threshold')
    spd.add_argument('--remove-threshold', type=float, default=0.5)
    spd.add_argument('--remove-topk', type=int, default=0)
    spd.add_argument('--remove-element', type=str, default='O')
    spd.add_argument('--cpu', action='store_true')
    spd.set_defaults(func=cmd_predict)

    return p


def main():
    set_seed(2025)
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()


