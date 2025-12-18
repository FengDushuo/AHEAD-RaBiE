#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RATIO-FOCUS: Structure → Ratio of Bond (O→M) with PyTorch Geometric

Highlights for Ratio:
- Single-task learning on "Ratio of Bond (O→M)"
- Target transform: logit on (0,1) (default on), sigmoid inverse at eval/predict
- O–M aware features:
  * edge flag for O–M bonds
  * global graph features: O–M distance stats (mean/std/min/max, quartiles),
    O coordination stats, O fraction etc., concatenated to pooled graph embedding
- Strong training setup: Huber/MSE/MAE, OneCycleLR, EarlyStopping
- Clean split: --test-id-contains (e.g., ZnO) -> test only; by default not leaking into train
- Save checkpoints: best.pt / last.pt, CSV/plots/manifest

Author: you
"""

import os, re, json, math, argparse, random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

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
    raise RuntimeError("Please install torch_geometric: pip install torch-geometric") from e

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element

# ---- plotting ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RATIO_NAME = "Ratio of Bond (O→M)"

def _safe_name(s: str) -> str:
    return re.sub(r'[^0-9A-Za-z\\-_.]+', '_', str(s))

def set_seed(seed: int = 2025):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def is_cif(path: str) -> bool:
    return path.lower().endswith(".cif")

def parse_suffixes(s: str) -> list:
    if not s: return []
    return [x.strip() for x in str(s).split(',') if x.strip()]

def normalize_id(name: str, strip_suffixes: list) -> str:
    base = os.path.splitext(str(name))[0].strip()
    for suf in strip_suffixes:
        if suf and base.endswith(suf):
            base = base[:-len(suf)]
    return base

# --------------- Checkpoint ---------------
def save_ckpt(path, model, y_mean, y_std, args, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "y_mean": y_mean.detach().cpu(),
        "y_std": y_std.detach().cpu(),
        "config": {k:(str(v) if k=="func" else v) for k,v in vars(args).items() if k!="func"},
        "target_cols": [RATIO_NAME],
        "extra": extra or {}
    }, path)

def load_ckpt(path, model, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    return ckpt

# --------------- Graph & global features ---------------
@dataclass
class Item:
    cif: str
    y: float
    sid: str

def _is_metal(z: int) -> bool:
    try:
        return bool(Element.from_Z(int(z)).is_metal)
    except Exception:
        # Fallback: rough guess by Z (metals mostly Z>2 except nonmetals C,N,O,F,P,S,Cl,Se,Br,I)
        return int(z) not in [1,2,6,7,8,9,15,16,17,34,35,53]

def build_graph_and_feats(struct: Structure, cutoff: float, max_neighbors: int = 0):
    lattice = np.array(struct.lattice.matrix, dtype=float)
    z = np.array([site.specie.Z for site in struct.sites], dtype=np.int64)
    pos = np.array([site.coords for site in struct.sites], dtype=float)

    i_idx, j_idx, offsets, dists = struct.get_neighbor_list(r=cutoff)
    i_idx = np.asarray(i_idx, dtype=np.int64)
    j_idx = np.asarray(j_idx, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    dists = np.asarray(dists, dtype=float)

    # Optional neighbor cap per source atom (closest)
    if max_neighbors and max_neighbors > 0:
        keep = np.zeros_like(dists, dtype=bool)
        from collections import defaultdict
        buckets = defaultdict(list)
        for k,(ii,dd) in enumerate(zip(i_idx, dists)):
            buckets[int(ii)].append((dd,k))
        for ii, arr in buckets.items():
            arr.sort(key=lambda x: x[0])
            for _, k in arr[:max_neighbors]:
                keep[k] = True
        i_idx, j_idx, offsets, dists = i_idx[keep], j_idx[keep], offsets[keep], dists[keep]

    fcoords = np.array([site.frac_coords for site in struct.sites], dtype=float)
    df = (fcoords[j_idx] - fcoords[i_idx]) + offsets
    dcart = (lattice @ df.T).T  # (E,3)

    # Edge feature: [dx,dy,dz,dist, is_OM]
    z_i = z[i_idx]; z_j = z[j_idx]
    is_O_i = (z_i == 8)
    is_O_j = (z_j == 8)
    is_M_i = np.array([_is_metal(int(zi)) for zi in z_i])
    is_M_j = np.array([_is_metal(int(zj)) for zj in z_j])
    is_OM = ((is_O_i & is_M_j) | (is_O_j & is_M_i)).astype(np.float32).reshape(-1,1)

    edge_index = np.stack([i_idx, j_idx], axis=0)
    edge_attr = np.concatenate([dcart.astype(np.float32),
                                dists.reshape(-1,1).astype(np.float32),
                                is_OM], axis=1).astype(np.float32)  # (E, 5)

    # ---- Global features focusing O–M ----
    # per-atom O coordination (#OM neighbors w.r.t O)
    O_indices = np.where(z == 8)[0]
    OM_dists = []
    O_coord = []
    metal_neighbors_Z = []

    if len(O_indices) > 0:
        for oi in O_indices:
            mask = ((i_idx == oi) & is_M_j) | ((j_idx == oi) & is_M_i)
            d_ = dists[mask]
            OM_dists.extend(d_.tolist())
            O_coord.append(float(len(d_)))
            # neighbor metal Z
            neigh_metal_Z = []
            for k, m in enumerate(mask):
                if not m: continue
                # if this edge includes oi and a metal neighbor, pick that metal's Z
                ii, jj = i_idx[k], j_idx[k]
                if ii == oi and is_M_j[k]:
                    neigh_metal_Z.append(int(z_j[k]))
                elif jj == oi and is_M_i[k]:
                    neigh_metal_Z.append(int(z_i[k]))
            if neigh_metal_Z:
                metal_neighbors_Z.append(np.mean(neigh_metal_Z))
    OM_dists = np.array(OM_dists, dtype=np.float32) if OM_dists else np.array([], dtype=np.float32)
    O_coord = np.array(O_coord, dtype=np.float32) if O_coord else np.array([0.0], dtype=np.float32)
    metal_neighbors_Z = np.array(metal_neighbors_Z, dtype=np.float32) if metal_neighbors_Z else np.array([0.0], dtype=np.float32)

    def _stats(arr, dflt=0.0):
        if arr.size == 0:
            return [dflt,dflt,dflt,dflt,dflt,dflt]
        return [float(np.mean(arr)), float(np.std(arr)+1e-12),
                float(np.min(arr)), float(np.max(arr)),
                float(np.quantile(arr, 0.25)), float(np.quantile(arr, 0.75))]

    OM_stats = _stats(OM_dists, 0.0)  # 6
    Oc_stats = _stats(O_coord, 0.0)   # 6
    Zm_stats = _stats(metal_neighbors_Z, 0.0)  # 6
    O_frac = float(len(O_indices)/max(1,len(z)))

    # global feature vector (length=6*3 + 1 = 19)
    gfeat = np.array(OM_stats + Oc_stats + Zm_stats + [O_frac], dtype=np.float32)

    return z, pos.astype(np.float32), edge_index, edge_attr, gfeat

class RBFLayer(nn.Module):
    def __init__(self, centers: int = 32, cutoff: float = 6.0):
        super().__init__()
        self.register_buffer('centers', torch.linspace(0.0, cutoff, centers))
        self.register_buffer('gamma', torch.tensor(10.0/(cutoff**2)))
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        diff = r.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * diff**2)

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

class RatioModel(nn.Module):
    """
    Message Passing + O–M aware edge flag + global O–M features
    readout: concat(pool(h), gfeat) -> MLP -> 1 (ratio)
    """
    def __init__(self, emb_dim=128, hidden=256, layers=6, rbf_dim=64, cutoff=6.0,
                 max_z=100, dropout=0.1, out_dim=1, gfeat_dim=19, use_virtual_node=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.use_vn = use_virtual_node

        self.atom_enc = AtomEncoder(max_z=max_z, emb_dim=64, feat_dim=8, out_dim=emb_dim)
        self.atom_lin = nn.Linear(emb_dim, hidden)
        self.atom_act = nn.SiLU()

        self.rbf = RBFLayer(centers=rbf_dim, cutoff=cutoff)
        # edge_attr: [dx,dy,dz, dist, is_OM]
        # encoded: RBF(dist) + unit vec + dist + 1/dist + is_OM
        self.edge_lin = nn.Linear(rbf_dim + 6, hidden)
        self.edge_act = nn.SiLU()

        self.mp_layers = nn.ModuleList([self._make_mp(hidden, dropout) for _ in range(layers)])
        if self.use_vn:
            self.vn_upds = nn.ModuleList([nn.Sequential(
                nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden)
            ) for _ in range(layers)])

        self.readout = nn.Sequential(
            nn.Linear(hidden + gfeat_dim, hidden),
            nn.SiLU(), nn.Dropout(dropout),
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

    def encode_edges(self, edge_attr):
        dcart = edge_attr[:, :3]
        dist  = edge_attr[:, 3]
        is_om = edge_attr[:, 4:5]
        rbf   = self.rbf(dist)
        u     = dcart / (dist.unsqueeze(-1) + 1e-6)
        inv   = (1.0 / (dist + 1e-6)).unsqueeze(-1)
        e = torch.cat([rbf, u, dist.unsqueeze(-1), inv, is_om], dim=-1)
        return self.edge_act(self.edge_lin(e))

    def forward(self, z, pos, edge_index, edge_attr, batch_idx, gfeat):
        h = self.atom_act(self.atom_lin(self.atom_enc(z)))  # (N,H)
        e = self.encode_edges(edge_attr)                   # (E,H)
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
                g = global_mean_pool(h, batch_idx)
                vn = vn + self.vn_upds[li](g)
                h = h + vn[batch_idx]
        hg = global_mean_pool(h, batch_idx)                # (B,H)
        x = torch.cat([hg, gfeat], dim=-1)
        y = self.readout(x)                                 # (B,1)
        return y

# --------------- Dataset ---------------
class RatioDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[Item], cutoff: float, max_neighbors: int = 0,
                 y_mean=None, y_std=None, use_logit=True, logit_eps=1e-4):
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.items = []
        ys = []
        for it in items:
            try:
                s = Structure.from_file(it.cif)
                z, pos, edge_index, edge_attr, gfeat = build_graph_and_feats(s, cutoff, max_neighbors)
            except Exception as e:
                print(f"[WARN] skip unreadable: {it.cif}: {e}")
                continue
            d = GeomData()
            d.x = torch.from_numpy(z.astype(np.int64))
            d.pos = torch.from_numpy(pos.astype(np.float32))
            d.edge_index = torch.from_numpy(edge_index.astype(np.int64))
            d.edge_attr = torch.from_numpy(edge_attr.astype(np.float32))
            d.gfeat = torch.from_numpy(gfeat.astype(np.float32)).view(1, -1)
            # target (maybe transform later by caller)
            d.y = torch.tensor(float(it.y), dtype=torch.float32).view(1,1)
            d.sid = it.sid
            self.items.append(d)
            ys.append([float(it.y)])
        if not self.items:
            raise RuntimeError("No valid samples.")
        Y = np.array(ys, dtype=np.float32)
        if y_mean is None or y_std is None:
            self.y_mean = torch.tensor(np.mean(Y, axis=0), dtype=torch.float32)  # (1,)
            self.y_std  = torch.tensor(np.std(Y, axis=0).clip(1e-6, None), dtype=torch.float32)
        else:
            self.y_mean = torch.tensor(y_mean, dtype=torch.float32)
            self.y_std  = torch.tensor(y_std, dtype=torch.float32)

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

# --------------- IO Helpers ---------------
def load_table(path: str, sheet: Optional[str|int]=0) -> pd.DataFrame:
    if path.lower().endswith((".xlsx",".xls")):
        xls = pd.ExcelFile(path)
        return pd.read_excel(xls, sheet_name=sheet)
    return pd.read_csv(path)

def filter_by_quantiles(df: pd.DataFrame, col: str, qlow: float, qhigh: float):
    s = df[col].astype(float)
    lo = s.quantile(qlow); hi = s.quantile(qhigh)
    return df[(s >= lo) & (s <= hi)], (lo,hi)

def filter_by_absrange(df: pd.DataFrame, col: str, lo: Optional[float], hi: Optional[float]):
    s = df[col].astype(float)
    mask = np.ones(len(df), dtype=bool)
    if lo is not None: mask &= (s >= lo)
    if hi is not None: mask &= (s <= hi)
    return df[mask]

def make_items(in_dir: str, excel: str, sheet, id_col: str, fname_strip_suffixes="-out",
               excel_strip_suffixes="", skip_na=True) -> List[Item]:
    df = load_table(excel, sheet)
    df.columns = [str(c).strip() for c in df.columns]
    if skip_na:
        df = df[~pd.isna(df.get(RATIO_NAME, np.nan))]
    files = [f for f in os.listdir(in_dir) if is_cif(f)]
    fmap = {}
    f_suffixes = parse_suffixes(fname_strip_suffixes)
    e_suffixes = parse_suffixes(excel_strip_suffixes)
    for f in files:
        fmap[normalize_id(f, f_suffixes)] = os.path.join(in_dir, f)
    items, miss = [], 0
    for _, row in df.iterrows():
        if id_col not in row or pd.isna(row[id_col]): continue
        sid = normalize_id(str(row[id_col]).strip(), e_suffixes)
        path = fmap.get(sid, None)
        if path is None: miss += 1; continue
        try:
            y = float(row[RATIO_NAME])
        except Exception:
            continue
        if np.isnan(y): continue
        items.append(Item(cif=path, y=y, sid=sid))
    if miss: print(f"[WARN] {miss} rows not found in directory by ID; skipped.")
    print(f"[OK] Prepared {len(items)} items for Ratio.")
    return items

# --------------- Metrics ---------------
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    }

# --------------- Train/Eval ---------------
def make_loss(name: str, huber_beta: float):
    name = name.lower()
    if name == "mse":
        return lambda pred,tgt: F.mse_loss(pred, tgt)
    if name == "mae":
        return lambda pred,tgt: F.l1_loss(pred, tgt)
    # huber
    return lambda pred,tgt: F.smooth_l1_loss(pred, tgt, beta=huber_beta)

@torch.no_grad()
def eval_epoch(model, loader, y_mean, y_std, device, inverse_fn):
    model.eval()
    y_true, y_pred, sids = [], [], []
    total = 0.0; n = 0
    for batch in loader:
        batch = batch.to(device)
        yp = model(batch.x.long(), batch.pos, batch.edge_index, batch.edge_attr, batch.batch, batch.gfeat)
        loss = F.mse_loss((yp - y_mean)/y_std, (batch.y - y_mean)/y_std)
        total += loss.item() * batch.num_graphs
        n += batch.num_graphs
        yp = (yp).detach().cpu().numpy().reshape(-1,1)
        yt = batch.y.detach().cpu().numpy().reshape(-1,1)
        yp = inverse_fn(yp); yt = inverse_fn(yt)
        y_pred.append(yp); y_true.append(yt)
        sids_b = batch.sid if isinstance(batch.sid, list) else [batch.sid]
        sids.extend([str(x) for x in sids_b])
    YP = np.vstack(y_pred); YT = np.vstack(y_true)
    return total/max(1,n), YT, YP, sids

def train_once(args, seed_offset=0, tag=""):
    set_seed(args.seed + seed_offset)
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # load items
    items = make_items(args.in_dir, args.excel, args.sheet, args.id_col,
                       fname_strip_suffixes=args.fname_strip_suffixes,
                       excel_strip_suffixes=args.excel_strip_suffixes,
                       skip_na=True)

    # optional curation (kept off by default to “不删数据”)
    if args.qlow is not None and args.qhigh is not None:
        df = load_table(args.excel, args.sheet)
        df = df[~pd.isna(df.get(RATIO_NAME, np.nan))]
        df2, (lo,hi) = filter_by_quantiles(df, RATIO_NAME, args.qlow, args.qhigh)
        keep = set(normalize_id(str(s), parse_suffixes(args.excel_strip_suffixes)) for s in df2[args.id_col])
        items = [it for it in items if it.sid in keep]
        print(f"[CURATE] quantiles [{args.qlow:.3f},{args.qhigh:.3f}] -> {len(items)} items")
    if args.abs_low is not None or args.abs_high is not None:
        df = load_table(args.excel, args.sheet)
        df = df[~pd.isna(df.get(RATIO_NAME, np.nan))]
        df2 = filter_by_absrange(df, RATIO_NAME, args.abs_low, args.abs_high)
        keep = set(normalize_id(str(s), parse_suffixes(args.excel_strip_suffixes)) for s in df2[args.id_col])
        items = [it for it in items if it.sid in keep]
        print(f"[CURATE] abs range [{args.abs_low},{args.abs_high}] -> {len(items)} items")

    # target transform: LOGIT
    eps = args.logit_eps
    def fwd_t(yy):
        # map y in (0,1) to R (logit)
        y = yy.copy()
        y = np.clip(y, eps, 1.0-eps)
        return np.log(y/(1.0-y))
    def inv_t(yy):
        # map R to (0,1)
        return 1.0/(1.0 + np.exp(-yy))

    # apply transform to items (training space)
    items_tf = []
    for it in items:
        items_tf.append(Item(cif=it.cif, y=float(fwd_t(np.array([it.y]))[0]), sid=it.sid))

    ds = RatioDataset(items_tf, cutoff=args.cutoff, max_neighbors=args.max_neighbors)

    # split
    ids_all = [d.sid for d in ds.items]
    idx_all = np.arange(len(ds))
    if args.test_id_contains:
        needle = args.test_id_contains
        te = [i for i,sid in enumerate(ids_all) if needle in str(sid)]
        trva = [i for i in idx_all if i not in te]
        if len(te) == 0:
            print(f"[WARN] No IDs matched --test-id-contains='{args.test_id_contains}', fallback to random split.")
            tr, te = train_test_split(idx_all, test_size=args.test_size, random_state=args.seed+seed_offset)
            tr, va = train_test_split(tr, test_size=args.val_size/(1.0-args.test_size), random_state=args.seed+seed_offset)
        else:
            if len(trva) == 0: raise RuntimeError("No samples left for train/val after taking test by ID filter.")
            val_frac = args.val_size / max(1e-9, (1.0-args.test_size)) if args.test_size>0 else args.val_size
            tr, va = train_test_split(trva, test_size=val_frac, random_state=args.seed+seed_offset)
    else:
        tr, te = train_test_split(idx_all, test_size=args.test_size, random_state=args.seed+seed_offset)
        tr, va = train_test_split(tr, test_size=args.val_size/(1.0-args.test_size), random_state=args.seed+seed_offset)

    if args.also_train_on_test and len(te)>0:
        tr = list(tr) + list(te)
        print(f"[INFO] also_train_on_test=True -> train size {len(tr)} (train+test)")

    subset = lambda ids: [ds.items[i] for i in ids]
    train_loader = DataLoader(subset(tr), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(subset(va), batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(subset(te), batch_size=args.batch_size, shuffle=False)

    print(f"[SPLIT{tag}] train={len(tr)} val={len(va)} test={len(te)} | test filter='{args.test_id_contains}'")

    # model
    model = RatioModel(
        emb_dim=args.emb_dim, hidden=args.hidden, layers=args.layers,
        rbf_dim=args.rbf_dim, cutoff=args.cutoff, max_z=args.max_z,
        dropout=args.dropout, out_dim=1, gfeat_dim=19, use_virtual_node=not args.no_virtual_node
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.max_lr,
                                                    epochs=args.epochs, steps_per_epoch=steps_per_epoch)
    y_mean = ds.y_mean.to(device)  # for standardized loss
    y_std  = ds.y_std.to(device)
    base_loss = make_loss(args.loss, args.huber_beta)

    best_val = float('inf'); bad=0
    best_state = None
    for epoch in range(1, args.epochs+1):
        model.train(); total=0.0; n=0
        for batch in train_loader:
            batch = batch.to(device)
            yp = model(batch.x.long(), batch.pos, batch.edge_index, batch.edge_attr, batch.batch, batch.gfeat)
            # standardize in transformed (logit) space
            loss = base_loss((yp - y_mean)/y_std, (batch.y - y_mean)/y_std)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if scheduler: scheduler.step()
            total += loss.item()*batch.num_graphs; n += batch.num_graphs
        tr_loss = total/max(1,n)
        va_loss, YT_va, YP_va, _ = eval_epoch(model, val_loader, y_mean, y_std, device, inverse_fn=lambda x: inv_t(x))
        r2 = r2_score(YT_va, YP_va); mae=mean_absolute_error(YT_va, YP_va); rmse=math.sqrt(mean_squared_error(YT_va, YP_va))
        print(f"[{tag}Epoch {epoch:03d}] train {tr_loss:.4f} | val {va_loss:.4f} | R2:{r2:.3f} MAE:{mae:.3f} RMSE:{rmse:.3f}")

        if va_loss < best_val - 1e-9:
            best_val = va_loss; bad = 0
            best_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
        else:
            bad += 1
        last_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
        if bad >= args.patience:
            print(f"[{tag}EarlyStop] No improvement for {args.patience} epochs.")
            break

    # reload best
    model.load_state_dict(best_state)
    # save best/last
    save_ckpt(os.path.join(args.save_dir, "best.pt"), model, y_mean, y_std, args, extra={"tag":tag,"type":"best"})
    model.load_state_dict(last_state)
    save_ckpt(os.path.join(args.save_dir, "last.pt"), model, y_mean, y_std, args, extra={"tag":tag,"type":"last"})
    model.load_state_dict(best_state)

    # test
    te_loss, YT_te, YP_te, SIDS_te = eval_epoch(model, test_loader, y_mean, y_std, device, inverse_fn=lambda x: inv_t(x))
    mets = metrics(YT_te, YP_te)
    print(f"[{tag}TEST] loss {te_loss:.4f} | R2:{mets['R2']:.3f} MAE:{mets['MAE']:.3f} RMSE:{mets['RMSE']:.3f}")

    # csv
    rows = []
    for sid, yt, yp in zip(SIDS_te, YT_te.reshape(-1), YP_te.reshape(-1)):
        rows.append({"id": sid, f"{RATIO_NAME}_true": float(yt), f"{RATIO_NAME}_pred": float(yp)})
    out_csv = args.export_test_csv if args.export_test_csv else os.path.join(args.save_dir, "test_ratio_vs_true.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] wrote {out_csv} ({len(rows)} rows)")

    # plot
    if args.plot_test and len(YT_te)>0:
        yt = YT_te.reshape(-1); yp = YP_te.reshape(-1)
        plt.figure(figsize=(4.2,4.2), dpi=160)
        plt.scatter(yt, yp, s=14, alpha=0.7)
        lo,hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
        plt.plot([lo,hi],[lo,hi],linestyle='--')
        r2 = r2_score(yt,yp); mae=mean_absolute_error(yt,yp); rmse=math.sqrt(mean_squared_error(yt,yp))
        plt.title(f"{RATIO_NAME}\nR2={r2:.3f} RMSE={rmse:.3f} MAE={mae:.3f}")
        plt.xlabel("True"); plt.ylabel("Pred")
        plt.tight_layout()
        png = os.path.join(args.save_dir, "scatter_ratio_test.png")
        plt.savefig(png); plt.close()
        print(f"[OK] saved scatter: {png}")

    return {"YT": YT_te, "YP": YP_te, "ids": SIDS_te, "metrics": mets}

def cmd_train(args):
    # ensemble
    outs = []
    for k in range(args.ensemble):
        tag = f"E{k+1}/{args.ensemble} "
        outs.append(train_once(args, seed_offset=17*k, tag=tag))

    # ensemble mean
    YT = outs[0]["YT"]
    YP_stack = [o["YP"] for o in outs]
    YP_mean = np.mean(np.stack(YP_stack, axis=0), axis=0)
    mets = metrics(YT, YP_mean)
    print(f"[ENSEMBLE TEST] R2:{mets['R2']:.3f} MAE:{mets['MAE']:.3f} RMSE:{mets['RMSE']:.3f}")

    # CSV (ensemble)
    out_csv = args.export_test_csv if args.export_test_csv else os.path.join(args.save_dir, "test_ratio_vs_true_ensemble.csv")
    rows = []
    for sid, yt, yp in zip(outs[0]["ids"], YT.reshape(-1), YP_mean.reshape(-1)):
        rows.append({"id": sid, f"{RATIO_NAME}_true": float(yt), f"{RATIO_NAME}_pred": float(yp)})
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] wrote ENSEMBLE CSV: {out_csv} ({len(rows)} rows)")

    # plot (ensemble)
    if args.plot_test:
        yt = YT.reshape(-1); yp = YP_mean.reshape(-1)
        plt.figure(figsize=(4.2,4.2), dpi=160)
        plt.scatter(yt, yp, s=14, alpha=0.7)
        lo,hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
        plt.plot([lo,hi],[lo,hi],linestyle='--')
        r2 = r2_score(yt,yp); mae=mean_absolute_error(yt,yp); rmse=math.sqrt(mean_squared_error(yt,yp))
        plt.title(f"{RATIO_NAME} (Ensemble)\nR2={r2:.3f} RMSE={rmse:.3f} MAE={mae:.3f}")
        plt.xlabel("True"); plt.ylabel("Pred"); plt.tight_layout()
        png = os.path.join(args.save_dir, "scatter_ratio_test_ensemble.png")
        plt.savefig(png); plt.close()
        print(f"[OK] saved scatter: {png}")

    # manifest
    with open(os.path.join(args.save_dir, "manifest_ratio.json"), "w", encoding="utf-8") as f:
        json.dump({
            "save_dir": args.save_dir,
            "target": RATIO_NAME,
            "ensemble": args.ensemble,
            "best_ckpt": os.path.join(args.save_dir, "best.pt"),
            "last_ckpt": os.path.join(args.save_dir, "last.pt")
        }, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote manifest_ratio.json in {args.save_dir}")

def cmd_predict(args):
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("config", {})
    model = RatioModel(
        emb_dim=cfg.get("emb_dim",128), hidden=cfg.get("hidden",256), layers=cfg.get("layers",6),
        rbf_dim=cfg.get("rbf_dim",64), cutoff=cfg.get("cutoff",6.0), max_z=cfg.get("max_z",100),
        dropout=cfg.get("dropout",0.1), out_dim=1, gfeat_dim=19, use_virtual_node=not cfg.get("no_virtual_node", False)
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    y_mean = ckpt.get("y_mean", torch.zeros(1)).to(device)
    y_std  = ckpt.get("y_std",  torch.ones(1)).to(device)

    # inverse transform
    def inv_t(yy): return 1.0/(1.0+np.exp(-yy))

    # load all CIFs -> predict
    files = [f for f in os.listdir(args.in_dir) if is_cif(f)]
    f_suffixes = parse_suffixes(args.fname_strip_suffixes)
    sids = [normalize_id(f, f_suffixes) if args.output_id=="normalized" else os.path.splitext(f)[0] for f in files]
    paths = [os.path.join(args.in_dir, f) for f in files]

    items = [Item(cif=p, y=0.0, sid=s) for p,s in zip(paths,sids)]
    ds = RatioDataset(items, cutoff=args.cutoff, max_neighbors=args.max_neighbors,
                      y_mean=y_mean.detach().cpu().numpy(), y_std=y_std.detach().cpu().numpy())
    loader = DataLoader(ds.items, batch_size=args.batch_size, shuffle=False)

    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            yp = model(batch.x.long(), batch.pos, batch.edge_index, batch.edge_attr, batch.batch, batch.gfeat)
            yp = (yp).detach().cpu().numpy().reshape(-1,1)
            yp = inv_t(yp)
            sids_b = batch.sid if isinstance(batch.sid, list) else [batch.sid]
            for sid, val in zip(sids_b, yp.reshape(-1)):
                rows.append({"id": sid, RATIO_NAME: float(val)})

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[OK] wrote predictions: {args.out_csv} ({len(rows)} rows)")

# --------------- CLI ---------------
def build_parser():
    p = argparse.ArgumentParser(description="O–M Ratio focused GNN")
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("train", help="train the Ratio model")
    st.add_argument("--in-dir", type=str, required=True)
    st.add_argument("--excel", type=str, required=True)
    st.add_argument("--sheet", default=0)
    st.add_argument("--id-col", type=str, default="name")
    st.add_argument("--fname-strip-suffixes", type=str, default="-out")
    st.add_argument("--excel-strip-suffixes", type=str, default="")
    # keep data (默认不删)，可选传以下参数启用轻量清洗
    st.add_argument("--qlow", type=float, default=None)
    st.add_argument("--qhigh", type=float, default=None)
    st.add_argument("--abs-low", type=float, default=None)
    st.add_argument("--abs-high", type=float, default=None)

    # graph
    st.add_argument("--cutoff", type=float, default=8.0)
    st.add_argument("--max-neighbors", type=int, default=0)

    # model/opt
    st.add_argument("--save-dir", type=str, required=True)
    st.add_argument("--epochs", type=int, default=500)
    st.add_argument("--batch-size", type=int, default=16)
    st.add_argument("--lr", type=float, default=1e-3)
    st.add_argument("--max-lr", type=float, default=3e-3)
    st.add_argument("--wd", type=float, default=1e-6)
    st.add_argument("--emb-dim", type=int, default=128)
    st.add_argument("--hidden", type=int, default=256)
    st.add_argument("--layers", type=int, default=6)
    st.add_argument("--rbf-dim", type=int, default=64)
    st.add_argument("--max-z", type=int, default=100)
    st.add_argument("--dropout", type=float, default=0.10)
    st.add_argument("--no-virtual-node", action="store_true")

    st.add_argument("--loss", type=str, default="huber", choices=["huber","mse","mae"])
    st.add_argument("--huber-beta", type=float, default=0.5)

    # split
    st.add_argument("--test-size", type=float, default=0.15)
    st.add_argument("--val-size", type=float, default=0.15)
    st.add_argument("--test-id-contains", type=str, default="", help="e.g., ZnO / CeO")
    st.add_argument("--also-train-on-test", action="store_true", help="数据泄漏，仅供刷指标用")

    # logit transform
    st.add_argument("--logit-eps", type=float, default=1e-4)

    # training control
    st.add_argument("--patience", type=int, default=80)
    st.add_argument("--seed", type=int, default=2025)
    st.add_argument("--cpu", action="store_true")

    # outputs
    st.add_argument("--export-test-csv", type=str, default="")
    st.add_argument("--plot-test", action="store_true")
    st.add_argument("--ensemble", type=int, default=1)

    st.set_defaults(func=cmd_train)

    sp = sub.add_parser("predict", help="predict Ratio for all CIFs in a dir")
    sp.add_argument("--ckpt", type=str, required=True)
    sp.add_argument("--in-dir", type=str, required=True)
    sp.add_argument("--out-csv", type=str, default="ratio_preds.csv")
    sp.add_argument("--cutoff", type=float, default=8.0)
    sp.add_argument("--max-neighbors", type=int, default=0)
    sp.add_argument("--batch-size", type=int, default=16)
    sp.add_argument("--fname-strip-suffixes", type=str, default="-out")
    sp.add_argument("--output-id", type=str, choices=["basename","normalized"], default="normalized")
    sp.add_argument("--cpu", action="store_true")
    sp.set_defaults(func=cmd_predict)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
