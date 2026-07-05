#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regress_run_singleview_strong.py

Single-structure strong training script:
- addH structure embedding only (single-structure route)
- strong fusion of graph + text + metadata
- default frozen text encoder
- optional light unfreezing of top text layers
- auxiliary tasks:
    * site_type classification
    * family classification
    * target_bin classification
- weighted sampling for small / imbalanced datasets
- fold-wise target standardization

Reads regress_train.yml, similar to your existing pipeline.

Recommended regress_train.yml fields
------------------------------------
run_name: addH_singleview_strong_fold0_seed42
train_path: /path/to/fold_0/nn_train.pkl
val_path: /path/to/fold_0/nn_val.pkl
ckpt_save_path: /path/to/ckpts
device: cuda
num_epochs: 40
batch_size: 16
loss_fn: SmoothL1Loss
standardize_target: true
seed: 42

# text
pretrain_ckpt: roberta-base
max_length: 256
use_text_raw: true
use_text_structured: true
freeze_text_encoder: true
unfreeze_top_n_layers: 0

# model dims
projection_dim: 256
meta_hidden_dim: 128
dropout: 0.15

# learning rates
lr_main: 2.0e-5
lr_text_projection: 2.0e-6
lr_text_top: 8.0e-7
weight_decay: 0.01

# auxiliary losses
aux_weight_site: 0.10
aux_weight_family: 0.05
aux_weight_target_bin: 0.05

# sampler
use_weighted_sampler: true
sampler_weight_site: 1.0
sampler_weight_target_bin: 1.0

# wandb
wandb_mode: disabled
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
import wandb
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")


# ---------------------------
# utilities
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_loss_fn(name: str):
    if name == "MSELoss":
        return nn.MSELoss()
    if name == "L1Loss":
        return nn.L1Loss()
    if name == "SmoothL1Loss":
        return nn.SmoothL1Loss()
    raise ValueError(f"Unsupported loss_fn: {name}")


@dataclass
class TargetScaler:
    mean: float = 0.0
    std: float = 1.0
    enabled: bool = True

    def fit(self, y: np.ndarray):
        if not self.enabled:
            self.mean = 0.0
            self.std = 1.0
            return self
        self.mean = float(np.mean(y))
        std = float(np.std(y))
        self.std = std if std > 1e-8 else 1.0
        return self

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return y
        return (y - self.mean) / self.std

    def inverse_torch(self, y: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return y
        return y * self.std + self.mean


# ---------------------------
# metadata preprocessor
# ---------------------------
class MetaPreprocessor:
    def __init__(self, cat_cols: Sequence[str], num_cols: Sequence[str]):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.cat_vocab: Dict[str, Dict[str, int]] = {}
        self.num_mean: Dict[str, float] = {}
        self.num_std: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame):
        for c in self.cat_cols:
            vals = df[c].fillna("unknown").astype(str).tolist()
            uniq = ["<UNK>"] + sorted(set(vals))
            self.cat_vocab[c] = {v: i for i, v in enumerate(uniq)}
        for c in self.num_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            mean = float(s.mean()) if s.notna().any() else 0.0
            std = float(s.std()) if s.notna().any() else 1.0
            if std <= 1e-8 or math.isnan(std):
                std = 1.0
            if math.isnan(mean):
                mean = 0.0
            self.num_mean[c] = mean
            self.num_std[c] = std
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.cat_cols:
            vocab = self.cat_vocab[c]
            out[c] = out[c].fillna("unknown").astype(str).map(lambda x: vocab.get(x, 0)).astype(int)
        for c in self.num_cols:
            s = pd.to_numeric(out[c], errors="coerce").fillna(self.num_mean.get(c, 0.0))
            out[c] = ((s - self.num_mean.get(c, 0.0)) / self.num_std.get(c, 1.0)).astype(np.float32)
        return out

    def state_dict(self) -> dict:
        return {
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
            "cat_vocab": self.cat_vocab,
            "num_mean": self.num_mean,
            "num_std": self.num_std,
        }

    @classmethod
    def from_state_dict(cls, state: dict):
        obj = cls(state["cat_cols"], state["num_cols"])
        obj.cat_vocab = state["cat_vocab"]
        obj.num_mean = state["num_mean"]
        obj.num_std = state["num_std"]
        return obj


def infer_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    reserved = {
        "id", "target", "eq_emb", "text_raw", "text_structured",
        "site_type_label", "family_label", "target_bin",
    }
    cat_cols: List[str] = []
    num_cols: List[str] = []
    aux_cols: List[str] = [c for c in ["site_type_label", "family_label", "target_bin"] if c in df.columns]

    for c in df.columns:
        if c in reserved:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    return cat_cols, num_cols, aux_cols


# ---------------------------
# dataset
# ---------------------------
class SingleviewStrongDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        cat_cols: Sequence[str],
        num_cols: Sequence[str],
        use_text_raw: bool = True,
        use_text_structured: bool = True,
        max_length: int = 256,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.use_text_raw = bool(use_text_raw)
        self.use_text_structured = bool(use_text_structured)
        self.max_length = int(max_length)

        self.eq = [np.asarray(x, dtype=np.float32).reshape(-1) for x in self.df["eq_emb"].values]
        dims = sorted(set(x.shape[0] for x in self.eq))
        if len(dims) != 1:
            raise ValueError(f"Inconsistent eq_emb dims: {dims}")
        self.graph_dim = int(dims[0])

    def __len__(self):
        return len(self.df)

    def _compose_text(self, row: pd.Series) -> str:
        parts = []
        if self.use_text_structured and "text_structured" in row and pd.notna(row["text_structured"]):
            parts.append(str(row["text_structured"]))
        if self.use_text_raw and "text_raw" in row and pd.notna(row["text_raw"]):
            parts.append(str(row["text_raw"]))
        if not parts:
            return ""
        return " </s> ".join(parts)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = self._compose_text(row)
        tok = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )

        item = {
            "id": str(row["id"]),
            "input_ids": torch.tensor(tok["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tok["attention_mask"], dtype=torch.long),
            "graph_embed": torch.tensor(self.eq[idx], dtype=torch.float),
            "target": torch.tensor(float(row["target"]), dtype=torch.float),
        }

        if self.cat_cols:
            cat_vals = np.asarray([int(row[c]) for c in self.cat_cols], dtype=np.int64)
            item["cat_feats"] = torch.tensor(cat_vals, dtype=torch.long)
        else:
            item["cat_feats"] = torch.zeros(0, dtype=torch.long)

        if self.num_cols:
            num_vals = np.asarray([float(row[c]) for c in self.num_cols], dtype=np.float32)
            item["num_feats"] = torch.tensor(num_vals, dtype=torch.float)
        else:
            item["num_feats"] = torch.zeros(0, dtype=torch.float)

        for lab in ["site_type_label", "family_label", "target_bin"]:
            if lab in row.index:
                item[lab] = torch.tensor(int(row[lab]), dtype=torch.long)
            else:
                item[lab] = torch.tensor(-1, dtype=torch.long)

        return item


# ---------------------------
# model
# ---------------------------
class TextEncoder(nn.Module):
    def __init__(self, pretrain_ckpt: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrain_ckpt)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "last_hidden_state"):
            cls = out.last_hidden_state[:, 0, :]
            return cls
        if isinstance(out, tuple):
            return out[0][:, 0, :]
        raise ValueError("Unexpected transformer output structure")


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        h = self.net(x) + self.skip(x)
        return self.norm(h)


class MetaEncoder(nn.Module):
    def __init__(self, cat_cardinalities: Sequence[int], num_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.cat_cardinalities = list(cat_cardinalities)
        self.num_dim = int(num_dim)

        self.embs = nn.ModuleList()
        emb_dim_total = 0
        for n in self.cat_cardinalities:
            emb_dim = min(32, max(4, int(round(math.sqrt(max(n, 2))))))
            self.embs.append(nn.Embedding(int(n), emb_dim))
            emb_dim_total += emb_dim

        in_dim = emb_dim_total + self.num_dim
        if in_dim == 0:
            self.encoder = None
            self.out_dim = 0
        else:
            self.encoder = ProjectionMLP(in_dim, out_dim, dropout=dropout)
            self.out_dim = out_dim

    def forward(self, cat_feats: torch.Tensor, num_feats: torch.Tensor):
        xs = []
        if len(self.embs) > 0 and cat_feats.numel() > 0:
            for i, emb in enumerate(self.embs):
                xs.append(emb(cat_feats[:, i]))
        if num_feats.numel() > 0:
            xs.append(num_feats)
        if not xs:
            return None
        x = torch.cat(xs, dim=-1)
        return self.encoder(x)


class SingleviewStrongModel(nn.Module):
    def __init__(
        self,
        pretrain_ckpt: str,
        graph_dim: int,
        cat_cardinalities: Sequence[int],
        num_dim: int,
        projection_dim: int = 256,
        meta_hidden_dim: int = 128,
        dropout: float = 0.15,
        n_site: int = 0,
        n_family: int = 0,
        n_target_bin: int = 0,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(pretrain_ckpt=pretrain_ckpt)
        hidden_size = int(getattr(self.text_encoder.model.config, "hidden_size", 768))

        self.text_proj = ProjectionMLP(hidden_size, projection_dim, dropout=dropout)
        self.graph_proj = ProjectionMLP(graph_dim, projection_dim, dropout=dropout)
        self.meta_encoder = MetaEncoder(cat_cardinalities=cat_cardinalities, num_dim=num_dim, out_dim=meta_hidden_dim, dropout=dropout)

        self.meta_to_proj = ProjectionMLP(meta_hidden_dim, projection_dim, dropout=dropout) if self.meta_encoder.out_dim > 0 else None

        self.gt_gate = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.Sigmoid(),
        )
        self.gtm_gate = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.Sigmoid(),
        )

        self.regressor = nn.Sequential(
            nn.LayerNorm(projection_dim),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, 1),
        )

        self.site_head = nn.Linear(projection_dim, n_site) if n_site > 1 else None
        self.family_head = nn.Linear(projection_dim, n_family) if n_family > 1 else None
        self.target_bin_head = nn.Linear(projection_dim, n_target_bin) if n_target_bin > 1 else None

    def forward(self, batch):
        text_feat = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        text_h = self.text_proj(text_feat)
        graph_h = self.graph_proj(batch["graph_embed"])

        gt_gate = self.gt_gate(torch.cat([graph_h, text_h], dim=-1))
        gt = gt_gate * graph_h + (1.0 - gt_gate) * text_h

        meta_h = self.meta_encoder(batch["cat_feats"], batch["num_feats"])
        if meta_h is not None and self.meta_to_proj is not None:
            meta_p = self.meta_to_proj(meta_h)
            gtm_gate = self.gtm_gate(torch.cat([gt, meta_p], dim=-1))
            fused = gtm_gate * gt + (1.0 - gtm_gate) * meta_p
        else:
            fused = gt

        out = {
            "pred": self.regressor(fused).squeeze(-1),
            "fused": fused,
        }
        if self.site_head is not None:
            out["site_logits"] = self.site_head(fused)
        if self.family_head is not None:
            out["family_logits"] = self.family_head(fused)
        if self.target_bin_head is not None:
            out["target_bin_logits"] = self.target_bin_head(fused)
        return out


# ---------------------------
# text freezing helpers
# ---------------------------
def set_module_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = bool(flag)


def _get_transformer_layers(backbone: nn.Module):
    candidate_paths = [
        ["encoder", "layer"],
        ["roberta", "encoder", "layer"],
        ["model", "encoder", "layer"],
        ["transformer", "layer"],
        ["transformer", "h"],
        ["encoder", "layers"],
    ]
    for path in candidate_paths:
        cur = backbone
        ok = True
        for p in path:
            if hasattr(cur, p):
                cur = getattr(cur, p)
            else:
                ok = False
                break
        if ok and hasattr(cur, "__len__"):
            return cur
    return None


def apply_text_freeze_policy(model: SingleviewStrongModel, freeze_text_encoder: bool, unfreeze_top_n_layers: int):
    if freeze_text_encoder:
        set_module_requires_grad(model.text_encoder, False)
    else:
        set_module_requires_grad(model.text_encoder, True)

    if freeze_text_encoder and int(unfreeze_top_n_layers) > 0:
        set_module_requires_grad(model.text_encoder, False)
        backbone = model.text_encoder.model
        layers = _get_transformer_layers(backbone)
        if layers is None:
            print("[WARN] Could not locate transformer layers. Unfreezing full text encoder.")
            set_module_requires_grad(model.text_encoder, True)
        else:
            n_layers = len(layers)
            k = min(int(unfreeze_top_n_layers), n_layers)
            for i in range(n_layers - k, n_layers):
                set_module_requires_grad(layers[i], True)
            for attr in ["pooler", "LayerNorm", "layer_norm"]:
                if hasattr(backbone, attr):
                    set_module_requires_grad(getattr(backbone, attr), True)


def build_optimizer(config: dict, model: SingleviewStrongModel):
    lr_main = float(config.get("lr_main", 2e-5))
    lr_text_proj = float(config.get("lr_text_projection", max(lr_main * 0.1, 1e-6)))
    lr_text_top = float(config.get("lr_text_top", 8e-7))
    wd = float(config.get("weight_decay", 0.01))

    text_encoder_params = [p for p in model.text_encoder.parameters() if p.requires_grad]
    text_proj_params = [p for p in model.text_proj.parameters() if p.requires_grad]
    main_modules = [model.graph_proj, model.regressor, model.gt_gate, model.gtm_gate]
    if model.meta_encoder is not None:
        main_modules.append(model.meta_encoder)
    if model.meta_to_proj is not None:
        main_modules.append(model.meta_to_proj)
    if model.site_head is not None:
        main_modules.append(model.site_head)
    if model.family_head is not None:
        main_modules.append(model.family_head)
    if model.target_bin_head is not None:
        main_modules.append(model.target_bin_head)

    main_params = []
    for m in main_modules:
        main_params.extend([p for p in m.parameters() if p.requires_grad])

    param_groups = []
    if main_params:
        param_groups.append({"params": main_params, "lr": lr_main, "weight_decay": wd})
    if text_proj_params:
        param_groups.append({"params": text_proj_params, "lr": lr_text_proj, "weight_decay": wd})
    if text_encoder_params:
        param_groups.append({"params": text_encoder_params, "lr": lr_text_top, "weight_decay": wd})

    if not param_groups:
        raise ValueError("No trainable parameters found.")
    return torch.optim.AdamW(param_groups)


def create_scheduler(config: dict, optimizer, train_size: int):
    schd = config.get("scheduler", "reduceLR")
    epochs = int(config["num_epochs"])
    train_bs = int(config["batch_size"])
    warmup = int(config.get("warmup_steps", 0))
    if schd == "reduceLR":
        return ReduceLROnPlateau(optimizer, mode="min", patience=3)
    train_steps = max(1, int(np.ceil(train_size / max(train_bs, 1))) * epochs)
    return transformers.get_scheduler(
        schd,
        optimizer=optimizer,
        num_warmup_steps=warmup,
        num_training_steps=train_steps,
    )


def build_weighted_sampler(df: pd.DataFrame, config: dict):
    use_weighted = bool(config.get("use_weighted_sampler", True))
    if not use_weighted:
        return None

    w_site = float(config.get("sampler_weight_site", 1.0))
    w_bin = float(config.get("sampler_weight_target_bin", 1.0))

    weights = np.ones(len(df), dtype=np.float64)

    if "site_type_label" in df.columns:
        s = df["site_type_label"].astype(int)
        vc = s.value_counts()
        inv = s.map(lambda x: 1.0 / max(vc.get(x, 1), 1)).to_numpy(dtype=np.float64)
        weights *= np.power(inv, w_site)

    if "target_bin" in df.columns:
        b = df["target_bin"].astype(int)
        vc = b.value_counts()
        inv = b.map(lambda x: 1.0 / max(vc.get(x, 1), 1) if x >= 0 else 1.0).to_numpy(dtype=np.float64)
        weights *= np.power(inv, w_bin)

    weights = weights / weights.mean()
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def summarize_trainable_parameters(model: nn.Module, max_lines: int = 30):
    rows = []
    total = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            total += p.numel()
            rows.append((name, tuple(p.shape), p.numel()))
    print(f"[INFO] trainable parameter tensors = {len(rows)}, params = {total:,}")
    for name, shape, n in rows[:max_lines]:
        print(f"    trainable: {name:70s} shape={shape} n={n}")
    if len(rows) > max_lines:
        print(f"    ... ({len(rows) - max_lines} more trainable tensors)")


# ---------------------------
# training / validation
# ---------------------------
def _aux_ce_loss(logits: torch.Tensor, labels: torch.Tensor):
    valid = labels >= 0
    if logits is None or valid.sum() == 0:
        return torch.tensor(0.0, device=labels.device)
    return nn.functional.cross_entropy(logits[valid], labels[valid])


def train_fn(data_loader, model, optimizer, device, scheduler, reg_loss_fn, scaler: TargetScaler, config: dict, log_interval: int, debug=False):
    model.train()
    lr_list = []
    train_losses = []
    batch_iteration = 0
    print("training...")

    w_site = float(config.get("aux_weight_site", 0.10))
    w_family = float(config.get("aux_weight_family", 0.05))
    w_tbin = float(config.get("aux_weight_target_bin", 0.05))

    for batch in tqdm(data_loader):
        ids = batch.pop("id", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        targets = batch["target"]
        targets_z = scaler.transform(targets)

        optimizer.zero_grad()
        out = model(batch)
        pred_z = out["pred"]

        reg_loss = reg_loss_fn(pred_z, targets_z)
        site_loss = _aux_ce_loss(out.get("site_logits"), batch["site_type_label"])
        family_loss = _aux_ce_loss(out.get("family_logits"), batch["family_label"])
        tbin_loss = _aux_ce_loss(out.get("target_bin_logits"), batch["target_bin"])

        loss = reg_loss + w_site * site_loss + w_family * family_loss + w_tbin * tbin_loss
        loss.backward()
        optimizer.step()

        train_losses.append(float(loss.item()))
        lr_list.append(float(optimizer.param_groups[0]["lr"]))

        if (batch_iteration != 0) and (batch_iteration % log_interval == 0) and (not debug):
            wandb.log({
                "iter_train_loss": float(loss.item()),
                "iter_reg_loss": float(reg_loss.item()),
                "iter_site_loss": float(site_loss.item()),
                "iter_family_loss": float(family_loss.item()),
                "iter_target_bin_loss": float(tbin_loss.item()),
            })

        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        batch_iteration += 1

    return float(np.mean(train_losses)), float(np.mean(lr_list))


def validate_fn(data_loader, model, device, reg_loss_fn, scaler: TargetScaler):
    model.eval()
    val_losses = []
    val_maes = []
    preds_all = []
    tgts_all = []

    with torch.no_grad():
        print("validating...")
        for batch in tqdm(data_loader):
            ids = batch.pop("id", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            pred_z = out["pred"]

            targets = batch["target"]
            targets_z = scaler.transform(targets)
            loss = reg_loss_fn(pred_z, targets_z)

            pred = scaler.inverse_torch(pred_z)
            mae = torch.mean(torch.abs(targets - pred))

            val_losses.append(float(loss.item()))
            val_maes.append(float(mae.item()))
            preds_all.append(pred.detach().cpu().numpy())
            tgts_all.append(targets.detach().cpu().numpy())

    pred = np.concatenate(preds_all, axis=0) if preds_all else np.empty((0,), dtype=float)
    tgt = np.concatenate(tgts_all, axis=0) if tgts_all else np.empty((0,), dtype=float)
    rmse = float(np.sqrt(np.mean((pred - tgt) ** 2))) if len(pred) > 0 else float("nan")
    return float(np.mean(val_losses)), float(np.mean(val_maes)), rmse


def save_checkpoint(path: str, epoch: int, model, optimizer, scheduler, best_loss: float, scaler: TargetScaler, feature_state: dict, config_snapshot: dict):
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_loss": float(best_loss),
            "target_mean": float(scaler.mean),
            "target_std": float(scaler.std),
            "standardize_target": bool(scaler.enabled),
            "feature_state": feature_state,
            "train_config": config_snapshot,
        },
        path,
    )


# ---------------------------
# main
# ---------------------------
def run_regression(config_file: str = "regress_train.yml"):
    config = load_yaml(config_file)
    set_seed(int(config.get("seed", 42)))

    train_path = config["train_path"]
    val_path = config["val_path"]
    ckpt_save_root = Path(config["ckpt_save_path"])
    ckpt_save_root.mkdir(parents=True, exist_ok=True)

    device = config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    batch_size = int(config.get("batch_size", 16))
    loss_name = config.get("loss_fn", "SmoothL1Loss")
    debug = bool(config.get("debug", False))
    if debug:
        device = "cpu"

    run_name = config["run_name"] + datetime.now().strftime("_%m%d_%H%M")
    ckpt_save_dir = ckpt_save_root / run_name
    ckpt_save_dir.mkdir(parents=True, exist_ok=True)

    pretrain_ckpt = config.get("pretrain_ckpt", "roberta-base")
    max_length = int(config.get("max_length", 256))
    use_text_raw = bool(config.get("use_text_raw", True))
    use_text_structured = bool(config.get("use_text_structured", True))
    projection_dim = int(config.get("projection_dim", 256))
    meta_hidden_dim = int(config.get("meta_hidden_dim", 128))
    dropout = float(config.get("dropout", 0.15))
    standardize_target = bool(config.get("standardize_target", True))
    freeze_text_encoder = bool(config.get("freeze_text_encoder", True))
    unfreeze_top_n_layers = int(config.get("unfreeze_top_n_layers", 0))
    num_epochs = int(config.get("num_epochs", 40))
    log_interval = int(config.get("log_interval", 10))

    print("=============================================================")
    print(f"{run_name} is launched")
    print("=============================================================")
    print("Model: singleview strong")
    print(f"pretrain_ckpt: {pretrain_ckpt}")
    print(f"freeze_text_encoder: {freeze_text_encoder}")
    print(f"unfreeze_top_n_layers: {unfreeze_top_n_layers}")
    print(f"projection_dim: {projection_dim}")
    print(f"meta_hidden_dim: {meta_hidden_dim}")
    print(f"dropout: {dropout}")
    print(f"epochs: {num_epochs}")
    print(f"batch_size: {batch_size}")
    print(f"standardize_target: {standardize_target}")
    print("=============================================================")

    wandb_mode = config.get("wandb_mode", os.environ.get("WANDB_MODE", "disabled"))
    if not debug:
        wandb.init(project="singleview-strong", name=run_name, mode=wandb_mode)

    if not debug:
        shutil.copyfile(config_file, ckpt_save_dir / Path(config_file).name)

    train_df = pd.read_pickle(train_path)
    val_df = pd.read_pickle(val_path)

    required = {"id", "target", "eq_emb"}
    if use_text_raw and "text_raw" not in train_df.columns:
        raise ValueError("use_text_raw=True but train dataframe has no text_raw column")
    if use_text_structured and "text_structured" not in train_df.columns:
        raise ValueError("use_text_structured=True but train dataframe has no text_structured column")
    miss = required - set(train_df.columns)
    if miss:
        raise ValueError(f"train dataframe missing required columns: {sorted(miss)}")

    cat_cols, num_cols, aux_cols = infer_feature_columns(train_df)
    # keep only columns that also exist in val
    cat_cols = [c for c in cat_cols if c in val_df.columns]
    num_cols = [c for c in num_cols if c in val_df.columns]

    meta_proc = MetaPreprocessor(cat_cols=cat_cols, num_cols=num_cols).fit(train_df)
    train_df = meta_proc.transform(train_df)
    val_df = meta_proc.transform(val_df)

    # aux class counts
    n_site = int(max(train_df["site_type_label"].max(), val_df["site_type_label"].max()) + 1) if "site_type_label" in train_df.columns else 0
    n_family = int(max(train_df["family_label"].max(), val_df["family_label"].max()) + 1) if "family_label" in train_df.columns else 0
    valid_tb_train = train_df["target_bin"][train_df["target_bin"] >= 0] if "target_bin" in train_df.columns else pd.Series([], dtype=int)
    valid_tb_val = val_df["target_bin"][val_df["target_bin"] >= 0] if "target_bin" in val_df.columns else pd.Series([], dtype=int)
    if len(valid_tb_train) + len(valid_tb_val) > 0:
        n_target_bin = int(max(valid_tb_train.max() if len(valid_tb_train) else -1, valid_tb_val.max() if len(valid_tb_val) else -1) + 1)
    else:
        n_target_bin = 0

    tokenizer = AutoTokenizer.from_pretrained(pretrain_ckpt)
    train_dataset = SingleviewStrongDataset(
        df=train_df,
        tokenizer=tokenizer,
        cat_cols=cat_cols,
        num_cols=num_cols,
        use_text_raw=use_text_raw,
        use_text_structured=use_text_structured,
        max_length=max_length,
    )
    val_dataset = SingleviewStrongDataset(
        df=val_df,
        tokenizer=tokenizer,
        cat_cols=cat_cols,
        num_cols=num_cols,
        use_text_raw=use_text_raw,
        use_text_structured=use_text_structured,
        max_length=max_length,
    )

    cat_cardinalities = [len(meta_proc.cat_vocab[c]) for c in cat_cols]
    model = SingleviewStrongModel(
        pretrain_ckpt=pretrain_ckpt,
        graph_dim=train_dataset.graph_dim,
        cat_cardinalities=cat_cardinalities,
        num_dim=len(num_cols),
        projection_dim=projection_dim,
        meta_hidden_dim=meta_hidden_dim,
        dropout=dropout,
        n_site=n_site,
        n_family=n_family,
        n_target_bin=n_target_bin,
    ).to(device)

    apply_text_freeze_policy(model, freeze_text_encoder=freeze_text_encoder, unfreeze_top_n_layers=unfreeze_top_n_layers)
    summarize_trainable_parameters(model)

    sampler = build_weighted_sampler(train_df, config)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    optimizer = build_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer, len(train_df))
    reg_loss_fn = get_loss_fn(loss_name)
    scaler = TargetScaler(enabled=standardize_target).fit(train_df["target"].astype(float).to_numpy())

    feature_state = {
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "meta_preprocessor": meta_proc.state_dict(),
        "pretrain_ckpt": pretrain_ckpt,
        "max_length": max_length,
        "use_text_raw": use_text_raw,
        "use_text_structured": use_text_structured,
        "projection_dim": projection_dim,
        "meta_hidden_dim": meta_hidden_dim,
        "dropout": dropout,
        "graph_dim": int(train_dataset.graph_dim),
        "cat_cardinalities": cat_cardinalities,
        "n_site": int(n_site),
        "n_family": int(n_family),
        "n_target_bin": int(n_target_bin),
    }
    with (ckpt_save_dir / "feature_state.json").open("w", encoding="utf-8") as f:
        json.dump(feature_state, f, indent=2, ensure_ascii=False)

    best_loss = 999999.0
    best_epoch = -1
    early_stop_threshold = int(config.get("early_stop_threshold", 5))
    early_stopping_counter = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, lr_now = train_fn(train_loader, model, optimizer, device, scheduler, reg_loss_fn, scaler, config, log_interval, debug=debug)
        val_loss, val_mae, val_rmse = validate_fn(val_loader, model, device, reg_loss_fn, scaler)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

        if not debug:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "lr": lr_now,
            })

        if val_loss < best_loss:
            save_checkpoint(
                str(ckpt_save_dir / "checkpoint.pt"),
                epoch,
                model,
                optimizer,
                scheduler,
                best_loss,
                scaler,
                feature_state=feature_state,
                config_snapshot=config,
            )
            print(
                f"Epoch: {epoch}, Train Loss = {round(train_loss, 3)}, "
                f"Val Loss = {round(val_loss, 3)}, Val MAE = {round(val_mae, 3)}, "
                f"Val RMSE = {round(val_rmse, 3)}, checkpoint saved."
            )
            best_loss = val_loss
            best_epoch = epoch
            early_stopping_counter = 0
        else:
            print(
                f"Epoch: {epoch}, Train Loss = {round(train_loss, 3)}, "
                f"Val Loss = {round(val_loss, 3)}, Val MAE = {round(val_mae, 3)}, "
                f"Val RMSE = {round(val_rmse, 3)}"
            )
            early_stopping_counter += 1
            if early_stopping_counter > early_stop_threshold:
                print(f"[INFO] Early stopping triggered at epoch {epoch}")
                break

    print("===== Training Termination =====")
    print(f"[INFO] best_epoch = {best_epoch}, best_val_loss = {best_loss:.6f}")
    if not debug:
        wandb.finish()


if __name__ == "__main__":
    run_regression("regress_train.yml")
