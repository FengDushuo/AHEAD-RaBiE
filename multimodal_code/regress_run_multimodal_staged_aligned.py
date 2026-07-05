#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import shutil
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List

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
from transformers import RobertaTokenizerFast

from model.modules import TextEncoder, ProjectionHead

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")


def _build_text_series(df: pd.DataFrame, preferred_col: str = "text", concat_cols: Optional[List[str]] = None) -> pd.Series:
    if concat_cols is None:
        concat_cols = ["text_structured", "text_raw", "text"]
    concat_cols = [c for c in concat_cols if c in df.columns]
    if preferred_col in df.columns and preferred_col not in concat_cols:
        concat_cols = [preferred_col] + concat_cols
    if not concat_cols:
        raise ValueError("No usable text columns found in dataframe")

    texts = []
    for _, row in df.iterrows():
        parts = []
        for c in concat_cols:
            val = row.get(c, "")
            if pd.isna(val):
                continue
            s = str(val).strip()
            if not s:
                continue
            if s not in parts:
                parts.append(s)
        texts.append(" </s> ".join(parts) if parts else "")
    return pd.Series(texts, index=df.index, name="text")


class MultimodalRegressionDataset(Dataset):
    def __init__(self, texts, targets, graph_emb, tokenizer, seq_len=256, sample_weights=None):
        self.texts = texts
        self.targets = np.asarray(targets, dtype=np.float32)
        self.graph_emb = graph_emb
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.sample_weights = np.asarray(sample_weights, dtype=np.float32) if sample_weights is not None else np.ones(len(self.targets), dtype=np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        tokenized = self.tokenizer(
            text,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        g = np.asarray(self.graph_emb[idx], dtype=np.float32).reshape(-1)
        return {
            "input_ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tokenized["attention_mask"], dtype=torch.long),
            "target": torch.tensor(self.targets[idx], dtype=torch.float),
            "graph_embed": torch.tensor(g, dtype=torch.float),
            "sample_weight": torch.tensor(self.sample_weights[idx], dtype=torch.float),
        }


class GraphProjectionHead(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), int(projection_dim))
        self.gelu = nn.GELU()
        self.fc = nn.Linear(int(projection_dim), int(projection_dim))
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(int(projection_dim))

    def forward(self, x):
        projected = self.proj(x)
        h = self.gelu(projected)
        h = self.fc(h)
        h = self.dropout(h)
        h = h + projected
        h = self.layer_norm(h)
        return h


class FusionRegressor(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class MultimodalRegressionModel(nn.Module):
    def __init__(self, config: dict, graph_input_dim: int):
        super().__init__()
        self.text_encoder = TextEncoder(config)
        self.text_projection = ProjectionHead(config)

        projection_dim = int(config["ProjectionConfig"]["projection_dim"])
        dropout_rate = float(config["ProjectionConfig"].get("dropout_rate", 0.1))

        self.graph_projection = GraphProjectionHead(
            input_dim=graph_input_dim,
            projection_dim=projection_dim,
            dropout_rate=dropout_rate,
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.Sigmoid(),
        )
        self.regressor = FusionRegressor(dim=projection_dim, hidden_dim=projection_dim, dropout=dropout_rate)

    def encode_fused(self, batch):
        text_features = self.text_encoder(batch)
        text_embeddings = self.text_projection(text_features)
        graph_embeddings = self.graph_projection(batch["graph_embed"])
        gate = self.fusion_gate(torch.cat([text_embeddings, graph_embeddings], dim=-1))
        fused = gate * text_embeddings + (1.0 - gate) * graph_embeddings
        return fused

    def forward(self, batch):
        fused = self.encode_fused(batch)
        return self.regressor(fused)


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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_loss_name(config: dict) -> str:
    return str(config.get("loss_fn", "SmoothL1Loss"))


def loss_reduction_none(loss_name: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if loss_name == "MSELoss":
        return (pred - target) ** 2
    if loss_name == "L1Loss":
        return torch.abs(pred - target)
    if loss_name == "SmoothL1Loss":
        return torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    raise ValueError(f"Unsupported loss_fn: {loss_name}")


def weighted_mean_loss(per_sample: torch.Tensor, sample_weight: Optional[torch.Tensor]) -> torch.Tensor:
    if sample_weight is None:
        return per_sample.mean()
    w = sample_weight.float().clamp(min=0.0)
    denom = torch.clamp(w.sum(), min=1e-12)
    return (per_sample * w).sum() / denom


def maybe_load_clip_text_weights(model: MultimodalRegressionModel, pt_ckpt_path: Optional[str], device: str):
    if not pt_ckpt_path:
        return
    print("loading pretrained text encoder and projection layer from")
    print(pt_ckpt_path)
    state = torch.load(pt_ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    own = model.state_dict()
    matched = {}
    for k, v in state.items():
        if k.startswith("text_encoder.") or k.startswith("text_projection."):
            if k in own and own[k].shape == v.shape:
                matched[k] = v
    missing = model.load_state_dict(matched, strict=False)
    print(f"[INFO] loaded pretrained text weights: {len(matched)} tensors")
    unexpected = getattr(missing, "unexpected_keys", [])
    if len(unexpected) > 0:
        print(f"[WARN] unexpected keys while loading pretrained text weights: {unexpected[:10]}")


def maybe_load_init_regress_weights(model: MultimodalRegressionModel, init_regress_ckpt_path: Optional[str], device: str):
    if not init_regress_ckpt_path:
        return
    print("[INFO] loading full multimodal warm-start checkpoint from")
    print(init_regress_ckpt_path)
    state = torch.load(init_regress_ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing = model.load_state_dict(state, strict=False)
    print(f"[INFO] warm-start load done. missing={len(getattr(missing, 'missing_keys', []))}, unexpected={len(getattr(missing, 'unexpected_keys', []))}")


def save_checkpoint(path: str, epoch: int, stage_name: str, model, optimizer, scheduler, best_loss: float, scaler: TargetScaler, graph_input_dim: int, config_snapshot: dict):
    torch.save(
        {
            "epoch": int(epoch),
            "stage_name": str(stage_name),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_loss": float(best_loss),
            "target_mean": float(scaler.mean),
            "target_std": float(scaler.std),
            "standardize_target": bool(scaler.enabled),
            "graph_input_dim": int(graph_input_dim),
            "train_config": config_snapshot,
        },
        path,
    )


def create_scheduler(config: dict, optimizer, train_size: int, num_epochs: int):
    schd = config.get("scheduler", "reduceLR")
    train_bs = int(config["batch_size"])
    warmup = int(config.get("warmup_steps", 0))
    if schd == "reduceLR":
        return ReduceLROnPlateau(optimizer, mode="min", patience=3)
    train_steps = max(1, int(np.ceil(train_size / max(train_bs, 1))) * num_epochs)
    return transformers.get_scheduler(
        schd,
        optimizer=optimizer,
        num_warmup_steps=warmup,
        num_training_steps=train_steps,
    )


def train_fn(data_loader, model, optimizer, device, scheduler, loss_name: str, scaler: TargetScaler, log_interval: int, debug=False):
    model.train()
    lr_list = []
    train_losses = []
    print("training...")
    batch_iteration = 0

    for batch in tqdm(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        targets = batch["target"]
        weights = batch.get("sample_weight")
        targets_z = scaler.transform(targets)

        optimizer.zero_grad()
        outputs_z = model(batch).squeeze(-1)
        per_sample = loss_reduction_none(loss_name, outputs_z, targets_z)
        loss = weighted_mean_loss(per_sample, weights)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        lr_list.append(float(optimizer.param_groups[0]["lr"]))
        if (batch_iteration != 0) and (batch_iteration % log_interval == 0) and (debug is False):
            wandb.log({"iter_train_loss": float(loss.item())})
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        batch_iteration += 1

    return float(np.mean(train_losses)), float(np.mean(lr_list))


def validate_fn(data_loader, model, device, loss_name: str, scaler: TargetScaler):
    model.eval()
    val_losses = []
    val_maes = []
    print("validating...")
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs_z = model(batch).squeeze(-1)
            targets = batch["target"]
            targets_z = scaler.transform(targets)
            per_sample = loss_reduction_none(loss_name, outputs_z, targets_z)
            loss = per_sample.mean()
            outputs = scaler.inverse_torch(outputs_z)
            mae = torch.mean(torch.abs(targets - outputs))
            val_losses.append(float(loss.item()))
            val_maes.append(float(mae.item()))
    return float(np.mean(val_losses)), float(np.mean(val_maes))


def _get_text_backbone_container(text_encoder_module: nn.Module) -> nn.Module:
    for attr in ["model", "roberta", "backbone", "encoder_model"]:
        if hasattr(text_encoder_module, attr):
            return getattr(text_encoder_module, attr)
    return text_encoder_module


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


def set_module_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = bool(flag)


def freeze_all_text_encoder(model: MultimodalRegressionModel):
    set_module_requires_grad(model.text_encoder, False)


def unfreeze_top_n_text_layers(model: MultimodalRegressionModel, top_n: int):
    freeze_all_text_encoder(model)
    if top_n <= 0:
        return
    backbone = _get_text_backbone_container(model.text_encoder)
    layers = _get_transformer_layers(backbone)
    if layers is None:
        print("[WARN] Could not locate transformer layers in text_encoder. Unfreezing full text_encoder instead.")
        set_module_requires_grad(model.text_encoder, True)
        return
    n_layers = len(layers)
    top_n = min(int(top_n), int(n_layers))
    for i in range(n_layers - top_n, n_layers):
        set_module_requires_grad(layers[i], True)
    for attr in ["pooler", "LayerNorm", "layer_norm"]:
        if hasattr(backbone, attr):
            set_module_requires_grad(getattr(backbone, attr), True)


def apply_stage_freeze_policy(model: MultimodalRegressionModel, config: dict, stage_name: str):
    if stage_name == "stage1":
        freeze_text_encoder = bool(config.get("freeze_text_encoder_stage1", True))
        freeze_text_projection = bool(config.get("freeze_text_projection_stage1", False))
        if freeze_text_encoder:
            freeze_all_text_encoder(model)
        else:
            set_module_requires_grad(model.text_encoder, True)
        set_module_requires_grad(model.text_projection, not freeze_text_projection)
        set_module_requires_grad(model.graph_projection, True)
        set_module_requires_grad(model.fusion_gate, True)
        set_module_requires_grad(model.regressor, True)
    elif stage_name == "stage2":
        top_n = int(config.get("unfreeze_top_n_layers_stage2", 2))
        unfreeze_top_n_text_layers(model, top_n=top_n)
        set_module_requires_grad(model.text_projection, True)
        set_module_requires_grad(model.graph_projection, True)
        set_module_requires_grad(model.fusion_gate, True)
        set_module_requires_grad(model.regressor, True)
    else:
        raise ValueError(f"Unknown stage_name: {stage_name}")


def _named_trainable_params(module: nn.Module):
    for _, p in module.named_parameters():
        if p.requires_grad:
            yield p


def build_stage_optimizer(config: dict, model: MultimodalRegressionModel, stage_name: str):
    wd = float(config.get("weight_decay", 0.01))
    if stage_name == "stage1":
        lr_new = float(config.get("lr_stage1_new", config.get("lr", 2e-5)))
        lr_text_proj = float(config.get("lr_stage1_text_projection", max(lr_new * 0.15, 1e-6)))
        param_groups = []
        tp = list(_named_trainable_params(model.text_projection))
        if tp:
            param_groups.append({"params": tp, "lr": lr_text_proj, "weight_decay": wd})
        new_params = list(_named_trainable_params(model.graph_projection)) + list(_named_trainable_params(model.fusion_gate)) + list(_named_trainable_params(model.regressor))
        if new_params:
            param_groups.append({"params": new_params, "lr": lr_new, "weight_decay": wd})
        if not param_groups:
            raise ValueError("No trainable parameters found for stage1.")
        return torch.optim.AdamW(param_groups)
    if stage_name == "stage2":
        lr_new = float(config.get("lr_stage2_new", config.get("lr", 1e-5)))
        lr_text_proj = float(config.get("lr_stage2_text_projection", max(lr_new * 0.2, 1e-6)))
        lr_text_top = float(config.get("lr_stage2_text_top", max(lr_new * 0.07, 1e-6)))
        param_groups = []
        text_params = list(_named_trainable_params(model.text_encoder))
        if text_params:
            param_groups.append({"params": text_params, "lr": lr_text_top, "weight_decay": wd})
        tp = list(_named_trainable_params(model.text_projection))
        if tp:
            param_groups.append({"params": tp, "lr": lr_text_proj, "weight_decay": wd})
        new_params = list(_named_trainable_params(model.graph_projection)) + list(_named_trainable_params(model.fusion_gate)) + list(_named_trainable_params(model.regressor))
        if new_params:
            param_groups.append({"params": new_params, "lr": lr_new, "weight_decay": wd})
        if not param_groups:
            raise ValueError("No trainable parameters found for stage2.")
        return torch.optim.AdamW(param_groups)
    raise ValueError(f"Unknown stage_name: {stage_name}")


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


def _extract_sample_weights(df: pd.DataFrame, weight_col: Optional[str]) -> np.ndarray:
    if weight_col is None or weight_col not in df.columns:
        return np.ones(len(df), dtype=np.float32)
    w = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
    w = np.where(w > 0, w, 1.0).astype(np.float32)
    return w


def run_regression(config_file: str = "regress_train.yml"):
    config = load_yaml(config_file)
    set_seed(int(config.get("seed", 42)))

    train_path = config["train_path"]
    val_path = config["val_path"]
    ckpt_save_root = Path(config["ckpt_save_path"])
    ckpt_save_root.mkdir(parents=True, exist_ok=True)
    model_config_path = config["model_config"]
    pt_ckpt_path = config.get("pt_ckpt_path")
    init_regress_ckpt_path = config.get("init_regress_ckpt_path")
    device = config.get("device", "cuda")
    batch_size = int(config["batch_size"])
    loss_name = get_loss_name(config)
    log_interval = int(config.get("log_interval", 10))
    debug = bool(config.get("debug", False))
    standardize_target = bool(config.get("standardize_target", True))
    train_strategy = str(config.get("train_strategy", "two_stage"))
    num_epochs = int(config.get("num_epochs", 24))
    stage1_epochs = int(config.get("stage1_epochs", max(1, min(8, num_epochs // 3))))
    stage2_epochs = int(config.get("stage2_epochs", max(1, num_epochs - stage1_epochs)))
    text_col = str(config.get("text_col", "text"))
    concat_text_cols = config.get("concat_text_cols", ["text_structured", "text_raw", "text"])
    seq_len = int(config.get("seq_len", 256))
    num_workers = int(config.get("num_workers", 2))
    sample_weight_col = config.get("sample_weight_col")
    use_weighted_sampler = bool(config.get("use_weighted_sampler", False))
    sampler_power = float(config.get("weighted_sampler_power", 1.0))

    if train_strategy == "single_stage":
        stage1_epochs = 0
        stage2_epochs = num_epochs

    if debug:
        device = "cpu"

    run_name = config["run_name"] + datetime.now().strftime("_%m%d_%H%M")
    ckpt_save_dir = ckpt_save_root / run_name
    ckpt_save_dir.mkdir(parents=True, exist_ok=True)

    print("=============================================================")
    print(f"{run_name} is launched")
    print("=============================================================")
    print("Model: multimodal regression (aligned staged)")
    print(f"Train strategy: {train_strategy}")
    print(f"Stage1 epochs: {stage1_epochs}")
    print(f"Stage2 epochs: {stage2_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Loss function: {loss_name}")
    print(f"Standardize target: {standardize_target}")
    print(f"Text col: {text_col}")
    print(f"Concat text cols: {concat_text_cols}")
    print(f"Seq len: {seq_len}")
    print(f"Sample weight col: {sample_weight_col}")
    print(f"Weighted sampler: {use_weighted_sampler}")
    print(f"Device: {device}")
    print("=============================================================")

    wandb_mode = config.get("wandb_mode", os.environ.get("WANDB_MODE", "disabled"))
    if not debug:
        wandb.init(project="clip-regress", name=run_name, mode=wandb_mode)

    if not debug:
        shutil.copyfile(config_file, ckpt_save_dir / Path(config_file).name)
        shutil.copyfile(model_config_path, ckpt_save_dir / Path(model_config_path).name)

    df_train = pd.read_pickle(train_path)
    df_val = pd.read_pickle(val_path)

    required_cols = {"id", "target", "eq_emb"}
    for name, df in [("train", df_train), ("val", df_val)]:
        miss = required_cols - set(df.columns)
        if miss:
            raise ValueError(f"{name} dataframe missing required columns for multimodal regression: {sorted(miss)}")

    df_train = df_train.copy()
    df_val = df_val.copy()
    df_train["text"] = _build_text_series(df_train, preferred_col=text_col, concat_cols=concat_text_cols)
    df_val["text"] = _build_text_series(df_val, preferred_col=text_col, concat_cols=concat_text_cols)

    if debug:
        df_train = df_train.sample(min(10, len(df_train)), random_state=0)
        df_val = df_val.sample(min(4, len(df_val)), random_state=0)

    graph_dims = sorted({int(np.asarray(x).reshape(-1).shape[0]) for x in df_train["eq_emb"]})
    if len(graph_dims) != 1:
        raise ValueError(f"Inconsistent eq_emb dims in train set: {graph_dims}")
    graph_input_dim = int(graph_dims[0])

    val_graph_dims = sorted({int(np.asarray(x).reshape(-1).shape[0]) for x in df_val["eq_emb"]})
    if len(val_graph_dims) != 1 or int(val_graph_dims[0]) != graph_input_dim:
        raise ValueError(f"Validation eq_emb dim mismatch: train={graph_input_dim}, val={val_graph_dims}")

    model_config = load_yaml(model_config_path)
    pretrain_ckpt = model_config.get("Path", {}).get("pretrain_ckpt", "roberta-base")
    tokenizer = RobertaTokenizerFast.from_pretrained(pretrain_ckpt)

    train_weights = _extract_sample_weights(df_train, sample_weight_col)
    val_weights = _extract_sample_weights(df_val, sample_weight_col)

    train_dataset = MultimodalRegressionDataset(
        texts=df_train["text"].values,
        targets=df_train["target"].values,
        graph_emb=df_train["eq_emb"].values,
        tokenizer=tokenizer,
        seq_len=seq_len,
        sample_weights=train_weights,
    )
    val_dataset = MultimodalRegressionDataset(
        texts=df_val["text"].values,
        targets=df_val["target"].values,
        graph_emb=df_val["eq_emb"].values,
        tokenizer=tokenizer,
        seq_len=seq_len,
        sample_weights=val_weights,
    )

    if use_weighted_sampler:
        sampler_weights = np.power(np.clip(train_weights.astype(np.float64), 1e-6, None), sampler_power)
        sampler = WeightedRandomSampler(weights=torch.as_tensor(sampler_weights, dtype=torch.double), num_samples=len(sampler_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = MultimodalRegressionModel(model_config, graph_input_dim=graph_input_dim).to(device)
    maybe_load_clip_text_weights(model, pt_ckpt_path, device)
    maybe_load_init_regress_weights(model, init_regress_ckpt_path, device)

    scaler = TargetScaler(enabled=standardize_target).fit(df_train["target"].astype(float).to_numpy())

    print(f"[INFO] graph_input_dim = {graph_input_dim}")
    print(f"[INFO] target_mean/std = {scaler.mean:.6f} / {scaler.std:.6f}")

    best_loss = 999999.0
    best_epoch = -1
    best_stage = None

    def run_stage(stage_name: str, stage_epochs_local: int, global_best_loss: float):
        nonlocal best_epoch, best_stage
        if stage_epochs_local <= 0:
            return global_best_loss

        print("\\n=============================================================")
        print(f"Starting {stage_name}")
        print("=============================================================")

        apply_stage_freeze_policy(model, config, stage_name=stage_name)
        summarize_trainable_parameters(model)

        optimizer = build_stage_optimizer(config, model, stage_name=stage_name)
        scheduler = create_scheduler(config, optimizer, len(df_train), stage_epochs_local)
        early_stop_threshold = int(config.get("early_stop_threshold", 4))
        early_stopping_counter = 0

        for epoch in range(1, stage_epochs_local + 1):
            train_loss, lr_now = train_fn(train_loader, model, optimizer, device, scheduler, loss_name, scaler, log_interval, debug=debug)
            val_loss, val_mae = validate_fn(val_loader, model, device, loss_name, scaler)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)

            if not debug:
                wandb.log({
                    f"{stage_name}/train_loss": train_loss,
                    f"{stage_name}/val_loss": val_loss,
                    f"{stage_name}/val_mae": val_mae,
                    f"{stage_name}/lr": lr_now,
                })

            if val_loss < global_best_loss:
                save_checkpoint(
                    str(ckpt_save_dir / "checkpoint.pt"),
                    epoch,
                    stage_name,
                    model,
                    optimizer,
                    scheduler,
                    global_best_loss,
                    scaler,
                    graph_input_dim,
                    config_snapshot=config,
                )
                print(f"{stage_name} Epoch: {epoch}, Train Loss = {round(train_loss, 3)}, Val Loss = {round(val_loss, 3)}, Val MAE = {round(val_mae, 3)}, checkpoint saved.")
                global_best_loss = val_loss
                best_epoch = epoch
                best_stage = stage_name
                early_stopping_counter = 0
            else:
                print(f"{stage_name} Epoch: {epoch}, Train Loss = {round(train_loss, 3)}, Val Loss = {round(val_loss, 3)}, Val MAE = {round(val_mae, 3)}")
                early_stopping_counter += 1
                if early_stopping_counter > early_stop_threshold:
                    print(f"[INFO] Early stopping triggered in {stage_name} at epoch {epoch}")
                    break
        return global_best_loss

    if train_strategy == "single_stage":
        best_loss = run_stage("stage2", stage2_epochs, best_loss)
    else:
        best_loss = run_stage("stage1", stage1_epochs, best_loss)
        best_loss = run_stage("stage2", stage2_epochs, best_loss)

    print("===== Training Termination =====")
    print(f"[INFO] best_stage = {best_stage}, best_epoch = {best_epoch}, best_val_loss = {best_loss:.6f}")
    if not debug:
        wandb.finish()


if __name__ == "__main__":
    run_regression("regress_train.yml")
