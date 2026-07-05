#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
import wandb
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import RobertaTokenizerFast

from model.modules import TextEncoder, ProjectionHead


class MultimodalRegressionDataset(Dataset):
    def __init__(self, texts, targets, graph_emb, tokenizer, seq_len=512):
        self.texts = texts
        self.targets = np.asarray(targets, dtype=np.float32)
        self.graph_emb = graph_emb
        self.tokenizer = tokenizer
        self.seq_len = seq_len

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

    def forward(self, batch):
        text_features = self.text_encoder(batch)
        text_embeddings = self.text_projection(text_features)

        graph_embeddings = batch["graph_embed"]
        graph_embeddings = self.graph_projection(graph_embeddings)

        gate = self.fusion_gate(torch.cat([text_embeddings, graph_embeddings], dim=-1))
        fused = gate * text_embeddings + (1.0 - gate) * graph_embeddings
        out = self.regressor(fused)
        return out


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


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(path: str, epoch: int, model, optimizer, scheduler, best_loss: float, scaler: TargetScaler, graph_input_dim: int):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_loss": best_loss,
            "target_mean": float(scaler.mean),
            "target_std": float(scaler.std),
            "standardize_target": bool(scaler.enabled),
            "graph_input_dim": int(graph_input_dim),
        },
        path,
    )


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
    print(f"[INFO] loaded multimodal text weights: {len(matched)} tensors")
    if len(getattr(missing, 'unexpected_keys', [])) > 0:
        print(f"[WARN] unexpected keys while loading pretrained text weights: {missing.unexpected_keys[:10]}")


def create_optimizer(config: dict, model: nn.Module):
    lr = float(config.get("lr", 1e-6))
    optimizer_name = config.get("optimizer", "AdamW")
    if optimizer_name != "AdamW":
        print(f"[WARN] optimizer {optimizer_name} not explicitly implemented; fallback to AdamW")
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)


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


def get_loss_fn(name: str):
    if name == "MSELoss":
        return nn.MSELoss()
    if name == "L1Loss":
        return nn.L1Loss()
    if name == "SmoothL1Loss":
        return nn.SmoothL1Loss()
    raise ValueError(f"Unsupported loss_fn: {name}")


def train_fn(data_loader, model, optimizer, device, scheduler, loss_fn, scaler: TargetScaler, log_interval, debug=False):
    model.train()
    lr_list = []
    train_losses = []
    print("training...")
    batch_iteration = 0

    for batch in tqdm(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        targets = batch["target"]
        targets_z = scaler.transform(targets)

        optimizer.zero_grad()
        outputs_z = model(batch).squeeze(-1)
        loss = loss_fn(outputs_z, targets_z)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        lr_list.append(optimizer.param_groups[0]["lr"])
        if (batch_iteration != 0) and (batch_iteration % log_interval == 0) and (debug is False):
            wandb.log({"iter_train_loss": loss.item()})
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        batch_iteration += 1

    return float(np.mean(train_losses)), float(np.mean(lr_list))


def validate_fn(data_loader, model, device, loss_fn, scaler: TargetScaler):
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
            loss = loss_fn(outputs_z, targets_z)

            outputs = scaler.inverse_torch(outputs_z)
            mae = torch.mean(torch.abs(targets - outputs))

            val_losses.append(loss.item())
            val_maes.append(mae.item())

    return float(np.mean(val_losses)), float(np.mean(val_maes))


def run_regression(config_file: str = "regress_train.yml"):
    config = load_yaml(config_file)

    run_name = config["run_name"] + datetime.now().strftime("_%m%d_%H%M")
    train_path = config["train_path"]
    val_path = config["val_path"]
    ckpt_save_dir = os.path.join(config["ckpt_save_path"], run_name)
    resume_path = config["resume_path"] if config.get("resume_path") else None
    resume_config = config["resume_config"] if config.get("resume_config") else None
    pt_ckpt_path = config["pt_ckpt_path"] if config.get("pt_ckpt_path") else None
    model_config_path = config["model_config"]
    device = config["device"]
    epochs = int(config["num_epochs"])
    early_stop_threshold = int(config["early_stop_threshold"])
    train_bs = int(config["batch_size"])
    val_bs = train_bs
    lr = float(config.get("lr", 1e-6))
    warmup = int(config.get("warmup_steps", 0))
    optim = config.get("optimizer", "AdamW")
    schd = config.get("scheduler", "reduceLR")
    loss_name = config.get("loss_fn", "L1Loss")
    log_interval = int(config.get("log_interval", 10))
    debug = bool(config.get("debug", False))
    standardize_target = bool(config.get("standardize_target", True))

    if debug:
        device = "cpu"
    if resume_path and resume_config:
        model_config_path = resume_config

    print("=============================================================")
    print(f"{run_name} is launched")
    print("=============================================================")
    print("Model: multimodal regression")
    print(f"Epochs: {epochs}")
    print(f"Early stopping threshold: {early_stop_threshold}")
    print(f"Training batch size: {train_bs}")
    print(f"Validation batch size: {val_bs}")
    print(f"Initial learning rate: {lr}")
    print(f"Warmup steps: {warmup}")
    print(f"Optimizer: {optim}")
    print(f"Scheduler: {schd}")
    print(f"Loss function: {loss_name}")
    print(f"Standardize target: {standardize_target}")
    print("=============================================================")

    if not debug:
        wandb.init(project="clip-regress", name=run_name)

    os.makedirs(ckpt_save_dir, exist_ok=True)
    if not debug:
        shutil.copyfile(config_file, os.path.join(ckpt_save_dir, Path(config_file).name))
        shutil.copyfile(model_config_path, os.path.join(ckpt_save_dir, Path(model_config_path).name))

    df_train = pd.read_pickle(train_path)
    df_val = pd.read_pickle(val_path)

    required_cols = {"id", "text", "target", "eq_emb"}
    for name, df in [("train", df_train), ("val", df_val)]:
        miss = required_cols - set(df.columns)
        if miss:
            raise ValueError(f"{name} dataframe missing required columns for multimodal regression: {sorted(miss)}")

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

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    train_dataset = MultimodalRegressionDataset(
        texts=df_train["text"].values,
        targets=df_train["target"].values,
        graph_emb=df_train["eq_emb"].values,
        tokenizer=tokenizer,
        seq_len=tokenizer.model_max_length,
    )
    val_dataset = MultimodalRegressionDataset(
        texts=df_val["text"].values,
        targets=df_val["target"].values,
        graph_emb=df_val["eq_emb"].values,
        tokenizer=tokenizer,
        seq_len=tokenizer.model_max_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, num_workers=2)

    model_config = load_yaml(model_config_path)
    model = MultimodalRegressionModel(model_config, graph_input_dim=graph_input_dim).to(device)

    if pt_ckpt_path:
        maybe_load_clip_text_weights(model, pt_ckpt_path, device)
    elif resume_path:
        print("resume training from", resume_path)
        state = torch.load(resume_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)

    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer, len(df_train))
    loss_fn = get_loss_fn(loss_name)

    scaler = TargetScaler(enabled=standardize_target).fit(df_train["target"].astype(float).to_numpy())
    print(f"[INFO] graph_input_dim = {graph_input_dim}")
    print(f"[INFO] target_mean/std = {scaler.mean:.6f} / {scaler.std:.6f}")

    best_loss = 999999.0
    early_stopping_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, lr_now = train_fn(
            train_loader, model, optimizer, device, scheduler, loss_fn, scaler, log_interval, debug=debug
        )
        val_loss, val_mae = validate_fn(val_loader, model, device, loss_fn, scaler)

        if schd == "reduceLR":
            scheduler.step(val_loss)

        if not debug:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "lr": lr_now,
            })

        if val_loss < best_loss:
            save_checkpoint(
                os.path.join(ckpt_save_dir, "checkpoint.pt"),
                epoch,
                model,
                optimizer,
                scheduler,
                best_loss,
                scaler,
                graph_input_dim,
            )
            print(
                f"Epoch: {epoch}, Train Loss = {round(train_loss,3)}, "
                f"Val Loss = {round(val_loss,3)}, Val MAE = {round(val_mae,3)}, checkpoint saved."
            )
            best_loss = val_loss
            early_stopping_counter = 0
        else:
            print(
                f"Epoch: {epoch}, Train Loss = {round(train_loss,3)}, "
                f"Val Loss = {round(val_loss,3)}, Val MAE = {round(val_mae,3)}"
            )
            early_stopping_counter += 1
            if early_stopping_counter > early_stop_threshold:
                print(f"Early stopping triggered at epoch {epoch}! Best Loss: {round(best_loss,3)}\n")
                break

    print("===== Training Termination =====")
    if not debug:
        wandb.finish()


if __name__ == "__main__":
    run_regression("regress_train.yml")
