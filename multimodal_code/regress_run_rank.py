#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
import wandb
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizerFast

from dataset import RegressionDataset
from model.models import RegressionModel, RegressionModel2
from utils import roberta_base_AdamW_grouped_LLRD


def _ensure_checkpoint_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _maybe_log_wandb(payload: dict, debug: bool) -> None:
    if not debug:
        wandb.log(payload)


def build_loss_fn(loss_name: str) -> nn.Module:
    if loss_name == "MSELoss":
        return nn.MSELoss()
    if loss_name == "L1Loss":
        return nn.L1Loss()
    if loss_name == "SmoothL1Loss":
        return nn.SmoothL1Loss()
    raise ValueError(f"Unsupported loss_fn: {loss_name}")


def pairwise_rank_hinge_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.1,
    min_delta: float = 0.2,
    max_pairs: Optional[int] = 4096,
    pair_mode: str = "all_pairs",
) -> torch.Tensor:
    """
    Pairwise ranking loss for scalar regression outputs.

    For each valid pair (i, j) with |y_i - y_j| >= min_delta, enforce
    sign(y_i - y_j) * (pred_i - pred_j) >= margin.
    """
    outputs = outputs.view(-1)
    targets = targets.view(-1)
    n = outputs.shape[0]
    device = outputs.device

    if n < 2:
        return torch.zeros((), device=device, dtype=outputs.dtype)

    diff_y = targets.unsqueeze(1) - targets.unsqueeze(0)
    diff_p = outputs.unsqueeze(1) - outputs.unsqueeze(0)

    valid = torch.abs(diff_y) >= float(min_delta)
    upper = torch.triu(torch.ones_like(valid, dtype=torch.bool), diagonal=1)
    valid = valid & upper

    if not torch.any(valid):
        return torch.zeros((), device=device, dtype=outputs.dtype)

    y_pairs = diff_y[valid]
    p_pairs = diff_p[valid]
    signs = torch.sign(y_pairs)

    if max_pairs is not None and y_pairs.numel() > int(max_pairs):
        if pair_mode == "random_subset":
            idx = torch.randperm(y_pairs.numel(), device=device)[: int(max_pairs)]
        else:
            # all_pairs fallback with cap to avoid quadratic explosion
            idx = torch.arange(int(max_pairs), device=device)
        y_pairs = y_pairs[idx]
        p_pairs = p_pairs[idx]
        signs = signs[idx]

    losses = torch.relu(float(margin) - signs * p_pairs)
    if losses.numel() == 0:
        return torch.zeros((), device=device, dtype=outputs.dtype)
    return losses.mean()


def load_state_dict_flex(model: nn.Module, ckpt_path: str, device: str) -> None:
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)


def train_fn(
    data_loader,
    model,
    optimizer,
    device,
    scheduler,
    base_loss_fn,
    log_interval,
    use_rank_loss: bool = False,
    rank_weight: float = 0.2,
    rank_margin: float = 0.1,
    rank_min_delta: float = 0.2,
    rank_max_pairs: Optional[int] = 4096,
    rank_pair_mode: str = "all_pairs",
    debug: bool = False,
):
    model.train()
    lr_list = []
    total_losses = []
    base_losses = []
    rank_losses = []

    print("training...")
    batch_iteration = 0
    for batch in tqdm(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        targets = batch["target"]

        optimizer.zero_grad()
        outputs = model(batch).squeeze(-1)

        base_loss = base_loss_fn(outputs, targets)
        rank_loss = torch.zeros((), device=device, dtype=outputs.dtype)
        if use_rank_loss:
            rank_loss = pairwise_rank_hinge_loss(
                outputs,
                targets,
                margin=rank_margin,
                min_delta=rank_min_delta,
                max_pairs=rank_max_pairs,
                pair_mode=rank_pair_mode,
            )

        total_loss = base_loss + float(rank_weight) * rank_loss
        total_loss.backward()
        optimizer.step()

        total_losses.append(total_loss.item())
        base_losses.append(base_loss.item())
        rank_losses.append(float(rank_loss.item()))
        lr_list.append(optimizer.param_groups[0]["lr"])

        if (batch_iteration != 0) and (batch_iteration % log_interval == 0) and (not debug):
            wandb.log(
                {
                    "iter_train_loss": total_loss.item(),
                    "iter_train_base_loss": base_loss.item(),
                    "iter_train_rank_loss": float(rank_loss.item()),
                }
            )

        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        batch_iteration += 1

    return (
        float(np.mean(total_losses)),
        float(np.mean(base_losses)),
        float(np.mean(rank_losses)),
        float(np.mean(lr_list)),
    )


def validate_fn(
    data_loader,
    model,
    device,
    base_loss_fn,
    use_rank_loss: bool = False,
    rank_weight: float = 0.2,
    rank_margin: float = 0.1,
    rank_min_delta: float = 0.2,
    rank_max_pairs: Optional[int] = 4096,
    rank_pair_mode: str = "all_pairs",
):
    model.eval()
    total_losses = []
    base_losses = []
    rank_losses = []
    val_maes = []

    print("validating...")
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch).squeeze(-1)
            targets = batch["target"]

            base_loss = base_loss_fn(outputs, targets)
            rank_loss = torch.zeros((), device=device, dtype=outputs.dtype)
            if use_rank_loss:
                rank_loss = pairwise_rank_hinge_loss(
                    outputs,
                    targets,
                    margin=rank_margin,
                    min_delta=rank_min_delta,
                    max_pairs=rank_max_pairs,
                    pair_mode=rank_pair_mode,
                )
            total_loss = base_loss + float(rank_weight) * rank_loss
            mae = torch.mean(torch.abs(targets - outputs))

            total_losses.append(total_loss.item())
            base_losses.append(base_loss.item())
            rank_losses.append(float(rank_loss.item()))
            val_maes.append(mae.item())

    return (
        float(np.mean(total_losses)),
        float(np.mean(base_losses)),
        float(np.mean(rank_losses)),
        float(np.mean(val_maes)),
    )


def run_regression(config_file: str) -> None:
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    run_name = config["run_name"] + datetime.now().strftime("_%m%d_%H%M")
    train_path = config["train_path"]
    val_path = config["val_path"]
    ckpt_save_dir = os.path.join(config["ckpt_save_path"], run_name)
    resume_path = config.get("resume_path") or None
    resume_config = config.get("resume_config") or None
    pt_ckpt_path = config.get("pt_ckpt_path") or None
    model_config_path = config["model_config"]
    head = config.get("head") or "regress"
    device = config["device"]
    epochs = int(config["num_epochs"])
    early_stop_threshold = int(config["early_stop_threshold"])
    train_bs = int(config["batch_size"])
    val_bs = train_bs
    lr = float(config.get("lr", 1e-6))
    warmup = int(config.get("warmup_steps", 0))
    optim = config.get("optimizer", "AdamW")
    schd = config.get("scheduler", "reduceLR")
    loss_name = config.get("loss_fn", "MSELoss")
    log_interval = int(config.get("log_interval", 10))
    debug = bool(config.get("debug", False))

    use_rank_loss = bool(config.get("use_rank_loss", False))
    rank_weight = float(config.get("rank_weight", 0.2))
    rank_margin = float(config.get("rank_margin", 0.1))
    rank_min_delta = float(config.get("rank_min_delta", 0.2))
    rank_max_pairs = config.get("rank_max_pairs", 4096)
    rank_max_pairs = None if rank_max_pairs is None else int(rank_max_pairs)
    rank_pair_mode = str(config.get("rank_pair_mode", "all_pairs"))

    if debug:
        device = "cpu"
    if resume_path and resume_config:
        model_config_path = resume_config

    print("=============================================================")
    print(f"{run_name} is launched")
    print("=============================================================")
    print(f"Head: {head}")
    print(f"Epochs: {epochs}")
    print(f"Early stopping threshold: {early_stop_threshold}")
    print(f"Training batch size: {train_bs}")
    print(f"Validation batch size: {val_bs}")
    print(f"Initial learning rate: {lr}")
    print(f"Warmup steps: {warmup}")
    print(f"Optimizer: {optim}")
    print(f"Scheduler: {schd}")
    print(f"Loss function: {loss_name}")
    print(f"Use ranking loss: {use_rank_loss}")
    if use_rank_loss:
        print(f"Rank weight: {rank_weight}")
        print(f"Rank margin: {rank_margin}")
        print(f"Rank min delta: {rank_min_delta}")
        print(f"Rank max pairs: {rank_max_pairs}")
        print(f"Rank pair mode: {rank_pair_mode}")
    print("=============================================================")

    if not debug:
        wandb.init(project="clip-regress-rank", name=run_name)

    _ensure_checkpoint_dir(ckpt_save_dir)
    if not debug:
        shutil.copyfile(config_file, os.path.join(ckpt_save_dir, Path(config_file).name))
        shutil.copyfile(model_config_path, os.path.join(ckpt_save_dir, Path(model_config_path).name))

    df_train = pd.read_pickle(train_path)
    df_val = pd.read_pickle(val_path)
    if debug:
        df_train = df_train.sample(min(10, len(df_train)))
        df_val = df_val.sample(min(2, len(df_val)))

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    train_dataset = RegressionDataset(
        texts=df_train["text"].values,
        targets=df_train["target"].values,
        tokenizer=tokenizer,
        seq_len=tokenizer.model_max_length,
    )
    val_dataset = RegressionDataset(
        texts=df_val["text"].values,
        targets=df_val["target"].values,
        tokenizer=tokenizer,
        seq_len=tokenizer.model_max_length,
    )
    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, num_workers=2)

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    model = RegressionModel2(model_config).to(device) if head == "pooler" else RegressionModel(model_config).to(device)

    if pt_ckpt_path:
        print("loading pretrained text encoder and projection layer from")
        print(pt_ckpt_path)
        load_state_dict_flex(model, pt_ckpt_path, device)
    elif resume_path:
        print("resume training from")
        print(resume_path)
        load_state_dict_flex(model, resume_path, device)

    if optim == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    elif optim == "gLLRD":
        optimizer, _ = roberta_base_AdamW_grouped_LLRD(model, lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optim}")

    if schd == "reduceLR":
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=3)
    else:
        train_steps = max(1, int(len(df_train) / max(train_bs, 1) * epochs))
        scheduler = transformers.get_scheduler(
            schd,
            optimizer=optimizer,
            num_warmup_steps=warmup,
            num_training_steps=train_steps,
        )

    base_loss_fn = build_loss_fn(loss_name)

    best_loss = 999999.0
    early_stopping_counter = 0

    for epoch in range(1, epochs + 1):
        train_total, train_base, train_rank, lr_mean = train_fn(
            train_loader,
            model,
            optimizer,
            device,
            scheduler,
            base_loss_fn,
            log_interval,
            use_rank_loss=use_rank_loss,
            rank_weight=rank_weight,
            rank_margin=rank_margin,
            rank_min_delta=rank_min_delta,
            rank_max_pairs=rank_max_pairs,
            rank_pair_mode=rank_pair_mode,
            debug=debug,
        )

        val_total, val_base, val_rank, val_mae = validate_fn(
            val_loader,
            model,
            device,
            base_loss_fn,
            use_rank_loss=use_rank_loss,
            rank_weight=rank_weight,
            rank_margin=rank_margin,
            rank_min_delta=rank_min_delta,
            rank_max_pairs=rank_max_pairs,
            rank_pair_mode=rank_pair_mode,
        )

        if schd == "reduceLR":
            scheduler.step(val_total)

        if not debug:
            wandb.log(
                {
                    "train_loss": train_total,
                    "train_base_loss": train_base,
                    "train_rank_loss": train_rank,
                    "val_loss": val_total,
                    "val_base_loss": val_base,
                    "val_rank_loss": val_rank,
                    "val_mae": val_mae,
                    "lr": lr_mean,
                }
            )

        if val_total < best_loss:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "config_file": config_file,
                },
                os.path.join(ckpt_save_dir, "checkpoint.pt"),
            )
            print(
                f"Epoch: {epoch}, Train Loss = {round(train_total,3)}, "
                f"Train Base = {round(train_base,3)}, Train Rank = {round(train_rank,3)}, "
                f"Val Loss = {round(val_total,3)}, Val Base = {round(val_base,3)}, "
                f"Val Rank = {round(val_rank,3)}, Val MAE = {round(val_mae,3)}, checkpoint saved."
            )
            best_loss = val_total
            early_stopping_counter = 0
        else:
            print(
                f"Epoch: {epoch}, Train Loss = {round(train_total,3)}, "
                f"Train Base = {round(train_base,3)}, Train Rank = {round(train_rank,3)}, "
                f"Val Loss = {round(val_total,3)}, Val Base = {round(val_base,3)}, "
                f"Val Rank = {round(val_rank,3)}, Val MAE = {round(val_mae,3)},"
            )
            early_stopping_counter += 1
            if early_stopping_counter > early_stop_threshold:
                print(f"Early stopping triggered at epoch {epoch}! Best Loss: {round(best_loss, 3)}\n")
                break

    print("===== Training Termination =====")
    if not debug:
        wandb.finish()


if __name__ == "__main__":
    run_regression("regress_train.yml")
