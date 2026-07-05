#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import RobertaTokenizerFast

from model.modules import TextEncoder, ProjectionHead


class MultimodalRegressionDataset(Dataset):
    def __init__(self, texts, graph_emb, ids=None, targets=None, tokenizer=None, seq_len=512):
        self.texts = texts
        self.graph_emb = graph_emb
        self.ids = ids if ids is not None else list(range(len(texts)))
        self.targets = np.asarray(targets, dtype=np.float32) if targets is not None else np.zeros(len(texts), dtype=np.float32)
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
            "id": self.ids[idx],
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

        graph_embeddings = self.graph_projection(batch["graph_embed"])

        gate = self.fusion_gate(torch.cat([text_embeddings, graph_embeddings], dim=-1))
        fused = gate * text_embeddings + (1.0 - gate) * graph_embeddings
        out = self.regressor(fused)
        return out


def parse_args():
    ap = argparse.ArgumentParser(description="Predict with multimodal regression model (text + eq_emb).")
    ap.add_argument("--data_path", required=True, help="Path to regress_test.pkl or addH_out_pred_input.pkl")
    ap.add_argument("--pt_ckpt_dir_path", required=True, help="Checkpoint directory containing checkpoint.pt and copied clip.yml")
    ap.add_argument("--save_path", required=True, help="Save path. If no .pkl suffix, treated as output directory.")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch_size", type=int, default=32)
    return ap.parse_args()


def resolve_checkpoint_dir(path_str: str) -> Path:
    p = Path(path_str).resolve()
    if p.is_dir():
        return p
    raise FileNotFoundError(f"Checkpoint directory not found: {p}")


def resolve_model_config(ckpt_dir: Path) -> Path:
    candidates = [
        ckpt_dir / "clip.yml",
        ckpt_dir / "model__clip.yml",
        ckpt_dir / "model" / "clip.yml",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find clip.yml under checkpoint dir: {ckpt_dir}")


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def resolve_output_file(save_path: str, ckpt_dir: Path) -> Path:
    p = Path(save_path).resolve()
    if p.suffix == ".pkl":
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{ckpt_dir.name}-strc.pkl"


def main():
    args = parse_args()

    data_path = Path(args.data_path).resolve()
    ckpt_dir = resolve_checkpoint_dir(args.pt_ckpt_dir_path)
    ckpt_path = ckpt_dir / "checkpoint.pt"
    model_config_path = resolve_model_config(ckpt_dir)
    output_file = resolve_output_file(args.save_path, ckpt_dir)

    if not data_path.exists():
        raise FileNotFoundError(data_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    print("=============================================================")
    print(f"Making predictions for {data_path}")
    print("=============================================================")
    print(f"Prediction made with {ckpt_dir.name}")
    print("=============================================================")

    df = pd.read_pickle(data_path)
    required_cols = {"id", "text", "eq_emb"}
    miss = required_cols - set(df.columns)
    if miss:
        raise ValueError(f"Prediction dataframe missing required columns for multimodal prediction: {sorted(miss)}")

    graph_dims = sorted({int(np.asarray(x).reshape(-1).shape[0]) for x in df["eq_emb"]})
    if len(graph_dims) != 1:
        raise ValueError(f"Inconsistent eq_emb dims in prediction dataset: {graph_dims}")
    graph_input_dim = int(graph_dims[0])

    model_cfg = load_yaml(model_config_path)
    pretrain_ckpt = model_cfg.get("Path", {}).get("pretrain_ckpt", "roberta-base")
    tokenizer = RobertaTokenizerFast.from_pretrained(pretrain_ckpt)

    dataset = MultimodalRegressionDataset(
        texts=df["text"].values,
        graph_emb=df["eq_emb"].values,
        ids=df["id"].tolist(),
        targets=df["target"].values if "target" in df.columns else None,
        tokenizer=tokenizer,
        seq_len=tokenizer.model_max_length,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = "cpu" if args.device == "cpu" or not torch.cuda.is_available() else "cuda"
    model = MultimodalRegressionModel(model_cfg, graph_input_dim=graph_input_dim).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)

    standardize_target = bool(state.get("standardize_target", False))
    target_mean = float(state.get("target_mean", 0.0))
    target_std = float(state.get("target_std", 1.0))
    if abs(target_std) < 1e-8:
        target_std = 1.0

    model.eval()
    pred_dict = {}
    with torch.no_grad():
        for batch in tqdm(loader):
            ids = batch["id"]
            batch = {k: v.to(device) for k, v in batch.items() if k != "id"}
            out = model(batch).squeeze(-1)
            if standardize_target:
                out = out * target_std + target_mean
            preds = out.detach().float().cpu().numpy().reshape(-1)
            for sid, pred in zip(ids, preds):
                pred_dict[str(sid)] = float(pred)

    with output_file.open("wb") as f:
        pickle.dump(pred_dict, f)

    print(f"[OK] raw prediction saved  -> {output_file}")


if __name__ == "__main__":
    main()
