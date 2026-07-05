#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import RobertaTokenizerFast

from model.modules import TextEncoder, ProjectionHead


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
    def __init__(self, texts, graph_emb, ids=None, targets=None, tokenizer=None, seq_len=256):
        self.texts = texts
        self.graph_emb = graph_emb
        self.ids = ids if ids is not None else list(range(len(texts)))
        self.targets = np.asarray(targets, dtype=np.float32) if targets is not None else np.zeros(len(texts), dtype=np.float32)
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)

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



class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None, depth: int = 2, dropout: float = 0.1, residual: bool = False):
        super().__init__()
        input_dim = int(input_dim)
        output_dim = int(output_dim)
        hidden_dim = int(hidden_dim or output_dim)
        depth = max(1, int(depth))
        self.residual = bool(residual and input_dim == output_dim)

        layers = []
        if depth == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)])
            for _ in range(depth - 2):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)])
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(output_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        y = self.net(x)
        if self.residual:
            y = y + x
        return self.norm(y)


class GraphProjectionHead(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int, dropout_rate: float = 0.1, depth: int = 2, hidden_mult: float = 1.0):
        super().__init__()
        hidden_dim = max(int(projection_dim), int(float(hidden_mult) * int(projection_dim)))
        self.proj = nn.Linear(int(input_dim), int(projection_dim))
        if int(depth) <= 1:
            self.net = nn.Identity()
        else:
            self.net = MLPBlock(
                input_dim=int(projection_dim),
                output_dim=int(projection_dim),
                hidden_dim=hidden_dim,
                depth=max(1, int(depth) - 1),
                dropout=dropout_rate,
                residual=True,
            )
        self.norm = nn.LayerNorm(int(projection_dim))
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            torch.nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        projected = self.proj(x)
        h = self.net(projected)
        if isinstance(self.net, nn.Identity):
            h = projected
        return self.norm(h)


class FusionRegressor(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1, depth: int = 2):
        super().__init__()
        hidden_dim = hidden_dim or dim
        depth = max(1, int(depth))
        layers = [nn.Dropout(dropout)]
        if depth <= 1:
            layers.append(nn.Linear(dim, 1))
        else:
            layers.extend([nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)])
            for _ in range(depth - 2):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)])
            layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
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
    """AddH-specific multimodal regressor with swappable fusion heads.

    model_variant options:
      - gated_sum: original gated weighted average of text and graph embeddings.
      - concat_interact: MLP over [text, graph, |text-graph|, text*graph].
      - residual_graph: text embedding plus learned graph/text residual.
      - graph_only: graph branch only, useful as a diagnostic model.
      - text_only: text branch only, useful as a diagnostic model.
    """
    def __init__(self, config: dict, graph_input_dim: int):
        super().__init__()
        self.text_encoder = TextEncoder(config)
        self.text_projection = ProjectionHead(config)

        projection_dim = int(config["ProjectionConfig"]["projection_dim"])
        dropout_rate = float(config["ProjectionConfig"].get("dropout_rate", 0.1))
        addh_cfg = config.get("AddHRegressConfig", {}) or {}
        self.model_variant = str(addh_cfg.get("model_variant", "gated_sum"))
        self.graph_noise_std = float(addh_cfg.get("graph_noise_std", 0.0))
        graph_proj_depth = int(addh_cfg.get("graph_proj_depth", 2))
        graph_hidden_mult = float(addh_cfg.get("graph_hidden_mult", 1.0))
        fusion_hidden_mult = float(addh_cfg.get("fusion_hidden_mult", 2.0))
        regressor_hidden_mult = float(addh_cfg.get("regressor_hidden_mult", 1.0))
        regressor_depth = int(addh_cfg.get("regressor_depth", 2))

        self.graph_projection = GraphProjectionHead(
            input_dim=graph_input_dim,
            projection_dim=projection_dim,
            dropout_rate=dropout_rate,
            depth=graph_proj_depth,
            hidden_mult=graph_hidden_mult,
        )

        fusion_hidden = max(projection_dim, int(projection_dim * fusion_hidden_mult))
        self.fusion_gate = nn.Sequential(nn.Linear(projection_dim * 2, projection_dim), nn.Sigmoid())
        self.concat_fusion = MLPBlock(
            input_dim=projection_dim * 4,
            output_dim=projection_dim,
            hidden_dim=fusion_hidden,
            depth=2,
            dropout=dropout_rate,
            residual=False,
        )
        self.residual_fusion = MLPBlock(
            input_dim=projection_dim * 4,
            output_dim=projection_dim,
            hidden_dim=fusion_hidden,
            depth=2,
            dropout=dropout_rate,
            residual=False,
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.regressor = FusionRegressor(
            dim=projection_dim,
            hidden_dim=max(projection_dim, int(projection_dim * regressor_hidden_mult)),
            dropout=dropout_rate,
            depth=regressor_depth,
        )

    def _maybe_noisy_graph(self, g):
        if self.training and self.graph_noise_std > 0:
            return g + torch.randn_like(g) * self.graph_noise_std
        return g

    def _encode_branches(self, batch):
        text_features = self.text_encoder(batch)
        text_embeddings = self.text_projection(text_features)
        graph_input = self._maybe_noisy_graph(batch["graph_embed"])
        graph_embeddings = self.graph_projection(graph_input)
        return text_embeddings, graph_embeddings

    def encode_fused(self, batch):
        text_embeddings, graph_embeddings = self._encode_branches(batch)
        variant = self.model_variant

        if variant == "text_only":
            return text_embeddings
        if variant == "graph_only":
            return graph_embeddings

        diff = torch.abs(text_embeddings - graph_embeddings)
        prod = text_embeddings * graph_embeddings
        concat = torch.cat([text_embeddings, graph_embeddings, diff, prod], dim=-1)

        if variant == "concat_interact":
            return self.concat_fusion(concat)
        if variant == "residual_graph":
            # small learned residual around text branch; safer on small data than a fully free fusion head
            return text_embeddings + torch.tanh(self.residual_scale) * self.residual_fusion(concat)

        # default: original gated sum, kept for compatibility
        gate = self.fusion_gate(torch.cat([text_embeddings, graph_embeddings], dim=-1))
        fused = gate * text_embeddings + (1.0 - gate) * graph_embeddings
        return fused

    def forward(self, batch):
        fused = self.encode_fused(batch)
        return self.regressor(fused)


def parse_args():
    ap = argparse.ArgumentParser(description="Predict with multimodal regression model (text + eq_emb).")
    ap.add_argument("--data_path", required=True, help="Path to regress_test.pkl or addH_out_pred_input.pkl")
    ap.add_argument("--pt_ckpt_dir_path", required=True, help="Checkpoint directory containing checkpoint.pt and copied clip.yml")
    ap.add_argument("--save_path", required=True, help="Save path. If no .pkl suffix, treated as output directory.")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--text_col", default=None, help="Optional override for preferred text column")
    ap.add_argument("--seq_len", type=int, default=None, help="Optional override for tokenizer sequence length")
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--save_latent_path", default=None, help="Optional output pickle for id -> fused embedding")
    return ap.parse_args()


def resolve_checkpoint_dir(path_str: str) -> Path:
    p = Path(path_str).resolve()
    if p.is_dir():
        return p
    raise FileNotFoundError(f"Checkpoint directory not found: {p}")


def resolve_model_config(ckpt_dir: Path) -> Path:
    candidates = [ckpt_dir / "clip.yml", ckpt_dir / "model__clip.yml", ckpt_dir / "model" / "clip.yml"]
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


def resolve_latent_output_file(save_path: Optional[str], ckpt_dir: Path) -> Optional[Path]:
    if save_path is None:
        return None
    p = Path(save_path).resolve()
    if p.suffix == ".pkl":
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{ckpt_dir.name}-fused.pkl"


def main():
    args = parse_args()

    data_path = Path(args.data_path).resolve()
    ckpt_dir = resolve_checkpoint_dir(args.pt_ckpt_dir_path)
    ckpt_path = ckpt_dir / "checkpoint.pt"
    model_config_path = resolve_model_config(ckpt_dir)
    output_file = resolve_output_file(args.save_path, ckpt_dir)
    latent_output_file = resolve_latent_output_file(args.save_latent_path, ckpt_dir)

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
    required_cols = {"id", "eq_emb"}
    miss = required_cols - set(df.columns)
    if miss:
        raise ValueError(f"Prediction dataframe missing required columns: {sorted(miss)}")

    state = torch.load(ckpt_path, map_location="cpu")
    train_cfg = state.get("train_config", {}) if isinstance(state, dict) else {}
    preferred_text_col = args.text_col or train_cfg.get("text_col", "text")
    concat_text_cols = train_cfg.get("concat_text_cols", ["text_structured", "text_raw", "text"])
    seq_len = int(args.seq_len or train_cfg.get("seq_len", 256))
    num_workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 2))

    df = df.copy()
    df["text"] = _build_text_series(df, preferred_col=preferred_text_col, concat_cols=concat_text_cols)

    graph_dims = sorted({int(np.asarray(x).reshape(-1).shape[0]) for x in df["eq_emb"]})
    if len(graph_dims) != 1:
        raise ValueError(f"Inconsistent eq_emb dims in prediction dataset: {graph_dims}")
    graph_input_dim = int(graph_dims[0])

    model_cfg = load_yaml(model_config_path)
    model_cfg["AddHRegressConfig"] = {
        "model_variant": str(train_cfg.get("regress_model_variant", train_cfg.get("model_variant", "gated_sum"))),
        "fusion_hidden_mult": float(train_cfg.get("fusion_hidden_mult", 2.0)),
        "regressor_hidden_mult": float(train_cfg.get("regressor_hidden_mult", 1.0)),
        "regressor_depth": int(train_cfg.get("regressor_depth", 2)),
        "graph_proj_depth": int(train_cfg.get("graph_proj_depth", 2)),
        "graph_hidden_mult": float(train_cfg.get("graph_hidden_mult", 1.0)),
        "graph_noise_std": 0.0,
    }
    pretrain_ckpt = model_cfg.get("Path", {}).get("pretrain_ckpt", "roberta-base")
    tokenizer = RobertaTokenizerFast.from_pretrained(pretrain_ckpt)

    dataset = MultimodalRegressionDataset(
        texts=df["text"].values,
        graph_emb=df["eq_emb"].values,
        ids=df["id"].tolist(),
        targets=df["target"].values if "target" in df.columns else None,
        tokenizer=tokenizer,
        seq_len=seq_len,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    device = "cpu" if args.device == "cpu" or not torch.cuda.is_available() else "cuda"
    model = MultimodalRegressionModel(model_cfg, graph_input_dim=graph_input_dim).to(device)
    model.load_state_dict(state["model_state_dict"], strict=False)

    standardize_target = bool(state.get("standardize_target", False))
    target_mean = float(state.get("target_mean", 0.0))
    target_std = float(state.get("target_std", 1.0))
    if abs(target_std) < 1e-8:
        target_std = 1.0

    model.eval()
    pred_dict = {}
    fused_dict = {} if latent_output_file is not None else None

    with torch.no_grad():
        for batch in tqdm(loader):
            ids = batch["id"]
            batch = {k: v.to(device) for k, v in batch.items() if k != "id"}
            fused = model.encode_fused(batch)
            out = model(batch).squeeze(-1)
            if standardize_target:
                out = out * target_std + target_mean
            preds = out.detach().float().cpu().numpy().reshape(-1)
            fused_np = fused.detach().float().cpu().numpy()
            for i, (sid, pred) in enumerate(zip(ids, preds)):
                pred_dict[str(sid)] = float(pred)
                if fused_dict is not None:
                    fused_dict[str(sid)] = fused_np[i].astype(np.float32)

    with output_file.open("wb") as f:
        pickle.dump(pred_dict, f)
    print(f"[OK] raw prediction saved  -> {output_file}")

    if latent_output_file is not None and fused_dict is not None:
        with latent_output_file.open("wb") as f:
            pickle.dump(fused_dict, f)
        print(f"[OK] fused embedding saved -> {latent_output_file}")


if __name__ == "__main__":
    main()
