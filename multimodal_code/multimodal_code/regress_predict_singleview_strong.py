#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regress_predict_singleview_strong.py

Prediction script for checkpoints produced by:
  regress_run_singleview_strong.py

Supports:
- validation / test prediction
- addH-out prediction
- optional metric JSON / merged CSV when target exists

Typical usage
-------------
python regress_predict_singleview_strong.py \
  --data-path singleview_local_data_cv_meta_v2/fold_0/nn_test.pkl \
  --ckpt-path singleview_strong_ckpts/run_xxx/checkpoint.pt \
  --save-dir pred_fold0_test \
  --device cuda \
  --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except Exception as e:
    raise SystemExit(f"scikit-learn is required: {e}")


# ---------------------------
# utilities
# ---------------------------
def save_json(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def metrics_from_df(df: pd.DataFrame, pred_col: str = "pred") -> Dict[str, float]:
    y_true = df["target"].to_numpy()
    y_pred = df[pred_col].to_numpy()
    return {
        "n": int(len(df)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


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

    @classmethod
    def from_state_dict(cls, state: dict):
        obj = cls(state["cat_cols"], state["num_cols"])
        obj.cat_vocab = state["cat_vocab"]
        obj.num_mean = state["num_mean"]
        obj.num_std = state["num_std"]
        return obj

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.cat_cols:
            if c not in out.columns:
                out[c] = "unknown"
            vocab = self.cat_vocab[c]
            out[c] = out[c].fillna("unknown").astype(str).map(lambda x: vocab.get(x, 0)).astype(int)
        for c in self.num_cols:
            if c not in out.columns:
                out[c] = np.nan
            s = pd.to_numeric(out[c], errors="coerce").fillna(self.num_mean.get(c, 0.0))
            out[c] = ((s - self.num_mean.get(c, 0.0)) / self.num_std.get(c, 1.0)).astype(np.float32)
        return out


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
        return " </s> ".join(parts) if parts else ""

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
        }

        if "target" in row.index and pd.notna(row["target"]):
            item["target"] = torch.tensor(float(row["target"]), dtype=torch.float)
        else:
            item["target"] = torch.tensor(float("nan"), dtype=torch.float)

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
            return out.last_hidden_state[:, 0, :]
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
# prediction
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True, help="nn_test.pkl / nn_val.pkl / addH_out_nn_pred_input.pkl")
    ap.add_argument("--ckpt-path", required=True, help="checkpoint.pt from regress_run_singleview_strong.py")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--write-pkl", action="store_true", help="Also save id->pred pickle")
    return ap.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ckpt = torch.load(args.ckpt_path, map_location=device)
    feature_state = ckpt["feature_state"]
    meta_proc = MetaPreprocessor.from_state_dict(feature_state["meta_preprocessor"])

    df = pd.read_pickle(args.data_path)
    df = meta_proc.transform(df.copy())

    tokenizer = AutoTokenizer.from_pretrained(feature_state["pretrain_ckpt"])
    dataset = SingleviewStrongDataset(
        df=df,
        tokenizer=tokenizer,
        cat_cols=feature_state["cat_cols"],
        num_cols=feature_state["num_cols"],
        use_text_raw=bool(feature_state["use_text_raw"]),
        use_text_structured=bool(feature_state["use_text_structured"]),
        max_length=int(feature_state["max_length"]),
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=2)

    model = SingleviewStrongModel(
        pretrain_ckpt=feature_state["pretrain_ckpt"],
        graph_dim=int(feature_state["graph_dim"]),
        cat_cardinalities=feature_state["cat_cardinalities"],
        num_dim=len(feature_state["num_cols"]),
        projection_dim=int(feature_state["projection_dim"]),
        meta_hidden_dim=int(feature_state["meta_hidden_dim"]),
        dropout=float(feature_state["dropout"]),
        n_site=int(feature_state["n_site"]),
        n_family=int(feature_state["n_family"]),
        n_target_bin=int(feature_state["n_target_bin"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    mean = float(ckpt.get("target_mean", 0.0))
    std = float(ckpt.get("target_std", 1.0))
    use_std = bool(ckpt.get("standardize_target", True))

    ids = []
    preds = []
    tgts = []
    with torch.no_grad():
        for batch in loader:
            batch_ids = batch.pop("id")
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            pred = out["pred"]
            if use_std:
                pred = pred * std + mean
            pred = pred.detach().cpu().numpy()

            tgt = batch["target"].detach().cpu().numpy()

            ids.extend(list(batch_ids))
            preds.extend(pred.tolist())
            tgts.extend(tgt.tolist())

    pred_df = pd.DataFrame({"id": ids, "pred": preds})
    if "target" in df.columns and pd.to_numeric(df["target"], errors="coerce").notna().any():
        tgt_series = pd.to_numeric(df["target"], errors="coerce")
        pred_df["target"] = tgt_series.to_numpy()

    pred_df.to_csv(save_dir / "predictions.csv", index=False)

    if args.write_pkl:
        import pickle
        pred_map = {str(i): float(p) for i, p in zip(ids, preds)}
        with (save_dir / "predictions.pkl").open("wb") as f:
            pickle.dump(pred_map, f)

    if "target" in pred_df.columns and pred_df["target"].notna().any():
        metrics = metrics_from_df(pred_df.dropna(subset=["target"]).copy(), pred_col="pred")
        save_json(save_dir / "metrics.json", metrics)

    feature_state_out = {
        "ckpt_path": str(Path(args.ckpt_path).resolve()),
        "data_path": str(Path(args.data_path).resolve()),
        "n_rows": int(len(pred_df)),
    }
    save_json(save_dir / "predict_info.json", feature_state_out)

    print("[OK] saved ->", save_dir / "predictions.csv")
    if (save_dir / "metrics.json").exists():
        print("[OK] metrics ->", save_dir / "metrics.json")


if __name__ == "__main__":
    main()
