#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_nn_residual_corrector.py

CatBoost + multiview two-way base + NN residual corrector.

Purpose
-------
Use CatBoost and multiview predictions as a strong two-way base predictor, then train
a tabular neural network to predict the residual:

    residual = target - pred_base

Final prediction:
    pred_final = pred_base + pred_residual_nn

Data assumptions
----------------
1) cv-root is produced by the singleview CV builder and contains:
   fold_0/nn_train.pkl
   fold_0/nn_val.pkl
   fold_0/nn_test.pkl
   fold_0/addH_out_nn_pred_input.pkl
   ...
   dataset_info.json

2) cat-root contains:
   test_pred_oof_ensemble.csv
   addH_out_pred_ensemble.csv
   OR the corresponding all-runs csvs

3) mv-root contains:
   test_pred_oof_ensemble.csv
   addH_out_pred_ensemble.csv
   OR the corresponding all-runs csvs

Outputs
-------
work-dir/
  fold0_seed42/
    base_ridge_info.json
    test_pred.csv
    addH_out_pred.csv
    test_base_pred.csv
    addH_out_base_pred.csv
    training_info.json
  ...
  test_pred_oof_ensemble.csv
  addH_out_pred_ensemble.csv
  test_pred_oof_base.csv
  addH_out_pred_base_ensemble.csv
  test_pred_oof_metrics.json
  test_pred_oof_base_metrics.json
  residual_summary.json

Recommended first run
---------------------
python train_nn_residual_corrector.py \
  --cv-root singleview_local_data_cv_meta_v2_mv \
  --cat-root singleview_catboost_runs_v2 \
  --mv-root multiview_cv_mm_merged_robust_aligned \
  --work-dir residual_nn_runs \
  --folds all \
  --seeds 42,52 \
  --graph-pca-dim 64 \
  --use-graph-scaler \
  --epochs 300 \
  --early-stop 30 \
  --batch-size 32 \
  --lr 1e-3 \
  --dropout 0.20 \
  --hidden-dims 256,128 \
  --base-method ridge \
  --base-ridge-alpha 1.0 \
  --residual-standardize \
  --loss-fn SmoothL1Loss

Notes
-----
- The two-way base model is fit INSIDE each fold using train split only.
- The residual NN is also fit using train split only and early-stopped on val.
- OOF final predictions come from each fold's test split predictions, aggregated across seeds.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-root", required=True)
    ap.add_argument("--cat-root", required=True)
    ap.add_argument("--mv-root", required=True)
    ap.add_argument("--work-dir", required=True)

    ap.add_argument("--folds", default="all")
    ap.add_argument("--seeds", default="42,52")

    ap.add_argument("--graph-pca-dim", type=int, default=64)
    ap.add_argument("--use-graph-scaler", action="store_true")

    ap.add_argument("--hidden-dims", default="256,128")
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--early-stop", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    ap.add_argument("--loss-fn", default="SmoothL1Loss", choices=["MSELoss", "L1Loss", "SmoothL1Loss"])
    ap.add_argument("--sample-weight-col", default="")

    ap.add_argument("--base-method", default="ridge", choices=["ridge", "mean", "weighted"])
    ap.add_argument("--base-ridge-alpha", type=float, default=1.0)
    ap.add_argument("--cat-weight", type=float, default=0.60)
    ap.add_argument("--mv-weight", type=float, default=0.40)

    ap.add_argument("--residual-standardize", action="store_true")
    ap.add_argument("--topk", type=int, default=20)
    return ap.parse_args()


def save_json(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_list_arg(raw: str) -> List[int]:
    raw = str(raw).strip()
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def list_fold_dirs(cv_root: Path) -> List[Path]:
    cands = [p for p in cv_root.iterdir() if p.is_dir() and p.name.startswith("fold_")]
    cands.sort(key=lambda p: int(p.name.split("_")[-1]))
    return cands


def metrics_from_df(df: pd.DataFrame, pred_col: str = "pred") -> Dict[str, float]:
    y_true = df["target"].to_numpy(dtype=float)
    y_pred = df[pred_col].to_numpy(dtype=float)
    return {
        "n": int(len(df)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _aggregate_all_runs_oof(df: pd.DataFrame) -> pd.DataFrame:
    need = {"id", "pred", "target"}
    if not need.issubset(df.columns):
        raise ValueError(f"All-runs OOF file missing columns: {sorted(need - set(df.columns))}")
    return df.groupby("id", as_index=False).agg(target=("target", "mean"), pred=("pred", "mean"))


def _aggregate_all_runs_out(df: pd.DataFrame) -> pd.DataFrame:
    need = {"id", "pred"}
    if not need.issubset(df.columns):
        raise ValueError(f"All-runs out file missing columns: {sorted(need - set(df.columns))}")
    return df.groupby("id", as_index=False).agg(pred=("pred", "mean"))


def _load_model_oof_out(root: Path, label: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    oof_candidates = [
        root / "test_pred_oof_ensemble.csv",
        root / "test_pred_all_runs.csv",
    ]
    out_candidates = [
        root / "addH_out_pred_ensemble.csv",
        root / "addH_out_pred_all_runs.csv",
    ]

    oof_df = None
    for p in oof_candidates:
        if p.exists():
            tmp = pd.read_csv(p)
            if p.name == "test_pred_all_runs.csv":
                tmp = _aggregate_all_runs_oof(tmp)
            else:
                need = {"id", "target", "pred"}
                if not need.issubset(tmp.columns):
                    raise ValueError(f"{p}: missing columns {sorted(need - set(tmp.columns))}")
                tmp = tmp[["id", "target", "pred"]].copy()
            oof_df = tmp
            break
    if oof_df is None:
        raise FileNotFoundError(f"{label}: could not find OOF file under {root}")

    out_df = None
    for p in out_candidates:
        if p.exists():
            tmp = pd.read_csv(p)
            if p.name == "addH_out_pred_all_runs.csv":
                tmp = _aggregate_all_runs_out(tmp)
            else:
                need = {"id", "pred"}
                if not need.issubset(tmp.columns):
                    raise ValueError(f"{p}: missing columns {sorted(need - set(tmp.columns))}")
                tmp = tmp[["id", "pred"]].copy()
            out_df = tmp
            break
    if out_df is None:
        raise FileNotFoundError(f"{label}: could not find addH-out file under {root}")

    oof_df["id"] = oof_df["id"].astype(str)
    out_df["id"] = out_df["id"].astype(str)
    oof_df = oof_df.rename(columns={"pred": f"pred_{label}"})
    out_df = out_df.rename(columns={"pred": f"pred_{label}"})
    return oof_df, out_df


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parse_hidden_dims(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError("hidden-dims must contain at least one int")
    return vals


class ResidualScaler:
    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self.mean = 0.0
        self.std = 1.0

    def fit(self, y: np.ndarray):
        if not self.enabled:
            self.mean = 0.0
            self.std = 1.0
            return self
        self.mean = float(np.mean(y))
        s = float(np.std(y))
        self.std = s if s > 1e-8 else 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return y.astype(np.float32)
        return ((y - self.mean) / self.std).astype(np.float32)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return y.astype(np.float32)
        return (y * self.std + self.mean).astype(np.float32)


class TabPreprocessor:
    def __init__(self, graph_pca_dim: int = 64, use_graph_scaler: bool = False):
        self.graph_pca_dim = int(graph_pca_dim)
        self.use_graph_scaler = bool(use_graph_scaler)
        self.graph_scaler = None
        self.graph_pca = None
        self.num_scaler = None
        self.num_fill_ = {}
        self.cat_maps_: Dict[str, Dict[str, int]] = {}
        self.cat_cols_: List[str] = []
        self.num_cols_: List[str] = []
        self.graph_cols_: List[str] = []

    def _stack_graph(self, s: pd.Series) -> np.ndarray:
        arrs = [np.asarray(x, dtype=np.float32).reshape(-1) for x in s.values]
        dims = sorted(set(a.shape[0] for a in arrs))
        if len(dims) != 1:
            raise ValueError(f"Inconsistent eq_emb dims: {dims}")
        return np.stack(arrs, axis=0)

    def fit(self, df_train: pd.DataFrame, num_cols: List[str], cat_cols: List[str]):
        self.num_cols_ = list(num_cols)
        self.cat_cols_ = list(cat_cols)

        # graph
        Xg = self._stack_graph(df_train["eq_emb"])
        if self.use_graph_scaler:
            self.graph_scaler = StandardScaler()
            Xg = self.graph_scaler.fit_transform(Xg)
        n_comp = int(min(self.graph_pca_dim, Xg.shape[0] - 1, Xg.shape[1]))
        n_comp = max(2, n_comp)
        self.graph_pca = PCA(n_components=n_comp, random_state=42)
        self.graph_pca.fit(Xg)
        self.graph_cols_ = [f"gpc_{i:03d}" for i in range(n_comp)]

        # numeric
        num_df = df_train[self.num_cols_].copy()
        for c in self.num_cols_:
            num_df[c] = pd.to_numeric(num_df[c], errors="coerce")
            fill = float(num_df[c].median()) if num_df[c].notna().any() else 0.0
            self.num_fill_[c] = fill
            num_df[c] = num_df[c].fillna(fill)
        self.num_scaler = StandardScaler()
        self.num_scaler.fit(num_df.to_numpy(dtype=np.float32))

        # categoricals
        self.cat_maps_ = {}
        for c in self.cat_cols_:
            vals = df_train[c].fillna("unknown").astype(str).tolist()
            vocab = {"<UNK>": 0}
            for v in vals:
                if v not in vocab:
                    vocab[v] = len(vocab)
            self.cat_maps_[c] = vocab
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # graph
        Xg = self._stack_graph(df["eq_emb"])
        if self.graph_scaler is not None:
            Xg = self.graph_scaler.transform(Xg)
        Xg = self.graph_pca.transform(Xg)

        # numeric
        num_df = df[self.num_cols_].copy()
        for c in self.num_cols_:
            num_df[c] = pd.to_numeric(num_df[c], errors="coerce").fillna(self.num_fill_[c])
        Xn = self.num_scaler.transform(num_df.to_numpy(dtype=np.float32))

        # concat numeric + graph
        X_num = np.concatenate([Xn, Xg.astype(np.float32)], axis=1).astype(np.float32)

        # categoricals to int codes
        cat_arrays = []
        for c in self.cat_cols_:
            vocab = self.cat_maps_[c]
            arr = df[c].fillna("unknown").astype(str).map(lambda x: vocab.get(x, 0)).to_numpy(dtype=np.int64)
            cat_arrays.append(arr.reshape(-1, 1))
        X_cat = np.concatenate(cat_arrays, axis=1).astype(np.int64) if cat_arrays else np.zeros((len(df), 0), dtype=np.int64)
        return X_num, X_cat


class ResidualDataset(Dataset):
    def __init__(self, X_num: np.ndarray, X_cat: np.ndarray, residual: np.ndarray, target: np.ndarray, pred_base: np.ndarray, sample_weight: np.ndarray):
        self.X_num = X_num.astype(np.float32)
        self.X_cat = X_cat.astype(np.int64)
        self.residual = residual.astype(np.float32)
        self.target = target.astype(np.float32)
        self.pred_base = pred_base.astype(np.float32)
        self.sample_weight = sample_weight.astype(np.float32)

    def __len__(self):
        return len(self.residual)

    def __getitem__(self, idx):
        return {
            "x_num": torch.tensor(self.X_num[idx], dtype=torch.float32),
            "x_cat": torch.tensor(self.X_cat[idx], dtype=torch.long),
            "residual": torch.tensor(self.residual[idx], dtype=torch.float32),
            "target": torch.tensor(self.target[idx], dtype=torch.float32),
            "pred_base": torch.tensor(self.pred_base[idx], dtype=torch.float32),
            "sample_weight": torch.tensor(self.sample_weight[idx], dtype=torch.float32),
        }


class ResidualMLP(nn.Module):
    def __init__(self, num_dim: int, cat_cardinalities: List[int], hidden_dims: List[int], dropout: float = 0.2):
        super().__init__()
        self.cat_cardinalities = list(cat_cardinalities)
        self.emb_layers = nn.ModuleList()
        emb_dims = []
        for card in self.cat_cardinalities:
            emb_dim = min(32, max(4, int(round(math.sqrt(card + 1)))))
            self.emb_layers.append(nn.Embedding(int(card), int(emb_dim)))
            emb_dims.append(int(emb_dim))

        in_dim = int(num_dim + sum(emb_dims))
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, int(h)),
                nn.BatchNorm1d(int(h)),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
            ])
            prev = int(h)
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        feats = [x_num]
        if x_cat.shape[1] > 0:
            cat_embs = []
            for j, emb in enumerate(self.emb_layers):
                cat_embs.append(emb(x_cat[:, j]))
            feats.append(torch.cat(cat_embs, dim=1))
        x = torch.cat(feats, dim=1)
        return self.mlp(x).squeeze(-1)


def loss_none(loss_name: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if loss_name == "MSELoss":
        return (pred - target) ** 2
    if loss_name == "L1Loss":
        return torch.abs(pred - target)
    if loss_name == "SmoothL1Loss":
        return torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    raise ValueError(f"Unsupported loss: {loss_name}")


def weighted_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    w = w.float().clamp(min=0.0)
    denom = torch.clamp(w.sum(), min=1e-12)
    return (x * w).sum() / denom


def _fit_base_predictor(train_df: pd.DataFrame, args) -> Dict[str, object]:
    X = train_df[["pred_cat", "pred_mv"]].to_numpy(dtype=float)
    y = train_df["target"].to_numpy(dtype=float)

    if args.base_method == "mean":
        return {"method": "mean"}

    if args.base_method == "weighted":
        s = float(args.cat_weight + args.mv_weight)
        wc, wm = float(args.cat_weight) / s, float(args.mv_weight) / s
        return {"method": "weighted", "cat_weight": wc, "mv_weight": wm}

    model = Ridge(alpha=float(args.base_ridge_alpha), fit_intercept=True)
    model.fit(X, y)
    return {
        "method": "ridge",
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "model": model,
        "alpha": float(args.base_ridge_alpha),
    }


def _apply_base_predictor(df: pd.DataFrame, info: Dict[str, object]) -> np.ndarray:
    Xc = df["pred_cat"].to_numpy(dtype=float)
    Xm = df["pred_mv"].to_numpy(dtype=float)
    if info["method"] == "mean":
        return ((Xc + Xm) / 2.0).astype(np.float32)
    if info["method"] == "weighted":
        return (float(info["cat_weight"]) * Xc + float(info["mv_weight"]) * Xm).astype(np.float32)
    model = info["model"]
    X = df[["pred_cat", "pred_mv"]].to_numpy(dtype=float)
    return model.predict(X).astype(np.float32)


def _extract_sample_weight(df: pd.DataFrame, col: str) -> np.ndarray:
    if not col or col not in df.columns:
        return np.ones(len(df), dtype=np.float32)
    s = pd.to_numeric(df[col], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
    s = np.where(s > 0, s, 1.0)
    return s.astype(np.float32)


def _train_one_fold_seed(
    fold_idx: int,
    seed: int,
    fold_dir: Path,
    args,
    info: dict,
    cat_oof: pd.DataFrame,
    mv_oof: pd.DataFrame,
    cat_out: pd.DataFrame,
    mv_out: pd.DataFrame,
    work_dir: Path,
):
    set_seed(seed)
    run_dir = work_dir / f"fold{fold_idx}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tr = pd.read_pickle(fold_dir / "nn_train.pkl").copy()
    va = pd.read_pickle(fold_dir / "nn_val.pkl").copy()
    te = pd.read_pickle(fold_dir / "nn_test.pkl").copy()
    out_df = pd.read_pickle(fold_dir / "addH_out_nn_pred_input.pkl").copy()

    for df in [tr, va, te, out_df]:
        df["id"] = df["id"].astype(str)

    # merge base model preds
    tr = tr.merge(cat_oof[["id", "pred_cat"]], on="id", how="inner").merge(mv_oof[["id", "pred_mv"]], on="id", how="inner")
    va = va.merge(cat_oof[["id", "pred_cat"]], on="id", how="inner").merge(mv_oof[["id", "pred_mv"]], on="id", how="inner")
    te = te.merge(cat_oof[["id", "pred_cat"]], on="id", how="inner").merge(mv_oof[["id", "pred_mv"]], on="id", how="inner")
    out_df = out_df.merge(cat_out[["id", "pred_cat"]], on="id", how="inner").merge(mv_out[["id", "pred_mv"]], on="id", how="inner")

    # fit two-way base inside fold
    base_info = _fit_base_predictor(tr, args)
    tr["pred_base"] = _apply_base_predictor(tr, base_info)
    va["pred_base"] = _apply_base_predictor(va, base_info)
    te["pred_base"] = _apply_base_predictor(te, base_info)
    out_df["pred_base"] = _apply_base_predictor(out_df, base_info)

    # residual target
    tr["residual_target"] = tr["target"].astype(float) - tr["pred_base"].astype(float)
    va["residual_target"] = va["target"].astype(float) - va["pred_base"].astype(float)
    te["residual_target"] = te["target"].astype(float) - te["pred_base"].astype(float)

    # features
    cat_cols = list(info.get("cat_cols", []))
    num_cols = list(info.get("num_cols", []))
    # keep only existing columns
    cat_cols = [c for c in cat_cols if c in tr.columns]
    num_cols = [c for c in num_cols if c in tr.columns]
    # base predictions are always numeric features
    for c in ["pred_cat", "pred_mv", "pred_base"]:
        if c not in num_cols:
            num_cols.append(c)

    pre = TabPreprocessor(graph_pca_dim=int(args.graph_pca_dim), use_graph_scaler=bool(args.use_graph_scaler))
    pre.fit(tr, num_cols=num_cols, cat_cols=cat_cols)

    Xn_tr, Xc_tr = pre.transform(tr)
    Xn_va, Xc_va = pre.transform(va)
    Xn_te, Xc_te = pre.transform(te)
    Xn_out, Xc_out = pre.transform(out_df)

    resid_scaler = ResidualScaler(enabled=bool(args.residual_standardize)).fit(tr["residual_target"].to_numpy(dtype=np.float32))
    y_tr_res = resid_scaler.transform(tr["residual_target"].to_numpy(dtype=np.float32))
    y_va_res = resid_scaler.transform(va["residual_target"].to_numpy(dtype=np.float32))
    y_te_res = resid_scaler.transform(te["residual_target"].to_numpy(dtype=np.float32))

    w_tr = _extract_sample_weight(tr, args.sample_weight_col)
    w_va = _extract_sample_weight(va, args.sample_weight_col)
    w_te = _extract_sample_weight(te, args.sample_weight_col)

    ds_tr = ResidualDataset(Xn_tr, Xc_tr, y_tr_res, tr["target"].to_numpy(dtype=np.float32), tr["pred_base"].to_numpy(dtype=np.float32), w_tr)
    ds_va = ResidualDataset(Xn_va, Xc_va, y_va_res, va["target"].to_numpy(dtype=np.float32), va["pred_base"].to_numpy(dtype=np.float32), w_va)
    ds_te = ResidualDataset(Xn_te, Xc_te, y_te_res, te["target"].to_numpy(dtype=np.float32), te["pred_base"].to_numpy(dtype=np.float32), w_te)
    ds_out = ResidualDataset(Xn_out, Xc_out, np.zeros(len(out_df), dtype=np.float32), np.zeros(len(out_df), dtype=np.float32), out_df["pred_base"].to_numpy(dtype=np.float32), np.ones(len(out_df), dtype=np.float32))

    dl_tr = DataLoader(ds_tr, batch_size=int(args.batch_size), shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=int(args.batch_size), shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=int(args.batch_size), shuffle=False)
    dl_out = DataLoader(ds_out, batch_size=int(args.batch_size), shuffle=False)

    device = "cpu" if args.device == "cpu" or not torch.cuda.is_available() else "cuda"
    model = ResidualMLP(
        num_dim=Xn_tr.shape[1],
        cat_cardinalities=[len(pre.cat_maps_[c]) for c in cat_cols],
        hidden_dims=_parse_hidden_dims(args.hidden_dims),
        dropout=float(args.dropout),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_state = None
    best_val_mae = float("inf")
    no_improve = 0
    train_curve = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        tr_losses = []
        for batch in dl_tr:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            pred_res_z = model(batch["x_num"], batch["x_cat"])
            per = loss_none(args.loss_fn, pred_res_z, batch["residual"])
            loss = weighted_mean(per, batch["sample_weight"])
            loss.backward()
            optimizer.step()
            tr_losses.append(float(loss.item()))

        # validation on final prediction
        model.eval()
        val_preds_final = []
        val_targets = []
        with torch.no_grad():
            for batch in dl_va:
                batch = {k: v.to(device) for k, v in batch.items()}
                pred_res_z = model(batch["x_num"], batch["x_cat"]).detach().cpu().numpy()
                pred_res = resid_scaler.inverse(pred_res_z)
                pred_final = batch["pred_base"].detach().cpu().numpy() + pred_res
                val_preds_final.append(pred_final.reshape(-1))
                val_targets.append(batch["target"].detach().cpu().numpy().reshape(-1))

        y_val = np.concatenate(val_targets)
        p_val = np.concatenate(val_preds_final)
        val_mae = float(mean_absolute_error(y_val, p_val))
        train_curve.append({"epoch": int(epoch), "train_loss": float(np.mean(tr_losses)), "val_mae": val_mae})

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            no_improve = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= int(args.early_stop):
                break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    # save preprocessors
    joblib.dump(pre, run_dir / "tab_preprocessor.joblib")
    joblib.dump(resid_scaler, run_dir / "residual_scaler.joblib")
    torch.save(model.state_dict(), run_dir / "residual_model.pt")

    # save base info without sklearn model object
    base_info_json = {k: v for k, v in base_info.items() if k != "model"}
    save_json(run_dir / "base_ridge_info.json", base_info_json)

    def _predict_loader(dl, ids, has_target: bool, target_arr=None):
        preds_res, preds_final = [], []
        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                pred_res_z = model(batch["x_num"], batch["x_cat"]).detach().cpu().numpy()
                pred_res = resid_scaler.inverse(pred_res_z)
                pred_final = batch["pred_base"].detach().cpu().numpy() + pred_res
                preds_res.append(pred_res.reshape(-1))
                preds_final.append(pred_final.reshape(-1))
        pred_res = np.concatenate(preds_res)
        pred_final = np.concatenate(preds_final)
        df = pd.DataFrame({"id": [str(x) for x in ids], "pred_residual": pred_res, "pred": pred_final})
        if has_target and target_arr is not None:
            df["target"] = target_arr
        return df

    te_pred = _predict_loader(dl_te, te["id"].tolist(), has_target=True, target_arr=te["target"].to_numpy(dtype=np.float32))
    out_pred = _predict_loader(dl_out, out_df["id"].tolist(), has_target=False)

    te_base = te[["id", "target", "pred_base"]].rename(columns={"pred_base": "pred"}).copy()
    out_base = out_df[["id", "pred_base"]].rename(columns={"pred_base": "pred"}).copy()

    te_pred["fold"] = int(fold_idx)
    te_pred["seed"] = int(seed)
    out_pred["fold"] = int(fold_idx)
    out_pred["seed"] = int(seed)

    te_base["fold"] = int(fold_idx)
    te_base["seed"] = int(seed)
    out_base["fold"] = int(fold_idx)
    out_base["seed"] = int(seed)

    te_pred.to_csv(run_dir / "test_pred.csv", index=False)
    out_pred.to_csv(run_dir / "addH_out_pred.csv", index=False)
    te_base.to_csv(run_dir / "test_base_pred.csv", index=False)
    out_base.to_csv(run_dir / "addH_out_base_pred.csv", index=False)

    training_info = {
        "fold": int(fold_idx),
        "seed": int(seed),
        "best_val_mae": float(best_val_mae),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "n_out": int(len(out_df)),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "train_curve": train_curve,
    }
    save_json(run_dir / "training_info.json", training_info)

    return te_pred, out_pred, te_base, out_base


def main():
    args = parse_args()
    cv_root = Path(args.cv_root).resolve()
    cat_root = Path(args.cat_root).resolve()
    mv_root = Path(args.mv_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    info_path = cv_root / "dataset_info.json"
    if not info_path.exists():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))

    cat_oof, cat_out = _load_model_oof_out(cat_root, "cat")
    mv_oof, mv_out = _load_model_oof_out(mv_root, "mv")

    folds_all = list_fold_dirs(cv_root)
    if not folds_all:
        raise FileNotFoundError(f"No fold_* dirs under {cv_root}")

    if args.folds == "all":
        fold_dirs = folds_all
    else:
        wanted = set(parse_list_arg(args.folds))
        fold_dirs = [p for p in folds_all if int(p.name.split("_")[-1]) in wanted]
        if not fold_dirs:
            raise ValueError(f"No requested folds found under {cv_root}: {sorted(wanted)}")

    seeds = parse_list_arg(args.seeds)
    if not seeds:
        raise ValueError("No seeds parsed from --seeds")

    test_parts = []
    out_parts = []
    test_base_parts = []
    out_base_parts = []

    for fold_dir in fold_dirs:
        fold_idx = int(fold_dir.name.split("_")[-1])
        for seed in seeds:
            te_pred, out_pred, te_base, out_base = _train_one_fold_seed(
                fold_idx=fold_idx,
                seed=seed,
                fold_dir=fold_dir,
                args=args,
                info=info,
                cat_oof=cat_oof,
                mv_oof=mv_oof,
                cat_out=cat_out,
                mv_out=mv_out,
                work_dir=work_dir,
            )
            test_parts.append(te_pred)
            out_parts.append(out_pred)
            test_base_parts.append(te_base)
            out_base_parts.append(out_base)

    # all-runs
    test_all = pd.concat(test_parts, axis=0, ignore_index=True)
    out_all = pd.concat(out_parts, axis=0, ignore_index=True)
    test_base_all = pd.concat(test_base_parts, axis=0, ignore_index=True)
    out_base_all = pd.concat(out_base_parts, axis=0, ignore_index=True)

    test_all.to_csv(work_dir / "test_pred_all_runs.csv", index=False)
    out_all.to_csv(work_dir / "addH_out_pred_all_runs.csv", index=False)
    test_base_all.to_csv(work_dir / "test_pred_base_all_runs.csv", index=False)
    out_base_all.to_csv(work_dir / "addH_out_pred_base_all_runs.csv", index=False)

    # aggregate across seeds for OOF
    test_ens = test_all.groupby("id", as_index=False).agg(target=("target", "mean"), pred=("pred", "mean"))
    out_ens = out_all.groupby("id", as_index=False).agg(pred=("pred", "mean"))
    test_base_ens = test_base_all.groupby("id", as_index=False).agg(target=("target", "mean"), pred=("pred", "mean"))
    out_base_ens = out_base_all.groupby("id", as_index=False).agg(pred=("pred", "mean"))

    test_ens.to_csv(work_dir / "test_pred_oof_ensemble.csv", index=False)
    out_ens.to_csv(work_dir / "addH_out_pred_ensemble.csv", index=False)
    test_base_ens.to_csv(work_dir / "test_pred_oof_base.csv", index=False)
    out_base_ens.to_csv(work_dir / "addH_out_pred_base_ensemble.csv", index=False)

    oof_metrics = metrics_from_df(test_ens, pred_col="pred")
    base_metrics = metrics_from_df(test_base_ens, pred_col="pred")
    save_json(work_dir / "test_pred_oof_metrics.json", oof_metrics)
    save_json(work_dir / "test_pred_oof_base_metrics.json", base_metrics)

    out_ens.sort_values("pred").head(args.topk).to_csv(work_dir / "addH_out_top20_low.csv", index=False)
    out_ens.sort_values("pred", ascending=False).head(args.topk).to_csv(work_dir / "addH_out_top20_high.csv", index=False)

    summary = {
        "cv_root": str(cv_root),
        "cat_root": str(cat_root),
        "mv_root": str(mv_root),
        "folds": [int(p.name.split("_")[-1]) for p in fold_dirs],
        "seeds": seeds,
        "base_method": args.base_method,
        "base_ridge_alpha": float(args.base_ridge_alpha),
        "graph_pca_dim": int(args.graph_pca_dim),
        "use_graph_scaler": bool(args.use_graph_scaler),
        "hidden_dims": _parse_hidden_dims(args.hidden_dims),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "early_stop": int(args.early_stop),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "loss_fn": args.loss_fn,
        "sample_weight_col": args.sample_weight_col,
        "residual_standardize": bool(args.residual_standardize),
        "base_oof_metrics": base_metrics,
        "final_oof_metrics": oof_metrics,
    }
    save_json(work_dir / "residual_summary.json", summary)

    print("[DONE] residual corrector finished ->", work_dir)
    print("[INFO] base OOF metrics  =", base_metrics)
    print("[INFO] final OOF metrics =", oof_metrics)


if __name__ == "__main__":
    main()
