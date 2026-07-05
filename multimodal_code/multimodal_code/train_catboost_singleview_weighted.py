#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_catboost_singleview_weighted.py

CatBoost single-structure training with optional sample weights.

New capabilities
----------------
1) Read sample weights directly from split pkls if a column like `w_domain` exists
2) Or load a separate CSV and merge `id -> w_domain` onto train/val/test/addH-out
3) Pass weights to CatBoost Pool for train / validation

Recommended usage
-----------------
python train_catboost_singleview_weighted.py \
  --cv-root singleview_local_data_cv_meta_v2 \
  --work-dir singleview_catboost_runs_weighted \
  --folds all \
  --seeds 42,52 \
  --graph-pca-dim 64 \
  --use-graph-scaler \
  --depth 6 \
  --learning-rate 0.03 \
  --l2-leaf-reg 8 \
  --iterations 4000 \
  --early-stopping-rounds 200 \
  --ensemble-method mean \
  --device CPU \
  --weight-col w_domain \
  --weight-map-csv addH_master_target_weighted.csv

Notes
-----
- If `w_domain` is already present in cat_train.pkl / cat_val.pkl / cat_test.pkl,
  this script uses it directly.
- If not, but `--weight-map-csv` is provided, the script merges weights by `id`.
- If neither is available, it falls back to unweighted training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from catboost import CatBoostRegressor, Pool
except Exception as e:
    raise SystemExit(f"catboost is required: {e}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-root", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--folds", default="all")
    ap.add_argument("--seeds", default="42,52")
    ap.add_argument("--ensemble-method", default="mean", choices=["mean", "median", "val_mae_weighted"])
    ap.add_argument("--topk", type=int, default=20)

    # graph features
    ap.add_argument("--graph-pca-dim", type=int, default=64)
    ap.add_argument("--use-graph-scaler", action="store_true")

    # catboost params
    ap.add_argument("--loss-function", default="MAE")
    ap.add_argument("--eval-metric", default="MAE")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--l2-leaf-reg", type=float, default=8.0)
    ap.add_argument("--iterations", type=int, default=4000)
    ap.add_argument("--random-strength", type=float, default=1.0)
    ap.add_argument("--bagging-temperature", type=float, default=0.0)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--early-stopping-rounds", type=int, default=200)
    ap.add_argument("--thread-count", type=int, default=-1)
    ap.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    ap.add_argument("--gpu-devices", default="0")

    # optional calibration
    ap.add_argument("--use-val-calibration", action="store_true")
    ap.add_argument("--calibration-mode", default="bias_only", choices=["bias_only", "affine"])
    ap.add_argument("--min-val-weight", type=float, default=1e-6)

    # weighting
    ap.add_argument("--weight-col", default="w_domain", help="Sample-weight column name")
    ap.add_argument("--weight-map-csv", default=None, help="Optional CSV containing id -> weight_col mapping")
    ap.add_argument("--use-val-weights", action="store_true", help="Also apply weights to validation Pool if available")
    ap.add_argument("--clip-weight-min", type=float, default=0.0, help="Optional lower bound applied to weights")
    ap.add_argument("--clip-weight-max", type=float, default=0.0, help="Optional upper bound applied to weights; <=0 disables")
    return ap.parse_args()


def parse_list_arg(raw: str) -> List[int]:
    raw = str(raw).strip()
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def list_fold_dirs(cv_root: Path) -> List[Path]:
    cands = [p for p in cv_root.iterdir() if p.is_dir() and p.name.startswith("fold_")]
    cands.sort(key=lambda p: int(p.name.split("_")[-1]))
    return cands


def save_json(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_dataset_info(cv_root: Path) -> Dict[str, object]:
    info_path = cv_root / "dataset_info.json"
    if not info_path.exists():
        return {}
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metrics_from_df(df: pd.DataFrame, pred_col: str = "pred") -> Dict[str, float]:
    y_true = df["target"].to_numpy()
    y_pred = df[pred_col].to_numpy()
    return {
        "n": int(len(df)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fit_calibration(y_pred: np.ndarray, y_true: np.ndarray, mode: str):
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    finite = np.isfinite(y_pred) & np.isfinite(y_true)
    y_pred = y_pred[finite]
    y_true = y_true[finite]
    if len(y_pred) == 0:
        return {"mode": mode, "a": 1.0, "b": 0.0}
    if mode == "bias_only":
        b = float(np.mean(y_true - y_pred))
        return {"mode": mode, "a": 1.0, "b": b}
    if len(y_pred) < 2 or float(np.std(y_pred)) < 1e-12:
        b = float(np.mean(y_true - y_pred))
        return {"mode": "affine", "a": 1.0, "b": b}
    a, b = np.polyfit(y_pred, y_true, deg=1)
    return {"mode": "affine", "a": float(a), "b": float(b)}


def apply_calibration_array(y_pred: np.ndarray, calib: Dict[str, float]) -> np.ndarray:
    return calib["a"] * np.asarray(y_pred, dtype=float) + calib["b"]


def aggregate_preds(df: pd.DataFrame, pred_col: str = "pred", method: str = "mean", weight_col: str = "run_weight") -> pd.DataFrame:
    if method == "mean":
        agg = df.groupby("id", as_index=False)[pred_col].mean()
        return agg.rename(columns={pred_col: "pred_ensemble"})
    if method == "median":
        agg = df.groupby("id", as_index=False)[pred_col].median()
        return agg.rename(columns={pred_col: "pred_ensemble"})
    if method == "val_mae_weighted":
        if weight_col not in df.columns:
            raise ValueError(f"Missing weight column: {weight_col}")
        tmp = df[["id", pred_col, weight_col]].copy()
        num = tmp.groupby("id").apply(lambda x: float(np.sum(x[pred_col].to_numpy() * x[weight_col].to_numpy()))).reset_index(name="num")
        den = tmp.groupby("id").apply(lambda x: float(np.sum(x[weight_col].to_numpy()))).reset_index(name="den")
        agg = num.merge(den, on="id", how="inner")
        agg["pred_ensemble"] = agg["num"] / agg["den"].replace(0.0, np.nan)
        return agg[["id", "pred_ensemble"]]
    raise ValueError(f"Unsupported ensemble method: {method}")


def _stack_graph(df: pd.DataFrame) -> np.ndarray:
    arrs = [np.asarray(x, dtype=np.float32).reshape(-1) for x in df["eq_emb"].values]
    dims = sorted(set(a.shape[0] for a in arrs))
    if len(dims) != 1:
        raise ValueError(f"Inconsistent eq_emb dims: {dims}")
    return np.stack(arrs, axis=0)


def _prepare_graph_features(train_df: pd.DataFrame, other_dfs: List[pd.DataFrame], graph_pca_dim: int, use_graph_scaler: bool, save_dir: Path):
    X_train = _stack_graph(train_df)
    scaler = None
    if use_graph_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        joblib.dump(scaler, save_dir / "graph_scaler.joblib")

    n_comp = int(min(graph_pca_dim, X_train.shape[0] - 1, X_train.shape[1]))
    n_comp = max(2, n_comp)
    pca = PCA(n_components=n_comp, random_state=42)
    X_train_p = pca.fit_transform(X_train)
    joblib.dump(pca, save_dir / "graph_pca.joblib")

    transformed = [X_train_p]
    for df in other_dfs:
        X = _stack_graph(df)
        if scaler is not None:
            X = scaler.transform(X)
        X = pca.transform(X)
        transformed.append(X)

    graph_cols = [f"gpc_{i:03d}" for i in range(X_train_p.shape[1])]
    return transformed, graph_cols


def _build_catboost_frame(
    df: pd.DataFrame,
    graph_feat: np.ndarray,
    graph_cols: List[str],
    cat_cols: List[str],
    num_cols: List[str],
    text_cols: List[str],
) -> pd.DataFrame:
    out = pd.DataFrame(graph_feat, columns=graph_cols, index=df.index)

    for c in num_cols:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            out[c] = np.nan

    for c in cat_cols:
        if c in df.columns:
            out[c] = df[c].fillna("unknown").astype(str)
        else:
            out[c] = "unknown"

    for c in text_cols:
        if c in df.columns:
            out[c] = df[c].fillna("").astype(str)
        else:
            out[c] = ""

    if "id" in df.columns:
        out["id"] = df["id"].astype(str)
    if "target" in df.columns:
        out["target"] = pd.to_numeric(df["target"], errors="coerce")
    return out


def _catboost_params(args, seed: int) -> Dict[str, object]:
    params = {
        "loss_function": args.loss_function,
        "eval_metric": args.eval_metric,
        "depth": int(args.depth),
        "learning_rate": float(args.learning_rate),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "iterations": int(args.iterations),
        "random_seed": int(seed),
        "random_strength": float(args.random_strength),
        "bagging_temperature": float(args.bagging_temperature),
        "subsample": float(args.subsample),
        "early_stopping_rounds": int(args.early_stopping_rounds),
        "verbose": 100,
        "allow_writing_files": False,
        "thread_count": int(args.thread_count),
    }
    if args.device == "GPU":
        params["task_type"] = "GPU"
        params["devices"] = args.gpu_devices
    else:
        params["task_type"] = "CPU"
    return params


def _load_weight_map(path: Optional[str], weight_col: str) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if "id" not in df.columns:
        raise ValueError(f"{p}: missing 'id' column")
    if weight_col not in df.columns:
        raise ValueError(f"{p}: missing weight column {weight_col!r}")
    out = df[["id", weight_col]].copy()
    out["id"] = out["id"].astype(str)
    out[weight_col] = pd.to_numeric(out[weight_col], errors="coerce")
    out = out.drop_duplicates(subset=["id"], keep="last")
    return out


def _attach_weights(df: pd.DataFrame, weight_col: str, weight_map: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = df.copy()
    if weight_col in out.columns:
        out[weight_col] = pd.to_numeric(out[weight_col], errors="coerce")
        return out
    if weight_map is not None:
        out["id"] = out["id"].astype(str)
        out = out.merge(weight_map, on="id", how="left")
        return out
    return out


def _clean_weights(series: pd.Series, clip_min: float, clip_max: float) -> Optional[np.ndarray]:
    if series is None:
        return None
    s = pd.to_numeric(series, errors="coerce")
    if not s.notna().any():
        return None
    s = s.fillna(1.0)
    if clip_min > 0:
        s = s.clip(lower=clip_min)
    if clip_max > 0:
        s = s.clip(upper=clip_max)
    return s.to_numpy(dtype=float)


def main():
    args = parse_args()
    cv_root = Path(args.cv_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    info = load_dataset_info(cv_root)
    cat_cols = list(info.get("cat_cols", []))
    num_cols = list(info.get("num_cols", []))
    text_cols = [c for c in ["text_raw", "text_structured"]]

    weight_map = _load_weight_map(args.weight_map_csv, args.weight_col)

    fold_dirs_all = list_fold_dirs(cv_root)
    if not fold_dirs_all:
        raise FileNotFoundError(f"No fold_* dirs found under {cv_root}")

    if args.folds == "all":
        fold_dirs = fold_dirs_all
    else:
        wanted = set(parse_list_arg(args.folds))
        fold_dirs = [p for p in fold_dirs_all if int(p.name.split("_")[-1]) in wanted]
        if not fold_dirs:
            raise ValueError(f"No requested folds found under {cv_root}: {sorted(wanted)}")

    seeds = parse_list_arg(args.seeds)
    if not seeds:
        raise ValueError("No seeds parsed from --seeds")

    all_run_metrics = []
    oof_parts = []
    out_parts = []

    for fold_dir in fold_dirs:
        fold_idx = int(fold_dir.name.split("_")[-1])
        train_df = pd.read_pickle(fold_dir / "cat_train.pkl")
        val_df = pd.read_pickle(fold_dir / "cat_val.pkl")
        test_df = pd.read_pickle(fold_dir / "cat_test.pkl")
        out_df = pd.read_pickle(fold_dir / "addH_out_cat_pred_input.pkl") if (fold_dir / "addH_out_cat_pred_input.pkl").exists() else None

        train_df = _attach_weights(train_df, args.weight_col, weight_map)
        val_df = _attach_weights(val_df, args.weight_col, weight_map)
        test_df = _attach_weights(test_df, args.weight_col, weight_map)
        if out_df is not None:
            out_df = _attach_weights(out_df, args.weight_col, weight_map)

        train_weights = _clean_weights(train_df.get(args.weight_col), args.clip_weight_min, args.clip_weight_max)
        val_weights = _clean_weights(val_df.get(args.weight_col), args.clip_weight_min, args.clip_weight_max) if args.use_val_weights else None

        for seed in seeds:
            run_tag = f"fold{fold_idx}_seed{seed}"
            run_dir = work_dir / run_tag
            run_dir.mkdir(parents=True, exist_ok=True)

            graph_list = [val_df, test_df] + ([out_df] if out_df is not None else [])
            transformed, graph_cols = _prepare_graph_features(
                train_df=train_df,
                other_dfs=graph_list,
                graph_pca_dim=args.graph_pca_dim,
                use_graph_scaler=bool(args.use_graph_scaler),
                save_dir=run_dir,
            )

            tr_g = transformed[0]
            va_g = transformed[1]
            te_g = transformed[2]
            out_g = transformed[3] if out_df is not None else None

            tr = _build_catboost_frame(train_df, tr_g, graph_cols, cat_cols, num_cols, text_cols)
            va = _build_catboost_frame(val_df, va_g, graph_cols, cat_cols, num_cols, text_cols)
            te = _build_catboost_frame(test_df, te_g, graph_cols, cat_cols, num_cols, text_cols)
            out_tab = _build_catboost_frame(out_df, out_g, graph_cols, cat_cols, num_cols, text_cols) if out_df is not None else None

            feature_cols = graph_cols + num_cols + cat_cols + text_cols
            cat_feature_names = list(cat_cols)
            text_feature_names = list(text_cols)

            train_pool = Pool(
                data=tr[feature_cols],
                label=tr["target"].to_numpy(),
                weight=train_weights,
                cat_features=cat_feature_names,
                text_features=text_feature_names,
            )
            val_pool = Pool(
                data=va[feature_cols],
                label=va["target"].to_numpy(),
                weight=val_weights,
                cat_features=cat_feature_names,
                text_features=text_feature_names,
            )

            model = CatBoostRegressor(**_catboost_params(args, seed=seed))
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            model.save_model(run_dir / "model.cbm")

            weight_info = {
                "weight_col": args.weight_col,
                "used_train_weights": bool(train_weights is not None),
                "used_val_weights": bool(val_weights is not None),
                "train_weight_mean": float(np.mean(train_weights)) if train_weights is not None else None,
                "train_weight_min": float(np.min(train_weights)) if train_weights is not None else None,
                "train_weight_max": float(np.max(train_weights)) if train_weights is not None else None,
            }
            save_json(run_dir / "weight_info.json", weight_info)

            # validation
            val_pred = model.predict(va[feature_cols])
            calib = {"mode": args.calibration_mode, "a": 1.0, "b": 0.0}
            if args.use_val_calibration:
                calib = fit_calibration(val_pred, va["target"].to_numpy(), mode=args.calibration_mode)
                val_pred = apply_calibration_array(val_pred, calib)
            save_json(run_dir / "calibration.json", calib)

            val_out = va[["id", "target"]].copy()
            val_out["pred"] = val_pred
            val_out.to_csv(run_dir / "val_pred.csv", index=False)
            val_metrics = metrics_from_df(val_out, pred_col="pred")
            save_json(run_dir / "val_metrics.json", val_metrics)

            run_weight = 1.0 / max(float(val_metrics["mae"]), float(args.min_val_weight))

            # test
            test_pred = model.predict(te[feature_cols])
            if args.use_val_calibration:
                test_pred = apply_calibration_array(test_pred, calib)

            test_out = te[["id", "target"]].copy()
            test_out["pred"] = test_pred
            test_out["fold"] = int(fold_idx)
            test_out["seed"] = int(seed)
            test_out["run_tag"] = run_tag
            test_out["run_weight"] = float(run_weight)
            test_out.to_csv(run_dir / "test_pred.csv", index=False)

            test_metrics = metrics_from_df(test_out, pred_col="pred")
            test_metrics["fold"] = int(fold_idx)
            test_metrics["seed"] = int(seed)
            test_metrics["val_mae"] = float(val_metrics["mae"])
            test_metrics["run_weight"] = float(run_weight)
            save_json(run_dir / "test_metrics.json", test_metrics)
            all_run_metrics.append(test_metrics)
            oof_parts.append(test_out)

            # addH-out
            if out_tab is not None and len(out_tab) > 0:
                pred = model.predict(out_tab[feature_cols])
                if args.use_val_calibration:
                    pred = apply_calibration_array(pred, calib)
                out_pred = out_tab[["id"]].copy()
                out_pred["pred"] = pred
                out_pred["fold"] = int(fold_idx)
                out_pred["seed"] = int(seed)
                out_pred["run_tag"] = run_tag
                out_pred["run_weight"] = float(run_weight)
                out_pred.to_csv(run_dir / "addH_out_pred.csv", index=False)
                out_parts.append(out_pred)

    if all_run_metrics:
        metrics_df = pd.DataFrame(all_run_metrics)
        metrics_df.to_csv(work_dir / "metrics_all_runs.csv", index=False)

        overall = {
            "mae_mean": float(metrics_df["mae"].mean()),
            "mae_std": float(metrics_df["mae"].std(ddof=1)) if len(metrics_df) > 1 else 0.0,
            "rmse_mean": float(metrics_df["rmse"].mean()),
            "rmse_std": float(metrics_df["rmse"].std(ddof=1)) if len(metrics_df) > 1 else 0.0,
            "r2_mean": float(metrics_df["r2"].mean()),
            "r2_std": float(metrics_df["r2"].std(ddof=1)) if len(metrics_df) > 1 else 0.0,
            "n_runs": int(len(metrics_df)),
            "ensemble_method": args.ensemble_method,
            "weight_col": args.weight_col,
            "weight_map_csv": args.weight_map_csv,
        }
        save_json(work_dir / "metrics_summary_overall.json", overall)

    if oof_parts:
        oof_all = pd.concat(oof_parts, axis=0, ignore_index=True)
        oof_all.to_csv(work_dir / "test_pred_all_runs.csv", index=False)

        oof_ens_parts = []
        for fold_idx, sub in oof_all.groupby("fold"):
            fold_ens = aggregate_preds(
                sub[["id", "pred", "run_weight"]].copy(),
                pred_col="pred",
                method=args.ensemble_method,
                weight_col="run_weight",
            )
            fold_truth = sub[["id", "target"]].drop_duplicates("id")
            fold_join = fold_truth.merge(fold_ens, on="id", how="inner").rename(columns={"pred_ensemble": "pred"})
            fold_join["fold"] = int(fold_idx)
            oof_ens_parts.append(fold_join)

        oof_ens = pd.concat(oof_ens_parts, axis=0, ignore_index=True)
        oof_ens.to_csv(work_dir / "test_pred_oof_ensemble.csv", index=False)
        save_json(work_dir / "test_pred_oof_ensemble_metrics.json", metrics_from_df(oof_ens, pred_col="pred"))

    if out_parts:
        out_all = pd.concat(out_parts, axis=0, ignore_index=True)
        out_all.to_csv(work_dir / "addH_out_pred_all_runs.csv", index=False)

        out_ens = aggregate_preds(
            out_all[["id", "pred", "run_weight"]].copy(),
            pred_col="pred",
            method=args.ensemble_method,
            weight_col="run_weight",
        )
        final = out_ens.rename(columns={"pred_ensemble": "pred"}).copy()
        final.to_csv(work_dir / "addH_out_pred_ensemble.csv", index=False)
        final.sort_values("pred").head(args.topk).to_csv(work_dir / "addH_out_top20_low.csv", index=False)
        final.sort_values("pred", ascending=False).head(args.topk).to_csv(work_dir / "addH_out_top20_high.csv", index=False)

    print("[DONE] Weighted CatBoost singleview training finished ->", work_dir)


if __name__ == "__main__":
    main()
