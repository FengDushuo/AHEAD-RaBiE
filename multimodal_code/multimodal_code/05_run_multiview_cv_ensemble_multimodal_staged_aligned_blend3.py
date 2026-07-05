#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--cv-root", required=True, help="Root dir produced by 04_make_multiview_data_cv_multimodal.py")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--python", default="python")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folds", default="all", help='Comma-separated fold indices, or "all"')
    ap.add_argument("--seeds", default="42", help='Comma-separated seeds, e.g. "42,52"')

    ap.add_argument("--epochs-clip", type=int, default=6)
    ap.add_argument("--epochs-regress", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=8)

    ap.add_argument("--lr", type=float, default=1e-5, help="Fallback LR if split LRs are not provided")
    ap.add_argument("--lr-clip", type=float, default=None)
    ap.add_argument("--lr-regress", type=float, default=None)

    ap.add_argument("--run-clip", action="store_true")
    ap.add_argument("--run-regress", action="store_true")
    ap.add_argument("--run-predict-val", action="store_true")
    ap.add_argument("--run-predict-test", action="store_true")
    ap.add_argument("--run-predict-out", action="store_true")

    ap.add_argument("--roberta-pretrain", default="roberta-base")
    ap.add_argument("--projection-dim", type=int, default=256)
    ap.add_argument("--dropout-rate", type=float, default=0.10)

    ap.add_argument("--text-col", default="text", help="Preferred primary text column")
    ap.add_argument("--concat-text-cols", default="text_structured,text_raw,text", help="Comma-separated text columns to concatenate if available")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=2)

    ap.add_argument("--sample-weight-col", default="", help="Optional sample-weight column already present in regress_train/val/test.pkl")
    ap.add_argument("--use-weighted-sampler", action="store_true")
    ap.add_argument("--weighted-sampler-power", type=float, default=1.0)

    ap.add_argument("--regress-loss-fn", default="SmoothL1Loss", choices=["MSELoss", "L1Loss", "SmoothL1Loss"])
    ap.add_argument("--ensemble-method", default="mean", choices=["mean", "median", "val_mae_weighted"])
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--standardize-target", action="store_true")
    ap.add_argument("--regress-script", default="regress_run_multimodal_staged_aligned.py")
    ap.add_argument("--predict-script", default="regress_predict_multimodal_aligned.py")
    ap.add_argument("--use-val-calibration", action="store_true")
    ap.add_argument("--calibration-mode", default="bias_only", choices=["affine", "bias_only"])
    ap.add_argument("--min-val-weight", type=float, default=1e-6)

    ap.add_argument("--train-strategy", default="two_stage", choices=["two_stage", "single_stage"])
    ap.add_argument("--stage1-epochs", type=int, default=8)
    ap.add_argument("--stage2-epochs", type=int, default=16)

    ap.add_argument("--freeze-text-encoder-stage1", dest="freeze_text_encoder_stage1", action="store_true")
    ap.add_argument("--no-freeze-text-encoder-stage1", dest="freeze_text_encoder_stage1", action="store_false")
    ap.set_defaults(freeze_text_encoder_stage1=True)

    ap.add_argument("--freeze-text-projection-stage1", dest="freeze_text_projection_stage1", action="store_true")
    ap.add_argument("--no-freeze-text-projection-stage1", dest="freeze_text_projection_stage1", action="store_false")
    ap.set_defaults(freeze_text_projection_stage1=False)

    ap.add_argument("--unfreeze-top-n-layers-stage2", type=int, default=2)

    ap.add_argument("--lr-stage1-new", type=float, default=2e-5)
    ap.add_argument("--lr-stage1-text-projection", type=float, default=3e-6)
    ap.add_argument("--lr-stage2-new", type=float, default=1e-5)
    ap.add_argument("--lr-stage2-text-projection", type=float, default=2e-6)
    ap.add_argument("--lr-stage2-text-top", type=float, default=7e-7)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--wandb-mode", default="disabled")

    ap.add_argument("--init-regress-ckpt-path", default=None, help="Optional full multimodal checkpoint to warm-start regression head/fusion/projections")

    # 3-way blend stage
    ap.add_argument("--run-threeway-blend", action="store_true", help="After multimodal stage, blend CatBoost + singleview NN + multiview")
    ap.add_argument("--cat-root", default=None, help="Root containing CatBoost predictions")
    ap.add_argument("--nn-root", default=None, help="Root containing singleview NN predictions")
    ap.add_argument("--mv-root-for-blend", default=None, help="Optional override. Defaults to --work-dir")
    ap.add_argument("--blend-method", default="ridge", choices=["ridge", "mean", "weighted"], help="Three-way final blend method")
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--cat-weight", type=float, default=0.45)
    ap.add_argument("--nn-weight", type=float, default=0.25)
    ap.add_argument("--mv-weight", type=float, default=0.30)
    ap.add_argument("--blend-prefix", default="blend3")
    ap.add_argument("--use-blend-calibration", action="store_true", help="Fit final blend calibration on OOF and apply to addH-out")
    ap.add_argument("--blend-calibration-mode", default="bias_only", choices=["affine", "bias_only"])
    return ap.parse_args()


def write_yaml(path: Path, obj: dict):
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def run(cmd, cwd=None):
    print("[RUN]", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def patch_root_yml(repo_root: Path, clip_cfg: dict, regress_cfg: dict, model_cfg: dict):
    backup_dir = repo_root / ".multiview_yaml_backup"
    backup_dir.mkdir(exist_ok=True)
    for name in ["clip_train.yml", "regress_train.yml", "model/clip.yml"]:
        src = repo_root / name
        dst = backup_dir / name.replace("/", "__")
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    write_yaml(repo_root / "clip_train.yml", clip_cfg)
    write_yaml(repo_root / "regress_train.yml", regress_cfg)
    write_yaml(repo_root / "model" / "clip.yml", model_cfg)


def restore_root_yml(repo_root: Path):
    backup_dir = repo_root / ".multiview_yaml_backup"
    mapping = {
        "clip_train.yml": repo_root / "clip_train.yml",
        "regress_train.yml": repo_root / "regress_train.yml",
        "model__clip.yml": repo_root / "model" / "clip.yml",
    }
    for bname, target in mapping.items():
        src = backup_dir / bname
        if src.exists():
            shutil.copyfile(src, target)


def latest_checkpoint_dir(base_dir: Path) -> Path:
    cands = [p for p in base_dir.iterdir() if p.is_dir()]
    if not cands:
        raise FileNotFoundError(f"No checkpoint subdirs found under {base_dir}")
    cands.sort(key=lambda p: p.stat().st_mtime)
    return cands[-1]


def resolve_pred_pkl(save_path: Path) -> Path:
    if save_path.is_file():
        return save_path
    if save_path.is_dir():
        cands = sorted(save_path.glob("*-strc.pkl"))
        if cands:
            return cands[-1]
        cands = sorted(save_path.glob("*.pkl"))
        if cands:
            return cands[-1]
    raise FileNotFoundError(f"Could not locate prediction pickle under {save_path}")


def save_json(path: Path, obj):
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def detect_eq_dim(data_dir: Path) -> int:
    clip_train_df = pd.read_pickle(data_dir / "clip_train.pkl")
    if len(clip_train_df) == 0:
        raise ValueError(f"{data_dir / 'clip_train.pkl'} is empty")
    eq_dims = sorted({int(np.asarray(x).reshape(-1).shape[0]) for x in clip_train_df["eq_emb"]})
    if len(eq_dims) != 1:
        raise ValueError(f"Inconsistent eq_emb dims in clip_train.pkl: {eq_dims}")
    return eq_dims[0]


def parse_list_arg(raw: str) -> List[int]:
    raw = str(raw).strip()
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def list_fold_dirs(cv_root: Path) -> List[Path]:
    cands = [p for p in cv_root.iterdir() if p.is_dir() and p.name.startswith("fold_")]
    cands.sort(key=lambda p: int(p.name.split("_")[-1]))
    return cands


def build_model_cfg(eq_dim: int, roberta_pretrain: str, projection_dim: int, dropout_rate: float) -> dict:
    proj_dim = int(min(max(32, projection_dim), max(32, eq_dim)))
    return {
        "CHGConfig": {"emb_tagging": False, "num_chg_dim": 64},
        "CLIPConfig": {"temperature": 1.0},
        "Path": {"pretrain_ckpt": roberta_pretrain},
        "ProjectionConfig": {"dropout_rate": float(dropout_rate), "projection_dim": proj_dim},
        "RobertaConfig": {
            "architectures": ["RobertaForMaskedLM"],
            "attention_probs_dropout_prob": 0.1,
            "bos_token_id": 0,
            "classifier_dropout": None,
            "eos_token_id": 2,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-5,
            "max_position_embeddings": 514,
            "model_type": "roberta",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 1,
            "position_embedding_type": "absolute",
            "transformers_version": "4.29.2",
            "type_vocab_size": 1,
            "use_cache": True,
            "vocab_size": 50265,
        },
    }


def pred_obj_to_df(pred_obj: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"id": list(pred_obj.keys()), "pred": list(pred_obj.values())})


def metrics_from_df(df: pd.DataFrame, pred_col: str = "pred") -> Dict[str, float]:
    y_true = df["target"].to_numpy()
    y_pred = df[pred_col].to_numpy()
    return {
        "n": int(len(df)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_prediction_frame(manifest_df: pd.DataFrame, pred_obj: Dict[str, float], save_json_path: Path | None = None) -> Tuple[Dict[str, float], pd.DataFrame]:
    pred_df = pred_obj_to_df(pred_obj)
    df = manifest_df.merge(pred_df, on="id", how="inner")
    metrics = metrics_from_df(df, pred_col="pred")
    if save_json_path is not None:
        save_json(save_json_path, metrics)
        df.to_csv(save_json_path.with_suffix(".csv"), index=False)
        print("[OK] metrics ->", save_json_path)
        print(metrics)
    return metrics, df


def fit_calibration(y_pred: np.ndarray, y_true: np.ndarray, mode: str) -> Dict[str, float]:
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


def apply_calibration_dict(pred_obj: Dict[str, float], calib: Dict[str, float]) -> Dict[str, float]:
    return {str(k): float(calib["a"] * float(v) + calib["b"]) for k, v in pred_obj.items()}


def aggregate_preds(df: pd.DataFrame, pred_col: str = "pred", method: str = "mean", weight_col: str = "run_weight") -> pd.DataFrame:
    if method == "mean":
        agg = df.groupby("id", as_index=False)[pred_col].mean()
        return agg.rename(columns={pred_col: "pred_ensemble"})
    if method == "median":
        agg = df.groupby("id", as_index=False)[pred_col].median()
        return agg.rename(columns={pred_col: "pred_ensemble"})
    if method == "val_mae_weighted":
        if weight_col not in df.columns:
            raise ValueError(f"Weighted ensemble requested but missing weight column: {weight_col}")
        tmp = df[["id", pred_col, weight_col]].copy()
        num = tmp.groupby("id").apply(lambda x: float(np.sum(x[pred_col].to_numpy() * x[weight_col].to_numpy()))).reset_index(name="num")
        den = tmp.groupby("id").apply(lambda x: float(np.sum(x[weight_col].to_numpy()))).reset_index(name="den")
        agg = num.merge(den, on="id", how="inner")
        agg["pred_ensemble"] = agg["num"] / agg["den"].replace(0.0, np.nan)
        return agg[["id", "pred_ensemble"]]
    raise ValueError(f"Unsupported ensemble method: {method}")


def _aggregate_all_runs_oof(df: pd.DataFrame) -> pd.DataFrame:
    need = {"id", "pred"}
    if not need.issubset(df.columns):
        raise ValueError(f"All-runs OOF file missing columns: {sorted(need - set(df.columns))}")
    if "target" not in df.columns:
        raise ValueError("All-runs OOF file must contain target for blending.")
    agg = df.groupby("id", as_index=False).agg(target=("target", "mean"), pred=("pred", "mean"))
    return agg


def _aggregate_all_runs_out(df: pd.DataFrame) -> pd.DataFrame:
    need = {"id", "pred"}
    if not need.issubset(df.columns):
        raise ValueError(f"All-runs out file missing columns: {sorted(need - set(df.columns))}")
    agg = df.groupby("id", as_index=False).agg(pred=("pred", "mean"))
    return agg


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


def _merge_threeway_oof(cat_df: pd.DataFrame, nn_df: pd.DataFrame, mv_df: pd.DataFrame) -> pd.DataFrame:
    merged = cat_df.merge(nn_df, on="id", how="inner", suffixes=("_cat_target", "_nn_target"))
    merged = merged.merge(mv_df, on="id", how="inner")

    target_cols = [c for c in merged.columns if c.startswith("target")]
    # unify target
    if "target" in merged.columns:
        pass
    else:
        merged["target"] = merged[target_cols[0]]
        for c in target_cols[1:]:
            both = merged[[target_cols[0], c]].dropna()
            if len(both) > 0:
                max_abs_diff = np.max(np.abs(both[target_cols[0]].to_numpy() - both[c].to_numpy()))
                if max_abs_diff > 1e-6:
                    print(f"[WARN] target mismatch between OOF sources; max abs diff = {max_abs_diff:.6g}")
        drop_cols = [c for c in target_cols if c != "target"]
        merged = merged.drop(columns=drop_cols)
    keep = ["id", "target", "pred_cat", "pred_nn", "pred_mv"]
    # rename target columns if needed
    for c in list(merged.columns):
        if c.startswith("target") and c != "target":
            merged = merged.drop(columns=[c])
    return merged[keep].copy()


def _merge_threeway_out(cat_df: pd.DataFrame, nn_df: pd.DataFrame, mv_df: pd.DataFrame) -> pd.DataFrame:
    merged = cat_df.merge(nn_df, on="id", how="inner").merge(mv_df, on="id", how="inner")
    return merged[["id", "pred_cat", "pred_nn", "pred_mv"]].copy()


def _fit_threeway_ridge(df_wide: pd.DataFrame, alpha: float) -> Ridge:
    X = df_wide[["pred_cat", "pred_nn", "pred_mv"]].to_numpy(dtype=float)
    y = df_wide["target"].to_numpy(dtype=float)
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(X, y)
    return model


def _apply_threeway_blend(df_wide: pd.DataFrame, method: str, cat_w: float, nn_w: float, mv_w: float, ridge_model: Optional[Ridge] = None) -> np.ndarray:
    Xc = df_wide["pred_cat"].to_numpy(dtype=float)
    Xn = df_wide["pred_nn"].to_numpy(dtype=float)
    Xm = df_wide["pred_mv"].to_numpy(dtype=float)

    if method == "mean":
        return (Xc + Xn + Xm) / 3.0
    if method == "weighted":
        s = float(cat_w + nn_w + mv_w)
        if s <= 0:
            raise ValueError("cat_weight + nn_weight + mv_weight must be > 0")
        wc, wn, wm = float(cat_w) / s, float(nn_w) / s, float(mv_w) / s
        return wc * Xc + wn * Xn + wm * Xm
    if method == "ridge":
        if ridge_model is None:
            raise ValueError("ridge_model is required for blend-method=ridge")
        X = df_wide[["pred_cat", "pred_nn", "pred_mv"]].to_numpy(dtype=float)
        return ridge_model.predict(X)
    raise ValueError(f"Unsupported blend method: {method}")


def _aggregate_pred_by_id_mean(df: pd.DataFrame, pred_col: str = "pred") -> pd.DataFrame:
    return df.groupby("id", as_index=False)[pred_col].mean()


def run_threeway_blend(args):
    if args.cat_root is None or args.nn_root is None:
        raise ValueError("--run-threeway-blend requires both --cat-root and --nn-root")

    work_dir = Path(args.work_dir).resolve()
    mv_root = Path(args.mv_root_for_blend).resolve() if args.mv_root_for_blend else work_dir
    cat_root = Path(args.cat_root).resolve()
    nn_root = Path(args.nn_root).resolve()

    cat_oof, cat_out = _load_model_oof_out(cat_root, "cat")
    nn_oof, nn_out = _load_model_oof_out(nn_root, "nn")
    mv_oof, mv_out = _load_model_oof_out(mv_root, "mv")

    blend_dir = work_dir / f"{args.blend_prefix}_{args.blend_method}"
    blend_dir.mkdir(parents=True, exist_ok=True)

    # save source tables for inspection
    cat_oof.to_csv(blend_dir / "cat_oof_source.csv", index=False)
    nn_oof.to_csv(blend_dir / "nn_oof_source.csv", index=False)
    mv_oof.to_csv(blend_dir / "mv_oof_source.csv", index=False)
    cat_out.to_csv(blend_dir / "cat_addH_out_source.csv", index=False)
    nn_out.to_csv(blend_dir / "nn_addH_out_source.csv", index=False)
    mv_out.to_csv(blend_dir / "mv_addH_out_source.csv", index=False)

    oof_wide = _merge_threeway_oof(cat_oof, nn_oof, mv_oof)
    out_wide = _merge_threeway_out(cat_out, nn_out, mv_out)

    oof_wide.to_csv(blend_dir / "threeway_oof_pairs.csv", index=False)
    out_wide.to_csv(blend_dir / "threeway_addH_out_pairs.csv", index=False)

    ridge_model = None
    ridge_info = None
    if args.blend_method == "ridge":
        ridge_model = _fit_threeway_ridge(oof_wide, alpha=float(args.ridge_alpha))
        ridge_info = {
            "coef": ridge_model.coef_.tolist(),
            "intercept": float(ridge_model.intercept_),
            "alpha": float(args.ridge_alpha),
        }
        save_json(blend_dir / "threeway_ridge_info.json", ridge_info)

    oof_wide["pred_blend_raw"] = _apply_threeway_blend(
        oof_wide,
        method=args.blend_method,
        cat_w=float(args.cat_weight),
        nn_w=float(args.nn_weight),
        mv_w=float(args.mv_weight),
        ridge_model=ridge_model,
    )

    blend_calib = {"mode": args.blend_calibration_mode, "a": 1.0, "b": 0.0}
    if args.use_blend_calibration:
        blend_calib = fit_calibration(
            y_pred=oof_wide["pred_blend_raw"].to_numpy(dtype=float),
            y_true=oof_wide["target"].to_numpy(dtype=float),
            mode=args.blend_calibration_mode,
        )
    save_json(blend_dir / "threeway_blend_calibration.json", blend_calib)

    oof_wide["pred_blend"] = blend_calib["a"] * oof_wide["pred_blend_raw"].to_numpy(dtype=float) + blend_calib["b"]
    oof_wide.to_csv(blend_dir / "threeway_oof_pairs_with_pred.csv", index=False)

    oof_final = _aggregate_pred_by_id_mean(oof_wide[["id", "target", "pred_blend"]].rename(columns={"pred_blend": "pred"}), pred_col="pred")
    target_final = oof_wide.groupby("id", as_index=False)["target"].mean()
    oof_final = target_final.merge(oof_final, on="id", how="inner")
    oof_final.to_csv(blend_dir / "threeway_oof_ensemble.csv", index=False)
    oof_metrics = metrics_from_df(oof_final, pred_col="pred")
    save_json(blend_dir / "threeway_oof_metrics.json", oof_metrics)

    # individual constituent metrics on same overlap set
    constituent_metrics = {}
    for col, name in [("pred_cat", "cat"), ("pred_nn", "nn"), ("pred_mv", "mv")]:
        constituent_metrics[name] = metrics_from_df(oof_wide[["id", "target", col]].rename(columns={col: "pred"}), pred_col="pred")
    save_json(blend_dir / "threeway_constituent_metrics_on_overlap.json", constituent_metrics)

    out_wide["pred_blend_raw"] = _apply_threeway_blend(
        out_wide,
        method=args.blend_method,
        cat_w=float(args.cat_weight),
        nn_w=float(args.nn_weight),
        mv_w=float(args.mv_weight),
        ridge_model=ridge_model,
    )
    out_wide["pred_blend"] = blend_calib["a"] * out_wide["pred_blend_raw"].to_numpy(dtype=float) + blend_calib["b"]
    out_wide.to_csv(blend_dir / "threeway_addH_out_pairs_with_pred.csv", index=False)

    out_final = _aggregate_pred_by_id_mean(out_wide[["id", "pred_blend"]].rename(columns={"pred_blend": "pred"}), pred_col="pred")
    out_final.to_csv(blend_dir / "threeway_addH_out_ensemble.csv", index=False)
    out_final.sort_values("pred").head(args.topk).to_csv(blend_dir / "threeway_addH_out_top20_low.csv", index=False)
    out_final.sort_values("pred", ascending=False).head(args.topk).to_csv(blend_dir / "threeway_addH_out_top20_high.csv", index=False)

    summary = {
        "blend_method": args.blend_method,
        "cat_root": str(cat_root),
        "nn_root": str(nn_root),
        "mv_root": str(mv_root),
        "cat_weight": float(args.cat_weight),
        "nn_weight": float(args.nn_weight),
        "mv_weight": float(args.mv_weight),
        "ridge_alpha": float(args.ridge_alpha),
        "use_blend_calibration": bool(args.use_blend_calibration),
        "blend_calibration_mode": args.blend_calibration_mode,
        "oof_metrics": oof_metrics,
        "constituent_metrics_on_overlap": constituent_metrics,
        "ridge_info": ridge_info,
    }
    save_json(blend_dir / "threeway_summary.json", summary)
    print("[DONE] three-way blend finished ->", blend_dir)
    print("[INFO] three-way OOF metrics =", oof_metrics)


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    cv_root = Path(args.cv_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    regress_script = repo_root / args.regress_script
    predict_script = repo_root / args.predict_script
    if not regress_script.exists():
        raise FileNotFoundError(f"Cannot find regress script: {regress_script}")
    if not predict_script.exists():
        raise FileNotFoundError(f"Cannot find predict script: {predict_script}")

    fold_dirs_all = list_fold_dirs(cv_root)
    if not fold_dirs_all:
        raise FileNotFoundError(f"No fold_* directories found under {cv_root}")

    if args.folds == "all":
        fold_dirs = fold_dirs_all
    else:
        wanted = set(parse_list_arg(args.folds))
        fold_dirs = [p for p in fold_dirs_all if int(p.name.split("_")[-1]) in wanted]
        if not fold_dirs:
            raise ValueError(f"No requested folds found under {cv_root}: {sorted(wanted)}")

    seeds = parse_list_arg(args.seeds)
    if not seeds:
        raise ValueError("No seeds were parsed from --seeds")

    lr_clip = args.lr_clip if args.lr_clip is not None else args.lr
    lr_regress = args.lr_regress if args.lr_regress is not None else args.lr
    need_val_preds = bool(args.run_predict_val or args.use_val_calibration or args.ensemble_method == "val_mae_weighted")

    all_run_metrics = []
    oof_parts = []
    out_parts = []

    for fold_dir in fold_dirs:
        fold_idx = int(fold_dir.name.split("_")[-1])
        eq_dim = detect_eq_dim(fold_dir)
        model_cfg = build_model_cfg(eq_dim, args.roberta_pretrain, args.projection_dim, args.dropout_rate)

        for seed in seeds:
            run_tag = f"fold{fold_idx}_seed{seed}"
            run_dir = work_dir / run_tag
            run_dir.mkdir(parents=True, exist_ok=True)

            clip_ckpt_base = run_dir / "clip_ckpts"
            regress_ckpt_base = run_dir / "regress_ckpts"
            clip_ckpt_base.mkdir(exist_ok=True)
            regress_ckpt_base.mkdir(exist_ok=True)

            clip_cfg = {
                "run_name": f"addH_clip_{run_tag}",
                "train_path": str(fold_dir / "clip_train.pkl"),
                "val_path": str(fold_dir / "clip_val.pkl"),
                "ckpt_save_path": str(clip_ckpt_base),
                "resume_ckpt_path": None,
                "model_config": "model/clip.yml",
                "device": args.device,
                "num_epochs": int(args.epochs_clip),
                "early_stop_threshold": 5,
                "batch_size": int(args.batch_size),
                "lr": float(lr_clip),
                "optimizer": "AdamW",
                "warmup_steps": 0,
                "scheduler": "reduceLR",
                "log_interval": 10,
                "patience": 3,
                "gnn_emb": "eq_emb",
                "debug": False,
                "seed": int(seed),
            }

            regress_cfg = {
                "run_name": f"addH_regress_mm_{run_tag}",
                "train_path": str(fold_dir / "regress_train.pkl"),
                "val_path": str(fold_dir / "regress_val.pkl"),
                "ckpt_save_path": str(regress_ckpt_base),
                "resume_path": None,
                "resume_config": None,
                "pt_ckpt_path": None,
                "init_regress_ckpt_path": args.init_regress_ckpt_path,
                "model_config": "model/clip.yml",
                "head": "pooler",
                "device": args.device,
                "num_epochs": int(args.epochs_regress),
                "early_stop_threshold": 4,
                "batch_size": int(args.batch_size),
                "lr": float(lr_regress),
                "warmup_steps": 0,
                "optimizer": "AdamW",
                "scheduler": "reduceLR",
                "loss_fn": args.regress_loss_fn,
                "log_interval": 10,
                "debug": False,
                "seed": int(seed),
                "standardize_target": bool(args.standardize_target),
                "train_strategy": args.train_strategy,
                "stage1_epochs": int(args.stage1_epochs),
                "stage2_epochs": int(args.stage2_epochs),
                "freeze_text_encoder_stage1": bool(args.freeze_text_encoder_stage1),
                "freeze_text_projection_stage1": bool(args.freeze_text_projection_stage1),
                "unfreeze_top_n_layers_stage2": int(args.unfreeze_top_n_layers_stage2),
                "lr_stage1_new": float(args.lr_stage1_new),
                "lr_stage1_text_projection": float(args.lr_stage1_text_projection),
                "lr_stage2_new": float(args.lr_stage2_new),
                "lr_stage2_text_projection": float(args.lr_stage2_text_projection),
                "lr_stage2_text_top": float(args.lr_stage2_text_top),
                "weight_decay": float(args.weight_decay),
                "wandb_mode": args.wandb_mode,
                "text_col": args.text_col,
                "concat_text_cols": [x.strip() for x in args.concat_text_cols.split(",") if x.strip()],
                "seq_len": int(args.seq_len),
                "num_workers": int(args.num_workers),
                "sample_weight_col": args.sample_weight_col if str(args.sample_weight_col).strip() else None,
                "use_weighted_sampler": bool(args.use_weighted_sampler),
                "weighted_sampler_power": float(args.weighted_sampler_power),
            }

            patch_root_yml(repo_root, clip_cfg, regress_cfg, model_cfg)
            try:
                if args.run_clip:
                    run([args.python, "clip_run.py"], cwd=repo_root)
                    clip_dir = latest_checkpoint_dir(clip_ckpt_base)
                else:
                    clip_dir = latest_checkpoint_dir(clip_ckpt_base)
                clip_ckpt = clip_dir / "checkpoint.pt"
                print("[INFO]", run_tag, "clip checkpoint ->", clip_ckpt)

                regress_cfg["pt_ckpt_path"] = str(clip_ckpt)
                write_yaml(repo_root / "regress_train.yml", regress_cfg)

                if args.run_regress:
                    run([args.python, args.regress_script], cwd=repo_root)
                    regress_dir = latest_checkpoint_dir(regress_ckpt_base)
                else:
                    regress_dir = latest_checkpoint_dir(regress_ckpt_base)
                print("[INFO]", run_tag, "regress checkpoint dir ->", regress_dir)

                calib = {"mode": args.calibration_mode, "a": 1.0, "b": 0.0}
                val_raw_metrics = None
                val_cal_metrics = None
                run_weight = None

                if need_val_preds:
                    pred_dir = run_dir / "val_pred_output"
                    run([
                        args.python, args.predict_script,
                        "--data_path", str(fold_dir / "regress_val.pkl"),
                        "--pt_ckpt_dir_path", str(regress_dir),
                        "--save_path", str(pred_dir),
                        "--device", str(args.device),
                        "--batch_size", str(args.batch_size),
                    ], cwd=repo_root)
                    pred_pkl = resolve_pred_pkl(pred_dir)
                    with open(pred_pkl, "rb") as f:
                        pred_obj_val = pickle.load(f)

                    val_manifest = pd.read_pickle(fold_dir / "regress_val.pkl")
                    val_raw_metrics, _ = evaluate_prediction_frame(val_manifest, pred_obj_val, run_dir / "val_metrics_raw.json")

                    if args.use_val_calibration:
                        val_raw_df = val_manifest.merge(pred_obj_to_df(pred_obj_val), on="id", how="inner")
                        calib = fit_calibration(val_raw_df["pred"].to_numpy(), val_raw_df["target"].to_numpy(), mode=args.calibration_mode)
                        pred_obj_val_cal = apply_calibration_dict(pred_obj_val, calib)
                        val_cal_metrics, _ = evaluate_prediction_frame(val_manifest, pred_obj_val_cal, run_dir / "val_metrics_calibrated.json")
                        save_json(run_dir / "calibration.json", calib)
                    else:
                        save_json(run_dir / "calibration.json", calib)
                        val_cal_metrics = val_raw_metrics

                    weight_source = val_cal_metrics["mae"] if val_cal_metrics is not None else val_raw_metrics["mae"]
                    run_weight = 1.0 / max(float(weight_source), float(args.min_val_weight))

                if args.run_predict_test:
                    pred_dir = run_dir / "test_pred_output"
                    run([
                        args.python, args.predict_script,
                        "--data_path", str(fold_dir / "regress_test.pkl"),
                        "--pt_ckpt_dir_path", str(regress_dir),
                        "--save_path", str(pred_dir),
                        "--device", str(args.device),
                        "--batch_size", str(args.batch_size),
                    ], cwd=repo_root)
                    pred_pkl = resolve_pred_pkl(pred_dir)
                    with open(pred_pkl, "rb") as f:
                        pred_obj = pickle.load(f)
                    if args.use_val_calibration:
                        pred_obj = apply_calibration_dict(pred_obj, calib)
                    test_manifest = pd.read_pickle(fold_dir / "regress_test.pkl")
                    metrics, test_df = evaluate_prediction_frame(test_manifest, pred_obj, run_dir / "test_metrics.json")
                    metrics["fold"] = int(fold_idx)
                    metrics["seed"] = int(seed)
                    if val_raw_metrics is not None:
                        metrics["val_mae_raw"] = float(val_raw_metrics["mae"])
                    if val_cal_metrics is not None:
                        metrics["val_mae_cal"] = float(val_cal_metrics["mae"])
                    metrics["calib_a"] = float(calib["a"])
                    metrics["calib_b"] = float(calib["b"])
                    metrics["run_weight"] = float(run_weight) if run_weight is not None else np.nan
                    all_run_metrics.append(metrics)

                    test_df["fold"] = int(fold_idx)
                    test_df["seed"] = int(seed)
                    test_df["run_tag"] = run_tag
                    test_df["calib_a"] = float(calib["a"])
                    test_df["calib_b"] = float(calib["b"])
                    test_df["run_weight"] = float(run_weight) if run_weight is not None else np.nan
                    oof_parts.append(test_df)

                if args.run_predict_out:
                    pred_dir = run_dir / "addH_out_pred_output"
                    run([
                        args.python, args.predict_script,
                        "--data_path", str(fold_dir / "addH_out_pred_input.pkl"),
                        "--pt_ckpt_dir_path", str(regress_dir),
                        "--save_path", str(pred_dir),
                        "--device", str(args.device),
                        "--batch_size", str(args.batch_size),
                    ], cwd=repo_root)
                    pred_pkl = resolve_pred_pkl(pred_dir)
                    with open(pred_pkl, "rb") as f:
                        pred_obj = pickle.load(f)
                    if args.use_val_calibration:
                        pred_obj = apply_calibration_dict(pred_obj, calib)
                    out_manifest = pd.read_csv(fold_dir / "addH_out_pred_manifest.csv")
                    pred_df = pred_obj_to_df(pred_obj)
                    merged = out_manifest.merge(pred_df, on="id", how="left")
                    merged["fold"] = int(fold_idx)
                    merged["seed"] = int(seed)
                    merged["run_tag"] = run_tag
                    merged["calib_a"] = float(calib["a"])
                    merged["calib_b"] = float(calib["b"])
                    merged["run_weight"] = float(run_weight) if run_weight is not None else np.nan
                    merged.to_csv(run_dir / "addH_out_pred_merged.csv", index=False)
                    out_parts.append(merged)

            finally:
                restore_root_yml(repo_root)

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
            "use_val_calibration": bool(args.use_val_calibration),
            "ensemble_method": args.ensemble_method,
            "train_strategy": args.train_strategy,
            "projection_dim": int(args.projection_dim),
        }
        save_json(work_dir / "metrics_summary_overall.json", overall)

    if oof_parts:
        oof_all = pd.concat(oof_parts, axis=0, ignore_index=True)
        oof_all.to_csv(work_dir / "test_pred_all_runs.csv", index=False)
        oof_ens_parts = []
        for fold_idx, sub in oof_all.groupby("fold"):
            fold_ens = aggregate_preds(sub[["id", "pred", "run_weight"]].copy(), pred_col="pred", method=args.ensemble_method, weight_col="run_weight")
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
        out_ens = aggregate_preds(out_all[["id", "pred", "run_weight"]].copy(), pred_col="pred", method=args.ensemble_method, weight_col="run_weight")
        manifest = out_all.drop(columns=["pred"], errors="ignore").drop_duplicates("id")
        out_ens_final = out_ens[["id", "pred_ensemble"]].rename(columns={"pred_ensemble": "pred"})
        final = manifest.merge(out_ens_final, on="id", how="left")
        final.to_csv(work_dir / "addH_out_pred_ensemble.csv", index=False)
        final.sort_values("pred").head(args.topk).to_csv(work_dir / "addH_out_top20_low.csv", index=False)
        final.sort_values("pred", ascending=False).head(args.topk).to_csv(work_dir / "addH_out_top20_high.csv", index=False)

    if args.run_threeway_blend:
        run_threeway_blend(args)

    print("[DONE] multimodal CV ensemble finished ->", work_dir)


if __name__ == "__main__":
    main()
