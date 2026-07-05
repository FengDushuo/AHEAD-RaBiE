#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run multi-view training/prediction over CV folds and aggregate ensemble outputs. Supports regress_run_rank.py."
    )
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--cv-root", required=True, help="Root produced by 04_make_multiview_data_cv.py")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--python", default="python")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folds", default="all", help="Comma-separated fold ids, or 'all'")
    ap.add_argument("--seeds", default="42", help="Comma-separated seeds, e.g. 42,52,62")
    ap.add_argument("--epochs-clip", type=int, default=6)
    ap.add_argument("--epochs-regress", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5, help="Fallback LR")
    ap.add_argument("--lr-clip", type=float, default=None)
    ap.add_argument("--lr-regress", type=float, default=None)
    ap.add_argument("--regress-loss-fn", default="L1Loss", choices=["MSELoss", "L1Loss", "SmoothL1Loss"])
    ap.add_argument("--roberta-pretrain", default="roberta-base")
    ap.add_argument("--run-clip", action="store_true")
    ap.add_argument("--run-regress", action="store_true")
    ap.add_argument("--run-predict-test", action="store_true")
    ap.add_argument("--run-predict-out", action="store_true")
    ap.add_argument("--ensemble-method", default="mean", choices=["mean", "median"])
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--clip-script", default="clip_run.py", help="Script used for clip stage, relative to repo-root or absolute path")
    ap.add_argument("--regress-script", default="regress_run_rank.py", help="Script used for regress stage, relative to repo-root or absolute path")
    ap.add_argument("--predict-script", default="regress_predict.py", help="Prediction script, relative to repo-root or absolute path")

    # ranking-loss related options for regress_run_rank.py
    ap.add_argument("--use-rank-loss", action="store_true", help="Enable ranking loss in regress stage")
    ap.add_argument("--rank-weight", type=float, default=0.2)
    ap.add_argument("--rank-margin", type=float, default=0.1)
    ap.add_argument("--rank-min-delta", type=float, default=0.2)
    ap.add_argument("--rank-max-pairs", type=int, default=4096)
    ap.add_argument("--rank-pair-mode", default="all_pairs", choices=["all_pairs", "random_subset"])
    return ap.parse_args()


def resolve_script(repo_root: Path, script_arg: str) -> str:
    p = Path(script_arg)
    if p.is_absolute():
        return str(p)
    return str((repo_root / script_arg).resolve())


def write_yaml(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def run(cmd, cwd=None, env=None):
    print("[RUN]", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


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


def detect_eq_dim(data_dir: Path) -> int:
    clip_train_df = pd.read_pickle(data_dir / "clip_train.pkl")
    if len(clip_train_df) == 0:
        raise ValueError(f"{data_dir}/clip_train.pkl is empty")
    eq_dims = sorted({int(np.asarray(x).reshape(-1).shape[0]) for x in clip_train_df["eq_emb"]})
    if len(eq_dims) != 1:
        raise ValueError(f"Inconsistent eq_emb dims: {eq_dims}")
    return eq_dims[0]


def parse_fold_ids(cv_root: Path, folds_arg: str) -> List[int]:
    fold_dirs = sorted([p for p in cv_root.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    available = [int(p.name.split("_")[1]) for p in fold_dirs]
    if folds_arg == "all":
        return available
    req = [int(x.strip()) for x in folds_arg.split(",") if x.strip()]
    missing = sorted(set(req) - set(available))
    if missing:
        raise FileNotFoundError(f"Requested folds not found: {missing}; available={available}")
    return req


def parse_seeds(seeds_arg: str) -> List[int]:
    return [int(x.strip()) for x in seeds_arg.split(",") if x.strip()]


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def aggregate_preds(rows: List[pd.DataFrame], ensemble_method: str) -> pd.DataFrame:
    merged = rows[0].copy()
    pred_cols = ["pred"]
    for i, df in enumerate(rows[1:], start=1):
        col = f"pred_{i}"
        merged = merged.merge(df.rename(columns={"pred": col}), on="id", how="inner")
        pred_cols.append(col)
    arr = merged[pred_cols].to_numpy(dtype=float)
    if ensemble_method == "mean":
        merged["pred_ensemble"] = np.mean(arr, axis=1)
    elif ensemble_method == "median":
        merged["pred_ensemble"] = np.median(arr, axis=1)
    else:
        raise ValueError(ensemble_method)
    return merged


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    cv_root = Path(args.cv_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    lr_clip = args.lr_clip if args.lr_clip is not None else args.lr
    lr_regress = args.lr_regress if args.lr_regress is not None else args.lr

    fold_ids = parse_fold_ids(cv_root, args.folds)
    seeds = parse_seeds(args.seeds)

    clip_script = resolve_script(repo_root, args.clip_script)
    regress_script = resolve_script(repo_root, args.regress_script)
    predict_script = resolve_script(repo_root, args.predict_script)

    all_test_rows = []
    all_out_rows = []

    try:
        for fold_id in fold_ids:
            fold_dir = cv_root / f"fold_{fold_id}"
            eq_dim = detect_eq_dim(fold_dir)
            print(f"[INFO] fold={fold_id} detected eq_emb dim -> {eq_dim}")
            for seed in seeds:
                run_tag = f"fold{fold_id}_seed{seed}"
                run_dir = work_dir / run_tag
                clip_ckpt_base = run_dir / "clip_ckpts"
                regress_ckpt_base = run_dir / "regress_ckpts"
                clip_ckpt_base.mkdir(parents=True, exist_ok=True)
                regress_ckpt_base.mkdir(parents=True, exist_ok=True)

                clip_cfg = {
                    "run_name": f"addH_clip_{run_tag}",
                    "train_path": str(fold_dir / "clip_train.pkl"),
                    "val_path": str(fold_dir / "clip_val.pkl"),
                    "ckpt_save_path": str(clip_ckpt_base),
                    "resume_ckpt_path": None,
                    "model_config": "model/clip.yml",
                    "device": args.device,
                    "num_epochs": args.epochs_clip,
                    "early_stop_threshold": 5,
                    "batch_size": args.batch_size,
                    "lr": lr_clip,
                    "optimizer": "AdamW",
                    "warmup_steps": 0,
                    "scheduler": "reduceLR",
                    "log_interval": 10,
                    "patience": 3,
                    "gnn_emb": "eq_emb",
                    "debug": False,
                    "seed": seed,
                }
                regress_cfg = {
                    "run_name": f"addH_regress_{run_tag}",
                    "train_path": str(fold_dir / "regress_train.pkl"),
                    "val_path": str(fold_dir / "regress_val.pkl"),
                    "ckpt_save_path": str(regress_ckpt_base),
                    "resume_path": None,
                    "resume_config": None,
                    "pt_ckpt_path": None,
                    "model_config": "model/clip.yml",
                    "head": "pooler",
                    "device": args.device,
                    "num_epochs": args.epochs_regress,
                    "early_stop_threshold": 4,
                    "batch_size": args.batch_size,
                    "lr": lr_regress,
                    "warmup_steps": 0,
                    "optimizer": "AdamW",
                    "scheduler": "reduceLR",
                    "loss_fn": args.regress_loss_fn,
                    "log_interval": 10,
                    "debug": False,
                    "seed": seed,
                    # ranking-loss config for regress_run_rank.py
                    "use_rank_loss": bool(args.use_rank_loss),
                    "rank_weight": float(args.rank_weight),
                    "rank_margin": float(args.rank_margin),
                    "rank_min_delta": float(args.rank_min_delta),
                    "rank_max_pairs": int(args.rank_max_pairs) if args.rank_max_pairs is not None else None,
                    "rank_pair_mode": str(args.rank_pair_mode),
                }
                model_cfg = {
                    "CHGConfig": {"emb_tagging": False, "num_chg_dim": 64},
                    "CLIPConfig": {"temperature": 1.0},
                    "Path": {"pretrain_ckpt": args.roberta_pretrain},
                    "ProjectionConfig": {"dropout_rate": 0.1, "projection_dim": eq_dim},
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

                patch_root_yml(repo_root, clip_cfg, regress_cfg, model_cfg)

                env = dict(os.environ)
                env["PYTHONHASHSEED"] = str(seed)
                env.setdefault("WANDB_MODE", "disabled")

                if args.run_clip:
                    run([args.python, clip_script], cwd=repo_root, env=env)
                clip_dir = latest_checkpoint_dir(clip_ckpt_base)
                clip_ckpt = clip_dir / "checkpoint.pt"
                print(f"[INFO] {run_tag} clip checkpoint -> {clip_ckpt}")

                regress_cfg["pt_ckpt_path"] = str(clip_ckpt)
                write_yaml(repo_root / "regress_train.yml", regress_cfg)

                if args.run_regress:
                    run([args.python, regress_script], cwd=repo_root, env=env)
                regress_dir = latest_checkpoint_dir(regress_ckpt_base)
                print(f"[INFO] {run_tag} regress checkpoint dir -> {regress_dir}")

                if args.run_predict_test:
                    pred_dir = run_dir / "test_pred_output"
                    run([
                        args.python, predict_script,
                        "--data_path", str(fold_dir / "regress_test.pkl"),
                        "--pt_ckpt_dir_path", str(regress_dir),
                        "--save_path", str(pred_dir),
                    ], cwd=repo_root, env=env)
                    pred_pkl = resolve_pred_pkl(pred_dir)
                    with open(pred_pkl, "rb") as f:
                        pred_obj = pickle.load(f)
                    pred_df = pd.DataFrame({"id": list(pred_obj.keys()), "pred": list(pred_obj.values())})
                    test_df = pd.read_pickle(fold_dir / "regress_test.pkl")
                    merged = test_df.merge(pred_df, on="id", how="inner")
                    metrics = evaluate_metrics(merged["target"].to_numpy(), merged["pred"].to_numpy())
                    metrics.update({"fold": fold_id, "seed": seed})
                    with (run_dir / "test_metrics.json").open("w") as f:
                        json.dump(metrics, f, indent=2)
                    merged.to_csv(run_dir / "test_metrics.csv", index=False)
                    all_test_rows.append(merged.assign(fold=fold_id, seed=seed))

                if args.run_predict_out:
                    pred_dir = run_dir / "addH_out_pred_output"
                    run([
                        args.python, predict_script,
                        "--data_path", str(fold_dir / "addH_out_pred_input.pkl"),
                        "--pt_ckpt_dir_path", str(regress_dir),
                        "--save_path", str(pred_dir),
                    ], cwd=repo_root, env=env)
                    pred_pkl = resolve_pred_pkl(pred_dir)
                    with open(pred_pkl, "rb") as f:
                        pred_obj = pickle.load(f)
                    pred_df = pd.DataFrame({"id": list(pred_obj.keys()), "pred": list(pred_obj.values())})
                    manifest = pd.read_csv(fold_dir / "addH_out_pred_manifest.csv")
                    merged = manifest.merge(pred_df, on="id", how="left")
                    merged.to_csv(run_dir / "addH_out_pred_merged.csv", index=False)
                    all_out_rows.append(merged[["id", "pred"]].assign(fold=fold_id, seed=seed))

        if all_test_rows:
            metric_rows = []
            for df in all_test_rows:
                metric_rows.append({
                    "fold": int(df["fold"].iloc[0]),
                    "seed": int(df["seed"].iloc[0]),
                    **evaluate_metrics(df["target"].to_numpy(), df["pred"].to_numpy()),
                })
            metrics_all = pd.DataFrame(metric_rows).sort_values(["fold", "seed"])
            metrics_all.to_csv(work_dir / "metrics_all_runs.csv", index=False)

            summary = {
                "mae_mean": float(metrics_all["mae"].mean()),
                "mae_std": float(metrics_all["mae"].std(ddof=0)),
                "rmse_mean": float(metrics_all["rmse"].mean()),
                "rmse_std": float(metrics_all["rmse"].std(ddof=0)),
                "r2_mean": float(metrics_all["r2"].mean()),
                "r2_std": float(metrics_all["r2"].std(ddof=0)),
                "n_runs": int(len(metrics_all)),
            }
            with (work_dir / "metrics_summary_overall.json").open("w") as f:
                json.dump(summary, f, indent=2)

            grouped = []
            for fold_id in sorted(metrics_all["fold"].unique()):
                sub = metrics_all[metrics_all["fold"] == fold_id]
                grouped.append({
                    "fold": int(fold_id),
                    "n_runs": int(len(sub)),
                    "mae_mean": float(sub["mae"].mean()),
                    "mae_std": float(sub["mae"].std(ddof=0)),
                    "rmse_mean": float(sub["rmse"].mean()),
                    "rmse_std": float(sub["rmse"].std(ddof=0)),
                    "r2_mean": float(sub["r2"].mean()),
                    "r2_std": float(sub["r2"].std(ddof=0)),
                })
            pd.DataFrame(grouped).to_csv(work_dir / "metrics_summary_by_fold.csv", index=False)

            # OOF ensemble within each test fold across seeds
            ensemble_rows = []
            for fold_id in sorted({int(df["fold"].iloc[0]) for df in all_test_rows}):
                fold_frames = []
                target_df = None
                for df in all_test_rows:
                    if int(df["fold"].iloc[0]) != fold_id:
                        continue
                    frame = df[["id", "pred"]].copy()
                    fold_frames.append(frame)
                    if target_df is None:
                        target_df = df[["id", "target"]].copy()
                agg = aggregate_preds(fold_frames, args.ensemble_method)
                agg = agg.merge(target_df, on="id", how="inner")
                agg["fold"] = fold_id
                ensemble_rows.append(agg[["id", "target", "pred_ensemble", "fold"]])
            oof = pd.concat(ensemble_rows, ignore_index=True)
            oof = oof.rename(columns={"pred_ensemble": "pred"})
            oof.to_csv(work_dir / "test_pred_oof_ensemble.csv", index=False)
            oof_metrics = evaluate_metrics(oof["target"].to_numpy(), oof["pred"].to_numpy())
            with (work_dir / "test_pred_oof_ensemble_metrics.json").open("w") as f:
                json.dump(oof_metrics, f, indent=2)

        if all_out_rows:
            out_df = pd.concat(all_out_rows, ignore_index=True)
            merged_runs = []
            for (_, _), sub in out_df.groupby(["fold", "seed"]):
                merged_runs.append(sub[["id", "pred"]].copy())
            out_ens = aggregate_preds(merged_runs, args.ensemble_method)
            # Keep only the ensemble prediction to avoid duplicate `pred` columns
            out_ens_final = out_ens[["id", "pred_ensemble"]].rename(columns={"pred_ensemble": "pred"})
            manifest = pd.read_csv(cv_root / f"fold_{fold_ids[0]}" / "addH_out_pred_manifest.csv")
            final = manifest.merge(out_ens_final, on="id", how="left")
            final.to_csv(work_dir / "addH_out_pred_ensemble.csv", index=False)
            final.sort_values("pred").head(args.topk).to_csv(work_dir / "addH_out_top20_low.csv", index=False)
            final.sort_values("pred", ascending=False).head(args.topk).to_csv(work_dir / "addH_out_top20_high.csv", index=False)
    finally:
        restore_root_yml(repo_root)


if __name__ == "__main__":
    main()
