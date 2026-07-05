#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--cv-root", required=True, help="Root dir produced by 04_make_multiview_data_cv_multimodal.py")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--python", default="python")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folds", default="all", help='Comma-separated fold indices, or "all"')
    ap.add_argument("--seeds", default="42", help='Comma-separated seeds, e.g. "42,52,62"')
    ap.add_argument("--epochs-clip", type=int, default=6)
    ap.add_argument("--epochs-regress", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5, help="Fallback LR if split LRs are not provided")
    ap.add_argument("--lr-clip", type=float, default=None)
    ap.add_argument("--lr-regress", type=float, default=None)
    ap.add_argument("--run-clip", action="store_true")
    ap.add_argument("--run-regress", action="store_true")
    ap.add_argument("--run-predict-test", action="store_true")
    ap.add_argument("--run-predict-out", action="store_true")
    ap.add_argument("--roberta-pretrain", default="roberta-base")
    ap.add_argument("--regress-loss-fn", default="L1Loss", choices=["MSELoss", "L1Loss", "SmoothL1Loss"])
    ap.add_argument("--ensemble-method", default="mean", choices=["mean", "median"])
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--standardize-target", action="store_true", help="Enable fold-wise target standardization in regress_run_multimodal.py")
    ap.add_argument("--regress-script", default="regress_run_multimodal.py")
    ap.add_argument("--predict-script", default="regress_predict_multimodal.py")
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


def evaluate_test(test_manifest: pd.DataFrame, pred_obj: dict, save_json: Path):
    pred_df = pd.DataFrame({"id": list(pred_obj.keys()), "pred": list(pred_obj.values())})
    df = test_manifest.merge(pred_df, on="id", how="inner")
    y_true = df["target"].to_numpy()
    y_pred = df["pred"].to_numpy()
    metrics = {
        "n": int(len(df)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
    with save_json.open("w") as f:
        json.dump(metrics, f, indent=2)
    df.to_csv(save_json.with_suffix(".csv"), index=False)
    print("[OK] test metrics ->", save_json)
    print(metrics)
    return metrics, df


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


def aggregate_preds(df: pd.DataFrame, pred_col: str = "pred", method: str = "mean") -> pd.DataFrame:
    if method == "mean":
        agg = df.groupby("id", as_index=False)[pred_col].mean()
    elif method == "median":
        agg = df.groupby("id", as_index=False)[pred_col].median()
    else:
        raise ValueError(f"Unsupported ensemble method: {method}")
    agg = agg.rename(columns={pred_col: "pred_ensemble"})
    return agg


def save_json(path: Path, obj):
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def build_model_cfg(eq_dim: int, roberta_pretrain: str) -> dict:
    return {
        "CHGConfig": {"emb_tagging": False, "num_chg_dim": 64},
        "CLIPConfig": {"temperature": 1.0},
        "Path": {"pretrain_ckpt": roberta_pretrain},
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

    all_run_metrics = []
    oof_parts = []
    out_parts = []

    for fold_dir in fold_dirs:
        fold_idx = int(fold_dir.name.split("_")[-1])
        eq_dim = detect_eq_dim(fold_dir)
        model_cfg = build_model_cfg(eq_dim, args.roberta_pretrain)

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
                    test_manifest = pd.read_pickle(fold_dir / "regress_test.pkl")
                    metrics, test_df = evaluate_test(test_manifest, pred_obj, run_dir / "test_metrics.json")
                    metrics["fold"] = int(fold_idx)
                    metrics["seed"] = int(seed)
                    all_run_metrics.append(metrics)
                    test_df["fold"] = int(fold_idx)
                    test_df["seed"] = int(seed)
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
                    out_manifest = pd.read_csv(fold_dir / "addH_out_pred_manifest.csv")
                    pred_df = pd.DataFrame({"id": list(pred_obj.keys()), "pred": list(pred_obj.values())})
                    merged = out_manifest.merge(pred_df, on="id", how="left")
                    merged["fold"] = int(fold_idx)
                    merged["seed"] = int(seed)
                    merged.to_csv(run_dir / "addH_out_pred_merged.csv", index=False)
                    out_parts.append(merged)

            finally:
                restore_root_yml(repo_root)

    if all_run_metrics:
        metrics_df = pd.DataFrame(all_run_metrics)
        metrics_df.to_csv(work_dir / "metrics_all_runs.csv", index=False)

        summary_by_fold = metrics_df.groupby("fold", as_index=False)[["mae", "rmse", "r2"]].agg(["mean", "std"])
        summary_by_fold.columns = ["_".join([c for c in col if c]).strip("_") for col in summary_by_fold.columns.to_flat_index()]
        summary_by_fold.to_csv(work_dir / "metrics_summary_by_fold.csv", index=False)

        overall = {
            "mae_mean": float(metrics_df["mae"].mean()),
            "mae_std": float(metrics_df["mae"].std(ddof=1)) if len(metrics_df) > 1 else 0.0,
            "rmse_mean": float(metrics_df["rmse"].mean()),
            "rmse_std": float(metrics_df["rmse"].std(ddof=1)) if len(metrics_df) > 1 else 0.0,
            "r2_mean": float(metrics_df["r2"].mean()),
            "r2_std": float(metrics_df["r2"].std(ddof=1)) if len(metrics_df) > 1 else 0.0,
            "n_runs": int(len(metrics_df)),
        }
        save_json(work_dir / "metrics_summary_overall.json", overall)

    if oof_parts:
        oof_all = pd.concat(oof_parts, axis=0, ignore_index=True)
        oof_all.to_csv(work_dir / "test_pred_all_runs.csv", index=False)

        oof_ens_parts = []
        for fold_idx, sub in oof_all.groupby("fold"):
            fold_ens = aggregate_preds(sub[["id", "pred"]], pred_col="pred", method=args.ensemble_method)
            fold_truth = sub[["id", "target"]].drop_duplicates("id")
            fold_join = fold_truth.merge(fold_ens, on="id", how="inner").rename(columns={"pred_ensemble": "pred"})
            fold_join["fold"] = int(fold_idx)
            oof_ens_parts.append(fold_join)
        oof_ens = pd.concat(oof_ens_parts, axis=0, ignore_index=True)
        oof_ens.to_csv(work_dir / "test_pred_oof_ensemble.csv", index=False)

        y_true = oof_ens["target"].to_numpy()
        y_pred = oof_ens["pred"].to_numpy()
        oof_metrics = {
            "n": int(len(oof_ens)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
        }
        save_json(work_dir / "test_pred_oof_ensemble_metrics.json", oof_metrics)

    if out_parts:
        out_all = pd.concat(out_parts, axis=0, ignore_index=True)
        out_all.to_csv(work_dir / "addH_out_pred_all_runs.csv", index=False)

        out_ens = aggregate_preds(out_all[["id", "pred"]], pred_col="pred", method=args.ensemble_method)
        manifest = out_all.drop(columns=["pred"], errors="ignore").drop_duplicates("id")
        out_ens_final = out_ens[["id", "pred_ensemble"]].rename(columns={"pred_ensemble": "pred"})
        final = manifest.merge(out_ens_final, on="id", how="left")
        final.to_csv(work_dir / "addH_out_pred_ensemble.csv", index=False)
        final.sort_values("pred").head(args.topk).to_csv(work_dir / "addH_out_top20_low.csv", index=False)
        final.sort_values("pred", ascending=False).head(args.topk).to_csv(work_dir / "addH_out_top20_high.csv", index=False)

    print("[DONE] multimodal CV ensemble finished ->", work_dir)


if __name__ == "__main__":
    main()
