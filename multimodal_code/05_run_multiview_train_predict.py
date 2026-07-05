#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, pickle, shutil, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--data-dir", required=True, help="Output dir of 04_make_multiview_data.py")
    ap.add_argument("--work-dir", required=True, help="Where checkpoints and logs will be written")
    ap.add_argument("--python", default="python")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs-clip", type=int, default=50)
    ap.add_argument("--epochs-regress", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5, help="Fallback LR used for both clip and regress if split LRs are not provided")
    ap.add_argument("--lr-clip", type=float, default=None, help="Learning rate for clip_run.py")
    ap.add_argument("--lr-regress", type=float, default=None, help="Learning rate for regress_run.py")
    ap.add_argument("--run-clip", action="store_true")
    ap.add_argument("--run-regress", action="store_true")
    ap.add_argument("--run-predict-test", action="store_true")
    ap.add_argument("--run-predict-out", action="store_true")
    ap.add_argument("--roberta-pretrain", default="roberta-base", help="keep as roberta-base if your repo is already configured for offline/local loading")
    ap.add_argument("--regress-loss-fn", default="L1Loss", choices=["MSELoss", "L1Loss", "SmoothL1Loss"], help="Loss function passed to regress_train.yml")
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


def detect_eq_dim(data_dir: Path) -> int:
    clip_train_df = pd.read_pickle(data_dir / "clip_train.pkl")
    if len(clip_train_df) == 0:
        raise ValueError("clip_train.pkl is empty")
    eq_dims = sorted({int(np.asarray(x).reshape(-1).shape[0]) for x in clip_train_df["eq_emb"]})
    if len(eq_dims) != 1:
        raise ValueError(f"Inconsistent eq_emb dims in clip_train.pkl: {eq_dims}")
    return eq_dims[0]


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    data_dir = Path(args.data_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    clip_ckpt_base = work_dir / "clip_ckpts"
    regress_ckpt_base = work_dir / "regress_ckpts"
    clip_ckpt_base.mkdir(exist_ok=True)
    regress_ckpt_base.mkdir(exist_ok=True)

    lr_clip = args.lr_clip if args.lr_clip is not None else args.lr
    lr_regress = args.lr_regress if args.lr_regress is not None else args.lr
    eq_dim = detect_eq_dim(data_dir)
    print("[INFO] detected eq_emb dim ->", eq_dim)
    print("[INFO] lr_clip ->", lr_clip)
    print("[INFO] lr_regress ->", lr_regress)

    clip_cfg = {
        "run_name": "addH_clip",
        "train_path": str(data_dir / "clip_train.pkl"),
        "val_path": str(data_dir / "clip_val.pkl"),
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
    }
    regress_cfg = {
        "run_name": "addH_regress",
        "train_path": str(data_dir / "regress_train.pkl"),
        "val_path": str(data_dir / "regress_val.pkl"),
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
    try:
        if args.run_clip:
            run([args.python, "clip_run.py"], cwd=repo_root)
            clip_dir = latest_checkpoint_dir(clip_ckpt_base)
        else:
            clip_dir = latest_checkpoint_dir(clip_ckpt_base)
        clip_ckpt = clip_dir / "checkpoint.pt"
        print("[INFO] clip checkpoint ->", clip_ckpt)

        regress_cfg["pt_ckpt_path"] = str(clip_ckpt)
        write_yaml(repo_root / "regress_train.yml", regress_cfg)

        if args.run_regress:
            run([args.python, "regress_run.py"], cwd=repo_root)
            regress_dir = latest_checkpoint_dir(regress_ckpt_base)
        else:
            regress_dir = latest_checkpoint_dir(regress_ckpt_base)
        print("[INFO] regress checkpoint dir ->", regress_dir)

        if args.run_predict_test:
            pred_dir = work_dir / "test_pred_output"
            run([
                args.python, "regress_predict.py",
                "--data_path", str(data_dir / "regress_test.pkl"),
                "--pt_ckpt_dir_path", str(regress_dir),
                "--save_path", str(pred_dir),
            ], cwd=repo_root)
            pred_pkl = resolve_pred_pkl(pred_dir)
            with open(pred_pkl, "rb") as f:
                pred_obj = pickle.load(f)
            test_manifest = pd.read_pickle(data_dir / "regress_test.pkl")
            evaluate_test(test_manifest, pred_obj, work_dir / "test_metrics.json")

        if args.run_predict_out:
            pred_dir = work_dir / "addH_out_pred_output"
            run([
                args.python, "regress_predict.py",
                "--data_path", str(data_dir / "addH_out_pred_input.pkl"),
                "--pt_ckpt_dir_path", str(regress_dir),
                "--save_path", str(pred_dir),
            ], cwd=repo_root)
            pred_pkl = resolve_pred_pkl(pred_dir)
            with open(pred_pkl, "rb") as f:
                pred_obj = pickle.load(f)
            out_manifest = pd.read_csv(data_dir / "addH_out_pred_manifest.csv")
            pred_df = pd.DataFrame({"id": list(pred_obj.keys()), "pred": list(pred_obj.values())})
            merged = out_manifest.merge(pred_df, on="id", how="left")
            merged.to_csv(work_dir / "addH_out_pred_merged.csv", index=False)
            merged.sort_values("pred").head(20).to_csv(work_dir / "addH_out_top20_low.csv", index=False)
            merged.sort_values("pred", ascending=False).head(20).to_csv(work_dir / "addH_out_top20_high.csv", index=False)
            print("[OK] addH-out predictions ->", work_dir / "addH_out_pred_merged.csv")
    finally:
        restore_root_yml(repo_root)


if __name__ == '__main__':
    main()
