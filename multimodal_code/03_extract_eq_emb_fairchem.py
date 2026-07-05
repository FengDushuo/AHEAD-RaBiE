#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract EquiformerV2 embeddings (eq_emb) from structures listed in a CSV
(e.g. addH_master.csv) using an OCP / FAIR-Chem v1 EquiformerV2 checkpoint + config.

Outputs
-------
1) <save-pkl>              : dict[id] -> np.ndarray (eq_emb)
2) <save-pkl>.meta.csv     : extraction metadata
3) Optional --save-dataset-pkl:
   DataFrame with [id, text, target, eq_emb] for downstream multi-view training.

Example
-------
python 03_extract_eq_emb_fairchem_modified.py \
  --master-csv addH_master.csv \
  --structure-col contcar_path \
  --id-col id \
  --text-col text \
  --target-col target \
  --model-dir /data/home/terminator/RL/multi-view/equiformer_v2_31m_allmd \
  --save-pkl addH_eq_emb.pkl \
  --save-dataset-pkl addH_eq_emb_dataset.pkl \
  --device cpu \
  --hook-module energy_block
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

try:
    from ase import Atoms
    from ase.io import read as ase_read
except Exception as e:
    raise SystemExit(f"ASE is required. Install it first.\nOriginal error: {e}")

try:
    from fairchem.core.common.relaxation.ase_utils import OCPCalculator
except Exception as e:
    raise SystemExit(
        "Could not import OCPCalculator from FAIR-Chem v1. "
        "Please ensure you are in the fairchem_v1 environment.\n"
        f"Original error: {e}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract EquiformerV2 embeddings from CONTCAR/POSCAR structures.")
    p.add_argument("--master-csv", required=True, help="CSV containing at least id/text/target and structure path columns.")
    p.add_argument("--structure-col", default="contcar_path", help="Column in master CSV pointing to structure file.")
    p.add_argument("--id-col", default="id", help="ID column.")
    p.add_argument("--text-col", default="text", help="Text column.")
    p.add_argument("--target-col", default="target", help="Target column (optional but recommended).")
    p.add_argument("--model-dir", required=True, help="Directory containing checkpoint.pt and config.yml")
    p.add_argument("--checkpoint", default=None, help="Path to checkpoint.pt (defaults to <model-dir>/checkpoint.pt)")
    p.add_argument("--config", default=None, help="Path to config.yml (defaults to <model-dir>/config.yml)")
    p.add_argument("--trainer", default="equiformerv2_forces", help="Trainer name for OCPCalculator.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Compute device.")
    p.add_argument("--cpu", action="store_true", help="Force CPU mode regardless of --device.")
    p.add_argument(
        "--cuda-fallback-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If CUDA inference fails because a FAIR-Chem/PyG extension such as "
            "torch_scatter was installed without CUDA support, rebuild the calculator "
            "on CPU and continue. Use --no-cuda-fallback-cpu to disable."
        ),
    )
    p.add_argument(
        "--hook-module",
        default="auto",
        help="Dotted module name to hook for extracting latent representation. Use 'auto' to probe called modules and select a valid atom/graph embedding automatically.",
    )
    p.add_argument(
        "--auto-hook-limit",
        type=int,
        default=0,
        help="Maximum number of modules to probe in auto mode. 0 means probe all named modules except the root.",
    )
    p.add_argument(
        "--require-success-min-frac",
        type=float,
        default=0.50,
        help="Fail with non-zero exit if the successful embedding fraction is below this threshold.",
    )
    p.add_argument("--adsorbate-symbol", default="H", help="Adsorbate symbol used in simple tag inference.")
    p.add_argument("--surface-z-tol", type=float, default=1.2, help="Top slab layer z tolerance (Å) for tag=1 assignment.")
    p.add_argument("--save-pkl", required=True, help="Output pickle file: dict[id] -> eq_emb")
    p.add_argument("--save-dataset-pkl", default=None, help="Optional output dataset pickle with id/text/target/eq_emb")
    p.add_argument("--meta-csv", default=None, help="Optional metadata CSV path; defaults to <save-pkl>.meta.csv")
    p.add_argument("--dump-modules-json", default=None, help="Optional JSON path to dump model module names.")
    p.add_argument("--limit", type=int, default=None, help="Optional sample limit for debugging.")
    p.add_argument("--strict", action="store_true", help="Stop on first failure.")
    p.add_argument("--debug-hook", action="store_true", help="Print the hook output type for each processed sample.")
    return p.parse_args()


def get_submodule(root: torch.nn.Module, dotted: str) -> torch.nn.Module:
    cur = root
    for token in dotted.split("."):
        if not token:
            continue
        if not hasattr(cur, token):
            raise AttributeError(f"Module has no submodule '{dotted}' (failed at token '{token}').")
        cur = getattr(cur, token)
    return cur


def collect_module_names(model: torch.nn.Module) -> List[str]:
    return [name for name, _ in model.named_modules()]


class HookCapture:
    def __init__(self) -> None:
        self.output: Any = None

    def __call__(self, module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        self.output = output


def _pool_atom_tensor(x: torch.Tensor, n_atoms: int) -> Optional[np.ndarray]:
    x = x.detach().float().cpu()

    # [1, N, D] -> [N, D]
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]

    # [N, D] : per-atom features
    if x.ndim == 2 and x.shape[0] == n_atoms:
        return x.mean(dim=0).numpy()

    # [1, D] : already graph-level
    if x.ndim == 2 and x.shape[0] == 1:
        return x[0].numpy()

    # [N, A, B, ...] : per-atom high-order features -> flatten then mean-pool
    if x.ndim >= 3 and x.shape[0] == n_atoms:
        x = x.reshape(x.shape[0], -1)
        return x.mean(dim=0).numpy()

    # [D]
    if x.ndim == 1:
        return x.numpy()

    return None


def choose_embedding_from_hook_output(obj: Any, n_atoms: int) -> np.ndarray:
    """
    More permissive extractor than the original version.

    Many FAIR-Chem / Equiformer modules return nested custom objects instead of
    a raw torch.Tensor. This function recursively searches tensors inside:
      - tensors
      - list / tuple
      - dict
      - custom objects exposing common attributes
      - custom objects' __dict__

    It then pools atom-wise tensors into one fixed-size graph embedding.
    """

    def _try(obj_: Any) -> Optional[np.ndarray]:
        # 1) raw tensor
        if isinstance(obj_, torch.Tensor):
            return _pool_atom_tensor(obj_, n_atoms=n_atoms)

        # 2) tuple / list
        if isinstance(obj_, (list, tuple)):
            for v in obj_:
                out = _try(v)
                if out is not None:
                    return out
            return None

        # 3) dict
        if isinstance(obj_, dict):
            preferred = ["embedding", "emb", "tensor", "x", "node_features", "features", "energy", "output"]
            for k in preferred:
                if k in obj_:
                    out = _try(obj_[k])
                    if out is not None:
                        return out
            for _, v in obj_.items():
                out = _try(v)
                if out is not None:
                    return out
            return None

        # 4) custom object with common attributes
        for attr in ["embedding", "emb", "tensor", "x", "node_features", "features", "energy", "output"]:
            if hasattr(obj_, attr):
                try:
                    out = _try(getattr(obj_, attr))
                    if out is not None:
                        return out
                except Exception:
                    pass

        # 5) search all fields in __dict__
        if hasattr(obj_, "__dict__"):
            for _, v in obj_.__dict__.items():
                out = _try(v)
                if out is not None:
                    return out

        return None

    emb = _try(obj)
    if emb is None:
        raise RuntimeError("Hook captured no tensor output.")
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    return emb



class AutoHookCollector:
    """Collect lightweight embedding candidates from many forward hooks for one probe structure."""

    def __init__(self, n_atoms: int, max_dim: int = 200000) -> None:
        self.n_atoms = int(n_atoms)
        self.max_dim = int(max_dim)
        self.candidates: Dict[str, Dict[str, Any]] = {}

    def make_hook(self, name: str):
        def _hook(module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            if name in self.candidates:
                return
            try:
                emb = choose_embedding_from_hook_output(output, n_atoms=self.n_atoms)
                if emb is None:
                    return
                emb = np.asarray(emb, dtype=np.float32).reshape(-1)
                dim = int(emb.shape[0])
                if dim < 8 or dim > self.max_dim:
                    return
                if not np.all(np.isfinite(emb)):
                    return
                self.candidates[name] = {"dim": dim, "mean_abs": float(np.mean(np.abs(emb)))}
            except Exception:
                return
        return _hook


def _module_priority(name: str, dim: int) -> Tuple[int, int, str]:
    """Lower tuple is better. Prefer late/backbone representations over scalar heads."""
    lname = name.lower()
    score = 100
    # These are common useful representation locations for Equiformer/FAIR-Chem wrappers.
    preferred_tokens = [
        "energy_block", "backbone", "blocks", "block", "norm", "ffn", "layer_norm", "sphere", "embedding",
    ]
    for i, tok in enumerate(preferred_tokens):
        if tok in lname:
            score = min(score, i)
    # Penalize obvious scalar/regression heads.
    bad_tokens = ["energy_head", "force", "output", "final", "scalar"]
    if any(tok in lname for tok in bad_tokens) and dim <= 32:
        score += 50
    # Prefer compact-but-informative graph features.
    dim_penalty = 0
    if dim < 32:
        dim_penalty += 50
    elif dim > 20000:
        dim_penalty += 20
    elif dim > 5000:
        dim_penalty += 10
    return (score + dim_penalty, abs(dim - 1024), name)


def auto_select_hook_module(model: torch.nn.Module, calc: Any, df: pd.DataFrame, args: argparse.Namespace) -> str:
    """Run one probe forward pass and choose a module whose output can be pooled as an embedding."""
    module_items = [(n, m) for n, m in model.named_modules() if n]
    if args.auto_hook_limit and args.auto_hook_limit > 0:
        module_items = module_items[: args.auto_hook_limit]

    # Find the first valid structure path.
    probe_atoms = None
    probe_id = None
    for _, row in df.iterrows():
        path = row.get(args.structure_col, None)
        if pd.isna(path):
            continue
        path = str(path)
        if not os.path.exists(path):
            continue
        try:
            probe_atoms = read_structure(path)
            probe_id = str(row[args.id_col])
            break
        except Exception:
            continue
    if probe_atoms is None:
        raise RuntimeError("Auto hook selection failed: no readable probe structure was found.")

    tags = infer_ocp_tags(probe_atoms, adsorbate_symbol=args.adsorbate_symbol, surface_z_tol=args.surface_z_tol)
    probe_atoms.set_tags(tags.tolist())
    collector = AutoHookCollector(n_atoms=len(probe_atoms))
    handles = []
    for name, module in module_items:
        try:
            handles.append(module.register_forward_hook(collector.make_hook(name)))
        except Exception:
            pass
    try:
        probe_atoms.calc = calc
        _ = float(probe_atoms.get_potential_energy())
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    if not collector.candidates:
        raise RuntimeError(
            "Auto hook selection failed: no module produced a poolable tensor. "
            "Run once with --debug-hook and inspect the dumped module list."
        )
    ranked = sorted(
        [(name, info["dim"], info.get("mean_abs", 0.0)) for name, info in collector.candidates.items()],
        key=lambda x: _module_priority(x[0], x[1]),
    )
    print(f"[INFO] auto hook probe id={probe_id} n_atoms={len(probe_atoms)} candidates={len(ranked)}")
    for name, dim, mean_abs in ranked[:10]:
        print(f"[INFO] auto candidate: {name} dim={dim} mean_abs={mean_abs:.4g}")
    chosen = ranked[0][0]
    print(f"[INFO] auto selected hook_module = {chosen}")
    return chosen

def read_structure(path: str) -> Atoms:
    return ase_read(path)


def infer_ocp_tags(atoms: Atoms, adsorbate_symbol: str = "H", surface_z_tol: float = 1.2) -> np.ndarray:
    if hasattr(atoms, "get_tags"):
        tags = np.array(atoms.get_tags(), dtype=int)
        if len(tags) == len(atoms) and np.any(tags != 0):
            return tags
    symbols = np.array(atoms.get_chemical_symbols())
    z = atoms.positions[:, 2].astype(float)
    tags = np.zeros(len(atoms), dtype=int)
    ads_idx = np.where(symbols == adsorbate_symbol)[0]
    ads_set = set()
    if len(ads_idx) > 0:
        top_ads = ads_idx[np.argmax(z[ads_idx])]
        ads_set.add(int(top_ads))
        tags[top_ads] = 2
    slab_idx = np.array([i for i in range(len(atoms)) if i not in ads_set], dtype=int)
    if len(slab_idx) > 0:
        zmax = float(np.max(z[slab_idx]))
        surf_idx = slab_idx[z[slab_idx] >= zmax - surface_z_tol]
        tags[surf_idx] = 1
    return tags


def build_calculator(checkpoint_path: Path, config_path: Path, trainer: str, use_cpu: bool):
    return OCPCalculator(
        checkpoint_path=str(checkpoint_path),
        config_yml=str(config_path),
        trainer=trainer,
        cpu=use_cpu,
    )


def _is_cuda_extension_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    patterns = [
        "not compiled with cuda support",
        "torch_scatter",
        "segment_sum_coo",
        "cuda error",
        "no kernel image is available",
        "invalid device function",
        "undefined symbol",
    ]
    return any(pat in msg for pat in patterns)


def _rebuild_calc_cpu(
    checkpoint_path: Path,
    config_path: Path,
    trainer: str,
    reason: BaseException,
) -> tuple[Any, torch.nn.Module]:
    print("[WARN] CUDA FAIR-Chem inference failed; falling back to CPU for embedding extraction.")
    print(f"[WARN] Fallback reason: {type(reason).__name__}: {reason}")
    print("[WARN] This usually means torch_scatter/torch_sparse/PyG in the fairchem env was installed as CPU-only.")
    print("[WARN] Multi-view training can still use CUDA later; only FAIR-Chem embedding extraction is moved to CPU.")
    calc_cpu = build_calculator(checkpoint_path, config_path, trainer, use_cpu=True)
    return calc_cpu, calc_cpu.trainer.model


def main() -> None:
    args = parse_args()
    master_csv = Path(args.master_csv).resolve()
    model_dir = Path(args.model_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else model_dir / "checkpoint.pt"
    config_path = Path(args.config).resolve() if args.config else model_dir / "config.yml"
    save_pkl = Path(args.save_pkl).resolve()
    meta_csv = Path(args.meta_csv).resolve() if args.meta_csv else Path(str(save_pkl) + ".meta.csv")
    save_dataset_pkl = Path(args.save_dataset_pkl).resolve() if args.save_dataset_pkl else None

    if not master_csv.exists():
        raise FileNotFoundError(master_csv)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    use_cpu = args.cpu or (args.device == "cpu") or (not torch.cuda.is_available())
    print(f"[INFO] master_csv  = {master_csv}")
    print(f"[INFO] checkpoint  = {checkpoint_path}")
    print(f"[INFO] config      = {config_path}")
    print(f"[INFO] trainer     = {args.trainer}")
    print(f"[INFO] device      = {'cpu' if use_cpu else 'cuda'}")
    print(f"[INFO] hook_module = {args.hook_module}")

    df = pd.read_csv(master_csv)
    for c in [args.id_col, args.structure_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {master_csv}")
    if args.limit:
        df = df.head(args.limit).copy()

    calc = build_calculator(checkpoint_path, config_path, args.trainer, use_cpu)
    model = calc.trainer.model
    module_names = collect_module_names(model)
    if args.dump_modules_json:
        with open(args.dump_modules_json, "w") as f:
            json.dump(module_names, f, indent=2)
        print(f"[OK] dumped module names -> {args.dump_modules_json}")

    try:
        if str(args.hook_module).lower() == "auto":
            selected_hook_module = auto_select_hook_module(model, calc, df, args)
        else:
            selected_hook_module = args.hook_module
    except Exception as e:
        if (not use_cpu) and bool(args.cuda_fallback_cpu) and _is_cuda_extension_error(e):
            use_cpu = True
            calc, model = _rebuild_calc_cpu(checkpoint_path, config_path, args.trainer, e)
            module_names = collect_module_names(model)
            if args.dump_modules_json:
                with open(args.dump_modules_json, "w") as f:
                    json.dump(module_names, f, indent=2)
                print(f"[OK] dumped CPU module names -> {args.dump_modules_json}")
            if str(args.hook_module).lower() == "auto":
                selected_hook_module = auto_select_hook_module(model, calc, df, args)
            else:
                selected_hook_module = args.hook_module
        else:
            raise

    hook_target = get_submodule(model, selected_hook_module)
    capture = HookCapture()
    handle = hook_target.register_forward_hook(capture)

    emb_dict: Dict[str, np.ndarray] = {}
    meta_rows: List[Dict[str, Any]] = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = str(row[args.id_col])
        structure_path = row[args.structure_col]
        info: Dict[str, Any] = {
            "id": sample_id,
            "structure_path": structure_path,
            "ok": False,
            "n_atoms": None,
            "pred_energy": None,
            "emb_dim": None,
            "notes": "",
            "hook_module": selected_hook_module,
            "device_used": "cpu" if use_cpu else "cuda",
        }
        try:
            if pd.isna(structure_path):
                raise ValueError("structure path is NaN")
            structure_path = str(structure_path)
            if not os.path.exists(structure_path):
                raise FileNotFoundError(structure_path)
            atoms = read_structure(structure_path)
            tags = infer_ocp_tags(atoms, adsorbate_symbol=args.adsorbate_symbol, surface_z_tol=args.surface_z_tol)
            atoms.set_tags(tags.tolist())

            capture.output = None
            atoms.calc = calc
            pred_energy = float(atoms.get_potential_energy())
            info["pred_energy"] = pred_energy
            info["n_atoms"] = len(atoms)

            if capture.output is None:
                raise RuntimeError("Forward hook captured nothing. Try a different --hook-module.")

            if args.debug_hook:
                print(f"[DEBUG] {sample_id}: hook output type = {type(capture.output)}")
                if isinstance(capture.output, dict):
                    print(f"[DEBUG] {sample_id}: hook keys = {list(capture.output.keys())[:20]}")
                elif isinstance(capture.output, (list, tuple)):
                    print(f"[DEBUG] {sample_id}: hook len  = {len(capture.output)}")

            emb = choose_embedding_from_hook_output(capture.output, n_atoms=len(atoms))
            emb_dict[sample_id] = emb
            info["emb_dim"] = int(emb.shape[0])
            info["ok"] = True
        except Exception as e:
            info["notes"] = f"{type(e).__name__}: {e}"
            if args.strict:
                handle.remove()
                raise
        meta_rows.append(info)

    handle.remove()

    with open(save_pkl, "wb") as f:
        pickle.dump(emb_dict, f)
    pd.DataFrame(meta_rows).to_csv(meta_csv, index=False)
    print(f"[OK] embeddings saved -> {save_pkl}")
    print(f"[OK] metadata saved   -> {meta_csv}")
    ok_count = int(sum(r["ok"] for r in meta_rows))
    total_count = int(len(meta_rows))
    ok_frac = (ok_count / total_count) if total_count else 0.0
    print(f"[INFO] success count  = {ok_count} / {total_count} ({ok_frac:.3f})")
    if ok_frac < float(args.require_success_min_frac):
        # Print the most frequent failure reasons before failing, so nohup logs are informative.
        try:
            notes = pd.Series([r.get("notes", "") for r in meta_rows if not r.get("ok", False)])
            print("[ERROR] embedding success fraction below threshold")
            print("[ERROR] top failure notes:")
            print(notes.value_counts().head(10).to_string())
        except Exception:
            pass
        raise SystemExit(2)

    if save_dataset_pkl is not None:
        keep_cols = [c for c in [args.id_col, args.text_col, args.target_col] if c in df.columns]
        ds = df[keep_cols].copy()
        ds = ds.rename(columns={args.id_col: "id"})
        if args.text_col in keep_cols and args.text_col != "text":
            ds = ds.rename(columns={args.text_col: "text"})
        if args.target_col in keep_cols and args.target_col != "target":
            ds = ds.rename(columns={args.target_col: "target"})
        ds["eq_emb"] = ds["id"].map(emb_dict)
        ds = ds[ds["eq_emb"].notna()].copy()
        ds.to_pickle(save_dataset_pkl)
        print(f"[OK] dataset pkl saved -> {save_dataset_pkl} (n={len(ds)})")


if __name__ == "__main__":
    main()
