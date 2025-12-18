#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch predictor for Structure→Structure GNN (vacancy-aware).

It loads the trained checkpoint ONCE, then iterates over all *.cif files in --input-dir
and writes predicted vacancy-removed structures into --output-dir.

Usage example:
  python s2s_batch_predict.py \
    --input-dir input \
    --output-dir output_pred \
    --ckpt runs/s2s_vac/best.pt \
    --cutoff 6.0 \
    --remove-element O \
    --remove-mode topk --remove-topk 1
"""
import os, sys, argparse, glob
from typing import Optional

import torch

# Import the original single-file predictor pieces
import s2s_gnn_dirpair as s2s


def main():
    p = argparse.ArgumentParser(description="Batch predict vacancy-removed structures for all CIFs in a directory.")
    p.add_argument("--input-dir", type=str, required=True, help="Directory containing input CIF files.")
    p.add_argument("--output-dir", type=str, required=True, help="Directory to save predicted CIF files.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint (e.g., runs/s2s_vac/best.pt).")
    p.add_argument("--cutoff", type=float, default=6.0)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--rbf-dim", type=int, default=32)
    p.add_argument("--max-z", type=int, default=100)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--remove-mode", type=str, choices=["threshold", "topk"], default="threshold")
    p.add_argument("--remove-threshold", type=float, default=0.5, help="Used when --remove-mode threshold.")
    p.add_argument("--remove-topk", type=int, default=1, help="Used when --remove-mode topk.")
    p.add_argument("--remove-element", type=str, default="O")
    p.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    p.add_argument("--pattern", type=str, default="*.cif", help="Glob pattern for inputs, default: *.cif")
    p.add_argument("--suffix", type=str, default="-pred-out.cif", help="Output filename suffix.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cpu" if (args.cpu or (not torch.cuda.is_available())) else "cuda"

    # Build dummy model and load checkpoint
    model = s2s.S2SModel(emb_dim=args.emb_dim, hidden=args.hidden, layers=args.layers,
                         rbf_dim=args.rbf_dim, cutoff=args.cutoff, max_z=args.max_z,
                         dropout=args.dropout).to(device)
    ckpt = s2s.load_checkpoint(args.ckpt, model, map_location=device)
    cfg = ckpt.get("config", None)
    dr_scale = ckpt.get("dr_scale", None)

    # If checkpoint contains architecture config, rebuild to match it exactly
    if cfg:
        model = s2s.S2SModel(emb_dim=cfg.get("emb_dim", args.emb_dim),
                             hidden=cfg.get("hidden", args.hidden),
                             layers=cfg.get("layers", args.layers),
                             rbf_dim=cfg.get("rbf_dim", args.rbf_dim),
                             cutoff=cfg.get("cutoff", args.cutoff),
                             max_z=cfg.get("max_z", args.max_z),
                             dropout=cfg.get("dropout", args.dropout)).to(device)
        s2s.load_checkpoint(args.ckpt, model, map_location=device)

    # Collect CIFs
    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        print(f"[ERROR] No files matched: {os.path.join(args.input_dir, args.pattern)}")
        sys.exit(2)

    print(f"[INFO] Found {len(files)} CIF(s) in {args.input_dir}. Writing to {args.output_dir}")
    n_ok, n_fail = 0, 0

    for f in files:
        try:
            pred_struct = s2s.predict_structure(
                model, f, cutoff=args.cutoff, device=device,
                remove_mode=args.remove_mode,
                remove_threshold=args.remove_threshold,
                remove_topk=args.remove_topk,
                remove_element=args.remove_element,
                dr_scale=dr_scale,
            )
            base = os.path.splitext(os.path.basename(f))[0]
            out_path = os.path.join(args.output_dir, base + args.suffix)
            pred_struct.to(fmt="cif", filename=out_path)
            n_ok += 1
            print(f"[OK] {base} -> {out_path}")
        except Exception as e:
            n_fail += 1
            print(f"[WARN] failed on {f}: {e}")

    print(f"[DONE] success={n_ok} failed={n_fail} (out: {args.output_dir})")


if __name__ == "__main__":
    main()
