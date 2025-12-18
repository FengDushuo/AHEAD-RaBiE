python s2s_gnn_dirpair.py predict \
  --in-cif data/ML-vac-full-cif/353-011-Au-out.cif \
  --ckpt runs/s2s_vac/best.pt \
  --out-cif pred-353-011-Au-out.cif \
  --cutoff 6.0 \
  --remove-element O \
  --remove-mode topk --remove-topk 1

