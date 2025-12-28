python s2s_gnn_dirpair.py train \
  --save-dir runs/s2s_vac \
  --cutoff 6.0 \
  --epochs 200 --batch-size 16 \
  --lr 1e-3 --max-lr 3e-3 \
  --layers 6 --hidden 256 --dropout 0.15 \
  --lambda-bce 0.3 --norm-dr \
  --match-max-dist 1.2 \
  --remove-element O

