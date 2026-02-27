export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python s2p_gnn_props_deltaE.py train \
  --full-dir ML-vac-full-cif \
  --out-dir  ML-vac-out-cif \
  --excel RaBiE-ML-with-CeO.xlsx \
  --sheet Sheet1 \
  --id-col name \
  --target-cols "ΔE" \
  --match-mode prefix \
  --test-id-contains "CeO" \
  --cv-folds 5 \
  --use-gpu \
  --workers 8 \
  --also-train-on-test \
  --save-dir runs/1-deltaE_ceo_max 

