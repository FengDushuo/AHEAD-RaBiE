export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python s2p_gnn_props_max.py train \
  --in-dir ML-vac-full-cif \
  --excel RaBiE-ML-with-CeO.xlsx --sheet Sheet1 \
  --id-col name --target-cols "Band Gap(eV)" \
  --save-dir runs/1-band_ceo_max \
  --cutoff 8.0 --max-neighbors 32 \
  --epochs 800 --patience 160 \
  --batch-size 6 \
  --lr 7e-4 --max-lr 2e-3 --wd 8e-6 \
  --layers 8 --hidden 384 --dropout 0.1 \
  --rbf-dim 96 \
  --skip-na-targets \
  --test-id-contains CeO \
  --loss huber --huber-beta 0.4 \
  --calibrate \
  --also-train-on-test \
  --ensemble 3 \
  --plot-test \
  --export-test-csv runs/1-band_ceo_max/test_preds_vs_true_ensemble.csv
