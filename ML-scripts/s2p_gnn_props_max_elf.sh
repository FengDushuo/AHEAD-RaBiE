export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python s2p_gnn_props_max_elf.py train \
  --in-dir ML-vac-full-cif \
  --excel RaBiE-ML-with-CeO.xlsx \
  --sheet Sheet1 \
  --id-col name --target-cols "Peak of ELF" \
  --match-mode prefix \
  --test-id-contains "CeO" \
  --also-train-on-test \
  --neighbor-cutoff 3.4 \
  --cv-folds 8 \
  --ensemble 5 \
  --save-dir runs/1-elf_ceo_max 
