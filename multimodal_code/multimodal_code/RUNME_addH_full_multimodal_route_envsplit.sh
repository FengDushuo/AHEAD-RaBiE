#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# addH/addH-2 -> addH-out full multimodal route with split conda environments.
# Place this script in: /data/home/terminator/RL/multi-view
# Run example:
#   cd /data/home/terminator/RL/multi-view
#   bash RUNME_addH_full_multimodal_route_envsplit.sh
#
# GPU selection:
#   GPU_ID=2 bash RUNME_addH_full_multimodal_route_envsplit.sh
#   GPU_ID=3 bash RUNME_addH_full_multimodal_route_envsplit.sh
# -----------------------------------------------------------------------------

# 0) Paths and environment-specific Python interpreters.
REPO_ROOT="${REPO_ROOT:-/data/home/terminator/RL/multi-view}"
SCRIPT_DIR="${SCRIPT_DIR:-$REPO_ROOT}"
cd "$SCRIPT_DIR"

# pymatgen / pandas / sklearn / transformers / multi-view training environment.
PY_MM="${PY_MM:-/data/home/terminator/anaconda3/envs/multiview/bin/python}"

# FAIR-Chem environment for Equiformer/FAIR-Chem embedding extraction.
PY_FAIR="${PY_FAIR:-/data/home/terminator/anaconda3/envs/fairchem/bin/python}"

# Use physical GPU 2 by default for multi-view training. Change to GPU_ID=3 if GPU 2 is busy.
GPU_ID="${GPU_ID:-2}"
DEVICE="${DEVICE:-cuda}"

# FAIR-Chem embedding extraction device.
# Your current fairchem env has torch_scatter installed without CUDA support, so the safe default is CPU.
# If you reinstall CUDA-enabled torch_scatter/torch_sparse/PyG in fairchem, run with FAIR_DEVICE=cuda.
FAIR_DEVICE="${FAIR_DEVICE:-cpu}"

# Data roots. If your data are elsewhere, export these before running.
ADDH_ROOT="${ADDH_ROOT:-$REPO_ROOT/addH}"
ADDH2_ROOT="${ADDH2_ROOT:-$REPO_ROOT/addH-2}"
ADDHOUT_ROOT="${ADDHOUT_ROOT:-$REPO_ROOT/addH-out}"

# Equiformer/FAIR-Chem checkpoint directory.
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/equiformer_v2_31m_allmd}"

# H reference energy. Keep consistent with your previous DFT convention.
EH_REF="${EH_REF:--0.0565}"

OUT="${OUT:-$REPO_ROOT/outputs_addh_full_mm_envsplit}"
mkdir -p "$OUT"

# Optional: avoid tokenizer parallel warnings and WandB popups.
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export WANDB_SILENT=true

# Basic checks.
if [[ ! -x "$PY_MM" ]]; then
  echo "[ERROR] PY_MM not found or not executable: $PY_MM" >&2
  exit 1
fi
if [[ ! -x "$PY_FAIR" ]]; then
  echo "[ERROR] PY_FAIR not found or not executable: $PY_FAIR" >&2
  exit 1
fi
if [[ ! -d "$ADDH_ROOT" ]]; then
  echo "[ERROR] ADDH_ROOT not found: $ADDH_ROOT" >&2
  exit 1
fi
if [[ ! -d "$ADDH2_ROOT" ]]; then
  echo "[ERROR] ADDH2_ROOT not found: $ADDH2_ROOT" >&2
  exit 1
fi
if [[ ! -d "$ADDHOUT_ROOT" ]]; then
  echo "[ERROR] ADDHOUT_ROOT not found: $ADDHOUT_ROOT" >&2
  exit 1
fi

echo "[INFO] SCRIPT_DIR = $SCRIPT_DIR"
echo "[INFO] REPO_ROOT  = $REPO_ROOT"
echo "[INFO] PY_MM      = $PY_MM"
echo "[INFO] PY_FAIR    = $PY_FAIR"
echo "[INFO] GPU_ID     = $GPU_ID"
echo "[INFO] DEVICE     = $DEVICE"
echo "[INFO] FAIR_DEVICE = $FAIR_DEVICE"
echo "[INFO] OUT        = $OUT"

# 1) Build normalized training masters. Uses multiview env because it needs pymatgen.
echo "[STEP 1] Building addH/addH-2 master tables with normalized schema..."
"$PY_MM" 01_build_addh_master_with_outlier_drop.py \
  --input-dir "$ADDH_ROOT" \
  --output-csv "$OUT/addH_master_flagged.csv" \
  --outlier-method iqr --outlier-action flag --iqr-multiplier 3.0 \
  --outlier-report-csv "$OUT/addH_outlier_report.csv" \
  --outlier-summary-json "$OUT/addH_outlier_summary.json"

"$PY_MM" build_addh_master_from_ml2_layout.py \
  --input-root "$ADDH2_ROOT" \
  --output-csv "$OUT/addH2_master_flagged.csv" \
  --base-miller-map "2542=100,2858=111,643=111" \
  --outlier-method iqr --outlier-action flag --iqr-multiplier 3.0 \
  --outlier-report-csv "$OUT/addH2_outlier_report.csv" \
  --outlier-summary-json "$OUT/addH2_outlier_summary.json"

"$PY_MM" merge_addh_master_tables_robust.py \
  --old-csv "$OUT/addH_master_flagged.csv" \
  --new-csv "$OUT/addH2_master_flagged.csv" \
  --output-csv "$OUT/addH_master_merged_robust.csv" \
  --merged-raw-csv "$OUT/addH_master_merged_raw.csv" \
  --dedup-policy quality_then_new \
  --outlier-method none --outlier-action flag

# 2) Build addH-out master. Uses multiview env because it may need pymatgen to generate bare_from_addH.
echo "[STEP 2] Building addH-out master and generated bare_from_addH structures..."
"$PY_MM" 02_build_addhout_master_normalized.py \
  --input-dir "$ADDHOUT_ROOT" \
  --output-csv "$OUT/addH_out_master_normalized.csv" \
  --miller-map "CeO2=111,ZnO=100" \
  --eh-ref "$EH_REF" \
  --target-source excel_preferred \
  --write-bare-from-addh \
  --bare-output-dir "$OUT/addH_out_bare_from_addH"

# 3) Extract addH and bare Equiformer embeddings. Uses fairchem env and binds physical GPU_ID.
echo "[STEP 3] Extracting FAIR-Chem/Equiformer embeddings on physical GPU $GPU_ID..."
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY_FAIR" 03_extract_eq_emb_fairchem.py \
  --master-csv "$OUT/addH_master_merged_robust.csv" \
  --structure-col contcar_path --id-col id --text-col text --target-col target \
  --model-dir "$MODEL_DIR" --save-pkl "$OUT/addH_addh_eq_emb.pkl" \
  --meta-csv "$OUT/addH_addh_eq_emb.meta.csv" --device "$FAIR_DEVICE" --hook-module auto --require-success-min-frac 0.80

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY_FAIR" 03_extract_eq_emb_fairchem.py \
  --master-csv "$OUT/addH_master_merged_robust.csv" \
  --structure-col bare_contcar_path --id-col id --text-col text --target-col target \
  --model-dir "$MODEL_DIR" --save-pkl "$OUT/addH_bare_eq_emb.pkl" \
  --meta-csv "$OUT/addH_bare_eq_emb.meta.csv" --device "$FAIR_DEVICE" --hook-module auto --require-success-min-frac 0.80

"$PY_MM" 06_build_dual_graph_embeddings.py \
  --master-csv "$OUT/addH_master_merged_robust.csv" \
  --addh-emb-pkl "$OUT/addH_addh_eq_emb.pkl" \
  --bare-emb-pkl "$OUT/addH_bare_eq_emb.pkl" \
  --save-pkl "$OUT/addH_dual_eq_emb.pkl" \
  --meta-csv "$OUT/addH_dual_eq_emb.meta.csv" \
  --missing-bare skip \
  --require-success-min-frac 0.80

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY_FAIR" 03_extract_eq_emb_fairchem.py \
  --master-csv "$OUT/addH_out_master_normalized.csv" \
  --structure-col contcar_path --id-col id --text-col text --target-col target \
  --model-dir "$MODEL_DIR" --save-pkl "$OUT/addH_out_addh_eq_emb.pkl" \
  --meta-csv "$OUT/addH_out_addh_eq_emb.meta.csv" --device "$FAIR_DEVICE" --hook-module auto --require-success-min-frac 0.80

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY_FAIR" 03_extract_eq_emb_fairchem.py \
  --master-csv "$OUT/addH_out_master_normalized.csv" \
  --structure-col bare_contcar_path --id-col id --text-col text --target-col target \
  --model-dir "$MODEL_DIR" --save-pkl "$OUT/addH_out_bare_eq_emb.pkl" \
  --meta-csv "$OUT/addH_out_bare_eq_emb.meta.csv" --device "$FAIR_DEVICE" --hook-module auto --require-success-min-frac 0.80

"$PY_MM" 06_build_dual_graph_embeddings.py \
  --master-csv "$OUT/addH_out_master_normalized.csv" \
  --addh-emb-pkl "$OUT/addH_out_addh_eq_emb.pkl" \
  --bare-emb-pkl "$OUT/addH_out_bare_eq_emb.pkl" \
  --save-pkl "$OUT/addH_out_dual_eq_emb.pkl" \
  --meta-csv "$OUT/addH_out_dual_eq_emb.meta.csv" \
  --missing-bare skip \
  --require-success-min-frac 0.80

# 4) Mild target-domain weighting. Uses multiview env.
echo "[STEP 4] Building mild target-domain weights..."
"$PY_MM" make_target_domain_weighted_train_table_mild.py \
  --train-csv "$OUT/addH_master_merged_robust.csv" \
  --target-csv "$OUT/addH_out_master_normalized.csv" \
  --output-csv "$OUT/addH_master_target_weighted_mild.csv" \
  --debug-csv "$OUT/target_domain_weight_debug_mild.csv" \
  --profile-json "$OUT/target_domain_profile_mild.json" \
  --train-eq-emb-pkl "$OUT/addH_dual_eq_emb.pkl" \
  --target-eq-emb-pkl "$OUT/addH_out_dual_eq_emb.pkl" \
  --use-emb --emb-min 0.97 --emb-max 1.06

# 5) Build strict group OOF splits. addH-out enters CLIP only as unlabeled graph-text pairs.
echo "[STEP 5] Building strict group OOF multi-view data, with addH-out included only in CLIP SSL..."
"$PY_MM" 04_make_multiview_data_cv_multimodal.py \
  --addh-master-csv "$OUT/addH_master_target_weighted_mild.csv" \
  --eq-emb-pkl "$OUT/addH_dual_eq_emb.pkl" \
  --addhout-master-csv "$OUT/addH_out_master_normalized.csv" \
  --addhout-eq-emb-pkl "$OUT/addH_out_dual_eq_emb.pkl" \
  --out-dir "$OUT/multiview_local_data_cv_mm_full" \
  --group-col family_base_miller \
  --cv-mode stratified_gkfold --n-splits 4 \
  --val-mode stratified_group_holdout --val-frac 0.25 \
  --stratify-bins 5 --stratify-target-col target \
  --seed 42 \
  --include-regress-eq-emb \
  --include-addhout-in-clip \
  --exclude-target-outliers \
  --target-abs-max "${TARGET_ABS_MAX:-20}"

# 6) Multi-view training/prediction. Uses multiview env and binds physical GPU_ID.
# The OOF ensemble ignores models whose validation MAE is clearly poor; this prevents
# unstable folds/seeds from dominating addH-out predictions.
echo "[STEP 6] Launching multi-view staged CV training/prediction on physical GPU $GPU_ID..."
echo "[INFO] Log file: $OUT/cv_mm_full_fixclip.gpu${GPU_ID}.log"
nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" WANDB_MODE=disabled WANDB_SILENT=true TOKENIZERS_PARALLELISM=false \
  "$PY_MM" 05_run_multiview_cv_ensemble_multimodal_staged_aligned_blend3_fixclip.py \
  --repo-root "$REPO_ROOT" \
  --cv-root "$OUT/multiview_local_data_cv_mm_full" \
  --work-dir "$OUT/multiview_cv_mm_full_fixclip" \
  --python "$PY_MM" --device "$DEVICE" \
  --folds all --seeds 42,52,62 \
  --epochs-clip 4 --epochs-regress 36 --batch-size 8 \
  --lr-clip 2e-6 --lr-regress 1e-6 \
  --regress-loss-fn SmoothL1Loss \
  --ensemble-method val_mae_weighted \
  --max-val-mae-for-ensemble "${MAX_VAL_MAE_FOR_ENSEMBLE:-3.0}" \
  --standardize-target \
  --regress-script regress_run_multimodal_staged_aligned.py \
  --predict-script regress_predict_multimodal_aligned.py \
  --train-strategy two_stage \
  --stage1-epochs 10 --stage2-epochs 26 \
  --freeze-text-encoder-stage1 --no-freeze-text-projection-stage1 \
  --unfreeze-top-n-layers-stage2 2 \
  --lr-stage1-new 4e-5 --lr-stage1-text-projection 4e-6 \
  --lr-stage2-new 1.5e-5 --lr-stage2-text-projection 2e-6 --lr-stage2-text-top 7e-7 \
  --weight-decay 0.01 \
  --sample-weight-col w_domain --use-weighted-sampler --weighted-sampler-power 0.5 \
  --use-val-calibration --calibration-mode bias_only \
  --run-clip --run-regress --run-predict-val --run-predict-test --run-predict-out \
  > "$OUT/cv_mm_full_fixclip.gpu${GPU_ID}.log" 2>&1 &

echo "[DONE] Background training started. Monitor with:"
echo "  tail -f $OUT/cv_mm_full_fixclip.gpu${GPU_ID}.log"
