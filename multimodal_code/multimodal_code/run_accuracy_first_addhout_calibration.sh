#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/home/terminator/RL/multi-view}"
cd "$ROOT"

PY_MM="${PY_MM:-/data/home/terminator/anaconda3/envs/multiview/bin/python}"
OUT_DIR="${OUT_DIR:-outputs_addh_accuracy_first_calibration}"
TARGET_COL="${TARGET_COL:-h_ads_excel}"
TOP_MODELS="${TOP_MODELS:-12}"
MIN_COVERAGE="${MIN_COVERAGE:-0.90}"
INCLUDE_RANK_FEATURES="${INCLUDE_RANK_FEATURES:-1}"

mkdir -p "$OUT_DIR" logs

EXTRA_ARGS=()
if [[ "$INCLUDE_RANK_FEATURES" == "1" ]]; then
  EXTRA_ARGS+=(--include-rank-features)
fi

echo "[INFO] ROOT       = $ROOT"
echo "[INFO] PY_MM      = $PY_MM"
echo "[INFO] OUT_DIR    = $OUT_DIR"
echo "[INFO] TARGET_COL = $TARGET_COL"
echo "[INFO] TOP_MODELS = $TOP_MODELS"
echo "[INFO] Start accuracy-first calibration: $(date)"

"$PY_MM" 15_accuracy_first_addhout_calibration.py \
  --roots outputs_addh_graph_ensemble_refine_v2 outputs_addh_graph_ensemble \
  --master-csv outputs_addh_full_mm_envsplit/addH_out_master_normalized.csv \
  --out-dir "$OUT_DIR" \
  --target-col "$TARGET_COL" \
  --min-coverage "$MIN_COVERAGE" \
  --top-models "$TOP_MODELS" \
  "${EXTRA_ARGS[@]}"

echo "[DONE] Accuracy-first calibration finished: $(date)"
echo "[RESULT] $OUT_DIR/base_model_accuracy_metrics.csv"
echo "[RESULT] $OUT_DIR/accuracy_first_loocv_model_summary.csv"
echo "[RESULT] $OUT_DIR/accuracy_first_loocv_predictions.csv"
echo "[RESULT] $OUT_DIR/accuracy_first_allfit_calibrated_predictions.csv"
