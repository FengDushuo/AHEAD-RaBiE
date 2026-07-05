#!/usr/bin/env bash
set -Eeuo pipefail

# Label-free rank/trend post-processing after pretrained delta-head.

ROOT="${ROOT:-$(pwd)}"
cd "$ROOT"

PY_MM="${PY_MM:-python}"
RUN_ROOT="${RUN_ROOT:-$ROOT}"

DELTA_HEAD_DIR="${DELTA_HEAD_DIR:-$RUN_ROOT/outputs_addh_pretrained_delta_head}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/outputs_addh_rank_trend_calibrated}"
AUDIT_LABELS_CSV="${AUDIT_LABELS_CSV:-$RUN_ROOT/outputs_addh_llm_element_priors/addhout_audit_labels.csv}"

PRED_CSV="${PRED_CSV:-$DELTA_HEAD_DIR/pretrained_delta_head_addhout_predictions.csv}"
VALUE_COL="${VALUE_COL:-pred_pretrained_delta_final}"
SCORE_COL="${SCORE_COL:-pred_existing_anchor}"
FALLBACK_COL="${FALLBACK_COL:-pred_pretrained_delta_final}"
FINAL_METHOD="${FINAL_METHOD:-quantile}"

echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] PRED_CSV=$PRED_CSV"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] VALUE_COL=$VALUE_COL SCORE_COL=$SCORE_COL FINAL_METHOD=$FINAL_METHOD"

"$PY_MM" 26_rank_trend_calibrate_addhout.py \
  --pred-csv "$PRED_CSV" \
  --out-dir "$OUT_DIR" \
  --value-col "$VALUE_COL" \
  --score-col "$SCORE_COL" \
  --fallback-col "$FALLBACK_COL" \
  --final-method "$FINAL_METHOD" \
  --audit-labels-csv "$AUDIT_LABELS_CSV"

echo "[RESULT] $OUT_DIR/rank_trend_calibrated_addhout_predictions.csv"
echo "[AUDIT]  $OUT_DIR/rank_trend_posthoc_audit.csv"
