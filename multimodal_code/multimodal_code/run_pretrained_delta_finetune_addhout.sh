#!/usr/bin/env bash
set -Eeuo pipefail

# Frozen-pretrained-embedding delta-head route.
# Reuses existing FAIR-Chem dual embeddings and trains only small sklearn heads.

ROOT="${ROOT:-$(pwd)}"
cd "$ROOT"

PY_MM="${PY_MM:-python}"
RUN_ROOT="${RUN_ROOT:-$ROOT}"

SRC_OUT="${SRC_OUT:-$RUN_ROOT/outputs_addh_full_mm_envsplit}"
LLM_FEATURE_DIR="${LLM_FEATURE_DIR:-$RUN_ROOT/outputs_addh_llm_element_priors}"
FAST_CAL_DIR="${FAST_CAL_DIR:-$RUN_ROOT/outputs_addh_target_calibrated_fast}"

DELTA_FEATURE_DIR="${DELTA_FEATURE_DIR:-$RUN_ROOT/outputs_addh_pretrained_delta_features}"
DELTA_HEAD_DIR="${DELTA_HEAD_DIR:-$RUN_ROOT/outputs_addh_pretrained_delta_head}"

TRAIN_FEATURES="${TRAIN_FEATURES:-$LLM_FEATURE_DIR/knowledge_features_train.csv}"
ADDHOUT_FEATURES="${ADDHOUT_FEATURES:-$LLM_FEATURE_DIR/knowledge_features_addhout.csv}"
TRAIN_DUAL_EMB="${TRAIN_DUAL_EMB:-$SRC_OUT/addH_dual_eq_emb.pkl}"
ADDHOUT_DUAL_EMB="${ADDHOUT_DUAL_EMB:-$SRC_OUT/addH_out_dual_eq_emb.pkl}"
EXISTING_PRED_CSV="${EXISTING_PRED_CSV:-$FAST_CAL_DIR/target_calibrated_addhout_predictions.csv}"
EXISTING_PRED_COL="${EXISTING_PRED_COL:-pred_fast_target_calibrated}"
AUDIT_LABELS_CSV="${AUDIT_LABELS_CSV:-auto}"

echo "[INFO] ROOT=$ROOT"
echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] PY_MM=$PY_MM"
echo "[INFO] TRAIN_FEATURES=$TRAIN_FEATURES"
echo "[INFO] ADDHOUT_FEATURES=$ADDHOUT_FEATURES"
echo "[INFO] TRAIN_DUAL_EMB=$TRAIN_DUAL_EMB"
echo "[INFO] ADDHOUT_DUAL_EMB=$ADDHOUT_DUAL_EMB"
echo "[INFO] EXISTING_PRED_CSV=$EXISTING_PRED_CSV"
echo "[INFO] DELTA_FEATURE_DIR=$DELTA_FEATURE_DIR"
echo "[INFO] DELTA_HEAD_DIR=$DELTA_HEAD_DIR"

"$PY_MM" 24_build_pretrained_delta_features.py \
  --train-features "$TRAIN_FEATURES" \
  --addhout-features "$ADDHOUT_FEATURES" \
  --train-dual-emb-pkl "$TRAIN_DUAL_EMB" \
  --addhout-dual-emb-pkl "$ADDHOUT_DUAL_EMB" \
  --out-dir "$DELTA_FEATURE_DIR" \
  --audit-labels-csv "$AUDIT_LABELS_CSV"

"$PY_MM" 25_train_pretrained_delta_head_addhout.py \
  --bundle-dir "$DELTA_FEATURE_DIR" \
  --out-dir "$DELTA_HEAD_DIR" \
  --existing-pred-csv "$EXISTING_PRED_CSV" \
  --existing-pred-col "$EXISTING_PRED_COL" \
  --audit-labels-csv "$AUDIT_LABELS_CSV" \
  --oracle-diagnostic-tune

echo "[RESULT] $DELTA_HEAD_DIR/pretrained_delta_head_addhout_predictions.csv"
echo "[AUDIT]  $DELTA_HEAD_DIR/pretrained_delta_head_posthoc_audit.csv"
