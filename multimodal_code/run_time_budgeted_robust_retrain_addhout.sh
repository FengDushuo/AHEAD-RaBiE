#!/usr/bin/env bash
set -Eeuo pipefail

# Time-budgeted robust retraining route.
# No FAIR-Chem embedding extraction is rerun unless the bundle is missing.
# No GPU is required for this script.

ROOT="${ROOT:-/data/home/terminator/RL/multi-view}"
PY_MM="${PY_MM:-/data/home/terminator/anaconda3/envs/multiview/bin/python}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034}"

RETRAIN_PROFILE="${RETRAIN_PROFILE:-fast}"      # fast, medium, thorough
SUPERBLEND_FINAL_METHOD="${SUPERBLEND_FINAL_METHOD:-balanced}"  # balanced, trend, mae_guarded

SRC_OUT="${SRC_OUT:-$RUN_ROOT/outputs_addh_full_mm_envsplit}"
LLM_FEATURE_DIR="${LLM_FEATURE_DIR:-$RUN_ROOT/outputs_addh_llm_element_priors}"
FAST_CAL_DIR="${FAST_CAL_DIR:-$RUN_ROOT/outputs_addh_target_calibrated_fast}"

DELTA_FEATURE_DIR="${DELTA_FEATURE_DIR:-$RUN_ROOT/outputs_addh_pretrained_delta_features}"
ROBUST_RETRAIN_DIR="${ROBUST_RETRAIN_DIR:-$RUN_ROOT/outputs_addh_robust_retrain_delta_${RETRAIN_PROFILE}}"
ROBUST_RANK_DIR="${ROBUST_RANK_DIR:-$RUN_ROOT/outputs_addh_robust_rank_trend_${RETRAIN_PROFILE}}"
ROBUST_SUPER_DIR="${ROBUST_SUPER_DIR:-$RUN_ROOT/outputs_addh_robust_superblend_${RETRAIN_PROFILE}_${SUPERBLEND_FINAL_METHOD}}"

TRAIN_FEATURES="${TRAIN_FEATURES:-$LLM_FEATURE_DIR/knowledge_features_train.csv}"
ADDHOUT_FEATURES="${ADDHOUT_FEATURES:-$LLM_FEATURE_DIR/knowledge_features_addhout.csv}"
TRAIN_DUAL_EMB="${TRAIN_DUAL_EMB:-$SRC_OUT/addH_dual_eq_emb.pkl}"
ADDHOUT_DUAL_EMB="${ADDHOUT_DUAL_EMB:-$SRC_OUT/addH_out_dual_eq_emb.pkl}"
EXISTING_PRED_CSV="${EXISTING_PRED_CSV:-$FAST_CAL_DIR/target_calibrated_addhout_predictions.csv}"
EXISTING_PRED_COL="${EXISTING_PRED_COL:-pred_fast_target_calibrated}"
AUDIT_LABELS_CSV="${AUDIT_LABELS_CSV:-$LLM_FEATURE_DIR/addhout_audit_labels.csv}"

if [[ ! -s "$TRAIN_FEATURES" || ! -s "$ADDHOUT_FEATURES" ]]; then
  echo "[WARN] features not found under RUN_ROOT=$RUN_ROOT"
  echo "[WARN] trying to auto-detect latest run directory under $ROOT"
  DETECTED_TRAIN_FEATURES="$(
    find "$ROOT" -path "*/outputs_addh_llm_element_priors/knowledge_features_train.csv" \
      -type f -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
  if [[ -z "$DETECTED_TRAIN_FEATURES" || ! -s "$DETECTED_TRAIN_FEATURES" ]]; then
    echo "[ERROR] could not find outputs_addh_llm_element_priors/knowledge_features_train.csv under $ROOT" >&2
    exit 2
  fi
  LLM_FEATURE_DIR="$(dirname "$DETECTED_TRAIN_FEATURES")"
  RUN_ROOT="$(dirname "$LLM_FEATURE_DIR")"
  SRC_OUT="$RUN_ROOT/outputs_addh_full_mm_envsplit"
  FAST_CAL_DIR="$RUN_ROOT/outputs_addh_target_calibrated_fast"
  DELTA_FEATURE_DIR="$RUN_ROOT/outputs_addh_pretrained_delta_features"
  ROBUST_RETRAIN_DIR="$RUN_ROOT/outputs_addh_robust_retrain_delta_${RETRAIN_PROFILE}"
  ROBUST_RANK_DIR="$RUN_ROOT/outputs_addh_robust_rank_trend_${RETRAIN_PROFILE}"
  ROBUST_SUPER_DIR="$RUN_ROOT/outputs_addh_robust_superblend_${RETRAIN_PROFILE}_${SUPERBLEND_FINAL_METHOD}"
  TRAIN_FEATURES="$DETECTED_TRAIN_FEATURES"
  ADDHOUT_FEATURES="$LLM_FEATURE_DIR/knowledge_features_addhout.csv"
  TRAIN_DUAL_EMB="$SRC_OUT/addH_dual_eq_emb.pkl"
  ADDHOUT_DUAL_EMB="$SRC_OUT/addH_out_dual_eq_emb.pkl"
  EXISTING_PRED_CSV="$FAST_CAL_DIR/target_calibrated_addhout_predictions.csv"
  if [[ "$AUDIT_LABELS_CSV" == "auto" || ! -s "$AUDIT_LABELS_CSV" ]]; then
    AUDIT_LABELS_CSV="$LLM_FEATURE_DIR/addhout_audit_labels.csv"
  fi
  echo "[INFO] auto-detected RUN_ROOT=$RUN_ROOT"
fi

cd "$ROOT"

echo "[INFO] ROOT=$ROOT"
echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] PY_MM=$PY_MM"
echo "[INFO] RETRAIN_PROFILE=$RETRAIN_PROFILE"
echo "[INFO] SUPERBLEND_FINAL_METHOD=$SUPERBLEND_FINAL_METHOD"
echo "[INFO] DELTA_FEATURE_DIR=$DELTA_FEATURE_DIR"
echo "[INFO] ROBUST_RETRAIN_DIR=$ROBUST_RETRAIN_DIR"
echo "[INFO] ROBUST_RANK_DIR=$ROBUST_RANK_DIR"
echo "[INFO] ROBUST_SUPER_DIR=$ROBUST_SUPER_DIR"

if [[ ! -f "$DELTA_FEATURE_DIR/pretrained_delta_feature_bundle.npz" ]]; then
  echo "[INFO] feature bundle missing; building it now"
  "$PY_MM" 24_build_pretrained_delta_features.py \
    --train-features "$TRAIN_FEATURES" \
    --addhout-features "$ADDHOUT_FEATURES" \
    --train-dual-emb-pkl "$TRAIN_DUAL_EMB" \
    --addhout-dual-emb-pkl "$ADDHOUT_DUAL_EMB" \
    --out-dir "$DELTA_FEATURE_DIR" \
    --audit-labels-csv "$AUDIT_LABELS_CSV"
else
  echo "[INFO] reusing feature bundle: $DELTA_FEATURE_DIR/pretrained_delta_feature_bundle.npz"
fi

"$PY_MM" 28_train_time_budgeted_robust_delta_addhout.py \
  --bundle-dir "$DELTA_FEATURE_DIR" \
  --out-dir "$ROBUST_RETRAIN_DIR" \
  --existing-pred-csv "$EXISTING_PRED_CSV" \
  --existing-pred-col "$EXISTING_PRED_COL" \
  --audit-labels-csv "$AUDIT_LABELS_CSV" \
  --profile "$RETRAIN_PROFILE"

"$PY_MM" 26_rank_trend_calibrate_addhout.py \
  --pred-csv "$ROBUST_RETRAIN_DIR/robust_retrain_addhout_predictions.csv" \
  --out-dir "$ROBUST_RANK_DIR" \
  --value-col pred_robust_retrain_final \
  --score-col pred_existing_anchor \
  --fallback-col pred_robust_retrain_final \
  --audit-labels-csv "$AUDIT_LABELS_CSV" \
  --final-method quantile

"$PY_MM" 27_superblend_precision_addhout.py \
  --rank-trend-csv "$ROBUST_RANK_DIR/rank_trend_calibrated_addhout_predictions.csv" \
  --delta-csv "$ROBUST_RETRAIN_DIR/robust_retrain_addhout_predictions.csv" \
  --target-csv "$FAST_CAL_DIR/target_calibrated_addhout_predictions.csv" \
  --knowledge-csv "$RUN_ROOT/outputs_addh_llm_element_knowledge_blend_scnet_deepseek_v4_pro/knowledge_enhanced_addhout_predictions.csv" \
  --out-dir "$ROBUST_SUPER_DIR" \
  --audit-labels-csv "$AUDIT_LABELS_CSV" \
  --final-method "$SUPERBLEND_FINAL_METHOD"

echo
echo "[RESULT] robust retrain audit:"
cat "$ROBUST_RETRAIN_DIR/robust_retrain_posthoc_audit.csv" || true

echo
echo "[RESULT] robust rank-trend audit:"
cat "$ROBUST_RANK_DIR/rank_trend_posthoc_audit.csv" || true

echo
echo "[RESULT] robust superblend audit:"
cat "$ROBUST_SUPER_DIR/superblend_precision_posthoc_audit.csv" || true

echo
echo "[FINAL] $ROBUST_SUPER_DIR/superblend_precision_addhout_predictions.csv"
echo "[FINAL COLUMN] pred_superblend_final"
