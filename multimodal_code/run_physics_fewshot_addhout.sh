#!/usr/bin/env bash
set -Eeuo pipefail

# Physics-enhanced + few-shot AddH-out route.
#
# Expected upstream files:
#   $RUN_ROOT/outputs_addh_llm_element_priors/knowledge_features_train.csv
#   $RUN_ROOT/outputs_addh_llm_element_priors/knowledge_features_addhout.csv
#   $RUN_ROOT/outputs_addh_full_mm_envsplit/addH_dual_eq_emb.pkl
#   $RUN_ROOT/outputs_addh_full_mm_envsplit/addH_out_dual_eq_emb.pkl
#
# If they do not exist yet, set BUILD_UPSTREAM=1 to run the data-prep +
# FAIR-Chem embedding part of run_addh_server_full_pipeline.sh first.

ROOT="${ROOT:-$(pwd)}"
cd "$ROOT"

PY_MM="${PY_MM:-python}"
PY_FAIRCHEM="${PY_FAIRCHEM:-$PY_MM}"
RUN_ROOT="${RUN_ROOT:-$ROOT}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
SRC_OUT="${SRC_OUT:-$RUN_ROOT/outputs_addh_full_mm_envsplit}"
LLM_FEATURE_DIR="${LLM_FEATURE_DIR:-$RUN_ROOT/outputs_addh_llm_element_priors}"

FAIRCHEM_MODEL_DIR="${FAIRCHEM_MODEL_DIR:-/data/home/terminator/RL/multi-view/equiformer_v2_31m_allmd}"
GPU_ID="${GPU_ID:-2}"
DEVICE="${DEVICE:-cuda}"
BUILD_UPSTREAM="${BUILD_UPSTREAM:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

DELTA_FEATURE_DIR="${DELTA_FEATURE_DIR:-$RUN_ROOT/outputs_addh_pretrained_delta_features}"
DELTA_HEAD_DIR="${DELTA_HEAD_DIR:-$RUN_ROOT/outputs_addh_pretrained_delta_head}"
RANK_TREND_DIR="${RANK_TREND_DIR:-$RUN_ROOT/outputs_addh_rank_trend_calibrated}"
SUPERBLEND_DIR="${SUPERBLEND_DIR:-$RUN_ROOT/outputs_addh_superblend_precision}"
FEWSHOT_DIR="${FEWSHOT_DIR:-$RUN_ROOT/outputs_addh_fewshot_domain_calibration}"

FAST_CAL_DIR="${FAST_CAL_DIR:-$RUN_ROOT/outputs_addh_target_calibrated_fast}"
KNOWLEDGE_DIR="${KNOWLEDGE_DIR:-$RUN_ROOT/outputs_addh_llm_element_knowledge_blend_scnet_deepseek_v4_pro}"
AUDIT_LABELS_CSV="${AUDIT_LABELS_CSV:-auto}"

TRAIN_FEATURES="${TRAIN_FEATURES:-$LLM_FEATURE_DIR/knowledge_features_train.csv}"
ADDHOUT_FEATURES="${ADDHOUT_FEATURES:-$LLM_FEATURE_DIR/knowledge_features_addhout.csv}"
TRAIN_DUAL_EMB="${TRAIN_DUAL_EMB:-$SRC_OUT/addH_dual_eq_emb.pkl}"
ADDHOUT_DUAL_EMB="${ADDHOUT_DUAL_EMB:-$SRC_OUT/addH_out_dual_eq_emb.pkl}"

SHOTS_PER_MATERIAL="${SHOTS_PER_MATERIAL:-0,1,2,3,4,5,6,8,10}"
REPEATS="${REPEATS:-200}"
OPERATIONAL_SHOTS_PER_MATERIAL="${OPERATIONAL_SHOTS_PER_MATERIAL:-4}"
OPERATIONAL_BASE_COL="${OPERATIONAL_BASE_COL:-pred_superblend_mae_guarded}"
OPERATIONAL_CALIBRATOR="${OPERATIONAL_CALIBRATOR:-guarded_auto}"
CALIBRATION_IDS="${CALIBRATION_IDS:-}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
mkdir -p "$LOG_DIR"

run_logged() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  echo "[CMD] $*" > "${log_file}.cmd"
  echo "[RUN] $*"
  "$@" > "$log_file" 2>&1
}

need_file() {
  local f="$1"
  if [[ ! -s "$f" ]]; then
    echo "[ERROR] missing required file: $f" >&2
    return 1
  fi
}

echo "[INFO] ROOT=$ROOT"
echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] PY_MM=$PY_MM"
echo "[INFO] PY_FAIRCHEM=$PY_FAIRCHEM"
echo "[INFO] GPU_ID=$GPU_ID DEVICE=$DEVICE"
echo "[INFO] FAIRCHEM_MODEL_DIR=$FAIRCHEM_MODEL_DIR"
echo "[INFO] BUILD_UPSTREAM=$BUILD_UPSTREAM"
echo "[INFO] FEWSHOT_DIR=$FEWSHOT_DIR"

if [[ "$BUILD_UPSTREAM" == "1" ]]; then
  echo "[STAGE A] Build upstream masters, FAIR-Chem embeddings, and knowledge feature tables"
  RUN_ROOT="$RUN_ROOT" \
  PY_MM="$PY_MM" \
  PY_FAIRCHEM="$PY_FAIRCHEM" \
  FAIRCHEM_MODEL_DIR="$FAIRCHEM_MODEL_DIR" \
  GPU_ID="$GPU_ID" \
  DEVICE="$DEVICE" \
  RUN_DATA_PREP=auto \
  RUN_EMBEDDINGS=auto \
  RUN_LLM_ROUTE=1 \
  RUN_SCNET_LLM=0 \
  RUN_GRAPH_GRID=0 \
  RUN_MULTIVIEW_GRID=0 \
  RUN_STRICT_BLIND=0 \
  RUN_SERVER_FINAL_BLEND=0 \
  RUN_AUDIT=0 \
  ./run_addh_server_full_pipeline.sh
fi

need_file "$TRAIN_FEATURES"
need_file "$ADDHOUT_FEATURES"
need_file "$TRAIN_DUAL_EMB"
need_file "$ADDHOUT_DUAL_EMB"
need_file "$FAST_CAL_DIR/target_calibrated_addhout_predictions.csv"

echo "[STAGE B] Build pretrained physical-feature bundle"
if [[ "$SKIP_COMPLETED" != "1" || ! -s "$DELTA_FEATURE_DIR/pretrained_delta_feature_bundle.npz" ]]; then
  run_logged "$LOG_DIR/stage_physics_delta_features.log" \
    "$PY_MM" 24_build_pretrained_delta_features.py \
      --train-features "$TRAIN_FEATURES" \
      --addhout-features "$ADDHOUT_FEATURES" \
      --train-dual-emb-pkl "$TRAIN_DUAL_EMB" \
      --addhout-dual-emb-pkl "$ADDHOUT_DUAL_EMB" \
      --out-dir "$DELTA_FEATURE_DIR" \
      --audit-labels-csv "$AUDIT_LABELS_CSV"
else
  echo "[SKIP] $DELTA_FEATURE_DIR/pretrained_delta_feature_bundle.npz"
fi

echo "[STAGE C] Train small delta heads on frozen physical embeddings"
if [[ "$SKIP_COMPLETED" != "1" || ! -s "$DELTA_HEAD_DIR/pretrained_delta_head_addhout_predictions.csv" ]]; then
  run_logged "$LOG_DIR/stage_physics_delta_head.log" \
    "$PY_MM" 25_train_pretrained_delta_head_addhout.py \
      --bundle-dir "$DELTA_FEATURE_DIR" \
      --out-dir "$DELTA_HEAD_DIR" \
      --existing-pred-csv "$FAST_CAL_DIR/target_calibrated_addhout_predictions.csv" \
      --existing-pred-col pred_fast_target_calibrated \
      --audit-labels-csv "$AUDIT_LABELS_CSV"
else
  echo "[SKIP] $DELTA_HEAD_DIR/pretrained_delta_head_addhout_predictions.csv"
fi

echo "[STAGE D] Rank/trend post-processing"
if [[ "$SKIP_COMPLETED" != "1" || ! -s "$RANK_TREND_DIR/rank_trend_calibrated_addhout_predictions.csv" ]]; then
  run_logged "$LOG_DIR/stage_physics_rank_trend.log" \
    "$PY_MM" 26_rank_trend_calibrate_addhout.py \
      --pred-csv "$DELTA_HEAD_DIR/pretrained_delta_head_addhout_predictions.csv" \
      --out-dir "$RANK_TREND_DIR" \
      --value-col pred_pretrained_delta_final \
      --score-col pred_existing_anchor \
      --fallback-col pred_pretrained_delta_final \
      --final-method quantile \
      --audit-labels-csv "$AUDIT_LABELS_CSV"
else
  echo "[SKIP] $RANK_TREND_DIR/rank_trend_calibrated_addhout_predictions.csv"
fi

echo "[STAGE E] Strict label-free superblend"
if [[ "$SKIP_COMPLETED" != "1" || ! -s "$SUPERBLEND_DIR/superblend_precision_addhout_predictions.csv" ]]; then
  run_logged "$LOG_DIR/stage_physics_superblend.log" \
    "$PY_MM" 27_superblend_precision_addhout.py \
      --rank-trend-csv "$RANK_TREND_DIR/rank_trend_calibrated_addhout_predictions.csv" \
      --delta-csv "$DELTA_HEAD_DIR/pretrained_delta_head_addhout_predictions.csv" \
      --target-csv "$FAST_CAL_DIR/target_calibrated_addhout_predictions.csv" \
      --knowledge-csv "$KNOWLEDGE_DIR/knowledge_enhanced_addhout_predictions.csv" \
      --out-dir "$SUPERBLEND_DIR" \
      --audit-labels-csv "$AUDIT_LABELS_CSV" \
      --final-method mae_guarded
else
  echo "[SKIP] $SUPERBLEND_DIR/superblend_precision_addhout_predictions.csv"
fi

echo "[STAGE F] Few-shot AddH-out calibration and held-out evaluation"
run_logged "$LOG_DIR/stage_physics_fewshot.log" \
  "$PY_MM" 31_fewshot_addhout_domain_adaptation.py \
    --pred-csv "$SUPERBLEND_DIR/superblend_precision_addhout_predictions.csv" \
    --labels "$AUDIT_LABELS_CSV" \
    --out-dir "$FEWSHOT_DIR" \
    --base-cols auto \
    --shots-per-material "$SHOTS_PER_MATERIAL" \
    --repeats "$REPEATS" \
    --operational-shots-per-material "$OPERATIONAL_SHOTS_PER_MATERIAL" \
    --operational-base-col "$OPERATIONAL_BASE_COL" \
    --operational-calibrator "$OPERATIONAL_CALIBRATOR" \
    --calibration-ids "$CALIBRATION_IDS" \
    --write-xlsx

echo "[DONE]"
echo "[STRICT]  $SUPERBLEND_DIR/superblend_precision_addhout_predictions.csv"
echo "[FEWSHOT] $FEWSHOT_DIR/fewshot_operational_predictions.csv"
echo "[AUDIT]   $FEWSHOT_DIR/fewshot_operational_audit.csv"
echo "[REPORT]  $FEWSHOT_DIR/fewshot_holdout_summary.csv"
