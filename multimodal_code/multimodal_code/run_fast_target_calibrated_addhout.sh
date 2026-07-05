#!/usr/bin/env bash
set -Eeuo pipefail

# Fast CPU route for AddH/AddH-2 -> AddH-out target-domain calibration.
# It reuses stage-3 knowledge feature tables and optional strict-blind outputs.

ROOT="${ROOT:-$(pwd)}"
cd "$ROOT"

PY_MM="${PY_MM:-python}"
RUN_ROOT="${RUN_ROOT:-$ROOT}"

LLM_FEATURE_DIR="${LLM_FEATURE_DIR:-$RUN_ROOT/outputs_addh_llm_element_priors}"
LLM_PRED_DIR="${LLM_PRED_DIR:-$RUN_ROOT/outputs_addh_llm_element_knowledge_blend_scnet_deepseek_v4_pro}"
STRICT_FINAL_DIR="${STRICT_FINAL_DIR:-$RUN_ROOT/outputs_addh_strict_blind_final}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/outputs_addh_target_calibrated_fast}"

TRAIN_FEATURES="${TRAIN_FEATURES:-$LLM_FEATURE_DIR/knowledge_features_train.csv}"
ADDHOUT_FEATURES="${ADDHOUT_FEATURES:-$LLM_FEATURE_DIR/knowledge_features_addhout.csv}"
LLM_PRED_CSV="${LLM_PRED_CSV:-$LLM_PRED_DIR/knowledge_enhanced_addhout_predictions.csv}"
STRICT_PRED_CSV="${STRICT_PRED_CSV:-$STRICT_FINAL_DIR/strict_blind_strategy_ensemble_predictions.csv}"
AUDIT_LABELS_CSV="${AUDIT_LABELS_CSV:-auto}"

MATERIAL_STRICT_WEIGHTS="${MATERIAL_STRICT_WEIGHTS:-CeO2=0.20,ZnO=0.90}"
FINAL_MODE="${FINAL_MODE:-auto}"
SEED="${SEED:-42}"
N_SPLITS="${N_SPLITS:-4}"

mkdir -p "$OUT_DIR"

echo "[INFO] ROOT=$ROOT"
echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] PY_MM=$PY_MM"
echo "[INFO] TRAIN_FEATURES=$TRAIN_FEATURES"
echo "[INFO] ADDHOUT_FEATURES=$ADDHOUT_FEATURES"
echo "[INFO] LLM_PRED_CSV=$LLM_PRED_CSV"
echo "[INFO] STRICT_PRED_CSV=$STRICT_PRED_CSV"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] FINAL_MODE=$FINAL_MODE"
echo "[INFO] MATERIAL_STRICT_WEIGHTS=$MATERIAL_STRICT_WEIGHTS"

"$PY_MM" 23_train_target_domain_calibrated_addhout.py \
  --train-features "$TRAIN_FEATURES" \
  --addhout-features "$ADDHOUT_FEATURES" \
  --feature-dir "$LLM_FEATURE_DIR" \
  --llm-pred-csv "$LLM_PRED_CSV" \
  --strict-pred-csv "$STRICT_PRED_CSV" \
  --out-dir "$OUT_DIR" \
  --n-splits "$N_SPLITS" \
  --seed "$SEED" \
  --material-strict-weights "$MATERIAL_STRICT_WEIGHTS" \
  --final-mode "$FINAL_MODE" \
  --audit-labels-csv "$AUDIT_LABELS_CSV"

echo "[RESULT] $OUT_DIR/target_calibrated_addhout_predictions.csv"
echo "[AUDIT]  $OUT_DIR/target_calibrated_posthoc_audit.csv"
