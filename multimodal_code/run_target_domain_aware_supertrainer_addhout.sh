#!/usr/bin/env bash
set -Eeuo pipefail

# Target-domain-aware strict-blind training route.
# Uses addH-out covariates for domain analysis and train weighting, but does
# not use addH-out labels for training/selection. Labels are audit-only.

ROOT="${ROOT:-$(pwd)}"
cd "$ROOT"

PY_MM="${PY_MM:-python}"
RUN_ROOT="${RUN_ROOT:-$ROOT}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"

LLM_FEATURE_DIR="${LLM_FEATURE_DIR:-$RUN_ROOT/outputs_addh_llm_element_priors}"
DELTA_FEATURE_DIR="${DELTA_FEATURE_DIR:-$RUN_ROOT/outputs_addh_pretrained_delta_features}"
SUPERBLEND_DIR="${SUPERBLEND_DIR:-$RUN_ROOT/outputs_addh_superblend_precision}"
FAST_CAL_DIR="${FAST_CAL_DIR:-$RUN_ROOT/outputs_addh_target_calibrated_fast}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/outputs_addh_target_domain_aware_supertrainer}"

TRAIN_FEATURES="${TRAIN_FEATURES:-$LLM_FEATURE_DIR/knowledge_features_train.csv}"
ADDHOUT_FEATURES="${ADDHOUT_FEATURES:-$LLM_FEATURE_DIR/knowledge_features_addhout.csv}"
GRAPH_BUNDLE="${GRAPH_BUNDLE:-$DELTA_FEATURE_DIR/pretrained_delta_feature_bundle.npz}"
GRAPH_TRAIN_META="${GRAPH_TRAIN_META:-$DELTA_FEATURE_DIR/pretrained_delta_train_meta.csv}"
GRAPH_ADDHOUT_META="${GRAPH_ADDHOUT_META:-$DELTA_FEATURE_DIR/pretrained_delta_addhout_meta.csv}"
ANCHOR_CSV="${ANCHOR_CSV:-$SUPERBLEND_DIR/superblend_precision_addhout_predictions.csv}"
FALLBACK_ANCHOR_CSV="${FALLBACK_ANCHOR_CSV:-$FAST_CAL_DIR/target_calibrated_addhout_predictions.csv}"
AUDIT_LABELS_CSV="${AUDIT_LABELS_CSV:-auto}"

N_SPLITS="${N_SPLITS:-5}"
TOP_K="${TOP_K:-10}"
FINAL_MODE="${FINAL_MODE:-mae_guarded}"
FEATURE_SETS="${FEATURE_SETS:-tabular,tabular_graph64,graph64,tabular_graph128}"
MODELS="${MODELS:-ridge,huber,elastic,extratrees,rf,hgb,gbr}"
TARGET_MODES="${TARGET_MODES:-absolute,residual_dopant,residual_llm}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

mkdir -p "$LOG_DIR" "$OUT_DIR"

echo "[INFO] ROOT=$ROOT"
echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] PY_MM=$PY_MM"
echo "[INFO] TRAIN_FEATURES=$TRAIN_FEATURES"
echo "[INFO] ADDHOUT_FEATURES=$ADDHOUT_FEATURES"
echo "[INFO] GRAPH_BUNDLE=$GRAPH_BUNDLE"
echo "[INFO] ANCHOR_CSV=$ANCHOR_CSV"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] MODELS=$MODELS"
echo "[INFO] FEATURE_SETS=$FEATURE_SETS"
echo "[INFO] TARGET_MODES=$TARGET_MODES"

if [[ ! -s "$TRAIN_FEATURES" ]]; then
  echo "[ERROR] missing train features: $TRAIN_FEATURES" >&2
  exit 2
fi
if [[ ! -s "$ADDHOUT_FEATURES" ]]; then
  echo "[ERROR] missing AddH-out features: $ADDHOUT_FEATURES" >&2
  exit 2
fi

"$PY_MM" -m py_compile 32_train_target_domain_aware_addhout.py

"$PY_MM" 32_train_target_domain_aware_addhout.py \
  --train-features "$TRAIN_FEATURES" \
  --addhout-features "$ADDHOUT_FEATURES" \
  --graph-bundle "$GRAPH_BUNDLE" \
  --graph-train-meta "$GRAPH_TRAIN_META" \
  --graph-addhout-meta "$GRAPH_ADDHOUT_META" \
  --anchor-csv "$ANCHOR_CSV" \
  --fallback-anchor-csv "$FALLBACK_ANCHOR_CSV" \
  --out-dir "$OUT_DIR" \
  --audit-labels-csv "$AUDIT_LABELS_CSV" \
  --n-splits "$N_SPLITS" \
  --top-k "$TOP_K" \
  --feature-sets "$FEATURE_SETS" \
  --models "$MODELS" \
  --target-modes "$TARGET_MODES" \
  --final-mode "$FINAL_MODE" \
  --write-xlsx

echo "[DONE]"
echo "[PRED]   $OUT_DIR/domain_aware_addhout_predictions.csv"
echo "[AUDIT]  $OUT_DIR/domain_aware_posthoc_audit.csv"
echo "[MODELS] $OUT_DIR/domain_aware_selected_models.csv"
echo "[SHIFT]  $OUT_DIR/domain_shift_feature_report.csv"
