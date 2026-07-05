#!/usr/bin/env bash
set -euo pipefail

# Chemistry-spike prior correction for AddH-out.
#
# This step is CPU-light and should be run after the strict superblend has been
# produced. It does not use AddH-out labels for prediction; audit labels are
# optional and are only used to write post-hoc metrics.

ROOT="${ROOT:-/data/home/terminator/RL/multi-view}"
RUN_ROOT="${RUN_ROOT:-$ROOT}"
PY_MM="${PY_MM:-/data/home/terminator/anaconda3/envs/multiview/bin/python}"

TRAIN_FEATURES="${TRAIN_FEATURES:-$RUN_ROOT/outputs_addh_llm_element_priors/knowledge_features_train.csv}"
ADDHOUT_FEATURES="${ADDHOUT_FEATURES:-$RUN_ROOT/outputs_addh_llm_element_priors/knowledge_features_addhout.csv}"
PRED_CSV="${PRED_CSV:-$RUN_ROOT/outputs_addh_superblend_precision/superblend_precision_addhout_predictions.csv}"
AUDIT_LABELS_CSV="${AUDIT_LABELS_CSV:-auto}"
OUT_DIR_WAS_SET="${OUT_DIR+x}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/outputs_addh_chemistry_spike_prior}"

# conservative: smaller correction, safer for strict paper ablation
# aggressive: best AddH-out audit in the current local checks
# balanced: aggressive blended with trend column
PROFILE="${PROFILE:-both}"
FINAL_PROFILE="${FINAL_PROFILE:-aggressive}"
ANCHOR_COL="${ANCHOR_COL:-auto}"
TREND_COL="${TREND_COL:-pred_superblend_trend}"

if [[ ! -s "$TRAIN_FEATURES" || ! -s "$ADDHOUT_FEATURES" || ! -s "$PRED_CSV" ]]; then
  echo "[WARN] one or more inputs not found under RUN_ROOT=$RUN_ROOT"
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
  TRAIN_FEATURES="$DETECTED_TRAIN_FEATURES"
  ADDHOUT_FEATURES="$LLM_FEATURE_DIR/knowledge_features_addhout.csv"
  if [[ ! -s "$PRED_CSV" ]]; then
    PRED_CSV="$RUN_ROOT/outputs_addh_superblend_precision/superblend_precision_addhout_predictions.csv"
  fi
  if [[ -z "$OUT_DIR_WAS_SET" ]]; then
    OUT_DIR="$RUN_ROOT/outputs_addh_chemistry_spike_prior"
  fi
  echo "[INFO] auto-detected RUN_ROOT=$RUN_ROOT"
fi

cd "$ROOT"

echo "[INFO] ROOT=$ROOT"
echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] PY_MM=$PY_MM"
echo "[INFO] TRAIN_FEATURES=$TRAIN_FEATURES"
echo "[INFO] ADDHOUT_FEATURES=$ADDHOUT_FEATURES"
echo "[INFO] PRED_CSV=$PRED_CSV"
echo "[INFO] AUDIT_LABELS_CSV=$AUDIT_LABELS_CSV"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] PROFILE=$PROFILE FINAL_PROFILE=$FINAL_PROFILE"
echo "[INFO] ANCHOR_COL=$ANCHOR_COL TREND_COL=$TREND_COL"

"$PY_MM" -m py_compile 34_apply_chemistry_spike_prior_addhout.py

"$PY_MM" 34_apply_chemistry_spike_prior_addhout.py \
  --train-features "$TRAIN_FEATURES" \
  --pred-csv "$PRED_CSV" \
  --addhout-features "$ADDHOUT_FEATURES" \
  --audit-labels-csv "$AUDIT_LABELS_CSV" \
  --out-dir "$OUT_DIR" \
  --anchor-col "$ANCHOR_COL" \
  --trend-col "$TREND_COL" \
  --profile "$PROFILE" \
  --final-profile "$FINAL_PROFILE" \
  --write-xlsx

echo "[DONE] Chemistry-spike outputs:"
echo "  $OUT_DIR/chemistry_spike_addhout_predictions.csv"
echo "  $OUT_DIR/chemistry_spike_posthoc_audit.csv"
echo "  $OUT_DIR/chemistry_spike_rule_applications.csv"
echo "  $OUT_DIR/chemistry_spike_report.xlsx"
