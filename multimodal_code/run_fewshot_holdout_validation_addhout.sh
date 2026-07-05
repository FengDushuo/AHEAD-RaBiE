#!/usr/bin/env bash
set -euo pipefail

# Internal AddH-out few-shot validation.
# This is the paper-safer route when no new external AddH-out-like dataset exists.

ROOT="${ROOT:-/data/home/terminator/RL/multi-view}"
RUN_ROOT="${RUN_ROOT:-$ROOT}"
PY_MM="${PY_MM:-/data/home/terminator/anaconda3/envs/multiview/bin/python}"

PRED_CSV="${PRED_CSV:-$RUN_ROOT/outputs_addh_bidirectional_chemistry_prior/bidirectional_chemistry_addhout_predictions.csv}"
LABELS_CSV="${LABELS_CSV:-$RUN_ROOT/outputs_addh_llm_element_priors/addhout_audit_labels.csv}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/outputs_addh_fewshot_holdout_validation}"
ANCHOR_COL="${ANCHOR_COL:-pred_chem_spike_final}"
N_REPEATS="${N_REPEATS:-500}"
TEST_FRAC="${TEST_FRAC:-0.35}"
SEED="${SEED:-42}"

if [[ ! -s "$PRED_CSV" ]]; then
  echo "[WARN] bidirectional prediction file not found under RUN_ROOT=$RUN_ROOT"
  DETECTED_PRED="$(
    find "$ROOT" -path "*/outputs_addh_bidirectional_chemistry_prior/bidirectional_chemistry_addhout_predictions.csv" \
      -type f -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
  if [[ -z "$DETECTED_PRED" || ! -s "$DETECTED_PRED" ]]; then
    echo "[ERROR] could not find bidirectional chemistry predictions under $ROOT" >&2
    exit 2
  fi
  PRED_CSV="$DETECTED_PRED"
  RUN_ROOT="$(dirname "$(dirname "$PRED_CSV")")"
  LABELS_CSV="$RUN_ROOT/outputs_addh_llm_element_priors/addhout_audit_labels.csv"
  OUT_DIR="$RUN_ROOT/outputs_addh_fewshot_holdout_validation"
  echo "[INFO] auto-detected RUN_ROOT=$RUN_ROOT"
fi

if [[ ! -s "$LABELS_CSV" ]]; then
  echo "[ERROR] missing labels csv: $LABELS_CSV" >&2
  echo "[HINT] create/copy outputs_addh_llm_element_priors/addhout_audit_labels.csv first." >&2
  exit 2
fi

cd "$ROOT"

echo "[INFO] ROOT=$ROOT"
echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] PY_MM=$PY_MM"
echo "[INFO] PRED_CSV=$PRED_CSV"
echo "[INFO] LABELS_CSV=$LABELS_CSV"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] ANCHOR_COL=$ANCHOR_COL N_REPEATS=$N_REPEATS TEST_FRAC=$TEST_FRAC SEED=$SEED"

"$PY_MM" -m py_compile 36_validate_addhout_fewshot_holdout.py

"$PY_MM" 36_validate_addhout_fewshot_holdout.py \
  --pred-csv "$PRED_CSV" \
  --labels-csv "$LABELS_CSV" \
  --out-dir "$OUT_DIR" \
  --anchor-col "$ANCHOR_COL" \
  --n-repeats "$N_REPEATS" \
  --test-frac "$TEST_FRAC" \
  --seed "$SEED" \
  --write-xlsx

echo "[DONE] Few-shot holdout validation outputs:"
echo "  $OUT_DIR/fewshot_holdout_summary.csv"
echo "  $OUT_DIR/fewshot_holdout_split_metrics.csv"
echo "  $OUT_DIR/fewshot_holdout_predictions.csv"
echo "  $OUT_DIR/full_data_reference_metrics.csv"
echo "  $OUT_DIR/fewshot_holdout_validation_report.xlsx"
