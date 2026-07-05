#!/usr/bin/env bash
set -euo pipefail

# Build AddH-out paper tables and figures.

ROOT="${ROOT:-/data/home/terminator/RL/multi-view}"
RUN_ROOT="${RUN_ROOT:-$ROOT}"
PY_MM="${PY_MM:-/data/home/terminator/anaconda3/envs/multiview/bin/python}"

HOLDOUT_DIR="${HOLDOUT_DIR:-$RUN_ROOT/outputs_addh_fewshot_holdout_validation}"
SENSITIVITY_DIR="${SENSITIVITY_DIR:-$RUN_ROOT/outputs_addh_calibration_fraction_sensitivity}"
PRED_CSV="${PRED_CSV:-$RUN_ROOT/outputs_addh_bidirectional_chemistry_prior/bidirectional_chemistry_addhout_predictions.csv}"
LABELS_CSV="${LABELS_CSV:-$RUN_ROOT/outputs_addh_llm_element_priors/addhout_audit_labels.csv}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/outputs_addh_paper_artifacts}"

cd "$ROOT"

echo "[INFO] ROOT=$ROOT"
echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] PY_MM=$PY_MM"
echo "[INFO] HOLDOUT_DIR=$HOLDOUT_DIR"
echo "[INFO] SENSITIVITY_DIR=$SENSITIVITY_DIR"
echo "[INFO] PRED_CSV=$PRED_CSV"
echo "[INFO] LABELS_CSV=$LABELS_CSV"
echo "[INFO] OUT_DIR=$OUT_DIR"

for f in \
  "$HOLDOUT_DIR/fewshot_holdout_summary.csv" \
  "$HOLDOUT_DIR/fewshot_holdout_split_metrics.csv" \
  "$HOLDOUT_DIR/fewshot_holdout_predictions.csv" \
  "$HOLDOUT_DIR/full_data_reference_metrics.csv" \
  "$PRED_CSV" \
  "$LABELS_CSV"
do
  if [[ ! -s "$f" ]]; then
    echo "[ERROR] missing required input: $f" >&2
    exit 2
  fi
done

"$PY_MM" -m py_compile 38_make_addhout_paper_tables_figures.py

"$PY_MM" 38_make_addhout_paper_tables_figures.py \
  --holdout-dir "$HOLDOUT_DIR" \
  --sensitivity-dir "$SENSITIVITY_DIR" \
  --pred-csv "$PRED_CSV" \
  --labels-csv "$LABELS_CSV" \
  --out-dir "$OUT_DIR"

echo "[DONE] Paper artifacts:"
echo "  $OUT_DIR/tables/addhout_main_results.csv"
echo "  $OUT_DIR/tables/addhout_main_results.md"
echo "  $OUT_DIR/figures/"
