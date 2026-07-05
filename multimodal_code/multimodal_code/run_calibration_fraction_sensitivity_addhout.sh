#!/usr/bin/env bash
set -euo pipefail

# Calibration-size sensitivity for AddH-out few-shot validation.

ROOT="${ROOT:-/data/home/terminator/RL/multi-view}"
RUN_ROOT="${RUN_ROOT:-$ROOT}"
PY_MM="${PY_MM:-/data/home/terminator/anaconda3/envs/multiview/bin/python}"

PRED_CSV="${PRED_CSV:-$RUN_ROOT/outputs_addh_bidirectional_chemistry_prior/bidirectional_chemistry_addhout_predictions.csv}"
LABELS_CSV="${LABELS_CSV:-$RUN_ROOT/outputs_addh_llm_element_priors/addhout_audit_labels.csv}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/outputs_addh_calibration_fraction_sensitivity}"
ANCHOR_COL="${ANCHOR_COL:-pred_chem_spike_final}"
CALIBRATION_FRACS="${CALIBRATION_FRACS:-0.10,0.20,0.30,0.50,0.65,0.70,0.80}"
N_REPEATS="${N_REPEATS:-500}"
SEED="${SEED:-42}"

cd "$ROOT"

echo "[INFO] ROOT=$ROOT"
echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] PY_MM=$PY_MM"
echo "[INFO] PRED_CSV=$PRED_CSV"
echo "[INFO] LABELS_CSV=$LABELS_CSV"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] ANCHOR_COL=$ANCHOR_COL CALIBRATION_FRACS=$CALIBRATION_FRACS N_REPEATS=$N_REPEATS"

if [[ ! -s "$PRED_CSV" ]]; then
  echo "[ERROR] missing prediction CSV: $PRED_CSV" >&2
  exit 2
fi
if [[ ! -s "$LABELS_CSV" ]]; then
  echo "[ERROR] missing labels CSV: $LABELS_CSV" >&2
  exit 2
fi

"$PY_MM" -m py_compile 36_validate_addhout_fewshot_holdout.py 37_calibration_fraction_sensitivity_addhout.py

"$PY_MM" 37_calibration_fraction_sensitivity_addhout.py \
  --pred-csv "$PRED_CSV" \
  --labels-csv "$LABELS_CSV" \
  --out-dir "$OUT_DIR" \
  --anchor-col "$ANCHOR_COL" \
  --calibration-fracs "$CALIBRATION_FRACS" \
  --n-repeats "$N_REPEATS" \
  --seed "$SEED" \
  --write-xlsx

echo "[DONE] Calibration sensitivity outputs:"
echo "  $OUT_DIR/calibration_fraction_summary.csv"
echo "  $OUT_DIR/calibration_fraction_paired_improvement.csv"
echo "  $OUT_DIR/calibration_fraction_sensitivity_report.xlsx"
