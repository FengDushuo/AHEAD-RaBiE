#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/home/terminator/RL/multi-view}"
RUN_ROOT="${RUN_ROOT:-$ROOT}"
PY_MM="${PY_MM:-/data/home/terminator/anaconda3/envs/multiview/bin/python}"

HOLDOUT_DIR="${HOLDOUT_DIR:-$RUN_ROOT/outputs_addh_fewshot_holdout_validation}"
SENSITIVITY_DIR="${SENSITIVITY_DIR:-$RUN_ROOT/outputs_addh_calibration_fraction_sensitivity}"
PRED_CSV="${PRED_CSV:-$RUN_ROOT/outputs_addh_bidirectional_chemistry_prior/bidirectional_chemistry_addhout_predictions.csv}"
LABELS_CSV="${LABELS_CSV:-$RUN_ROOT/outputs_addh_llm_element_priors/addhout_audit_labels.csv}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/outputs_addh_science_figures}"
FIG_DPI="${FIG_DPI:-600}"
FIG_FORMATS="${FIG_FORMATS:-pdf,svg,png}"
TRAINING_LOG_CSV="${TRAINING_LOG_CSV:-}"

echo "[INFO] ROOT=$ROOT"
echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] PY_MM=$PY_MM"
echo "[INFO] HOLDOUT_DIR=$HOLDOUT_DIR"
echo "[INFO] SENSITIVITY_DIR=$SENSITIVITY_DIR"
echo "[INFO] PRED_CSV=$PRED_CSV"
echo "[INFO] LABELS_CSV=$LABELS_CSV"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] FIG_DPI=$FIG_DPI FIG_FORMATS=$FIG_FORMATS"
if [[ -n "$TRAINING_LOG_CSV" ]]; then
  echo "[INFO] TRAINING_LOG_CSV=$TRAINING_LOG_CSV"
fi

cd "$ROOT"

cmd=(
  "$PY_MM" "$ROOT/39_make_science_submission_figures.py"
  --holdout-dir "$HOLDOUT_DIR"
  --sensitivity-dir "$SENSITIVITY_DIR"
  --pred-csv "$PRED_CSV"
  --labels-csv "$LABELS_CSV"
  --out-dir "$OUT_DIR"
  --dpi "$FIG_DPI"
  --formats "$FIG_FORMATS"
)

if [[ -n "$TRAINING_LOG_CSV" ]]; then
  cmd+=(--training-log-csv "$TRAINING_LOG_CSV")
fi

"${cmd[@]}"

echo "[DONE] Science-style figures:"
find "$OUT_DIR" -maxdepth 3 -type f | sort
