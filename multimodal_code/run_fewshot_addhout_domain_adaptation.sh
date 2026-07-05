#!/usr/bin/env bash
set -Eeuo pipefail

# Few-shot AddH-out target-domain adaptation.
# This route is NOT strict-blind: selected AddH-out labels are used only for
# calibration, and the remaining AddH-out rows are used for held-out reporting.

ROOT="${ROOT:-$(pwd)}"
cd "$ROOT"

PY_MM="${PY_MM:-python}"
RUN_ROOT="${RUN_ROOT:-$ROOT}"

PRED_CSV="${PRED_CSV:-$RUN_ROOT/outputs_addh_superblend_precision/superblend_precision_addhout_predictions.csv}"
if [[ ! -s "$PRED_CSV" && -s "$ROOT/outputs_addh_superblend_precision/superblend_precision_addhout_predictions.csv" ]]; then
  PRED_CSV="$ROOT/outputs_addh_superblend_precision/superblend_precision_addhout_predictions.csv"
fi

LABELS="${LABELS:-auto}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/outputs_addh_fewshot_domain_calibration}"

SHOTS_PER_MATERIAL="${SHOTS_PER_MATERIAL:-0,1,2,3,4,5,6,8,10}"
REPEATS="${REPEATS:-200}"
BASE_COLS="${BASE_COLS:-auto}"

# For MAE-first use pred_superblend_mae_guarded.
# For trend-preserving adaptation use pred_superblend_balanced or pred_superblend_trend.
OPERATIONAL_SHOTS_PER_MATERIAL="${OPERATIONAL_SHOTS_PER_MATERIAL:-4}"
OPERATIONAL_BASE_COL="${OPERATIONAL_BASE_COL:-pred_superblend_mae_guarded}"
OPERATIONAL_CALIBRATOR="${OPERATIONAL_CALIBRATOR:-guarded_auto}"
CALIBRATION_IDS="${CALIBRATION_IDS:-}"

echo "[INFO] ROOT=$ROOT"
echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] PY_MM=$PY_MM"
echo "[INFO] PRED_CSV=$PRED_CSV"
echo "[INFO] LABELS=$LABELS"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] SHOTS_PER_MATERIAL=$SHOTS_PER_MATERIAL REPEATS=$REPEATS"
echo "[INFO] OPERATIONAL_SHOTS_PER_MATERIAL=$OPERATIONAL_SHOTS_PER_MATERIAL"
echo "[INFO] OPERATIONAL_BASE_COL=$OPERATIONAL_BASE_COL"
echo "[INFO] OPERATIONAL_CALIBRATOR=$OPERATIONAL_CALIBRATOR"

"$PY_MM" -m py_compile 31_fewshot_addhout_domain_adaptation.py

"$PY_MM" 31_fewshot_addhout_domain_adaptation.py \
  --pred-csv "$PRED_CSV" \
  --labels "$LABELS" \
  --out-dir "$OUT_DIR" \
  --base-cols "$BASE_COLS" \
  --shots-per-material "$SHOTS_PER_MATERIAL" \
  --repeats "$REPEATS" \
  --operational-shots-per-material "$OPERATIONAL_SHOTS_PER_MATERIAL" \
  --operational-base-col "$OPERATIONAL_BASE_COL" \
  --operational-calibrator "$OPERATIONAL_CALIBRATOR" \
  --calibration-ids "$CALIBRATION_IDS" \
  --write-xlsx

echo "[RESULT] $OUT_DIR/fewshot_holdout_summary.csv"
echo "[RESULT] $OUT_DIR/fewshot_recommended_by_holdout.csv"
echo "[RESULT] $OUT_DIR/fewshot_operational_predictions.csv"
echo "[AUDIT]  $OUT_DIR/fewshot_operational_audit.csv"
