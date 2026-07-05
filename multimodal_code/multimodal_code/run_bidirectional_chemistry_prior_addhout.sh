#!/usr/bin/env bash
set -euo pipefail

# Second-stage bidirectional chemistry prior for AddH-out.
# Run after outputs_addh_chemistry_spike_prior has been generated.

ROOT="${ROOT:-/data/home/terminator/RL/multi-view}"
RUN_ROOT="${RUN_ROOT:-$ROOT}"
PY_MM="${PY_MM:-/data/home/terminator/anaconda3/envs/multiview/bin/python}"

PRED_CSV="${PRED_CSV:-$RUN_ROOT/outputs_addh_chemistry_spike_prior/chemistry_spike_addhout_predictions.csv}"
AUDIT_LABELS_CSV="${AUDIT_LABELS_CSV:-auto}"
OUT_DIR_WAS_SET="${OUT_DIR+x}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/outputs_addh_bidirectional_chemistry_prior}"
ANCHOR_COL="${ANCHOR_COL:-pred_chem_spike_final}"
PROFILE="${PROFILE:-both}"
FINAL_PROFILE="${FINAL_PROFILE:-aggressive}"

if [[ ! -s "$PRED_CSV" ]]; then
  echo "[WARN] prediction file not found under RUN_ROOT=$RUN_ROOT"
  echo "[WARN] trying to auto-detect latest chemistry-spike output under $ROOT"
  DETECTED_PRED="$(
    find "$ROOT" -path "*/outputs_addh_chemistry_spike_prior/chemistry_spike_addhout_predictions.csv" \
      -type f -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
  if [[ -z "$DETECTED_PRED" || ! -s "$DETECTED_PRED" ]]; then
    echo "[ERROR] could not find outputs_addh_chemistry_spike_prior/chemistry_spike_addhout_predictions.csv under $ROOT" >&2
    exit 2
  fi
  PRED_CSV="$DETECTED_PRED"
  RUN_ROOT="$(dirname "$(dirname "$PRED_CSV")")"
  if [[ -z "$OUT_DIR_WAS_SET" ]]; then
    OUT_DIR="$RUN_ROOT/outputs_addh_bidirectional_chemistry_prior"
  fi
  echo "[INFO] auto-detected RUN_ROOT=$RUN_ROOT"
fi

cd "$ROOT"

echo "[INFO] ROOT=$ROOT"
echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] PY_MM=$PY_MM"
echo "[INFO] PRED_CSV=$PRED_CSV"
echo "[INFO] AUDIT_LABELS_CSV=$AUDIT_LABELS_CSV"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] ANCHOR_COL=$ANCHOR_COL PROFILE=$PROFILE FINAL_PROFILE=$FINAL_PROFILE"

"$PY_MM" -m py_compile 35_apply_bidirectional_chemistry_prior_addhout.py

"$PY_MM" 35_apply_bidirectional_chemistry_prior_addhout.py \
  --pred-csv "$PRED_CSV" \
  --out-dir "$OUT_DIR" \
  --anchor-col "$ANCHOR_COL" \
  --audit-labels-csv "$AUDIT_LABELS_CSV" \
  --profile "$PROFILE" \
  --final-profile "$FINAL_PROFILE" \
  --write-xlsx

echo "[DONE] Bidirectional chemistry outputs:"
echo "  $OUT_DIR/bidirectional_chemistry_addhout_predictions.csv"
echo "  $OUT_DIR/bidirectional_chemistry_posthoc_audit.csv"
echo "  $OUT_DIR/bidirectional_chemistry_rule_applications.csv"
echo "  $OUT_DIR/bidirectional_chemistry_report.xlsx"
