#!/usr/bin/env bash
set -euo pipefail

INP="${INP:-outputs/hads_canonical_records.jsonl}"
OUTDIR="${OUTDIR:-outputs/hads_db}"
LOGDIR="${LOGDIR:-logs_hads_step4}"
FORCE="${FORCE:-0}"

mkdir -p "$LOGDIR"

timestamp(){ date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(timestamp)] [INFO] Hads Step4 build_normalized_database starting"

if [[ ! -s "$INP" ]]; then
  echo "[$(timestamp)] [ERROR] Missing or empty input: $INP" | tee -a "$LOGDIR/step4_build_hads_normalized_database.log"
  exit 1
fi

if [[ "$FORCE" == "1" ]]; then
  echo "[$(timestamp)] [INFO] FORCE=1 -> removing existing output dir: $OUTDIR" | tee -a "$LOGDIR/step4_build_hads_normalized_database.log"
  rm -rf "$OUTDIR"
fi

python 4_build_hads_normalized_database.py --in "$INP" --outdir "$OUTDIR" > "$LOGDIR/step4_build_hads_normalized_database.log" 2>&1

REQUIRED_FILES=(
  "$OUTDIR/paper_table.csv"
  "$OUTDIR/system_table.csv"
  "$OUTDIR/site_table.csv"
  "$OUTDIR/adsorption_table.csv"
  "$OUTDIR/model_feature_table.csv"
  "$OUTDIR/system_summary_table.csv"
  "$OUTDIR/qc_database_summary.csv"
  "$OUTDIR/dft_priority_candidates.csv"
)

for f in "${REQUIRED_FILES[@]}"; do
  if [[ ! -s "$f" ]]; then
    echo "[$(timestamp)] [ERROR] Expected output missing or empty: $f" | tee -a "$LOGDIR/step4_build_hads_normalized_database.log"
    exit 1
  fi
done

echo "[$(timestamp)] [DONE] Hads Step4 build_normalized_database finished successfully"
printf '  %s\n' "${REQUIRED_FILES[@]}"
