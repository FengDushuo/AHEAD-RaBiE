#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-pdf-files}"
OUT="${OUT:-pdf-files-parsed}"
GROBID_URL="${GROBID_URL:-http://127.0.0.1:8070}"
TABLE_ENGINES="${TABLE_ENGINES:-camelot}"
WORKERS="${WORKERS:-8}"
FORCE="${FORCE:-0}"
SKIP_LEGACY="${SKIP_LEGACY:-0}"
LEGACY_PARSER="${LEGACY_PARSER:-./1_parse_corpus.py}"
LOGDIR="${LOGDIR:-logs_step1}"
mkdir -p "$LOGDIR"
timestamp(){ date "+%Y-%m-%d %H:%M:%S"; }
echo "[$(timestamp)] [INFO] Step1 parse_hads_corpus starting"
[[ -d "$ROOT" ]] || { echo "[$(timestamp)] [ERROR] Input root directory not found: $ROOT" | tee -a "$LOGDIR/step1_parse_hads_corpus.log"; exit 1; }
[[ -f "$LEGACY_PARSER" ]] || { echo "[$(timestamp)] [ERROR] Legacy parser not found: $LEGACY_PARSER" | tee -a "$LOGDIR/step1_parse_hads_corpus.log"; exit 1; }
CMD=(python 1_parse_echem_corpus.py --root "$ROOT" --out "$OUT" --grobid-url "$GROBID_URL" --table-engines "$TABLE_ENGINES" --workers "$WORKERS" --legacy-parser "$LEGACY_PARSER")
if [[ "$FORCE" == "1" ]]; then CMD+=(--force); fi
if [[ "$SKIP_LEGACY" == "1" ]]; then CMD+=(--skip-legacy); fi
"${CMD[@]}" > "$LOGDIR/step1_parse_hads_corpus.log" 2>&1
REQUIRED_FILES=("$OUT/manifest.csv" "$OUT/qc_report.csv" "$OUT/paper_domain_tags.csv" "$OUT/section_manifest.csv" "$OUT/table_header_semantics.csv" "$OUT/enhanced_parse_summary.json")
for f in "${REQUIRED_FILES[@]}"; do [[ -s "$f" ]] || { echo "[$(timestamp)] [ERROR] Expected output missing or empty: $f" | tee -a "$LOGDIR/step1_parse_hads_corpus.log"; exit 1; }; done
echo "[$(timestamp)] [DONE] Step1 parse_hads_corpus finished successfully"
printf '  %s\n' "${REQUIRED_FILES[@]}"
