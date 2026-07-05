#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# HADS extraction pipeline: SCNet DeepSeek-V4-Flash accurate mode
# Step3 extraction -> Step4 normalized DB -> Step5 stats/ML -> figures
# ============================================================
# Usage:
#   export SCNET_API_KEY="sk-xxxx"
#   nohup bash run_hads_scnet_v4flash_accurate_full.sh > run_hads_scnet_v4flash_accurate_full.log 2>&1 &
# Resume instead of overwrite:
#   FORCE=0 nohup bash run_hads_scnet_v4flash_accurate_full.sh > run_hads_scnet_v4flash_accurate_full.log 2>&1 &
# ============================================================

# ---------- safety checks ----------
: "${SCNET_API_KEY:?Please export SCNET_API_KEY first, e.g. export SCNET_API_KEY='sk-xxxx'}"

# ---------- basic paths ----------
export PARSED_ROOT="${PARSED_ROOT:-pdf-files-parsed}"
export MANIFEST="${MANIFEST:-$PARSED_ROOT/manifest.csv}"
export PAPER_TAGS="${PAPER_TAGS:-$PARSED_ROOT/paper_domain_tags.csv}"
export QC_REPORT="${QC_REPORT:-$PARSED_ROOT/qc_report.csv}"

RUN_TAG="${RUN_TAG:-scnet_v4flash_accurate}"
STEP3_OUT="${STEP3_OUT:-outputs_${RUN_TAG}_hads}"
STEP3_LOG="${STEP3_LOG:-logs_${RUN_TAG}_step3}"
DB_OUT="${DB_OUT:-outputs_${RUN_TAG}_hads_db}"
STEP4_LOG="${STEP4_LOG:-logs_${RUN_TAG}_step4}"
STEP5_OUT="${STEP5_OUT:-outputs_${RUN_TAG}_step5_stats_ml}"
FIG_OUT="${FIG_OUT:-outputs_${RUN_TAG}_publication_figures}"
MASTER_LOGDIR="${MASTER_LOGDIR:-logs_${RUN_TAG}_master}"

mkdir -p "$STEP3_OUT" "$STEP3_LOG" "$DB_OUT" "$STEP4_LOG" "$STEP5_OUT" "$FIG_OUT" "$MASTER_LOGDIR"

# ---------- SCNet / LLM settings ----------
export SCNET_MODE=1
export SCNET_BASE_URL="${SCNET_BASE_URL:-https://api.scnet.cn/api/llm/v1}"
export API_BASES="${API_BASES:-$SCNET_BASE_URL}"
export MODEL_ID="${MODEL_ID:-DeepSeek-V4-Flash}"

# Use the same long-context model for verification by default.
export VERIFY_LLM="${VERIFY_LLM:-1}"
export VERIFIER_API_BASES="${VERIFIER_API_BASES:-$API_BASES}"
export VERIFIER_MODEL_ID="${VERIFIER_MODEL_ID:-$MODEL_ID}"
export VERIFY_MAX_TOKENS="${VERIFY_MAX_TOKENS:-7000}"

# ---------- retrieval / embedding settings ----------
export EMBED_MODEL="${EMBED_MODEL:-/data/home/terminator/LLM/llm-models/bge-m3}"
export DEVICE="${DEVICE:-cuda}"
export QDRANT_PREFIX="${QDRANT_PREFIX:-qdrant_local_shard}"
export COLLECTION_PREFIX="${COLLECTION_PREFIX:-papers}"
export SHARD_IDS="${SHARD_IDS:-0,1,2}"

# ---------- run mode ----------
# FORCE=1: start a clean, most reproducible extraction.
# FORCE=0: resume from progress files where possible.
export FORCE="${FORCE:-1}"
export FAST_MODE=0
export HIGH_RECALL=1

# External API: keep workers conservative to avoid rate-limit and cross-request instability.
export WORKERS_3B="${WORKERS_3B:-1}"
export WORKERS_3C="${WORKERS_3C:-1}"
export WORKERS_3D="${WORKERS_3D:-1}"
export TIMEOUT="${TIMEOUT:-420}"
export MAX_RETRIES="${MAX_RETRIES:-6}"

# Triage thresholds: slightly relaxed to reduce false negatives.
export MIN_RELEVANCE="${MIN_RELEVANCE:-0.14}"
export MIN_SYSTEM_SIGNAL="${MIN_SYSTEM_SIGNAL:-0.10}"
export MIN_METRIC_SIGNAL="${MIN_METRIC_SIGNAL:-0.08}"

# ---------- 1M-context-aware evidence settings ----------
# Important: we still do evidence-guided extraction rather than sending full papers.
# This keeps the evidence relevant and reduces cross-system numeric contamination.

# Step3b: catalyst / surface / system extraction
export TOPK_MAIN_3B="${TOPK_MAIN_3B:-36}"
export TOPK_TABLE_3B="${TOPK_TABLE_3B:-56}"
export TOPK_ENTITY_3B="${TOPK_ENTITY_3B:-14}"
export MAIN_BUDGET_CHARS_3B="${MAIN_BUDGET_CHARS_3B:-60000}"
export TABLE_BUDGET_CHARS_3B="${TABLE_BUDGET_CHARS_3B:-70000}"
export ENTITY_BUDGET_CHARS_3B="${ENTITY_BUDGET_CHARS_3B:-9000}"
export MAX_ITEMS_MAIN_3B="${MAX_ITEMS_MAIN_3B:-28}"
export MAX_ITEMS_TABLE_3B="${MAX_ITEMS_TABLE_3B:-34}"
export MAX_ITEMS_ENTITY_3B="${MAX_ITEMS_ENTITY_3B:-8}"
export MAX_TOKENS_3B="${MAX_TOKENS_3B:-6500}"

# Step3c: active-site / local descriptor extraction
export TOPK_MAIN_3C="${TOPK_MAIN_3C:-44}"
export TOPK_TABLE_3C="${TOPK_TABLE_3C:-72}"
export TOPK_ENTITY_3C="${TOPK_ENTITY_3C:-16}"
export MAIN_BUDGET_CHARS_3C="${MAIN_BUDGET_CHARS_3C:-80000}"
export TABLE_BUDGET_CHARS_3C="${TABLE_BUDGET_CHARS_3C:-90000}"
export ENTITY_BUDGET_CHARS_3C="${ENTITY_BUDGET_CHARS_3C:-11000}"
export MAX_ITEMS_MAIN_3C="${MAX_ITEMS_MAIN_3C:-34}"
export MAX_ITEMS_TABLE_3C="${MAX_ITEMS_TABLE_3C:-42}"
export MAX_ITEMS_ENTITY_3C="${MAX_ITEMS_ENTITY_3C:-8}"
export MAX_TOKENS_3C="${MAX_TOKENS_3C:-8500}"

# Step3d: H adsorption / deprotonation / Volmer metrics extraction
# This is the most important step for ΔE_H*, ΔG_H*, deprotonation energy, and barriers.
export TOPK_MAIN_3D="${TOPK_MAIN_3D:-56}"
export TOPK_TABLE_3D="${TOPK_TABLE_3D:-96}"
export TOPK_ENTITY_3D="${TOPK_ENTITY_3D:-18}"
export MAIN_BUDGET_CHARS_3D="${MAIN_BUDGET_CHARS_3D:-100000}"
export TABLE_BUDGET_CHARS_3D="${TABLE_BUDGET_CHARS_3D:-140000}"
export ENTITY_BUDGET_CHARS_3D="${ENTITY_BUDGET_CHARS_3D:-14000}"
export MAX_ITEMS_MAIN_3D="${MAX_ITEMS_MAIN_3D:-42}"
export MAX_ITEMS_TABLE_3D="${MAX_ITEMS_TABLE_3D:-56}"
export MAX_ITEMS_ENTITY_3D="${MAX_ITEMS_ENTITY_3D:-9}"
export MAX_TOKENS_3D="${MAX_TOKENS_3D:-11000}"

# Per-evidence quote length. These are intentionally larger than the old defaults.
export MAX_TEXT_CHARS="${MAX_TEXT_CHARS:-5000}"
export MAX_TABLE_CHARS="${MAX_TABLE_CHARS:-8000}"
export MAX_CAPTION_CHARS="${MAX_CAPTION_CHARS:-5000}"

# ---------- Step5 / plotting settings ----------
export QC_FILTER="${QC_FILTER:-1}"
export GROUP_COL="${GROUP_COL:-paper_id}"
export N_SPLITS="${N_SPLITS:-5}"
export SEED="${SEED:-2026}"

# ---------- helper functions ----------
timestamp(){ date "+%Y-%m-%d %H:%M:%S"; }
log(){ echo "[$(timestamp)] $*"; }
require_file(){ [[ -s "$1" ]] || { log "[ERROR] Missing or empty file: $1"; exit 1; }; }
run_logged(){
  local name="$1"; shift
  local logf="$1"; shift
  log "[RUN] $name"
  if ! "$@" > "$logf" 2>&1; then
    log "[ERROR] $name failed. Log tail: $logf"
    tail -n 180 "$logf" || true
    exit 1
  fi
  log "[DONE] $name"
}

# ---------- preflight ----------
log "[INFO] Starting HADS SCNet accurate full pipeline"
log "[INFO] MODEL_ID=$MODEL_ID"
log "[INFO] VERIFIER_MODEL_ID=$VERIFIER_MODEL_ID VERIFY_LLM=$VERIFY_LLM"
log "[INFO] STEP3_OUT=$STEP3_OUT"
log "[INFO] DB_OUT=$DB_OUT"
log "[INFO] STEP5_OUT=$STEP5_OUT"
log "[INFO] FIG_OUT=$FIG_OUT"
log "[INFO] FORCE=$FORCE; set FORCE=0 to resume instead of overwrite"

require_file "$MANIFEST"
for f in 3_run_extraction_pipeline_hads.sh 4_build_hads_normalized_database.sh 5_run_stats_ml_hads.sh; do
  [[ -f "$f" ]] || { log "[ERROR] Missing script: $f"; exit 1; }
done

if [[ -f scnet_smoke_test.py ]]; then
  run_logged "SCNet smoke test" "$MASTER_LOGDIR/scnet_smoke_test.log" \
    python scnet_smoke_test.py --api-base "$SCNET_BASE_URL" --model-id "$MODEL_ID"
else
  log "[WARN] scnet_smoke_test.py not found; skip smoke test"
fi

# ---------- Step3 ----------
run_logged "Step3 evidence-guided SCNet extraction" "$MASTER_LOGDIR/step3_wrapper.log" \
  env OUTDIR="$STEP3_OUT" LOGDIR="$STEP3_LOG" \
  bash 3_run_extraction_pipeline_hads.sh

require_file "$STEP3_OUT/hads_canonical_records.jsonl"

# ---------- Step4 ----------
run_logged "Step4 normalized database" "$MASTER_LOGDIR/step4_wrapper.log" \
  env INP="$STEP3_OUT/hads_canonical_records.jsonl" OUTDIR="$DB_OUT" LOGDIR="$STEP4_LOG" FORCE=1 \
  bash 4_build_hads_normalized_database.sh

require_file "$DB_OUT/model_feature_table.csv"
require_file "$DB_OUT/dft_priority_candidates.csv"
require_file "$DB_OUT/qc_database_summary.csv"

# ---------- Step5 ----------
run_logged "Step5 descriptive statistics / ML / reports" "$MASTER_LOGDIR/step5_wrapper.log" \
  env DB_OUTDIR="$DB_OUT" OUTROOT="$STEP5_OUT" QC_FILTER="$QC_FILTER" GROUP_COL="$GROUP_COL" N_SPLITS="$N_SPLITS" SEED="$SEED" \
  bash 5_run_stats_ml_hads.sh

# ---------- Figures ----------
PLOT_SCRIPT="plot_hads_publication_figures_v2_refined_layoutfix15_deltaE_normalized.py"
if [[ -f "$PLOT_SCRIPT" ]]; then
  run_logged "Publication figures" "$MASTER_LOGDIR/figures_wrapper.log" \
    python "$PLOT_SCRIPT" \
      --dbdir "$DB_OUT" \
      --step5dir "$STEP5_OUT" \
      --outdir "$FIG_OUT" \
      --dpi 900 \
      --font Arial \
      --formats png,pdf \
      --topk 10 \
      --bubble-max 260 \
      --deltae-detail \
      --deltae-target H_adsorption_energy_value_eV \
      --deltae-topn 12
else
  log "[WARN] Plot script not found: $PLOT_SCRIPT; skip figure generation"
fi

# ---------- quick summary ----------
log "[DONE] Full HADS SCNet accurate pipeline finished"
log "[RESULT] Step3 canonical records: $STEP3_OUT/hads_canonical_records.jsonl"
log "[RESULT] Database: $DB_OUT"
log "[RESULT] Step5 stats/ML: $STEP5_OUT"
log "[RESULT] Figures: $FIG_OUT"
log "[TIP] Check QC summary: $DB_OUT/qc_database_summary.csv"
