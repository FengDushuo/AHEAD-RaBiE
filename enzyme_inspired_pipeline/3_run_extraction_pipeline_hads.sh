#!/usr/bin/env bash
set -euo pipefail

PARSED_ROOT="${PARSED_ROOT:-pdf-files-parsed}"
OUTDIR="${OUTDIR:-outputs}"
LOGDIR="${LOGDIR:-logs_hads_step3}"
MANIFEST="${MANIFEST:-$PARSED_ROOT/manifest.csv}"
PAPER_TAGS="${PAPER_TAGS:-$PARSED_ROOT/paper_domain_tags.csv}"
QC_REPORT="${QC_REPORT:-$PARSED_ROOT/qc_report.csv}"

# -----------------------------------------------------------------------------
# LLM backend configuration
# -----------------------------------------------------------------------------
# Local vLLM mode (default): use API_BASES/MODEL_ID as before.
# SCNet mode: export SCNET_MODE=1 and SCNET_API_KEY=sk-...
#   Optional: export MODEL_ID="model name copied from SCNet console"
SCNET_MODE="${SCNET_MODE:-0}"
SCNET_BASE_URL="${SCNET_BASE_URL:-https://api.scnet.cn/api/llm/v1}"
SCNET_MODEL="${SCNET_MODEL:-DeepSeek-R1-Distill-Qwen-32B}"

if [[ "$SCNET_MODE" == "1" ]]; then
  API_BASES="${API_BASES:-$SCNET_BASE_URL}"
  MODEL_ID="${MODEL_ID:-$SCNET_MODEL}"
  VERIFY_LLM="${VERIFY_LLM:-1}"
  # External APIs usually have stricter rate limits; keep concurrency conservative.
  WORKERS_3B="${WORKERS_3B:-1}"; WORKERS_3C="${WORKERS_3C:-1}"; WORKERS_3D="${WORKERS_3D:-1}"
else
  API_BASES="${API_BASES:-http://192.168.94.12:8001/v1,http://192.168.94.12:8002/v1,http://192.168.94.12:8003/v1,http://192.168.94.12:8004/v1}"
  MODEL_ID="${MODEL_ID:-/data/home/terminator/LLM/llm-models/glm-4-9b}"
  VERIFY_LLM="${VERIFY_LLM:-0}"
fi
VERIFIER_API_BASES="${VERIFIER_API_BASES:-$API_BASES}"
VERIFIER_MODEL_ID="${VERIFIER_MODEL_ID:-$MODEL_ID}"
VERIFY_MAX_TOKENS="${VERIFY_MAX_TOKENS:-2800}"

# Descriptor repair after Step3e. Deterministic repair is cheap and improves
# joint coverage between ΔE_H* and site descriptors. Optional LLM repair is
# controlled separately because it adds API cost.
REPAIR_DESCRIPTORS="${REPAIR_DESCRIPTORS:-1}"
REPAIR_LLM="${REPAIR_LLM:-0}"
REPAIR_MAX_LLM_RECORDS="${REPAIR_MAX_LLM_RECORDS:-0}"
REPAIR_MIN_FUZZY_SCORE="${REPAIR_MIN_FUZZY_SCORE:-0.26}"

EMBED_MODEL="${EMBED_MODEL:-/data/home/terminator/LLM/llm-models/bge-m3}"
DEVICE="${DEVICE:-cuda}"
QDRANT_PREFIX="${QDRANT_PREFIX:-qdrant_local_shard}"
COLLECTION_PREFIX="${COLLECTION_PREFIX:-papers}"
SHARD_IDS="${SHARD_IDS:-0,1,2}"
FORCE="${FORCE:-0}"
FAST_MODE="${FAST_MODE:-0}"
HIGH_RECALL="${HIGH_RECALL:-1}"
TIMEOUT="${TIMEOUT:-240}"
MAX_RETRIES="${MAX_RETRIES:-5}"
MAX_TOKENS_3B="${MAX_TOKENS_3B:-2600}"
MAX_TOKENS_3C="${MAX_TOKENS_3C:-3400}"
MAX_TOKENS_3D="${MAX_TOKENS_3D:-4200}"
MIN_RELEVANCE="${MIN_RELEVANCE:-0.16}"
MIN_SYSTEM_SIGNAL="${MIN_SYSTEM_SIGNAL:-0.12}"
MIN_METRIC_SIGNAL="${MIN_METRIC_SIGNAL:-0.10}"

if [[ "$FAST_MODE" == "1" ]]; then
  WORKERS_3B="${WORKERS_3B:-2}"; WORKERS_3C="${WORKERS_3C:-2}"; WORKERS_3D="${WORKERS_3D:-2}"
  TOPK_MAIN_3B="${TOPK_MAIN_3B:-8}"; TOPK_TABLE_3B="${TOPK_TABLE_3B:-10}"; TOPK_ENTITY_3B="${TOPK_ENTITY_3B:-3}"
  TOPK_MAIN_3C="${TOPK_MAIN_3C:-8}"; TOPK_TABLE_3C="${TOPK_TABLE_3C:-12}"; TOPK_ENTITY_3C="${TOPK_ENTITY_3C:-3}"
  TOPK_MAIN_3D="${TOPK_MAIN_3D:-10}"; TOPK_TABLE_3D="${TOPK_TABLE_3D:-16}"; TOPK_ENTITY_3D="${TOPK_ENTITY_3D:-3}"
  MAIN_BUDGET_CHARS_3B="${MAIN_BUDGET_CHARS_3B:-5000}"; TABLE_BUDGET_CHARS_3B="${TABLE_BUDGET_CHARS_3B:-3000}"; ENTITY_BUDGET_CHARS_3B="${ENTITY_BUDGET_CHARS_3B:-800}"
  MAIN_BUDGET_CHARS_3C="${MAIN_BUDGET_CHARS_3C:-6000}"; TABLE_BUDGET_CHARS_3C="${TABLE_BUDGET_CHARS_3C:-3600}"; ENTITY_BUDGET_CHARS_3C="${ENTITY_BUDGET_CHARS_3C:-900}"
  MAIN_BUDGET_CHARS_3D="${MAIN_BUDGET_CHARS_3D:-7000}"; TABLE_BUDGET_CHARS_3D="${TABLE_BUDGET_CHARS_3D:-4800}"; ENTITY_BUDGET_CHARS_3D="${ENTITY_BUDGET_CHARS_3D:-900}"
  MAX_ITEMS_MAIN_3B="${MAX_ITEMS_MAIN_3B:-7}"; MAX_ITEMS_TABLE_3B="${MAX_ITEMS_TABLE_3B:-5}"; MAX_ITEMS_ENTITY_3B="${MAX_ITEMS_ENTITY_3B:-2}"
  MAX_ITEMS_MAIN_3C="${MAX_ITEMS_MAIN_3C:-7}"; MAX_ITEMS_TABLE_3C="${MAX_ITEMS_TABLE_3C:-5}"; MAX_ITEMS_ENTITY_3C="${MAX_ITEMS_ENTITY_3C:-2}"
  MAX_ITEMS_MAIN_3D="${MAX_ITEMS_MAIN_3D:-8}"; MAX_ITEMS_TABLE_3D="${MAX_ITEMS_TABLE_3D:-6}"; MAX_ITEMS_ENTITY_3D="${MAX_ITEMS_ENTITY_3D:-2}"
elif [[ "$HIGH_RECALL" == "1" ]]; then
  WORKERS_3B="${WORKERS_3B:-4}"; WORKERS_3C="${WORKERS_3C:-4}"; WORKERS_3D="${WORKERS_3D:-4}"
  TOPK_MAIN_3B="${TOPK_MAIN_3B:-18}"; TOPK_TABLE_3B="${TOPK_TABLE_3B:-28}"; TOPK_ENTITY_3B="${TOPK_ENTITY_3B:-8}"
  TOPK_MAIN_3C="${TOPK_MAIN_3C:-18}"; TOPK_TABLE_3C="${TOPK_TABLE_3C:-30}"; TOPK_ENTITY_3C="${TOPK_ENTITY_3C:-8}"
  TOPK_MAIN_3D="${TOPK_MAIN_3D:-22}"; TOPK_TABLE_3D="${TOPK_TABLE_3D:-36}"; TOPK_ENTITY_3D="${TOPK_ENTITY_3D:-8}"
  MAIN_BUDGET_CHARS_3B="${MAIN_BUDGET_CHARS_3B:-10000}"; TABLE_BUDGET_CHARS_3B="${TABLE_BUDGET_CHARS_3B:-8500}"; ENTITY_BUDGET_CHARS_3B="${ENTITY_BUDGET_CHARS_3B:-2200}"
  MAIN_BUDGET_CHARS_3C="${MAIN_BUDGET_CHARS_3C:-11000}"; TABLE_BUDGET_CHARS_3C="${TABLE_BUDGET_CHARS_3C:-9500}"; ENTITY_BUDGET_CHARS_3C="${ENTITY_BUDGET_CHARS_3C:-2400}"
  MAIN_BUDGET_CHARS_3D="${MAIN_BUDGET_CHARS_3D:-12000}"; TABLE_BUDGET_CHARS_3D="${TABLE_BUDGET_CHARS_3D:-11000}"; ENTITY_BUDGET_CHARS_3D="${ENTITY_BUDGET_CHARS_3D:-2600}"
  MAX_ITEMS_MAIN_3B="${MAX_ITEMS_MAIN_3B:-12}"; MAX_ITEMS_TABLE_3B="${MAX_ITEMS_TABLE_3B:-12}"; MAX_ITEMS_ENTITY_3B="${MAX_ITEMS_ENTITY_3B:-5}"
  MAX_ITEMS_MAIN_3C="${MAX_ITEMS_MAIN_3C:-12}"; MAX_ITEMS_TABLE_3C="${MAX_ITEMS_TABLE_3C:-12}"; MAX_ITEMS_ENTITY_3C="${MAX_ITEMS_ENTITY_3C:-5}"
  MAX_ITEMS_MAIN_3D="${MAX_ITEMS_MAIN_3D:-14}"; MAX_ITEMS_TABLE_3D="${MAX_ITEMS_TABLE_3D:-14}"; MAX_ITEMS_ENTITY_3D="${MAX_ITEMS_ENTITY_3D:-5}"
else
  WORKERS_3B="${WORKERS_3B:-4}"; WORKERS_3C="${WORKERS_3C:-4}"; WORKERS_3D="${WORKERS_3D:-4}"
  TOPK_MAIN_3B="${TOPK_MAIN_3B:-10}"; TOPK_TABLE_3B="${TOPK_TABLE_3B:-14}"; TOPK_ENTITY_3B="${TOPK_ENTITY_3B:-4}"
  TOPK_MAIN_3C="${TOPK_MAIN_3C:-10}"; TOPK_TABLE_3C="${TOPK_TABLE_3C:-16}"; TOPK_ENTITY_3C="${TOPK_ENTITY_3C:-5}"
  TOPK_MAIN_3D="${TOPK_MAIN_3D:-12}"; TOPK_TABLE_3D="${TOPK_TABLE_3D:-20}"; TOPK_ENTITY_3D="${TOPK_ENTITY_3D:-5}"
  MAIN_BUDGET_CHARS_3B="${MAIN_BUDGET_CHARS_3B:-7000}"; TABLE_BUDGET_CHARS_3B="${TABLE_BUDGET_CHARS_3B:-4200}"; ENTITY_BUDGET_CHARS_3B="${ENTITY_BUDGET_CHARS_3B:-1200}"
  MAIN_BUDGET_CHARS_3C="${MAIN_BUDGET_CHARS_3C:-7500}"; TABLE_BUDGET_CHARS_3C="${TABLE_BUDGET_CHARS_3C:-4600}"; ENTITY_BUDGET_CHARS_3C="${ENTITY_BUDGET_CHARS_3C:-1400}"
  MAIN_BUDGET_CHARS_3D="${MAIN_BUDGET_CHARS_3D:-8000}"; TABLE_BUDGET_CHARS_3D="${TABLE_BUDGET_CHARS_3D:-5500}"; ENTITY_BUDGET_CHARS_3D="${ENTITY_BUDGET_CHARS_3D:-1400}"
  MAX_ITEMS_MAIN_3B="${MAX_ITEMS_MAIN_3B:-8}"; MAX_ITEMS_TABLE_3B="${MAX_ITEMS_TABLE_3B:-6}"; MAX_ITEMS_ENTITY_3B="${MAX_ITEMS_ENTITY_3B:-3}"
  MAX_ITEMS_MAIN_3C="${MAX_ITEMS_MAIN_3C:-8}"; MAX_ITEMS_TABLE_3C="${MAX_ITEMS_TABLE_3C:-6}"; MAX_ITEMS_ENTITY_3C="${MAX_ITEMS_ENTITY_3C:-3}"
  MAX_ITEMS_MAIN_3D="${MAX_ITEMS_MAIN_3D:-9}"; MAX_ITEMS_TABLE_3D="${MAX_ITEMS_TABLE_3D:-8}"; MAX_ITEMS_ENTITY_3D="${MAX_ITEMS_ENTITY_3D:-3}"
fi
MAX_TEXT_CHARS="${MAX_TEXT_CHARS:-1000}"; MAX_TABLE_CHARS="${MAX_TABLE_CHARS:-2000}"; MAX_CAPTION_CHARS="${MAX_CAPTION_CHARS:-1600}"

mkdir -p "$OUTDIR" "$LOGDIR"
timestamp(){ date "+%Y-%m-%d %H:%M:%S"; }
log(){ echo "[$(timestamp)] $*"; }
die(){ log "[ERROR] $*"; exit 1; }
check_file(){ [[ -s "$1" ]] || die "Missing or empty required file: $1"; }
run_step(){ local name="$1"; shift; local logf="$1"; shift; log "[RUN] $name"; if ! "$@" > "$logf" 2>&1; then echo "---- LOG TAIL: $logf ----"; tail -n 160 "$logf" || true; echo "--------------------------"; die "$name failed"; fi; log "[DONE] $name"; }
count_jsonl(){ python - "$1" <<'PY'
import sys, os
fp=sys.argv[1]
print(sum(1 for line in open(fp, encoding='utf-8') if line.strip()) if os.path.exists(fp) else 0)
PY
}
count_extractable(){ python - "$1" <<'PY'
import sys, os, json
n=0
fp=sys.argv[1]
if os.path.exists(fp):
  for line in open(fp, encoding='utf-8'):
    try:
      o=json.loads(line)
      if int(float(o.get('should_extract',0)))==1: n+=1
    except Exception: pass
print(n)
PY
}
need_run(){ [[ "$FORCE" == "1" || ! -s "$1" ]]; }
FORCE_ARGS=(); [[ "$FORCE" == "1" ]] && FORCE_ARGS=(--force)
VERIFY_ARGS=(); [[ "$VERIFY_LLM" == "1" ]] && VERIFY_ARGS=(--verify-llm --verifier-api-bases "$VERIFIER_API_BASES" --verifier-model-id "$VERIFIER_MODEL_ID" --verify-max-tokens "$VERIFY_MAX_TOKENS")

TRIAGE_JSONL="$OUTDIR/hads_paper_triage.jsonl"
TRIAGE_CSV="$OUTDIR/hads_paper_triage.csv"
TRIAGE_SUMMARY="$OUTDIR/hads_paper_triage_summary.json"
HADS_SYSTEM_JSONL="$OUTDIR/hads_system_candidates.jsonl"
HADS_SITE_JSONL="$OUTDIR/hads_site_descriptor_candidates.jsonl"
HADS_ADS_JSONL="$OUTDIR/hads_adsorption_candidates.jsonl"
HADS_CANONICAL_JSONL="$OUTDIR/hads_canonical_records.jsonl"

check_file "$MANIFEST"
[[ -f "$PAPER_TAGS" ]] || log "[WARN] missing paper tags: $PAPER_TAGS; continuing with empty tags"
[[ -f "$QC_REPORT" ]] || log "[WARN] missing QC report: $QC_REPORT; continuing with empty QC"
for f in 3a_triage_papers.py 3b_extract_hads_systems.py 3c_extract_hads_site_descriptors.py 3d_extract_hads_adsorption.py 3e_link_hads_records.py; do [[ -f "$f" ]] || die "Missing script: $f"; done
if [[ "$REPAIR_DESCRIPTORS" == "1" ]]; then [[ -f "3f_repair_missing_descriptors.py" ]] || die "Missing script: 3f_repair_missing_descriptors.py"; fi

log "[CONFIG] API_BASES=$API_BASES"
log "[CONFIG] MODEL_ID=$MODEL_ID"
log "[CONFIG] HIGH_RECALL=$HIGH_RECALL VERIFY_LLM=$VERIFY_LLM REPAIR_DESCRIPTORS=$REPAIR_DESCRIPTORS REPAIR_LLM=$REPAIR_LLM WORKERS=($WORKERS_3B,$WORKERS_3C,$WORKERS_3D)"

if need_run "$TRIAGE_JSONL"; then
  run_step "Step3a triage" "$LOGDIR/step3a_triage.log" \
    python 3a_triage_papers.py --manifest "$MANIFEST" --parsed-root "$PARSED_ROOT" --paper-tags "$PAPER_TAGS" --qc-report "$QC_REPORT" --out-jsonl "$TRIAGE_JSONL" --out-csv "$TRIAGE_CSV" --out-summary "$TRIAGE_SUMMARY" --min-relevance "$MIN_RELEVANCE" --min-system-signal "$MIN_SYSTEM_SIGNAL" --min-metric-signal "$MIN_METRIC_SIGNAL"
else log "[SKIP] Step3a triage"; fi
check_file "$TRIAGE_JSONL"
N_EXTRACT="$(count_extractable "$TRIAGE_JSONL")"
log "[INFO] extractable papers: $N_EXTRACT"
if [[ "$N_EXTRACT" == "0" ]]; then
  : > "$HADS_SYSTEM_JSONL"; : > "$HADS_SITE_JSONL"; : > "$HADS_ADS_JSONL"; : > "$HADS_CANONICAL_JSONL"
  log "[WARN] no extractable papers; created empty outputs"
  exit 0
fi

if need_run "$HADS_SYSTEM_JSONL"; then
  run_step "Step3b systems" "$LOGDIR/step3b_systems.log" \
    python 3b_extract_hads_systems.py --triage "$TRIAGE_JSONL" --parsed-root "$PARSED_ROOT" --api-bases "$API_BASES" --model-id "$MODEL_ID" --embed-model "$EMBED_MODEL" --device "$DEVICE" --qdrant-prefix "$QDRANT_PREFIX" --collection-prefix "$COLLECTION_PREFIX" --shard-ids "$SHARD_IDS" --topk-main "$TOPK_MAIN_3B" --topk-table "$TOPK_TABLE_3B" --topk-entity "$TOPK_ENTITY_3B" --main-budget-chars "$MAIN_BUDGET_CHARS_3B" --table-budget-chars "$TABLE_BUDGET_CHARS_3B" --entity-budget-chars "$ENTITY_BUDGET_CHARS_3B" --max-items-main "$MAX_ITEMS_MAIN_3B" --max-items-table "$MAX_ITEMS_TABLE_3B" --max-items-entity "$MAX_ITEMS_ENTITY_3B" --max-text-chars "$MAX_TEXT_CHARS" --max-table-chars "$MAX_TABLE_CHARS" --max-caption-chars "$MAX_CAPTION_CHARS" --workers "$WORKERS_3B" --timeout "$TIMEOUT" --max-retries "$MAX_RETRIES" --max-tokens "$MAX_TOKENS_3B" --out "$HADS_SYSTEM_JSONL" --progress "$OUTDIR/progress_hads_step3b_systems.json" "${VERIFY_ARGS[@]}" "${FORCE_ARGS[@]}"
else log "[SKIP] Step3b systems"; fi
check_file "$HADS_SYSTEM_JSONL"

if need_run "$HADS_SITE_JSONL"; then
  run_step "Step3c sites" "$LOGDIR/step3c_sites.log" \
    python 3c_extract_hads_site_descriptors.py --triage "$TRIAGE_JSONL" --systems "$HADS_SYSTEM_JSONL" --parsed-root "$PARSED_ROOT" --api-bases "$API_BASES" --model-id "$MODEL_ID" --embed-model "$EMBED_MODEL" --device "$DEVICE" --qdrant-prefix "$QDRANT_PREFIX" --collection-prefix "$COLLECTION_PREFIX" --shard-ids "$SHARD_IDS" --topk-main "$TOPK_MAIN_3C" --topk-table "$TOPK_TABLE_3C" --topk-entity "$TOPK_ENTITY_3C" --main-budget-chars "$MAIN_BUDGET_CHARS_3C" --table-budget-chars "$TABLE_BUDGET_CHARS_3C" --entity-budget-chars "$ENTITY_BUDGET_CHARS_3C" --max-items-main "$MAX_ITEMS_MAIN_3C" --max-items-table "$MAX_ITEMS_TABLE_3C" --max-items-entity "$MAX_ITEMS_ENTITY_3C" --max-text-chars "$MAX_TEXT_CHARS" --max-table-chars "$MAX_TABLE_CHARS" --max-caption-chars "$MAX_CAPTION_CHARS" --workers "$WORKERS_3C" --timeout "$TIMEOUT" --max-retries "$MAX_RETRIES" --max-tokens "$MAX_TOKENS_3C" --out "$HADS_SITE_JSONL" --progress "$OUTDIR/progress_hads_step3c_sites.json" "${VERIFY_ARGS[@]}" "${FORCE_ARGS[@]}"
else log "[SKIP] Step3c sites"; fi
[[ -s "$HADS_SITE_JSONL" ]] || { log "[WARN] no site output; creating empty site file"; : > "$HADS_SITE_JSONL"; }

if need_run "$HADS_ADS_JSONL"; then
  run_step "Step3d adsorption/deprotonation" "$LOGDIR/step3d_adsorption.log" \
    python 3d_extract_hads_adsorption.py --triage "$TRIAGE_JSONL" --systems "$HADS_SYSTEM_JSONL" --sites "$HADS_SITE_JSONL" --parsed-root "$PARSED_ROOT" --api-bases "$API_BASES" --model-id "$MODEL_ID" --embed-model "$EMBED_MODEL" --device "$DEVICE" --qdrant-prefix "$QDRANT_PREFIX" --collection-prefix "$COLLECTION_PREFIX" --shard-ids "$SHARD_IDS" --topk-main "$TOPK_MAIN_3D" --topk-table "$TOPK_TABLE_3D" --topk-entity "$TOPK_ENTITY_3D" --main-budget-chars "$MAIN_BUDGET_CHARS_3D" --table-budget-chars "$TABLE_BUDGET_CHARS_3D" --entity-budget-chars "$ENTITY_BUDGET_CHARS_3D" --max-items-main "$MAX_ITEMS_MAIN_3D" --max-items-table "$MAX_ITEMS_TABLE_3D" --max-items-entity "$MAX_ITEMS_ENTITY_3D" --max-text-chars "$MAX_TEXT_CHARS" --max-table-chars "$MAX_TABLE_CHARS" --max-caption-chars "$MAX_CAPTION_CHARS" --workers "$WORKERS_3D" --timeout "$TIMEOUT" --max-retries "$MAX_RETRIES" --max-tokens "$MAX_TOKENS_3D" --out "$HADS_ADS_JSONL" --progress "$OUTDIR/progress_hads_step3d_adsorption.json" "${VERIFY_ARGS[@]}" "${FORCE_ARGS[@]}"
else log "[SKIP] Step3d adsorption/deprotonation"; fi
[[ -s "$HADS_ADS_JSONL" ]] || { log "[WARN] no adsorption output; creating empty adsorption file"; : > "$HADS_ADS_JSONL"; }

if need_run "$HADS_CANONICAL_JSONL"; then
  run_step "Step3e canonical linking" "$LOGDIR/step3e_link.log" \
    python 3e_link_hads_records.py --triage "$TRIAGE_JSONL" --systems "$HADS_SYSTEM_JSONL" --sites "$HADS_SITE_JSONL" --adsorption "$HADS_ADS_JSONL" --out "$HADS_CANONICAL_JSONL" --progress "$OUTDIR/progress_step3e_hads_link.json" "${FORCE_ARGS[@]}"
else log "[SKIP] Step3e canonical linking"; fi
check_file "$HADS_CANONICAL_JSONL"

if [[ "$REPAIR_DESCRIPTORS" == "1" ]]; then
  REPAIR_INPUT="$HADS_CANONICAL_JSONL.before_descriptor_repair"
  REPAIR_TMP="$HADS_CANONICAL_JSONL.repair_tmp"
  cp "$HADS_CANONICAL_JSONL" "$REPAIR_INPUT"
  REPAIR_ARGS=()
  if [[ "$REPAIR_LLM" == "1" ]]; then
    REPAIR_ARGS=(--llm-repair --parsed-root "$PARSED_ROOT" --api-bases "$API_BASES" --model-id "$MODEL_ID" --max-llm-records "$REPAIR_MAX_LLM_RECORDS")
  fi
  run_step "Step3f descriptor repair" "$LOGDIR/step3f_descriptor_repair.log"     python 3f_repair_missing_descriptors.py --canonical "$REPAIR_INPUT" --sites "$HADS_SITE_JSONL" --out "$REPAIR_TMP" --summary "$OUTDIR/hads_descriptor_repair_summary.json" --min-fuzzy-score "$REPAIR_MIN_FUZZY_SCORE" "${REPAIR_ARGS[@]}"
  mv "$REPAIR_TMP" "$HADS_CANONICAL_JSONL"
fi

log "[ALL DONE] HADS Step3 pipeline finished. canonical rows=$(count_jsonl "$HADS_CANONICAL_JSONL")"
