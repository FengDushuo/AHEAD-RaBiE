#!/usr/bin/env bash
set -euo pipefail

# Re-link + descriptor-repair + rebuild database/Step5/figures without rerunning costly Step3b/3c/3d.
# Use this after an existing Step3 extraction has already produced:
#   hads_paper_triage.jsonl, hads_system_candidates.jsonl, hads_site_descriptor_candidates.jsonl, hads_adsorption_candidates.jsonl

STEP3_OUT="${STEP3_OUT:-outputs_scnet_v4flash_accurate_hads}"
DB_OUT="${DB_OUT:-outputs_scnet_v4flash_accurate_hads_db_repaired}"
STEP5_OUT="${STEP5_OUT:-outputs_scnet_v4flash_accurate_step5_stats_ml_repaired}"
FIG_OUT="${FIG_OUT:-outputs_scnet_v4flash_accurate_publication_figures_repaired}"
LOGDIR="${LOGDIR:-logs_hads_relink_repair}"
PARSED_ROOT="${PARSED_ROOT:-pdf-files-parsed}"

# Descriptor repair settings.
REPAIR_LLM="${REPAIR_LLM:-0}"              # set 1 to enable SCNet model repair for still-missing descriptors
REPAIR_MAX_LLM_RECORDS="${REPAIR_MAX_LLM_RECORDS:-0}"  # 0 = no cap
REPAIR_MIN_FUZZY_SCORE="${REPAIR_MIN_FUZZY_SCORE:-0.26}"
MODEL_ID="${MODEL_ID:-DeepSeek-V4-Flash}"
API_BASES="${API_BASES:-https://api.scnet.cn/api/llm/v1}"

mkdir -p "$STEP3_OUT" "$DB_OUT" "$STEP5_OUT" "$FIG_OUT" "$LOGDIR"

ts(){ date '+%Y-%m-%d %H:%M:%S'; }
log(){ echo "[$(ts)] $*"; }
run_step(){ local name="$1"; shift; local logf="$1"; shift; log "[RUN] $name"; if ! "$@" > "$logf" 2>&1; then echo "---- LOG TAIL: $logf ----"; tail -n 160 "$logf" || true; echo "--------------------------"; exit 1; fi; log "[DONE] $name"; }
check(){ [[ -s "$1" ]] || { log "[ERROR] missing or empty: $1"; exit 1; }; }

TRIAGE="$STEP3_OUT/hads_paper_triage.jsonl"
SYSTEMS="$STEP3_OUT/hads_system_candidates.jsonl"
SITES="$STEP3_OUT/hads_site_descriptor_candidates.jsonl"
ADS="$STEP3_OUT/hads_adsorption_candidates.jsonl"
CANON_RELINK="$STEP3_OUT/hads_canonical_records_relinked.jsonl"
CANON_REPAIRED="$STEP3_OUT/hads_canonical_records_repaired.jsonl"

check "$TRIAGE"; check "$SYSTEMS"; check "$SITES"; check "$ADS"

run_step "Step3e enhanced relinking" "$LOGDIR/step3e_relink.log" \
  python 3e_link_hads_records.py \
    --triage "$TRIAGE" \
    --systems "$SYSTEMS" \
    --sites "$SITES" \
    --adsorption "$ADS" \
    --out "$CANON_RELINK" \
    --progress "$STEP3_OUT/progress_step3e_hads_relink_enhanced.json" \
    --force

REPAIR_ARGS=()
if [[ "$REPAIR_LLM" == "1" ]]; then
  : "${SCNET_API_KEY:?Set SCNET_API_KEY before REPAIR_LLM=1}"
  REPAIR_ARGS=(--llm-repair --parsed-root "$PARSED_ROOT" --api-bases "$API_BASES" --model-id "$MODEL_ID" --max-llm-records "$REPAIR_MAX_LLM_RECORDS")
fi

run_step "Step3f missing descriptor repair" "$LOGDIR/step3f_repair.log" \
  python 3f_repair_missing_descriptors.py \
    --canonical "$CANON_RELINK" \
    --sites "$SITES" \
    --out "$CANON_REPAIRED" \
    --summary "$STEP3_OUT/hads_descriptor_repair_summary.json" \
    --min-fuzzy-score "$REPAIR_MIN_FUZZY_SCORE" \
    "${REPAIR_ARGS[@]}"

# Make repaired file the standard canonical file used by downstream scripts.
cp "$CANON_REPAIRED" "$STEP3_OUT/hads_canonical_records.jsonl"

run_step "Step4 rebuild normalized database" "$LOGDIR/step4_build_db.log" \
  bash -c "INP='$CANON_REPAIRED' OUTDIR='$DB_OUT' FORCE=1 bash 4_build_hads_normalized_database.sh"

run_step "Step5 stats/ML" "$LOGDIR/step5_stats_ml.log" \
  bash -c "DB_OUTDIR='$DB_OUT' OUTROOT='$STEP5_OUT' QC_FILTER=1 GROUP_COL=paper_id N_SPLITS=5 SEED=2026 bash 5_run_stats_ml_hads.sh"

run_step "Publication figures" "$LOGDIR/plot_publication_figures.log" \
  python plot_hads_publication_figures_v2_refined_layoutfix15_deltaE_normalized.py \
    --dbdir "$DB_OUT" \
    --step5dir "$STEP5_OUT" \
    --outdir "$FIG_OUT" \
    --dpi 900 \
    --font Arial \
    --formats png,pdf \
    --topk 15 \
    --bubble-max 260 \
    --deltae-detail \
    --deltae-target H_adsorption_energy_value_eV \
    --deltae-topn 15

log "[DONE] relink + repair + rebuild finished"
log "[DONE] repaired canonical: $CANON_REPAIRED"
log "[DONE] database: $DB_OUT"
log "[DONE] figures: $FIG_OUT"
