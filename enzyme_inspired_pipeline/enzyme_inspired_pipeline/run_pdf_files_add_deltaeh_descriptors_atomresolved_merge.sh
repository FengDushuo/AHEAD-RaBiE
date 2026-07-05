#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Detailed add-on pipeline for pdf-files-add
# Focus: high-quality joint extraction of ΔE_H* + DFT mechanism descriptors
# Strategy: parse added PDFs only -> detailed two-stage LLM extraction -> merge with old data
# =============================================================================

# ---------- Python environment ----------
PDFPARSE_ENV="${PDFPARSE_ENV:-/data/home/terminator/anaconda3/envs/pdfparse}"
PYTHON_BIN="${PYTHON_BIN:-$PDFPARSE_ENV/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: $PYTHON_BIN"
  echo "        Set PYTHON_BIN=/path/to/python or PDFPARSE_ENV=/path/to/env before running."
  exit 1
fi
export PATH="$PDFPARSE_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHON_BIN
PY_ENV_PREFIX="export PATH='$PDFPARSE_ENV/bin':\$PATH; export PYTHONNOUSERSITE=1; export PYTHON_BIN='$PYTHON_BIN';"

# ---------- SCNet / OpenAI-compatible model ----------
SCNET_BASE_URL="${SCNET_BASE_URL:-https://api.scnet.cn/api/llm/v1}"
API_BASES="${API_BASES:-$SCNET_BASE_URL}"
MODEL_ID="${MODEL_ID:-DeepSeek-V4-Flash}"
: "${SCNET_API_KEY:?Please export SCNET_API_KEY first, e.g. export SCNET_API_KEY='sk-xxxx'}"

# ---------- Input / output ----------
ADD_ROOT="${ADD_ROOT:-pdf-files-add}"
ADD_PARSED_ROOT="${ADD_PARSED_ROOT:-pdf-files-add-parsed}"
ADD_OUTDIR="${ADD_OUTDIR:-outputs_scnet_v4flash_add_deltaeh_atomresolved}"
ADD_LOGDIR="${ADD_LOGDIR:-logs_scnet_v4flash_add_deltaeh_detailed_qc}"

PREV_CANONICAL="${PREV_CANONICAL:-}"
if [[ -z "$PREV_CANONICAL" ]]; then
  if [[ -s "outputs_scnet_v4flash_accurate_hads/hads_canonical_records_repaired.jsonl" ]]; then
    PREV_CANONICAL="outputs_scnet_v4flash_accurate_hads/hads_canonical_records_repaired.jsonl"
  elif [[ -s "outputs_scnet_v4flash_accurate_hads/hads_canonical_records.jsonl" ]]; then
    PREV_CANONICAL="outputs_scnet_v4flash_accurate_hads/hads_canonical_records.jsonl"
  elif [[ -s "outputs/hads_canonical_records.jsonl" ]]; then
    PREV_CANONICAL="outputs/hads_canonical_records.jsonl"
  else
    echo "[ERROR] Could not find previous canonical records. Set PREV_CANONICAL=/path/to/hads_canonical_records*.jsonl"
    exit 1
  fi
fi

ADD_CANONICAL="$ADD_OUTDIR/hads_canonical_records_add_deltaeh_atomresolved.jsonl"
MERGED_CANONICAL="${MERGED_CANONICAL:-$ADD_OUTDIR/hads_canonical_records_merged_with_add_atomresolved.jsonl}"
MERGED_DB="${MERGED_DB:-outputs_scnet_v4flash_merged_add_deltaeh_atomresolved_db}"
MERGED_STEP5="${MERGED_STEP5:-outputs_scnet_v4flash_merged_add_deltaeh_atomresolved_step5_stats_ml}"
MERGED_FIGS="${MERGED_FIGS:-outputs_scnet_v4flash_merged_add_deltaeh_atomresolved_publication_figures}"

# ---------- Run controls ----------
# If pdf-files-add-parsed already exists, keep FORCE_PARSE=0 to avoid wasting time.
FORCE_PARSE="${FORCE_PARSE:-0}"
# Detailed extraction is normally run from scratch. Set FORCE_EXTRACT=0 to resume.
FORCE_EXTRACT="${FORCE_EXTRACT:-1}"
RUN_STEP5="${RUN_STEP5:-1}"
RUN_PLOT="${RUN_PLOT:-1}"

# ---------- Detailed extraction settings ----------
# This script uses up to two LLM calls per candidate paper. Start with workers=1 or 2.
WORKERS_ADD="${WORKERS_ADD:-1}"
TIMEOUT="${TIMEOUT:-420}"
MAX_RETRIES="${MAX_RETRIES:-5}"
TARGET_MAX_TOKENS="${TARGET_MAX_TOKENS:-3000}"
DESC_MAX_TOKENS="${DESC_MAX_TOKENS:-4500}"
ENERGY_ABS_MAX_EV="${ENERGY_ABS_MAX_EV:-5.0}"
KEEP_H2_ADSORPTION="${KEEP_H2_ADSORPTION:-0}"
MAX_TARGETS_PER_PAPER="${MAX_TARGETS_PER_PAPER:-24}"

TARGET_TOPK_TEXT="${TARGET_TOPK_TEXT:-16}"
TARGET_TOPK_TABLE="${TARGET_TOPK_TABLE:-30}"
TARGET_TOPK_CAPTION="${TARGET_TOPK_CAPTION:-8}"
DESC_TOPK_TEXT="${DESC_TOPK_TEXT:-24}"
DESC_TOPK_TABLE="${DESC_TOPK_TABLE:-32}"
DESC_TOPK_CAPTION="${DESC_TOPK_CAPTION:-10}"

TARGET_TEXT_BUDGET_CHARS="${TARGET_TEXT_BUDGET_CHARS:-14000}"
TARGET_TABLE_BUDGET_CHARS="${TARGET_TABLE_BUDGET_CHARS:-22000}"
TARGET_CAPTION_BUDGET_CHARS="${TARGET_CAPTION_BUDGET_CHARS:-6000}"
DESC_TEXT_BUDGET_CHARS="${DESC_TEXT_BUDGET_CHARS:-22000}"
DESC_TABLE_BUDGET_CHARS="${DESC_TABLE_BUDGET_CHARS:-26000}"
DESC_CAPTION_BUDGET_CHARS="${DESC_CAPTION_BUDGET_CHARS:-8000}"
MAX_QUOTE_CHARS="${MAX_QUOTE_CHARS:-2200}"

# Parser controls
GROBID_URL="${GROBID_URL:-http://127.0.0.1:8070}"
TABLE_ENGINES="${TABLE_ENGINES:-camelot}"
PARSE_WORKERS="${PARSE_WORKERS:-8}"

mkdir -p "$ADD_OUTDIR" "$ADD_LOGDIR" "$MERGED_DB" "$MERGED_STEP5" "$MERGED_FIGS"

timestamp(){ date "+%Y-%m-%d %H:%M:%S"; }
log(){ echo "[$(timestamp)] $*"; }
check_file(){ [[ -s "$1" ]] || { echo "[$(timestamp)] [ERROR] Missing or empty: $1"; exit 1; }; }
run_step(){ local name="$1"; shift; local logf="$1"; shift; log "[RUN] $name"; if ! "$@" > "$logf" 2>&1; then echo "---- LOG TAIL: $logf ----"; tail -n 160 "$logf" || true; echo "--------------------------"; exit 1; fi; log "[DONE] $name"; }
count_jsonl(){ "$PYTHON_BIN" - "$1" <<'PY'
import sys, os
fp=sys.argv[1]
print(sum(1 for line in open(fp, encoding='utf-8') if line.strip()) if os.path.exists(fp) else 0)
PY
}

dedup_parsed_manifest(){
  local manifest_fp="$ADD_PARSED_ROOT/manifest.csv"
  local dup_report="$ADD_OUTDIR/duplicate_paper_id_rows_in_add_manifest.csv"
  [[ -s "$manifest_fp" ]] || return 1
  "$PYTHON_BIN" - "$manifest_fp" "$dup_report" <<'PYDEDUP'
import sys
from pathlib import Path
import pandas as pd
manifest = Path(sys.argv[1]); dup_report = Path(sys.argv[2])
df = pd.read_csv(manifest)
if "paper_id" not in df.columns:
    print(f"[WARN] manifest has no paper_id column: {manifest}")
    sys.exit(0)
df["paper_id"] = df["paper_id"].astype(str).str.strip()
df = df[df["paper_id"].ne("")].copy()
if df["paper_id"].duplicated().any():
    dup = df[df["paper_id"].duplicated(keep=False)].sort_values("paper_id")
    dup_report.parent.mkdir(parents=True, exist_ok=True)
    dup.to_csv(dup_report, index=False)
    before = len(df)
    df = df.drop_duplicates(subset=["paper_id"], keep="last").copy()
    df.to_csv(manifest, index=False)
    print(f"[WARN] duplicate paper_id rows in manifest: {before-len(df)} removed; kept rows={len(df)}; report={dup_report}")
else:
    print(f"[INFO] manifest paper_id is unique; rows={len(df)}")
PYDEDUP
}

log "[CONFIG] ADD_ROOT=$ADD_ROOT"
log "[CONFIG] ADD_PARSED_ROOT=$ADD_PARSED_ROOT"
log "[CONFIG] PREV_CANONICAL=$PREV_CANONICAL"
log "[CONFIG] MODEL_ID=$MODEL_ID"
log "[CONFIG] PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import sys
print("[PYTHON] executable=", sys.executable)
import fitz
print("[PYTHON] fitz/PyMuPDF=OK")
PY
log "[CONFIG] detailed extraction v2_qc: workers=$WORKERS_ADD target_tokens=$TARGET_MAX_TOKENS desc_tokens=$DESC_MAX_TOKENS energy_abs_max=$ENERGY_ABS_MAX_EV keep_h2=$KEEP_H2_ADSORPTION"

[[ -d "$ADD_ROOT" ]] || { echo "[ERROR] pdf add directory not found: $ADD_ROOT"; exit 1; }
check_file "$PREV_CANONICAL"

# ---------- Step A: parse added PDFs only ----------
if [[ "$FORCE_PARSE" == "1" || ! -s "$ADD_PARSED_ROOT/manifest.csv" ]]; then
  log "[RUN] Parse pdf-files-add"
  if ! bash -lc "$PY_ENV_PREFIX ROOT='$ADD_ROOT' OUT='$ADD_PARSED_ROOT' GROBID_URL='$GROBID_URL' TABLE_ENGINES='$TABLE_ENGINES' WORKERS='$PARSE_WORKERS' FORCE='$FORCE_PARSE' bash 1_parse_echem_corpus.sh" > "$ADD_LOGDIR/stepA_parse_pdf_files_add.log" 2>&1; then
    echo "---- LOG TAIL: $ADD_LOGDIR/stepA_parse_pdf_files_add.log ----"
    tail -n 160 "$ADD_LOGDIR/stepA_parse_pdf_files_add.log" || true
    echo "--------------------------"
    if [[ -s "$ADD_PARSED_ROOT/manifest.csv" ]]; then
      log "[WARN] parser returned non-zero, but manifest.csv exists. Trying manifest de-duplication and continuing."
    else
      log "[ERROR] parser failed and manifest.csv was not created."
      exit 1
    fi
  else
    log "[DONE] Parse pdf-files-add"
  fi
else
  log "[SKIP] Parse pdf-files-add; existing parsed root found: $ADD_PARSED_ROOT"
fi
check_file "$ADD_PARSED_ROOT/manifest.csv"
dedup_parsed_manifest
check_file "$ADD_PARSED_ROOT/manifest.csv"

# ---------- Step B: detailed two-stage extraction ----------
EXTRACT_FORCE_ARGS=()
if [[ "$FORCE_EXTRACT" == "1" ]]; then EXTRACT_FORCE_ARGS=(--force); fi
if [[ "$KEEP_H2_ADSORPTION" == "1" ]]; then EXTRACT_FORCE_ARGS+=(--keep-h2-adsorption); fi
run_step "Detailed two-stage ΔE_H* + descriptor extraction" "$ADD_LOGDIR/stepB_detailed_extract_deltaeh_descriptors.log" \
  "$PYTHON_BIN" 3g_detailed_extract_deltaeh_descriptor_joint_v3_atomresolved.py \
    --manifest "$ADD_PARSED_ROOT/manifest.csv" \
    --parsed-root "$ADD_PARSED_ROOT" \
    --api-bases "$API_BASES" \
    --model-id "$MODEL_ID" \
    --workers "$WORKERS_ADD" \
    --timeout "$TIMEOUT" \
    --max-retries "$MAX_RETRIES" \
    --target-max-tokens "$TARGET_MAX_TOKENS" \
    --desc-max-tokens "$DESC_MAX_TOKENS" \
    --target-topk-text "$TARGET_TOPK_TEXT" \
    --target-topk-table "$TARGET_TOPK_TABLE" \
    --target-topk-caption "$TARGET_TOPK_CAPTION" \
    --desc-topk-text "$DESC_TOPK_TEXT" \
    --desc-topk-table "$DESC_TOPK_TABLE" \
    --desc-topk-caption "$DESC_TOPK_CAPTION" \
    --target-text-budget-chars "$TARGET_TEXT_BUDGET_CHARS" \
    --target-table-budget-chars "$TARGET_TABLE_BUDGET_CHARS" \
    --target-caption-budget-chars "$TARGET_CAPTION_BUDGET_CHARS" \
    --desc-text-budget-chars "$DESC_TEXT_BUDGET_CHARS" \
    --desc-table-budget-chars "$DESC_TABLE_BUDGET_CHARS" \
    --desc-caption-budget-chars "$DESC_CAPTION_BUDGET_CHARS" \
    --max-quote-chars "$MAX_QUOTE_CHARS" \
    --max-targets-per-paper "$MAX_TARGETS_PER_PAPER" \
    --energy-abs-max-ev "$ENERGY_ABS_MAX_EV" \
    --out "$ADD_CANONICAL" \
    --progress "$ADD_OUTDIR/progress_3g_detailed_extract_deltaeh_descriptor_joint_v3_atomresolved.json" \
    "${EXTRACT_FORCE_ARGS[@]}"

touch "$ADD_CANONICAL"
log "[INFO] Add-on detailed canonical rows=$(count_jsonl "$ADD_CANONICAL")"

# ---------- Step C: merge with previous canonical records ----------
run_step "Merge previous + detailed added canonical records" "$ADD_LOGDIR/stepC_merge_canonical.log" \
  "$PYTHON_BIN" merge_hads_canonical_records.py \
    --old "$PREV_CANONICAL" \
    --new "$ADD_CANONICAL" \
    --out "$MERGED_CANONICAL" \
    --summary "$ADD_OUTDIR/merge_summary.json"
check_file "$MERGED_CANONICAL"
log "[INFO] Merged canonical rows=$(count_jsonl "$MERGED_CANONICAL")"

# ---------- Step D: rebuild normalized DB ----------
rm -rf "$MERGED_DB"
run_step "Build merged normalized database" "$ADD_LOGDIR/stepD_build_merged_db.log" \
  bash -lc "$PY_ENV_PREFIX INP='$MERGED_CANONICAL' OUTDIR='$MERGED_DB' LOGDIR='$ADD_LOGDIR/step4_merged_db' FORCE=1 python 4_build_hads_normalized_database_atomresolved.py --in '$MERGED_CANONICAL' --outdir '$MERGED_DB'"
check_file "$MERGED_DB/model_feature_table.csv"

# ---------- Step E: optional Step5 ----------
if [[ "$RUN_STEP5" == "1" ]]; then
  run_step "Step5 on merged DB" "$ADD_LOGDIR/stepE_step5_merged.log" \
    bash -lc "$PY_ENV_PREFIX DB_OUTDIR='$MERGED_DB' OUTROOT='$MERGED_STEP5' QC_FILTER=1 GROUP_COL=paper_id N_SPLITS=5 SEED=2026 bash 5_run_stats_ml_hads.sh"
else
  log "[SKIP] Step5"
fi

# ---------- Step F: optional figures ----------
if [[ "$RUN_PLOT" == "1" ]]; then
  if [[ -f "plot_hads_publication_figures_v2_refined_layoutfix15_deltaE_normalized.py" ]]; then
    run_step "Plot merged figures" "$ADD_LOGDIR/stepF_plot_merged.log" \
      "$PYTHON_BIN" plot_hads_publication_figures_v2_refined_layoutfix15_deltaE_normalized.py \
        --dbdir "$MERGED_DB" \
        --step5dir "$MERGED_STEP5" \
        --outdir "$MERGED_FIGS" \
        --dpi 900 \
        --font Arial \
        --formats png,pdf \
        --topk 12 \
        --bubble-max 260 \
        --deltae-detail \
        --deltae-target H_adsorption_energy_value_eV \
        --deltae-topn 12
  else
    log "[WARN] plotting script not found; skip figures"
  fi
else
  log "[SKIP] plotting"
fi

# ---------- Diagnostic ----------
"$PYTHON_BIN" - <<PY
import pandas as pd
fp = "$MERGED_DB/model_feature_table.csv"
df = pd.read_csv(fp)
target = "H_adsorption_energy_value_eV"
features = ["d_band_center","bader_charge","coordination_number","oxidation_state_value","vacancy_flag_yes","hydroxyl_flag_yes","bridge_oxygen_flag_yes","work_function","PZC","surface_facet"]
rows=[]
for f in features:
    if f not in df.columns:
        rows.append({"feature": f, "status": "missing_column"}); continue
    rows.append({
        "feature": f,
        "feature_nonmissing": int(df[f].notna().sum()),
        "pair_with_DeltaE_H": int((df[f].notna() & df[target].notna()).sum()) if target in df.columns else 0,
        "target_nonmissing": int(df[target].notna().sum()) if target in df.columns else 0,
    })
out = pd.DataFrame(rows)
out.to_csv("$ADD_OUTDIR/merged_DeltaEH_descriptor_joint_coverage_detailed_qc.csv", index=False)
print("\n[DONE] Detailed joint coverage diagnostic:")
print(out.to_string(index=False))
PY

log "[ALL DONE] Detailed add-on ΔE_H* descriptor pipeline finished"
log "[RESULT] detailed add canonical: $ADD_CANONICAL"
log "[RESULT] merged canonical: $MERGED_CANONICAL"
log "[RESULT] merged DB: $MERGED_DB"
log "[RESULT] coverage CSV: $ADD_OUTDIR/merged_DeltaEH_descriptor_joint_coverage_detailed_qc.csv"
log "[RESULT] figures: $MERGED_FIGS"
