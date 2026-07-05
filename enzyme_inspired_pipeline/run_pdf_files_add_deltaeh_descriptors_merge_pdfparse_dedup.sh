#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Add-on pipeline for pdf-files-add
# Focus: ΔE_H* + DFT mechanism descriptors joint coverage
# Goal: process only new PDFs, keep token usage controlled, then merge with previous data.
# =============================================================================

# ---------- Python environment ----------
# Force all Python calls, including subprocesses invoked by legacy parsers, to use
# the pdfparse conda environment instead of the base environment.
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

# Make Python path resolution explicit for child bash -lc commands.
PY_ENV_PREFIX="export PATH='$PDFPARSE_ENV/bin':\$PATH; export PYTHONNOUSERSITE=1; export PYTHON_BIN='$PYTHON_BIN';"

# ---------- SCNet / OpenAI-compatible model ----------
SCNET_BASE_URL="${SCNET_BASE_URL:-https://api.scnet.cn/api/llm/v1}"
API_BASES="${API_BASES:-$SCNET_BASE_URL}"
MODEL_ID="${MODEL_ID:-DeepSeek-V4-Flash}"
: "${SCNET_API_KEY:?Please export SCNET_API_KEY first, e.g. export SCNET_API_KEY='sk-xxxx'}"

# ---------- Input / output ----------
ADD_ROOT="${ADD_ROOT:-pdf-files-add}"
ADD_PARSED_ROOT="${ADD_PARSED_ROOT:-pdf-files-add-parsed}"
ADD_OUTDIR="${ADD_OUTDIR:-outputs_scnet_v4flash_add_deltaeh}"
ADD_LOGDIR="${ADD_LOGDIR:-logs_scnet_v4flash_add_deltaeh}"

# Previous canonical data. The script tries common names if PREV_CANONICAL is not set.
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

ADD_CANONICAL="$ADD_OUTDIR/hads_canonical_records_add_deltaeh.jsonl"
MERGED_CANONICAL="${MERGED_CANONICAL:-$ADD_OUTDIR/hads_canonical_records_merged_with_add.jsonl}"
MERGED_DB="${MERGED_DB:-outputs_scnet_v4flash_merged_add_deltaeh_db}"
MERGED_STEP5="${MERGED_STEP5:-outputs_scnet_v4flash_merged_add_deltaeh_step5_stats_ml}"
MERGED_FIGS="${MERGED_FIGS:-outputs_scnet_v4flash_merged_add_deltaeh_publication_figures}"

# ---------- Run controls ----------
FORCE_PARSE="${FORCE_PARSE:-0}"
FORCE_EXTRACT="${FORCE_EXTRACT:-1}"
RUN_STEP5="${RUN_STEP5:-1}"
RUN_PLOT="${RUN_PLOT:-1}"

# Token-controlled evidence settings: one LLM call per paper, table-first.
WORKERS_ADD="${WORKERS_ADD:-1}"
TIMEOUT="${TIMEOUT:-360}"
MAX_RETRIES="${MAX_RETRIES:-5}"
MAX_TOKENS_ADD="${MAX_TOKENS_ADD:-3500}"
TOPK_TEXT_ADD="${TOPK_TEXT_ADD:-10}"
TOPK_TABLE_ADD="${TOPK_TABLE_ADD:-18}"
TOPK_CAPTION_ADD="${TOPK_CAPTION_ADD:-6}"
TEXT_BUDGET_CHARS_ADD="${TEXT_BUDGET_CHARS_ADD:-9000}"
TABLE_BUDGET_CHARS_ADD="${TABLE_BUDGET_CHARS_ADD:-12000}"
CAPTION_BUDGET_CHARS_ADD="${CAPTION_BUDGET_CHARS_ADD:-4500}"
MAX_QUOTE_CHARS_ADD="${MAX_QUOTE_CHARS_ADD:-1800}"

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
manifest = Path(sys.argv[1])
dup_report = Path(sys.argv[2])
df = pd.read_csv(manifest)
if "paper_id" not in df.columns:
    print(f"[WARN] manifest has no paper_id column: {manifest}")
    sys.exit(0)
# Normalize paper_id to avoid hidden whitespace duplicates.
df["paper_id"] = df["paper_id"].astype(str).str.strip()
df = df[df["paper_id"].ne("")].copy()
if df["paper_id"].duplicated().any():
    dup = df[df["paper_id"].duplicated(keep=False)].sort_values("paper_id")
    dup_report.parent.mkdir(parents=True, exist_ok=True)
    dup.to_csv(dup_report, index=False)
    before = len(df)
    # Keep the last row because legacy parser normally writes the latest completed parse last.
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
log "[CONFIG] PDFPARSE_ENV=$PDFPARSE_ENV"
log "[CONFIG] PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import sys
print("[PYTHON] executable=", sys.executable)
try:
    import fitz
    print("[PYTHON] fitz/PyMuPDF=OK")
except Exception as e:
    print("[PYTHON] fitz/PyMuPDF=FAILED:", repr(e))
    raise
PY
log "[CONFIG] controlled evidence: table_budget=$TABLE_BUDGET_CHARS_ADD text_budget=$TEXT_BUDGET_CHARS_ADD max_tokens=$MAX_TOKENS_ADD workers=$WORKERS_ADD"

[[ -d "$ADD_ROOT" ]] || { echo "[ERROR] pdf add directory not found: $ADD_ROOT"; exit 1; }
check_file "$PREV_CANONICAL"

# ---------- Step A: parse added PDFs only ----------
# The legacy parser can finish PDF parsing but then fail if manifest.csv contains duplicate
# paper_id values. We keep the completed parse output, de-duplicate manifest.csv, and
# continue because the focused add-on extractor only needs manifest.csv + parsed chunks.
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

# ---------- Step B: focused extraction from added PDFs ----------
EXTRACT_FORCE_ARGS=()
if [[ "$FORCE_EXTRACT" == "1" ]]; then EXTRACT_FORCE_ARGS=(--force); fi
run_step "Focused ΔE_H* + descriptor extraction" "$ADD_LOGDIR/stepB_extract_deltaeh_descriptors.log" \
  "$PYTHON_BIN" 3g_extract_deltaeh_descriptor_joint.py \
    --manifest "$ADD_PARSED_ROOT/manifest.csv" \
    --parsed-root "$ADD_PARSED_ROOT" \
    --api-bases "$API_BASES" \
    --model-id "$MODEL_ID" \
    --workers "$WORKERS_ADD" \
    --timeout "$TIMEOUT" \
    --max-retries "$MAX_RETRIES" \
    --max-tokens "$MAX_TOKENS_ADD" \
    --topk-text "$TOPK_TEXT_ADD" \
    --topk-table "$TOPK_TABLE_ADD" \
    --topk-caption "$TOPK_CAPTION_ADD" \
    --text-budget-chars "$TEXT_BUDGET_CHARS_ADD" \
    --table-budget-chars "$TABLE_BUDGET_CHARS_ADD" \
    --caption-budget-chars "$CAPTION_BUDGET_CHARS_ADD" \
    --max-quote-chars "$MAX_QUOTE_CHARS_ADD" \
    --out "$ADD_CANONICAL" \
    --progress "$ADD_OUTDIR/progress_3g_extract_deltaeh_descriptor_joint.json" \
    "${EXTRACT_FORCE_ARGS[@]}"

# Allow zero records but keep an empty file for downstream reporting.
touch "$ADD_CANONICAL"
log "[INFO] Add-on canonical rows=$(count_jsonl "$ADD_CANONICAL")"

# ---------- Step C: merge with previous canonical records ----------
run_step "Merge previous + added canonical records" "$ADD_LOGDIR/stepC_merge_canonical.log" \
  "$PYTHON_BIN" merge_hads_canonical_records.py \
    --old "$PREV_CANONICAL" \
    --new "$ADD_CANONICAL" \
    --out "$MERGED_CANONICAL" \
    --summary "$ADD_OUTDIR/merge_summary.json"
check_file "$MERGED_CANONICAL"
log "[INFO] Merged canonical rows=$(count_jsonl "$MERGED_CANONICAL")"

# ---------- Step D: rebuild normalized DB from merged canonical records ----------
run_step "Build merged normalized database" "$ADD_LOGDIR/stepD_build_merged_db.log" \
  bash -lc "$PY_ENV_PREFIX INP='$MERGED_CANONICAL' OUTDIR='$MERGED_DB' LOGDIR='$ADD_LOGDIR/step4_merged_db' FORCE=1 bash 4_build_hads_normalized_database.sh"
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

# ---------- Diagnostic: joint coverage after merge ----------
"$PYTHON_BIN" - <<PY
import pandas as pd, json
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
out.to_csv("$ADD_OUTDIR/merged_DeltaEH_descriptor_joint_coverage.csv", index=False)
print("\n[DONE] Joint coverage diagnostic:")
print(out.to_string(index=False))
PY

log "[ALL DONE] Add-on ΔE_H* descriptor pipeline finished"
log "[RESULT] add canonical: $ADD_CANONICAL"
log "[RESULT] merged canonical: $MERGED_CANONICAL"
log "[RESULT] merged DB: $MERGED_DB"
log "[RESULT] coverage CSV: $ADD_OUTDIR/merged_DeltaEH_descriptor_joint_coverage.csv"
log "[RESULT] figures: $MERGED_FIGS"
