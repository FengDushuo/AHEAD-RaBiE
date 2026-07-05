#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Server full pipeline for AddH/AddH-2 -> AddH-out prediction
#
# Stages:
#   0) sanity checks
#   1) build master tables from addH/addH-2/addH-out if needed
#   2) extract FAIR-Chem graph embeddings if needed
#   3) build/call/train LLM + element-knowledge route
#   4) run graph-embedding classical ensemble grid
#   5) run multimodal grid
#   6) build strict-blind graph/multiview strategy ensemble
#   7) fuse LLM/element and strict-blind predictions
#
# Strict-blind rule:
#   addH-out labels are never used for training, selection, or blending.
#   They are used only when RUN_AUDIT=1, and then only for post-hoc audit files.
# ============================================================

ROOT="${ROOT:-$(pwd)}"
cd "$ROOT"

PY_MM="${PY_MM:-python}"
PY_FAIRCHEM="${PY_FAIRCHEM:-$PY_MM}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs_addh_server/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
SRC_OUT="${SRC_OUT:-$RUN_ROOT/outputs_addh_full_mm_envsplit}"

ADDH_DIR="${ADDH_DIR:-addH}"
ADDH2_DIR="${ADDH2_DIR:-addH-2}"
ADDHOUT_DIR="${ADDHOUT_DIR:-addH-out}"
ADDHOUT_EXCEL="${ADDHOUT_EXCEL:-addH-out/氢吸附能.xlsx}"
ADDH2_BASE_MILLER_MAP="${ADDH2_BASE_MILLER_MAP:-2542=100,2858=111,643=111}"
ADDHOUT_MILLER_MAP="${ADDHOUT_MILLER_MAP:-CeO2=111,ZnO=100}"
EH_REF="${EH_REF:--0.0565}"
TARGET_ABS_MAX="${TARGET_ABS_MAX:-10}"

FAIRCHEM_MODEL_DIR="${FAIRCHEM_MODEL_DIR:-/data/home/terminator/RL/multi-view/equiformer_v2_31m_allmd}"
FAIRCHEM_TRAINER="${FAIRCHEM_TRAINER:-equiformerv2_forces}"
DEVICE="${DEVICE:-cuda}"
GPU_ID="${GPU_ID:-2}"
MISSING_BARE="${MISSING_BARE:-zeros}"
DUAL_MIN_FRAC="${DUAL_MIN_FRAC:-0.80}"

RUN_DATA_PREP="${RUN_DATA_PREP:-auto}"       # auto | 1 | 0
RUN_EMBEDDINGS="${RUN_EMBEDDINGS:-auto}"     # auto | 1 | 0
RUN_LLM_ROUTE="${RUN_LLM_ROUTE:-1}"
RUN_SCNET_LLM="${RUN_SCNET_LLM:-0}"          # requires SCNET_API_KEY
RUN_GRAPH_GRID="${RUN_GRAPH_GRID:-1}"
RUN_MULTIVIEW_GRID="${RUN_MULTIVIEW_GRID:-1}"
RUN_STRICT_BLIND="${RUN_STRICT_BLIND:-1}"
RUN_SERVER_FINAL_BLEND="${RUN_SERVER_FINAL_BLEND:-1}"
RUN_AUDIT="${RUN_AUDIT:-0}"                  # post-hoc only

SCNET_MODEL="${SCNET_MODEL:-DeepSeek-V4-Pro}"
SCNET_SLEEP="${SCNET_SLEEP:-0.8}"
SCNET_MAX_RETRIES="${SCNET_MAX_RETRIES:-3}"
LLM_FEATURE_DIR="${LLM_FEATURE_DIR:-$RUN_ROOT/outputs_addh_llm_element_priors}"
LLM_PRED_DIR="${LLM_PRED_DIR:-$RUN_ROOT/outputs_addh_llm_element_knowledge_blend_scnet_deepseek_v4_pro}"
LLM_PRIOR_JSONL="${LLM_PRIOR_JSONL:-$LLM_FEATURE_DIR/llm_prior_scnet_deepseek_v4_pro_addhout.jsonl}"
LLM_SCAN_PRED_ROOT="${LLM_SCAN_PRED_ROOT:-logs}"

SEEDS="${SEEDS:-42,52,62,72,82,92,102,112,122,132}"
N_SPLITS="${N_SPLITS:-4}"
VAL_FRAC="${VAL_FRAC:-0.25}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MULTIVIEW_SCOPE="${MULTIVIEW_SCOPE:-targeted}" # targeted | all | off
TOP_K="${TOP_K:-16}"
MIN_COVERAGE="${MIN_COVERAGE:-0.90}"
WEIGHT_MODE="${WEIGHT_MODE:-soft_inverse_rmse}"

GRAPH_GRID_ROOT="${GRAPH_GRID_ROOT:-$RUN_ROOT/outputs_addh_graph_ensemble_refine_v2}"
BASE_GRAPH_GRID_ROOT="${BASE_GRAPH_GRID_ROOT:-$ROOT/outputs_addh_graph_ensemble}"
MULTIVIEW_GRID_ROOT="${MULTIVIEW_GRID_ROOT:-$RUN_ROOT/outputs_addh_modelgrid_v2_full}"
BASE_MULTIVIEW_GRID_ROOT="${BASE_MULTIVIEW_GRID_ROOT:-$ROOT/outputs_addh_modelgrid_v2}"
STRICT_ROOT="${STRICT_ROOT:-$RUN_ROOT/outputs_addh_strict_blind}"
STRICT_FINAL_DIR="${STRICT_FINAL_DIR:-$RUN_ROOT/outputs_addh_strict_blind_final}"
SERVER_FINAL_DIR="${SERVER_FINAL_DIR:-$RUN_ROOT/outputs_addh_server_final_blend}"

TRAIN_MASTER_ORIG="$SRC_OUT/addH_master_orig.csv"
TRAIN_MASTER_ML2="$SRC_OUT/addH_master_ml2.csv"
TRAIN_MASTER_MERGED="$SRC_OUT/addH_master_merged_robust.csv"
TRAIN_MASTER="$SRC_OUT/addH_master_target_weighted_mild.csv"
ADDHOUT_MASTER="$SRC_OUT/addH_out_master_normalized.csv"

ADDH_EQ_EMB="$SRC_OUT/addH_eq_emb.pkl"
ADDH_BARE_EQ_EMB="$SRC_OUT/addH_bare_eq_emb.pkl"
ADDH_DUAL_EMB="$SRC_OUT/addH_dual_eq_emb.pkl"
ADDHOUT_EQ_EMB="$SRC_OUT/addH_out_eq_emb.pkl"
ADDHOUT_BARE_EQ_EMB="$SRC_OUT/addH_out_bare_eq_emb.pkl"
ADDHOUT_DUAL_EMB="$SRC_OUT/addH_out_dual_eq_emb.pkl"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_SILENT="${WANDB_SILENT:-true}"

mkdir -p "$RUN_ROOT" "$LOG_DIR" "$SRC_OUT"

run_logged() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  echo "[CMD] $*" > "${log_file}.cmd"
  echo "[RUN] $*"
  "$@" > "$log_file" 2>&1
}

stage_wants_file() {
  local mode="$1"
  local marker="$2"
  if [[ "$FORCE_RERUN" == "1" ]]; then
    [[ "$mode" != "0" ]]
    return
  fi
  case "$mode" in
    1) return 0 ;;
    0) return 1 ;;
    auto) [[ ! -s "$marker" ]] ;;
    *) echo "[ERROR] invalid stage mode: $mode" >&2; exit 2 ;;
  esac
}

require_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] missing required file: $f" >&2
    exit 2
  fi
}

require_script() {
  local f="$1"
  require_file "$f"
}

print_config() {
  cat <<INFO
============================================================
[CONFIG] ROOT=$ROOT
[CONFIG] RUN_ROOT=$RUN_ROOT
[CONFIG] PY_MM=$PY_MM
[CONFIG] PY_FAIRCHEM=$PY_FAIRCHEM
[CONFIG] DEVICE=$DEVICE GPU_ID=$GPU_ID
[CONFIG] SRC_OUT=$SRC_OUT
[CONFIG] ADDH_DIR=$ADDH_DIR
[CONFIG] ADDH2_DIR=$ADDH2_DIR
[CONFIG] ADDHOUT_DIR=$ADDHOUT_DIR
[CONFIG] ADDHOUT_EXCEL=$ADDHOUT_EXCEL
[CONFIG] ADDH2_BASE_MILLER_MAP=$ADDH2_BASE_MILLER_MAP
[CONFIG] ADDHOUT_MILLER_MAP=$ADDHOUT_MILLER_MAP
[CONFIG] RUN_DATA_PREP=$RUN_DATA_PREP
[CONFIG] RUN_EMBEDDINGS=$RUN_EMBEDDINGS
[CONFIG] RUN_LLM_ROUTE=$RUN_LLM_ROUTE RUN_SCNET_LLM=$RUN_SCNET_LLM
[CONFIG] RUN_GRAPH_GRID=$RUN_GRAPH_GRID
[CONFIG] RUN_MULTIVIEW_GRID=$RUN_MULTIVIEW_GRID MULTIVIEW_SCOPE=$MULTIVIEW_SCOPE
[CONFIG] RUN_STRICT_BLIND=$RUN_STRICT_BLIND RUN_SERVER_FINAL_BLEND=$RUN_SERVER_FINAL_BLEND RUN_AUDIT=$RUN_AUDIT
[CONFIG] SEEDS=$SEEDS
============================================================
INFO
}

print_config

echo "[STAGE 0] Sanity checks"
for f in \
  01_build_addh_master_with_outlier_drop.py \
  build_addh_master_from_ml2_layout.py \
  merge_addh_master_tables_robust.py \
  02_build_addhout_master_normalized.py \
  make_target_domain_weighted_train_table_mild.py \
  03_extract_eq_emb_fairchem.py \
  06_build_dual_graph_embeddings.py \
  19_build_llm_element_prior_features.py \
  20_train_llm_element_knowledge_blend.py \
  21_call_scnet_deepseek_priors.py \
  22_fuse_llm_strict_blind_server.py \
  run_graph_embedding_refine_addh_full_once.sh \
  run_addh_model_experiment_grid_v2_full.sh \
  run_strict_blind_addhout_prediction_diverse.sh \
  17_build_strict_blind_strategy_ensemble.py; do
  require_script "$f"
done

run_logged "$LOG_DIR/stage0_py_compile.log" \
  "$PY_MM" -m py_compile \
    19_build_llm_element_prior_features.py \
    20_train_llm_element_knowledge_blend.py \
    21_call_scnet_deepseek_priors.py \
    22_fuse_llm_strict_blind_server.py \
    13_train_graph_embedding_ensemble_v2.py \
    14_summarize_graph_ensemble_addhout_v2.py \
    16_make_strict_blind_addhout_ensemble_v2.py \
    17_build_strict_blind_strategy_ensemble.py

DATA_PREP_NEEDED=0
case "$RUN_DATA_PREP" in
  1) DATA_PREP_NEEDED=1 ;;
  0) DATA_PREP_NEEDED=0 ;;
  auto)
    if [[ ! -s "$TRAIN_MASTER" || ! -s "$ADDHOUT_MASTER" ]]; then
      DATA_PREP_NEEDED=1
    fi
    ;;
  *) echo "[ERROR] invalid RUN_DATA_PREP=$RUN_DATA_PREP" >&2; exit 2 ;;
esac

if [[ "$DATA_PREP_NEEDED" == "1" ]]; then
  echo "[STAGE 1] Build master tables"
  run_logged "$LOG_DIR/stage1a_addh_master_orig.log" \
    "$PY_MM" 01_build_addh_master_with_outlier_drop.py \
      --input-dir "$ADDH_DIR" \
      --energy-bare "$ADDH_DIR/energy.dat" \
      --energy-addh "$ADDH_DIR/energy-addH.dat" \
      --output-csv "$TRAIN_MASTER_ORIG" \
      --eh-ref "$EH_REF" \
      --outlier-method none

  run_logged "$LOG_DIR/stage1b_addh2_master_ml2.log" \
    "$PY_MM" build_addh_master_from_ml2_layout.py \
      --input-root "$ADDH2_DIR" \
      --output-csv "$TRAIN_MASTER_ML2" \
      --eh-ref "$EH_REF" \
      --base-miller-map "$ADDH2_BASE_MILLER_MAP" \
      --outlier-method none

  run_logged "$LOG_DIR/stage1c_merge_train_masters.log" \
    "$PY_MM" merge_addh_master_tables_robust.py \
      --old-csv "$TRAIN_MASTER_ORIG" \
      --new-csv "$TRAIN_MASTER_ML2" \
      --output-csv "$TRAIN_MASTER_MERGED" \
      --merged-raw-csv "$SRC_OUT/addH_master_merged_raw.csv" \
      --outlier-method iqr \
      --outlier-action drop \
      --outlier-report-csv "$SRC_OUT/addH_master_merged_outlier_report.csv" \
      --outlier-summary-json "$SRC_OUT/addH_master_merged_outlier_summary.json"

  run_logged "$LOG_DIR/stage1d_addhout_master.log" \
    "$PY_MM" 02_build_addhout_master_normalized.py \
      --input-dir "$ADDHOUT_DIR" \
      --excel-path "$ADDHOUT_EXCEL" \
      --output-csv "$ADDHOUT_MASTER" \
      --eh-ref "$EH_REF" \
      --miller-map "$ADDHOUT_MILLER_MAP" \
      --write-bare-from-addh \
      --bare-output-dir "$SRC_OUT/addhout_bare_from_addH"

  run_logged "$LOG_DIR/stage1e_target_domain_weights.log" \
    "$PY_MM" make_target_domain_weighted_train_table_mild.py \
      --train-csv "$TRAIN_MASTER_MERGED" \
      --target-csv "$ADDHOUT_MASTER" \
      --output-csv "$TRAIN_MASTER" \
      --debug-csv "$SRC_OUT/target_domain_weight_debug_mild.csv" \
      --profile-json "$SRC_OUT/target_domain_profile_mild.json"
else
  echo "[SKIP] Stage 1 master tables already available: $TRAIN_MASTER and $ADDHOUT_MASTER"
fi

if stage_wants_file "$RUN_EMBEDDINGS" "$ADDH_DUAL_EMB" || stage_wants_file "$RUN_EMBEDDINGS" "$ADDHOUT_DUAL_EMB"; then
  echo "[STAGE 2] Extract FAIR-Chem graph embeddings"
  if [[ -z "$FAIRCHEM_MODEL_DIR" ]]; then
    echo "[ERROR] FAIRCHEM_MODEL_DIR is required when embeddings need to be generated." >&2
    echo "[ERROR] Set FAIRCHEM_MODEL_DIR=/path/to/model_dir containing checkpoint.pt and config.yml" >&2
    exit 2
  fi
  require_file "$FAIRCHEM_MODEL_DIR/checkpoint.pt"
  require_file "$FAIRCHEM_MODEL_DIR/config.yml"

  if stage_wants_file "$RUN_EMBEDDINGS" "$ADDH_EQ_EMB"; then
    run_logged "$LOG_DIR/stage2a_addh_eq_emb.log" \
      env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY_FAIRCHEM" 03_extract_eq_emb_fairchem.py \
        --master-csv "$TRAIN_MASTER" \
        --structure-col contcar_path \
        --model-dir "$FAIRCHEM_MODEL_DIR" \
        --trainer "$FAIRCHEM_TRAINER" \
        --device "$DEVICE" \
        --save-pkl "$ADDH_EQ_EMB" \
        --meta-csv "$SRC_OUT/addH_eq_emb.meta.csv"
  fi

  if stage_wants_file "$RUN_EMBEDDINGS" "$ADDH_BARE_EQ_EMB"; then
    run_logged "$LOG_DIR/stage2b_addh_bare_eq_emb.log" \
      env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY_FAIRCHEM" 03_extract_eq_emb_fairchem.py \
        --master-csv "$TRAIN_MASTER" \
        --structure-col bare_contcar_path \
        --model-dir "$FAIRCHEM_MODEL_DIR" \
        --trainer "$FAIRCHEM_TRAINER" \
        --device "$DEVICE" \
        --save-pkl "$ADDH_BARE_EQ_EMB" \
        --meta-csv "$SRC_OUT/addH_bare_eq_emb.meta.csv"
  fi

  if stage_wants_file "$RUN_EMBEDDINGS" "$ADDH_DUAL_EMB"; then
    run_logged "$LOG_DIR/stage2c_addh_dual_emb.log" \
      "$PY_MM" 06_build_dual_graph_embeddings.py \
        --master-csv "$TRAIN_MASTER" \
        --addh-emb-pkl "$ADDH_EQ_EMB" \
        --bare-emb-pkl "$ADDH_BARE_EQ_EMB" \
        --save-pkl "$ADDH_DUAL_EMB" \
        --meta-csv "$SRC_OUT/addH_dual_eq_emb.meta.csv" \
        --missing-bare "$MISSING_BARE" \
        --require-success-min-frac "$DUAL_MIN_FRAC"
  fi

  if stage_wants_file "$RUN_EMBEDDINGS" "$ADDHOUT_EQ_EMB"; then
    run_logged "$LOG_DIR/stage2d_addhout_eq_emb.log" \
      env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY_FAIRCHEM" 03_extract_eq_emb_fairchem.py \
        --master-csv "$ADDHOUT_MASTER" \
        --structure-col contcar_path \
        --model-dir "$FAIRCHEM_MODEL_DIR" \
        --trainer "$FAIRCHEM_TRAINER" \
        --device "$DEVICE" \
        --save-pkl "$ADDHOUT_EQ_EMB" \
        --meta-csv "$SRC_OUT/addH_out_eq_emb.meta.csv"
  fi

  if stage_wants_file "$RUN_EMBEDDINGS" "$ADDHOUT_BARE_EQ_EMB"; then
    run_logged "$LOG_DIR/stage2e_addhout_bare_eq_emb.log" \
      env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY_FAIRCHEM" 03_extract_eq_emb_fairchem.py \
        --master-csv "$ADDHOUT_MASTER" \
        --structure-col bare_contcar_path \
        --model-dir "$FAIRCHEM_MODEL_DIR" \
        --trainer "$FAIRCHEM_TRAINER" \
        --device "$DEVICE" \
        --save-pkl "$ADDHOUT_BARE_EQ_EMB" \
        --meta-csv "$SRC_OUT/addH_out_bare_eq_emb.meta.csv"
  fi

  if stage_wants_file "$RUN_EMBEDDINGS" "$ADDHOUT_DUAL_EMB"; then
    run_logged "$LOG_DIR/stage2f_addhout_dual_emb.log" \
      "$PY_MM" 06_build_dual_graph_embeddings.py \
        --master-csv "$ADDHOUT_MASTER" \
        --addh-emb-pkl "$ADDHOUT_EQ_EMB" \
        --bare-emb-pkl "$ADDHOUT_BARE_EQ_EMB" \
        --save-pkl "$ADDHOUT_DUAL_EMB" \
        --meta-csv "$SRC_OUT/addH_out_dual_eq_emb.meta.csv" \
        --missing-bare "$MISSING_BARE" \
        --require-success-min-frac "$DUAL_MIN_FRAC"
  fi
else
  echo "[SKIP] Stage 2 embeddings already available."
fi

if [[ "$RUN_LLM_ROUTE" == "1" ]]; then
  echo "[STAGE 3] LLM + element-knowledge strict-blind route"
  mkdir -p "$LLM_FEATURE_DIR" "$LLM_PRED_DIR"

  run_logged "$LOG_DIR/stage3a_llm_feature_prompts.log" \
    "$PY_MM" 19_build_llm_element_prior_features.py \
      --addh-dir "$ADDH_DIR" \
      --addh2-root "$ADDH2_DIR" \
      --addhout-dir "$ADDHOUT_DIR" \
      --addhout-excel "$ADDHOUT_EXCEL" \
      --out-dir "$LLM_FEATURE_DIR" \
      --eh-ref "$EH_REF" \
      --addh2-base-miller-map "$ADDH2_BASE_MILLER_MAP" \
      --addhout-miller-map "$ADDHOUT_MILLER_MAP" \
      --target-abs-max "$TARGET_ABS_MAX" \
      --write-audit-labels

  if [[ "$RUN_SCNET_LLM" == "1" ]]; then
    if [[ -z "${SCNET_API_KEY:-}" ]]; then
      echo "[ERROR] RUN_SCNET_LLM=1 but SCNET_API_KEY is not set." >&2
      exit 2
    fi
    run_logged "$LOG_DIR/stage3b_scnet_deepseek_priors.log" \
      "$PY_MM" 21_call_scnet_deepseek_priors.py \
        --input-jsonl "$LLM_FEATURE_DIR/llm_prior_prompts.jsonl" \
        --output-jsonl "$LLM_PRIOR_JSONL" \
        --id-regex "^(CeO2|ZnO)-" \
        --model "$SCNET_MODEL" \
        --prompt-style compact \
        --disable-thinking \
        --no-response-format \
        --sleep "$SCNET_SLEEP" \
        --max-retries "$SCNET_MAX_RETRIES"
  elif [[ -s "$LLM_PRIOR_JSONL" ]]; then
    echo "[INFO] Reusing existing LLM prior JSONL: $LLM_PRIOR_JSONL"
  else
    echo "[INFO] No SCNet call and no LLM prior JSONL. Deterministic element priors will be used."
  fi

  LLM_PRIOR_ARGS=()
  if [[ -s "$LLM_PRIOR_JSONL" ]]; then
    LLM_PRIOR_ARGS=(--llm-prior-jsonl "$LLM_PRIOR_JSONL")
  fi

  run_logged "$LOG_DIR/stage3c_llm_features_with_priors.log" \
    "$PY_MM" 19_build_llm_element_prior_features.py \
      --addh-dir "$ADDH_DIR" \
      --addh2-root "$ADDH2_DIR" \
      --addhout-dir "$ADDHOUT_DIR" \
      --addhout-excel "$ADDHOUT_EXCEL" \
      --out-dir "$LLM_FEATURE_DIR" \
      --eh-ref "$EH_REF" \
      --addh2-base-miller-map "$ADDH2_BASE_MILLER_MAP" \
      --addhout-miller-map "$ADDHOUT_MILLER_MAP" \
      --target-abs-max "$TARGET_ABS_MAX" \
      "${LLM_PRIOR_ARGS[@]}" \
      --write-audit-labels

  LLM_AUDIT_ARGS=()
  if [[ "$RUN_AUDIT" == "1" ]]; then
    LLM_AUDIT_ARGS=(--audit-labels-csv "$LLM_FEATURE_DIR/addhout_audit_labels.csv")
  fi

  run_logged "$LOG_DIR/stage3d_llm_train_blend.log" \
    "$PY_MM" 20_train_llm_element_knowledge_blend.py \
      --feature-dir "$LLM_FEATURE_DIR" \
      --out-dir "$LLM_PRED_DIR" \
      --target-abs-max "$TARGET_ABS_MAX" \
      --scan-pred-root "$LLM_SCAN_PRED_ROOT" \
      "${LLM_AUDIT_ARGS[@]}"
else
  echo "[SKIP] Stage 3 LLM route"
fi

if [[ "$RUN_GRAPH_GRID" == "1" ]]; then
  echo "[STAGE 4] Graph embedding ensemble grid"
  run_logged "$LOG_DIR/stage4_graph_grid.log" \
    env ROOT="$ROOT" PY_MM="$PY_MM" SRC_OUT="$SRC_OUT" \
      BASE_GRID_ROOT="$BASE_GRAPH_GRID_ROOT" GRID_ROOT="$GRAPH_GRID_ROOT" \
      ADDH_MASTER="$TRAIN_MASTER" ADDH_DUAL_EMB="$ADDH_DUAL_EMB" \
      ADDHOUT_MASTER="$ADDHOUT_MASTER" ADDHOUT_DUAL_EMB="$ADDHOUT_DUAL_EMB" \
      SEEDS="$SEEDS" N_SPLITS="$N_SPLITS" VAL_FRAC="$VAL_FRAC" \
      SKIP_COMPLETED="$SKIP_COMPLETED" FORCE_RERUN="$FORCE_RERUN" \
      OMP_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$MKL_NUM_THREADS" OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
      bash run_graph_embedding_refine_addh_full_once.sh
else
  echo "[SKIP] Stage 4 graph grid"
fi

if [[ "$RUN_MULTIVIEW_GRID" == "1" && "$MULTIVIEW_SCOPE" != "off" ]]; then
  echo "[STAGE 5] Multi-view model grid"
  if [[ "$MULTIVIEW_SCOPE" == "all" ]]; then
    MV_EXPS="all"
  else
    MV_EXPS="${MV_EXPS:-m13_t10_ssl_graphonly_u2_wp05,m16_t9_ssl_graphonly_u2_wp05,m17_t8_ssl_graphonly_u2_wp05,m18_t7_ssl_graphonly_u2_wp05,m19_t10_ssl_graphonly_u2_wp00,m20_t10_ssl_graphonly_u2_wp02,m21_t10_ssl_graphonly_u2_wp08,m22_t10_ssl_graphonly_u2_wp10,m23_t10_ssl_graphonly_u2_drop00,m24_t10_ssl_graphonly_u2_drop05,m25_t10_ssl_graphonly_u2_drop15,m26_t10_ssl_graphonly_u2_drop20,m27_t10_ssl_graphonly_u2_drop30,m28_t10_ssl_graphonly_u2_noise002,m29_t10_ssl_graphonly_u2_noise005,m30_t10_ssl_graphonly_u2_noise010,m31_t10_ssl_graphonly_depth1,m32_t10_ssl_graphonly_depth3,m34_t10_ssl_graphonly_proj128,m35_t10_ssl_graphonly_proj512,m42_t10_nossl_graphonly_u2_wp05,m43_t10_ssl_graphonly_valweighted,m01_t10_ssl_concat_u2_wp05,m02_t10_ssl_residual_u2_wp05,m44_t9_ssl_gated_u2_wp05,m45_t8_ssl_gated_u2_wp05,m46_t10_ssl_concat_proj128,m47_t10_ssl_concat_proj512,m50_t10_ssl_concat_valweighted,m51_t10_ssl_residual_drop20}"
  fi
  run_logged "$LOG_DIR/stage5_multiview_grid.log" \
    env ROOT="$ROOT" PY_MM="$PY_MM" GPU_ID="$GPU_ID" DEVICE="$DEVICE" SRC_OUT="$SRC_OUT" \
      GRID_ROOT="$MULTIVIEW_GRID_ROOT" REPO_ROOT="$ROOT" \
      ADDH_MASTER="$TRAIN_MASTER" ADDH_DUAL_EMB="$ADDH_DUAL_EMB" \
      ADDHOUT_MASTER="$ADDHOUT_MASTER" ADDHOUT_DUAL_EMB="$ADDHOUT_DUAL_EMB" \
      SEEDS="$SEEDS" EXPS_TO_RUN="$MV_EXPS" RUN_FINAL_ALL=0 USE_REPO_LOCK=1 \
      SKIP_COMPLETED="$SKIP_COMPLETED" FORCE_RERUN="$FORCE_RERUN" \
      CUDA_VISIBLE_DEVICES="$GPU_ID" \
      bash run_addh_model_experiment_grid_v2_full.sh
else
  echo "[SKIP] Stage 5 multi-view grid"
fi

if [[ "$RUN_STRICT_BLIND" == "1" ]]; then
  echo "[STAGE 6] Strict-blind strategy ensemble"
  STRICT_BLIND_ROOTS="$GRAPH_GRID_ROOT $BASE_GRAPH_GRID_ROOT $MULTIVIEW_GRID_ROOT $BASE_MULTIVIEW_GRID_ROOT"
  mkdir -p "$STRICT_ROOT"
  BALANCED="$STRICT_ROOT/diverse_balanced"
  CONSERVATIVE="$STRICT_ROOT/diverse_conservative"
  STABILITY="$STRICT_ROOT/diverse_stability"
  OOF="$STRICT_ROOT/diverse_oof"

  run_logged "$LOG_DIR/stage6a_strict_balanced.log" \
    env ROOT="$ROOT" PY_MM="$PY_MM" OUT_DIR="$BALANCED" \
      STRICT_BLIND_ROOTS="$STRICT_BLIND_ROOTS" ADDHOUT_MASTER_CSV="$ADDHOUT_MASTER" TRAIN_MASTER_CSV="$TRAIN_MASTER" \
      TOP_K="$TOP_K" MIN_COVERAGE="$MIN_COVERAGE" SCORE_MODE=conservative WEIGHT_MODE="$WEIGHT_MODE" FAMILY_DIVERSE=1 \
      AUDIT_WITH_LABELS=0 STRICT_OUTPUT_NO_LABELS=1 \
      FEATURE_MODE_CAP_MAP="full=3,bare_delta=2,addh_delta=2,delta=1,addh=3,addh_bare=3,bare=1,graph_only=2,concat_interact=2,gated_sum=1,residual_graph=1,text_only=0" \
      MODEL_FAMILY_CAP_MAP="graph_ensemble=12,multiview=4,unknown=2" \
      REQUIRE_FEATURE_MODES="addh_bare=1,addh=1,full=1,graph_only=1" \
      bash run_strict_blind_addhout_prediction_diverse.sh

  run_logged "$LOG_DIR/stage6b_strict_conservative.log" \
    env ROOT="$ROOT" PY_MM="$PY_MM" OUT_DIR="$CONSERVATIVE" \
      STRICT_BLIND_ROOTS="$STRICT_BLIND_ROOTS" ADDHOUT_MASTER_CSV="$ADDHOUT_MASTER" TRAIN_MASTER_CSV="$TRAIN_MASTER" \
      TOP_K="$TOP_K" MIN_COVERAGE="$MIN_COVERAGE" SCORE_MODE=conservative WEIGHT_MODE="$WEIGHT_MODE" FAMILY_DIVERSE=1 \
      AUDIT_WITH_LABELS=0 STRICT_OUTPUT_NO_LABELS=1 \
      FEATURE_MODE_CAP_MAP="full=2,bare_delta=1,addh_delta=1,delta=1,addh=4,addh_bare=4,bare=1,graph_only=3,concat_interact=1,gated_sum=1,residual_graph=1,text_only=0" \
      MODEL_FAMILY_CAP_MAP="graph_ensemble=12,multiview=4,unknown=2" \
      REQUIRE_FEATURE_MODES="addh_bare=2,addh=2,full=1,graph_only=1" \
      bash run_strict_blind_addhout_prediction_diverse.sh

  run_logged "$LOG_DIR/stage6c_strict_stability.log" \
    env ROOT="$ROOT" PY_MM="$PY_MM" OUT_DIR="$STABILITY" \
      STRICT_BLIND_ROOTS="$STRICT_BLIND_ROOTS" ADDHOUT_MASTER_CSV="$ADDHOUT_MASTER" TRAIN_MASTER_CSV="$TRAIN_MASTER" \
      TOP_K="$TOP_K" MIN_COVERAGE="$MIN_COVERAGE" SCORE_MODE=stability_first WEIGHT_MODE="$WEIGHT_MODE" FAMILY_DIVERSE=1 \
      AUDIT_WITH_LABELS=0 STRICT_OUTPUT_NO_LABELS=1 \
      FEATURE_MODE_CAP_MAP="full=2,bare_delta=2,addh_delta=2,delta=1,addh=3,addh_bare=3,bare=1,graph_only=3,concat_interact=1,gated_sum=1,residual_graph=1,text_only=0" \
      MODEL_FAMILY_CAP_MAP="graph_ensemble=12,multiview=4,unknown=2" \
      REQUIRE_FEATURE_MODES="addh_bare=1,addh=1,graph_only=1" \
      bash run_strict_blind_addhout_prediction_diverse.sh

  run_logged "$LOG_DIR/stage6d_strict_oof.log" \
    env ROOT="$ROOT" PY_MM="$PY_MM" OUT_DIR="$OOF" \
      STRICT_BLIND_ROOTS="$STRICT_BLIND_ROOTS" ADDHOUT_MASTER_CSV="$ADDHOUT_MASTER" TRAIN_MASTER_CSV="$TRAIN_MASTER" \
      TOP_K="$TOP_K" MIN_COVERAGE="$MIN_COVERAGE" SCORE_MODE=oof_first WEIGHT_MODE="$WEIGHT_MODE" FAMILY_DIVERSE=1 \
      AUDIT_WITH_LABELS=0 STRICT_OUTPUT_NO_LABELS=1 \
      FEATURE_MODE_CAP_MAP="full=3,bare_delta=2,addh_delta=2,delta=1,addh=2,addh_bare=2,bare=1,graph_only=2,concat_interact=2,gated_sum=1,residual_graph=1,text_only=0" \
      MODEL_FAMILY_CAP_MAP="graph_ensemble=12,multiview=4,unknown=2" \
      REQUIRE_FEATURE_MODES="full=1,graph_only=1,addh_bare=1" \
      bash run_strict_blind_addhout_prediction_diverse.sh

  run_logged "$LOG_DIR/stage6e_strategy_ensemble.log" \
    "$PY_MM" 17_build_strict_blind_strategy_ensemble.py \
      --strategy-dirs "$BALANCED" "$CONSERVATIVE" "$STABILITY" "$OOF" \
      --addhout-master-csv "$ADDHOUT_MASTER" \
      --train-master-csv "$TRAIN_MASTER" \
      --out-dir "$STRICT_FINAL_DIR" \
      --weight-mode soft_inverse_oof \
      --min-coverage 0.80 \
      --strict-output-no-labels
else
  echo "[SKIP] Stage 6 strict-blind strategy ensemble"
fi

if [[ "$RUN_SERVER_FINAL_BLEND" == "1" ]]; then
  echo "[STAGE 7] Final server blend"
  FINAL_AUDIT_ARGS=()
  if [[ "$RUN_AUDIT" == "1" && -s "$LLM_FEATURE_DIR/addhout_audit_labels.csv" ]]; then
    FINAL_AUDIT_ARGS=(--audit-labels-csv "$LLM_FEATURE_DIR/addhout_audit_labels.csv")
  fi
  run_logged "$LOG_DIR/stage7_server_final_blend.log" \
    "$PY_MM" 22_fuse_llm_strict_blind_server.py \
      --llm-pred-csv "$LLM_PRED_DIR/knowledge_enhanced_addhout_predictions.csv" \
      --strict-pred-csv "$STRICT_FINAL_DIR/strict_blind_strategy_ensemble_predictions.csv" \
      --out-dir "$SERVER_FINAL_DIR" \
      --train-features-csv "$LLM_FEATURE_DIR/knowledge_features_train.csv" \
      --target-abs-max "$TARGET_ABS_MAX" \
      "${FINAL_AUDIT_ARGS[@]}"
else
  echo "[SKIP] Stage 7 final server blend"
fi

cat <<DONE
============================================================
[ALL DONE] AddH server pipeline finished.
[RUN_ROOT]      $RUN_ROOT
[LOG_DIR]       $LOG_DIR
[LLM_RESULT]    $LLM_PRED_DIR/knowledge_enhanced_addhout_predictions.csv
[STRICT_RESULT] $STRICT_FINAL_DIR/strict_blind_strategy_ensemble_predictions.csv
[FINAL_RESULT]  $SERVER_FINAL_DIR/server_final_addhout_predictions.csv
[FINAL_XLSX]    $SERVER_FINAL_DIR/server_final_addhout_predictions.xlsx
============================================================
DONE
