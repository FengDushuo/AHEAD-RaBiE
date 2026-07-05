#!/usr/bin/env bash
set -Eeuo pipefail

# Long but guarded AddH-out optimization route.
#
# The route is intentionally conservative:
#   1) run a heavier source-trained robust retrain candidate;
#   2) run a heavier target-domain-aware candidate using unlabeled AddH-out features;
#   3) apply the chemistry-spike prior to the original and new anchors;
#   4) write audits for comparison if AddH-out labels are available.
#
# AddH-out labels are never used for fitting in these scripts. They are audit-only.

ROOT="${ROOT:-/data/home/terminator/RL/multi-view}"
RUN_ROOT="${RUN_ROOT:-$ROOT}"
PY_MM="${PY_MM:-/data/home/terminator/anaconda3/envs/multiview/bin/python}"
LOG_DIR_WAS_SET="${LOG_DIR+x}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"

RUN_ROBUST="${RUN_ROBUST:-1}"
RUN_DOMAIN="${RUN_DOMAIN:-1}"
RUN_CHEMISTRY="${RUN_CHEMISTRY:-1}"

RETRAIN_PROFILE="${RETRAIN_PROFILE:-thorough}"
SUPERBLEND_FINAL_METHOD="${SUPERBLEND_FINAL_METHOD:-mae_guarded}"

AUDIT_LABELS_CSV="${AUDIT_LABELS_CSV:-auto}"
LLM_FEATURE_DIR="${LLM_FEATURE_DIR:-$RUN_ROOT/outputs_addh_llm_element_priors}"
TRAIN_FEATURES="${TRAIN_FEATURES:-$LLM_FEATURE_DIR/knowledge_features_train.csv}"
ADDHOUT_FEATURES="${ADDHOUT_FEATURES:-$LLM_FEATURE_DIR/knowledge_features_addhout.csv}"

if [[ ! -s "$TRAIN_FEATURES" || ! -s "$ADDHOUT_FEATURES" ]]; then
  echo "[WARN] features not found under RUN_ROOT=$RUN_ROOT"
  echo "[WARN] trying to auto-detect latest run directory under $ROOT"
  DETECTED_TRAIN_FEATURES="$(
    find "$ROOT" -path "*/outputs_addh_llm_element_priors/knowledge_features_train.csv" \
      -type f -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
  if [[ -n "$DETECTED_TRAIN_FEATURES" && -s "$DETECTED_TRAIN_FEATURES" ]]; then
    LLM_FEATURE_DIR="$(dirname "$DETECTED_TRAIN_FEATURES")"
    RUN_ROOT="$(dirname "$LLM_FEATURE_DIR")"
    TRAIN_FEATURES="$DETECTED_TRAIN_FEATURES"
    ADDHOUT_FEATURES="$LLM_FEATURE_DIR/knowledge_features_addhout.csv"
    if [[ -z "$LOG_DIR_WAS_SET" ]]; then
      LOG_DIR="$RUN_ROOT/logs"
    fi
    echo "[INFO] auto-detected RUN_ROOT=$RUN_ROOT"
    echo "[INFO] auto-detected LLM_FEATURE_DIR=$LLM_FEATURE_DIR"
  else
    echo "[ERROR] could not find outputs_addh_llm_element_priors/knowledge_features_train.csv under $ROOT" >&2
    echo "[HINT] run this on server to locate it:" >&2
    echo "       find $ROOT -path '*/outputs_addh_llm_element_priors/knowledge_features_train.csv' -type f" >&2
    exit 2
  fi
fi

BASE_SUPERBLEND_CSV="${BASE_SUPERBLEND_CSV:-$RUN_ROOT/outputs_addh_superblend_precision/superblend_precision_addhout_predictions.csv}"
ROBUST_SUPER_DIR="${ROBUST_SUPER_DIR:-$RUN_ROOT/outputs_addh_robust_superblend_${RETRAIN_PROFILE}_${SUPERBLEND_FINAL_METHOD}}"
ROBUST_SUPER_CSV="$ROBUST_SUPER_DIR/superblend_precision_addhout_predictions.csv"
DOMAIN_LONG_DIR="${DOMAIN_LONG_DIR:-$RUN_ROOT/outputs_addh_target_domain_aware_supertrainer_long}"
DOMAIN_LONG_CSV="$DOMAIN_LONG_DIR/domain_aware_addhout_predictions.csv"

BASE_CHEM_DIR="${BASE_CHEM_DIR:-$RUN_ROOT/outputs_addh_chemistry_spike_prior}"
ROBUST_CHEM_DIR="${ROBUST_CHEM_DIR:-$RUN_ROOT/outputs_addh_chemistry_spike_prior_robust_${RETRAIN_PROFILE}_${SUPERBLEND_FINAL_METHOD}}"
DOMAIN_CHEM_DIR="${DOMAIN_CHEM_DIR:-$RUN_ROOT/outputs_addh_chemistry_spike_prior_domain_long}"

cd "$ROOT"
mkdir -p "$LOG_DIR"

echo "[INFO] ROOT=$ROOT"
echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] PY_MM=$PY_MM"
echo "[INFO] RUN_ROBUST=$RUN_ROBUST RUN_DOMAIN=$RUN_DOMAIN RUN_CHEMISTRY=$RUN_CHEMISTRY"
echo "[INFO] RETRAIN_PROFILE=$RETRAIN_PROFILE SUPERBLEND_FINAL_METHOD=$SUPERBLEND_FINAL_METHOD"
echo "[INFO] AUDIT_LABELS_CSV=$AUDIT_LABELS_CSV"

for f in \
  24_build_pretrained_delta_features.py \
  26_rank_trend_calibrate_addhout.py \
  27_superblend_precision_addhout.py \
  28_train_time_budgeted_robust_delta_addhout.py \
  32_train_target_domain_aware_addhout.py \
  34_apply_chemistry_spike_prior_addhout.py \
  run_time_budgeted_robust_retrain_addhout.sh \
  run_target_domain_aware_supertrainer_addhout.sh \
  run_chemistry_spike_prior_addhout.sh
do
  if [[ ! -s "$f" ]]; then
    echo "[ERROR] missing required file: $ROOT/$f" >&2
    exit 2
  fi
done

"$PY_MM" -m py_compile \
  28_train_time_budgeted_robust_delta_addhout.py \
  32_train_target_domain_aware_addhout.py \
  34_apply_chemistry_spike_prior_addhout.py

if [[ "$RUN_ROBUST" == "1" ]]; then
  echo
  echo "[STEP 1] robust retrain candidate: profile=$RETRAIN_PROFILE"
  ROOT="$ROOT" \
  RUN_ROOT="$RUN_ROOT" \
  PY_MM="$PY_MM" \
  RETRAIN_PROFILE="$RETRAIN_PROFILE" \
  SUPERBLEND_FINAL_METHOD="$SUPERBLEND_FINAL_METHOD" \
  AUDIT_LABELS_CSV="$AUDIT_LABELS_CSV" \
  bash run_time_budgeted_robust_retrain_addhout.sh
else
  echo "[SKIP] robust retrain"
fi

if [[ "$RUN_DOMAIN" == "1" ]]; then
  echo
  echo "[STEP 2] long target-domain-aware candidate"
  ROOT="$ROOT" \
  RUN_ROOT="$RUN_ROOT" \
  PY_MM="$PY_MM" \
  OUT_DIR="$DOMAIN_LONG_DIR" \
  AUDIT_LABELS_CSV="$AUDIT_LABELS_CSV" \
  N_SPLITS="${N_SPLITS:-5}" \
  TOP_K="${TOP_K:-18}" \
  FINAL_MODE="${FINAL_MODE:-mae_guarded}" \
  FEATURE_SETS="${FEATURE_SETS:-tabular,tabular_graph64,graph64,tabular_graph128}" \
  MODELS="${MODELS:-ridge,huber,elastic,extratrees,rf,hgb,gbr}" \
  TARGET_MODES="${TARGET_MODES:-absolute,residual_dopant,residual_llm}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-12}" \
  bash run_target_domain_aware_supertrainer_addhout.sh
else
  echo "[SKIP] domain-aware long candidate"
fi

if [[ "$RUN_CHEMISTRY" == "1" ]]; then
  echo
  echo "[STEP 3A] chemistry-spike on base strict superblend"
  if [[ -s "$BASE_SUPERBLEND_CSV" ]]; then
    ROOT="$ROOT" RUN_ROOT="$RUN_ROOT" PY_MM="$PY_MM" \
    TRAIN_FEATURES="$TRAIN_FEATURES" \
    ADDHOUT_FEATURES="$ADDHOUT_FEATURES" \
    PRED_CSV="$BASE_SUPERBLEND_CSV" \
    AUDIT_LABELS_CSV="$AUDIT_LABELS_CSV" \
    OUT_DIR="$BASE_CHEM_DIR" \
    PROFILE=both FINAL_PROFILE=aggressive \
    ANCHOR_COL=auto TREND_COL=pred_superblend_trend \
    bash run_chemistry_spike_prior_addhout.sh
  else
    echo "[WARN] missing base superblend: $BASE_SUPERBLEND_CSV"
  fi

  echo
  echo "[STEP 3B] chemistry-spike on robust retrain superblend"
  if [[ -s "$ROBUST_SUPER_CSV" ]]; then
    ROOT="$ROOT" RUN_ROOT="$RUN_ROOT" PY_MM="$PY_MM" \
    TRAIN_FEATURES="$TRAIN_FEATURES" \
    ADDHOUT_FEATURES="$ADDHOUT_FEATURES" \
    PRED_CSV="$ROBUST_SUPER_CSV" \
    AUDIT_LABELS_CSV="$AUDIT_LABELS_CSV" \
    OUT_DIR="$ROBUST_CHEM_DIR" \
    PROFILE=both FINAL_PROFILE=aggressive \
    ANCHOR_COL=auto TREND_COL=pred_superblend_trend \
    bash run_chemistry_spike_prior_addhout.sh
  else
    echo "[WARN] missing robust superblend: $ROBUST_SUPER_CSV"
  fi

  echo
  echo "[STEP 3C] chemistry-spike on long domain-aware candidate"
  if [[ -s "$DOMAIN_LONG_CSV" ]]; then
    ROOT="$ROOT" RUN_ROOT="$RUN_ROOT" PY_MM="$PY_MM" \
    TRAIN_FEATURES="$TRAIN_FEATURES" \
    ADDHOUT_FEATURES="$ADDHOUT_FEATURES" \
    PRED_CSV="$DOMAIN_LONG_CSV" \
    AUDIT_LABELS_CSV="$AUDIT_LABELS_CSV" \
    OUT_DIR="$DOMAIN_CHEM_DIR" \
    PROFILE=both FINAL_PROFILE=aggressive \
    ANCHOR_COL=pred_domain_aware_final TREND_COL=pred_domain_aware_trend \
    bash run_chemistry_spike_prior_addhout.sh
  else
    echo "[WARN] missing domain-aware long predictions: $DOMAIN_LONG_CSV"
  fi
else
  echo "[SKIP] chemistry-spike prior"
fi

echo
echo "[SUMMARY] post-hoc audit files, if labels are available:"
for f in \
  "$BASE_CHEM_DIR/chemistry_spike_posthoc_audit.csv" \
  "$ROBUST_CHEM_DIR/chemistry_spike_posthoc_audit.csv" \
  "$DOMAIN_CHEM_DIR/chemistry_spike_posthoc_audit.csv" \
  "$ROBUST_SUPER_DIR/superblend_precision_posthoc_audit.csv" \
  "$DOMAIN_LONG_DIR/domain_aware_posthoc_audit.csv"
do
  if [[ -s "$f" ]]; then
    echo
    echo "===== $f ====="
    cat "$f"
  fi
done

echo
echo "[RECOMMENDED DEFAULT]"
echo "  predictions: $BASE_CHEM_DIR/chemistry_spike_addhout_predictions.csv"
echo "  column:      pred_chem_spike_final"
echo
echo "[PACK]"
echo "  cd \"$RUN_ROOT\" && tar -czf long_chemistry_guided_status_outputs.tgz \\"
echo "    outputs_addh_chemistry_spike_prior \\"
echo "    outputs_addh_chemistry_spike_prior_robust_${RETRAIN_PROFILE}_${SUPERBLEND_FINAL_METHOD} \\"
echo "    outputs_addh_chemistry_spike_prior_domain_long \\"
echo "    outputs_addh_robust_superblend_${RETRAIN_PROFILE}_${SUPERBLEND_FINAL_METHOD} \\"
echo "    outputs_addh_target_domain_aware_supertrainer_long \\"
echo "    logs/nohup_long_chemistry_guided_*.log"
