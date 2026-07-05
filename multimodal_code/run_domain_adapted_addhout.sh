#!/usr/bin/env bash
set -Eeuo pipefail

# Domain-adapted AddH-out route.
# Uses unlabeled AddH-out feature distribution for importance weighting and
# target-like validation. It does not use addH-out labels for training/selection.

ROOT="${ROOT:-/data/home/terminator/RL/multi-view}"
PY_MM="${PY_MM:-/data/home/terminator/anaconda3/envs/multiview/bin/python}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034}"

DOMAIN_PROFILE="${DOMAIN_PROFILE:-fast}"                  # fast, medium, thorough
SUPERBLEND_FINAL_METHOD="${SUPERBLEND_FINAL_METHOD:-balanced}"  # balanced, trend, mae_guarded

SRC_OUT="${SRC_OUT:-$RUN_ROOT/outputs_addh_full_mm_envsplit}"
LLM_FEATURE_DIR="${LLM_FEATURE_DIR:-$RUN_ROOT/outputs_addh_llm_element_priors}"
FAST_CAL_DIR="${FAST_CAL_DIR:-$RUN_ROOT/outputs_addh_target_calibrated_fast}"

DELTA_FEATURE_DIR="${DELTA_FEATURE_DIR:-$RUN_ROOT/outputs_addh_pretrained_delta_features}"
DOMAIN_DIR="${DOMAIN_DIR:-$RUN_ROOT/outputs_addh_domain_adapted_delta_${DOMAIN_PROFILE}}"
DOMAIN_RANK_DIR="${DOMAIN_RANK_DIR:-$RUN_ROOT/outputs_addh_domain_rank_trend_${DOMAIN_PROFILE}}"
DOMAIN_SUPER_DIR="${DOMAIN_SUPER_DIR:-$RUN_ROOT/outputs_addh_domain_superblend_${DOMAIN_PROFILE}_${SUPERBLEND_FINAL_METHOD}}"

TRAIN_FEATURES="${TRAIN_FEATURES:-$LLM_FEATURE_DIR/knowledge_features_train.csv}"
ADDHOUT_FEATURES="${ADDHOUT_FEATURES:-$LLM_FEATURE_DIR/knowledge_features_addhout.csv}"
TRAIN_DUAL_EMB="${TRAIN_DUAL_EMB:-$SRC_OUT/addH_dual_eq_emb.pkl}"
ADDHOUT_DUAL_EMB="${ADDHOUT_DUAL_EMB:-$SRC_OUT/addH_out_dual_eq_emb.pkl}"
EXISTING_PRED_CSV="${EXISTING_PRED_CSV:-$FAST_CAL_DIR/target_calibrated_addhout_predictions.csv}"
EXISTING_PRED_COL="${EXISTING_PRED_COL:-pred_fast_target_calibrated}"
AUDIT_LABELS_CSV="${AUDIT_LABELS_CSV:-$LLM_FEATURE_DIR/addhout_audit_labels.csv}"

cd "$ROOT"

echo "[INFO] ROOT=$ROOT"
echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] PY_MM=$PY_MM"
echo "[INFO] DOMAIN_PROFILE=$DOMAIN_PROFILE"
echo "[INFO] SUPERBLEND_FINAL_METHOD=$SUPERBLEND_FINAL_METHOD"
echo "[INFO] DELTA_FEATURE_DIR=$DELTA_FEATURE_DIR"
echo "[INFO] DOMAIN_DIR=$DOMAIN_DIR"
echo "[INFO] DOMAIN_RANK_DIR=$DOMAIN_RANK_DIR"
echo "[INFO] DOMAIN_SUPER_DIR=$DOMAIN_SUPER_DIR"

if [[ ! -f "$DELTA_FEATURE_DIR/pretrained_delta_feature_bundle.npz" ]]; then
  echo "[INFO] feature bundle missing; building it now"
  "$PY_MM" 24_build_pretrained_delta_features.py \
    --train-features "$TRAIN_FEATURES" \
    --addhout-features "$ADDHOUT_FEATURES" \
    --train-dual-emb-pkl "$TRAIN_DUAL_EMB" \
    --addhout-dual-emb-pkl "$ADDHOUT_DUAL_EMB" \
    --out-dir "$DELTA_FEATURE_DIR" \
    --audit-labels-csv "$AUDIT_LABELS_CSV"
else
  echo "[INFO] reusing feature bundle: $DELTA_FEATURE_DIR/pretrained_delta_feature_bundle.npz"
fi

"$PY_MM" 29_train_domain_adapted_addhout.py \
  --bundle-dir "$DELTA_FEATURE_DIR" \
  --out-dir "$DOMAIN_DIR" \
  --existing-pred-csv "$EXISTING_PRED_CSV" \
  --existing-pred-col "$EXISTING_PRED_COL" \
  --audit-labels-csv "$AUDIT_LABELS_CSV" \
  --profile "$DOMAIN_PROFILE"

"$PY_MM" 26_rank_trend_calibrate_addhout.py \
  --pred-csv "$DOMAIN_DIR/domain_adapted_addhout_predictions.csv" \
  --out-dir "$DOMAIN_RANK_DIR" \
  --value-col pred_domain_adapted_final \
  --score-col pred_existing_anchor \
  --fallback-col pred_domain_adapted_final \
  --audit-labels-csv "$AUDIT_LABELS_CSV" \
  --final-method quantile

"$PY_MM" 27_superblend_precision_addhout.py \
  --rank-trend-csv "$DOMAIN_RANK_DIR/rank_trend_calibrated_addhout_predictions.csv" \
  --delta-csv "$DOMAIN_DIR/domain_adapted_addhout_predictions.csv" \
  --target-csv "$FAST_CAL_DIR/target_calibrated_addhout_predictions.csv" \
  --knowledge-csv "$RUN_ROOT/outputs_addh_llm_element_knowledge_blend_scnet_deepseek_v4_pro/knowledge_enhanced_addhout_predictions.csv" \
  --out-dir "$DOMAIN_SUPER_DIR" \
  --audit-labels-csv "$AUDIT_LABELS_CSV" \
  --final-method "$SUPERBLEND_FINAL_METHOD"

echo
echo "[RESULT] domain-adapted audit:"
cat "$DOMAIN_DIR/domain_adapted_posthoc_audit.csv" || true

echo
echo "[RESULT] domain rank-trend audit:"
cat "$DOMAIN_RANK_DIR/rank_trend_posthoc_audit.csv" || true

echo
echo "[RESULT] domain superblend audit:"
cat "$DOMAIN_SUPER_DIR/superblend_precision_posthoc_audit.csv" || true

echo
echo "[FINAL] $DOMAIN_SUPER_DIR/superblend_precision_addhout_predictions.csv"
echo "[FINAL COLUMN] pred_superblend_final"
