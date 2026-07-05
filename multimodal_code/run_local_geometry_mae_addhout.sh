#!/usr/bin/env bash
set -Eeuo pipefail

# Local-geometry MAE-first route.
# Adds H/O/dopant local geometry features to both train and AddH-out, then
# retrains the frozen-embedding delta head and final MAE-guarded superblend.

ROOT="${ROOT:-/data/home/terminator/RL/multi-view}"
PY_MM="${PY_MM:-/data/home/terminator/anaconda3/envs/multiview/bin/python}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034}"

SRC_OUT="${SRC_OUT:-$RUN_ROOT/outputs_addh_full_mm_envsplit}"
LLM_FEATURE_DIR="${LLM_FEATURE_DIR:-$RUN_ROOT/outputs_addh_llm_element_priors}"
FAST_CAL_DIR="${FAST_CAL_DIR:-$RUN_ROOT/outputs_addh_target_calibrated_fast}"

GEOM_DIR="${GEOM_DIR:-$RUN_ROOT/outputs_addh_local_geometry_features}"
GEOM_BUNDLE_DIR="${GEOM_BUNDLE_DIR:-$RUN_ROOT/outputs_addh_geometry_pretrained_delta_features}"
GEOM_HEAD_DIR="${GEOM_HEAD_DIR:-$RUN_ROOT/outputs_addh_geometry_pretrained_delta_head}"
GEOM_RANK_DIR="${GEOM_RANK_DIR:-$RUN_ROOT/outputs_addh_geometry_rank_trend}"
GEOM_SUPER_DIR="${GEOM_SUPER_DIR:-$RUN_ROOT/outputs_addh_geometry_mae_superblend}"

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
echo "[INFO] GEOM_DIR=$GEOM_DIR"
echo "[INFO] GEOM_BUNDLE_DIR=$GEOM_BUNDLE_DIR"
echo "[INFO] GEOM_HEAD_DIR=$GEOM_HEAD_DIR"
echo "[INFO] GEOM_RANK_DIR=$GEOM_RANK_DIR"
echo "[INFO] GEOM_SUPER_DIR=$GEOM_SUPER_DIR"

"$PY_MM" 30_build_local_geometry_features.py \
  --root "$ROOT" \
  --train-features "$TRAIN_FEATURES" \
  --addhout-features "$ADDHOUT_FEATURES" \
  --out-dir "$GEOM_DIR" \
  --pbc-axes xy

"$PY_MM" 24_build_pretrained_delta_features.py \
  --train-features "$GEOM_DIR/knowledge_features_train_geom.csv" \
  --addhout-features "$GEOM_DIR/knowledge_features_addhout_geom.csv" \
  --train-dual-emb-pkl "$TRAIN_DUAL_EMB" \
  --addhout-dual-emb-pkl "$ADDHOUT_DUAL_EMB" \
  --out-dir "$GEOM_BUNDLE_DIR" \
  --audit-labels-csv "$AUDIT_LABELS_CSV"

"$PY_MM" 25_train_pretrained_delta_head_addhout.py \
  --bundle-dir "$GEOM_BUNDLE_DIR" \
  --out-dir "$GEOM_HEAD_DIR" \
  --existing-pred-csv "$EXISTING_PRED_CSV" \
  --existing-pred-col "$EXISTING_PRED_COL" \
  --audit-labels-csv "$AUDIT_LABELS_CSV" \
  --top-k 12 \
  --min-oof-improvement 0.03 \
  --max-pred-mean-shift 2.2 \
  --max-delta-blend-weight 0.22 \
  --oracle-diagnostic-tune

"$PY_MM" 26_rank_trend_calibrate_addhout.py \
  --pred-csv "$GEOM_HEAD_DIR/pretrained_delta_head_addhout_predictions.csv" \
  --out-dir "$GEOM_RANK_DIR" \
  --value-col pred_pretrained_delta_final \
  --score-col pred_existing_anchor \
  --fallback-col pred_pretrained_delta_final \
  --audit-labels-csv "$AUDIT_LABELS_CSV" \
  --final-method quantile

"$PY_MM" 27_superblend_precision_addhout.py \
  --rank-trend-csv "$GEOM_RANK_DIR/rank_trend_calibrated_addhout_predictions.csv" \
  --delta-csv "$GEOM_HEAD_DIR/pretrained_delta_head_addhout_predictions.csv" \
  --target-csv "$FAST_CAL_DIR/target_calibrated_addhout_predictions.csv" \
  --knowledge-csv "$RUN_ROOT/outputs_addh_llm_element_knowledge_blend_scnet_deepseek_v4_pro/knowledge_enhanced_addhout_predictions.csv" \
  --out-dir "$GEOM_SUPER_DIR" \
  --audit-labels-csv "$AUDIT_LABELS_CSV" \
  --final-method mae_guarded

echo
echo "[RESULT] geometry manifest:"
cat "$GEOM_DIR/local_geometry_manifest.json" || true

echo
echo "[RESULT] geometry delta-head audit:"
cat "$GEOM_HEAD_DIR/pretrained_delta_head_posthoc_audit.csv" || true

echo
echo "[RESULT] geometry rank-trend audit:"
cat "$GEOM_RANK_DIR/rank_trend_posthoc_audit.csv" || true

echo
echo "[RESULT] geometry MAE superblend audit:"
cat "$GEOM_SUPER_DIR/superblend_precision_posthoc_audit.csv" || true

echo
echo "[FINAL] $GEOM_SUPER_DIR/superblend_precision_addhout_predictions.csv"
echo "[FINAL COLUMN] pred_superblend_final"
