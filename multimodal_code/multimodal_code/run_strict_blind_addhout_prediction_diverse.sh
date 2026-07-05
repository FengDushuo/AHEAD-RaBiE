#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/home/terminator/RL/multi-view}"
cd "$ROOT"

PY_MM="${PY_MM:-/data/home/terminator/anaconda3/envs/multiview/bin/python}"
OUT_DIR="${OUT_DIR:-outputs_addh_strict_blind_diverse}"
STRICT_BLIND_ROOTS="${STRICT_BLIND_ROOTS:-outputs_addh_graph_ensemble_refine_v2 outputs_addh_graph_ensemble outputs_addh_modelgrid_v2 outputs_addh_modelgrid_v2_full}"
ADDHOUT_MASTER_CSV="${ADDHOUT_MASTER_CSV:-outputs_addh_full_mm_envsplit/addH_out_master_normalized.csv}"
TRAIN_MASTER_CSV="${TRAIN_MASTER_CSV:-outputs_addh_full_mm_envsplit/addH_master_target_weighted_mild.csv}"
TOP_K="${TOP_K:-16}"
MIN_COVERAGE="${MIN_COVERAGE:-0.90}"
SCORE_MODE="${SCORE_MODE:-conservative}"          # conservative | balanced | oof_first | stability_first
WEIGHT_MODE="${WEIGHT_MODE:-soft_inverse_rmse}"  # soft_inverse_rmse | rank | uniform
FAMILY_DIVERSE="${FAMILY_DIVERSE:-1}"
AUDIT_WITH_LABELS="${AUDIT_WITH_LABELS:-0}"
CLIP_TO_SOURCE_RANGE="${CLIP_TO_SOURCE_RANGE:-0}"

# Default caps are designed to avoid full/delta dominance.
FEATURE_MODE_CAP="${FEATURE_MODE_CAP:-3}"
FEATURE_MODE_CAP_MAP="${FEATURE_MODE_CAP_MAP:-full=3,bare_delta=2,addh_delta=2,delta=1,addh=3,addh_bare=3,bare=1,graph_only=2,concat_interact=2,gated_sum=1,residual_graph=1,text_only=0}"
MODEL_FAMILY_CAP_MAP="${MODEL_FAMILY_CAP_MAP:-graph_ensemble=12,multiview=4,unknown=2}"
REQUIRE_FEATURE_MODES="${REQUIRE_FEATURE_MODES:-addh_bare=1,addh=1,full=1}"
CONSERVATIVE_MODE_PENALTY="${CONSERVATIVE_MODE_PENALTY:-full=0.08,bare_delta=0.06,addh_delta=0.05,delta=0.10,text_only=0.25,addh=0.00,addh_bare=0.00,bare=0.03,graph_only=0.03,concat_interact=0.05,gated_sum=0.06,residual_graph=0.06}"
PREFER_FEATURE_MODES="${PREFER_FEATURE_MODES:-addh_bare,addh,graph_only,full}"
STRICT_OUTPUT_NO_LABELS="${STRICT_OUTPUT_NO_LABELS:-1}"

mkdir -p "$OUT_DIR"

echo "[INFO] ROOT=$ROOT"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] TOP_K=$TOP_K"
echo "[INFO] MIN_COVERAGE=$MIN_COVERAGE"
echo "[INFO] SCORE_MODE=$SCORE_MODE"
echo "[INFO] WEIGHT_MODE=$WEIGHT_MODE"
echo "[INFO] FAMILY_DIVERSE=$FAMILY_DIVERSE"
echo "[INFO] STRICT_BLIND_ROOTS=$STRICT_BLIND_ROOTS"
echo "[INFO] ADDHOUT_MASTER_CSV=$ADDHOUT_MASTER_CSV"
echo "[INFO] TRAIN_MASTER_CSV=$TRAIN_MASTER_CSV"
echo "[INFO] FEATURE_MODE_CAP_MAP=$FEATURE_MODE_CAP_MAP"
echo "[INFO] MODEL_FAMILY_CAP_MAP=$MODEL_FAMILY_CAP_MAP"
echo "[INFO] REQUIRE_FEATURE_MODES=$REQUIRE_FEATURE_MODES"
echo "[INFO] PREFER_FEATURE_MODES=$PREFER_FEATURE_MODES"
echo "[INFO] AUDIT_WITH_LABELS=$AUDIT_WITH_LABELS"

ARGS=(
  --roots
)
read -r -a STRICT_ROOT_ARRAY <<< "$STRICT_BLIND_ROOTS"
ARGS+=("${STRICT_ROOT_ARRAY[@]}")
ARGS+=(
  --addhout-master-csv "$ADDHOUT_MASTER_CSV"
  --train-master-csv "$TRAIN_MASTER_CSV"
  --out-dir "$OUT_DIR"
  --top-k "$TOP_K"
  --min-coverage "$MIN_COVERAGE"
  --score-mode "$SCORE_MODE"
  --weight-mode "$WEIGHT_MODE"
  --feature-mode-cap "$FEATURE_MODE_CAP"
  --feature-mode-cap-map "$FEATURE_MODE_CAP_MAP"
  --model-family-cap-map "$MODEL_FAMILY_CAP_MAP"
  --require-feature-modes "$REQUIRE_FEATURE_MODES"
  --conservative-mode-penalty "$CONSERVATIVE_MODE_PENALTY"
  --prefer-feature-modes "$PREFER_FEATURE_MODES"
)

if [[ "$FAMILY_DIVERSE" == "1" ]]; then
  ARGS+=(--family-diverse)
fi
if [[ "$AUDIT_WITH_LABELS" == "1" ]]; then
  ARGS+=(--audit-with-labels)
fi
if [[ "$CLIP_TO_SOURCE_RANGE" == "1" ]]; then
  ARGS+=(--clip-to-source-range)
fi
if [[ "$STRICT_OUTPUT_NO_LABELS" == "1" ]]; then
  ARGS+=(--strict-output-no-labels)
fi

"$PY_MM" 16_make_strict_blind_addhout_ensemble_v2.py "${ARGS[@]}"

echo "[DONE] family-diverse strict blind AddH-out prediction finished."
echo "[RESULT] $OUT_DIR/strict_blind_model_selection_summary.csv"
echo "[RESULT] $OUT_DIR/strict_blind_selected_models.csv"
echo "[RESULT] $OUT_DIR/strict_blind_addhout_predictions.csv"
