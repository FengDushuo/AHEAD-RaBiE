#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/home/terminator/RL/multi-view}"
cd "$ROOT"

PY_MM="${PY_MM:-/data/home/terminator/anaconda3/envs/multiview/bin/python}"
OUT_DIR="${OUT_DIR:-outputs_addh_strict_blind}"
TOP_K="${TOP_K:-12}"
MIN_COVERAGE="${MIN_COVERAGE:-0.90}"
SCORE_MODE="${SCORE_MODE:-balanced}"          # balanced | oof_first | stability_first
WEIGHT_MODE="${WEIGHT_MODE:-soft_inverse_rmse}" # soft_inverse_rmse | rank | uniform
PREFER_FEATURE_MODES="${PREFER_FEATURE_MODES:-}" # e.g. addh_bare,addh,full. Empty = no preference.
AUDIT_WITH_LABELS="${AUDIT_WITH_LABELS:-0}"   # 0 = strict blind output only; 1 = add post-hoc metrics, not used for selection
CLIP_TO_SOURCE_RANGE="${CLIP_TO_SOURCE_RANGE:-0}"

mkdir -p "$OUT_DIR"

echo "[INFO] ROOT=$ROOT"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] TOP_K=$TOP_K"
echo "[INFO] MIN_COVERAGE=$MIN_COVERAGE"
echo "[INFO] SCORE_MODE=$SCORE_MODE"
echo "[INFO] WEIGHT_MODE=$WEIGHT_MODE"
echo "[INFO] PREFER_FEATURE_MODES=$PREFER_FEATURE_MODES"
echo "[INFO] AUDIT_WITH_LABELS=$AUDIT_WITH_LABELS"
echo "[INFO] CLIP_TO_SOURCE_RANGE=$CLIP_TO_SOURCE_RANGE"

ARGS=(
  --roots outputs_addh_graph_ensemble_refine_v2 outputs_addh_graph_ensemble outputs_addh_modelgrid_v2 outputs_addh_modelgrid_v2_full
  --addhout-master-csv outputs_addh_full_mm_envsplit/addH_out_master_normalized.csv
  --train-master-csv outputs_addh_full_mm_envsplit/addH_master_target_weighted_mild.csv
  --out-dir "$OUT_DIR"
  --top-k "$TOP_K"
  --min-coverage "$MIN_COVERAGE"
  --score-mode "$SCORE_MODE"
  --weight-mode "$WEIGHT_MODE"
)

if [[ -n "$PREFER_FEATURE_MODES" ]]; then
  ARGS+=(--prefer-feature-modes "$PREFER_FEATURE_MODES")
fi
if [[ "$AUDIT_WITH_LABELS" == "1" ]]; then
  ARGS+=(--audit-with-labels)
fi
if [[ "$CLIP_TO_SOURCE_RANGE" == "1" ]]; then
  ARGS+=(--clip-to-source-range)
fi

"$PY_MM" 16_make_strict_blind_addhout_ensemble.py "${ARGS[@]}"

echo "[DONE] strict blind AddH-out prediction finished."
echo "[RESULT] $OUT_DIR/strict_blind_model_selection_summary.csv"
echo "[RESULT] $OUT_DIR/strict_blind_selected_models.csv"
echo "[RESULT] $OUT_DIR/strict_blind_addhout_predictions.csv"
