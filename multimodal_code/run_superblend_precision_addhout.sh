#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/home/terminator/RL/multi-view}"
PY_MM="${PY_MM:-/data/home/terminator/anaconda3/envs/multiview/bin/python}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034}"

FINAL_METHOD="${FINAL_METHOD:-balanced}"
OUT_DIR="${OUT_DIR:-$RUN_ROOT/outputs_addh_superblend_precision}"

cd "$ROOT"

echo "[INFO] ROOT=$ROOT"
echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] PY_MM=$PY_MM"
echo "[INFO] FINAL_METHOD=$FINAL_METHOD"
echo "[INFO] OUT_DIR=$OUT_DIR"

if [[ ! -f "$RUN_ROOT/outputs_addh_rank_trend_calibrated/rank_trend_calibrated_addhout_predictions.csv" ]]; then
  echo "[INFO] rank-trend output missing; running run_rank_trend_calibrated_addhout.sh first"
  RUN_ROOT="$RUN_ROOT" PY_MM="$PY_MM" bash run_rank_trend_calibrated_addhout.sh
fi

args=(
  27_superblend_precision_addhout.py
  --rank-trend-csv "$RUN_ROOT/outputs_addh_rank_trend_calibrated/rank_trend_calibrated_addhout_predictions.csv"
  --delta-csv "$RUN_ROOT/outputs_addh_pretrained_delta_head/pretrained_delta_head_addhout_predictions.csv"
  --target-csv "$RUN_ROOT/outputs_addh_target_calibrated_fast/target_calibrated_addhout_predictions.csv"
  --knowledge-csv "$RUN_ROOT/outputs_addh_llm_element_knowledge_blend_scnet_deepseek_v4_pro/knowledge_enhanced_addhout_predictions.csv"
  --out-dir "$OUT_DIR"
  --audit-labels-csv "$RUN_ROOT/outputs_addh_llm_element_priors/addhout_audit_labels.csv"
  --final-method "$FINAL_METHOD"
)

if [[ "${ALLOW_AUDIT_SELECTION:-0}" == "1" ]]; then
  args+=(--allow-audit-label-selection)
fi

if [[ "${ALLOW_AUDIT_OFFSET:-0}" == "1" ]]; then
  args+=(--allow-audit-offset-calibration)
fi

if [[ "${CLIP_TO_CANDIDATE_ENVELOPE:-0}" == "1" ]]; then
  args+=(--clip-to-candidate-envelope)
fi

"$PY_MM" "${args[@]}"

echo
echo "[RESULT] audit:"
if [[ -f "$OUT_DIR/superblend_precision_posthoc_audit.csv" ]]; then
  cat "$OUT_DIR/superblend_precision_posthoc_audit.csv"
else
  echo "[WARN] audit csv not found; predictions were still written."
fi

echo
echo "[RESULT] final predictions:"
echo "$OUT_DIR/superblend_precision_addhout_predictions.csv"
echo "[RESULT] final column: pred_superblend_final"
