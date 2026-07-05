#!/usr/bin/env bash
set -euo pipefail

# Strict-blind LLM/element-prior route for AddH/AddH-2 -> AddH-out.
#
# Labels in addH-out/氢吸附能.xlsx are written only to addhout_audit_labels.csv
# and are used only when AUDIT_WITH_LABELS=1.

PY_MM="${PY_MM:-python}"
OUT_FEATURE_DIR="${OUT_FEATURE_DIR:-outputs_addh_llm_element_priors}"
OUT_PRED_DIR="${OUT_PRED_DIR:-outputs_addh_llm_element_knowledge_blend}"
TARGET_ABS_MAX="${TARGET_ABS_MAX:-10.0}"
AUDIT_WITH_LABELS="${AUDIT_WITH_LABELS:-1}"
ADDHOUT_EXCEL="${ADDHOUT_EXCEL:-addH-out/氢吸附能.xlsx}"
LLM_PRIOR_JSONL="${LLM_PRIOR_JSONL:-}"
RUN_SCNET_LLM="${RUN_SCNET_LLM:-0}"
SCNET_MODEL="${SCNET_MODEL:-DeepSeek-V4-Pro}"
SCNET_PRIOR_JSONL="${SCNET_PRIOR_JSONL:-$OUT_FEATURE_DIR/llm_prior_scnet_deepseek_v4_pro_addhout.jsonl}"
SCNET_PROMPT_STYLE="${SCNET_PROMPT_STYLE:-compact}"
SCNET_SLEEP="${SCNET_SLEEP:-0.8}"
SCNET_MAX_RETRIES="${SCNET_MAX_RETRIES:-3}"

echo "[STEP 1] Build element + LLM/literature prior features"
cmd=(
  "$PY_MM" 19_build_llm_element_prior_features.py
  --addh-dir addH
  --addh2-root addH-2
  --addhout-dir addH-out
  --addhout-excel "$ADDHOUT_EXCEL"
  --out-dir "$OUT_FEATURE_DIR"
  --target-abs-max "$TARGET_ABS_MAX"
  --write-audit-labels
)
if [[ -n "$LLM_PRIOR_JSONL" ]]; then
  cmd+=(--llm-prior-jsonl "$LLM_PRIOR_JSONL")
fi
"${cmd[@]}"

if [[ "$RUN_SCNET_LLM" == "1" ]]; then
  echo "[STEP 1B] Call SCNet $SCNET_MODEL for addH-out LLM priors"
  "$PY_MM" 21_call_scnet_deepseek_priors.py \
    --input-jsonl "$OUT_FEATURE_DIR/llm_prior_prompts.jsonl" \
    --output-jsonl "$SCNET_PRIOR_JSONL" \
    --id-regex "^(CeO2|ZnO)-" \
    --model "$SCNET_MODEL" \
    --prompt-style "$SCNET_PROMPT_STYLE" \
    --disable-thinking \
    --no-response-format \
    --sleep "$SCNET_SLEEP" \
    --max-retries "$SCNET_MAX_RETRIES"

  echo "[STEP 1C] Rebuild features with SCNet LLM priors"
  "$PY_MM" 19_build_llm_element_prior_features.py \
    --addh-dir addH \
    --addh2-root addH-2 \
    --addhout-dir addH-out \
    --addhout-excel "$ADDHOUT_EXCEL" \
    --out-dir "$OUT_FEATURE_DIR" \
    --target-abs-max "$TARGET_ABS_MAX" \
    --llm-prior-jsonl "$SCNET_PRIOR_JSONL" \
    --write-audit-labels
fi

echo "[STEP 2] Train strict-blind knowledge model and blend existing predictions"
cmd=(
  "$PY_MM" 20_train_llm_element_knowledge_blend.py
  --feature-dir "$OUT_FEATURE_DIR"
  --out-dir "$OUT_PRED_DIR"
  --target-abs-max "$TARGET_ABS_MAX"
  --scan-pred-root logs
)
if [[ "$AUDIT_WITH_LABELS" == "1" ]]; then
  cmd+=(--audit-labels-csv "$OUT_FEATURE_DIR/addhout_audit_labels.csv")
fi
"${cmd[@]}"

echo "[DONE]"
echo "[PRED]  $OUT_PRED_DIR/knowledge_enhanced_addhout_predictions.csv"
echo "[XLSX]  $OUT_PRED_DIR/knowledge_enhanced_addhout_predictions.xlsx"
echo "[OOF]   $OUT_PRED_DIR/knowledge_model_oof_metrics.csv"
echo "[BASE]  $OUT_PRED_DIR/base_prediction_file_selection.csv"
