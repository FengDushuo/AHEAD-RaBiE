# LLM / Element Knowledge Route for AddH-out

This route uses LLM/literature knowledge as structured prior features, not as a direct energy oracle.

## 1. Build Features And LLM Prompt Pack

```bash
python 19_build_llm_element_prior_features.py \
  --addh-dir addH \
  --addh2-root addH-2 \
  --addhout-dir addH-out \
  --addhout-excel "addH-out/氢吸附能.xlsx" \
  --out-dir outputs_addh_llm_element_priors \
  --target-abs-max 10 \
  --write-audit-labels
```

Key outputs:

- `outputs_addh_llm_element_priors/knowledge_features_train.csv`
- `outputs_addh_llm_element_priors/knowledge_features_addhout.csv`
- `outputs_addh_llm_element_priors/llm_prior_prompts.jsonl`
- `outputs_addh_llm_element_priors/addhout_audit_labels.csv` only for post-hoc audit

## 2. Optional: Run LLM Priors

Send `llm_prior_prompts.jsonl` to an LLM/batch workflow. Each response should be strict JSON with fields such as:

```json
{
  "id": "ZnO-7-Cu",
  "material": "ZnO",
  "dopant": "Cu",
  "llm_prior_h_ads_eV_guess": null,
  "llm_expected_rank_score": -1.5,
  "llm_h_binding_strength_score": 1.5,
  "llm_oxygen_affinity_score": 2.3,
  "llm_oxide_reducibility_score": 1.2,
  "llm_charge_compensation_complexity": 1.0,
  "llm_host_similarity_to_training": 0.2,
  "llm_confidence": 0.65,
  "rationale_short": "Cu on ZnO is expected to bind H relatively strongly compared with closed-shell dopants.",
  "sources": []
}
```

Then rebuild features with:

```bash
python 19_build_llm_element_prior_features.py \
  --out-dir outputs_addh_llm_element_priors \
  --llm-prior-jsonl your_llm_prior_responses.jsonl \
  --write-audit-labels
```

### SCNet DeepSeek-V4-Pro

SCNet's official OpenAI-compatible endpoint is:

- `https://api.scnet.cn/api/llm/v1/chat/completions`

Use the helper script. The API key must come from `SCNET_API_KEY`; do not write it into scripts.

PowerShell:

```powershell
$env:SCNET_API_KEY="your_api_key"
python 21_call_scnet_deepseek_priors.py `
  --input-jsonl outputs_addh_llm_element_priors/llm_prior_prompts.jsonl `
  --output-jsonl outputs_addh_llm_element_priors/llm_prior_scnet_deepseek_v4_pro_addhout.jsonl `
  --id-regex "^(CeO2|ZnO)-" `
  --model DeepSeek-V4-Pro `
  --prompt-style compact `
  --disable-thinking `
  --no-response-format `
  --max-retries 3 `
  --sleep 0.8
```

Bash:

```bash
export SCNET_API_KEY="your_api_key"
python 21_call_scnet_deepseek_priors.py \
  --input-jsonl outputs_addh_llm_element_priors/llm_prior_prompts.jsonl \
  --output-jsonl outputs_addh_llm_element_priors/llm_prior_scnet_deepseek_v4_pro_addhout.jsonl \
  --id-regex "^(CeO2|ZnO)-" \
  --model DeepSeek-V4-Pro \
  --prompt-style compact \
  --disable-thinking \
  --no-response-format \
  --max-retries 3 \
  --sleep 0.8
```

Then rebuild features:

```bash
python 19_build_llm_element_prior_features.py \
  --out-dir outputs_addh_llm_element_priors \
  --llm-prior-jsonl outputs_addh_llm_element_priors/llm_prior_scnet_deepseek_v4_pro_addhout.jsonl \
  --write-audit-labels
```

Or use the one-shot script:

```bash
export SCNET_API_KEY="your_api_key"
RUN_SCNET_LLM=1 bash run_llm_element_knowledge_blend_addhout.sh
```

## 3. Train And Predict

```bash
python 20_train_llm_element_knowledge_blend.py \
  --feature-dir outputs_addh_llm_element_priors \
  --out-dir outputs_addh_llm_element_knowledge_blend_scnet_deepseek_v4_pro \
  --target-abs-max 10 \
  --scan-pred-root logs
```

For post-hoc audit only:

```bash
python 20_train_llm_element_knowledge_blend.py \
  --feature-dir outputs_addh_llm_element_priors \
  --out-dir outputs_addh_llm_element_knowledge_blend_scnet_deepseek_v4_pro \
  --target-abs-max 10 \
  --scan-pred-root logs \
  --audit-labels-csv outputs_addh_llm_element_priors/addhout_audit_labels.csv
```

Main prediction output:

- `outputs_addh_llm_element_knowledge_blend_scnet_deepseek_v4_pro/knowledge_enhanced_addhout_predictions.csv`
- `outputs_addh_llm_element_knowledge_blend_scnet_deepseek_v4_pro/knowledge_enhanced_addhout_predictions.xlsx`

Final prediction column:

- `pred_llm_element_knowledge_blend`

## Current Local Audit

With 40 SCNet DeepSeek-V4-Pro priors loaded, this route produced on the provided example:

- final MAE: `1.928 eV`
- final RMSE: `2.249 eV`
- final bias: `-0.072 eV`
- Pearson: `0.462`
- Spearman: `0.441`

This is a strict-blind prediction route; audit labels are not used for training, model selection, or blending weights.
