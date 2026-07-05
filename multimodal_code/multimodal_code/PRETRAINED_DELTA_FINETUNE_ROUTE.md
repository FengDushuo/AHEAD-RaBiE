# Pretrained Delta Fine-Tune Route

This route trains a small delta-head on frozen pretrained FAIR-Chem dual
embeddings:

```text
dual_emb = concat(addH_emb, bare_emb, addH_emb - bare_emb)
target   = E(addH) - E(bare) - E_H
```

It is designed for the current small addH/addH-2 training set. It does not
fine-tune the full EquiformerV2 backbone by default, because that is very easy
to overfit with only a few hundred labels.

## Files

Upload these files to `/data/home/terminator/RL/multi-view`:

```text
24_build_pretrained_delta_features.py
25_train_pretrained_delta_head_addhout.py
run_pretrained_delta_finetune_addhout.sh
```

This route expects that the previous pipeline has already produced:

```text
$RUN_ROOT/outputs_addh_full_mm_envsplit/addH_dual_eq_emb.pkl
$RUN_ROOT/outputs_addh_full_mm_envsplit/addH_out_dual_eq_emb.pkl
$RUN_ROOT/outputs_addh_llm_element_priors/knowledge_features_train.csv
$RUN_ROOT/outputs_addh_llm_element_priors/knowledge_features_addhout.csv
$RUN_ROOT/outputs_addh_target_calibrated_fast/target_calibrated_addhout_predictions.csv
```

If the dual embedding files are missing, rerun only the embedding stage on the
GPU node first, or reuse the existing run where they already exist.

## Run On Server

This step can run on `mgt` because it uses existing embeddings and trains only
small sklearn models:

```bash
cd /data/home/terminator/RL/multi-view

export PY_MM=/data/home/terminator/anaconda3/envs/multiview/bin/python
export RUN_ROOT=/data/home/terminator/RL/multi-view/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034

bash run_pretrained_delta_finetune_addhout.sh
```

Outputs:

```text
$RUN_ROOT/outputs_addh_pretrained_delta_features/
$RUN_ROOT/outputs_addh_pretrained_delta_head/pretrained_delta_head_addhout_predictions.csv
$RUN_ROOT/outputs_addh_pretrained_delta_head/pretrained_delta_head_posthoc_audit.csv
```

Use this final column:

```text
pred_pretrained_delta_final
```

## How It Protects Accuracy

The script computes grouped OOF metrics on addH/addH-2. Delta-head models are
allowed into the final blend only when they beat the dopant-prior baseline by
at least `--min-oof-improvement` and have sane addH-out prediction statistics.

If no delta-head model passes the gates, the final prediction safely falls back
to the current best calibrated prediction:

```text
pred_fast_target_calibrated
```

That means the route should not make the current result worse just because a
large embedding model looks attractive.

## More Aggressive Experiments

To test whether the pretrained delta head has useful signal without changing
the default guard too much:

```bash
export DELTA_HEAD_DIR="$RUN_ROOT/outputs_addh_pretrained_delta_head_aggressive"

$PY_MM 25_train_pretrained_delta_head_addhout.py \
  --bundle-dir "$RUN_ROOT/outputs_addh_pretrained_delta_features" \
  --out-dir "$DELTA_HEAD_DIR" \
  --existing-pred-csv "$RUN_ROOT/outputs_addh_target_calibrated_fast/target_calibrated_addhout_predictions.csv" \
  --existing-pred-col pred_fast_target_calibrated \
  --min-oof-improvement 0.00 \
  --max-delta-blend-weight 0.35 \
  --audit-labels-csv "$RUN_ROOT/outputs_addh_llm_element_priors/addhout_audit_labels.csv" \
  --oracle-diagnostic-tune
```

Treat `pred_oracle_delta_blend_diagnostic` as diagnostics only, because it uses
addH-out labels.
