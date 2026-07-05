# Time-Budgeted Robust Retraining Route

This is the recommended retraining route when time cost matters.

It does **not** rerun FAIR-Chem embedding extraction and does **not** fine-tune
EquiformerV2. It reuses the cached dual embeddings and LLM/element features,
then trains robust small heads with repeated grouped CV.

## Upload These Files

Upload these files to:

```bash
/data/home/terminator/RL/multi-view
```

Files:

```text
28_train_time_budgeted_robust_delta_addhout.py
run_time_budgeted_robust_retrain_addhout.sh
ROBUST_RETRAIN_ROUTE.md
```

The route also uses already-uploaded scripts:

```text
24_build_pretrained_delta_features.py
26_rank_trend_calibrate_addhout.py
27_superblend_precision_addhout.py
```

## Fast Run

Fast is the first run I recommend. It should be much cheaper than rerunning the
full multiview/FAIR-Chem pipeline.

```bash
cd /data/home/terminator/RL/multi-view
export PY_MM=/data/home/terminator/anaconda3/envs/multiview/bin/python
export RUN_ROOT=/data/home/terminator/RL/multi-view/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034
export RETRAIN_PROFILE=fast
export SUPERBLEND_FINAL_METHOD=balanced

nohup bash run_time_budgeted_robust_retrain_addhout.sh \
  > "$RUN_ROOT/logs/nohup_robust_retrain_fast_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

## Medium Run

Run this if `fast` improves or is close. It trains more candidates and repeats
the grouped CV more times.

```bash
cd /data/home/terminator/RL/multi-view
export PY_MM=/data/home/terminator/anaconda3/envs/multiview/bin/python
export RUN_ROOT=/data/home/terminator/RL/multi-view/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034
export RETRAIN_PROFILE=medium
export SUPERBLEND_FINAL_METHOD=balanced

nohup bash run_time_budgeted_robust_retrain_addhout.sh \
  > "$RUN_ROOT/logs/nohup_robust_retrain_medium_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

## MAE-First Variant

If MAE matters more than trend:

```bash
export SUPERBLEND_FINAL_METHOD=mae_guarded
```

Then rerun the same command.

## What To Check

```bash
cat "$RUN_ROOT/outputs_addh_robust_retrain_delta_fast/robust_retrain_posthoc_audit.csv"
cat "$RUN_ROOT/outputs_addh_robust_rank_trend_fast/rank_trend_posthoc_audit.csv"
cat "$RUN_ROOT/outputs_addh_robust_superblend_fast_balanced/superblend_precision_posthoc_audit.csv"
cat "$RUN_ROOT/outputs_addh_robust_superblend_fast_balanced/superblend_precision_manifest.json"
```

Final strict prediction file:

```bash
$RUN_ROOT/outputs_addh_robust_superblend_fast_balanced/superblend_precision_addhout_predictions.csv
```

Final column:

```text
pred_superblend_final
```

For strict use, the manifest should say:

```json
"labels_used_for_selection": false
"labels_used_for_offset_calibration": false
```

## Expected Behavior

The robust retrain head may or may not beat the earlier delta head by itself.
That is normal. The important final result is after rank-trend and superblend.

Use `fast` first. If the final audit beats the current strict result
`MAE=1.7577 / Spearman=0.5903`, keep it. If it does not, use the previous
`outputs_addh_superblend_precision` strict result.
