# Domain-Adapted AddH-out Route

This route is designed for the failure mode we observed:

```text
source OOF looks good, but AddH-out gets worse
```

It still trains only on addH/addH-2 labels. AddH-out labels are not used for
training or model selection. AddH-out unlabeled features are used for:

1. source importance weighting through a domain classifier;
2. target-like OOF validation;
3. addH-out distribution gates before blending with the anchor.

## Upload These Files

Upload to:

```bash
/data/home/terminator/RL/multi-view
```

Files:

```text
29_train_domain_adapted_addhout.py
run_domain_adapted_addhout.sh
DOMAIN_ADAPTED_ROUTE.md
```

Also make sure these existing scripts are on the server:

```text
24_build_pretrained_delta_features.py
26_rank_trend_calibrate_addhout.py
27_superblend_precision_addhout.py
```

## Fast Run

Start here.

```bash
cd /data/home/terminator/RL/multi-view
export PY_MM=/data/home/terminator/anaconda3/envs/multiview/bin/python
export RUN_ROOT=/data/home/terminator/RL/multi-view/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034
export DOMAIN_PROFILE=fast
export SUPERBLEND_FINAL_METHOD=balanced

nohup bash run_domain_adapted_addhout.sh \
  > "$RUN_ROOT/logs/nohup_domain_adapted_fast_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

## Medium Run

Only run this if fast is close or improves.

```bash
cd /data/home/terminator/RL/multi-view
export PY_MM=/data/home/terminator/anaconda3/envs/multiview/bin/python
export RUN_ROOT=/data/home/terminator/RL/multi-view/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034
export DOMAIN_PROFILE=medium
export SUPERBLEND_FINAL_METHOD=balanced

nohup bash run_domain_adapted_addhout.sh \
  > "$RUN_ROOT/logs/nohup_domain_adapted_medium_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

## MAE-First Variant

If you care more about MAE than trend:

```bash
export SUPERBLEND_FINAL_METHOD=mae_guarded
```

Then rerun the same command.

## What To Check

```bash
cat "$RUN_ROOT/outputs_addh_domain_adapted_delta_fast/domain_adapted_posthoc_audit.csv"
cat "$RUN_ROOT/outputs_addh_domain_rank_trend_fast/rank_trend_posthoc_audit.csv"
cat "$RUN_ROOT/outputs_addh_domain_superblend_fast_balanced/superblend_precision_posthoc_audit.csv"
cat "$RUN_ROOT/outputs_addh_domain_superblend_fast_balanced/superblend_precision_manifest.json"
cat "$RUN_ROOT/outputs_addh_domain_adapted_delta_fast/domain_adapted_manifest.json"
```

Final prediction file:

```bash
$RUN_ROOT/outputs_addh_domain_superblend_fast_balanced/superblend_precision_addhout_predictions.csv
```

Final column:

```text
pred_superblend_final
```

## Strict Baselines To Beat

Current best strict results:

```text
MAE-first:  MAE=1.6987, Spearman=0.5472
Balanced:   MAE=1.7577, Spearman=0.5903
Trend-first MAE=1.7645, Spearman=0.5960
```

The domain-adapted route is useful only if it beats one of these according to
your priority.

## Package Results

After running fast:

```bash
cd /data/home/terminator/RL/multi-view
export RUN_ROOT=/data/home/terminator/RL/multi-view/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034

tar -czf domain_adapted_fast_status_outputs.tgz \
  "$RUN_ROOT/logs"/*domain_adapted* \
  "$RUN_ROOT/outputs_addh_domain_adapted_delta_fast" \
  "$RUN_ROOT/outputs_addh_domain_rank_trend_fast" \
  "$RUN_ROOT/outputs_addh_domain_superblend_fast_balanced" \
  "$RUN_ROOT/outputs_addh_superblend_precision" \
  "$RUN_ROOT/outputs_addh_rank_trend_calibrated" \
  "$RUN_ROOT/outputs_addh_target_calibrated_fast"
```

Put the `.tgz` in the local workspace and I can judge whether to adopt it.
