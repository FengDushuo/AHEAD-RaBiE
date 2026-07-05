# AddH-out Superblend Precision Route

This route is the final lightweight step after:

1. `run_fast_target_calibrated_addhout.sh`
2. `run_pretrained_delta_finetune_addhout.sh`
3. `run_rank_trend_calibrated_addhout.sh`

It does not rerun FAIR-Chem and does not need GPU. It only reads the existing
prediction CSV files and writes a more robust final blend.

## Recommended Strict Run

Use this when addH-out labels must not be used for model selection.

```bash
cd /data/home/terminator/RL/multi-view
export PY_MM=/data/home/terminator/anaconda3/envs/multiview/bin/python
export RUN_ROOT=/data/home/terminator/RL/multi-view/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034

nohup bash run_superblend_precision_addhout.sh \
  > "$RUN_ROOT/logs/nohup_superblend_precision_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

Final file:

```bash
$RUN_ROOT/outputs_addh_superblend_precision/superblend_precision_addhout_predictions.csv
```

Final column:

```text
pred_superblend_final
```

Default `FINAL_METHOD=balanced` produces `pred_superblend_final` from
`pred_superblend_balanced`, which is designed to improve the current
rank-trend result while keeping trend stable.

## MAE-First Run

Use this if you care more about lower MAE than rank trend.

```bash
cd /data/home/terminator/RL/multi-view
export PY_MM=/data/home/terminator/anaconda3/envs/multiview/bin/python
export RUN_ROOT=/data/home/terminator/RL/multi-view/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034
export FINAL_METHOD=mae_guarded

nohup bash run_superblend_precision_addhout.sh \
  > "$RUN_ROOT/logs/nohup_superblend_mae_guarded_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

This is more aggressive because it uses the delta-head ensemble shape more
strongly. Check both MAE and Spearman before adopting it.

## Explicit Audit-Label Calibration

Use this only if you intentionally allow addH-out labels to calibrate this same
batch. This is not strict-blind and should not be reported as pure
generalization.

```bash
cd /data/home/terminator/RL/multi-view
export PY_MM=/data/home/terminator/anaconda3/envs/multiview/bin/python
export RUN_ROOT=/data/home/terminator/RL/multi-view/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034
export FINAL_METHOD=auto_audit
export ALLOW_AUDIT_SELECTION=1
export ALLOW_AUDIT_OFFSET=1

nohup bash run_superblend_precision_addhout.sh \
  > "$RUN_ROOT/logs/nohup_superblend_audit_calibrated_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

Audit-calibrated output will be clearly marked in:

```bash
$RUN_ROOT/outputs_addh_superblend_precision/superblend_precision_manifest.json
$RUN_ROOT/outputs_addh_superblend_precision/audit_offset_calibration.json
```

## What To Check

```bash
cat "$RUN_ROOT/outputs_addh_superblend_precision/superblend_precision_posthoc_audit.csv"
cat "$RUN_ROOT/outputs_addh_superblend_precision/superblend_precision_manifest.json"
```

For strict use, confirm:

```json
"labels_used_for_selection": false
"labels_used_for_offset_calibration": false
```

For the current audit package, the expected strict balanced result should be
slightly better than the previous rank-trend output. The aggressive MAE-first
and audit-calibrated modes may reduce MAE further, but the latter uses labels.
