# Target-domain-aware supertrainer route

This is a stricter and more professional route for AddH/AddH-2 -> AddH-out.

## What it does

1. Reads addH/addH-2 train features and AddH-out covariates.
2. Estimates train-sample target-likeness using a domain classifier and nearest-neighbor distance to AddH-out.
3. Trains absolute and residual models with target-domain sample weights.
4. Selects models using grouped OOF MAE, target-weighted OOF MAE, target-like OOF MAE, and AddH-out prediction-distribution guards.
5. Blends the new target-aware model conservatively with the current strict-blind anchor.

AddH-out labels are audit-only by default.

## Run on server

```bash
cd /data/home/terminator/RL/multi-view

export RUN_ROOT=/data/home/terminator/RL/multi-view/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034
export PY_MM=/data/home/terminator/anaconda3/envs/multiview/bin/python

nohup bash run_target_domain_aware_supertrainer_addhout.sh \
  > "$RUN_ROOT/logs/nohup_target_domain_aware_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

Fast diagnostic version:

```bash
MODELS=ridge,hgb FEATURE_SETS=tabular,tabular_graph64 TOP_K=6 \
nohup bash run_target_domain_aware_supertrainer_addhout.sh \
  > "$RUN_ROOT/logs/nohup_target_domain_aware_fast_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

Trend-priority version:

```bash
FINAL_MODE=trend nohup bash run_target_domain_aware_supertrainer_addhout.sh \
  > "$RUN_ROOT/logs/nohup_target_domain_aware_trend_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

## Results to inspect

```bash
cat "$RUN_ROOT/outputs_addh_target_domain_aware_supertrainer/domain_aware_posthoc_audit.csv"
cat "$RUN_ROOT/outputs_addh_target_domain_aware_supertrainer/domain_aware_selected_models.csv"
head -40 "$RUN_ROOT/outputs_addh_target_domain_aware_supertrainer/domain_shift_feature_report.csv"
```

Key prediction columns:

- `pred_domain_aware_final`: default final column.
- `pred_domain_aware_guarded`: MAE-oriented conservative blend.
- `pred_domain_aware_balanced`: blends MAE and trend anchors.
- `pred_domain_aware_trend`: more trend-oriented.
- `pred_domain_aware_model`: raw target-domain-aware model ensemble.

Compare against:

- `pred_superblend_final`
- `pred_superblend_mae_guarded`
- `pred_superblend_trend`

## Package for local review

```bash
cd "$RUN_ROOT"

tar -czf target_domain_aware_status_outputs.tgz \
  outputs_addh_target_domain_aware_supertrainer \
  outputs_addh_superblend_precision \
  outputs_addh_pretrained_delta_head \
  outputs_addh_rank_trend_calibrated \
  logs/nohup_target_domain_aware*.log
```

Download `target_domain_aware_status_outputs.tgz` to the local workspace.
