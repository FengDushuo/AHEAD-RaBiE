# Few-shot / Physics-enhanced AddH-out route

This route is for paper-safe target-domain adaptation.

## Strict-blind vs few-shot

- Strict-blind result:
  - Train/model selection uses only `addH + addH-2`.
  - AddH-out labels are used only for post-hoc audit.
  - Use `outputs_addh_superblend_precision/superblend_precision_addhout_predictions.csv`.

- Few-shot target-domain adaptation:
  - A small selected subset of AddH-out labels is used for calibration.
  - Scores are reported only on the remaining held-out AddH-out rows.
  - Use `outputs_addh_fewshot_domain_calibration/fewshot_holdout_summary.csv`.

## Recommended server command

```bash
cd /data/home/terminator/RL/multi-view

export RUN_ROOT=/data/home/terminator/RL/multi-view/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034
export PY_MM=/data/home/terminator/anaconda3/envs/multiview/bin/python
export PY_FAIRCHEM=/data/home/terminator/anaconda3/envs/fairchem/bin/python
export FAIRCHEM_MODEL_DIR=/data/home/terminator/RL/multi-view/equiformer_v2_31m_allmd
export GPU_ID=2

nohup bash run_physics_fewshot_addhout.sh \
  > "$RUN_ROOT/logs/nohup_physics_fewshot_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

If upstream FAIR-Chem embeddings or knowledge feature tables are missing, run:

```bash
BUILD_UPSTREAM=1 nohup bash run_physics_fewshot_addhout.sh \
  > "$RUN_ROOT/logs/nohup_physics_fewshot_upstream_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

## Faster command using existing strict predictions

```bash
cd /data/home/terminator/RL/multi-view

export RUN_ROOT=/data/home/terminator/RL/multi-view/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034
export PY_MM=/data/home/terminator/anaconda3/envs/multiview/bin/python

nohup bash run_fewshot_addhout_domain_adaptation.sh \
  > "$RUN_ROOT/logs/nohup_fewshot_addhout_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

## Important knobs

MAE-first:

```bash
export OPERATIONAL_BASE_COL=pred_superblend_mae_guarded
```

Trend-first:

```bash
export OPERATIONAL_BASE_COL=pred_superblend_trend
```

Balanced:

```bash
export OPERATIONAL_BASE_COL=pred_superblend_balanced
```

Use a fixed calibration set:

```bash
export CALIBRATION_IDS=CeO2-0-Ce,CeO2-3-Cr,ZnO-3-Cr,ZnO-19-Zr
```

Use 6 labels per material:

```bash
export OPERATIONAL_SHOTS_PER_MATERIAL=6
```

The default calibrator is `guarded_auto`. It checks calibration labels by leave-one-out and falls back to the original strict prediction if calibration is not trustworthy.

## Files to inspect

- `fewshot_holdout_summary.csv`: main few-shot paper table.
- `fewshot_recommended_by_holdout.csv`: best held-out configuration for each shot count.
- `fewshot_operational_predictions.csv`: calibrated prediction file.
- `fewshot_operational_audit.csv`: audit for all labeled / held-out / calibration rows.
- `fewshot_calibration_selection.csv`: selected AddH-out rows to label or use as calibration.
- `fewshot_manifest.json`: exact label usage and calibration parameters.
