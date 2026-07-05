# Local Geometry MAE Route

This route is for MAE-first improvement. It adds local H/O/dopant geometry
features to both addH/addH-2 training rows and AddH-out prediction rows, then
reruns the current best lightweight delta-head and MAE-guarded superblend.

It does not rerun FAIR-Chem embedding extraction and does not use AddH-out
labels for training or model selection.

## Added Local Features

The extractor parses CONTCAR files directly and produces unified train/AddH-out
columns such as:

```text
geom_addh_h_o_min
geom_addh_h_dopant_min
geom_addh_h_host_min
geom_addh_h_minus_o_cart_z
geom_addh_h_nearest_o_metal_coord_3p0
geom_addh_dopant_o_min
geom_addh_dopant_o_coord_2p5
geom_bare_dopant_o_min
geom_bare_dopant_o_coord_2p5
geom_delta_dopant_o_min
geom_delta_dopant_o_coord_2p5
```

These columns are merged into:

```text
knowledge_features_train_geom.csv
knowledge_features_addhout_geom.csv
```

## Upload These Files

Upload to:

```bash
/data/home/terminator/RL/multi-view
```

Files:

```text
30_build_local_geometry_features.py
run_local_geometry_mae_addhout.sh
LOCAL_GEOMETRY_MAE_ROUTE.md
```

The route also uses existing scripts:

```text
24_build_pretrained_delta_features.py
25_train_pretrained_delta_head_addhout.py
26_rank_trend_calibrate_addhout.py
27_superblend_precision_addhout.py
```

## Run

```bash
cd /data/home/terminator/RL/multi-view
export PY_MM=/data/home/terminator/anaconda3/envs/multiview/bin/python
export RUN_ROOT=/data/home/terminator/RL/multi-view/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034

nohup bash run_local_geometry_mae_addhout.sh \
  > "$RUN_ROOT/logs/nohup_local_geometry_mae_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

Watch:

```bash
tail -f "$RUN_ROOT/logs/nohup_local_geometry_mae_"*.log
```

## What To Check

```bash
cat "$RUN_ROOT/outputs_addh_local_geometry_features/local_geometry_manifest.json"
cat "$RUN_ROOT/outputs_addh_geometry_pretrained_delta_head/pretrained_delta_head_posthoc_audit.csv"
cat "$RUN_ROOT/outputs_addh_geometry_rank_trend/rank_trend_posthoc_audit.csv"
cat "$RUN_ROOT/outputs_addh_geometry_mae_superblend/superblend_precision_posthoc_audit.csv"
cat "$RUN_ROOT/outputs_addh_geometry_mae_superblend/superblend_precision_manifest.json"
```

Final file:

```bash
$RUN_ROOT/outputs_addh_geometry_mae_superblend/superblend_precision_addhout_predictions.csv
```

Final column:

```text
pred_superblend_final
```

## Baseline To Beat

Current strict MAE-first best:

```text
MAE = 1.6987
Spearman = 0.5472
```

The local-geometry route is worth adopting only if:

```text
pred_superblend_final MAE < 1.6987
```

## Package Results

```bash
cd /data/home/terminator/RL/multi-view
export RUN_ROOT=/data/home/terminator/RL/multi-view/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034

tar -czf local_geometry_mae_status_outputs.tgz \
  "$RUN_ROOT/logs"/*local_geometry_mae* \
  "$RUN_ROOT/outputs_addh_local_geometry_features" \
  "$RUN_ROOT/outputs_addh_geometry_pretrained_delta_features" \
  "$RUN_ROOT/outputs_addh_geometry_pretrained_delta_head" \
  "$RUN_ROOT/outputs_addh_geometry_rank_trend" \
  "$RUN_ROOT/outputs_addh_geometry_mae_superblend" \
  "$RUN_ROOT/outputs_addh_superblend_precision" \
  "$RUN_ROOT/outputs_addh_target_calibrated_fast"
```

Put `local_geometry_mae_status_outputs.tgz` in the local workspace and I can
judge whether it improved the current best result.
