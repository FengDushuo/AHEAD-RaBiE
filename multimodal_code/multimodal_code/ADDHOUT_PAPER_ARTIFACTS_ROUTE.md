# AddH-out Paper Artifacts Route

## 当前论文主线

在没有新的外部 AddH-out-like 数据时，推荐主结果写成：

```text
target-domain few-shot calibration with repeated held-out AddH-out validation
```

不要写：

```text
strict blind external validation
```

## 推荐结果表述

主结果使用 `fewshot_calibrated` 的 held-out 指标：

```text
material-stratified random holdout, 500 repeats
MAE 0.388 +/- 0.087
RMSE 0.492 +/- 0.112
Pearson 0.981 +/- 0.011
Spearman 0.967 +/- 0.025
```

对照：

```text
strict superblend:     MAE 1.667 +/- 0.350
chemistry-spike prior: MAE 0.795 +/- 0.139
```

`pred_bidir_chem_final` 的 full-data MAE 0.216 只能写成：

```text
post-hoc upper bound, not held-out
```

## 服务器运行顺序

先确保已经有：

```text
outputs_addh_fewshot_holdout_validation/
outputs_addh_bidirectional_chemistry_prior/
outputs_addh_chemistry_spike_prior/
outputs_addh_llm_element_priors/addhout_audit_labels.csv
```

然后运行校准比例敏感性：

```bash
cd /data/home/terminator/RL/multi-view
chmod +x run_calibration_fraction_sensitivity_addhout.sh

export ROOT=/data/home/terminator/RL/multi-view
export RUN_ROOT=/data/home/terminator/RL/multi-view/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034
export PY_MM=/data/home/terminator/anaconda3/envs/multiview/bin/python
export N_REPEATS=500
export CALIBRATION_FRACS=0.10,0.20,0.30,0.50,0.65,0.70,0.80

mkdir -p "$RUN_ROOT/logs"
nohup bash run_calibration_fraction_sensitivity_addhout.sh \
  > "$RUN_ROOT/logs/nohup_calibration_fraction_sensitivity_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

生成论文表格和图：

```bash
cd /data/home/terminator/RL/multi-view
chmod +x run_make_addhout_paper_artifacts.sh

export ROOT=/data/home/terminator/RL/multi-view
export RUN_ROOT=/data/home/terminator/RL/multi-view/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034
export PY_MM=/data/home/terminator/anaconda3/envs/multiview/bin/python

nohup bash run_make_addhout_paper_artifacts.sh \
  > "$RUN_ROOT/logs/nohup_make_addhout_paper_artifacts_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

## 主要输出

论文主表：

```text
outputs_addh_paper_artifacts/tables/addhout_main_results.csv
outputs_addh_paper_artifacts/tables/addhout_main_results.md
```

校准比例表：

```text
outputs_addh_paper_artifacts/tables/addhout_calibration_fraction_summary.csv
outputs_addh_paper_artifacts/tables/addhout_calibration_fraction_paired_improvement.csv
```

图：

```text
outputs_addh_paper_artifacts/figures/parity_strict_superblend.png
outputs_addh_paper_artifacts/figures/parity_chemistry_spike.png
outputs_addh_paper_artifacts/figures/parity_fewshot_holdout_median.png
outputs_addh_paper_artifacts/figures/mae_distribution_repeated_holdout.png
outputs_addh_paper_artifacts/figures/calibration_fraction_vs_mae.png
outputs_addh_paper_artifacts/figures/dopant_wise_error.png
```

## 打包回传

```bash
cd "$RUN_ROOT"
tar -czf addhout_paper_artifacts_status_outputs.tgz \
  outputs_addh_paper_artifacts \
  outputs_addh_calibration_fraction_sensitivity \
  outputs_addh_fewshot_holdout_validation \
  logs/nohup_calibration_fraction_sensitivity_*.log \
  logs/nohup_make_addhout_paper_artifacts_*.log
```
