# AddH-out Few-shot Holdout Paper Route

## 核心判断

如果只有当前 AddH-out 这一批数据，就不能声称完成了新的外部验证。更稳妥、可以写进论文主结果的路线是：

```text
target-domain few-shot calibration with repeated held-out AddH-out validation
```

也就是每次只使用一部分 AddH-out 标签做校准，剩余 AddH-out 标签完全不参与校准，只用于测试。

## 推荐主结果

主结果不要用 full-data post-hoc 的 `pred_bidir_chem_final = MAE 0.2156`。

推荐主结果用：

```text
fewshot_calibrated
```

当前本地 500 次 material-stratified repeated holdout：

```text
fewshot_calibrated:    MAE 0.3877 +/- 0.0874, Pearson 0.9815, Spearman 0.9673
chemistry-spike base:  MAE 0.7947 +/- 0.1386, Pearson 0.9173, Spearman 0.8865
strict superblend:     MAE 1.6667 +/- 0.3504, Pearson 0.5312, Spearman 0.5464
```

Leave-one-dopant-out:

```text
fewshot_calibrated:    MAE 0.3902
chemistry-spike base:  MAE 0.8068
strict superblend:     MAE 1.6852
```

## 论文写法

可以写：

```text
Because no independent AddH-out-like external set was available, we evaluated
target-domain transfer by repeated held-out validation within AddH-out. In each
split, a subset of AddH-out labels was used only to calibrate chemistry-regime
residual priors, and the remaining AddH-out samples were held out for testing.
```

不要写：

```text
external blind validation on AddH-out
```

## full-data bidirectional 结果怎么用

`pred_bidir_chem_final` 的 MAE 约 0.2156，只能作为：

- post-hoc upper bound；
- chemistry-rule error attribution；
- hypothesis generator；
- frozen-rule 后续外部验证的候选方案。

不要把它当作 strict blind 主结果。

## 服务器运行

```bash
cd /data/home/terminator/RL/multi-view
chmod +x run_fewshot_holdout_validation_addhout.sh

export ROOT=/data/home/terminator/RL/multi-view
export RUN_ROOT=/data/home/terminator/RL/multi-view/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034
export PY_MM=/data/home/terminator/anaconda3/envs/multiview/bin/python
export N_REPEATS=500
export TEST_FRAC=0.35

mkdir -p "$RUN_ROOT/logs"
nohup bash run_fewshot_holdout_validation_addhout.sh \
  > "$RUN_ROOT/logs/nohup_fewshot_holdout_validation_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

## 看结果

```bash
cat "$RUN_ROOT/outputs_addh_fewshot_holdout_validation/fewshot_holdout_summary.csv"
cat "$RUN_ROOT/outputs_addh_fewshot_holdout_validation/full_data_reference_metrics.csv"
```

## 打包

```bash
cd "$RUN_ROOT"
tar -czf fewshot_holdout_validation_status_outputs.tgz \
  outputs_addh_fewshot_holdout_validation \
  outputs_addh_bidirectional_chemistry_prior \
  outputs_addh_chemistry_spike_prior \
  logs/nohup_fewshot_holdout_validation_*.log
```
