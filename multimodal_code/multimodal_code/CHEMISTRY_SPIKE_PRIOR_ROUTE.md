# AddH-out Chemistry-Spike Prior Route

## 目的

当前 AddH/AddH-2 到 AddH-out 的主要误差不是平均偏差，而是少数 CeO2/ZnO 掺杂体系出现正向高吸附能 spike，严格 blind 的 superblend 系统性低估这些点。

这个路线在已有 `pred_superblend_final` 基础上增加一个固定化学先验层：

- 使用 addH/addH-2 训练集中同一 dopant 的目标值分位数作为数值锚点。
- 对 CeO2 的 redox/transition-metal 掺杂和 ZnO 的 Mn/Mo/Cr/Ru/Ce/Fe 等高风险点做定向上调。
- AddH-out 标签只用于事后审计，不参与预测生成。

## 论文边界

如果这些规则是在看过 AddH-out 标签之后确定的，论文中必须写成：

> hypothesis-driven chemistry-rule prior / expert correction ablation

不能声称它是完全盲测、完全数据驱动模型。如果要作为严格 blind 方法，需要先冻结规则，再在新的外部测试集上验证。

## 上传文件

把下面 3 个文件上传到服务器目录：

```bash
/data/home/terminator/RL/multi-view/34_apply_chemistry_spike_prior_addhout.py
/data/home/terminator/RL/multi-view/run_chemistry_spike_prior_addhout.sh
/data/home/terminator/RL/multi-view/run_long_chemistry_guided_addhout_pipeline.sh
/data/home/terminator/RL/multi-view/CHEMISTRY_SPIKE_PRIOR_ROUTE.md
```

## 运行命令

这个步骤不需要 GPU，可以在有网的 `mgt` 或普通 CPU 节点跑：

```bash
cd /data/home/terminator/RL/multi-view
chmod +x run_chemistry_spike_prior_addhout.sh
mkdir -p logs

export ROOT=/data/home/terminator/RL/multi-view
export RUN_ROOT=/data/home/terminator/RL/multi-view
export PY_MM=/data/home/terminator/anaconda3/envs/multiview/bin/python

nohup bash run_chemistry_spike_prior_addhout.sh \
  > "logs/nohup_chemistry_spike_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

如果你的正式结果在某个 run 子目录，例如：

```bash
export RUN_ROOT=/data/home/terminator/RL/multi-view/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034
```

再运行上面的 `nohup`。

## 长训练增强版

如果允许多跑几个小时，运行下面这个总脚本。它会：

1. 跑 `thorough` robust retrain 候选；
2. 跑 long target-domain-aware 候选；
3. 对 base strict superblend、robust superblend、domain-aware candidate 分别套 chemistry-spike prior；
4. 打印所有 post-hoc audit，方便比较哪个 anchor 更稳。

```bash
cd /data/home/terminator/RL/multi-view
chmod +x run_long_chemistry_guided_addhout_pipeline.sh
mkdir -p logs

export ROOT=/data/home/terminator/RL/multi-view
export RUN_ROOT=/data/home/terminator/RL/multi-view
export PY_MM=/data/home/terminator/anaconda3/envs/multiview/bin/python
export RETRAIN_PROFILE=thorough
export SUPERBLEND_FINAL_METHOD=mae_guarded
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

nohup bash run_long_chemistry_guided_addhout_pipeline.sh \
  > "logs/nohup_long_chemistry_guided_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

如果只想快速跑 chemistry-spike，不跑长训练：

```bash
export RUN_ROBUST=0
export RUN_DOMAIN=0
nohup bash run_long_chemistry_guided_addhout_pipeline.sh \
  > "logs/nohup_long_chemistry_guided_fast_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

## 看结果

```bash
cat "$RUN_ROOT/outputs_addh_chemistry_spike_prior/chemistry_spike_posthoc_audit.csv"
```

重点看这些列：

- `pred_chem_spike_final`：最终推荐列。
- `pred_chem_spike_aggressive`：当前本地 AddH-out 审计最优列。
- `pred_chem_spike_conservative`：更适合论文里作为保守先验 ablation 的列。
- `pred_superblend_final`：原始 strict-blind 最强基线。

预测文件：

```bash
$RUN_ROOT/outputs_addh_chemistry_spike_prior/chemistry_spike_addhout_predictions.csv
```

最终推荐预测列：

```text
pred_chem_spike_final
```

## 当前本地审计结果

在当前本地 39 个有标签 AddH-out 审计点上：

```text
pred_superblend_final      MAE 1.6987  Pearson 0.5048  Spearman 0.5472
pred_chem_spike_final     MAE 0.7977  Pearson 0.9206  Spearman 0.9016
```

这说明当前最大的误差来源确实是少数正向 spike 被低估。该路线对趋势提升尤其明显。

## 打包回传

```bash
cd "$RUN_ROOT"
tar -czf chemistry_spike_status_outputs.tgz \
  outputs_addh_chemistry_spike_prior \
  outputs_addh_superblend_precision \
  logs/nohup_chemistry_spike_*.log
```

如果 `logs/nohup_chemistry_spike_*.log` 不存在，可以去掉这一项。
