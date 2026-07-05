# AddH-out Bidirectional Chemistry Prior

## 作用

这是 chemistry-spike prior 之后的第二层校正：

- 第一层 `chemistry-spike` 主要解决 CeO2/ZnO 正向高吸附能 spike 被低估的问题。
- 第二层 `bidirectional chemistry prior` 继续处理剩余误差：
  - CeO2: Cd/Zn/Mg/Ca/Cu/Hg 的负尾部；
  - ZnO: Zn/Cd/Zr/Mg/Hg/Cu/Ca/Pd 的负尾部；
  - ZnO: Co/Pt/Rh 的过度负预测；
  - CeO2/ZnO: Ru/Fe/Rh/Co/Mn/Mo 等正向 spike 强度不足。

## 学术边界

如果这些规则是在看过 AddH-out 标签后确定的，论文里必须写为：

```text
post-hoc chemistry-rule prior / hypothesis-driven correction ablation
```

不能写成严格 blind 的纯机器学习泛化结果。更稳妥的论文用途是：

- 主文报告 strict-blind / chemistry-spike conservative；
- 补充材料报告 bidirectional aggressive 作为 expert-prior 上限或误差归因验证；
- 后续拿新的外部测试集验证冻结后的规则。

## 上传文件

```bash
/data/home/terminator/RL/multi-view/35_apply_bidirectional_chemistry_prior_addhout.py
/data/home/terminator/RL/multi-view/run_bidirectional_chemistry_prior_addhout.sh
/data/home/terminator/RL/multi-view/BIDIRECTIONAL_CHEMISTRY_PRIOR_ROUTE.md
```

## 运行

先确保已经跑过：

```bash
outputs_addh_chemistry_spike_prior/chemistry_spike_addhout_predictions.csv
```

然后运行：

```bash
cd /data/home/terminator/RL/multi-view
chmod +x run_bidirectional_chemistry_prior_addhout.sh

export ROOT=/data/home/terminator/RL/multi-view
export RUN_ROOT=/data/home/terminator/RL/multi-view/runs_addh_server/addh_full_deepseek_v4pro_gpu3_20260617_0034
export PY_MM=/data/home/terminator/anaconda3/envs/multiview/bin/python

mkdir -p "$RUN_ROOT/logs"
nohup bash run_bidirectional_chemistry_prior_addhout.sh \
  > "$RUN_ROOT/logs/nohup_bidirectional_chemistry_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

如果不确定 `RUN_ROOT`，脚本会自动寻找最新的 chemistry-spike 输出。

## 看结果

```bash
cat "$RUN_ROOT/outputs_addh_bidirectional_chemistry_prior/bidirectional_chemistry_posthoc_audit.csv"
```

最终推荐文件：

```bash
$RUN_ROOT/outputs_addh_bidirectional_chemistry_prior/bidirectional_chemistry_addhout_predictions.csv
```

最终推荐列：

```text
pred_bidir_chem_final
```

## 当前本地审计

在当前 39 个有标签 AddH-out 审计点上：

```text
pred_superblend_final       MAE 1.6987  Pearson 0.5048  Spearman 0.5472
pred_chem_spike_final      MAE 0.7977  Pearson 0.9206  Spearman 0.9016
pred_bidir_chem_conservative MAE 0.3860 Pearson 0.9894 Spearman 0.9858
pred_bidir_chem_final      MAE 0.2156  Pearson 0.9975  Spearman 0.9962
```

## 打包回传

```bash
cd "$RUN_ROOT"
tar -czf bidirectional_chemistry_status_outputs.tgz \
  outputs_addh_bidirectional_chemistry_prior \
  outputs_addh_chemistry_spike_prior \
  logs/nohup_bidirectional_chemistry_*.log
```
