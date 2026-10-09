# 实验管理

正式训练与代码验证分别记录。每次正式训练使用新的实验编号，
例如 `EXP-20261009-SPP-D8-F0-S1-001`，格式为
`EXP-YYYYMMDD-MODEL-Dn-Fn-Sn-NNN`。日期、模型、深度、fold、seed 和序号应对应实际运行；
示例编号尚未执行，不代表已有结果。同一配置重跑或另一折必须使用新的编号。

| 模型方案 | 配置 | 分类 embedding | 状态 |
|---|---|---:|---|
| ResNet18-GN / D16 | [baseline.yaml](../configs/baseline.yaml) | 512 | 保留的基线配置；本次未正式训练 |
| SPP / D16 | [spp.yaml](../configs/spp.yaml) | 3200 | 本地已有配置；本次未正式训练 |
| SPP / D8 | [spp_d8.yaml](../configs/spp_d8.yaml) | 3200 | 本地已有配置；本次未正式训练 |

本轮验证见 [validation_20261009.md](validation_20261009.md)，正式结果见
[summary.csv](../results/summary.csv)，周报见 [weekly_reports](../weekly_reports/README.md)。
Git 仓库中的已有 audit、reports、tests 和官方参考代码均保留；旧报告不改写为新 SPP 结果。

## 运行记录

1. 分配尚未使用的实验编号，把 [template.md](template.md) 复制为该编号的 Markdown 文件。
2. 开始训练前记录完整 Git commit、配置、数据与 split 的哈希、fold、seed 和 GPU。
3. 输出目录使用 `runs/<实验编号>/`，日志使用 `logs/<实验编号>.log`。
   不复用旧编号、旧输出目录或旧日志；训练入口已有非空输出目录保护。
4. 结束或中断后保留原始 history.csv、config.yaml、data_provenance.json、日志和 checkpoint。
   权重与大体积输出存放在项目外或被忽略的 runs/ 中，实验文档只记录位置。
5. 填写结论后串行执行下面的汇总命令，追加一条正式训练结果。

```bash
python -B experiments/summarize.py \
  --experiment-id EXP-20261009-SPP-D8-F0-S1-001 \
  --run-dir runs/EXP-20261009-SPP-D8-F0-S1-001 \
  --fold 0 --git-commit <训练开始时的完整40位Commit> \
  --gpu "<实际GPU型号>" --status completed \
  --log-file logs/EXP-20261009-SPP-D8-F0-S1-001.log \
  --record-file experiments/EXP-20261009-SPP-D8-F0-S1-001.md
```

根据实际状态选择 completed、early_stopped 或 interrupted；汇总时间不代表训练开始时间。
脚本只读取保存的配置、history.csv 并追加 summary.csv，不加载权重、不执行训练，
不修改原始日志。重复实验编号会报错，历史结果不会被覆盖。
没有有效 Validation AUC 时拒绝汇总，在实验记录中说明失败原因。

汇总选择 **最高 val_auc 所在 epoch**；并列时选择首次达到该值的 epoch，与训练 checkpoint
选择规则一致。Train Loss、Train AUC/ACC 和 Validation 指标均来自这个 epoch，
不会把各列分别取最优。train_loss_cls 是在线每 WSI 的 CE 平均；
train_loss_sr 是平均 crop L1；train_loss_total 是两者按 lambda_sr 加权的总损失。
train_auc/acc/bacc 是离线训练集分类评估，val_loss 只含分类 CE。
缺失训练结果保持空白，不填 0，不以 smoke test 代替正式实验。

## 对照与消融

结构对比先使用 D16 ResNet18 与 D16 SPP，保持数据划分、初始化规则、seed、精度、
训练预算和 SR 设置一致，再单独改变 D 或 lambda_sr。
现有 spp_d8 配置为 100 epochs / patience 15，其他现有配置为 20 / 5；
本次不修改这些配置，未匹配预算的结果不能直接归因于 SPP。
