# 实验管理

正常训练命令保持不变。实验编号、GPU、训练开始时的 Git Commit、fold、seed 和指标
由程序记录，不需要手填，也不需要先复制模板。

| 模型方案 | 配置 | 分类 embedding | 状态 |
|---|---|---:|---|
| ResNet18-GN / D16 | [baseline.yaml](../configs/baseline.yaml) | 512 | 保留的基线配置；尚无正式结果 |
| SPP / D16 | [spp.yaml](../configs/spp.yaml) | 3200 | 已有配置；尚无正式结果 |
| SPP / D8 | [spp_d8.yaml](../configs/spp_d8.yaml) | 3200 | 已有配置；尚无正式结果 |

首次同步验证见 [validation_20261009.md](validation_20261009.md)，正式结果见
[summary.csv](../results/summary.csv)，周报见 [weekly_reports](../weekly_reports/README.md)。
已有 audit、reports、tests 和官方参考代码保留；旧报告不改写为新 SPP 结果。

## AutoDL 上训练

从 GitHub 拉取代码后，在 E2E_MetaSR 目录运行：

```bash
python -u -B train_e2e.py --config configs/spp_d8.yaml \
  --data-root /path/to/full_data --fold 0 --output runs/spp_d8_full_fold0
```

每次使用新的 output 目录；原有非空目录保护仍有效。
启动后自动生成日期加随机后缀的唯一编号，写入 output/run_info.json。
正常结束或早停后，自动追加 results/summary.csv，并生成 experiments/EXP-*.md。
记录包含最高 Validation AUC 对应 epoch 的 Train Loss、AUC、ACC、BACC 等指标；
你只需按需补充实验目的、结论或周报。[template.md](template.md) 可用于扩展说明，无需逐项手填。

Commit 用于以后找到这次训练实际使用的代码，由 Git 自动读取。直接复制的无 Git 目录
或没有安装 Git 时，该项留空；旧运行缺失的 GPU、fold、状态等信息也保持空白。
训练中断时 run_info.json 保持 incomplete，不会记成完成。

## 补汇总旧运行或中断运行

正常训练结束不需要执行这一步。需要补汇总时只传输出目录：

```bash
python -B experiments/summarize.py runs/spp_d8_full_fold0
```

脚本读取保存的配置和 history.csv，不加载权重或启动训练。
相同编号重复汇总不会重复追加或覆盖已有文档、结论；重跑应使用新的输出目录和编号。
没有有效 Validation AUC 时不写汇总。终端日志为可选，自动识别 output/training.log
或 logs/<输出目录名>.log；原始逐轮指标始终在 output/history.csv。

汇总选择最高 val_auc 所在 epoch，并列取首次，与训练 checkpoint 规则一致。
所有指标来自同一个 epoch，不把各列分别取最优。
train_loss_cls 是在线每 WSI 的 CE 平均，train_loss_sr 是 crop L1 平均，
train_loss_total 是两者按 lambda_sr 加权的总损失。
train_auc/acc/bacc 是离线训练集分类评估，val_loss 只含分类 CE。
不填虚构结果，也不把轻量测试当作正式实验。

## 结果回 GitHub

代码从本地提交到 GitHub，再由 AutoDL 拉取；训练后在 AutoDL 仓库根目录执行：

```bash
git add E2E_MetaSR/results/summary.csv E2E_MetaSR/experiments/EXP-*.md
git commit -m "Record training results"
git push origin main
```

有周报时也可提交对应 weekly_reports/ 文件。回到本地 Git 工作副本后执行
git pull --ff-only 即可取回记录。训练程序只生成本地记录，不自动执行 Git 提交或推送。
数据、权重、runs/ 和 logs/ 继续留在 AutoDL；记录中的输出路径指向训练机器。

## 对照与消融

结构对比先使用 D16 ResNet18 与 D16 SPP，保持数据划分、初始化规则、seed、精度、
训练预算和 SR 设置一致，再单独改变 D 或 lambda_sr。
现有 spp_d8 配置为 100 epochs / patience 15，其他现有配置为 20 / 5；
未匹配预算的结果不能直接归因于 SPP。
