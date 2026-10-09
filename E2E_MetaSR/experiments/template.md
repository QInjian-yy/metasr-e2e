# 实验补充说明（可选）

训练结束会自动生成 experiments/EXP-*.md 和 results/summary.csv。
通常只需在自动记录的“实验目的”和“实验结论”中补充几句；需要更完整说明时使用此模板。
无需为自动汇总预先创建文件。

## 实验目的

- 实验编号：使用 run_info.json 中自动生成的编号。
- 目的、假设：
- 对照实验编号：
- 本次改动：

## 模型与运行信息

- 模型结构、训练配置：见输出目录 config.yaml；可补充结构差异。
- 数据划分、文件哈希：见 data_provenance.json；可补充数据集名称。
- 随机种子：见 config.yaml。
- Git Commit、GPU、fold、开始与结束时间、状态：见 run_info.json。
- 训练日志与指标：见 history.csv；终端日志如有则记录路径。
- 权重位置：输出目录中的 best.pth、last.pth，不提交到 GitHub。

## 指标

自动记录采用最高 Validation AUC 对应的同一个 epoch，并列取首次。
未运行的指标留空，不填 0。

| 指标 | 值 |
|---|---|
| Best epoch | |
| Train Loss CE / SR / Total | |
| Train AUC / ACC / BACC | |
| Validation Loss CE | |
| Validation AUC / ACC / BACC | |
| Independent Test AUC / ACC / BACC（若实际执行） | |

## 实验结论

- 观察到的结果：
- 相对对照的变化：
- 异常或局限：
- 下一步：
