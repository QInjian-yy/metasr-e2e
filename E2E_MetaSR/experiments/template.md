---
experiment_id: 待分配唯一实验编号
status: planned
started_at: 待填写
finished_at: 待填写
---

# 实验记录

## 实验编号与目的

- 实验编号：
- 目的、假设：
- 对照实验编号：
- 本次唯一改动；无法控制的差异：

## 模型结构与配置

- 分类分支：SPP / ResNet18-GN
- RDN 深度 D、checkpoint_rdb、scale、rgb_range：
- SPP levels 与 embedding 维度，或 ResNet18 配置：
- ABMIL、分类层：
- 配置文件与 SHA256；保存的 runs/<编号>/config.yaml：
- lambda_sr、SR crop、region microbatch、precision：
- optimizer、lr、weight_decay、epochs、early_stopping_patience：

## 数据划分与随机性

- 数据集名称、版本和外部 DATA_ROOT：
- manifest、标签文件、split 文件位置与 SHA256：
- fold；train/val WSI 数；region 数范围：
- 随机种子：
- 数据预处理、增强、LR/HR 配对检查：

## 代码与环境

- 训练开始时完整 Git Commit：
- 工作树是否干净；额外本地修改：
- GPU 型号、数量、显存：
- OS、Python、PyTorch、torchvision、CUDA：
- 训练命令：

## 日志与输出

- 输出目录：runs/<实验编号>/
- 训练日志：logs/<实验编号>.log
- history.csv、config.yaml、data_provenance.json：
- best.pth、last.pth 所在外部或忽略目录：
- 完成状态、实际 epochs、停止原因、耗时、显存：

## 指标

仅填真实训练结果。下面各指标使用最高 val_auc 对应的同一个 epoch；
并列取首次，未运行或缺失项留空。

| 指标 | 值 |
|---|---|
| Best epoch | |
| Train Loss CE / SR / Total | |
| Train AUC / ACC / BACC | |
| Validation Loss CE | |
| Validation AUC / ACC / BACC | |
| Independent Test AUC / ACC / BACC（若实际执行） | |

## 实验结论

- 观察到的结果与证据：
- 相对匹配对照的变化：
- 失败、异常及证据边界：
- 本次能支持的结论：
- 下一步：

正式训练结束后可使用 summarize.py 追加汇总。同一编号不能再次写入；
需要重跑时创建新编号，原始记录、日志与输出保留。
