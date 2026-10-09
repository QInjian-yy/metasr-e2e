# Meta-SR E2E

当前代码支持共享 Meta-RDN 特征上的 SPP + ABMIL，以及保留的 ResNet18-GN + ABMIL。
本次代码来源为 2026-10-08 完成修改的本地项目；同步整理日期为 2026-10-09。

| 内容 | 入口 |
|---|---|
| 模型、数据和运行说明 | [E2E_MetaSR/README.md](E2E_MetaSR/README.md) |
| 服务器运行步骤 | [SERVER_RUN.md](E2E_MetaSR/SERVER_RUN.md) |
| 实验自动记录与结果回传 | [experiments/README.md](E2E_MetaSR/experiments/README.md) |
| 实验补充说明模板（可选） | [template.md](E2E_MetaSR/experiments/template.md) |
| 正式训练结果 | [results/summary.csv](E2E_MetaSR/results/summary.csv) |
| 每周研究进展 | [weekly_reports/README.md](E2E_MetaSR/weekly_reports/README.md) |
| 首次同步代码验证 | [validation_20261009.md](E2E_MetaSR/experiments/validation_20261009.md) |

SPP 使用 1、2、3、6 四级 adaptive max pooling，将每个 region 的 64 通道特征变为
3200 维 embedding，再进入 gated ABMIL 和二分类层。SPP 自身没有可学习参数。
当前默认训练入口仍使用 ResNet18 的 baseline 配置；SPP-D8 请显式选择
[configs/spp_d8.yaml](E2E_MetaSR/configs/spp_d8.yaml)。

首次同步通过原有 24 个回归测试、新增 6 个 SPP 测试和 2 个结果汇总测试，
没有启动正式长训练。summary.csv 当前仅包含表头；轻量测试不能证明完整数据集上的
分类效果、收敛或大规模 GPU 显存需求。

日常流程：本地修改代码 → GitHub main → AutoDL 拉取并训练 → 结果记录回 GitHub。
训练命令保持不变，编号、GPU、Commit 自动记录；结束或早停自动生成实验文档和结果汇总。
无需手填汇总参数，详细步骤见上面的服务器运行说明。数据与权重保留在 AutoDL。

保留目录：

- [Meta-SR-Pytorch-0.4.0/](Meta-SR-Pytorch-0.4.0/)：原始官方参考代码和已有示例。
- [E2E_MetaSR/audit/](E2E_MetaSR/audit/)：已有审计文件。
- [E2E_MetaSR/reports/](E2E_MetaSR/reports/)：已有历史实验和验证记录。
- [E2E_MetaSR/tests/](E2E_MetaSR/tests/)：原有测试及本轮新增回归测试。

历史报告保留原始内容，不作为 SPP 新模型的正式训练结果。旧报告和 probe 中引用的
本地路径、配置或命令反映当时环境；复现前需核对对应提交。
运行所需的官方实现已内置于 E2E_MetaSR/vendor/official，无需依赖外部参考项目。
数据集、权重、原始训练输出、缓存和本地环境文件不纳入新增提交。
