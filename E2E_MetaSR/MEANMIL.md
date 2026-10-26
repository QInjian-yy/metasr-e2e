# 可选 MeanMIL / ABMIL 独立版本

所有旧模型、配置、训练/评估/SR 推理入口保持原文件不变。
新入口 `train_e2e_mil.py` 接受现有 YAML，新增 `--mil-pooling meanmil|abmil`。
不指定 CLI 时读取 YAML 的 `mil_pooling`，YAML 也未指定时默认 `abmil`。

## 聚合含义

同一 WSI 的所有 Region 编码后先拼成 `[N,D]`，再聚合：

- `abmil`：沿用现有 GatedAttentionMIL。
- `meanmil`：`embeddings.mean(dim=0, keepdim=True)` 得到 `[1,D]`，然后使用原分类头。

MeanMIL 没有可训练参数；没有新增投影、归一化、激活或 Loss。
这里是 Region 之间的平均，不是将 Region 内的 Max SPP 改为 Avg SPP。
现有 SPP、SPP-LN、SPP-Avg、SPP-1344、ResNet18-GN、Region Gate 和已有合法分类头组合均可用。
RDN、SR、增强、数据划分、microbatch、每张 WSI 一次 backward/Adam、训练计划均由原 YAML 决定。
整袋平均不会先对各 microbatch 求均值再平均，避免最后一个小 batch 权重错误。

## 训练

在 `E2E_MetaSR` 目录，沿用你正在比较的配置、fold、data-root、labels/split 参数，仅改变聚合选项和输出目录。
例如沿用当前 Region Gate / SR0.1 / micro8 / 300轮配置：

```bash
python -B train_e2e_mil.py \
  --config configs/native512_d1_region_gate_sr01_noaug_m8_300.yaml \
  --mil-pooling meanmil \
  --data-root /path/to/full_data --fold 0 \
  --output runs/native512_d1_region_gate_meanmil_sr01_noaug_m8_300
```

要跑同条件 ABMIL，把上面 `--mil-pooling` 改为 `abmil`，并使用另一个新输出目录。
原 Max SPP 或其他版本直接换成其原有 YAML，不必改写原文件。
输出目录非空时报错；当前没有 resume。

两种聚合先按旧实现初始化共享模块，再将 MeanMIL 分支的聚合头替换为无参数均值。
因此相同 seed 下共享参数初值和模型初始化后的 Torch RNG 与对应旧 ABMIL 相同；
MeanMIL 分支不保留、不优化被替换的 ABMIL 参数。

## 评估与 SR 推理

```bash
python -B eval_e2e_mil.py \
  --checkpoint runs/native512_d1_region_gate_meanmil_sr01_noaug_m8_300/best.pth \
  --data-root /path/to/full_data \
  --predictions-csv runs/native512_d1_region_gate_meanmil_sr01_noaug_m8_300/eval_predictions.csv

python -B infer_sr_mil.py \
  --checkpoint runs/native512_d1_region_gate_meanmil_sr01_noaug_m8_300/best.pth \
  --lr-image /path/to/region.png --output /path/to/new_region_sr.npy
```

若训练显式指定了 `--labels-csv`、`--split-dir`，评估也指定对应路径。
新入口保存 `metasr-mil-v1` checkpoint，配置和 architecture 均记录 `mil_pooling`，严格检查结构和权重。
新评估/推理入口也兼容旧 ABMIL v1/v2 与 Region Gate checkpoint。
旧入口继续只负责原格式；新格式使用上述新入口。
评估保留数据来源哈希校验；预测和 SR 输出已有文件时拒绝覆盖。

## 日志与验证范围

history、best/last 权重、逐WSI验证预测、config、data_provenance、equivalence、run_info 均保存到新运行目录。
当前旧 `results/summary.csv` 没有聚合类型字段，因此新入口不自动向该旧表追加，也不自动生成旧格式实验卡，防止 MeanMIL 与 ABMIL 混记。
共享 engine 的梯度日志保留旧键 `abmil`；MeanMIL 对应值为0，因为均值头没有参数，分类梯度仍会回传 RDN 和可训练 Region 编码器。

```bash
python -B -m unittest discover -s tests -v
```

测试包含 N=1/5/35 均值及梯度、共享初始化/RNG、全部现有编码器、旧 checkpoint 兼容、
新 checkpoint 严格恢复、SR开/关、完整袋与不等长 microbatch、Adam更新及独立入口的小尺寸合成检查。
这些检查不代表真实512输入的显存、完整8K重建质量或分类效果已验证。

2026-10-26 本机验证：全套110项测试通过（原103项加新增7项），包括CUDA上scale16/32的FP32和BF16联合反向与Adam更新。
新增前后原有103个Python/YAML文件的SHA-256全部一致；未启动正式数据训练。
