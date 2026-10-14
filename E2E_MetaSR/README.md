# Meta-SR E2E：SPP / ResNet18 + ABMIL

独立的 CAMELYON16 E2E 训练项目。运行所需的官方 Meta-RDN 源码已放在
`vendor/official/`，无需额外上传 `Meta-SR-Pytorch-0.4.0/` 或原 `E2E/`。

历史审计输出、调试脚本、测试文件和 Python 缓存已清理。训练启动仍会调用
`validation.py`，通过 AST 读取 `reference/trainer.py` 的参考坐标函数；
这两个文件属于当前运行依赖，必须保留。

## 配置与模型

| 配置 | 分类分支 | RDB 数 D |
|---|---|---:|
| `configs/spp_d8.yaml` | SPP，较小的 RDN 配置 | 8 |
| `configs/spp_d8_mlp.yaml` | SPP，ABMIL 后接 128 维 MLP 分类头 | 8 |
| `configs/spp.yaml` | SPP | 16 |
| `configs/baseline.yaml` | ResNet18-GN | 16 |
| `configs/rdn_d8.yaml` | ResNet18-GN | 8 |
| `configs/rdn_d4.yaml` | ResNet18-GN | 4 |

训练入口默认仍使用 `baseline.yaml`；运行较小的 SPP 模型时明确传
`--config configs/spp_d8.yaml`。D8 只减少 RDB 数量，每个 RDB 仍有 8 层 dense conv，
growth=64，输出保持 64 通道；RDN 的 GFF 输入随 D 调整。

SPP 分类路径：

```text
一张 WSI 的全部 LR regions [N,3,256,256]
  → 共享 RDN 完整 feature [N,64,256,256]
  → adaptive max pooling：1×1、2×2、3×3、6×6
  → 各分支 flatten 后依次拼接 [N,3200]
  → GatedAttentionMIL(3200,128) → [1,3200]
  → Linear(3200,2) → logits [1,2]
```

顶层配置 `classification_head` 默认为 `linear`，保留原分类头及旧权重键名。
SPP 可选择 `classification_head: mlp`，仅将 ABMIL 后的分类头改为
`Linear(3200,128) → ReLU → nn.Dropout(0.3) → Linear(128,2)`。
该选项只支持 `classification_encoder: spp`，不改变 RDN、SPP、ABMIL 或 SR 分支。
新配置 `configs/spp_d8_mlp.yaml` 沿用已记录的 D8 / lr=1e-4 / WD=1e-4 /
SR=0 / MeanShift 冻结 / BF16 / 整批设置，仅新增 MLP 分类头；运行时指定新的输出目录。

每次 `train_wsi` 都调用 `model.train()`，开启 Dropout；轮末训练集评估、验证集评估
以及 `eval_e2e.py` 独立评估共用 `model.eval()` 和 `torch.no_grad()`，关闭 Dropout 且不更新参数。
下一轮首个训练步骤自动恢复训练模式。自行加载 checkpoint 推理时，同样需要
先调用 `model.eval()`，并在 `torch.no_grad()` 中执行前向。

SPP 没有可学习参数，没有 ResNet 实例、额外投影、归一化或激活。
SR 分支使用同一次编码得到的完整 feature，以绝对 HR 坐标查询每个 region 的 HR crop。
scale=32，LR/HR 图像为 RGB [0,1]。完整 8K SR 推理复用完整 feature，在 no_grad 下逐块解码并写入 memmap。

`metasr.freeze_meanshift` 默认为 `false`，保留旧实验中 MeanShift 参数可训练的行为。
在新配置的 `metasr` 下设置 `freeze_meanshift: true`，会冻结 `sub_mean` 和 `add_mean`
的 weight/bias；不改官方代码、网络结构或前向计算。相同权重下输出不变，后续训练更新会不同。
开关随配置保存在 checkpoint 中，评估和 SR 推理加载时自动恢复；旧 checkpoint 缺少该字段时仍默认关闭。

## 可选配对几何增强

增强默认关闭。可在主 YAML 中添加以下字段，或在原训练命令上增加
`--augmentation-config configs/augmentation_geometry.yaml`，使用独立配置覆盖增强设置：

```yaml
augmentation:
  enabled: true
  horizontal_flip: 0.5
  vertical_flip: 0.5
  rotation90: true
```

每次读取训练 WSI 时，对每个 LR/HR region 对独立抽取一次变换：先水平翻转，
再垂直翻转，最后从 0°/90°/180°/270° 中等概率抽取逆时针旋转。没有颜色增强。
参数固定在路径包装中，不随 region micro-batch 划分改变；关闭开关时不额外抽取随机数。
独立配置只覆盖 `augmentation`，不修改分类头、RDN 深度、损失系数或优化器参数；
实际增强设置会写入运行目录的 `config.yaml` 和 checkpoint。

LR 在 CPU 上完成几何变换。SR crop 的 `(top,left,height,width)` 定义在增强后的
8192×8192 图像上；`paired_geometry.py` 将其逆映射为原始 HR 的裁剪区域，先裁剪，
再对小 crop 应用相同变换，保证非对齐坐标及边界逐像素对应。HR 图像的文件解码仍在 CPU，
只将 crop 张量传入 GPU；分类与 SR 共用同一次增强后的 LR 编码，SR=0 时不读取 HR。
轮末训练集评估使用 `evaluation_view()`；验证集、`eval_e2e.py` 和 `infer_sr.py` 均不增强。
旧配置未提供 `augmentation` 时保留原路径，不改变网络参数或旧 checkpoint 加载方式。

## 数据与三折划分

`--data-root` 指向真实数据根目录：

```text
DATA_ROOT/
  images_256/
  images_8192/
  manifests/
    patch_manifest.csv
```

manifest 使用 `filename,slide_id` 配对 LR/HR，同名图像放在对应图片目录。
HR 保留原始 8192×8192 图像，训练时读取随机 crop；不要提前缩小 HR。
`lambda_sr=0` 时不需要 HR 文件，也不加载 HR 或运行 SR 解码。

标签和三折划分保留在 `downstream_train/`：
`camelyon16_labels.csv` 与 `splits_0.csv`、`splits_1.csv`、`splits_2.csv`。
项目内 `manifests/` 保留为数据元信息，实际 loader 读取的是 DATA_ROOT 下的 manifest。
使用自己的已划分数据时，可通过 `--labels-csv` 和 `--split-dir` 指定原文件。
标签列为 `case_id,slide_id,label`；划分列为 `train,val`，内容是 case_id。
每次 `--fold 0/1/2` 只运行所选的一折，不重新划分。

## 运行

安装匹配的 CUDA PyTorch/torchvision，其余依赖见 `requirements.txt`。
从本项目目录执行，详细服务器步骤见 `SERVER_RUN.md`：

```bash
python -u -B train_e2e.py --config configs/spp_d8.yaml \
  --data-root /path/to/full_data --fold 0 --output runs/spp_d8_full_fold0

python -B eval_e2e.py --checkpoint runs/spp_d8_full_fold0/best.pth \
  --data-root /path/to/full_data

python -B infer_sr.py --checkpoint runs/spp_d8_full_fold0/best.pth \
  --lr-image /path/to/region.png --output runs/spp_d8_sr.npy
```

如果训练时显式指定标签和划分，评估时也应指定相同的 `--labels-csv`、`--split-dir`。
训练启动会按所选 D 执行小尺寸 SR 等价性与 checkpoint 校验，失败则停止。
输出目录必须为空或不存在；当前入口不支持断点续训。

## 整批、分批与损失

SPP 配置默认整批：

```yaml
training:
  use_region_microbatch: false
  region_microbatch_size: 1
```

整批时 size 不生效，同一 WSI 的全部 N 个 region 一次编码。
需要分批时，复制配置为新文件，只修改：

```yaml
training:
  use_region_microbatch: true
  region_microbatch_size: 2
```

两种模式都汇总全部 N 个 embedding 后执行一次 ABMIL，每张 WSI 一次 backward、
一次 Adam.step。分批顺序编码并保留各 chunk 的计算图，没有 GradPool，
不能把全部激活显存简单视为按 chunk_size/N 缩减。OOM 会报错，不自动分批或截断 region。
BF16 整批与分批可能存在数值差异，不能直接认定为等价。

总损失为一张 WSI 的分类 CE 加上
`lambda_sr × 所有 region 的平均 crop L1`。
每个 region 独立采样一个 HR crop，坐标在 region 分批前生成。
`lambda_sr=0` 时分类梯度仍回传 RDN，P2W/add_mean 没有 SR 梯度。

## 日志与 checkpoint

每张 WSI 打印分类/SR/总损失与 CUDA allocated/reserved 峰值；
每 10 张 WSI 打印参数梯度范数。SPP 本身无参数，`region_encoder: 0.0` 属于正常行为。

`history.csv` 保存每轮在线分类/SR/总损失、离线 train/val AUC、ACC、BACC、
验证分类 loss 和训练 step 峰值显存。显存单位为字节，除以 2**30 得到 GiB。
新运行额外记录 `train_eval_loss`：轮末固定模型、Dropout 关闭时的训练集分类 CE，
与 `val_loss` 使用同一评估计算方式。原 `train_loss_cls/sr/total` 仍是训练步骤的在线均值
（MLP 头训练时 Dropout 开启）；原指标含义不变，旧 CSV 不会改写。
输出还包括 `config.yaml`、`data_provenance.json`、`equivalence.json`、`run_info.json`、
`last.pth`、`best.pth`、`best_val_predictions.csv`。
最优 checkpoint 按 val_auc 选择；`spp_d8.yaml` 和 `spp_d8_mlp.yaml` 最多 100 epochs、patience=15，其他现有配置仍为 20/5。

新 checkpoint 格式 `metasr-abmil-v2` 明确保存分类结构与维度；
评估和 SR 推理据此构造模型并严格加载。旧 `metasr-abmil-v1` 明确使用 ResNet18。
MLP checkpoint 在配置中记录分类头类型，并在 architecture 中记录类型、隐藏维度及 Dropout 概率；
旧 checkpoint 缺少分类头选项时按 `linear` 加载，仍保留严格的 state_dict 检查。
未知结构、配置冲突、缺失键或尺寸不符均报错。
D16/D8/D4 的 RDB/GFF 权重形状不同，不能跨深度直接严格加载整个模型。
当前没有旧权重迁移入口；新 SPP 实验从随机初始化建立模型与 Adam。

## 实验与研究进展

- [实验索引与编号规则](experiments/README.md)
- [统一实验模板](experiments/template.md)
- [正式训练结果汇总](results/summary.csv)
- [每周研究进展](weekly_reports/README.md)
- [2026-10-09 代码同步验证](experiments/validation_20261009.md)

按原来的命令训练，使用新的输出目录。实验编号、GPU、Git Commit 自动记录；
正常结束或早停自动追加 results/summary.csv，并生成 experiments/EXP-*.md。
无需手填汇总参数。可选补汇总只需 `python -B experiments/summarize.py <输出目录>`。
历史实验和已有结论不会被覆盖；当前尚无正式训练结果。
Git 仓库副本继续保留已有审计、测试和历史报告，旧记录不作为新 SPP 的正式结果。
