# Official Meta-RDN + Meta-SR + ResNet18-GN + ABMIL

独立的 CAMELYON16 E2E baseline。模型数学的唯一来源是本地
`Meta-SR-Pytorch-0.4.0/model/metardn.py`；数据、split 和下游工程来自本地 `E2E/`。
这两个原目录不参与运行时导入，也不需要修改。根据后续要求，ResNet18 使用
E2E 已有替换函数把 20 个 BatchNorm 全部改为 GroupNorm；gated ABMIL 未改。

## 模型与来源

- `vendor/official/metardn.py`：完整复制官方文件，仅把 `common` 改为相对导入。
  RDB、SFENet1/2、GFF、Pos2Weight 和 MeanShift 均保持官方实现及参数行为。
- `vendor/official/common.py`：官方原文件逐字节复制。
- `reference/trainer.py`、`reference/option.py`：官方参考文件；测试从 trainer 的 AST
  提取原样的 `input_matrix_wpn`，避免导入旧 trainer 的依赖与参数解析器。
- `models/metasr.py`：继承官方 MetaRDN，提取同一 feature 路径，只在各 RDB 外加 checkpoint，
  用整数 scale 的 offset 复用和 chunk 改变 Meta-Upscale 的计算顺序。
  可用 `rdn_blocks` 减少 RDB 数量，并同步调整 GFF 的输入通道。
- `models/abmil.py`：E2E 的 `GatedAttentionMIL` 类原样提取。
- `models/baseline.py`：E2E 相同的 64 通道 ResNet18、512 维 embedding、ABMIL、Linear(512,2)，
  按用户要求使用原工程的 BN→GN 函数。
- `wsi_data.py`、`augmentation.py`、`downstream_train/*.csv`、`manifests/*.csv`：复制 E2E。
  loader 仅新增 `require_hr`，使 SR OFF 和分类验证无需 HR 文件。训练入口不启用 augmentation。
  CAMELYON16 case split 只读取 `splits_<fold>.csv`，忽略 manifest 的旧 split 字段。
- `reference/provenance.json` 与 `reports/source_snapshot.json`：复制来源及原目录 SHA-256。

默认官方 RDN-B：G0=64、kernel=3、D=16、C=8、growth=64。
另提供 `configs/rdn_d8.yaml`：仅将 D 改为 8，属于 RDN-B-D8 实验变体。
Pos2Weight 始终为 `Linear(3,256) → ReLU → Linear(256,1728)`。
输入/GT 沿用 E2E 的 RGB `[0,1]`，所以官方可配置参数 `rgb_range=1`，输入不额外标准化。
保留官方 MeanShift 的原始参数行为，包括源文件的 `self.requires_grad = False` 写法；
没有擅自改为 `requires_grad_(False)`，两者在当前 PyTorch 中并不等价。

## 张量流程

| 步骤 | 形状 |
|---|---|
| 一个 WSI 的所有 LR regions | `[N,3,256,256]` |
| sub_mean → SFENet1/2 → D 个 RDB → GFF + shallow residual（默认 D=16） | `[N,64,256,256]` |
| ResNet18-GN → 完整 bag | `[N,512]` |
| 原 gated ABMIL → classifier | `[1,512] → [1,2]` |
| SR 分支一次 unfold（逐个 region 解码独立 crop） | `[1,576,65536]` |
| ×32 的 position / weight table | `[1024,3] → [1024,1728] → [1024,576,3]` |
| 训练时每个完整 LR chunk 的输出 | `[1,L,1024,3]`，`L ≤ lr_chunk_size` |
| 默认连续 HR crop 监督（逐个 region） | prediction / GT 均为 `[1,3,256,256]` |
| 显式 full inference | 流式 tiles，逻辑输出 `[N,3,8192,8192]` |

N 是 WSI region 数；`training.use_region_microbatch=false` 时 B=N，直接一次编码全部 regions；
设为 `true` 时 B=`region_microbatch_size`。两种模式下所有 N 个 embedding 都进入一次 ABMIL。
两条分支使用同一次 RDN forward 的同一 feature。一次 WSI 执行一次联合 backward 和 Adam step。
没有 GradPool、BN replay、手工 RNG restore、通信模块、Frequency Router、WiKG 或额外 attention。

## 仅涉及内存的优化

1. 可选 `checkpoint_rdb` 只包裹原来的每个 RDB，使用 `use_reentrant=False`。
2. 不调用官方 `repeat_x`；每次 decoder 调用只对 feature unfold 一次。
3. ×32 每个像素使用 `(1/32, (y%32)/32, (x%32)/32)`，LR index 为
   `(y//32)*W + x//32`。只预测 1024 组权重；权重每次调用重新计算，不跨 optimizer step 缓存。
4. LR 位置分块；crop 边界只选需要的 offset。不会生成整张 HR position matrix 或逐像素权重。
5. 训练仅构建 crop 图，GT 在 CPU 解码裁剪，再把 crop 传入 GPU；LR feature 仍覆盖完整 256×256。
6. 完整 SR 通过 `iter_full_sr()` 在 `no_grad` 下逐 tile 输出；`infer_sr.py` 写入磁盘 memmap，
   不在 GPU 拼接完整 8K，也不保留完整 SR autograd 图。FP32 `.npy` 每张约 768 MiB。

## 运行

在本目录使用已安装匹配版本的 PyTorch/torchvision；其余依赖列在 `requirements.txt`。
本机已验证解释器：`D:\conda\envs\clam_latest\python.exe`，PyTorch 2.13.0+cu130。

```powershell
python -B validation.py
python -B validation.py --device cuda:0 --output reports/equivalence_cuda.json
python -B -m unittest discover -s tests -v
python -B scripts/probe_cuda.py --n 1 --output reports/cuda_n1.json
python -B scripts/probe_cuda.py --n 6 --output reports/cuda_n6.json
python -B scripts/probe_cuda.py --n 1 --mode full-sr --precision fp32 --output reports/cuda_full8k.json
```

正式训练入口如下。启动时会按所选 RDN 深度运行小尺寸解码等价性与 checkpoint gate，失败就停止。
参考前向来自 vendored 实现；D8 会调整参考模型的深度，不标记为完整官方 RDN-B。
数据根目录应包含真实 `images_256/`、`images_8192/`、`manifests/patch_manifest.csv`。
标签和 fold 默认使用本项目中从 E2E 复制的 CSV，也可以显式指定路径。

```powershell
python -B train_e2e.py --data-root <DATA_ROOT> --fold 0 --output runs/fold0
python -B eval_e2e.py --checkpoint runs/fold0/best.pth --data-root <DATA_ROOT>
python -B infer_sr.py --checkpoint runs/fold0/best.pth --lr-image <LR_IMAGE> --output runs/sr.npy
```

编辑 `configs/baseline.yaml` 可切换 `lambda_sr: 0 / 0.1 / 1.0`、checkpoint、crop 和 chunk。
默认 `sr_train_crop: 256`，CUDA 探测脚本也默认使用 256；每个 region 每次参与训练时独立随机采样一个连续 crop。
每张 WSI 在 region 分批前一次生成 `[N,2]` 坐标；同一随机状态下，full/micro 模式的逐 region crop 一致。
下一次访问 WSI 会重新采样，允许随机重叠，不维护历史去重。每个 region 的预测与 GT 使用相同坐标。
RDN/ResNet 仍按 full/micro 配置编码，SR 则从共享 feature 逐个解码 crop；解码调用次数随 N 增加。
历史显存报告来自本地 GPU；A800 上需以实际最大 region 数重新验证完整训练 step。
region 默认使用 full batch；显存不足时设 `training.use_region_microbatch: true`，
并用 `training.region_microbatch_size` 指定 micro-batch 大小。
`lambda_sr=0` 时不读取 HR、不调用 P2W/Meta-Upscale，P2W/add_mean 不产生梯度。
Loss 为一张 WSI 的 CE 加上 `lambda_sr × 所有 region 的平均 crop L1`。
不使用 LPIPS。SR OFF 仍保留完整的分类梯度路径。

默认 Adam、lr=1e-5、weight_decay=1e-4、20 epochs、val_auc patience=5，沿用 E2E 常用设置。
`history.csv` 记录在线 cls/SR/total loss 和离线 train/val AUC、ACC、BACC；
保存 `last.pth`、`best.pth`、`best_val_predictions.csv`、配置和数据哈希。
每 10 张 WSI 在 optimizer step 前打印梯度范数；每张 WSI 记录 CUDA allocated/reserved 峰值。
当前入口为新实验，不提供 resume；输出目录非空时拒绝覆盖。

## RDN-D8 轻量配置

`configs/rdn_d8.yaml` 只将 `metasr.rdn_blocks` 从 16 改为 8。
每块仍有 8 个 dense conv，growth=64，最终特征仍为 64 通道。
ResNet18-GN、ABMIL、Meta-SR 解码坐标、损失权重与训练设置沿用 baseline。
RDB 拼接通道从 1024 降至 512，GFF 的第一个卷积同步改为 `512 -> 64`。

| 参数统计 | baseline D16 | D8 | 降幅 |
|---|---:|---:|---:|
| RDN 编码器（SFENet + RDB + GFF） | 21,973,952 | 11,024,832 | 49.83% |
| Meta-SR 整体（含 P2W、MeanShift） | 22,419,096 | 11,469,976 | 48.84% |
| 完整 E2E | 33,919,386 | 22,970,266 | 32.28% |

参数统计记录在 `reports/rdn_d8_model_summary.json`。D8 已在本地 RTX 3060 6GB 上完成
真实 `normal_001` 的 N=1、N=2 单步前向、反向与 Adam 更新（BF16、整批编码、256 HR crop）。
峰值 allocated 分别为 0.841/1.528 GiB，reserved 为 1.125/1.820 GiB；
汇总见 `reports/rdn_d8_real_smoke_summary.json`。这次使用 YAML 的 seed=1，早先 D16 探测为 seed=7，
权重与裁剪坐标不同；单步结果不用于判断长期训练效果或严格提速倍数。

本地两对真实图片的 D8 冒烟命令（在项目目录执行）：

```powershell
& D:\conda\envs\clam_latest\python.exe -u -B scripts\probe_cuda.py --config configs\rdn_d8.yaml --data-root D:\ddata --slide-id normal_001 --n 2 --output reports\real_n2_rdn_d8.json
```

probe 使用 YAML 中的模型、种子、优化器和训练步设置；显式提供的 precision、crop、lambda_sr
及区域分批 CLI 参数覆盖对应设置。无 `--config` 时保留原先的 D16、seed=7 探测默认值。

完整数据集训练时选择新配置：

```bash
python -B train_e2e.py --config configs/rdn_d8.yaml --data-root <DATA_ROOT> --fold 0 --output runs/fold0_rdn_d8
```

本地 `D:\ddata` 仅有一个 WSI 的图像，适用于上述单 WSI 冒烟；完整训练还需要划分中涉及的其余图像。
D8 与 D16 的 RDB/GFF 参数形状不同，D16 checkpoint 不能直接 strict-load 为 D8；使用新实验目录从头训练。
已完成 D8 真实 CUDA N=2 整批/micro1 的梯度及 Adam 更新对照：BF16 未通过；
FP32 梯度通过，但严格逐元素参数更新检查未全部通过，确定性控制组也有同类差异。
详见 [完整审计与复查命令](reports/d8_n2_equivalence/README.md)。当前保留整批默认设置，不能将分批标为已验证无损。

## RDN-D4 轻量配置与真实 N=2 显存

`configs/rdn_d4.yaml` 相比 D8 只将 `metasr.rdn_blocks` 改为 4，保持整批 BF16、
checkpoint、256 HR crop 和原有优化器设置。RDN 编码器 5,550,272 参数，完整 E2E 17,495,706 参数。

已用 `D:\ddata` 的 `normal_001` 两对真实图片完成 N=2 整批前向、反向和一次 Adam 更新。
两个 LR 区域一起编码；每个对应的 HR 图独立裁一个 256×256 区域，SR 损失按两个区域取平均；
两个 embedding 组成同一完整 WSI 袋子进行分类。

| N=2 整批 BF16 | 峰值 allocated | 峰值 reserved |
|---|---:|---:|
| 之前 D8 实测 | 1.528 GiB | 1.820 GiB |
| D4 实测 | 1.384 GiB | 1.627 GiB |

D4 allocated 比之前 D8 测量低约 9.43%。这些是 PyTorch 显存指标，包含首次 Adam 状态分配。
汇总见 `reports/rdn_d4_real_smoke_summary.json`，原始记录见 `reports/real_normal_001_d4_n2_20260930.json`。
不同深度会改变初始化和随机数消耗，因此即使 seed 相同，权重与裁剪坐标也不同；
该单步冒烟不证明模型精度或稳定提速比例，也没有验证 D4 的整批/分批等价性。

在项目目录复跑同一 WSI 的两个区域：

```powershell
& D:\conda\envs\clam_latest\python.exe -u -B scripts\probe_cuda.py --config configs\rdn_d4.yaml --data-root D:\ddata --slide-id normal_001 --n 2 --output reports\real_n2_rdn_d4.json
```

完整数据集训练时使用 `--config configs/rdn_d4.yaml` 并选择新的输出目录。
D8/D16 checkpoint 不能直接完整 strict-load 到 D4。

## 历史基线验证边界

见 `REPORT.md` 和 `reports/` 中的原始结果。CPU FP32 等价测试使用完整官方 RDN-B，
只缩小空间尺寸；CUDA 探测也没有缩减模型。CUDA BF16 是训练精度设置，不能称为 FP32 精确等价。
早期基线验证使用合成张量或临时图像；后续真实 N=2 比较见 `reports/real_n2_batch_compare_20260930/README.md`。
本版保留普通整袋 autograd 图，因此显存仍随 WSI 的 N 增长，不能保证任意 N 都适合 6 GB GPU。
