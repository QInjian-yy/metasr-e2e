# E2E_MetaSR 只读审计报告

审计日期：2026-09-27。本次仅做源代码读取与定向验证；未启动 CAMELYON16 数据训练。
审计期间 production code 未改。按明确要求只新增了本报告。审计前保存的
`audit/before_sha256.json` 共 247 个 E2E_MetaSR 文件，本次复核无缺失、无散列变化；
`reports/source_integrity.json` 记录的 Meta-SR 官方目录与原 E2E 共 203 个文件也均未变。
审计证据及逐项数据在 `audit/math_cpu.json`、`audit/metadata.json`、
`audit/optimizer_parameters.csv`、`reports/` 中。

## 1. 架构与官方 Meta-SR 对照

独立加载 `Meta-SR-Pytorch-0.4.0/model/metardn.py` 与 `trainer.py` 作对照：新模型
继承随项目复制的官方 `MetaRDN`；唯一 vendor 源码差别是把 `from model import common`
改为包内相对导入。官方类与新模型有 **304 个 state-dict 项目，名称及 shape 全部相同**。

| 模块 | 核验结果 |
|---|---|
| `MeanShift` | 官方 `common.MeanShift`，位置不变 |
| SFENet1 / SFENet2 | 官方 Conv，顺序和参数 shape 不变 |
| RDB_Conv / RDB / LFF | 官方定义：dense concat、1×1 LFF、局部 residual；block 内未改写 |
| RDB 数量 / 每块卷积数 | B 配置，16 / 8，growth=64，G0=64 |
| Global Feature Fusion | 原 1×1 Conv → 3×3 Conv |
| global residual | `x = GFF(cat(RDBs_out))` 后 `x += f__1` |
| Pos2Weight | 原 `Linear(3,256) → ReLU → Linear(256,1728)` |
| kernel size | 3×3 |

`extract_features()` 与官方 `forward()` 的 feature 路径逐操作相同，返回正是
GFF 加 shallow residual 后的 `[N,64,H,W]`，无额外的 SR feature Conv、Norm、activation、
projection 或 attention。真实输入尺寸由 dataset loader 限定为 `[N,3,256,256]`，输出
`[N,64,256,256]`；CUDA synthetic joint probe 实测了该尺寸。

## 2. 与原 E2E downstream 的差异

原 `E2E/downstream/shared_model.py:SharedE2EModel.__init__` 使用 64→64 的 ResNet18
首层替换和 Identity fc，**实际 ResNet18 为 BatchNorm**；文件里的 BN→GN helper 没在此构造路径
调用。新 baseline 保留首层、ResNet18 stage、512 维 embedding、`GatedAttentionMIL` 和
Linear(512,2)，并按用户后续要求启用现有算法对应的 BN→GN helper，故有 20 个 GroupNorm、
0 个 BatchNorm。GroupNorm 使每个 region 独立于 micro-batch 分组；全 batch / micro-batch
对照见第 11 节。BN→GN 会改变归一化与训练动态，是两项目做严格 downstream 公平对照时要记录
的一项差异；本次按用户要求保留 GN，未改回 BN。数据分组/loss 实现是独立复制版本，ABMIL 类
逐 AST 对照与 E2E 原类一致。

## 3. ×32 optimized Meta-Upscale 数学等价性

固定整数 `r=32` 时，官方 `input_matrix_wpn()` 按 HR row-major 顺序产生：

```text
[1/32, (y % 32)/32, (x % 32)/32]
```

每 LR 点的 `(dy,dx)` 组合恰有 1024 个。`position_table()` 的顺序为
`offset_id = dy*32 + dx`，得到官方 Pos2Weight 输出后 reshape 为
`[1024,576,3]`。LR patch gather 索引为 `floor(y/32)*W + floor(x/32)`。
输出 reshape/permutation 对应 `(LR_y,dy,LR_x,dx)`，与官方 `view(...).permute(...)`
到最终 HR 行列的顺序一致。offset 值是二进制可精确表示的小数，1024 个 position 与
官方 `input_matrix_wpn()` **逐值完全相等**。

实现每次 decode 对 `[B,64,H,W]` 执行一次 `unfold(3,padding=1)`，随后按 LR chunk
gather patch 并乘 position weight；不执行 `repeat_x`，不产生 scale² 份 feature 或逐 HR pixel
权重矩阵。weight table 按当前参数重新计算，不跨 optimizer step 缓存。数学与运算顺序之外的
临时张量布局发生改变，这些属于 memory optimization。

## 4. scale=32 官方 vs optimized 小输入结果

独立执行官方源码的 `MetaRDN.forward()`、`Trainer.input_matrix_wpn()` 和 decoder 语句。
FP32 CPU，完整 RDN-B，只有输入空间尺寸缩小。输出/输入梯度的误差按元素最大值和均值统计；
参数梯度分列 RDN、Pos2Weight。

| LR shape | 输出 shape | 输出 max / mean | 输入梯度 max / mean | RDN 梯度 max / mean | P2W 梯度 max / mean |
|---|---|---:|---:|---:|---:|
| `[1,3,2,2]` | `[1,3,64,64]` | 7.749e-7 / 8.469e-8 | 6.706e-8 / 3.632e-8 | 2.161e-7 / 2.125e-11 | 4.843e-8 / 2.919e-10 |
| `[1,3,4,4]` | `[1,3,128,128]` | 1.431e-6 / 9.966e-8 | 4.470e-8 / 1.470e-8 | 1.192e-7 / 4.411e-11 | 1.080e-7 / 4.474e-10 |

第二尺寸完整参数梯度最大误差 `8.345e-7`，全体参数 mean absolute error
`5.216e-11`。所有对应参数均有匹配的梯度存在状态。scale=32 官方等价性测试通过。

## 5. 非对齐 HR crop 测试

×4 官方整图 `[1,3,20,28]` 与从绝对坐标 `(3,7)` 起的 `13×17` optimized crop 比较：
663 个 RGB pixel 值，max error `5.245e-6`，mean `7.794e-7`，容差内通过。

×32 另按题目示例验证 `(y,x,h,w)=(137,291,35,53)`。crop 第一像素的绝对 LR
index 是 `floor(137/32)*16+floor(291/32)=73`，位置为
`[1/32,9/32,3/32] = [0.03125,0.28125,0.09375]`。crop 内所有绝对 position 与官方整张
position matrix 精确相等；独立按官方 absolute position + LR gather 构造的 oracle 与
`decode_crop` **逐像素相等**（max/mean error 均 0）。crop 左上角没有被当成新图 `(0,0)`。

## 6. 边界测试

×4、LR `[1,64,5,7]` 的全部 HR 子像素，四角及所有边缘均对照官方 decoder：

| 项目 | max absolute error |
|---|---:|
| 左上角 4×4 子像素 | 1.431e-6 |
| 右上角 | 0 |
| 左下角 | 2.384e-6 |
| 右下角 | 0 |
| 全部边缘 | 4.768e-6 |

×32 也对比四角及全部边缘，全部边缘 max error `5.722e-6`。`F.unfold(...,padding=1)`
与显式 constant-zero padding 后手工抽取 3×3 patch **逐值相等**；coverage map 的 min/max 都为
1。实现没有 clamp、reflect 或 replicate padding。

## 7. checkpoint 等价性

固定 seed，`model.train()`、FP32、同权重和输入，对比 checkpoint OFF/ON（只包各官方 RDB，
`use_reentrant=False`）：

- 输出 max/mean error：0 / 0；输入梯度：0 / 0。
- 全 16 个 RDB 各自的参数梯度、GFF 梯度、全部 304 个参数梯度：max error **0**。
- 分别执行 Adam step 后参数差：max error **0**。

官方 RDB 内部没有 dropout/BN 等状态随机模块，未引入手工 RNG replay。

## 8. classification-only 梯度链

`lambda_sr=0` 的 FP32、N=6 synthetic test 覆盖 `[6,3,16,16]`；禁止 HR loader、P2W 和
decoder 的 mock 确认它们均未被调用。各列为有限、存在梯度的参数整体 L2 norm；RDB 和 GFF
检查了全部参数张量，而非只检查一个入口。

| 位置 | 梯度 L2 |
|---|---:|
| ABMIL（5/5 tensors） | 0.186103 |
| ResNet 首层 conv | 43.313818 |
| ResNet layer4 | 49.670503 |
| ResNet 末层 conv | 23.041645 |
| SFENet1（2/2） / SFENet2（2/2） | 4.061695 / 4.386513 |
| RDB1（18/18） / RDB8（18/18） / RDB16（18/18） | 1.871217 / 1.551148 / 0.568300 |
| GFF（4/4） | 8.272468 |

这些梯度由 `wsi_loss → model.encode_regions → RDN feature → ResNet18 → cat(embeddings)
→ ABMIL → CE` 正常 autograd 反传。逐文件检查该路径没有 `detach()`、`.cpu()`、`.numpy()`
或 `no_grad()` 断开 RDN 的操作。P2W 4 个梯度为 `None` 是 SR OFF 的预期行为。

## 9. SR-only 梯度链

`lambda_sr>0`、scale 32、小型 synthetic LR/crop 的 SR-only L1 backward：SFENet1/2 为
0.365561 / 0.301250，RDB1/8/16 为 0.365560 / 0.301250 / 0.116316，GFF 为
1.962469，Pos2Weight 为 **3.231714（4/4 tensors）**。ABMIL、classifier、ResNet 没有梯度，
符合 SR-only 计算图。

## 10. joint 梯度链

`L_total = L_cls + 0.1 L_sr` 的独立 joint probe 中，ABMIL L2=0.0967778，ResNet 首层
18.936392；SFENet1/2=1.835275 / 1.929412；RDB1/8/16=0.732303 / 0.590009 /
0.227271；GFF=3.431899；P2W=0.323171。所有要求的分支均有梯度。
另分别量出 classification 与 SR 梯度，二者按 0.1 加和后与 joint 梯度的 max 差
`2.328e-7`、relative L2 `6.455e-8`。

## 11. micro-batch vs full-batch 等价性

eval mode、classification-only、N=6，在同模型权重和相同六个 LR 输入下比较一次性 B=6
与 B=1 六次 encoder、再合并完整 embedding 交给同一个 ABMIL。生产路径保留每次 embedding
的 autograd graph，用 `torch.cat` 组 bag；没有 GradPool、detach/replay、梯度缓存或 BN replay。
GroupNorm 不跨 region 计算统计量。

| 比较项（FP32） | max / mean absolute error |
|---|---:|
| region embedding | 1.097e-5 / 1.283e-6 |
| ABMIL attention | 5.960e-8 / 2.980e-8 |
| WSI logits | 7.749e-7 / 7.451e-7 |
| CE loss | 0 / 0 |
| ResNet 参数梯度 | 1.594e-6 / 9.166e-9 |
| ABMIL 参数梯度 | 1.565e-7 / 2.089e-9 |

**FP32 RDN 梯度的严格逐元素比较出现一处容差失败，不能隐藏。** 全 RDN 梯度 max
difference 为 `5.432e-4`、mean `1.846e-7`、relative L2 `4.215e-4`；少数低幅度分量未通过
`atol=1e-5, rtol=5e-4`，所以最初严格 assert 失败。该失败已记录在
`audit/math_cpu.log`，不是改代码后消失。独立 FP64 诊断（RDN 数学一致但不经生产 `wsi_loss`
将 embedding `.float()` 的路径）中，RDN 梯度 max difference `6.661e-16`、relative L2
`1.150e-15`；attention/logit/loss 与 ResNet/ABMIL 梯度也为 0 或约 `1e-15`。这支持两条
路径数学一致，FP32 细微差异与批量卷积浮点累加顺序相符，但不等同于生产 BF16 在所有 bag 上
都已证明等价。现有结论是输出与分类结果接近，FP32 RDN 梯度存在可复现的小量级差异，非逐元素
相同。该数值差异不是本次 SR decoder 修改；本审计没有重构训练路径。

## 12. optimizer 参数覆盖

Adam 唯一参数组覆盖 371 个 `requires_grad=True` tensors、33,919,386 个元素：Meta-RDN /
MeanShift 300、Pos2Weight 4、ResNet18 60、ABMIL 5、classifier 2。按 parameter object
identity 计数，所有 trainable parameter 恰好出现一次；遗漏 0、重复 0。
SR-only 时 Pos2Weight 获得梯度；SR OFF 时 P2W 不建图、梯度 `None`，Adam（含 weight decay）
step 后 P2W 参数保持原值。

注意官方 `common.MeanShift.__init__()` 使用 `self.requires_grad = False`，这只是给 module
赋普通属性，**没有冻结它的 weight/bias Parameter**。复核发现 sub_mean/add_mean 四个参数的
`requires_grad` 均为 True，官方 trainer 的 `model.parameters()` optimizer 和新 optimizer
都会优化它们。新项目忠实保留官方代码与行为；本次未自行“修复”为冻结，因为那会改变官方
训练行为与可比性。

## 13. joint loss normalization

实现为：

```python
sr_terms.append(F.l1_loss(prediction.float(), truth) * batch_size / N)
L_sr = torch.stack(sr_terms).sum()
L_total = L_cls + lambda_sr * L_sr
```

每个 `F.l1_loss` 对其 RGB crop 的所有 pixel/channel 取 mean。乘 micro-batch size 再除以 WSI
region 总数后求和，等于 `(1/N) Σ_i mean_pixel(L1_i)`；不会因 WSI patch 数增多而按 region
累加放大。N=1/B=1、N=5/B=2、N=6/B=1、N=6/B=6 都对照单区域独立均值；最大差
`5.961e-8`。训练同一配置对每个 region 使用相同 crop 尺寸。

## 14. 真实数据 metadata dry-run

从复制自 E2E 的原始 CSV 读取真实 metadata（未打开图像）：262 张 WSI、2,788 个 region；
Normal=151、Tumor=111；每张 WSI `N=1…37`。manifest slide_id 集与 labels slide_id 集
完全相同；重复文件名 0；文件名前缀 label 错误 0；`filename = slide_id_x_y` 坐标名错误 0。
fold 0 为 train 174 / val 88、case overlap 0；其他 fold 也执行了 metadata loader dry-run。
抽检 Normal/Tumor 的少 region 与多 region WSI：一个 WSI 的所有 LR/HR path 依 manifest 行顺序
一一配对、文件名一致、label 与 CSV 相同。variable N 会全部编码后交给 ABMIL。6 个 manifest /
label/split CSV 与原 E2E 文件字节一致。

但本机 `E2E_MetaSR/images_256`、`images_8192` 和 `E2E/images_256`、`images_8192`
目录均不存在。未能真实打开图像，未能核对图像尺寸、decode、RGB 内容或 LR/HR 像素/坐标配准；
这一项是 metadata-only dry-run，不是完整真实图像 dry-run。

## 15. 完整 8K streaming 正确性

真实尺寸 synthetic FP32 CUDA probe 返回完整逻辑输出 shape `[1,3,8192,8192]`、
67,108,864 个 RGB pixel、256 tiles，全部 tile 都检查 finite。对 `[1,64,256,256]`
及默认 `lr_chunk_size=256`，每个 LR row 产生一个 `[1,3,32,8192]` tile；256 个 LR row
严格覆盖 8192 行，x 从 0 开始单 tile 覆盖 8192 列，tile 地址互斥，故全图 coverage 恰为 1。
相同 iterator 在小 scale 32 官方对照中还实测 coverage map min=max=1、逐像素比较官方整图，
max/mean error `5.722e-6 / 3.607e-7`。`infer_sr.py` 按这些地址写 CPU/disk memmap，不在
CUDA 聚合完整 8K tensor；memmap 小尺寸自动化 test 验证文件 shape 和写入 crop。完整 8K probe
计数完整推理像素，但未另存一张约 768 MiB 的 8K `.npy` 作为本次审计产物。

## 16. 显存重新测量

GPU 为 NVIDIA GeForce RTX 3060 Laptop GPU，PyTorch 报告总显存 `6,441,926,656` bytes
（6 GiB / `nvidia-smi` 6144 MiB）。本次新测每次模型/optimizer 建好后记录 allocator baseline，
执行 `torch.cuda.empty_cache()` 与 `reset_peak_memory_stats()`，合成输入下做 joint forward、
backward、gradient finite 检查和一次 Adam step。原始 allocator 数值见下表，GiB=bytes/2³⁰。

| N / micro-batch | allocated before → after empty_cache | peak allocated | peak reserved | 完整单步 |
|---|---:|---:|---:|---|
| 1 / 1 | 0.127 → 0.127 GiB | 1.072 GiB | 1.500 GiB | 通过 |
| 6 / 1 | 0.142 → 0.142 GiB | 2.998 GiB | 3.266 GiB | 通过 |
| 6 / 6，本轮新测 | baseline 约 0.14 GiB，未完成 step | 不报告 | 不报告 | WDDM 下运行过久，已中止 |

同批 N=6 的先前留档测试完整执行到 Adam step（109.0 s），结果 allocated `5.508 GiB`、reserved
`6.691 GiB`，状态 passed。本轮在当前 Windows GPU 负载下也尝试重测，但在 forward/backward/
Adam step 完成前耗时过久而中止，所以不能提供新的 B=6 peak 结果。`max_memory_reserved` 是 caching allocator
本次管理的 reserve 峰值，含 allocator 为峰值请求保留的 segment，并非仍被 tensor 使用的 live
memory，也不是 NVIDIA 面板剩余显存。此 Windows WDDM 桌面会共享/占用显存；旧同批测量的
reserved 超过标称 6 GiB 仍完成一步，不代表在 6 GiB dedicated VRAM 上有足够部署余量。
默认 B=1 所有 N 个 region 仍参与同一 WSI 的 ABMIL 与分类 loss。

## 17. 已发现的问题及证据边界

1. full-vs-micro-batch 的 FP32 RDN gradient 严格逐元素阈值有记录在案的失败，误差与 FP64
   诊断分别见第 11 节；没有隐藏失败或修改 model math 来消除它。
2. 原官方 MeanShift 的 `self.requires_grad=False` 没冻结其 parameters；新项目忠实保留，optimizer
   会更新这四个参数。见第 12 节。
3. 原 E2E 下游是 BN，新项目按用户要求是 GN；这是明确的分类 branch 实验差异，见第 2 节。
4. 真实图像文件当前缺失；数据内容及 LR/HR 实际配准未验证。第 14 节只覆盖真实 metadata。
5. 同批 N=6 已完成的旧显存测量 reserve 超出标称显存；Windows 测试成功不能外推为其他 GPU
   显存配置可运行。

目前没有发现 Meta-SR crop 坐标、公式、输出排布、边界 padding、分类计算图或 loss/N normalization
需要修改的确定性代码 bug。

## 18. 是否修改及最小修改内容

未修改 `E2E_MetaSR` production code，也未修改原 `E2E/` 或 `Meta-SR-Pytorch-0.4.0/`。
按用户要求新增本文件 `AUDIT_REPORT.md`。没有借机重构、改阈值、改归一化或改写官方网络。

## 19. 最终全部测试结果

| 验证项 | 结果 |
|---|---|
| 官方模块/state-dict shape 和 RDN-B 结构 | 通过 |
| scale 32 官方 vs optimized forward/input/RDN/P2W gradients | CPU 小尺寸通过 |
| absolute crop、×4 非对齐 crop、四角/全部边界 zero-padding | 通过 |
| 小 feature streaming 与官方整图逐像素、tile coverage | 通过 |
| checkpoint train-mode OFF/ON、各 RDB/GFF/input gradients、Adam step | 全部差 0，通过 |
| classification-only 到 RDB16/GFF 的完整梯度链 | 通过；P2W 不计算梯度符合 SR OFF |
| SR-only 和 joint 梯度路径、梯度相加关系 | 通过；P2W 在 SR ON 获得梯度 |
| full/micro-batch outputs、ResNet/ABMIL gradients | 通过，见第 11 节的 FP32 RDN 细项 |
| optimizer coverage / duplicated parameters | 371/371 恰好一次，通过 |
| SR pixel→region→WSI normalization，N=1/5/6 | 误差 ≤5.961e-8，通过 |
| real CAMELYON16 CSV grouping/labels/folds/variable N | metadata-only 通过；图像 dry-run 不可用 |
| 8K FP32 CUDA streaming shape/pixel count | `[1,3,8192,8192]`，67,108,864 pixels，通过 |
| 当前轮 CUDA peak memory | N=1、N=6/B=1 重测通过；N=6/B=6 新测中止，引用上一轮已完成数值 |
| 审计期间项目文件 | 247 个既有散列未变；两原目录 203 个文件未变 |

### 最终结论

**A. 可以确认 Meta-SR 数学保持官方定义。** Meta-RDN/P2W 来源、scale-32 官方对照和 crop/stream
像素验证均支持这一结论；改动位于 memory 计算顺序和临时 tensor 布局。GroupNorm 属于下游
归一化差异，不属于 Meta-SR 变化。

**B. 可以确认 classification loss 能反传到 Meta-RDN。** synthetic `lambda_sr=0` 实测 SFENet、
RDB1/8/16、GFF 梯度均 finite 且非零，forward 没有 detach/no-grad 断点。

**C. 当前目录尚不能完成 CAMELYON16 小规模真实训练 dry-run。** 模型和合成输入路径通过审计，
但当前机器没有对应真实 256/8192 图像目录；真实图像尺寸、读取、配对无法确认。本轮未启动
真实数据训练。取得并指向真实 image root 后再做单 WSI/单 optimizer step 的数据 dry-run，才能确认
真实训练入口已就绪。
