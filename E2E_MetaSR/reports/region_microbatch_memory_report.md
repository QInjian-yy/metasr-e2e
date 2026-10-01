# Region full-batch / micro-batch 审计与显存验证

本轮仅使用合成输入，没有读取 CAMELYON16 图像，也没有开始真实数据训练。当前设备是 **NVIDIA GeForce RTX 3060 Laptop GPU，6 GiB，WDDM**；PyTorch `2.13.0+cu130`。A800 80GB 未实测，相关数字在第 8 节明确标为估算。

## 1. 当前 micro-batch 路径

原配置 `region_batch_size: 1` 是指：一个 WSI 有 `N` 个 region 时，Meta-RDN 和 ResNet18 每次处理 `[1,3,256,256]`；将全部 region 的 `[1,512]` embedding 保留在普通 autograd 图里，拼成 `[N,512]` 后一次送进 ABMIL。随后对 WSI 分类 loss 做一次 backward 和一次 optimizer step。SR 开启时，每个 region chunk 同步解码 crop，按 region 数加权汇总 L1。

训练路径**没有** detach、GradPool、replay 或 `no_grad`；micro-batch 没有切断梯度链。它只改变 region encoder 的单次输入批量。验证阶段的 `no_grad` 是原有评估行为。

## 2. 新开关与执行路径

`configs/baseline.yaml` 现在使用：

```yaml
training:
  use_region_microbatch: false
  region_microbatch_size: 1
```

`false` 时，`wsi_loss` 和 `evaluate` 每张 WSI 用一个 `[N,3,256,256]` batch 编码，不拆 region；计算图直接经过 Meta-RDN、ResNet18-GN、ABMIL 和 loss。`true` 时才按 `region_microbatch_size` 分块，embedding 仍合并成整袋后只调用一次 ABMIL。两种模式都保留普通 autograd，没有 detach 或 replay。`region_microbatch_size` 必须为正整数。

ResNet18-GN 沿用现有实现，对应你的批注 :codex-annotation{index="1"}。

## 3. Classification-only CUDA 单步

设置 `lambda_sr=0`、RDB checkpoint ON、BF16、train mode；计时包含 forward、backward 和 Adam step。N=1 的 full 与 micro 有效 batch 都是 1，因此一次测量适用于两种设置。

| N | Full-region batch | Micro-batch=1 | 备注 |
|---:|---|---|---|
| 1 | 通过；peak allocated/reserved `1.008 / 1.436 GiB`；`18.51 s` | 与 full 的有效计算完全相同 | 本轮新测 |
| 6 | 本轮在 `nvidia-smi 5926/6144 MiB`、GPU 100% 时主动中断；未取得 CUDA peak 或完成状态 | 通过；`2.643 / 3.068 GiB`；`105.59 s` | 本轮新测；full 未计作通过 |
| 15 | 未测 | 未测 | Full N=6 已逼近上限；micro N=15 的分类-only线性估算也约 `5.59 GiB allocated`，故不冒险实跑 |
| 25 | 未测 | 未测 | 两种模式均不在 6 GiB WDDM 卡上继续探测 |
| 35 | 未测 | 未测 | 两种模式均不在 6 GiB WDDM 卡上继续探测 |

中断记录在 `reports/region_mem_cls_n6_full_aborted.json`。它只记录中断前 `nvidia-smi` 的整机 GPU 占用，**不代表**模型自身的 allocated/reserved 峰值。`reports/region_mem_cls_n1_full.json` 和 `reports/region_mem_cls_n6_micro1.json` 保存了完成单步的完整结果。

## 4. Joint training CUDA 单步

使用 `lambda_sr=0.1`、Meta-SR crop=256、RDB checkpoint ON、BF16、train mode；SR loss 为 crop L1，计时包含 optimizer step。下列 N=1/N=6 来自项目已有、完整通过的同机合成 CUDA 记录；旧探测器的 `micro_batch=6` 与现在 `use_region_microbatch=false` 在 N=6 时走相同的单次 `[6,3,256,256]` encoder forward。

| N | Full-region batch | Micro-batch=1 |
|---:|---|---|
| 1 | `1.081 / 1.502 GiB` allocated/reserved，`23.20 s`（有效 batch=1） | 同一测量 |
| 6 | `5.508 / 6.691 GiB`，`108.99 s` | `2.860 / 3.266 GiB`，`131.18 s` |
| 15 | 未测 | 未测 |
| 35 | 未测 | 未测 |

N=6 full joint 的完整单步已通过；其 reserved 超过设备标称显存，说明 WDDM 下 reserved 不能直接当作可用物理余量。两个 N=6 完整 joint 记录分别为 `reports/cuda_n6_batch6_256_bf16.json` 和 `reports/cuda_n6_256_bf16.json`。在这组旧合成测量中 full step 约快 17%，但只代表该 RTX 3060 和该输入配置。

## 5. Full vs micro FP32 数值比较

固定 seed=811、N=6、FP32、`lambda_sr=0`，相同模型参数和输入；比较一次 `[6,3,16,16]` 编码与六次 `[1,3,16,16]` 编码。模型处于 eval mode；本模型中的 RDN、GroupNorm、ABMIL 没有依赖 train/eval 状态的随机层或运行统计。详细结果在 `reports/region_microbatch_equivalence.json`。

| 项目 | max_abs_error | 结论 |
|---|---:|---|
| Region embeddings | `1.097e-5` | 近似一致 |
| ABMIL attention | `5.960e-8` | 近似一致 |
| WSI logits | `7.749e-7` | 近似一致 |
| Classification loss | `0` | 相同 |
| Meta-RDN gradients | `5.432e-4`；relative L2 `4.215e-4` | 逐元素原阈值有若干张量超差 |
| ResNet18-GN gradients | `1.594e-6`；relative L2 `1.911e-6` | 近似一致 |
| ABMIL gradients | `1.565e-7`；relative L2 `8.626e-6` | 近似一致 |
| Classifier gradients | `3.338e-6`；relative L2 `6.437e-7` | 近似一致 |

结论是实数运算下两路实现相同，但 FP32 实际卷积会因 batch 形状改变归约顺序；因此不能声称逐元素完全相等。RDN 梯度的绝对最大误差较大，平均绝对误差为 `1.846e-7`、relative L2 为 `4.215e-4`。这次误差已保留，没有放宽阈值掩盖。

## 6. 速度

本轮完成的分类-only N=6 micro step 用时 `105.59 s`；N=6 full 分类-only因 GPU 接近满载被中断，没有可比较的完整 step 时间。已有 joint N=6 合成测量为 full `108.99 s`、micro `131.18 s`，只作为同机旧记录，不代表 A800 速度。

## 7. 测试和显存统计

- `python -B -m unittest discover -s tests -v`：**22/22 通过**，包括 full-batch 一次调用且 embedding 保留 grad_fn、micro-batch 分块、SR OFF、ABMIL backward、scale=32 和 Meta-SR 等价测试。
- `load_config` 成功读取默认 `use_region_microbatch=false`、`region_microbatch_size=1`。
- 官方 Meta-Upscale 与优化实现 ×2/×3/×4 最大误差分别 `2.384e-6 / 1.907e-6 / 2.861e-6`；checkpoint forward、input-gradient、parameter-gradient 最大误差均为 0。记录也由本轮完整测试输出确认。
- 分类-only 本轮完整 CUDA 单步：N=1 full、N=6 micro=1；另有旧记录 N=1 joint、N=6 full/micro joint。
- `lambda_sr=0` 的单测确认不读取 HR、不调用 decoder，Pos2Weight 梯度为 `None`。
- N=6 full 分类-only中断前的 WDDM 占用接近总显存；N≥15 未在本机强行尝试。所有实测均为合成数据，未启动真实数据长训练。

## 8. A800 80GB 估算和建议

以下是**估算，不是 A800 实测**。用 RTX 3060 已完成的 full joint N=1（`1.081 GiB`）和 N=6（`5.508 GiB`）做线性外推。分类-only 没有取得 full N=6 的 CUDA allocator 峰值，因此使用 full joint 的 N=6 峰值作为保守上界代理，并从分类-only N=1 锚点外推。模型、batch 算法和 cuDNN workspace 可能随硬件变化；额外列出 1.5× 线性结果作为压力余量，不把它当硬件保证。

| N | Classification-only full batch 估算 allocated | Joint full batch 估算 allocated | Joint 的 1.5×压力值 |
|---:|---:|---:|---:|
| 6 | ≤`5.51 GiB`（用 joint 峰值作上界代理） | `5.51 GiB`（RTX实测锚点） | `8.26 GiB` |
| 15 | `13.61 GiB` | `13.48 GiB` | `20.22 GiB` |
| 25 | `22.61 GiB` | `22.34 GiB` | `33.51 GiB` |
| 35 | `31.61 GiB` | `31.19 GiB` | `46.79 GiB` |

按这个保守但粗糙的外推，N≤35 在 A800 80GB 上很可能可整批运行，joint N=35 加 1.5×余量约 `46.8 GiB`。因此当前配置默认保持：

```yaml
use_region_microbatch: false
region_microbatch_size: 1
```

micro-batch 作为低显存 fallback。建议真实 CAMELYON16 长训练前，先在目标 A800 上用合成 N=35、相同 crop、精度和 optimizer 完成一次 forward/backward/step，并记录 allocated/reserved；本轮没有 A800 实测，估算不能代替这项目标机验证。
