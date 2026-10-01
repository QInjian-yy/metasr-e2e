# MetaSR baseline 实现与验证报告

日期：2026-09-27。所有新增内容位于 `D:\sdx_model\E2E_MetaSR`。
原 `E2E/` 和 `Meta-SR-Pytorch-0.4.0/` 共 203 个文件的 SHA-256 校验通过，无修改、删除或新增。
根据后续要求，ResNet18 的 20 个 BN 已全部替换为 E2E 现成实现的 GroupNorm；ABMIL 未改。

1. **主要新增文件**：`models/metasr.py`、`models/baseline.py`、`models/abmil.py`；
   `engine.py`、`train_e2e.py`、`eval_e2e.py`、`infer_sr.py`；`configs/baseline.yaml`；
   `validation.py`、`tests/test_metasr.py`、`tests/test_engine.py`、`scripts/probe_cuda.py`；
   `README.md`、`requirements.txt`、本报告和 `reports/*`。
   复制文件包含 `vendor/official/*`、`reference/*`、`wsi_data.py`、`augmentation.py`、
   `tests/test_wsi_data.py`、`downstream_train/*.csv`、`manifests/*.csv`。
2. **官方复用**：Meta-RDN-B 的全部参数模块保持官方定义：SFENet1/2、16 个 RDB
   （8 conv/block、growth=64）、GFF、MeanShift、Pos2Weight `3→256→1728`。
   vendor 仅调整 common 导入路径。测试使用官方 forward 和原样 `input_matrix_wpn`。
   官方代码可配置的 `rgb_range` 设为 1，以匹配现有 E2E 的 `[0,1]` 数据。
3. **内存优化**：RDB 外层 checkpoint；feature 只 unfold 一次；按 `(1/r,dy,dx)`
   复用 1024 组 P2W 输出；按 LR chunk/offset 计算连续 crop；完整 SR 以无梯度 tile 输出。
   这些操作没有修改 RDB/P2W/Meta-Upscale 的数学公式。训练不默认生成 8K。
4. **shape**：`[N,3,256,256] → [N,64,256,256]`。分类：ResNet18-GN
   `→[N,512]→ABMIL [1,512]→logits [1,2]`。SR：unfold `[B,576,65536]`；
   positions `[1024,3]`；weights `[1024,576,3]`；chunk `[B,L,1024,3]`；
   默认 crop `[B,3,256,256]`；完整输出逻辑形状 `[N,3,8192,8192]`。
   B 为 region micro-batch，L≤配置的 `lr_chunk_size`。两分支共享同次 forward 的 feature。
5. **checkpoint ON/OFF（FP32，eval 且启用梯度）**：CPU 输出、输入梯度、参数梯度最大误差均 **0**。
   CUDA 输出误差 **0**，输入梯度 **1.863e-9**，参数梯度 **1.490e-8**。
6. **官方与优化实现等价性**：完整官方 B 网络，小空间尺寸，不缩减通道/层数。

| scale | CPU forward 最大绝对误差 | CUDA forward 最大绝对误差 | CPU/CUDA 参数梯度最大误差上界 |
|---|---:|---:|---:|
| ×2 | 2.384e-6 | 2.384e-6 | 3.577e-7 |
| ×3 | 1.907e-6 | 2.384e-6 | 4.769e-7 |
| ×4 | 2.861e-6 | 2.027e-6 | 2.385e-7 |

另测非 LR 边界对齐 crop，forward 最大误差 7.153e-6、参数梯度 2.385e-6，
在 FP32 浮点容差内。×32 的 1024 个 offset 与官方 position matrix 精确一致。

7. **CUDA 峰值**：RTX 3060 Laptop 6 GB，PyTorch 2.13.0+cu130。
   joint 为 BF16、RDB checkpoint ON、256×256 GT crop、lambda_sr=0.1，包含一次 Adam step；
   原始字节数保存在 JSON。N 是完整 WSI 的 region 数，B 是同时编码的 region 数。

| 场景 | N | B | allocated GiB | reserved GiB | 结果 |
|---|---:|---:|---:|---:|---|
| Joint，LR 256×256 | 1 | 1 | 1.081 | 1.502 | forward/backward/Adam 通过 |
| Joint，LR 256×256 | 6 | 1 | 2.860 | 3.266 | 完整 6-region bag 通过 |
| Joint，LR 256×256，同批输入 | 6 | 6 | 5.508 | 6.691 | forward/backward/Adam 通过 |
| Full SR，FP32 256→8192 | 1 | 1 | 0.675 | 1.293 | 256 tiles、67,108,864 pixels 通过 |

完整 8K 探测逐 tile 校验 finite 和像素总数，消费后释放，不在显存收集输出；
实际 `.npy` memmap 写入另用小输入自动测试验证。PyTorch allocator 峰值不含桌面/驱动的额外占用。
同批 B=6 的 reserved 已超过本机物理显存容量；这是当前 Windows WDDM 环境的运行结果，
不能据此保证 6 GB 纯显存能容纳该配置。默认 B=1；它不减少 WSI 中参与 ABMIL 的 region 数。
N=1 时 ABMIL softmax 权重恒为 1，scorer 梯度为 0 属于正常数学结果；N=6 的 ABMIL 梯度非零。

8. **已运行测试**：`python -B -m unittest discover -s tests -v`，**21/21 通过**。
   覆盖官方 forward、checkpoint 输出/梯度、×2/×3/×4 数学等价、×32 N=1/N=6 小输入、
   SR backward、ResNet18-GN/ABMIL backward、联合 loss 的 0/0.1/1.0 三种权重、
   SR OFF 无 HR 读取且 P2W/add_mean 无梯度、一次共享 feature、chunk 和边缘 crop、
   GN 不依赖 region batch、一次参数更新、分类评估、memmap、CAMELYON16 loader/split。
   CPU 和 CUDA 独立等价性 gate 均通过；完整 8K 推理与上述实际尺寸 CUDA joint 探测通过。

**证据边界**：使用合成张量/临时图像；本机未提供真实 CAMELYON16 256/8192 图像。
未运行真实数据训练、长时间正式训练或报告预测性能。新训练入口每次启动都会先运行官方等价性 gate。
默认保留普通整袋计算图，显存随 N 增长；N=6 的通过结果不代表任意更大 WSI 都能放入 6 GB。

原始证据：`reports/equivalence.json`、`reports/equivalence_cuda.json`、`reports/tests.log`、
`reports/cuda_n1_256_bf16.json`、`reports/cuda_n6_256_bf16.json`、
`reports/cuda_n6_batch6_256_bf16.json`、`reports/cuda_full8k_n1_fp32.json`、
`reports/source_integrity.json`。
