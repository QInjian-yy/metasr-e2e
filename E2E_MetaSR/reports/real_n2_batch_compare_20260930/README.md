# 真实 N=2：整批与区域微批次比较

## 结论

- BF16：micro1 的峰值 allocated 降低 23.32%，但整体梯度相对 L2 差异为 14.8136%，未通过既有严格阈值。
- FP32（关闭 CUDA matmul / cuDNN TF32）：micro1 的峰值 allocated 降低 29.34%，梯度相对 L2 差异为 0.0000674871%，通过阈值，但不是逐位相同。
- 两次独立 BF16 整批运行的 33,919,386 个梯度元素逐位相同。低精度下的整批/分批差异在本次对照中可复现。
- 这些是 normal_001 的一次联合训练步结果，不能据此推断长期分类指标。

## 数据与控制条件

- 数据目录：`D:\ddata`，WSI：`normal_001`，真实标签：0。
- 两对真实输入：256×256 LR / 8192×8192 HR。
- 区域顺序：`normal_001_31168_86336.jpg`、`normal_001_57152_100032.jpg`。
- 两个 HR 裁剪框 `(y,x,h,w)`：`(4116,1603,256,256)`、`(5702,7172,256,256)`。
- 每次独立进程、seed=7、同一初始权重 SHA256：`44a6c7beafe2ab714fbb597b0a434721f4c7f7f95199cbc1531891911374ab34`。
- 正式 `MetaSRABMIL` 与 `engine.wsi_loss`；RDB checkpoint 开启，GroupNorm，SR 权重 0.1，无几何增强。
- 整批一次编码两个区域；micro1 分两次编码，保留普通计算图，合成同一个 `[2,512]` bag，再做一次 ABMIL、一次 backward、一次 Adam 更新。
- Adam：lr=1e-5，weight_decay=1e-4。
- GPU：RTX 3060 Laptop 6GB；PyTorch 2.13.0+cu130。
- 所有五次训练步均完成 forward、backward、optimizer_step，全部梯度有限。

## 实测显存

| 精度 | 编码方式 | 峰值 allocated / GiB | 峰值 reserved / GiB |
|---|---|---:|---:|
| BF16 | 整批 2 | 1.820028 | 2.408203 |
| BF16 | micro1 | 1.395573 | 1.767578 |
| FP32，TF32 关闭 | 整批 2 | 3.798807 | 5.048828 |
| FP32，TF32 关闭 | micro1 | 2.684354 | 3.400391 |

这是包含优化器更新的 PyTorch 峰值计数，不包含其他桌面应用，也不是整张显卡物理驻留显存的独占测量。
CPU 梯度快照在记录峰值和耗时之后导出。Windows WDDM 下，FP32 整批测试期间整卡占用一度达到 5835 MiB。
单步耗时波动明显，原始 JSON 保留测量值，本报告不据此给出稳定吞吐结论。

## 梯度核对

对比 371 个参数张量、33,919,386 个梯度元素。定义：

`相对 L2 差异 = ||g_micro1 - g_full||₂ / ||g_full||₂`。

沿用 `tests/test_engine.py` 的阈值：`abs(delta) <= 2e-5 + 2e-4 * abs(g_full)`，未针对 BF16 放宽。

| 指标 | BF16 | FP32，TF32 关闭 |
|---|---:|---:|
| 全部梯度相对 L2 差异 | 14.8136285% | 0.0000674871% |
| 最大绝对差异 | 0.097900390625 | 9.536743e-7 |
| 梯度余弦相似度 | 0.989364998 | 0.999999999999775 |
| 超阈值元素数 | 15,712,561 | 0 |
| 超阈值参数张量数 | 367 / 371 | 0 / 371 |
| 分类损失差异 | 0 | 0 |
| SR 损失差异 | 3.933907e-5 | 0 |
| 总损失差异 | 3.933907e-6 | 0 |

BF16 各模块的梯度相对 L2 差异：RDN 19.0832%、ResNet18 16.1976%、ABMIL 10.2645%、分类头 0.3318%、Pos2Weight 0.2447%。
FP32 的分类头、ABMIL、Pos2Weight 梯度逐位相同；RDN 和 ResNet18 存在通过阈值的微小浮点差异。
比较程序还检查了同一梯度与自身完全相同，并确认能检出人为将梯度乘以 2 的负对照。

FP32 与 BF16 重复对照支持差异主要来自低精度下随批大小变化的数值行为；本次未逐算子定位具体误差来源。

## 文件与复查

- `bf16_comparison.json`、`fp32_comparison.json`：每个参数和模块的误差统计。
- `bf16_full.json`、`bf16_micro1.json`、`fp32_full.json`、`fp32_micro1.json`：独立训练步报告。
- `bf16_full_repeat.json`：整批重复运行报告；逐元素重复检查写入 `bf16_comparison.json`。
- `*_gradients.pt`：仅含 CPU 梯度的比较快照。

在项目目录运行微批次测量：

```powershell
& D:\conda\envs\clam_latest\python.exe -u -B scripts\probe_cuda.py --data-root D:\ddata --slide-id normal_001 --n 2 --precision bf16 --sr-crop 256 --lambda-sr 0.1 --mode joint --use-region-microbatch --region-microbatch-size 1 --output reports\my_n2_micro1.json
```

复查已保存的梯度快照：

```powershell
& D:\conda\envs\clam_latest\python.exe -B audit\compare_real_batch_gradients.py reports\real_n2_batch_compare_20260930 bf16
& D:\conda\envs\clam_latest\python.exe -B audit\compare_real_batch_gradients.py reports\real_n2_batch_compare_20260930 fp32
```
