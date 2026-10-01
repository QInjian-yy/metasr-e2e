# D8 真实 CUDA：N=2 整批与 micro1 训练步等价性审计

日期：2026-09-30。结论：**BF16 未通过；FP32 梯度通过，但 Adam 更新的严格逐元素检查未通过。不能据此认定为严格等价或保证长期性能不变。**

## 实验条件与范围

- RTX 3060 Laptop 6GB，PyTorch 2.13.0+cu130，CUDA 13.0。
- `configs/rdn_d8.yaml`，真实 `D:\ddata` 的 `normal_001`，两对 256×256 / 8192×8192 RGB 图片，真实标签 0。
- RDN D=8 / C=8 / G=64，checkpoint 开启；完整模型 22,970,266 个参数，227 个参数张量全部参与梯度/更新检查。
- seed=1，同初始权重 SHA256：`f60161eef8a0a3d2453e7b93a8f91bb6f7d910f11a9f917d84d3ad48f0b054c4`。
- 固定图片顺序和实际 HR crop `(y,x,h,w)`：`(7700,6701,256,256)`、`(5804,3659,256,256)`。
- Adam：lr=1e-5，weight_decay=1e-4，betas=(0.9,0.999)，eps=1e-8；每次都是相同初始化的第一个训练步，optimizer state 初始为空。
- 直接调用生产 `engine.wsi_loss`，随后一次 backward、一次 Adam step；CE + 0.1×平均 SR L1。
- 整批编码形状 `[2,64,256,256]`；micro1 两次 `[1,64,256,256]`。两者都只调用一次 ABMIL，输入完整 `[2,512]`，hook 实测 requires_grad/grad_fn 均保留。没有分成两个 WSI 优化器更新，没有 detach 或梯度回放。
- 共完成 9 次真实 CUDA 前向、反向、Adam 更新：普通 BF16/FP32 各 full、micro1、full_repeat；确定性 FP32 三次。所有执行成功、梯度有限。执行日志的 `status=passed` 只代表该次训练步完成。

## 阈值

逐元素条件为 `abs(actual-reference) <= atol + rtol*abs(reference)`。以下阈值在比较前设置，失败后未放宽。相对 L2 为 `||actual-reference||2 / ||reference||2`，更新指实际 `step 后参数 - 初始参数`。

| 检查 | atol | rtol |
|---|---:|---:|
| logits / loss | 3e-6 | 3e-5 |
| 梯度（沿用 tests/test_engine.py） | 2e-5 | 2e-4 |
| 更新后参数 | 1e-7 | 1e-5 |
| 参数更新量 | 1e-7 | 1e-4 |
| Adam 一阶矩 | 2e-6 | 2e-4 |
| Adam 二阶矩 | 2e-8 | 4e-4 |

这是本次审计的数值判定标准，不是模型性能下降的判定标准。

## 整批与 micro1 实测

| 指标 | BF16 普通 CUDA | FP32 普通 CUDA | FP32 确定性 CUDA |
|---|---:|---:|---:|
| logits 最大绝对差 | 0.0029296875 | 0 | 0 |
| forward 通过 | 否 | 是 | 是 |
| 梯度相对 L2 差 | 13.6858% | 0.0000652404% | 0.0000852354% |
| 梯度最大绝对差 | 0.13134765625 | 6.10948e-7 | 9.61125e-7 |
| 梯度超阈值元素 | 11,893,308 | 0 | 0 |
| 更新量相对 L2 差 | 45.1954% | 0.0179526% | 0.0278293% |
| 更新后参数最大绝对差 | 2.00272e-5 | 3.94369e-6 | 1.09002e-5 |
| 更新后参数超阈值元素 | 1,478,544 | 199 | 219 |
| 更新量超阈值元素 | 1,604,596 | 373 | 430 |
| Adam 两个矩均通过 | 否 | 是 | 是 |
| 完整训练步通过 | **否** | **否** | **否** |

原始结果：[BF16](bf16_comparison.json)、[普通 FP32](fp32_comparison.json)、[确定性 FP32](deterministic/fp32_comparison.json)。比较程序对这些失败结果返回退出码 1，并保留完整 JSON。

## 重复运行对照：不能把所有波动都归因于分批

- BF16 full 与 full_repeat：输出、全部梯度、更新后参数、Adam 状态完全相同。
- 普通 FP32 full 与 full_repeat：输出相同；梯度相对 L2 差 0.0000603723%，更新相对 L2 差 0.0258536%，最大参数差 9.70624e-6。因此普通 FP32 的 full/micro 差异与自身重复波动量级接近，不能全部归因于 microbatch。
- 为排除这项混杂，额外开启 `torch.use_deterministic_algorithms(True)`、`cudnn.deterministic=True`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`；FP32 的 matmul/cuDNN TF32 均关闭。
- 确定性 FP32 full 与 full_repeat：输出、全部梯度、更新后参数、Adam 状态完全相同。但 full 与 micro1 仍未通过完整训练步的逐元素检查。确定性保证本次相同执行方式可复现，未使不同 batch shape 的浮点计算逐位一致。
- 比较器自比为零，并确认把一个非零梯度乘 2 会被拒绝，防止失效比较器误报通过。

## FP32：为什么梯度通过，参数更新仍可能失败

确定性试验中最大差异发生在 `sr.GFF.0.weight` 的展平索引 1740。初始值 0.021303441375494003，权重衰减贡献约 +2.13034414e-6。

| 量 | full | micro1 |
|---|---:|---:|
| 原始梯度 | -2.14891043e-6 | -2.12248415e-6 |
| 加权重衰减后的有效梯度（由 Adam 一阶矩还原） | -1.85663418e-8 | +7.85993604e-9 |
| 实际参数更新 | +6.49876893e-6 | -4.40143049e-6 |

原始梯度与权重衰减近乎抵消，有效梯度接近 eps=1e-8；小的原始梯度差可以改变有效梯度符号，被 Adam 放大为局部更新差异。用保存的 Adam 两个矩重算更新，吻合实测。见 [确定性诊断](deterministic/fp32_worst_update_diagnostic.json) 和 [普通 FP32 诊断](fp32_worst_update_diagnostic.json)。未为获得通过而改 eps、weight_decay、lr 或阈值。

## 显存（普通 CUDA 实测）

| 精度 | full allocated / reserved | micro1 allocated / reserved |
|---|---:|---:|
| BF16 | 1.528 / 1.820 GiB | 1.104 / 1.334 GiB |
| FP32 | 3.258 / 3.697 GiB | 2.138 / 2.695 GiB |

均包含前向、反向和首次 Adam 状态分配，CPU 快照在显存峰值采集后保存。allocated/reserved 是 PyTorch 指标，不代表 WDDM 下进程独占的全部物理显存。单次耗时含读图等开销，不据此声称稳定提速。

## 如何复查或测试新的 N

在项目根目录用下面的 PowerShell 命令，设置一个新的输出目录避免覆盖已有证据。`--deterministic` 用于确定性控制组；省略它可复查普通运行。BF16 单独更换 `$auditPrecision` 后运行。N>2 需要同一 WSI 真实存在足够的配对图片，且整批参考能在同一 GPU 上完成。

```powershell
$auditPython = 'D:\conda\envs\clam_latest\python.exe'
$auditDirectory = 'reports/d8_n2_recheck'
$auditPrecision = 'fp32'
$auditN = 2
foreach ($auditMode in @('full', 'micro1', 'full_repeat')) {
    $auditBatch = if ($auditMode -eq 'micro1') { '--use-region-microbatch' } else { '--no-use-region-microbatch' }
    & $auditPython -u -B scripts/probe_cuda.py --config configs/rdn_d8.yaml --data-root D:\ddata --slide-id normal_001 --n $auditN --precision $auditPrecision --deterministic $auditBatch --region-microbatch-size 1 --gradient-output "$auditDirectory/${auditPrecision}_${auditMode}_gradients.pt" --training-state-output "$auditDirectory/${auditPrecision}_${auditMode}_state.pt" --output "$auditDirectory/${auditPrecision}_${auditMode}.json"
    if ($LASTEXITCODE -ne 0) { throw "Probe failed: $auditMode" }
}
& $auditPython -u -B audit/compare_real_batch_gradients.py $auditDirectory $auditPrecision
```

## 使用结论与未覆盖项

数学上的损失和完整 WSI 聚合结构保持一致；实际 CUDA 浮点轨迹并非严格一致。BF16 差异明显；FP32 整体差异很小，但不能掩盖少量逐元素更新失败。**当前维持 YAML 的整批编码默认值，不把 micro1 标为已验证无损。**

这次只覆盖一个真实 WSI、seed=1、相同随机初始化下的首次 Adam 步。没有证明长期 AUC/ACC/BACC 下降，也没有证明它们不变。后续 N>2 应继续对照输出、全参数梯度、参数更新和 Adam 状态，并增加多个 seed、连续训练步及实际 checkpoint/optimizer 状态；若要宣称性能不降低，还需要匹配训练条件的完整实验。此处没有把未来测试标记为通过。
