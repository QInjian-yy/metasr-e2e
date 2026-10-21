# native512 数据支持与旧功能兼容性报告

日期：2026-10-21。修改前提交 `cf4e3104c740436e6640586a58b44ffab10002ca`，工作区干净；实现位于本地分支 `codex/native512-scale16`。没有运行正式训练，没有改写历史实验、原 YAML、原测试或任何历史 checkpoint。

机器可读结果：[native512_validation_20261021.json](native512_validation_20261021.json)。完整回归日志：[native512_tests_20261021.log](native512_tests_20261021.log)。

## 结论与验证边界

| 要求 | 实际结论 |
|---|---|
| 旧 256 数据按旧命令运行 | PASS：缺省 legacy256；原 68 项测试全部保留并通过，旧 CLI 默认标签/划分路径不变 |
| 旧 checkpoint 正常加载 | PASS：v1/v2 合成回归；真实 D4 fold1 best.pth 严格加载通过，原文件 hash 不变 |
| 新 512 数据加载 | PASS：真实 schema 的临时 JPG、Fold 0/1、完整 WSI 分组；25 对真实 native 验证样例的文件头与 LR RGB 解码通过 |
| 目标 tissue20 完整数据加载 | NOT TESTED：本地只有真实清单、report 和划分，没有其 images_512/images_8192 |
| scale16 SR crop | PASS：小尺寸官方 Decoder oracle、边界/非对齐/随机 crop、反向、streaming；512 CUDA SR=0.1 单步通过 |
| 数据增强 | PASS：16 种 H/V/Rot90 组合、416 个虚拟 8K crop 检查、已保存 JPEG 解码像素 oracle |
| 新旧模式隔离 | PASS：显式 mode、scale 合同、路径安全、native report 与计数检查；旧模式拒绝明确 native 数据证据 |
| 完整 8192 SR 导出 | NOT TESTED：小图 streaming 与入口尺寸测试通过，本轮未生成完整 8K 测试张量/输出 |
| 正式训练 | 未启动；等待用户确认 |

这里的 PASS 指上述范围，不表示新数据已完成全部图像内容、病灶覆盖、源 WSI 像素或模型收敛验证。

## 修改前检查与实现范围

修改前发现的固定假设：

- `wsi_data.py`：从 filename 构造 images_256/images_8192 配对，按 case-level CSV 分组。
- `engine.py`：训练及评估固定读取 256；HR crop 抽样边界固定为 8192。
- `train_e2e.py`：scale=32 校验、固定 legacy 标签默认路径，checkpoint 没有显式输入数据合同。
- `eval_e2e.py`：未从 checkpoint 传递数据模式；原 provenance 有 manifest/labels/split 三个 hash。
- `infer_sr.py`：固定读取 256，重建内部本来就按实际 feature 空间尺寸与 scale 输出。
- `validation.py`：默认只做 scale2/3/4 小图等价检查。

`models/metasr.py` 的 feature extraction、Pos2Weight、position/weight table、unfold、tile/chunk、边界和 streaming 均已按 `self.scale` 工作；SPP、ABMIL 与配对增强也无需改数学。因此采用新增数据合同和独立 native 解析层，未改这些模块。

## 显式数据合同

| 内容 | legacy256 | native512 |
|---|---|---|
| 选择方式 | 不写字段或 data_mode: legacy256 | 必须显式 data_mode: native512 |
| 输入 | images_256，256×256 | images_512，512×512 |
| HR | images_8192，8192×8192 | images_8192，8192×8192 |
| Meta-SR scale | 32 | 16 |
| 默认清单 | manifests/patch_manifest.csv，filename/slide_id | 显式配置详细导出清单 |
| 默认 labels/split | 原项目 ROOT/downstream_train | DATA_ROOT/downstream_train |
| provenance | 原 manifest/labels/split | 详细 manifest/labels/split，加 verification/report.json |
| 图像操作 | 原 RGB float32 /255，无 resize | 相同 RGB float32 /255，无 resize |

模型保持五种 encoder：spp=3200、spp_1344=1344、spp_ln=3200、spp_avg=3200、resnet18=512。RDN 输出通道、SPP 层级、ABMIL、分类头、损失、Adam、early stopping、增强抽样、checkpoint_rdb 和 MeanShift 行为没有改变。

新保存的 best/last 仍采用 metasr-abmil-v2，增加独立顶层 `data_spec={mode,lr_size,hr_size,scale}`；architecture 字典和权重键名不变。无新字段的历史 checkpoint 仍走原加载路径；显式模式或新 data_spec 不一致时拒绝加载。历史小尺寸 model-only checkpoint 的既有 API 也保留。

## 真实目标清单与坐标来源

服务器根目录：

```text
c16_native_l1_8192_l5_512_tissue20_jpg90_nonoverlap_v2_train/
  downstream_train/
  images_512/
  images_8192/
  manifests/       # 训练派生清单
  Manifests/       # 原始详细清单
  verification/
```

Linux 区分 `manifests` 与 `Manifests`。根据用户提供的本地真实文件：

- `D:\c16_metadata\Manifests_upper\patch_manifest.csv` 对应服务器详细清单，表头为 `slide_id,x,y,hr_level,lr_level,scale,tissue_ratio,padded,hr_path,lr_path,hr_pixel_sha256,lr_pixel_sha256`。
- `D:\c16_metadata\manifests_lower\manifests\patch_manifest.csv` 是派生版，表头为 `filename,slide_id,patient_id,x,y`。
- 两份清单的 `(slide_id,x,y,basename)` 集合精确相等，slide_manifest 的计数也一致。

新示例明确配置 `data_manifest: Manifests/patch_manifest.csv`，优先使用真实 lr_path/hr_path，不根据文件名推断坐标，不回退猜测大小写目录，也不凭空补齐派生清单缺失的几何字段。详细清单路径若不同，需要显式修改该新配置字段。

已核对目标数据元信息：4343 对 JPG，264 张非零块 WSI，153 normal / 111 tumor，每张 1–45 Region；统一 levels=1/5、scale=16。report 为 complete，tissue_threshold=0.20，JPEG quality=90/subsampling=0/optimize=true。报告有 269 条已处理 WSI：5 张零块 normal_016/019/027/042/045，另显式跳过 normal_144；标签与有效清单完全吻合。

| Fold | Train WSI | Val WSI | Train Region | Val Region |
|---|---:|---:|---:|---:|
| 0 | 176 | 88 | 2784 | 1559 |
| 1 | 176 | 88 | 2839 | 1504 |
| 2 | 176 | 88 | 3063 | 1280 |

各折 case 无交集或遗漏，三个验证集互不重叠并覆盖全部病例。不要将这套 264 WSI 的划分替换为旧实验 262 WSI 的划分。

已阅读实际 v2 生成脚本：OpenSlide 的 `read_region((x,y),1,(8192,8192))` 与 `read_region((x,y),5,(512,512))` 共用 Level0 原点。真实 downsample 为 2/32，视野都是 Level0 的 16384×16384。训练 decoder 和 GT crop 的 `(top,left,height,width)` 是已导出 HR 图片的局部像素坐标，不再乘 2；只有回读源 WSI 的 HR 局部偏移才涉及该换算。

真实 report 明确 `pixel_zero_alignment_claimed=false`，原生采样相位另行审核，probe 存在亚 LR 像素相位差。本轮保证局部 crop/增强的数学对应，不声称两种原生层已验证为逐像素零偏移；没有 resize、配准或移动像素。JPEG 的验证 oracle 是保存文件解码后的像素，不要求它与未压缩源 WSI 完全相同。

## 新配置比较

- [spp_d1_native512_aug_sr0.yaml](../configs/spp_d1_native512_aug_sr0.yaml)：原 SPP，整批 false/size1，100 epochs / patience15。
- [spp_d1_1344_native512_aug_sr0.yaml](../configs/spp_d1_1344_native512_aug_sr0.yaml)：仅 classification_encoder 改为 spp_1344，其余字段与上项完全相同。
- 两个 `_micro1.yaml` 是明确独立的显存配置，只把 use_region_microbatch 改为 true；不把它们称为与原整批完全相同的实验条件。
- [spp_d1_native512_aug_sr01_smoke.yaml](../configs/spp_d1_native512_aug_sr01_smoke.yaml)：单独的 SR=0.1 / micro1 / 1 epoch 示例，本轮未运行它进行正式数据训练。

将主 SPP 新配置移除 data_mode/data_manifest 并把 scale 恢复为 32 后，与已记录 `runs/spp_d1_aug_sr0_a100/config.yaml` 无其他差异。

## 实际测试

修改前原 68/68 PASS；修改后 **103/103 PASS，无跳过或放宽旧断言**，最终完整回归耗时 29.523 秒。新测试共 35 项：

| 新测试文件 | 项数 | 覆盖 |
|---|---:|---|
| test_native512_data.py | 14 | 真实 schema、JPG/RGB、尺寸、SR0 不触碰 HR、缺图、重复、路径穿越/symlink、report/计数和模式混用拒绝 |
| test_native512_geometry.py | 3 | 16 组合、416 个虚拟 8K crop、JPEG 解码后的增强/crop、真实 512 JPG 读取 |
| test_native512_integration.py | 13 | 五 encoder、两种模式、SR0/SR0.1、梯度、full/micro1/micro2 FP32 等价、Fold0/1 完整袋、checkpoint、train/eval/infer 和 hash |
| test_native512_metasr.py | 5 | scale16 Pos2Weight、官方 oracle、随机/边界/非对齐 crop、参数和 feature 梯度、checkpoint_rdb、streaming |

额外实际执行：

- native 启动 gate 使用 `(2,3,4,16)`，D1/CPU/FP32 PASS；scale16 输出为 `[1,3,64,80]`，最大前向绝对差 4.7684e-6、参数梯度差 8.3447e-7，沿用原 assert_close 容差，未放宽断言。
- 从 Git 基线内存加载旧代码，与新代码对比：五 encoder 初始化/RNG/权重/小图输出逐位一致；legacy SR0/SR0.1 Engine 的输出、有效梯度及结束 RNG 逐位一致（便宜可微 decoder fixture 隔离流程）；默认 equivalence gate 的结果和结束 RNG 逐位一致。
- 真实旧 `runs/spp_d4_fold1_sr0_aug_300/best.pth`（epoch104/fold1）严格加载通过，文件 SHA256 不变；未重跑完整数据集推理。
- `D:\ddata\native_jpeg_validation_20261021\mixed_export` 的 25 对真实 native 样例（7 PNG/18 JPG，2 WSI）通过 LR/HR 文件头与全部 LR RGB/float32 解码。这是另一组 tissue35 验证样例，不是目标完整训练集，不使用 test_001 伪造训练标签。
- train/eval/infer 的 `--help` 检查通过；最后修改的新模式 hash 报错信息后又运行 13 项 integration 回归，通过。

## CUDA 单步与显存

本机 RTX 3060 Laptop 6GB、Torch 2.13.0+cu130；每次为 3 个合成 512×512 RGB Region、D1、真实 feature/decoder 数学、一个完整 ABMIL、一次 backward 和一次 Adam.step。HR GT 仅合成 256 crop，未分配完整 8K 张量。所有检查均 PASS，没有 OOM。

| Encoder | 精度 | Microbatch | SR | allocated 峰值 GiB | reserved 峰值 GiB |
|---|---|---:|---:|---:|---:|
| spp | BF16 | 1 | 0 | 2.6400 | 3.2285 |
| spp | BF16 | 2 | 0 | 4.6082 | 5.3203 |
| spp | BF16 | 1 | 0.1 | 2.6992 | 3.3125 |
| spp | FP32 | 1 | 0 | 5.2393 | 6.3750 |
| spp_1344 | BF16 | 1 | 0 | 2.6355 | 3.2344 |

SR0 的 P2W 无梯度，SR0.1 的 P2W 有有效梯度；RDN/ABMIL/Classifier 梯度均有限且非零。每次 MIL 输入均含全部 3 个 Region。BF16 micro1/2 的本次 logits 相同，但不据此宣称一般 BF16 逐位等价。

这些值仅覆盖 N=3/D1。最大 45 个 Region、D4/D8、整批、新数据 I/O、实际训练速度和更长运行均未做 CUDA 压力验证。microbatch 保留整个 bag 的计算图，显存不能按单张 Region 直接外推。6GB 建议从显式 micro1/BF16 配置做真实 bag 单步核验，不承诺它适配任意 N。Windows 本次 FP32 的 reserved API 记录为 6.375 GiB，不能将其解释为有充足独占物理显存余量；保留原始测量值，不推算显存保证。

## 文件修改原因

| 文件 | 修改目的 |
|---|---|
| data_modes.py（新增） | 明确两套尺寸/scale 合同，checkpoint 元数据验证 |
| native_data.py（新增） | 独立解析真实 native schema，路径/报告/配对/文件头/计数与防混用检查 |
| wsi_data.py | native 显式分支，原 legacy 分组/划分继续复用，记录实际来源文件 |
| engine.py | 根据 mode 选择输入边长，检测样本/模型不匹配；优化目标不变 |
| train_e2e.py | 模式校验、新数据默认 labels/split、native scale16 gate、保存 data_spec 和来源 |
| eval_e2e.py | 按 checkpoint 选择模式/清单，核对四项来源 hash |
| infer_sr.py | 按 checkpoint 读取 256/512；原 streaming 重建不改 |
| models/baseline.py | 只在 checkpoint 重建入口增加数据合同校验，未改网络 |
| validation.py | 只新增可选 scales 参数，默认 gate 完全保留 |
| 新 YAML ×5 | 两种 encoder、独立 micro1 变体和 SR-on 小规模示例 |
| 新测试 ×4、probe_native512.py | CPU 合同/几何/梯度/兼容回归和可重复 CUDA 合成单步 |
| README 与本报告/JSON/测试日志 | 数据约定、实测结果、限制和命令 |

原 `models/metasr.py`、所有 SPP 模块、ABMIL、augmentation.py、paired_geometry.py、官方 vendor 均未修改。修改前已跟踪的全部旧 YAML、旧测试、downstream CSV、runs/config/history、logs、experiments、summary.csv 均经 SHA256 核对保持不变。

## 验证与后续命令

以下命令已列出供复核；本轮只执行测试和合成 probe，没有执行正式训练命令。

在 E2E_MetaSR 目录运行全套测试：

```bash
python -B -m unittest discover -s tests -v
```

可重复的合成单步（output 必须不存在）：

```bash
python -B scripts/probe_native512.py \
  --config configs/spp_d1_native512_aug_sr0.yaml \
  --regions 3 --microbatch 1 --precision bf16 --lambda-sr 0.1 \
  --output /tmp/native512_sr01_probe.json
```

服务器真实数据预检，不训练：

```bash
export DATA_ROOT=/media/ub/shidang_plu/MT/py_project/yujian/double_wsi/CAMELYON16_WSI_DATA/c16_native_l1_8192_l5_512_tissue20_jpg90_nonoverlap_v2_train
python -B - <<'PY'
import os
from wsi_data import load_fold_datasets
for fold in (0, 1, 2):
    train, val = load_fold_datasets(
        os.environ['DATA_ROOT'], fold, data_mode='native512',
        manifest_csv='Manifests/patch_manifest.csv', require_hr=False)
    print(fold, len(train), len(val),
          sum(s['n_regions'] for s in train.samples),
          sum(s['n_regions'] for s in val.samples))
PY
```

这会实际要求全部 LR 存在且 header 为 512；要额外检查全部 HR 的存在/尺寸，将 require_hr 改为 True，不会将 8K HR 全部搬到 GPU。本机无目标图片，未执行该完整目标预检。

用户确认后，在新输出目录启动，例如显式 micro1 方案：

```bash
python -u -B train_e2e.py \
  --config configs/spp_d1_native512_aug_sr0_micro1.yaml \
  --data-root "$DATA_ROOT" --fold 0 \
  --output runs/spp_d1_native512_fold0_s1_micro1
```

SPP-1344 改用 `configs/spp_d1_1344_native512_aug_sr0_micro1.yaml` 并指定另一个 output。整批对照使用不带 `_micro1` 的两个主配置；不能在显存未验证时假定最大 bag 可整批放入。

评估与显式 SR 推理命令保持原 CLI：

```bash
python -B eval_e2e.py \
  --checkpoint runs/spp_d1_native512_fold0_s1_micro1/best.pth \
  --data-root "$DATA_ROOT"

python -B infer_sr.py \
  --checkpoint runs/spp_d1_native512_fold0_s1_micro1/best.pth \
  --lr-image "$DATA_ROOT/images_512/REAL_MANIFEST_FILENAME.jpg" \
  --output /tmp/native512_region_sr.npy
```

最后一个文件名须从真实详细清单选取；两模式全图输出合同都是 `[1,3,8192,8192]`。SR0 的 decoder 未得到 SR 监督，该命令仅说明接口，不代表其重建质量。
