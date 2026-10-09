# SPP D8 E2E 服务器运行

## 上传

上传清理后的整个 `E2E_MetaSR/` 文件夹。保留 `configs/`、`models/`、
`vendor/`、`reference/trainer.py`、`validation.py`、`downstream_train/` 及根目录运行代码。
无需额外上传 `Meta-SR-Pytorch-0.4.0/` 或原 `E2E/`，无需预训练权重。
历史测试输出和调试脚本已删除，旧 probe/unittest 命令不再适用。

真实图片可以放在项目外：

```text
/path/to/full_data/
  images_256/                 # LR 256×256 RGB
  images_8192/                # 配对 HR 8192×8192 RGB
  manifests/patch_manifest.csv
```

图片文件名需对应 manifest 的 filename。不要把 HR 预先缩小。
现有三折划分无需重做：默认使用 `downstream_train/` 中的标签和三个 split 文件；
使用服务器已有划分时，将下方 `--labels-csv` 和 `--split-dir` 替换成对应路径。

## 环境

进入上传后的 E2E_MetaSR，激活匹配的 CUDA PyTorch/torchvision 环境：

```bash
python -m pip install -r requirements.txt
```

`configs/spp_d8.yaml` 使用 SPP、8 个 RDB、BF16、RDB checkpoint、
scale=32、lambda_sr=0.1、256 HR crop。BF16 不支持时会报错，
需要明确选择 FP32 配置；不会自动切换精度。

## 整批训练

每个 step 处理一张 WSI 的全部 N 个 region，N 从数据自动读取。
默认 `training.use_region_microbatch=false`。

在 Linux 服务器执行，替换真实数据路径：

```bash
mkdir -p logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u -B train_e2e.py \
  --config configs/spp_d8.yaml \
  --data-root /path/to/full_data \
  --labels-csv downstream_train/camelyon16_labels.csv \
  --split-dir downstream_train \
  --fold 0 \
  --output runs/spp_d8_full_fold0 \
  2>&1 | tee logs/spp_d8_full_fold0.log
```

分别将 fold 改为 1、2，并使用不同的 output 和日志文件，运行其他两折。
日志写到 output 目录之外，避免启动时的非空目录保护阻止运行。
当前不支持 resume，已有实验目录不能覆盖。训练启动自动执行保留的校验逻辑。
可在服务器已有 tmux/screen 会话中运行。

## 显式分批

复制 `configs/spp_d8.yaml` 为 `configs/spp_d8_micro2.yaml`，只修改：

```yaml
training:
  use_region_microbatch: true
  region_microbatch_size: 2
```

训练命令改用 `--config configs/spp_d8_micro2.yaml`，
并使用新的 output 和日志路径。没有训练入口的 microbatch CLI 参数。
N=5 时按 2、2、1 编码，全部 embedding 汇总后仍只执行一次 ABMIL/backward/step。
分批保留计算图；BF16 下可能与整批产生数值差异。显存不足会报错，不会自动截断数据。

## 分类评估与完整 SR 推理

```bash
CUDA_VISIBLE_DEVICES=0 python -B eval_e2e.py \
  --checkpoint runs/spp_d8_full_fold0/best.pth \
  --data-root /path/to/full_data \
  --labels-csv downstream_train/camelyon16_labels.csv \
  --split-dir downstream_train \
  --predictions-csv runs/spp_d8_full_fold0/eval_predictions.csv

CUDA_VISIBLE_DEVICES=0 python -B infer_sr.py \
  --checkpoint runs/spp_d8_full_fold0/best.pth \
  --lr-image /path/to/full_data/images_256/region.png \
  --output runs/spp_d8_full_fold0/region_sr.npy
```

评估使用 checkpoint 中保存的 fold，并核对原 manifest、标签和 split 的哈希。
SR 推理为单张 LR 输出完整 8K，逐块写入磁盘；每张 FP32 文件约 768 MiB。
评估和推理按 checkpoint 保存的实际结构构造模型并严格加载。

训练指标与峰值显存保存到 `history.csv`，终端日志通过 tee 保存。
训练损失含分类与 SR；val_loss 只含分类。完整资源需求需在目标服务器实测。

