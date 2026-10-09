# SPP D8 E2E 服务器运行

## 从 GitHub 获取代码

本地修改提交到 GitHub，AutoDL 拉取 main 进行训练。首次在 AutoDL 终端执行：

```bash
cd /root/autodl-tmp
git clone https://github.com/QInjian-yy/metasr-e2e.git
cd metasr-e2e/E2E_MetaSR
```

后续更新在仓库中执行 `git pull --ff-only`。训练需要 `E2E_MetaSR/` 内的
configs、models、vendor、reference、validation.py、downstream_train 等运行文件；
无需依赖外部 `Meta-SR-Pytorch-0.4.0/` 或原 `E2E/`，无需预训练权重。
网络和磁盘说明见 AutoDL 官方的 [Git 文档](https://www.autodl.com/docs/git/) 与
[目录说明](https://www.autodl.com/docs/env/)。

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

进入仓库中的 E2E_MetaSR，激活匹配的 CUDA PyTorch/torchvision 环境：

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

只需原来的训练参数，无需手填实验编号、GPU 或 Commit。
以下 tee 用于可选的终端日志保存，自动记录不依赖它。

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

## 自动记录与回传结果

训练开始自动保存唯一编号、GPU、Git Commit 等到 output/run_info.json。
正常结束或早停自动追加 results/summary.csv，并生成 experiments/EXP-*.md；
中断保持 incomplete，不会记成已完成。可选补汇总只需一行：

```bash
python -B experiments/summarize.py runs/spp_d8_full_fold0
```

在 AutoDL 的仓库根目录回传小型结果记录：

```bash
cd /root/autodl-tmp/metasr-e2e
git add E2E_MetaSR/results/summary.csv E2E_MetaSR/experiments/EXP-*.md
git commit -m "Record training results"
git push origin main
```

数据、权重、原始 runs/ 和 logs/ 已被忽略，不回传。
本地 Git 工作副本执行 `git pull --ff-only` 即可取回记录；
当前工作副本为 `D:\metasr-e2e-sync-20261009`，`D:\sdx_model\E2E_MetaSR` 是无 Git 的源目录。
完整记录说明见 [experiments/README.md](experiments/README.md)。

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

