# A800 full-region batch 运行说明

每个训练 step 处理一张 WSI，自动读取该 WSI 的全部 N 个 region，一次 encoder forward，
一次 ABMIL、联合 backward 和 Adam.step。N 不需要手动固定。

## 上传与数据

上传代码包并解压得到 `E2E_MetaSR/`。代码包已包含 `vendor/official/`，
无需额外上传原来的 `Meta-SR-Pytorch-0.4.0/` 或 `E2E/`，无需预训练权重。
模型从随机初始化开始。代码包不含真实图像、运行结果或 Python 环境。

真实数据可放在项目外，下面用 `/path/to/camelyon16_data` 作为占位路径：

```text
/path/to/camelyon16_data/
  images_256/                 # 256×256 RGB LR
  images_8192/                # 对应的原始 8192×8192 HR
  manifests/patch_manifest.csv
```

LR、HR 文件名必须对应 manifest 的 filename。使用与这些图像匹配的 manifest；
不要用其他数据集的 manifest 覆盖。HR 保留 8192 原图，训练时随机取 256 crop，
不要提前把 HR resize 到 256。服务器已有标签和划分时，用下面命令中的
`--labels-csv` 和 `--split-dir` 显式指定服务器文件，不使用代码包自带的默认 CSV。
标签 CSV 需要 `case_id,slide_id,label` 列（0/1）；划分目录需要 `splits_0.csv` 等文件，
包含 `train,val` 列，值为 case_id。

## 环境与配置

激活服务器已有的 CUDA PyTorch 环境，进入解压后的 E2E_MetaSR 目录执行：

```bash
python -m pip install -r requirements.txt
python -c "import torch, torchvision; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

若没有可用的 CUDA PyTorch 环境，先根据服务器驱动安装匹配的 torch/torchvision：
https://pytorch.org/get-started/locally/

`configs/baseline.yaml` 的关键配置：

```yaml
sr_train_crop: 256
lambda_sr: 0.1
precision: bf16
training:
  use_region_microbatch: false
  region_microbatch_size: 1
```

`metasr.scale=32`，`metasr.checkpoint_rdb=true`。microbatch 开关为 false 时，
size=1 不生效；所有 N 个 region 一起编码。SR decoder 的 LR chunk 仍正常使用。

## 先验证，再训练

```bash
python -B -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=0 python -B validation.py --device cuda:0 --output reports/equivalence_a800.json
CUDA_VISIBLE_DEVICES=0 python -B scripts/probe_cuda.py --n 37 --size 256 --sr-crop 256 --precision bf16 --lambda-sr 0.1 --output reports/a800_n37_crop256.json
```

当前随包 manifest 最大 region 数为 37。若服务器使用其他 manifest，按其最大 N 探测。
probe 默认 full-region batch，不要添加 `--use-region-microbatch`。
它使用合成张量，执行完整 forward/backward/Adam.step，不读取真实图像。
不能只凭进程退出码认定通过；读取 JSON：

```bash
python - <<'PY'
import json
from pathlib import Path
r = json.loads(Path('reports/a800_n37_crop256.json').read_text())
print('status:', r['status'], 'stages:', r['training_stages'])
print('allocated GiB:', r['max_memory_allocated'] / 2**30)
print('reserved GiB:', r['max_memory_reserved'] / 2**30)
print('step seconds:', r['elapsed_seconds'])
assert r['status'] == 'passed' and all(r['training_stages'].values()), r
PY
```

上述验证通过、显存有余量后，替换服务器已有的数据、标签和划分路径再启动 fold 0：

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_e2e.py \
  --config configs/baseline.yaml \
  --data-root /path/to/camelyon16_data \
  --labels-csv /path/to/camelyon16_labels.csv \
  --split-dir /path/to/cross_validation_splits \
  --fold 0 \
  --output runs/fold0_full_crop256
```

输出目录必须为空或不存在。当前入口不支持 resume。默认最多 20 epochs，
按验证 AUC 早停；输出 `history.csv`、`best.pth`、`last.pth` 和运行配置。
远程断连后仍需运行时，可在已有 tmux/screen 会话中执行同一训练命令。

若只想先做一轮真实数据检查，将配置复制为新的 YAML，把 `epochs` 改为 1，
用 `--config` 指向该文件，并指定另一个空输出目录。
