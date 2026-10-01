"""Explicit full 8K inference; stream GPU tiles directly into a CPU/disk .npy array."""

import argparse
from pathlib import Path

import numpy as np
import torch

from engine import autocast, memory_summary
from models.baseline import MetaSRABMIL
from wsi_data import load_images


@torch.no_grad()
def reconstruct(model, lr, output_path):
    feature = model.sr.extract_features(lr)
    b, _, h, w = feature.shape
    scale = model.sr.scale
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32,
                                     shape=(b, 3, h*scale, w*scale))
    for y, x, tile in model.sr.iter_full_sr(feature):
        output[:, :, y:y+tile.shape[2], x:x+tile.shape[3]] = tile.float().cpu().numpy()
    output.flush()
    del output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lr-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("format") != "metasr-abmil-v1":
        raise ValueError("Expected a MetaSR baseline checkpoint")
    device = torch.device(args.device)
    model = MetaSRABMIL(**saved["config"]["metasr"]).to(device).eval()
    model.load_state_dict(saved["model_state"], strict=True)
    lr = load_images([args.lr_image], 256).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with autocast(device, saved["config"]["precision"]):
        reconstruct(model, lr, args.output)
    print({"output": str(args.output), "shape": [1, 3, 8192, 8192], **memory_summary(device)})


if __name__ == "__main__":
    main()
