"""Restore a MeanMIL/ABMIL checkpoint and reuse the existing streamed SR inference."""

import argparse
from pathlib import Path

import torch

from data_modes import checkpoint_data_spec
from engine import autocast, memory_summary
from infer_sr import reconstruct
from models.mil_model import model_from_checkpoint
from wsi_data import load_images


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
    spec = checkpoint_data_spec(saved)
    device = torch.device(args.device)
    model = model_from_checkpoint(saved, device).eval()
    lr = load_images([args.lr_image], spec.lr_size).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with autocast(device, saved["config"]["precision"]):
        reconstruct(model, lr, args.output)
    print({"output": str(args.output), "shape": [1, 3, spec.hr_size, spec.hr_size], **memory_summary(device)})


if __name__ == "__main__":
    main()
