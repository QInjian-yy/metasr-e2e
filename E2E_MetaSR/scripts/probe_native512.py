"""One synthetic native512 WSI step; never reads a dataset or starts training."""

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from engine import gradient_norm, memory_summary, wsi_loss
from models.baseline import MetaSRABMIL
from train_e2e import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--regions", type=int, default=3)
    parser.add_argument("--microbatch", type=int, choices=(1, 2), default=1)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--lambda-sr", type=float, choices=(0.0, 0.1), default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.regions < 2:
        parser.error("Use at least two regions to check full-bag attention gradients")
    if args.output.exists():
        raise FileExistsError(args.output)
    config = load_config(args.config)
    if config.get("data_mode") != "native512":
        parser.error("This probe requires an explicit native512 config")
    config.update(precision=args.precision, lambda_sr=args.lambda_sr)
    config["training"] = {"use_region_microbatch": True, "region_microbatch_size": args.microbatch}
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 unsupported on this GPU")
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    lr = torch.rand(args.regions, 3, 512, 512)
    crop = config["sr_train_crop"]
    truth = torch.rand(args.regions, 3, crop, crop) if args.lambda_sr else None
    sample = {"data_mode": "native512", "slide_id": "synthetic-native512", "label": 1,
              "n_regions": args.regions, "lr_paths": list(range(args.regions)),
              "hr_paths": list(range(args.regions))}
    model = MetaSRABMIL(classification_encoder=config["classification_encoder"],
                       classification_head=config["classification_head"], **config["metasr"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    bags = []
    model.mil_head.register_forward_pre_hook(lambda module, inputs: bags.append(list(inputs[0].shape)))
    report = {"synthetic": True, "config": config, "input_shape": list(lr.shape), "mil_inputs": bags,
              "device": str(device), "torch": torch.__version__, "optimizer_steps": 0}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        report.update(gpu=torch.cuda.get_device_name(device), free_before_bytes=free, total_bytes=total)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    phase = "forward"

    def load_lr(paths, size):
        assert size == 512
        return lr[paths]

    def load_hr(paths, box):
        if truth is None:
            raise AssertionError("SR-off attempted an HR read")
        assert 0 <= box[0] <= 8192 - crop and 0 <= box[1] <= 8192 - crop
        return truth[paths]

    try:
        with patch("engine.load_images", side_effect=load_lr), patch("engine.load_hr_crops", side_effect=load_hr):
            model.train()
            total_loss, cls, sr, logits = wsi_loss(model, sample, device, config)
            assert torch.isfinite(total_loss)
            phase = "backward"
            total_loss.backward()
            norms = {name: gradient_norm(module) for name, module in
                     (("rdn", model.sr), ("abmil", model.mil_head),
                      ("classifier", model.classifier), ("p2w", model.sr.P2W))}
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            assert all(norms[name] > 0 for name in ("rdn", "abmil", "classifier"))
            assert (norms["p2w"] > 0) if args.lambda_sr else (norms["p2w"] == 0)
            assert len(bags) == 1 and bags[0][0] == args.regions
            phase = "optimizer_step"
            optimizer.step()
            report.update(status="passed", optimizer_steps=1, loss_cls=cls.item(), loss_sr=sr.item(),
                          loss_total=total_loss.item(), logits=logits.detach().float().cpu().tolist(),
                          gradients=norms)
    except torch.OutOfMemoryError as error:
        report.update(status="oom", failed_phase=phase, error=str(error))
    report.update(elapsed_seconds=time.perf_counter() - started, **memory_summary(device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
