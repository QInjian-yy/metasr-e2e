"""One-step synthetic or real-image CUDA measurement through the WSI loss path."""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

import engine
from models.baseline import MetaSRABMIL
from train_e2e import load_config


def load_real_sample(root, slide_id, n):
    with (root / "manifests/patch_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["slide_id"] == slide_id]
    with (root / "downstream_train/camelyon16_labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        labels = [row for row in csv.DictReader(handle) if row["slide_id"] == slide_id]
    if len(labels) != 1 or labels[0]["label"] not in ("0", "1"):
        raise ValueError(f"Expected one binary label for {slide_id}")
    if not 1 <= n <= len(rows):
        raise ValueError(f"{slide_id} has {len(rows)} manifest regions; requested {n}")
    sample = {"slide_id": slide_id, "n_regions": n, "label": int(labels[0]["label"]),
              "lr_paths": [], "hr_paths": [], "manifest_n_regions": len(rows)}
    for row in rows[:n]:
        for key, folder, size in (("lr_paths", "images_256", 256),
                                  ("hr_paths", "images_8192", 8192)):
            path = root / folder / row["filename"]
            with Image.open(path) as image:
                if image.size != (size, size):
                    raise ValueError(f"{path}: expected {(size, size)}, got {image.size}")
            sample[key].append(str(path))
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Training YAML; explicit probe options override its step settings")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--use-region-microbatch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--region-microbatch-size", "--micro-batch", dest="micro_batch_size",
                        type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--lambda-sr", type=float, default=0.1)
    parser.add_argument("--sr-crop", type=int, default=256)
    parser.add_argument("--mode", choices=("joint", "full-sr"), default="joint")
    parser.add_argument("--data-root", type=Path, help="Use real paired images and labels below this root")
    parser.add_argument("--slide-id", help="WSI to select from the real-data manifest")
    parser.add_argument("--gradient-output", type=Path, help="Save CPU gradients after measurement for comparison")
    parser.add_argument("--training-state-output", type=Path,
                        help="Save post-step parameters and Adam state on CPU for equivalence checks")
    parser.add_argument("--deterministic", action="store_true",
                        help="Require deterministic CUDA algorithms for a separate controlled comparison")
    parser.add_argument("--output", type=Path, required=True)
    requested, _ = parser.parse_known_args()
    file_config = load_config(requested.config) if requested.config else {}
    if file_config:
        parser.set_defaults(precision=file_config["precision"], lambda_sr=file_config["lambda_sr"],
                            sr_crop=file_config["sr_train_crop"],
                            use_region_microbatch=file_config["training"]["use_region_microbatch"],
                            micro_batch_size=file_config["training"]["region_microbatch_size"])
    args = parser.parse_args()
    model_config = file_config.get("metasr", {"scale": 32, "lr_chunk_size": 256,
                                              "checkpoint_rdb": True, "rgb_range": 1, "rdn_blocks": 16})
    if args.n < 1 or args.size < 1 or args.micro_batch_size < 1 or args.sr_crop < 1:
        parser.error("n, size, micro-batch-size and sr-crop must be positive")
    if (args.data_root is None) != (args.slide_id is None):
        parser.error("--data-root and --slide-id must be supplied together")
    if args.data_root is not None and (args.mode != "joint" or args.size != 256):
        parser.error("Real-data measurement requires --mode joint --size 256")
    real_sample = load_real_sample(args.data_root, args.slide_id, args.n) if args.data_root else None
    torch.set_num_threads(4)
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
    seed = file_config.get("seed", 7)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    if args.precision == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    crop = min(args.sr_crop, args.size * 32)
    report = {**vars(args), "output": str(args.output),
              "config": str(args.config) if args.config else None, "model_config": model_config, "seed": seed,
              "gradient_output": str(args.gradient_output) if args.gradient_output else None,
              "training_state_output": str(args.training_state_output) if args.training_state_output else None,
              "data_root": str(args.data_root) if args.data_root else None, "gpu": props.name,
              "gpu_total_bytes": props.total_memory, "torch": torch.__version__,
              "cuda": torch.version.cuda, "checkpoint_rdb": model_config["checkpoint_rdb"],
              "normalization": "GroupNorm", "input_shape": [args.n, 3, args.size, args.size],
              "synthetic": real_sample is None, "includes_adam_step": args.mode == "joint",
              "sr_crop": crop, "lr_chunk_size": model_config["lr_chunk_size"],
              "effective_region_batch_size": min(args.micro_batch_size, args.n)
                  if args.use_region_microbatch else args.n,
              "training_stages": {"forward": False, "backward": False, "optimizer_step": False}}
    report["tf32"] = {"matmul": torch.backends.cuda.matmul.allow_tf32,
                      "cudnn": torch.backends.cudnn.allow_tf32}
    report["determinism"] = {"algorithms": torch.are_deterministic_algorithms_enabled(),
                             "cudnn": torch.backends.cudnn.deterministic,
                             "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    if real_sample is not None:
        report["sample"] = real_sample
    model = MetaSRABMIL(**model_config).to(device)
    report["parameter_counts"] = {
        "rdn_encoder": sum(p.numel() for name, p in model.sr.named_parameters()
                           if name.startswith(("SFENet", "RDBs.", "GFF."))),
        "metasr_total": sum(p.numel() for p in model.sr.parameters()),
        "e2e_total": sum(p.numel() for p in model.parameters()),
    }
    if args.gradient_output or args.training_state_output:
        digest = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
        report["initial_state_sha256"] = digest.hexdigest()
        report["encoder_inputs"], report["mil_inputs"] = [], []
        model.region_encoder.register_forward_pre_hook(lambda module, inputs: report["encoder_inputs"].append(
            {"shape": list(inputs[0].shape), "requires_grad": inputs[0].requires_grad,
             "has_grad_fn": inputs[0].grad_fn is not None}))
        model.mil_head.register_forward_pre_hook(lambda module, inputs: report["mil_inputs"].append(
            {"shape": list(inputs[0].shape), "requires_grad": inputs[0].requires_grad,
             "has_grad_fn": inputs[0].grad_fn is not None}))
    optimizer = torch.optim.Adam(model.parameters(), lr=file_config.get("lr", 1e-5),
                                 weight_decay=file_config.get("weight_decay", 1e-4))
    report["optimizer"] = {key: optimizer.param_groups[0][key] for key in ("lr", "weight_decay", "betas", "eps")}
    torch.cuda.empty_cache()
    report["allocated_before_step"] = torch.cuda.memory_allocated(device)
    report["reserved_before_step"] = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    phase = "setup"
    try:
        if args.mode == "joint":
            if real_sample is None:
                lr_cpu = torch.rand(args.n, 3, args.size, args.size)
                hr_cpu = torch.rand(args.n, 3, crop, crop)
                sample = {"slide_id": "synthetic", "n_regions": args.n, "label": 1,
                          "lr_paths": list(range(args.n)), "hr_paths": list(range(args.n))}
                loader_scope = (
                    patch("engine.load_images", side_effect=lambda paths, size: lr_cpu[paths]),
                    patch("engine.load_hr_crops", side_effect=lambda paths, box: hr_cpu[paths]),
                )
            else:
                sample = real_sample
                loader_scope = (nullcontext(), nullcontext())
            config = {"training": {"use_region_microbatch": args.use_region_microbatch,
                                   "region_microbatch_size": args.micro_batch_size},
                      "lambda_sr": args.lambda_sr, "sr_train_crop": crop,
                      "precision": args.precision}
            model.train()
            optimizer.zero_grad(set_to_none=True)
            phase = "forward"
            print(f"[probe] N={args.n} synthetic={report['synthetic']} phase={phase}", flush=True)
            with loader_scope[0], loader_scope[1], \
                    patch("engine.load_hr_crops", wraps=engine.load_hr_crops) as hr_loader:
                total, cls, sr, logits = engine.wsi_loss(model, sample, device, config)
                report["hr_crops"] = [{"paths": [str(path) for path in call.args[0]],
                                       "box_y_x_h_w": list(call.args[1])}
                                      for call in hr_loader.call_args_list]
                if not torch.isfinite(total):
                    raise FloatingPointError("Non-finite loss")
                report["training_stages"]["forward"] = True
                phase = "backward"
                print(f"[probe] phase={phase}", flush=True)
                total.backward()
                report["training_stages"]["backward"] = True
                finite_gradients = [torch.isfinite(p.grad).all() for p in model.parameters()
                                    if p.grad is not None]
                gradient_finite = bool(torch.stack(finite_gradients).all().item())
                if not gradient_finite:
                    raise FloatingPointError("Non-finite gradient")
                norms = {"rdn": engine.gradient_norm(model.sr),
                         "resnet18_gn": engine.gradient_norm(model.region_encoder),
                         "abmil": engine.gradient_norm(model.mil_head),
                         "classifier": engine.gradient_norm(model.classifier),
                         "pos2weight": engine.gradient_norm(model.sr.P2W)}
                if norms["rdn"] <= 0 or norms["resnet18_gn"] <= 0:
                    raise AssertionError(f"Expected nonzero classification gradients: {norms}")
                if args.lambda_sr == 0 and any(p.grad is not None for p in model.sr.P2W.parameters()):
                    raise AssertionError("SR OFF unexpectedly constructed Pos2Weight gradients")
                report.update(loss_total=total.item(), loss_cls=cls.item(), loss_sr=sr.item(),
                              logits=logits.detach().float().cpu().tolist(),
                              gradient_l2=norms, gradients_finite=gradient_finite)
                phase = "optimizer_step"
                print(f"[probe] phase={phase}", flush=True)
                optimizer.step()
                report["training_stages"]["optimizer_step"] = True
        else:
            model.eval()
            total_pixels, tiles = 0, 0
            feature_batch = report["effective_region_batch_size"]
            with torch.no_grad(), engine.autocast(device, args.precision):
                for start in range(0, args.n, feature_batch):
                    b = min(feature_batch, args.n-start)
                    feature = model.sr.extract_features(torch.rand(b, 3, args.size, args.size, device=device))
                    for y, x, tile in model.sr.iter_full_sr(feature):
                        assert not tile.requires_grad and torch.isfinite(tile).all()
                        total_pixels += tile.numel() // 3
                        tiles += 1
            assert total_pixels == args.n * (args.size * 32) ** 2
            report.update(pixels=total_pixels, tiles=tiles,
                          output_shape=[args.n, 3, args.size*32, args.size*32])
        report["status"] = "passed"
    except torch.cuda.OutOfMemoryError as exc:
        report.update(status="oom", phase=phase, error=str(exc))
    except Exception as exc:
        report.update(status="failed", phase=phase, error=f"{type(exc).__name__}: {exc}")
    if report.get("status") == "passed":
        torch.cuda.synchronize(device)
    report["max_memory_allocated"] = torch.cuda.max_memory_allocated(device)
    report["max_memory_reserved"] = torch.cuda.max_memory_reserved(device)
    report["elapsed_seconds"] = time.perf_counter() - started
    if args.gradient_output and report["status"] == "passed" and args.mode == "joint":
        # CPU copies happen after peak/timing capture and do not change the training graph.
        args.gradient_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({name: p.grad.detach().cpu() if p.grad is not None else None
                    for name, p in model.named_parameters()}, args.gradient_output)
    if args.training_state_output and report["status"] == "passed" and args.mode == "joint":
        args.training_state_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "parameters": {name: p.detach().cpu() for name, p in model.named_parameters()},
            "adam": {name: {key: value.detach().cpu() if torch.is_tensor(value) else value
                            for key, value in optimizer.state[p].items()}
                     for name, p in model.named_parameters()},
        }, args.training_state_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
