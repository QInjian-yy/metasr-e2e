"""Trace one production joint-loss step using synthetic LR and full CPU HR tensors."""

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import engine
from models.baseline import MetaSRABMIL
from train_e2e import ROOT, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(71)
    torch.backends.cudnn.benchmark = False
    config = load_config(ROOT / "configs/baseline.yaml")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    lr_cpu = torch.rand(args.n, 3, 256, 256)
    hr_cpu = torch.randint(256, (args.n, 3, 8192, 8192), dtype=torch.uint8)
    model = MetaSRABMIL(**config["metasr"]).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    report = {"gpu": props.name, "gpu_total_bytes": props.total_memory,
              "config": config, "synthetic": True,
              "input_note": "Independent random LR and HR; tests computation, not reconstruction quality.",
              "shapes": {}, "stages": {"forward": False, "backward": False, "optimizer_step": False}}

    def record(name, tensor):
        report["shapes"].setdefault(name, {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                                          "device": str(tensor.device), "requires_grad": tensor.requires_grad})

    def hook(name):
        def observe(module, inputs, output):
            record(name + ".input", inputs[0])
            for i, value in enumerate(output if isinstance(output, tuple) else (output,)):
                record(name + f".output{i}", value)
        return observe

    modules = {"sub_mean": model.sr.sub_mean, "SFENet1": model.sr.SFENet1,
               "SFENet2": model.sr.SFENet2, "RDB1": model.sr.RDBs[0],
               "RDB16": model.sr.RDBs[-1], "GFF": model.sr.GFF,
               "Pos2Weight": model.sr.P2W, "ResNet18_GN": model.region_encoder,
               "ABMIL_V": model.mil_head.attention_V, "ABMIL_U": model.mil_head.attention_U,
               "ABMIL_w": model.mil_head.attention_w,
               "ABMIL": model.mil_head, "classifier": model.classifier}
    modules.update({"ResNet." + name: getattr(model.region_encoder, name)
                    for name in ("conv1", "bn1", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc")})
    handles = [module.register_forward_hook(hook(name)) for name, module in modules.items()]
    record("LR_CPU", lr_cpu)
    record("HR_CPU", hr_cpu)
    report["normalization"] = {"GroupNorm": sum(isinstance(m, torch.nn.GroupNorm) for m in model.modules()),
                               "BatchNorm2d": sum(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())}
    original_decode = model.sr.decode_crop
    original_unfold = torch.nn.functional.unfold
    original_matmul = torch.matmul

    def load_crop(paths, box):
        y, x, h, w = box
        crop = torch.stack([hr_cpu[i, :, y:y+h, x:x+w] for i in paths]).float().div_(255)
        report.setdefault("hr_crops", []).extend(
            {"region_index": i, "box_y_x_h_w": list(box)} for i in paths)
        record("GT_crop_CPU", crop)
        return crop

    def unfold(*values, **kwargs):
        result = original_unfold(*values, **kwargs)
        record("unfold", result)
        return result

    def matmul(a, b):
        result = original_matmul(a, b)
        record("first_chunk.patches", a)
        record("first_chunk.kernel", b)
        record("first_chunk.matmul", result)
        report["decoder_matmul_calls"] = report.get("decoder_matmul_calls", 0) + 1
        return result

    def decode(feature, box):
        record("shared_feature", feature)
        with patch("models.metasr.F.unfold", side_effect=unfold), patch("torch.matmul", side_effect=matmul):
            result = original_decode(feature, box)
        record("SR_prediction", result)
        return result

    sample = {"n_regions": args.n, "label": 1,
              "lr_paths": list(range(args.n)), "hr_paths": list(range(args.n))}
    tracked = {"MetaRDN.SFENet1": model.sr.SFENet1.weight,
               "ResNet.conv1": model.region_encoder.conv1.weight,
               "ABMIL.attention_V": model.mil_head.attention_V.weight,
               "classifier": model.classifier.weight,
               "Pos2Weight.output": model.sr.P2W.meta_block[2].weight}
    before = {name: parameter.detach().cpu().clone() for name, parameter in tracked.items()}
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    print(f"Starting full-batch N={args.n}, crop={config['sr_train_crop']}", flush=True)
    try:
        with patch("engine.load_images", side_effect=lambda paths, size: lr_cpu[paths]), \
             patch("engine.load_hr_crops", side_effect=load_crop), \
             patch.object(model.sr, "decode_crop", side_effect=decode):
            total, cls, sr, logits = engine.wsi_loss(model, sample, device, config)
        record("logits", logits)
        for name, loss in (("loss_total", total), ("loss_cls", cls), ("loss_sr", sr)):
            assert torch.isfinite(loss)
            record(name, loss)
            report[name] = loss.item()
        report["stages"]["forward"] = True
        print("Forward complete; starting backward", flush=True)
        total.backward()
        report["stages"]["backward"] = True
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        report["gradient_l2"] = {name: parameter.grad.float().norm().item() for name, parameter in tracked.items()}
        assert all(value > 0 for name, value in report["gradient_l2"].items()
                   if args.n > 1 or name != "ABMIL.attention_V")
        report["attention_note"] = "N=1 softmax is identically 1; zero attention gradient is expected." if args.n == 1 else "Multiple regions participate in attention."
        print("Backward complete; starting Adam step", flush=True)
        optimizer.step()
        report["stages"]["optimizer_step"] = True
        report["parameter_update_max_abs"] = {name: (p.detach().cpu() - before[name]).abs().max().item()
                                               for name, p in tracked.items()}
        assert all(torch.isfinite(p).all() for p in model.parameters())
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        torch.cuda.synchronize()
        report["elapsed_seconds"] = time.perf_counter() - started
        report.update(engine.memory_summary(device))
        for handle in handles:
            handle.remove()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "shapes"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
