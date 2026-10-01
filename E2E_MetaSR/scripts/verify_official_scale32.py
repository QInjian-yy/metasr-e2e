"""Audit x32 against verified official source; preserve strict numerical failures.

No training/model/config files are modified. Exit 1 if the CPU FP32/FP64
checks fail; CUDA BF16 is reported separately against the FP32 tolerance.
"""

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from models.metasr import MemoryEfficientMetaRDN, official_args
from validation import gradients, input_matrix_wpn, official_decode, run_equivalence
from vendor.official.metardn import MetaRDN

ROOT = Path(__file__).resolve().parents[1]
FORWARD_TOL = (3e-6, 3e-5)  # Existing FP32 gate, unchanged.
GRAD_TOL = (2e-5, 2e-4)     # Existing FP32 gradient gate, unchanged.
DOUBLE_TOL = (1e-10, 1e-10)
BOXES = ((0, 0, 64, 96), (0, 32, 64, 64), (7, 11, 37, 52),
         (63, 95, 1, 1), (31, 31, 2, 2))


def compare(actual, expected, tolerance):
    a, b = actual.detach().double(), expected.detach().double()
    difference = (a - b).abs()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    failures = int((difference > tolerance[0] + tolerance[1] * b.abs()).sum())
    return {"max_abs_error": difference.max().item(), "failed_elements": failures,
            "relative_l2_error": (difference.norm() / b.norm().clamp_min(1e-30)).item(),
            "finite": finite, "passed": finite and failures == 0}


def compare_grads(actual, expected, tolerance):
    assert actual.keys() == expected.keys()
    present = all((actual[k] is None) == (expected[k] is None) for k in actual)
    values = {k: compare(actual[k], expected[k], tolerance) for k in actual
              if actual[k] is not None and expected[k] is not None}
    return {"presence_equal": present, "parameters": values,
            "max_abs_error": max(v["max_abs_error"] for v in values.values()),
            "passed": present and all(v["passed"] for v in values.values())}


def verify_source(official):
    result = {}
    for original, local in (("model/metardn.py", "vendor/official/metardn.py"),
                            ("model/common.py", "vendor/official/common.py"),
                            ("trainer.py", "reference/trainer.py"),
                            ("option.py", "reference/option.py")):
        a, b = official / original, ROOT / local
        source = a.read_text(encoding="utf-8")
        if original == "model/metardn.py":
            source = source.replace("from model import common", "from . import common")
        assert source == b.read_text(encoding="utf-8"), original
        result[original] = {"verified": True,
                            "official_sha256": hashlib.sha256(a.read_bytes()).hexdigest()}
    return result


def decoder_cases(dtype):
    results = []
    tol = DOUBLE_TOL if dtype == torch.float64 else FORWARD_TOL
    grad_tol = DOUBLE_TOL if dtype == torch.float64 else GRAD_TOL
    for seed in (0, 71, 20260928):
        torch.manual_seed(seed)
        model = MemoryEfficientMetaRDN().to(dtype).eval()
        pos, mask = input_matrix_wpn(2, 3, 32)
        table = model.position_table("cpu").reshape(32, 32, 3)
        expanded = table[None, :, None].expand(2, 32, 3, 32, 3).reshape(1, -1, 3)
        assert torch.equal(pos, expanded) and mask.all()
        for n in (1, 6):
            feature = torch.randn(n, 64, 2, 3).to(dtype)
            with torch.no_grad():
                reference = official_decode(model, feature, pos.to(dtype))
                for chunk in (1, 2, 256):
                    model.lr_chunk_size = chunk
                    for box in BOXES:
                        y, x, h, w = box
                        with patch.object(model, "repeat_x", side_effect=AssertionError("repeat_x forbidden")):
                            with patch("models.metasr.F.unfold", wraps=F.unfold) as unfold:
                                actual = model.decode_crop(feature, box)
                                assert unfold.call_count == 1
                        results.append({"kind": "forward", "seed": seed, "n": n,
                                        "chunk": chunk, "box": box,
                                        **compare(actual, reference[:, :, y:y+h, x:x+w], tol)})
            # Random cotangent checks pixel-specific derivatives, not only mean loss.
            fa = feature.clone().requires_grad_()
            fb = feature.clone().requires_grad_()
            y, x, h, w = BOXES[2]
            cotangent = torch.randn(n, 3, h, w).to(dtype)
            model.zero_grad(set_to_none=True)
            expected = official_decode(model, fa, pos.to(dtype))[:, :, y:y+h, x:x+w]
            (expected * cotangent).mean().backward()
            expected_grads = gradients(model)
            model.zero_grad(set_to_none=True)
            actual = model.decode_crop(fb, BOXES[2])
            (actual * cotangent).mean().backward()
            results.append({"kind": "random_cotangent_backward", "seed": seed, "n": n,
                            "feature": compare(fb.grad, fa.grad, grad_tol),
                            "parameters": compare_grads(gradients(model), expected_grads, grad_tol)})
        del model, actual, expected, expected_grads, fa, fb
        gc.collect()
    return results


def negative_controls():
    torch.manual_seed(32)
    model = MemoryEfficientMetaRDN().double().eval()
    feature = torch.randn(1, 64, 2, 3, dtype=torch.float64, requires_grad=True)
    pos, _ = input_matrix_wpn(2, 3, 32)
    with torch.no_grad():
        expected = official_decode(model, feature, pos.double())
        table = model.position_table("cpu")
        with patch.object(model, "position_table", return_value=table[:, [0, 2, 1]]):
            wrong = model.decode_crop(feature, BOXES[0])
        swapped = compare(wrong, expected, DOUBLE_TOL)
    original_table = model.weight_table
    with patch.object(model, "weight_table", side_effect=lambda f: original_table(f).detach()):
        model.decode_crop(feature, BOXES[2]).square().mean().backward()
    detached = all(p.grad is None for p in model.P2W.parameters())
    assert not swapped["passed"] and detached
    return {"swapped_offsets_rejected": not swapped["passed"],
            "swapped_offsets_error": swapped["max_abs_error"],
            "detached_weight_gradient_rejected": detached}


def whole_model(device, precision, adam_step=False):
    torch.manual_seed(813)
    dtype = torch.float64 if precision == "float64" else torch.float32
    reference = MetaRDN(official_args()).to(device=device, dtype=dtype).train()
    reference.set_scale(0)
    optimized = MemoryEfficientMetaRDN(checkpoint_rdb=True).to(device=device, dtype=dtype).train()
    optimized.load_state_dict(reference.state_dict(), strict=True)
    xa = torch.rand(1, 3, 2, 3, device=device, dtype=dtype).requires_grad_()
    xb = xa.detach().clone().requires_grad_()
    pos, _ = input_matrix_wpn(2, 3, 32)
    pos = pos.to(device=device, dtype=dtype)
    truth = torch.rand(1, 3, 64, 96, device=device, dtype=dtype)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=precision == "bf16"):
        a, b = reference(xa, pos), optimized(xb, BOXES[0])
        la, lb = F.l1_loss(a.to(dtype), truth), F.l1_loss(b.to(dtype), truth)
    la.backward()
    lb.backward()
    tol = DOUBLE_TOL if dtype == torch.float64 else FORWARD_TOL
    grad_tol = DOUBLE_TOL if dtype == torch.float64 else GRAD_TOL
    result = {"device": device, "precision": precision, "optimized_checkpoint": True,
              "output": compare(b, a, tol), "loss_abs_error": abs(lb.item() - la.item()),
              "input_gradient": compare(xb.grad, xa.grad, grad_tol),
              "parameter_gradients": compare_grads(gradients(optimized), gradients(reference), grad_tol)}
    if adam_step:
        before = optimized.P2W.meta_block[2].weight.detach().clone()
        for model in (reference, optimized):
            torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4).step()
        result["adam_parameter_comparison"] = compare_grads(
            dict(optimized.named_parameters()), dict(reference.named_parameters()), tol)
        assert not torch.equal(before, optimized.P2W.meta_block[2].weight)
        with torch.no_grad():
            result["after_update_output"] = compare(optimized(xb, BOXES[0]), reference(xa, pos), tol)
    return result


def all_passed(value):
    if isinstance(value, dict):
        return value.get("passed", True) and all(all_passed(v) for v in value.values())
    return all(all_passed(v) for v in value) if isinstance(value, list) else True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, default=ROOT.parent / "Meta-SR-Pytorch-0.4.0")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/official_scale32_strict_audit.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    watched = list((ROOT / "models").glob("*.py")) + [ROOT / "engine.py", ROOT / "validation.py", ROOT / "configs/baseline.yaml"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    report = {"torch": torch.__version__, "source": verify_source(args.official_root),
              "tolerances": {"fp32_forward": FORWARD_TOL, "fp32_gradients": GRAD_TOL, "fp64": DOUBLE_TOL},
              "scope": "Same parameters/rgb_range=1; crop compared to the same official output crop. No full-8K official reconstruction or real-data training."}
    projected = torch.arange(8192).float().mul(1 / 32)
    assert torch.equal(projected.floor().long(), torch.arange(8192) // 32)
    assert torch.equal(projected - projected.floor(), (torch.arange(8192) % 32).float() / 32)
    report["axis_8192_exact"] = True
    report["existing_gate"] = run_equivalence()
    for dtype in (torch.float32, torch.float64):
        name = str(dtype).split(".")[-1]
        report[name] = decoder_cases(dtype)
        print(name, "forward cases=90, backward cases=6, passed=", all_passed(report[name]), flush=True)
    report["negative_controls"] = negative_controls()
    report["float64_whole_model_adam"] = whole_model("cpu", "float64", adam_step=True)
    report["cuda"] = []
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        report["gpu"] = torch.cuda.get_device_name()
        for precision in ("fp32", "bf16"):
            if precision == "bf16" and not torch.cuda.is_bf16_supported():
                continue
            report["cuda"].append(whole_model("cuda", precision))
            gc.collect()
            torch.cuda.empty_cache()
    report["runtime_files_unchanged"] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    assert report["runtime_files_unchanged"]
    report["fp64_passed"] = all_passed(report["float64"]) and all_passed(report["float64_whole_model_adam"])
    report["fp32_passed"] = all_passed(report["float32"]) and all(all_passed(r) for r in report["cuda"] if r["precision"] == "fp32")
    report["bf16_meets_fp32_tolerance"] = next((all_passed(r) for r in report["cuda"] if r["precision"] == "bf16"), None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("fp64_passed", "fp32_passed", "bf16_meets_fp32_tolerance", "runtime_files_unchanged")}), flush=True)
    print("Report:", args.output, flush=True)
    raise SystemExit(0 if report["fp64_passed"] and report["fp32_passed"] else 1)


if __name__ == "__main__":
    main()
