"""Small FP32 equivalence gate, using only the locally vendored official source."""

import ast
import json
import math
from pathlib import Path

import torch
from torch import nn

from models.metasr import MemoryEfficientMetaRDN, official_args
from vendor.official.metardn import MetaRDN


def input_matrix_wpn(h, w, scale):
    # Execute the verbatim official method without importing its legacy trainer
    # dependencies or option.py's argument-parser side effects.
    source = Path(__file__).parent / "reference" / "trainer.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Trainer")
    method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef)
                  and node.name == "input_matrix_wpn")
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["input_matrix_wpn"](None, h, w, scale)


def official_decode(model, feature, positions):
    # Exact decoder statements from official MetaRDN.forward, starting at P2W.
    source = Path(__file__).parent / "vendor" / "official" / "metardn.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MetaRDN")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "local_weight" for t in n.targets))
    method.body = method.body[start:]
    namespace = {"torch": torch, "nn": nn, "math": math}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["forward"](model, feature, positions)


def max_error(a, b):
    return float((a.detach() - b.detach()).abs().max())


def gradients(module):
    return {name: None if p.grad is None else p.grad.detach().clone()
            for name, p in module.named_parameters()}


def compare_gradients(first, second):
    if first.keys() != second.keys():
        raise AssertionError("Different parameter names")
    error = 0.0
    for name, a in first.items():
        b = second[name]
        if a is None or b is None:
            if a is not None or b is not None:
                raise AssertionError("Different gradient presence: " + name)
        else:
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4, msg=name)
            error = max(error, max_error(a, b))
    return error


def run_equivalence(device="cpu", rdn_blocks=16):
    torch.manual_seed(20260927)
    device = torch.device(device)
    reference = MetaRDN(official_args()).to(device).eval()
    optimized = MemoryEfficientMetaRDN(checkpoint_rdb=False, lr_chunk_size=2,
                                       rdn_blocks=rdn_blocks).to(device).eval()
    if rdn_blocks != 16:
        # Keep the vendored reference forward; match the selected variant's depth.
        reference.D = rdn_blocks
        reference.RDBs = reference.RDBs[:rdn_blocks]
        reference.GFF[0] = nn.Conv2d(rdn_blocks * 64, 64, 1).to(device)
    optimized.load_state_dict(reference.state_dict(), strict=True)
    results = {"device": str(device), "dtype": "float32",
               "official_config": "RDN-B/G0=64/D=16/C=8/G=64" if rdn_blocks == 16 else None,
               "selected_config": f"RDN-B/G0=64/D={rdn_blocks}/C=8/G=64",
               "is_official_rdn_b": rdn_blocks == 16,
               "upsampling": []}
    for scale in (2, 3, 4):
        reference.scale = optimized.scale = scale
        lr = torch.randn(1, 3, 4, 5, device=device)
        positions, mask = input_matrix_wpn(4, 5, scale)
        positions = positions.to(device)
        assert mask.all()
        a = reference(lr, positions)
        b = optimized(lr, (0, 0, 4 * scale, 5 * scale))
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
        a.square().mean().backward()
        b.square().mean().backward()
        results["upsampling"].append({"scale": scale, "shape": list(a.shape),
            "forward_max_abs_error": max_error(a, b),
            "parameter_gradient_max_abs_error": compare_gradients(gradients(reference), gradients(optimized))})
        reference.zero_grad(set_to_none=True)
        optimized.zero_grad(set_to_none=True)
    del reference, a, b

    lr = torch.randn(1, 3, 4, 5, device=device, requires_grad=True)
    a = optimized.extract_features(lr)
    a.square().mean().backward()
    expected, expected_input = gradients(optimized), lr.grad.clone()
    optimized.zero_grad(set_to_none=True)
    lr.grad = None
    optimized.checkpoint_rdb = True
    b = optimized.extract_features(lr)
    b.square().mean().backward()
    torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(expected_input, lr.grad, atol=1e-7, rtol=1e-6)
    results["checkpoint"] = {"feature_shape": list(a.shape), "eval_with_grad_enabled": True,
        "forward_max_abs_error": max_error(a, b),
        "input_gradient_max_abs_error": max_error(expected_input, lr.grad),
        "parameter_gradient_max_abs_error": compare_gradients(expected, gradients(optimized))}
    results["passed"] = True
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/equivalence.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rdn-blocks", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = run_equivalence(args.device, rdn_blocks=args.rdn_blocks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
