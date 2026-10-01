"""Read-only audit of production code; all generated evidence stays in audit/."""
import argparse
import ast
import csv
import hashlib
import json
import math
import sys
import types
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from engine import wsi_loss
from models.baseline import MetaSRABMIL
from models.metasr import MemoryEfficientMetaRDN, official_args

SOURCE = ROOT.parent / "Meta-SR-Pytorch-0.4.0"


def load_official():
    common = types.ModuleType("audit_official_common")
    p = SOURCE / "model/common.py"
    exec(compile(p.read_text(encoding="utf-8"), str(p), "exec"), common.__dict__)
    p = SOURCE / "model/metardn.py"
    tree = ast.parse(p.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == "model")]
    namespace = {"common": common}
    exec(compile(tree, str(p), "exec"), namespace)
    return namespace["MetaRDN"]


OfficialMetaRDN = load_official()


def official_positions(h, w, scale):
    p = SOURCE / "trainer.py"
    tree = ast.parse(p.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "input_matrix_wpn")
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(p), "exec"), namespace)
    return namespace["input_matrix_wpn"](None, h, w, scale)


def official_decoder(model, feature, positions):
    p = SOURCE / "model/metardn.py"
    tree = ast.parse(p.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MetaRDN")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "local_weight")
    method.body = method.body[start:]
    namespace = {"torch": torch, "nn": nn, "math": math}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(p), "exec"), namespace)
    return namespace["forward"](model, feature, positions)


def error(a, b, atol=1e-5, rtol=5e-4, enforce=True):
    within = bool(torch.isclose(a, b, atol=atol, rtol=rtol).all())
    if enforce:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    delta = (a.detach().double() - b.detach().double()).abs()
    return {"max_abs_error": delta.max().item(), "mean_abs_error": delta.mean().item(),
            "elements": delta.numel(), "within_atol_rtol": within}


def grads(model):
    return {n: None if p.grad is None else p.grad.detach().clone() for n, p in model.named_parameters()}


def compare_dicts(a, b, prefix="", atol=1e-5, enforce=True):
    peak, total, count, tensors, absent, worst = 0., 0., 0, 0, 0, None
    failed, delta2, reference2 = [], 0., 0.
    for name in a:
        if not name.startswith(prefix):
            continue
        x, y = a[name], b[name]
        if x is None or y is None:
            assert x is None and y is None, name
            absent += 1
            continue
        result = error(x, y, atol=atol, enforce=enforce)
        if not result["within_atol_rtol"]:
            failed.append(name)
        delta2 += (x.detach().double()-y.detach().double()).square().sum().item()
        reference2 += x.detach().double().square().sum().item()
        if result["max_abs_error"] > peak:
            peak, worst = result["max_abs_error"], name
        total += result["mean_abs_error"] * result["elements"]
        count += result["elements"]
        tensors += 1
    return {"max_abs_error": peak, "mean_abs_error": total / count if count else 0.,
            "elements": count, "tensors": tensors, "none_tensors": absent, "worst_parameter": worst,
            "outside_original_tolerance": failed, "relative_l2_error": math.sqrt(delta2/max(reference2, 1e-300))}


def norm(module):
    parameters = list(module.named_parameters())
    present = [(n, p.grad) for n, p in parameters if p.grad is not None]
    return {"l2": math.sqrt(sum(g.detach().double().square().sum().item() for _, g in present)),
            "tensors": len(parameters), "with_grad": len(present),
            "all_finite": all(bool(torch.isfinite(g).all()) for _, g in present),
            "zero_tensor_names": [n for n, g in present if not torch.count_nonzero(g)],
            "missing_names": [n for n, p in parameters if p.grad is None]}


def chain(model):
    modules = {"ABMIL": model.mil_head, "classifier": model.classifier,
               "ResNet_first_conv": model.region_encoder.conv1,
               "ResNet_last_conv": model.region_encoder.layer4[-1].conv2,
               "ResNet_layer4": model.region_encoder.layer4,
               "SFENet1": model.sr.SFENet1, "SFENet2": model.sr.SFENet2,
               "RDB1": model.sr.RDBs[0], "RDB8": model.sr.RDBs[7],
               "RDB16": model.sr.RDBs[15], "GFF": model.sr.GFF, "P2W": model.sr.P2W}
    return {n: norm(m) for n, m in modules.items()}


def architecture():
    official = OfficialMetaRDN(official_args())
    model = MetaSRABMIL()
    assert list(official.state_dict()) == list(model.sr.state_dict())
    assert all(official.state_dict()[n].shape == v.shape for n, v in model.sr.state_dict().items())
    a = (SOURCE / "model/metardn.py").read_bytes().replace(b"from model import common", b"from . import common")
    assert a == (ROOT / "vendor/official/metardn.py").read_bytes()
    assert (SOURCE / "model/common.py").read_bytes() == (ROOT / "vendor/official/common.py").read_bytes()
    assert (SOURCE / "trainer.py").read_bytes() == (ROOT / "reference/trainer.py").read_bytes()
    assert (SOURCE / "option.py").read_bytes() == (ROOT / "reference/option.py").read_bytes()
    assert model.sr.D == len(model.sr.RDBs) == 16
    for block in model.sr.RDBs:
        assert len(block.convs) == 8
        assert block.LFF.weight.shape == (64, 576, 1, 1)
        for index, conv in enumerate(block.convs):
            assert conv.conv[0].weight.shape == (64, 64*(index+1), 3, 3)
    def node(path, name):
        return next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    assert ast.dump(node(ROOT / "models/abmil.py", "GatedAttentionMIL")) == ast.dump(
        node(ROOT.parent / "E2E/downstream/attention_resnet/model.py", "GatedAttentionMIL"))
    assert ast.dump(node(ROOT / "models/baseline.py", "_replace_bn_with_gn")) == ast.dump(
        node(ROOT.parent / "E2E/downstream/shared_model.py", "_replace_bn_with_gn"))
    return {"official_state_entries": len(official.state_dict()), "state_names_and_shapes_equal": True,
            "D": 16, "C": 8, "growth": 64, "G0": 64, "kernel": 3,
            "P2W": [3, 256, 1728], "GroupNorm": sum(isinstance(m, nn.GroupNorm) for m in model.modules()),
            "BatchNorm": sum(isinstance(m, nn.BatchNorm2d) for m in model.modules()),
            "MeanShift_parameters": {n: p.requires_grad for n, p in model.sr.named_parameters()
                                     if n.startswith(("sub_mean", "add_mean"))}}


def scale32_equivalence(device):
    output = []
    for size in (2, 4):
        torch.manual_seed(113 + size)
        reference = OfficialMetaRDN(official_args(32)).to(device).eval()
        reference.set_scale(0)
        model = MemoryEfficientMetaRDN(scale=32, checkpoint_rdb=False, lr_chunk_size=3).to(device).eval()
        model.load_state_dict(reference.state_dict())
        x = torch.rand(1, 3, size, size, device=device, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        positions, mask = official_positions(size, size, 32)
        assert mask.all()
        a = reference(x, positions.to(device))
        b = model(y, (0, 0, size*32, size*32))
        weights = torch.linspace(0.3, 1.3, a.numel(), device=device).reshape_as(a)
        (a.square() * weights).mean().backward()
        (b.square() * weights).mean().backward()
        ga, gb = grads(reference), grads(model)
        rdn_a = {n: g for n, g in ga.items() if not n.startswith(("P2W", "add_mean"))}
        rdn_b = {n: g for n, g in gb.items() if n in rdn_a}
        output.append({"input_shape": list(x.shape), "output_shape": list(a.shape),
                       "output": error(a, b), "input_gradient": error(x.grad, y.grad),
                       "RDN_gradient": compare_dicts(rdn_a, rdn_b),
                       "P2W_gradient": compare_dicts(ga, gb, "P2W."),
                       "all_parameter_gradients": compare_dicts(ga, gb)})
        print("scale32", device, size, output[-1], flush=True)
        del reference, model, a, b, ga, gb, rdn_a, rdn_b
    return output


@torch.no_grad()
def crops_boundaries_stream():
    results = {}
    for scale, h, w in ((4, 5, 7), (32, 2, 3)):
        torch.manual_seed(2)
        model = MemoryEfficientMetaRDN(scale=scale, lr_chunk_size=2).eval()
        reference = OfficialMetaRDN(official_args(scale)).eval()
        reference.set_scale(0)
        reference.load_state_dict(model.state_dict())
        feature = torch.randn(1, 64, h, w)
        pos, _ = official_positions(h, w, scale)
        full = official_decoder(reference, feature, pos)
        streamed, coverage = torch.empty_like(full), torch.zeros(h*scale, w*scale, dtype=torch.int16)
        with patch.object(model, "repeat_x", side_effect=AssertionError("repeat_x forbidden")), \
             patch("models.metasr.F.unfold", wraps=F.unfold) as unfold:
            for y, x, tile in model.iter_full_sr(feature):
                th, tw = tile.shape[-2:]
                assert not coverage[y:y+th, x:x+tw].any()
                coverage[y:y+th, x:x+tw] += 1
                streamed[:, :, y:y+th, x:x+tw] = tile
            assert unfold.call_count == 1
        assert coverage.eq(1).all()
        corner = {}
        for name, iy, ix in (("top_left", 0, 0), ("top_right", 0, w-1),
                             ("bottom_left", h-1, 0), ("bottom_right", h-1, w-1)):
            corner[name] = error(streamed[:, :, iy*scale:(iy+1)*scale, ix*scale:(ix+1)*scale],
                                 full[:, :, iy*scale:(iy+1)*scale, ix*scale:(ix+1)*scale])
        edge = torch.zeros(h, w, dtype=torch.bool)
        edge[0, :] = edge[-1, :] = True
        edge[:, 0] = edge[:, -1] = True
        edge = edge.repeat_interleave(scale, 0).repeat_interleave(scale, 1)
        padded = F.pad(feature, (1, 1, 1, 1), mode="constant", value=0)
        manual = torch.stack([padded[:, :, y:y+3, x:x+3].reshape(1, -1)
                              for y in range(h) for x in range(w)], dim=2)
        assert torch.equal(manual, F.unfold(feature, 3, padding=1))
        entry = {"stream_vs_official": error(streamed, full), "coverage_min": int(coverage.min()),
                 "coverage_max": int(coverage.max()), "corner_subpixels": corner,
                 "all_edge_subpixels": error(streamed[:, :, edge], full[:, :, edge]),
                 "manual_zero_padding_exact": True}
        if scale == 4:
            box = (3, 7, 13, 17)
            crop = model.decode_crop(feature, box)
            entry["unaligned_crop"] = {"box": box, **error(crop, full[:, :, 3:16, 7:24])}
        results[str(scale)] = entry

    # Absolute-coordinate independent oracle at the user's example, without
    # allocating every HR dynamic kernel for this larger feature image.
    model = MemoryEfficientMetaRDN(scale=32, lr_chunk_size=2).eval()
    feature = torch.randn(1, 64, 16, 16)
    pos, _ = official_positions(16, 16, 32)
    box = (137, 291, 35, 53)
    yy, xx = torch.meshgrid(torch.arange(137, 172), torch.arange(291, 344), indexing="ij")
    official_pos = pos.reshape(512, 512, 3)[yy, xx].reshape(-1, 3)
    manual_pos = torch.stack((torch.full_like(yy, 1/32, dtype=torch.float32),
                             (yy % 32).float()/32, (xx % 32).float()/32), -1).reshape(-1, 3)
    assert torch.equal(official_pos, manual_pos)
    indices = ((yy//32)*16 + xx//32).reshape(-1)
    patches = F.unfold(feature, 3, padding=1).index_select(2, indices).transpose(1, 2)
    kernels = model.P2W(official_pos).reshape(-1, 576, 3)
    oracle = torch.einsum("bpk,pkc->bpc", patches, kernels).transpose(1, 2).reshape(1, 3, 35, 53)
    oracle = model.add_mean(oracle)
    results["absolute_crop32"] = {"box": box, "first_LR_index": int(indices[0]),
        "first_position": official_pos[0].tolist(), "positions_exact": True,
        "crop_vs_absolute_oracle": error(model.decode_crop(feature, box), oracle)}
    return results


def checkpoint_train():
    torch.manual_seed(319)
    a = MemoryEfficientMetaRDN(checkpoint_rdb=False, lr_chunk_size=2).train()
    b = MemoryEfficientMetaRDN(checkpoint_rdb=True, lr_chunk_size=2).train()
    b.load_state_dict(a.state_dict())
    optimizers = [torch.optim.Adam(m.parameters(), lr=1e-5, weight_decay=1e-4) for m in (a, b)]
    x = torch.rand(1, 3, 3, 4, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    oa, ob = a(x, (3, 7, 40, 49)), b(y, (3, 7, 40, 49))
    oa.square().mean().backward()
    ob.square().mean().backward()
    ga, gb = grads(a), grads(b)
    output = {"train_mode": True, "dtype": "float32", "forward": error(oa, ob),
              "input_gradient": error(x.grad, y.grad), "all_gradients": compare_dicts(ga, gb),
              "RDB_gradients": {str(i+1): compare_dicts(ga, gb, f"RDBs.{i}.") for i in range(16)},
              "GFF_gradient": compare_dicts(ga, gb, "GFF.")}
    for optimizer in optimizers:
        optimizer.step()
    output["after_adam_parameters"] = compare_dicts(dict(a.named_parameters()), dict(b.named_parameters()))
    return output


def classifier_microbatch(dtype=torch.float32):
    torch.manual_seed(811)
    model = MetaSRABMIL().eval()
    lr = torch.rand(6, 3, 16, 16)
    model, lr = model.to(dtype), lr.to(dtype)
    sample = {"n_regions": 6, "label": 1, "lr_paths": list(range(6)), "hr_paths": []}
    cfg = {"lambda_sr": 0., "precision": "fp32", "sr_train_crop": 32}
    captures = []
    for batch in (6, 1):
        model.zero_grad(set_to_none=True)
        embeddings, attention = [], []
        handles = [model.region_encoder.register_forward_hook(lambda m, i, o: embeddings.append(o.detach().clone())),
                   model.mil_head.register_forward_hook(lambda m, i, o: attention.append(o[1].detach().clone()))]
        with patch("engine.load_images", side_effect=lambda paths, size: lr[paths]), \
             patch("engine.load_hr_crops", side_effect=AssertionError("SR OFF HR read")), \
             patch.object(model.sr, "decode_crop", side_effect=AssertionError("SR OFF decoder")), \
             patch.object(model.sr.P2W, "forward", side_effect=AssertionError("SR OFF P2W")):
            if dtype == torch.float32:
                region_cfg = {"use_region_microbatch": batch != 6, "region_microbatch_size": batch}
                total, cls, sr, logits = wsi_loss(model, sample, torch.device("cpu"),
                    dict(cfg, training=region_cfg))
            else:
                # Diagnostic only: preserve FP64 instead of the production engine's
                # explicit embedding.float(). The architecture and bag math are identical.
                bag = torch.cat([model.encode_regions(part)[1] for part in lr.split(batch)])
                logits = model.forward_embeddings(bag)
                cls = total = F.cross_entropy(logits, torch.tensor([1]))
                sr = cls.new_zeros(())
            total.backward()
        for handle in handles:
            handle.remove()
        stats = chain(model)
        for name, item in stats.items():
            if name != "P2W":
                assert item["l2"] > 0 and item["all_finite"] and not item["missing_names"], (name, item)
        assert stats["P2W"]["with_grad"] == 0 and sr.item() == 0
        captures.append({"embedding": torch.cat(embeddings), "attention": attention[0],
                         "logits": logits.detach(), "loss": cls.detach(), "grads": grads(model), "chain": stats})
    a, b = captures
    result = {key: error(a[key], b[key], atol=2e-5, enforce=False) for key in ("embedding", "attention", "logits", "loss")}
    result.update({"RDN_gradients": compare_dicts(a["grads"], b["grads"], "sr.", enforce=False),
                   "ResNet_gradients": compare_dicts(a["grads"], b["grads"], "region_encoder.", enforce=False),
                   "ABMIL_gradients": compare_dicts(a["grads"], b["grads"], "mil_head.", enforce=False),
                   "classifier_gradients": compare_dicts(a["grads"], b["grads"], "classifier.", enforce=False),
                   "classification_only_chain_micro1": b["chain"], "input_shape": list(lr.shape),
                   "dtype": str(dtype), "SR_OFF_decoder_calls": 0, "SR_OFF_HR_reads": 0})
    return result


def branches_and_optimizer():
    torch.manual_seed(101)
    model = MetaSRABMIL().eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    counts = Counter(id(p) for group in optimizer.param_groups for p in group["params"])
    inventory = []
    for name, parameter in model.named_parameters():
        group = ("Pos2Weight" if name.startswith("sr.P2W") else "Meta-RDN/MeanShift" if name.startswith("sr.")
                 else "ResNet18" if name.startswith("region_encoder") else "ABMIL" if name.startswith("mil_head")
                 else "classifier")
        inventory.append({"name": name, "shape": list(parameter.shape), "numel": parameter.numel(),
                          "requires_grad": parameter.requires_grad, "group": group,
                          "optimizer_occurrences": counts[id(parameter)]})
        assert counts[id(parameter)] == 1
    assert set(counts) == {id(p) for p in model.parameters() if p.requires_grad}
    with (ROOT / "audit/optimizer_parameters.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    lr = torch.rand(3, 3, 16, 16)
    target = torch.tensor([1])
    gt = torch.rand(3, 3, 32, 33)
    branches, gradients = {}, {}
    for name in ("cls", "sr", "joint"):
        model.zero_grad(set_to_none=True)
        if name == "sr":
            loss = F.l1_loss(model.sr(lr, (3, 7, 32, 33)), gt)
        else:
            total, cls, sr, _ = model.joint_loss(lr, target, lambda_sr=0.1 if name == "joint" else 0,
                                               hr_crop=gt, box=(3, 7, 32, 33))
            loss = total
        loss.backward()
        branches[name] = {"loss": loss.item(), "chain": chain(model)}
        gradients[name] = grads(model)
        assert branches[name]["chain"]["SFENet1"]["l2"] > 0
        if name != "cls":
            assert branches[name]["chain"]["P2W"]["l2"] > 0
        if name == "sr":
            assert branches[name]["chain"]["ABMIL"]["with_grad"] == 0
            assert branches[name]["chain"]["ResNet_first_conv"]["with_grad"] == 0
    summed = {}
    for name, g in gradients["joint"].items():
        a, b = gradients["cls"][name], gradients["sr"][name]
        summed[name] = None if a is None and b is None else ((a if a is not None else torch.zeros_like(b))
                                                           + 0.1*(b if b is not None else torch.zeros_like(a)))
    branches["joint_gradient_equals_cls_plus_0.1sr"] = compare_dicts(summed, gradients["joint"])
    model.zero_grad(set_to_none=True)
    before = {n: p.detach().clone() for n, p in model.sr.P2W.named_parameters()}
    model.joint_loss(lr, target, lambda_sr=0)[0].backward()
    optimizer.step()
    assert all(torch.equal(before[n], p) for n, p in model.sr.P2W.named_parameters())
    groups = {}
    for row in inventory:
        item = groups.setdefault(row["group"], {"tensors": 0, "elements": 0})
        item["tensors"] += 1
        item["elements"] += row["numel"]
    return {"branches": branches, "optimizer_coverage": {"groups": groups, "missing": [], "duplicates": [],
             "trainable_tensors": len(inventory), "trainable_elements": sum(r["numel"] for r in inventory),
             "SR_OFF_P2W_unchanged_after_weight_decay_Adam": True}}


def normalization():
    torch.manual_seed(87)
    model = MetaSRABMIL().eval()
    lr = torch.rand(1, 3, 4, 4)
    gt = torch.rand(1, 3, 13, 15)
    results = []
    with torch.no_grad():
        expected = F.l1_loss(model.sr(lr, (3, 7, 13, 15)), gt).item()
        # Engine assumes square crops. Use 13x13 and independently calculate the mean.
        gt = gt[:, :, :, :13]
        expected = F.l1_loss(model.sr(lr, (3, 7, 13, 13)), gt)
        for n, batch in ((1, 1), (5, 2), (6, 1), (6, 6)):
            sample = {"n_regions": n, "label": 1, "lr_paths": list(range(n)), "hr_paths": list(range(n))}
            cfg = {"lambda_sr": 0.1, "precision": "fp32", "sr_train_crop": 13,
                   "training": {"use_region_microbatch": batch != n, "region_microbatch_size": batch}}
            with patch("engine.load_images", side_effect=lambda paths, size: lr.repeat(len(paths), 1, 1, 1)), \
                 patch("engine.load_hr_crops", side_effect=lambda paths, box: gt.repeat(len(paths), 1, 1, 1)), \
                 patch("engine.torch.randint", return_value=torch.tensor([[3, 7]] * n)):
                total, cls, sr, _ = wsi_loss(model, sample, torch.device("cpu"), cfg)
            results.append({"N": n, "micro_batch": batch, "sr_mean": sr.item(),
                            "expected": expected.item(), "error": error(sr, expected),
                            "total_formula": error(total, cls + .1*sr)})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--only-scale32", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "audit/math_cpu.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    results = {"device": args.device, "dtype": "float32", "seeded": True,
               "reference_root": str(SOURCE), "production_files_modified": False}
    checks = {"scale32": lambda: scale32_equivalence(args.device)}
    if not args.only_scale32:
        checks.update(architecture=architecture, crops_boundaries_stream=crops_boundaries_stream,
                      checkpoint_train=checkpoint_train, classification_microbatch=classifier_microbatch,
                      branches_optimizer=branches_and_optimizer, loss_normalization=normalization)
        checks["microbatch_fp64_diagnostic"] = lambda: classifier_microbatch(torch.float64)
    for name, function in checks.items():
        print("START", name, flush=True)
        results[name] = function()
        print("PASS", name, flush=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    results["all_checks_completed"] = True
    results["microbatch_uses_recorded_errors_not_silent_tolerance_relaxation"] = not args.only_scale32
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
