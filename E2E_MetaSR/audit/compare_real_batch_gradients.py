"""Compare full/micro1 real WSI gradients and optional post-Adam states."""

import json
import math
import sys
from pathlib import Path

import torch


ATOL, RTOL = 2e-5, 2e-4  # Existing tests/test_engine.py batching comparison tolerances.


def group_name(name):
    if name.startswith("sr.P2W."):
        return "pos2weight"
    if name.startswith("sr."):
        return "rdn"
    return name.split(".")[0]


def tensor_stats(reference, actual, atol=ATOL, rtol=RTOL):
    a, b = reference.double().flatten(), actual.double().flatten()
    delta = b - a
    absolute = delta.abs()
    tolerance = atol + rtol * a.abs()
    significant = a.abs() > atol
    return {"elements": a.numel(), "reference_sq": float(a.square().sum()),
            "actual_sq": float(b.square().sum()), "difference_sq": float(delta.square().sum()),
            "dot": float(torch.dot(a, b)), "max_abs_error": float(absolute.max()),
            "outside_tolerance": int((absolute > tolerance).sum()),
            "bitwise_different": int((a != b).sum()),
            "significant_reference_elements": int(significant.sum()),
            "sign_flips_on_significant_elements": int(((a * b < 0) & significant).sum())}


def summarize(values):
    additive = ("elements", "reference_sq", "actual_sq", "difference_sq", "dot",
                "outside_tolerance", "bitwise_different", "significant_reference_elements",
                "sign_flips_on_significant_elements")
    total = {key: sum(value[key] for value in values) for key in additive}
    total["max_abs_error"] = max(value["max_abs_error"] for value in values)
    total["reference_l2"] = math.sqrt(total["reference_sq"])
    total["actual_l2"] = math.sqrt(total["actual_sq"])
    total["relative_l2_error"] = math.sqrt(total["difference_sq"]) / max(total["reference_l2"], 1e-30)
    denominator = math.sqrt(total["reference_sq"] * total["actual_sq"])
    total["cosine_similarity"] = max(-1.0, min(1.0, total["dot"] / denominator)) if denominator else None
    total["allclose"] = total["outside_tolerance"] == 0
    total["bitwise_equal"] = total["bitwise_different"] == 0
    return total


def compare_training_states(directory, names, full):
    paths = [directory / f"{name}_state.pt" for name in names]
    if not any(path.is_file() for path in paths):
        return None
    assert all(path.is_file() for path in paths), "Both training-state snapshots are required"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models.baseline import MetaSRABMIL
    import hashlib

    torch.manual_seed(full["seed"])
    initial = MetaSRABMIL(**full["model_config"])
    digest = hashlib.sha256()
    for name, value in initial.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().numpy().tobytes())
    assert digest.hexdigest() == full["initial_state_sha256"]
    first, second = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    assert first["parameters"].keys() == second["parameters"].keys()
    assert first["adam"].keys() == second["adam"].keys()
    # Fixed before examining results. Update atol is 1% of this experiment's lr=1e-5.
    tolerances = {"parameters": (1e-7, 1e-5), "updates": (1e-7, 1e-4),
                  "exp_avg": (2e-6, 2e-4), "exp_avg_sq": (2e-8, 4e-4)}
    values = {key: [] for key in tolerances}
    worst_updates, steps, absent = [], [], []
    for name, parameter in initial.named_parameters():
        a, b = first["parameters"][name], second["parameters"][name]
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        values["parameters"].append(tensor_stats(a, b, *tolerances["parameters"]))
        delta_a = a.double() - parameter.detach().double()
        delta_b = b.double() - parameter.detach().double()
        update = tensor_stats(delta_a, delta_b, *tolerances["updates"])
        values["updates"].append(update)
        worst_updates.append({"name": name, **summarize([update])})
        state_a, state_b = first["adam"][name], second["adam"][name]
        assert state_a.keys() == state_b.keys()
        if not state_a:
            absent.append(name)
            continue
        assert state_a["step"].item() == state_b["step"].item() == 1
        steps.append(name)
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.isfinite(state_a[key]).all() and torch.isfinite(state_b[key]).all()
            values[key].append(tensor_stats(state_a[key], state_b[key], *tolerances[key]))
    metrics = {key: summarize(rows) for key, rows in values.items()}
    return {"tolerances": {key: {"atol": pair[0], "rtol": pair[1]} for key, pair in tolerances.items()},
            "metrics": metrics, "all_adam_steps_equal_one": True, "adam_parameter_count": len(steps),
            "parameters_without_adam_state": absent,
            "worst_updates": sorted(worst_updates, key=lambda row: row["max_abs_error"], reverse=True)[:10],
            "numerically_close": all(value["allclose"] for value in metrics.values()),
            "bitwise_equal": all(value["bitwise_equal"] for value in metrics.values())}


def main():
    directory = Path(sys.argv[1])
    precision = sys.argv[2]
    torch.set_num_threads(4)
    names = (f"{precision}_full", f"{precision}_micro1")
    reports = [json.loads((directory / f"{name}.json").read_text()) for name in names]
    for report in reports:
        assert report["status"] == "passed" and all(report["training_stages"].values())
        assert report["gradients_finite"] and not report["synthetic"]
    full, micro = reports
    assert full["initial_state_sha256"] == micro["initial_state_sha256"]
    assert full["hr_crops"] == micro["hr_crops"]
    assert full["sample"] == micro["sample"]
    n = full["sample"]["n_regions"]
    assert full["effective_region_batch_size"] == n and micro["effective_region_batch_size"] == 1
    for key in ("precision", "lambda_sr", "sr_crop", "checkpoint_rdb", "tf32"):
        assert full[key] == micro[key], key
    for key in ("model_config", "seed", "optimizer", "determinism"):
        assert full.get(key) == micro.get(key), key
    if "encoder_inputs" in full:
        assert [row["shape"][0] for row in full["encoder_inputs"]] == [n]
        assert [row["shape"][0] for row in micro["encoder_inputs"]] == [1] * n
        for report in reports:
            assert len(report["mil_inputs"]) == 1 and report["mil_inputs"][0]["shape"] == [n, 512]
            assert all(row["requires_grad"] and row["has_grad_fn"]
                       for row in report["encoder_inputs"] + report["mil_inputs"])
    snapshots = [torch.load(directory / f"{name}_gradients.pt", map_location="cpu", weights_only=True)
                 for name in names]
    reference, actual = snapshots
    assert reference.keys() == actual.keys()
    parameters, missing = {}, []
    for name, grad in reference.items():
        other = actual[name]
        if grad is None or other is None:
            assert grad is None and other is None, name
            missing.append(name)
            continue
        assert grad.shape == other.shape and torch.isfinite(grad).all() and torch.isfinite(other).all()
        parameters[name] = tensor_stats(grad, other)
    groups = {group: summarize([value for name, value in parameters.items() if group_name(name) == group])
              for group in sorted({group_name(name) for name in parameters})}
    global_stats = summarize(list(parameters.values()))
    repeat_stats = None
    repeat_path = directory / f"{precision}_full_repeat_gradients.pt"
    if repeat_path.is_file():
        repeat_report = json.loads((directory / f"{precision}_full_repeat.json").read_text())
        assert repeat_report["status"] == "passed" and all(repeat_report["training_stages"].values())
        assert repeat_report["initial_state_sha256"] == full["initial_state_sha256"]
        assert repeat_report["hr_crops"] == full["hr_crops"]
        for key in ("sample", "model_config", "seed", "optimizer", "tf32", "determinism"):
            assert repeat_report.get(key) == full.get(key), key
        repeated = torch.load(repeat_path, map_location="cpu", weights_only=True)
        assert reference.keys() == repeated.keys()
        assert all((reference[name] is None) == (repeated[name] is None) for name in reference)
        repeat_stats = summarize([tensor_stats(grad, repeated[name]) for name, grad in reference.items()
                                  if grad is not None])
    first = next(name for name, grad in reference.items() if grad is not None and grad.abs().max() > ATOL)
    self_check = tensor_stats(reference[first], reference[first])
    scale_negative = tensor_stats(reference[first], reference[first] * 2)
    assert self_check["bitwise_different"] == 0 and scale_negative["outside_tolerance"] > 0
    losses = {key: {"full": full[key], "micro1": micro[key], "abs_error": abs(full[key]-micro[key])}
              for key in ("loss_total", "loss_cls", "loss_sr")}
    logits_error = float((torch.tensor(full["logits"]) - torch.tensor(micro["logits"])).abs().max())
    forward_stats = tensor_stats(torch.tensor([full[key] for key in losses] + full["logits"][0]),
                                 torch.tensor([micro[key] for key in losses] + micro["logits"][0]),
                                 atol=3e-6, rtol=3e-5)
    forward_close = forward_stats["outside_tolerance"] == 0
    training_states = compare_training_states(directory, names, full)
    repeat_training_states = None
    if (directory / f"{precision}_full_repeat_state.pt").is_file():
        repeat_training_states = compare_training_states(directory,
            (f"{precision}_full", f"{precision}_full_repeat"), full)
    result = {"precision": precision, "data_root": full["data_root"], "sample": full["sample"],
              "same_initial_state": True, "initial_state_sha256": full["initial_state_sha256"],
              "same_hr_crops": True, "hr_crops": full["hr_crops"],
              "atol": ATOL, "rtol": RTOL, "tolerance_source": "tests/test_engine.py batching gradient comparison",
              "losses": losses, "logits_max_abs_error": logits_error,
              "parameter_tensors_compared": len(parameters), "none_in_both": missing,
              "all_parameters": global_stats, "groups": groups,
              "forward_tolerance": {"atol": 3e-6, "rtol": 3e-5}, "forward_close": forward_close,
              "post_adam": training_states,
              "numerical_equivalence_passed": forward_close and global_stats["allclose"]
                  and (training_states is None or training_states["numerically_close"]),
              "bitwise_equal": forward_stats["bitwise_different"] == 0 and global_stats["bitwise_equal"]
                  and (training_states is None or training_states["bitwise_equal"]),
              "full_vs_full_repeat": repeat_stats,
              "full_vs_full_repeat_post_adam": repeat_training_states,
              "parameter_tensors_outside_tolerance": sum(value["outside_tolerance"] > 0 for value in parameters.values()),
              "worst_parameters_by_max_abs_error": sorted(
                  [{"name": name, **summarize([value])} for name, value in parameters.items()],
                  key=lambda value: value["max_abs_error"], reverse=True)[:10],
              "negative_control": {"parameter": first, "self_comparison_exact": True,
                                   "doubling_gradient_detected": True},
              "runs": [{"mode": mode, "max_memory_allocated": report["max_memory_allocated"],
                        "max_memory_reserved": report["max_memory_reserved"],
                        "elapsed_seconds": report["elapsed_seconds"], "tf32": report["tf32"]}
                       for mode, report in zip(("full", "micro1"), reports)],
              "parameters": parameters,
              "status": "comparison_complete", "strict_tolerance_passed": global_stats["allclose"]}
    path = directory / f"{precision}_comparison.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("parameters", "sample", "worst_parameters_by_max_abs_error")}, indent=2))
    print("SAVED", str(path))
    return 0 if result["numerical_equivalence_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
