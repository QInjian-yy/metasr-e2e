"""Independent CAMELYON16 Meta-RDN + Meta-SR + ResNet18 + ABMIL training."""

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from engine import evaluate, train_wsi
from models.baseline import MetaSRABMIL
from validation import run_equivalence
from wsi_data import collate_one_wsi, load_fold_datasets

ROOT = Path(__file__).resolve().parent


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config["metasr"]["scale"] != 32 or config["metasr"]["rgb_range"] != 1:
        raise ValueError("This dataset uses 256 -> 8192 and RGB in [0,1]: scale=32, rgb_range=1")
    if config["precision"] not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    if not isinstance(config["metasr"]["checkpoint_rdb"], bool):
        raise ValueError("checkpoint_rdb must be a boolean")
    rdn_blocks = config["metasr"].get("rdn_blocks", 16)
    if isinstance(rdn_blocks, bool) or not isinstance(rdn_blocks, int) or not 1 <= rdn_blocks <= 16:
        raise ValueError("metasr.rdn_blocks must be an integer from 1 to 16")
    training = config["training"]
    if not isinstance(training["use_region_microbatch"], bool):
        raise ValueError("training.use_region_microbatch must be a boolean")
    for name in ("region_microbatch_size",):
        value = training[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    for name in ("sr_train_crop", "epochs", "early_stopping_patience"):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config["sr_train_crop"] >= 8192:
        raise ValueError("Training requires a crop smaller than 8192; use infer_sr.py for full reconstruction")
    if not math.isfinite(config["lambda_sr"]) or config["lambda_sr"] < 0:
        raise ValueError("lambda_sr must be finite and nonnegative")
    return config


def provenance(data_root, dataset):
    paths = {"manifest": Path(data_root) / "manifests/patch_manifest.csv",
             "labels": dataset.labels_csv, "split": dataset.split_csv}
    return {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in paths.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.yaml")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--labels-csv", type=Path, default=ROOT / "downstream_train/camelyon16_labels.csv")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "downstream_train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 unsupported on this GPU; configure fp32")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new or empty output directory to preserve existing runs")
    args.output.mkdir(parents=True, exist_ok=True)
    # Check the selected depth against the vendored forward before training.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    gate = run_equivalence(rdn_blocks=config["metasr"].get("rdn_blocks", 16))
    (args.output / "equivalence.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(f"Decoder equivalence and checkpoint gate passed: {gate['selected_config']}", flush=True)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    train_data, val_data = load_fold_datasets(args.data_root, args.fold, args.labels_csv,
        args.split_dir, require_hr=config["lambda_sr"] > 0)
    if set(sample["label"] for sample in val_data.samples) != {0, 1}:
        raise ValueError("Validation needs both classes for val_auc checkpoint selection")
    loaders = [DataLoader(data, batch_size=1, shuffle=shuffle, collate_fn=collate_one_wsi)
               for data, shuffle in ((train_data, True), (train_data.evaluation_view(), False), (val_data, False))]
    model = MetaSRABMIL(**config["metasr"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    data_info = provenance(args.data_root, train_data)
    (args.output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (args.output / "data_provenance.json").write_text(json.dumps(data_info, indent=2), encoding="utf-8")
    best_auc, stale, iteration = -math.inf, 0, 0
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for epoch in range(1, config["epochs"] + 1):
            online = []
            for sample in loaders[0]:
                iteration += 1
                result = train_wsi(model, sample, optimizer, device, config, iteration % 10 == 0)
                online.append(result)
                print(f"[WSI] epoch={epoch} iteration={iteration} slide={sample['slide_id']} "
                      f"N={sample['n_regions']} {result}", flush=True)
            train_result = evaluate(model, loaders[1], device, config)
            val_result = evaluate(model, loaders[2], device, config)
            row = {"epoch": epoch, **{key: sum(r[key] for r in online) / len(online)
                   for key in ("train_loss_cls", "train_loss_sr", "train_loss_total")},
                   **{"train_"+key: train_result[key] for key in ("auc", "acc", "bacc")},
                   **{"val_"+key: val_result[key] for key in ("loss", "auc", "acc", "bacc")},
                   **{key: max((r[key] or 0) for r in online)
                      for key in ("max_memory_allocated", "max_memory_reserved")}}
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            improved = val_result["auc"] > best_auc
            if improved:
                best_auc, stale = val_result["auc"], 0
            else:
                stale += 1
            saved = {"format": "metasr-abmil-v1", "epoch": epoch, "fold": args.fold,
                     "config": config, "data_provenance": data_info, "model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict(), "best_val_auc": best_auc}
            torch.save(saved, args.output / "last.pth")
            if improved:
                torch.save(saved, args.output / "best.pth")
                with (args.output / "best_val_predictions.csv").open("w", newline="", encoding="utf-8") as out:
                    predictions = val_result["predictions"]
                    table = csv.DictWriter(out, fieldnames=list(predictions[0]))
                    table.writeheader()
                    table.writerows(predictions)
            print(f"[epoch] {row}", flush=True)
            if stale >= config["early_stopping_patience"]:
                break


if __name__ == "__main__":
    main()
