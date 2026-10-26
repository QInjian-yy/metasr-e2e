"""Train region-gate V1 through the existing WSI engine, writing only a new run."""

import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from augmentation import AugmentationConfig, load_augmentation_config
from data_modes import data_spec, validate_data_config
from engine import evaluate, train_wsi
from models.region_gate_model import CHECKPOINT_FORMAT, RegionGateMetaSRABMIL
from train_e2e import ROOT, provenance, run_metadata
from validation import run_equivalence
from wsi_data import collate_one_wsi, load_fold_datasets


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("classification_encoder") != "spp_region_gate":
        raise ValueError("This entrypoint requires classification_encoder=spp_region_gate")
    config.setdefault("classification_head", "linear")
    if config["classification_head"] != "linear":
        raise ValueError("Region-gate V1 requires classification_head=linear")
    spec = validate_data_config(config)
    if config["precision"] not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    config["metasr"].setdefault("freeze_meanshift", False)
    for key in ("checkpoint_rdb", "freeze_meanshift"):
        if not isinstance(config["metasr"][key], bool):
            raise ValueError(f"metasr.{key} must be a boolean")
    depth = config["metasr"].get("rdn_blocks", 16)
    if type(depth) is not int or not 1 <= depth <= 16:
        raise ValueError("metasr.rdn_blocks must be an integer from 1 to 16")
    if not isinstance(config["training"]["use_region_microbatch"], bool):
        raise ValueError("training.use_region_microbatch must be a boolean")
    for key in ("sr_train_crop", "epochs", "early_stopping_patience"):
        value = config[key]
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    value = config["training"]["region_microbatch_size"]
    if type(value) is not int or value < 1:
        raise ValueError("training.region_microbatch_size must be a positive integer")
    if config["sr_train_crop"] >= spec.hr_size:
        raise ValueError("Training requires a crop smaller than 8192")
    if not math.isfinite(config["lambda_sr"]) or config["lambda_sr"] < 0:
        raise ValueError("lambda_sr must be finite and nonnegative")
    augmentation = config.get("augmentation", {})
    if not isinstance(augmentation, dict):
        raise ValueError("augmentation must be a mapping")
    AugmentationConfig(**augmentation)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/native512_d1_region_gate_sr01_noaug_m8_300.yaml")
    parser.add_argument("--augmentation-config", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--labels-csv", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    spec = data_spec(config)
    if spec.mode == "legacy256":
        args.labels_csv = args.labels_csv or ROOT / "downstream_train/camelyon16_labels.csv"
        args.split_dir = args.split_dir or ROOT / "downstream_train"
    if args.augmentation_config is not None:
        config["augmentation"] = asdict(load_augmentation_config(args.augmentation_config))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 unsupported on this GPU; configure fp32")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new or empty output directory to preserve existing runs")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    native_options = ({"data_mode": spec.mode, "manifest_csv": config.get(
        "data_manifest", "manifests/patch_manifest.csv")} if spec.mode == "native512" else {})
    gate = run_equivalence(rdn_blocks=config["metasr"].get("rdn_blocks", 16),
                           **({"scales": (2, 3, 4, 16)} if native_options else {}))
    (args.output / "equivalence.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(f"Decoder equivalence and checkpoint gate passed: {gate['selected_config']}", flush=True)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    train_data, val_data = load_fold_datasets(
        args.data_root, args.fold, args.labels_csv, args.split_dir,
        require_hr=config["lambda_sr"] > 0, augmentation=config.get("augmentation"), **native_options)
    if set(sample["label"] for sample in val_data.samples) != {0, 1}:
        raise ValueError("Validation needs both classes for val_auc checkpoint selection")
    loaders = [DataLoader(data, batch_size=1, shuffle=shuffle, collate_fn=collate_one_wsi)
               for data, shuffle in ((train_data, True), (train_data.evaluation_view(), False),
                                     (val_data, False))]
    model = RegionGateMetaSRABMIL(classification_encoder=config["classification_encoder"],
                                 classification_head=config["classification_head"],
                                 **config["metasr"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    data_info = provenance(args.data_root, train_data)
    (args.output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (args.output / "data_provenance.json").write_text(json.dumps(data_info, indent=2), encoding="utf-8")
    info = run_metadata(args.fold, device)
    info_file = args.output / "run_info.json"
    info_file.write_text(json.dumps(info, indent=2), encoding="utf-8")
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
                   **{"train_" + key: train_result[key] for key in ("auc", "acc", "bacc")},
                   "train_eval_loss": train_result["loss"],
                   **{"val_" + key: val_result[key] for key in ("loss", "auc", "acc", "bacc")},
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
            saved = {"format": CHECKPOINT_FORMAT, "architecture": model.architecture(),
                     "data_spec": spec.metadata(), "epoch": epoch, "fold": args.fold,
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
    info["status"] = "early_stopped" if stale >= config["early_stopping_patience"] else "completed"
    info["finished_at"] = datetime.now().astimezone().isoformat()
    info_file.write_text(json.dumps(info, indent=2), encoding="utf-8")
    # Keep records in this run; do not append to the legacy experiment summary.
    print(f"[run] {info['experiment_id']} {info['status']} best_val_auc={best_auc}", flush=True)


if __name__ == "__main__":
    main()
