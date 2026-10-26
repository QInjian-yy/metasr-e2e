"""Train a new MeanMIL/ABMIL run using the unchanged WSI engine and data contracts."""

import argparse
import csv
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from augmentation import load_augmentation_config
from data_modes import data_spec
from engine import evaluate, train_wsi
from models.mil_model import CHECKPOINT_FORMAT, MetaSRMIL
from train_e2e import ROOT, load_config as load_baseline_config, provenance, run_metadata
from validation import run_equivalence
from wsi_data import collate_one_wsi, load_fold_datasets


def load_config(path):
    original = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(original, dict) and original.get("classification_encoder") == "spp_region_gate":
        from train_e2e_region_gate import load_config as load_region_gate_config

        config = load_region_gate_config(path)
    else:
        config = load_baseline_config(path)
    config.setdefault("mil_pooling", "abmil")
    if config["mil_pooling"] not in ("abmil", "meanmil"):
        raise ValueError("mil_pooling must be abmil or meanmil")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.yaml")
    parser.add_argument("--mil-pooling", choices=("abmil", "meanmil"),
                        help="Override mil_pooling in the YAML; omitted options default to abmil")
    parser.add_argument("--augmentation-config", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--labels-csv", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mil_pooling is not None:
        config["mil_pooling"] = args.mil_pooling
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
    print(f"[MIL] pooling={config['mil_pooling']} encoder={config['classification_encoder']}", flush=True)
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
    model = MetaSRMIL(classification_encoder=config["classification_encoder"],
                      classification_head=config["classification_head"],
                      mil_pooling=config["mil_pooling"], **config["metasr"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    data_info = provenance(args.data_root, train_data)
    (args.output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (args.output / "data_provenance.json").write_text(json.dumps(data_info, indent=2), encoding="utf-8")
    info = run_metadata(args.fold, device)
    info["mil_pooling"] = config["mil_pooling"]
    info_file = args.output / "run_info.json"
    info_file.write_text(json.dumps(info, indent=2), encoding="utf-8")
    best_auc, stale, iteration = -float("inf"), 0, 0
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
    # The legacy summary schema cannot distinguish pooling; keep this run's records local.
    print(f"[run] {info['experiment_id']} {info['status']} pooling={config['mil_pooling']} "
          f"best_val_auc={best_auc}", flush=True)


if __name__ == "__main__":
    main()
