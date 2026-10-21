"""Evaluate the saved fold using all regions and classification only."""

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_modes import checkpoint_data_spec
from engine import evaluate
from models.baseline import model_from_checkpoint
from train_e2e import ROOT, provenance
from wsi_data import collate_one_wsi, load_fold_datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--predictions-csv", type=Path)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    spec = checkpoint_data_spec(saved)
    if spec.mode == "legacy256":
        args.labels_csv = args.labels_csv or ROOT / "downstream_train/camelyon16_labels.csv"
        args.split_dir = args.split_dir or ROOT / "downstream_train"
    device = torch.device(args.device)
    model = model_from_checkpoint(saved, device)
    _, val = load_fold_datasets(args.data_root, saved["fold"], args.labels_csv, args.split_dir,
                               require_hr=False,
                               **({"data_mode": spec.mode,
                                   "manifest_csv": saved["config"].get("data_manifest", "manifests/patch_manifest.csv")}
                                  if spec.mode == "native512" else {}))
    hashes = lambda info: {key: value["sha256"] for key, value in info.items()}
    if hashes(provenance(args.data_root, val)) != hashes(saved["data_provenance"]):
        if spec.mode == "native512":
            raise ValueError("Manifest/labels/split/verification differ from the training checkpoint")
        raise ValueError("Manifest/labels/split differ from the training checkpoint")
    loader = DataLoader(val, batch_size=1, collate_fn=collate_one_wsi)
    result = evaluate(model, loader, device, saved["config"])
    predictions = result.pop("predictions")
    print(result)
    if args.predictions_csv:
        args.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)


if __name__ == "__main__":
    main()
