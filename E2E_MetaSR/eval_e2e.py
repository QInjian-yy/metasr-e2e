"""Evaluate the saved fold using all regions and classification only."""

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from engine import evaluate
from models.baseline import MetaSRABMIL
from train_e2e import ROOT, provenance
from wsi_data import collate_one_wsi, load_fold_datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, default=ROOT / "downstream_train/camelyon16_labels.csv")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "downstream_train")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--predictions-csv", type=Path)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("format") != "metasr-abmil-v1":
        raise ValueError("Expected a MetaSR baseline checkpoint")
    _, val = load_fold_datasets(args.data_root, saved["fold"], args.labels_csv, args.split_dir, require_hr=False)
    hashes = lambda info: {key: value["sha256"] for key, value in info.items()}
    if hashes(provenance(args.data_root, val)) != hashes(saved["data_provenance"]):
        raise ValueError("Manifest/labels/split differ from the training checkpoint")
    device = torch.device(args.device)
    model = MetaSRABMIL(**saved["config"]["metasr"]).to(device)
    model.load_state_dict(saved["model_state"], strict=True)
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
