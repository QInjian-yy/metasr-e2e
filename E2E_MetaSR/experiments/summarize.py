"""Append one real training run to the summary; never rewrite an existing experiment."""
import argparse
import csv
import math
import re
from datetime import date
from pathlib import Path

import yaml

FIELDS = ["experiment_id","recorded_at","status","classification_encoder","rdn_blocks","fold","seed","git_commit","gpu","best_epoch","train_loss_cls","train_loss_sr","train_loss_total","train_auc","train_acc","train_bacc","val_loss","val_auc","val_acc","val_bacc","run_dir","history_file","log_file","config_file","record_file"]
METRICS = ("train_loss_cls", "train_loss_sr", "train_loss_total", "train_auc", "train_acc",
           "train_bacc", "val_loss", "val_auc", "val_acc", "val_bacc")


def summarize(args):
    if not re.fullmatch(r"EXP-\d{8}-[A-Z0-9-]+-F[012]-S\d+-\d{3}", args.experiment_id):
        raise ValueError("Use EXP-YYYYMMDD-MODEL-Dn-Fn-Sn-NNN")
    if not re.fullmatch(r"[0-9a-f]{40}", args.git_commit):
        raise ValueError("Record the full Git commit used when training started")
    if not args.record_file.is_file() or not args.log_file.is_file():
        raise ValueError("Create the experiment record and keep the actual training log")
    with args.summary.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError("Summary columns differ from the expected schema")
        if any(row["experiment_id"] == args.experiment_id for row in reader):
            raise ValueError("Experiment ID already exists; historical results cannot be overwritten")
    history_file = args.run_dir / "history.csv"
    config_file = args.run_dir / "config.yaml"
    with history_file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    valid = [row for row in rows if math.isfinite(float(row["val_auc"]))]
    if not valid:
        raise ValueError("No finite validation AUC; do not invent a best epoch")
    # Python max keeps the first tie, matching the trainer's strict val_auc improvement.
    best = max(valid, key=lambda row: float(row["val_auc"]))
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    row = {"experiment_id": args.experiment_id, "recorded_at": date.today().isoformat(),
           "status": args.status, "classification_encoder": config.get("classification_encoder", "resnet18"),
           "rdn_blocks": config["metasr"].get("rdn_blocks", 16), "fold": args.fold,
           "seed": config["seed"], "git_commit": args.git_commit, "gpu": args.gpu,
           "best_epoch": best["epoch"], **{key: best[key] for key in METRICS},
           "run_dir": str(args.run_dir), "history_file": str(history_file),
           "log_file": str(args.log_file), "config_file": str(config_file),
           "record_file": str(args.record_file)}
    with args.summary.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=FIELDS).writerow(row)
    return row


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--status", choices=("completed", "early_stopped", "interrupted"), required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--record-file", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=root / "results/summary.csv")
    print(summarize(parser.parse_args()))


if __name__ == "__main__":
    main()
