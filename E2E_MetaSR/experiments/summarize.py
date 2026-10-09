"""Summarize a training output directory without changing historical results."""
import argparse
import csv
import hashlib
import json
import math
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["experiment_id","recorded_at","status","classification_encoder","rdn_blocks","fold","seed","git_commit","gpu","best_epoch","train_loss_cls","train_loss_sr","train_loss_total","train_auc","train_acc","train_bacc","val_loss","val_auc","val_acc","val_bacc","run_dir","history_file","log_file","config_file","record_file"]
METRICS = ("train_loss_cls", "train_loss_sr", "train_loss_total", "train_auc", "train_acc",
           "train_bacc", "val_loss", "val_auc", "val_acc", "val_bacc")


def summarize(run_dir):
    run_dir = Path(run_dir).resolve()
    info_file = run_dir / "run_info.json"
    info = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}
    experiment_id = info.get("experiment_id",
                             "LEGACY-" + hashlib.sha256(str(run_dir).encode()).hexdigest()[:12])
    summary = ROOT / "results/summary.csv"
    if summary.exists():
        with summary.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDS:
                raise ValueError("Summary columns differ from the expected schema")
            for existing in reader:
                if existing["experiment_id"] == experiment_id:
                    return existing
    history_file = run_dir / "history.csv"
    config_file = run_dir / "config.yaml"
    with history_file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    valid = [row for row in rows if math.isfinite(float(row["val_auc"]))]
    if not valid:
        raise ValueError("No finite validation AUC; do not invent a best epoch")
    # First tie matches the trainer's strict val_auc improvement.
    best = max(valid, key=lambda row: float(row["val_auc"]))
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    record = ROOT / "experiments" / (experiment_id + ".md")
    log = run_dir / "training.log"
    if not log.is_file():
        log = ROOT / "logs" / (run_dir.name + ".log")
    row = {"experiment_id": experiment_id, "recorded_at": date.today().isoformat(),
           "status": info.get("status", ""),
           "classification_encoder": config.get("classification_encoder", "resnet18"),
           "rdn_blocks": config["metasr"].get("rdn_blocks", 16), "fold": info.get("fold", ""),
           "seed": config["seed"], "git_commit": info.get("git_commit", ""), "gpu": info.get("gpu", ""),
           "best_epoch": best["epoch"], **{key: best[key] for key in METRICS},
           "run_dir": str(run_dir), "history_file": str(history_file),
           "log_file": str(log) if log.exists() else "", "config_file": str(config_file),
           "record_file": str(record)}
    record.parent.mkdir(parents=True, exist_ok=True)
    if not record.exists():
        lines = [f"# {experiment_id}", "", f"- 状态：{row['status'] or '未记录'}",
                 f"- 模型：{row['classification_encoder']} / D{row['rdn_blocks']}",
                 f"- Fold / seed：{row['fold']} / {row['seed']}",
                 f"- GPU：{row['gpu'] or '未记录'}", f"- Git Commit：{row['git_commit'] or '未记录'}",
                 f"- 开始：{info.get('started_at', '未记录')}",
                 f"- 结束：{info.get('finished_at', '未记录')}",
                 f"- 输出目录：{run_dir}", f"- 配置：{config_file}",
                 f"- 数据划分与哈希：{run_dir / 'data_provenance.json'}",
                 f"- 训练明细：{history_file}", f"- 终端日志：{row['log_file'] or '未保存'}",
                 "", "## 实验目的", "", "待填写（可选）。",
                 "", "## 最高 Validation AUC 对应 epoch", "",
                 "| 指标 | 值 |", "|---|---|", f"| epoch | {best['epoch']} |",
                 *[f"| {key} | {best[key]} |" for key in METRICS],
                 "", "## 实验结论", "", "待填写（可选）。"]
        record.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary.parent.mkdir(parents=True, exist_ok=True)
    new_file = not summary.exists()
    with summary.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    row = summarize(parser.parse_args().run_dir)
    print(f"{row['experiment_id']}: epoch {row['best_epoch']}, val_auc {row['val_auc']}")


if __name__ == "__main__":
    main()
