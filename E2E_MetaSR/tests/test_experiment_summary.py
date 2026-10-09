import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.summarize import FIELDS, METRICS, summarize
import train_e2e


class ExperimentSummaryTests(unittest.TestCase):
    def fixture(self, root, aucs, metadata=True):
        run = root / "run"
        run.mkdir()
        (run / "config.yaml").write_text(
            "classification_encoder: spp\nmetasr:\n  rdn_blocks: 8\nseed: 1\n", encoding="utf-8")
        if metadata:
            (run / "run_info.json").write_text(json.dumps({
                "experiment_id": "EXP-test", "fold": 0, "git_commit": "f" * 40,
                "gpu": "synthetic CPU", "status": "completed"}), encoding="utf-8")
        with (run / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", *METRICS])
            writer.writeheader()
            for epoch, auc in enumerate(aucs, 1):
                writer.writerow({"epoch": epoch, **{key: 0.25 for key in METRICS}, "val_auc": auc})
        return run

    def test_one_directory_best_epoch_and_repeat_preserve_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = self.fixture(root, (0.6, 0.9, 0.9))
            (root / "logs").mkdir()
            log = root / "logs/run.log"
            log.write_text("synthetic test log", encoding="utf-8")
            history_before = (run / "history.csv").read_bytes()
            with patch("experiments.summarize.ROOT", root):
                row = summarize(run)
                summary = root / "results/summary.csv"
                record = Path(row["record_file"])
                record.write_text(record.read_text(encoding="utf-8") + "\nMy conclusion\n",
                                  encoding="utf-8")
                before = summary.read_bytes(), record.read_bytes()
                self.assertEqual(summarize(run)["experiment_id"], row["experiment_id"])
            self.assertEqual(row["best_epoch"], "2")
            self.assertEqual(row["val_auc"], "0.9")
            self.assertEqual(row["git_commit"], "f" * 40)
            self.assertEqual(row["gpu"], "synthetic CPU")
            self.assertEqual(row["log_file"], str(log))
            self.assertEqual((summary.read_bytes(), record.read_bytes()), before)
            self.assertEqual((run / "history.csv").read_bytes(), history_before)
            with summary.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)

    def test_no_valid_auc_does_not_write_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = self.fixture(root, ("nan",))
            with patch("experiments.summarize.ROOT", root):
                with self.assertRaisesRegex(ValueError, "No finite"):
                    summarize(run)
            self.assertFalse((root / "results/summary.csv").exists())
            self.assertFalse((root / "experiments").exists())

    def test_old_run_missing_metadata_is_not_fabricated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = self.fixture(root, (0.7,), metadata=False)
            with patch("experiments.summarize.ROOT", root):
                row = summarize(run)
            for key in ("fold", "git_commit", "gpu", "status", "log_file"):
                self.assertEqual(row[key], "")
            self.assertTrue(Path(row["record_file"]).is_file())

    def test_existing_summary_rows_are_preserved_when_appending(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = self.fixture(root, (0.7,))
            summary = root / "results/summary.csv"
            summary.parent.mkdir()
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow({"experiment_id": "historical", "val_auc": "0.42"})
            with patch("experiments.summarize.ROOT", root):
                summarize(run)
            with summary.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["experiment_id"] for row in rows], ["historical", "EXP-test"])
            self.assertEqual(rows[0]["val_auc"], "0.42")

    def test_run_metadata_is_automatic_and_keeps_start_commit(self):
        from types import SimpleNamespace
        import torch
        result = SimpleNamespace(returncode=0, stdout="a" * 40 + "\n")
        with patch("train_e2e.subprocess.run", return_value=result):
            first = train_e2e.run_metadata(2, torch.device("cpu"))
            second = train_e2e.run_metadata(2, torch.device("cpu"))
        self.assertEqual(first["fold"], 2)
        self.assertEqual(first["git_commit"], "a" * 40)
        self.assertEqual(first["gpu"], "cpu")
        self.assertEqual(first["status"], "incomplete")
        self.assertNotEqual(first["experiment_id"], second["experiment_id"])
        with patch("train_e2e.subprocess.run", return_value=result), \
                patch("train_e2e.torch.cuda.get_device_name", return_value="AutoDL test GPU") as gpu:
            info = train_e2e.run_metadata(0, torch.device("cuda:0"))
        self.assertEqual(info["gpu"], "AutoDL test GPU")
        gpu.assert_called_once_with(torch.device("cuda:0"))

    def test_copy_without_git_can_still_record_training(self):
        import torch
        with patch("train_e2e.subprocess.run", side_effect=FileNotFoundError):
            info = train_e2e.run_metadata(0, torch.device("cpu"))
        self.assertEqual(info["git_commit"], "")
        self.assertEqual(info["gpu"], "cpu")


if __name__ == "__main__":
    unittest.main()
