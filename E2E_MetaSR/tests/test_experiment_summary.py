import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from experiments.summarize import FIELDS, METRICS, summarize


class ExperimentSummaryTests(unittest.TestCase):
    def fixture(self, root, aucs):
        run = root / "run"
        run.mkdir()
        (run / "config.yaml").write_text(
            "classification_encoder: spp\nmetasr:\n  rdn_blocks: 8\nseed: 1\n", encoding="utf-8")
        history = run / "history.csv"
        with history.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", *METRICS])
            writer.writeheader()
            for epoch, auc in enumerate(aucs, 1):
                writer.writerow({"epoch": epoch, **{key: 0.25 for key in METRICS}, "val_auc": auc})
        summary = root / "summary.csv"
        summary.write_text(",".join(FIELDS) + "\n", encoding="utf-8")
        log = root / "training.log"
        log.write_text("synthetic fixture only", encoding="utf-8")
        record = root / "experiment.md"
        record.write_text("synthetic fixture only", encoding="utf-8")
        return Namespace(experiment_id="EXP-20261009-SPP-D8-F0-S1-001", run_dir=run, fold=0,
                         git_commit="f" * 40, gpu="synthetic CPU", status="completed",
                         log_file=log, record_file=record, summary=summary)

    def test_best_epoch_first_tie_and_duplicate_id_preserve_history(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp), (0.6, 0.9, 0.9))
            history_before = (args.run_dir / "history.csv").read_bytes()
            row = summarize(args)
            self.assertEqual(row["best_epoch"], "2")
            self.assertEqual(row["val_auc"], "0.9")
            self.assertEqual(row["classification_encoder"], "spp")
            summary_before = args.summary.read_bytes()
            with self.assertRaisesRegex(ValueError, "already exists"):
                summarize(args)
            self.assertEqual(args.summary.read_bytes(), summary_before)
            self.assertEqual((args.run_dir / "history.csv").read_bytes(), history_before)
            with args.summary.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)

    def test_no_valid_auc_does_not_create_a_result(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp), ("nan",))
            before = args.summary.read_bytes()
            with self.assertRaisesRegex(ValueError, "No finite"):
                summarize(args)
            self.assertEqual(args.summary.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
