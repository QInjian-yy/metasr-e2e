import csv
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import train_e2e


class TrainingRecordTests(unittest.TestCase):
    def run_entry(self, epochs, patience, fail=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run"
            config = {"seed": 1, "classification_encoder": "spp", "metasr": {"rdn_blocks": 8},
                      "lambda_sr": 0.1, "lr": 0.001, "weight_decay": 0,
                      "epochs": epochs, "early_stopping_patience": patience}
            train = SimpleNamespace(samples=[{"label": 0, "slide_id": "test", "n_regions": 1}])
            train.evaluation_view = lambda: train
            val = SimpleNamespace(samples=[{"label": 0}, {"label": 1}])
            model = torch.nn.Linear(1, 2)
            model.architecture = lambda: {"synthetic_test": True}
            online = {"train_loss_cls": 0.25, "train_loss_sr": 0.1, "train_loss_total": 0.26,
                      "max_memory_allocated": None, "max_memory_reserved": None}
            evaluated = {"loss": 0.25, "auc": 0.9, "acc": 0.5, "bacc": 0.5,
                         "predictions": [{"slide_id": "test", "label": 0, "probability": 0.5}]}
            with ExitStack() as stack:
                replacements = {
                    "train_e2e.ROOT": root, "experiments.summarize.ROOT": root,
                    "train_e2e.load_config": lambda path: config,
                    "train_e2e.load_fold_datasets": lambda *args, **kwargs: (train, val),
                    "train_e2e.DataLoader": lambda data, **kwargs: data.samples,
                    "train_e2e.MetaSRABMIL": lambda **kwargs: model,
                    "train_e2e.provenance": lambda *args: {},
                    "train_e2e.run_equivalence": lambda **kwargs: {"selected_config": "synthetic test"},
                    "sys.argv": ["train_e2e.py", "--data-root", str(root / "data"),
                                 "--output", str(run), "--device", "cpu"],
                }
                for name, value in replacements.items():
                    stack.enter_context(patch(name, value))
                stack.enter_context(patch("train_e2e.subprocess.run",
                                          return_value=SimpleNamespace(returncode=0, stdout="a" * 40)))
                trainer = stack.enter_context(patch("train_e2e.train_wsi", return_value=online,
                                                    side_effect=RuntimeError("test interruption") if fail else None))
                stack.enter_context(patch("train_e2e.evaluate", return_value=evaluated))
                stack.enter_context(redirect_stdout(io.StringIO()))
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "test interruption"):
                        train_e2e.main()
                else:
                    train_e2e.main()
            info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["git_commit"], "a" * 40)
            if fail:
                self.assertEqual(info["status"], "incomplete")
                self.assertNotIn("finished_at", info)
                self.assertFalse((root / "results/summary.csv").exists())
                return
            self.assertEqual(info["status"], "early_stopped" if patience == 1 else "completed")
            with (root / "results/summary.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["experiment_id"], info["experiment_id"])
            self.assertEqual(rows[0]["status"], info["status"])
            self.assertEqual(rows[0]["best_epoch"], "1")
            self.assertTrue(Path(rows[0]["record_file"]).is_file())
            self.assertTrue((run / "best.pth").is_file())
            self.assertEqual(trainer.call_count, 2 if patience == 1 else epochs)

    def test_normal_finish_records_results(self):
        self.run_entry(epochs=1, patience=5)

    def test_early_stop_records_results_without_extra_epochs(self):
        self.run_entry(epochs=3, patience=1)

    def test_interruption_is_not_recorded_as_completed(self):
        self.run_entry(epochs=1, patience=5, fail=True)


if __name__ == "__main__":
    unittest.main()
