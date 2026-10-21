"""CPU contract tests; small tensors test scale 16 without allocating an 8K GT."""

import copy
import csv
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml
from PIL import Image

import eval_e2e
import infer_sr
import train_e2e
from augmentation import AugmentedPath
from data_modes import checkpoint_data_spec, data_spec, validate_data_config
from engine import evaluate, train_wsi, wsi_loss
from models.baseline import MetaSRABMIL, model_from_checkpoint
from native_data import NATIVE_COLUMNS
from wsi_data import load_fold_datasets, load_images


ROOT = Path(__file__).resolve().parents[1]


class Native512IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def setUp(self):
        torch.manual_seed(219)
        self.config = {
            "data_mode": "native512", "data_manifest": "Manifests/patch_manifest.csv",
            "classification_encoder": "spp",
            "classification_head": "linear", "metasr": {
                "rdn_blocks": 1, "scale": 16, "rgb_range": 1,
                "checkpoint_rdb": True, "lr_chunk_size": 2},
            "precision": "fp32", "lambda_sr": 0.1, "sr_train_crop": 32,
            "training": {"use_region_microbatch": False, "region_microbatch_size": 1},
        }
        self.lr = torch.rand(3, 3, 16, 16)
        self.truth = torch.rand(3, 3, 32, 32)
        self.sample = {"data_mode": "native512", "slide_id": "synthetic", "label": 1,
                       "n_regions": 3, "lr_paths": [0, 1, 2], "hr_paths": [0, 1, 2]}
        self.coordinates = torch.tensor([[0, 0], [17, 31], [224, 224]])

    def model(self, config=None):
        config = self.config if config is None else config
        return MetaSRABMIL(classification_encoder=config["classification_encoder"],
                           classification_head=config.get("classification_head", "linear"),
                           **config["metasr"])

    def load_lr(self, paths, size):
        self.assertEqual(size, 512)
        return self.lr[paths]

    def saved(self, model, config=None):
        config = self.config if config is None else config
        return {"format": "metasr-abmil-v2", "architecture": model.architecture(),
                "config": copy.deepcopy(config), "data_spec": data_spec(config).metadata(),
                "model_state": model.state_dict(), "fold": 1,
                "data_provenance": {name: {"sha256": name} for name in
                                    ("manifest", "labels", "split", "verification")}}

    def test_all_old_yamls_retain_legacy_default_and_new_configs_are_isolated(self):
        for path in (ROOT / "configs").glob("*.yaml"):
            original = yaml.safe_load(path.read_text(encoding="utf-8"))
            if "metasr" not in original or original.get("data_mode") == "native512":
                continue
            with self.subTest(config=path.name):
                config = train_e2e.load_config(path)
                self.assertEqual(data_spec(config).metadata(), {
                    "mode": "legacy256", "lr_size": 256, "hr_size": 8192, "scale": 32})
                self.assertNotIn("data_mode", config)
        first = train_e2e.load_config(ROOT / "configs/spp_d1_native512_aug_sr0.yaml")
        second = train_e2e.load_config(ROOT / "configs/spp_d1_1344_native512_aug_sr0.yaml")
        self.assertEqual(first["classification_encoder"], "spp")
        self.assertEqual(second["classification_encoder"], "spp_1344")
        first.pop("classification_encoder")
        second.pop("classification_encoder")
        self.assertEqual(first, second)
        self.assertEqual(data_spec(first).metadata(), {
            "mode": "native512", "lr_size": 512, "hr_size": 8192, "scale": 16})
        for mode, scale, rgb in (("native512", 32, 1), ("legacy256", 16, 1),
                                 ("native512", 16, 255), ("unknown", 16, 1)):
            invalid = copy.deepcopy(self.config)
            invalid["data_mode"] = mode
            invalid["metasr"].update(scale=scale, rgb_range=rgb)
            with self.subTest(mode=mode, scale=scale, rgb=rgb), self.assertRaises(ValueError):
                validate_data_config(invalid)

    def test_all_encoders_preserve_dimensions_at_both_scales(self):
        for scale in (32, 16):
            for encoder, width in (("spp", 3200), ("spp_1344", 1344),
                                   ("spp_ln", 3200), ("spp_avg", 3200), ("resnet18", 512)):
                with self.subTest(scale=scale, encoder=encoder), torch.no_grad():
                    model = MetaSRABMIL(classification_encoder=encoder, rdn_blocks=1,
                                       scale=scale, checkpoint_rdb=False).eval()
                    feature, embedding = model.encode_regions(self.lr[:2])
                    self.assertEqual(tuple(feature.shape), (2, 64, 16, 16))
                    self.assertEqual(tuple(embedding.shape), (2, width))
                    self.assertEqual(tuple(model.forward_embeddings(embedding).shape), (1, 2))

    def test_native_sr_off_never_reads_hr_decodes_or_samples(self):
        self.config["lambda_sr"] = 0
        model = self.model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        with patch("engine.load_images", side_effect=self.load_lr), \
             patch("engine.load_hr_crops", side_effect=AssertionError("SR OFF reads HR")), \
             patch("engine.torch.randint", side_effect=AssertionError("SR OFF samples crop")), \
             patch.object(model.sr, "decode_crop", side_effect=AssertionError("SR OFF decoder")), \
             patch.object(optimizer, "step", wraps=optimizer.step) as step, \
             patch.object(model, "forward_embeddings", wraps=model.forward_embeddings) as mil:
            result = train_wsi(model, self.sample, optimizer, torch.device("cpu"), self.config)
            self.assertEqual(step.call_count, 1)
            self.assertEqual(mil.call_count, 1)
            self.assertEqual(tuple(mil.call_args.args[0].shape), (3, 3200))
            measured = evaluate(model, [self.sample, dict(self.sample, label=0)],
                                 torch.device("cpu"), self.config)
        self.assertEqual(result["train_loss_sr"], 0)
        self.assertEqual(len(measured["predictions"]), 2)
        self.assertTrue(all(p.grad is None for p in model.sr.P2W.parameters()))
        for parameter in (model.sr.SFENet1.weight, model.mil_head.attention_V.weight,
                          model.classifier.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.norm().item(), 0)

    def test_native_sr_full_micro1_micro2_preserve_one_bag_loss_and_gradients(self):
        model = self.model().train()
        reference = None
        for micro, size in ((False, 1), (True, 1), (True, 2)):
            self.config["training"].update(use_region_microbatch=micro, region_microbatch_size=size)
            model.zero_grad(set_to_none=True)
            with patch("engine.load_images", side_effect=self.load_lr) as load, \
                 patch("engine.load_hr_crops", side_effect=lambda paths, box: self.truth[paths]) as hr, \
                 patch("engine.torch.randint", return_value=self.coordinates) as random, \
                 patch.object(model, "forward_embeddings", wraps=model.forward_embeddings) as mil:
                result = wsi_loss(model, self.sample, torch.device("cpu"), self.config)
            self.assertEqual(mil.call_count, 1)
            self.assertEqual(tuple(mil.call_args.args[0].shape), (3, 3200))
            self.assertEqual(load.call_count, (3 + size - 1) // size if micro else 1)
            random.assert_called_once_with(8192 - 32 + 1, (3, 2))
            self.assertEqual([call.args for call in hr.call_args_list],
                             [([i], (y, x, 32, 32)) for i, (y, x) in enumerate(self.coordinates.tolist())])
            torch.testing.assert_close(result[0], result[1] + 0.1 * result[2])
            self.assertGreater(result[2].item(), 0)
            result[0].backward()
            gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()
                         if p.grad is not None}
            for name in ("sr.SFENet1.weight", "sr.RDBs.0.LFF.weight", "sr.P2W.meta_block.2.weight",
                         "mil_head.attention_V.weight", "classifier.weight"):
                self.assertTrue(torch.isfinite(gradients[name]).all())
                self.assertGreater(gradients[name].norm().item(), 0)
            if reference is None:
                reference = [x.detach().clone() for x in result], gradients
            else:
                for actual, expected in zip(result, reference[0]):
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
                self.assertEqual(gradients.keys(), reference[1].keys())
                for name, actual in gradients.items():
                    torch.testing.assert_close(actual, reference[1][name], atol=2e-5, rtol=2e-4, msg=name)

    def test_data_mode_and_model_mismatches_fail_before_image_io(self):
        model = self.model()
        with patch("engine.load_images", side_effect=AssertionError("must reject before image IO")):
            for sample, config in ((dict(self.sample, data_mode="legacy256"), self.config),
                                   (self.sample, dict(self.config, data_mode="legacy256"))):
                with self.subTest(sample=sample["data_mode"], config=config["data_mode"]):
                    with self.assertRaisesRegex(ValueError, "data mode"):
                        wsi_loss(model, sample, torch.device("cpu"), config)
                    with self.assertRaisesRegex(ValueError, "data mode"):
                        evaluate(model, [sample], torch.device("cpu"), config)
            model.sr.scale = 32
            with self.assertRaisesRegex(ValueError, "scale=16"):
                wsi_loss(model, self.sample, torch.device("cpu"), self.config)

    def test_native_checkpoint_roundtrip_and_legacy_model_only_formats(self):
        model = self.model().eval()
        saved = self.saved(model)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "native.pth"
            torch.save(saved, path)
            restored = model_from_checkpoint(torch.load(path, weights_only=False)).eval()
        self.assertEqual(restored.sr.scale, 16)
        with torch.no_grad():
            torch.testing.assert_close(restored(self.lr), model(self.lr), atol=0, rtol=0)
        for mutate in (lambda x: x.pop("data_spec"),
                       lambda x: x["data_spec"].update(lr_size=256),
                       lambda x: x["config"]["metasr"].update(scale=32),
                       lambda x: x["config"].pop("data_mode")):
            invalid = copy.deepcopy(saved)
            mutate(invalid)
            with self.assertRaises(ValueError):
                model_from_checkpoint(invalid)
        for encoder, version, scale in (("spp", "metasr-abmil-v2", 32),
                                         ("spp", "metasr-abmil-v2", 2),
                                         ("resnet18", "metasr-abmil-v1", 32)):
            config = {"classification_encoder": encoder, "metasr": {"scale": scale, "rdn_blocks": 1}}
            legacy = self.model(config).eval()
            old = {"config": config, "format": version, "model_state": legacy.state_dict(),
                   "architecture": legacy.architecture()}
            self.assertEqual(checkpoint_data_spec(old).mode, "legacy256")
            restored = model_from_checkpoint(old).eval()
            self.assertEqual(restored.sr.scale, scale)
            for name, value in legacy.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)

    def test_eval_entry_uses_native_mode_and_rejects_every_source_hash_change(self):
        saved = self.saved(self.model())
        samples = [self.sample, dict(self.sample, label=0)]
        for changed in (None, "manifest", "labels", "split", "verification"):
            current = copy.deepcopy(saved["data_provenance"])
            if changed:
                current[changed]["sha256"] = "wrong-data-version"
            with self.subTest(changed=changed), \
                 patch("sys.argv", ["eval_e2e.py", "--checkpoint", "unused.pth",
                                    "--data-root", "unused", "--device", "cpu"]), \
                 patch("eval_e2e.torch.load", return_value=saved), \
                 patch("eval_e2e.load_fold_datasets", return_value=([], samples)) as loader, \
                 patch("eval_e2e.provenance", return_value=current), \
                 patch("engine.load_images", side_effect=self.load_lr) as images, \
                 redirect_stdout(io.StringIO()):
                if changed:
                    with self.assertRaisesRegex(ValueError, "differ"):
                        eval_e2e.main()
                    images.assert_not_called()
                else:
                    eval_e2e.main()
                    self.assertEqual(images.call_count, 2)
                self.assertEqual(loader.call_args.kwargs, {
                    "require_hr": False, "data_mode": "native512",
                    "manifest_csv": "Manifests/patch_manifest.csv"})
                self.assertEqual(loader.call_args.args[2:], (None, None))

    def test_native_provenance_hashes_selected_manifest_and_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = {"manifest": root / "Manifests/patch_manifest.csv",
                     "labels": root / "downstream_train/labels.csv",
                     "split": root / "downstream_train/splits_1.csv",
                     "verification": root / "verification/report.json"}
            for key, path in files.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(key.encode("ascii"))
            dataset = SimpleNamespace(manifest_csv=files["manifest"], labels_csv=files["labels"],
                                      split_csv=files["split"], verification_json=files["verification"])
            actual = train_e2e.provenance(root, dataset)
            self.assertEqual(set(actual), set(files))
            for key, path in files.items():
                self.assertEqual(actual[key], {"path": str(path.resolve()),
                                 "sha256": hashlib.sha256(key.encode("ascii")).hexdigest()})

    def test_infer_entry_requests_exact_mode_size_without_full_8k_allocation(self):
        for native in (False, True):
            config = copy.deepcopy(self.config)
            if not native:
                config.pop("data_mode")
                config.pop("data_manifest")
                config["metasr"]["scale"] = 32
            saved = self.saved(self.model(config), config)
            if not native:
                saved.pop("data_spec")
            with tempfile.TemporaryDirectory() as temp, \
                 patch("sys.argv", ["infer_sr.py", "--checkpoint", "unused.pth", "--lr-image", "lr.jpg",
                                    "--output", str(Path(temp) / "full.npy"), "--device", "cpu"]), \
                 patch("infer_sr.torch.load", return_value=saved), \
                 patch("infer_sr.load_images", return_value=self.lr[:1]) as images, \
                 patch("infer_sr.reconstruct") as reconstruct, redirect_stdout(io.StringIO()) as output:
                infer_sr.main()
                self.assertEqual(images.call_args.args[1], 512 if native else 256)
                self.assertEqual(reconstruct.call_args.args[0].sr.scale, 16 if native else 32)
                self.assertIn("8192, 8192", output.getvalue())

    def test_training_entry_saves_data_spec_in_best_and_last(self):
        config = dict(self.config, seed=1, lr=1e-4, weight_decay=1e-4, epochs=1,
                      early_stopping_patience=15)
        train = SimpleNamespace(samples=[self.sample])
        train.evaluation_view = lambda: train
        val = SimpleNamespace(samples=[self.sample, dict(self.sample, label=0)])
        model = self.model()
        measured = {"loss": 0.1, "auc": 0.6, "acc": 0.5, "bacc": 0.5,
                    "predictions": [{"slide_id": "synthetic", "label": 1, "p_tumor": 0.6}]}
        online = {"train_loss_cls": 0.1, "train_loss_sr": 0.2, "train_loss_total": 0.12,
                  "max_memory_allocated": None, "max_memory_reserved": None}
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            replacements = {
                "sys.argv": ["train_e2e.py", "--data-root", str(root / "data"),
                             "--output", str(root / "run"), "--device", "cpu"],
                "train_e2e.load_config": lambda path: config,
                "train_e2e.DataLoader": lambda data, **kwargs: data.samples,
                "train_e2e.MetaSRABMIL": lambda **kwargs: model,
                "train_e2e.provenance": lambda *args: {},
                "train_e2e.run_equivalence": lambda **kwargs: {"selected_config": "synthetic contract test"},
                "train_e2e.summarize": lambda *args: {"experiment_id": "synthetic", "val_auc": 0.6},
            }
            for name, value in replacements.items():
                stack.enter_context(patch(name, value))
            load = stack.enter_context(patch("train_e2e.load_fold_datasets", return_value=(train, val)))
            stack.enter_context(patch("train_e2e.train_wsi", return_value=online))
            stack.enter_context(patch("train_e2e.evaluate", return_value=measured))
            stack.enter_context(redirect_stdout(io.StringIO()))
            train_e2e.main()
            self.assertEqual(load.call_args.kwargs["data_mode"], "native512")
            self.assertEqual(load.call_args.kwargs["manifest_csv"], "Manifests/patch_manifest.csv")
            self.assertEqual(load.call_args.args[2:], (None, None))
            for filename in ("best.pth", "last.pth"):
                saved = torch.load(root / "run" / filename, weights_only=False)
                self.assertEqual(saved["data_spec"], data_spec(config).metadata())
                self.assertEqual(saved["config"]["data_mode"], "native512")
                self.assertEqual(saved["epoch"], 1)
                restored = model_from_checkpoint(saved)
                self.assertEqual(restored.architecture(), model.architecture())


class Native512WSILoaderTests(unittest.TestCase):
    """Actual JPEG/CSV/report integration; HR files are deliberately absent."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.manifest = self.root / "Manifests/patch_manifest.csv"
        self.labels_path = self.root / "downstream_train/camelyon16_labels.csv"
        self.report_path = self.root / "verification/report.json"
        self.labels, self.regions, slides = [], [], []
        geometry = {"level_downsamples": [1., 2., 4., 8., 16., 32.],
                    "level_dimensions": [[131072 // (2 ** level)] * 2 for level in range(6)],
                    "patch_size_level0": 16384, "status": "complete"}
        self.expected = {"a0": ("case_a", 0, 1), "a1": ("case_a", 0, 2),
                         "b0": ("case_b", 1, 3), "c0": ("case_c", 0, 1),
                         "d0": ("case_d", 1, 2)}
        for slide_id, (case_id, label, count) in self.expected.items():
            self.labels.append({"slide_id": slide_id, "case_id": case_id, "label": label})
            slides.append(dict(geometry, slide_id=slide_id, pairs=count))
            for region in range(count):
                name = "opaque_{}_{}.jpg".format(slide_id, region)
                row = {"slide_id": slide_id, "x": 17 + region * 16384, "y": 33,
                       "hr_level": 1, "lr_level": 5, "scale": 16,
                       "hr_path": "images_8192/" + name, "lr_path": "images_512/" + name}
                self.regions.append(row)
                path = self.root / row["lr_path"]
                path.parent.mkdir(exist_ok=True)
                Image.new("RGB", (512, 512), (label * 100, 128, region * 60)).save(path, quality=90)
        self.write_csv(self.manifest, NATIVE_COLUMNS, self.regions)
        self.write_csv(self.labels_path, ("case_id", "slide_id", "label"), self.labels)
        report = {"status": "complete", "hr_level": 1, "lr_level": 5, "scale": 16,
                  "hr_size": 8192, "lr_size": 512, "slides": slides,
                  "sampling": {"method": "tissue_windows_nonoverlap_v2", "overlap_allowed": False},
                  "zero_patch_slides": [], "skipped_slides": []}
        self.report_path.parent.mkdir()
        self.report_path.write_text(json.dumps(report), encoding="utf-8")
        self.write_fold(0, ["case_a", "case_b"], ["case_c", "case_d"])
        self.write_fold(1, ["case_c", "case_d"], ["case_a", "case_b"])

    def write_csv(self, path, fields, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def write_fold(self, fold, train, val):
        rows = [{"train": train[i] if i < len(train) else "",
                 "val": val[i] if i < len(val) else ""} for i in range(max(len(train), len(val)))]
        self.write_csv(self.root / "downstream_train/splits_{}.csv".format(fold),
                       ("train", "val"), rows)

    def load(self, fold=0, **kwargs):
        return load_fold_datasets(self.root, fold, data_mode="native512",
                                  manifest_csv="Manifests/patch_manifest.csv", require_hr=False, **kwargs)

    def test_fold0_fold1_complete_native_bags_labels_paths_and_provenance_without_hr_io(self):
        original_open, original_file, original_resolve = Image.open, Path.is_file, Path.resolve

        def guarded_open(path, *args, **kwargs):
            self.assertNotIn("images_8192", Path(path).parts)
            return original_open(path, *args, **kwargs)

        def guarded_file(path):
            self.assertNotIn("images_8192", path.parts)
            return original_file(path)

        def guarded_resolve(path, *args, **kwargs):
            self.assertNotIn("images_8192", path.parts)
            return original_resolve(path, *args, **kwargs)

        with patch("native_data.Image.open", side_effect=guarded_open), \
             patch.object(Path, "is_file", guarded_file), patch.object(Path, "resolve", guarded_resolve):
            for fold, train_cases in ((0, {"case_a", "case_b"}), (1, {"case_c", "case_d"})):
                with self.subTest(fold=fold):
                    train, val = self.load(fold)
                    self.assertEqual({s["case_id"] for s in train.samples}, train_cases)
                    self.assertFalse({s["case_id"] for s in val.samples} & train_cases)
                    self.assertEqual({s["slide_id"] for s in train.samples + val.samples}, set(self.expected))
                    self.assertEqual(len(train) + len(val), len(self.expected))
                    self.assertEqual({s["label"] for s in val.samples}, {0, 1})
                    for dataset in (train, val):
                        self.assertEqual(dataset.manifest_csv, self.manifest)
                        self.assertEqual(dataset.verification_json, self.report_path)
                        info = train_e2e.provenance(self.root, dataset)
                        self.assertEqual(set(info), {"manifest", "labels", "split", "verification"})
                        self.assertEqual(info["manifest"]["sha256"],
                                         hashlib.sha256(self.manifest.read_bytes()).hexdigest())
                        for sample in dataset.samples:
                            case_id, label, count = self.expected[sample["slide_id"]]
                            self.assertEqual((sample["case_id"], sample["label"], sample["n_regions"]),
                                             (case_id, label, count))
                            self.assertEqual(sample["data_mode"], "native512")
                            expected = [r for r in self.regions if r["slide_id"] == sample["slide_id"]]
                            self.assertEqual(sample["lr_paths"], [self.root / r["lr_path"] for r in expected])
                            self.assertEqual(sample["hr_paths"], [self.root / r["hr_path"] for r in expected])
                            images = load_images(sample["lr_paths"], 512)
                            self.assertEqual(tuple(images.shape), (count, 3, 512, 512))
                            self.assertEqual(images.dtype, torch.float32)
                            self.assertTrue(0 <= images.min().item() <= images.max().item() <= 1)

    def test_only_training_wraps_paired_paths_and_evaluation_never_draws_geometry(self):
        train, val = self.load(augmentation={"enabled": True, "horizontal_flip": 0.5,
                                             "vertical_flip": 0.5, "rotation90": True})
        self.assertTrue(train.augmentation.enabled)
        self.assertFalse(val.augmentation.enabled)
        for index in range(len(train)):
            sample = train[index]
            self.assertEqual(len(sample["lr_paths"]), sample["n_regions"])
            for lr, hr in zip(sample["lr_paths"], sample["hr_paths"]):
                self.assertIsInstance(lr, AugmentedPath)
                self.assertIsInstance(hr, AugmentedPath)
                self.assertIs(lr.params, hr.params)
            self.assertEqual(tuple(load_images(sample["lr_paths"], 512).shape),
                             (sample["n_regions"], 3, 512, 512))
        clean = train.evaluation_view()
        self.assertFalse(clean.augmentation.enabled)
        self.assertIs(clean.samples, train.samples)
        self.assertEqual(clean.manifest_csv, train.manifest_csv)
        self.assertEqual(clean.verification_json, train.verification_json)
        with patch("augmentation.random.random", side_effect=AssertionError("eval geometry RNG")), \
             patch("augmentation.random.randrange", side_effect=AssertionError("eval rotation RNG")):
            for dataset in (clean, val):
                for index in range(len(dataset)):
                    self.assertIs(dataset[index], dataset.samples[index])
                    self.assertTrue(all(isinstance(p, Path) for p in dataset[index]["lr_paths"]))

    def test_native_missing_labels_regions_and_invalid_case_folds_are_rejected(self):
        self.write_csv(self.labels_path, ("case_id", "slide_id", "label"), self.labels[1:])
        with self.assertRaisesRegex(ValueError, "missing real labels"):
            self.load()
        self.write_csv(self.labels_path, ("case_id", "slide_id", "label"),
                       self.labels + [{"case_id": "case_a", "slide_id": "absent", "label": 0}])
        with self.assertRaisesRegex(ValueError, "missing manifest regions"):
            self.load()
        self.write_csv(self.labels_path, ("case_id", "slide_id", "label"), self.labels)
        self.write_fold(0, ["case_a", "case_b", "case_c"], ["case_c", "case_d"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.load()
        self.write_fold(0, ["case_a"], ["case_c", "case_d"])
        with self.assertRaisesRegex(ValueError, "omits labelled cases"):
            self.load()
        self.write_fold(0, ["case_a", "case_b"], ["case_c", "case_d", "unknown"])
        with self.assertRaisesRegex(ValueError, "unknown cases"):
            self.load()


if __name__ == "__main__":
    unittest.main()
