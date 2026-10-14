"""Small CPU fixtures exercise real paired loading, decoder and WSI gradients."""

import io
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import eval_e2e
import infer_sr
import train_e2e
from augmentation import AugmentationConfig, AugmentedPath, TransformParams
from engine import evaluate, load_hr_crops, wsi_loss
from models.baseline import MetaSRABMIL, model_from_checkpoint
from train_e2e import load_config, provenance
from wsi_data import Camelyon16WSI, load_fold_datasets, load_images


class PairedAugmentationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(123)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sample = {"slide_id": "slide0", "case_id": "case0", "label": 1,
                       "n_regions": 3, "lr_paths": [], "hr_paths": []}
        y, x = np.indices((8, 8))
        for region in range(3):
            pixels = np.stack((x * 29, y * 31, (x + y + region * 3) * 11), axis=-1).astype(np.uint8)
            for directory, array in (("images_256", pixels),
                                     ("images_8192", pixels.repeat(32, 0).repeat(32, 1))):
                path = self.root / directory / f"region{region}.png"
                path.parent.mkdir(exist_ok=True)
                Image.fromarray(array).save(path)
                self.sample["lr_paths" if directory == "images_256" else "hr_paths"].append(path)
        self.config = {"classification_encoder": "spp", "metasr": {"rdn_blocks": 1, "scale": 32},
                       "training": {"use_region_microbatch": False, "region_microbatch_size": 1},
                       "precision": "fp32", "lambda_sr": 0.1, "sr_train_crop": 8}

    def model(self):
        return MetaSRABMIL(classification_encoder="spp", rdn_blocks=1, scale=32).train()

    def loss(self, model, sample):
        # Keep production scale=32 and all real PIL/model operations, using 8 -> 256
        # fixtures instead of 256 -> 8192; sample origin, unaligned and far edge crops.
        with patch("engine.load_images", side_effect=lambda paths, size: load_images(paths, 8)), \
             patch("engine.load_hr_crops", side_effect=lambda paths, box: load_hr_crops(paths, box, 256)), \
             patch("engine.torch.randint", return_value=torch.tensor([[0, 0], [7, 13], [248, 248]])), \
             patch("wsi_data.generate_transform_params", side_effect=AssertionError("Resampled during batching")):
            return wsi_loss(model, sample, torch.device("cpu"), self.config)

    def folds(self, require_hr=True):
        records = {
            "manifests/patch_manifest.csv": ("filename,slide_id", "region0.png,slide0\nregion1.png,slide1\nregion2.png,slide2"),
            "downstream_train/camelyon16_labels.csv": ("case_id,slide_id,label", "case0,slide0,1\ncase1,slide1,0\ncase2,slide2,1"),
            "downstream_train/splits_0.csv": ("train,val", "case0,case1\n,case2"),
        }
        for name, (header, rows) in records.items():
            path = self.root / name
            path.parent.mkdir(exist_ok=True)
            path.write_text(header + "\n" + rows + "\n", encoding="utf-8")
        return load_fold_datasets(self.root, 0, augmentation={"enabled": True}, require_hr=require_hr)

    def test_yaml_default_and_enabled_validation(self):
        baseline = Path(__file__).resolve().parents[1] / "configs/baseline.yaml"
        original = load_config(baseline)
        self.assertFalse(AugmentationConfig(**original.get("augmentation", {})).enabled)
        selected = dict(original, augmentation={"enabled": True, "horizontal_flip": 1.0})
        with patch("train_e2e.yaml.safe_load", return_value=selected):
            self.assertTrue(load_config(baseline)["augmentation"]["enabled"])
        for invalid in ({"enabled": "false"}, {"horizontal_flip": -0.1}, {"vertical_flip": 1.1},
                        {"rotation90": 1}):
            with self.subTest(invalid=invalid), \
                 patch("train_e2e.yaml.safe_load", return_value=dict(original, augmentation=invalid)), \
                 self.assertRaises(ValueError):
                load_config(baseline)

    def test_training_entry_forwards_yaml_or_standalone_augmentation(self):
        baseline = Path(__file__).resolve().parents[1] / "configs/baseline.yaml"
        override = self.root / "augmentation.yaml"
        override.write_text("augmentation:\n  enabled: false\n", encoding="utf-8")
        for index, (enabled, standalone) in enumerate(((None, False), (True, False), (True, True))):
            config = load_config(baseline)
            if enabled is not None:
                config["augmentation"] = {"enabled": enabled}
            argv = ["train_e2e.py", "--data-root", str(self.root), "--output", str(self.root / f"run{index}"),
                    "--device", "cpu"]
            if standalone:
                argv += ["--augmentation-config", str(override)]
            with self.subTest(enabled=enabled, standalone=standalone), \
                 patch("sys.argv", argv), patch("train_e2e.load_config", return_value=config), \
                 patch("train_e2e.run_equivalence", return_value={"selected_config": "synthetic fixture"}), \
                 patch("train_e2e.load_fold_datasets", side_effect=RuntimeError("dataset reached")) as loader, \
                 patch("sys.stdout", new_callable=io.StringIO), \
                 self.assertRaisesRegex(RuntimeError, "dataset reached"):
                train_e2e.main()
            selected = loader.call_args.kwargs.get("augmentation")
            self.assertEqual(selected.get("enabled") if selected else None, False if standalone else enabled)

    def test_sampling_is_lazy_paired_and_training_only(self):
        train, val = self.folds()
        with patch("wsi_data.Image.open", side_effect=AssertionError("Eager image decoding")):
            sample = train[0]
            self.assertIsInstance(sample["lr_paths"][0], AugmentedPath)
            self.assertIs(sample["lr_paths"][0].params, sample["hr_paths"][0].params)
            for dataset in (train.evaluation_view(), val):
                before = random.getstate()
                plain = dataset[0]
                self.assertEqual(random.getstate(), before)
                self.assertIsInstance(plain["lr_paths"][0], Path)
                self.assertIsInstance(plain["hr_paths"][0], Path)
        self.assertIsInstance(train.samples[0]["lr_paths"][0], Path)

    def test_lr_and_hr_loaders_share_augmented_coordinates(self):
        params = TransformParams(True, False, 1)
        lr = load_images([AugmentedPath(self.sample["lr_paths"][0], params)], 8)
        expanded = lr.repeat_interleave(32, -2).repeat_interleave(32, -1)
        for box in ((0, 0, 8, 8), (7, 13, 19, 37), (248, 249, 8, 7)):
            with self.subTest(box=box):
                y, x, h, w = box
                truth = load_hr_crops([AugmentedPath(self.sample["hr_paths"][0], params)], box, 256)
                self.assertEqual(tuple(truth.shape), (1, 3, h, w))
                torch.testing.assert_close(truth, expanded[:, :, y:y+h, x:x+w], atol=0, rtol=0)

    def test_lr_augmented_path_with_legacy_pillow_path_detection(self):
        path = self.sample["lr_paths"][0]
        plain = load_images([path], 8)
        params = TransformParams(True, True, 1)
        expected = torch.rot90(torch.flip(plain, (-2, -1)), 1, (-2, -1))
        # Pillow 10 recognizes pathlib.Path, but not arbitrary os.PathLike objects.
        with patch("PIL.Image.is_path", side_effect=lambda f: isinstance(f, (bytes, str, Path))):
            for ordinary in (path, str(path)):
                torch.testing.assert_close(load_images([ordinary], 8), plain, atol=0, rtol=0)
            actual = load_images([AugmentedPath(path, params)], 8)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertFalse(torch.equal(actual, plain))

    def test_hr_augmented_path_with_legacy_pillow_path_detection(self):
        path = self.sample["hr_paths"][0]
        plain = load_images([path], 256)
        params = TransformParams(True, True, 1)
        transformed = torch.rot90(torch.flip(plain, (-2, -1)), 1, (-2, -1))
        with patch("PIL.Image.is_path", side_effect=lambda f: isinstance(f, (bytes, str, Path))):
            for box in ((0, 0, 8, 8), (7, 13, 19, 37), (248, 249, 8, 7)):
                y, x, h, w = box
                for ordinary in (path, str(path)):
                    torch.testing.assert_close(load_hr_crops([ordinary], box, 256),
                                               plain[:, :, y:y+h, x:x+w], atol=0, rtol=0)
                actual = load_hr_crops([AugmentedPath(path, params)], box, 256)
                torch.testing.assert_close(actual, transformed[:, :, y:y+h, x:x+w], atol=0, rtol=0)

    def test_joint_gradients_and_geometry_independent_of_microbatch(self):
        dataset = Camelyon16WSI([self.sample], augmentation={"enabled": True})
        model = self.model()
        for coefficient in (0, 0.1):
            self.config["lambda_sr"] = coefficient
            reference = None
            for micro, size in ((False, 1), (True, 1), (True, 2)):
                with self.subTest(lambda_sr=coefficient, micro=micro, size=size):
                    random.seed(72)
                    sample = dataset[0]
                    rng = random.getstate()
                    model.zero_grad(set_to_none=True)
                    self.config["training"] = {"use_region_microbatch": micro, "region_microbatch_size": size}
                    result = self.loss(model, sample)
                    self.assertEqual(random.getstate(), rng)
                    result[0].backward()
                    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
                    self.assertTrue(all(torch.isfinite(value).all() for value in gradients.values()))
                    for name in ("sr.SFENet1.weight", "sr.RDBs.0.LFF.weight", "mil_head.attention_V.weight", "classifier.weight"):
                        self.assertTrue(torch.isfinite(gradients[name]).all())
                        self.assertGreater(gradients[name].abs().sum().item(), 0)
                    if coefficient:
                        self.assertGreater(result[2].item(), 0)
                        self.assertGreater(gradients["sr.P2W.meta_block.2.weight"].abs().sum().item(), 0)
                    else:
                        self.assertEqual(result[2].item(), 0)
                        self.assertTrue(all(p.grad is None for p in model.sr.P2W.parameters()))
                    if reference is None:
                        reference = (sample, [value.detach().clone() for value in result], gradients)
                    else:
                        self.assertEqual(sample, reference[0])
                        for actual, expected in zip(result, reference[1]):
                            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
                        self.assertEqual(gradients.keys(), reference[2].keys())
                        for name, value in gradients.items():
                            torch.testing.assert_close(value, reference[2][name], atol=2e-5, rtol=2e-4)

    def test_disabled_preserves_plain_paths_rng_losses_and_gradients(self):
        model = self.model()
        for coefficient in (0, 0.1):
            self.config["lambda_sr"] = coefficient
            reference = None
            for augmentation in (None, {"enabled": False, "horizontal_flip": 1, "vertical_flip": 1}):
                with self.subTest(lambda_sr=coefficient, augmentation=augmentation):
                    dataset = Camelyon16WSI([self.sample], augmentation=augmentation)
                    rng = random.getstate()
                    sample = dataset[0]
                    self.assertIs(sample, self.sample)
                    self.assertEqual(random.getstate(), rng)
                    model.zero_grad(set_to_none=True)
                    result = self.loss(model, sample)
                    result[0].backward()
                    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
                    if reference is None:
                        reference = ([value.detach().clone() for value in result], gradients)
                    else:
                        for actual, expected in zip(result, reference[0]):
                            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                        for name, value in gradients.items():
                            torch.testing.assert_close(value, reference[1][name], atol=0, rtol=0)

    def test_sr_off_never_reads_hr_or_samples_crop(self):
        for path in self.sample["hr_paths"]:
            path.unlink()
        train, _ = self.folds(require_hr=False)
        model = self.model()
        self.config["lambda_sr"] = 0
        with patch("engine.load_images", side_effect=lambda paths, size: load_images(paths, 8)), \
             patch("engine.load_hr_crops", side_effect=AssertionError("Read HR with SR off")), \
             patch("engine.torch.randint", side_effect=AssertionError("Sampled SR crop with SR off")), \
             patch.object(model.sr, "decode_crop", side_effect=AssertionError("Decoded SR with SR off")):
            result = wsi_loss(model, train[0], torch.device("cpu"), self.config)
            result[0].backward()
        self.assertEqual(result[2].item(), 0)

    def test_offline_train_val_and_standalone_eval_do_not_augment(self):
        train, val = self.folds()
        model = self.model()
        self.config["augmentation"] = {"enabled": True}
        state = {name: value.clone() for name, value in model.state_dict().items()}

        def plain_images(paths, size):
            self.assertTrue(all(isinstance(path, Path) for path in paths))
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(model.training)
            return load_images(paths, 8)

        with patch("engine.load_images", side_effect=plain_images), \
             patch("engine.load_hr_crops", side_effect=AssertionError("Evaluation read HR")), \
             patch("wsi_data.generate_transform_params", side_effect=AssertionError("Evaluation sampled geometry")):
            train_samples = list(train.evaluation_view())
            evaluate(model, train_samples + [dict(train_samples[0], label=0)], torch.device("cpu"), self.config)
            evaluate(model, val, torch.device("cpu"), self.config)
            saved = {"config": self.config, "fold": 0, "data_provenance": provenance(self.root, val)}
            with patch("sys.argv", ["eval_e2e.py", "--checkpoint", "unused.pth", "--data-root", str(self.root),
                                    "--labels-csv", str(val.labels_csv), "--split-dir", str(val.split_csv.parent),
                                    "--device", "cpu"]), \
                 patch("eval_e2e.torch.load", return_value=saved), \
                 patch("eval_e2e.model_from_checkpoint", return_value=model), \
                 patch("sys.stdout", new_callable=io.StringIO):
                eval_e2e.main()
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], atol=0, rtol=0)

    def test_sr_inference_passes_plain_image_with_augmented_training_config(self):
        path = self.sample["lr_paths"][0]
        model = self.model()
        self.config["augmentation"] = {"enabled": True}
        with patch("sys.argv", ["infer_sr.py", "--checkpoint", "unused.pth", "--lr-image", str(path),
                                "--output", str(self.root / "sr.npy"), "--device", "cpu"]), \
             patch("infer_sr.torch.load", return_value={"config": self.config}), \
             patch("infer_sr.model_from_checkpoint", return_value=model), \
             patch("infer_sr.load_images", side_effect=lambda paths, size: load_images(paths, 8)) as loader, \
             patch("infer_sr.reconstruct") as reconstruct, \
             patch("wsi_data.generate_transform_params", side_effect=AssertionError("Inference sampled geometry")), \
             patch("sys.stdout", new_callable=io.StringIO):
            infer_sr.main()
        loader.assert_called_once_with([path], 256)
        self.assertFalse(model.training)
        torch.testing.assert_close(reconstruct.call_args.args[1], load_images([path], 8), atol=0, rtol=0)

    def test_checkpoint_config_does_not_change_model_or_inference(self):
        model = self.model().eval()
        images = load_images(self.sample["lr_paths"], 8)
        with torch.no_grad():
            expected = model(images)
            for enabled in (None, False, True):
                config = dict(self.config)
                if enabled is not None:
                    config["augmentation"] = {"enabled": enabled}
                saved = {"format": "metasr-abmil-v2", "architecture": model.architecture(),
                         "config": config, "model_state": model.state_dict()}
                with self.subTest(enabled=enabled):
                    restored = model_from_checkpoint(saved).eval()
                    torch.testing.assert_close(restored(images), expected, atol=0, rtol=0)
                    for name, value in restored.state_dict().items():
                        torch.testing.assert_close(value, saved["model_state"][name], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
