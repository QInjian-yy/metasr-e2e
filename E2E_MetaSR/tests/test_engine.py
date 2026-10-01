import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from engine import evaluate, load_hr_crops, train_wsi, wsi_loss
from infer_sr import reconstruct
from models.baseline import MetaSRABMIL


class EngineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(17)
        self.model = MetaSRABMIL().eval()
        self.lr = torch.rand(3, 3, 16, 16)
        self.truth = torch.rand(3, 3, 32, 32)
        self.sample = {"slide_id": "synthetic", "n_regions": 3, "label": 1,
                       "lr_paths": [0, 1, 2], "hr_paths": [0, 1, 2]}
        self.config = {"training": {"use_region_microbatch": True, "region_microbatch_size": 2},
                       "lambda_sr": 0.1,
                       "sr_train_crop": 32, "precision": "fp32"}

    def test_shared_feature_once_and_region_weighted_loss(self):
        expected = self.model.joint_loss(self.lr, torch.tensor([1]), lambda_sr=0.1,
            hr_crop=self.truth, box=(17, 31, 32, 32))
        with patch("engine.load_images", side_effect=lambda paths, size: self.lr[paths]), \
             patch("engine.load_hr_crops", side_effect=lambda paths, box: self.truth[paths]), \
             patch("engine.torch.randint", return_value=torch.tensor([[17, 31]] * 3)), \
             patch.object(self.model.sr, "extract_features", wraps=self.model.sr.extract_features) as encode:
            actual = wsi_loss(self.model, self.sample, torch.device("cpu"), self.config)
        self.assertEqual(encode.call_count, 2)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)

    def test_full_region_batch_is_one_attached_encoder_forward(self):
        self.config["training"] = {"use_region_microbatch": False, "region_microbatch_size": 1}
        calls = []
        original = self.model.encode_regions

        def capture(lr):
            calls.append(tuple(lr.shape))
            result = original(lr)
            self.assertIsNotNone(result[1].grad_fn)
            return result

        with patch("engine.load_images", side_effect=lambda paths, size: self.lr[paths]), \
             patch("engine.load_hr_crops", side_effect=lambda paths, box: self.truth[paths]), \
             patch("engine.torch.randint", return_value=torch.tensor([[17, 31]] * 3)), \
             patch.object(self.model, "encode_regions", side_effect=capture):
            loss, _, _, _ = wsi_loss(self.model, self.sample, torch.device("cpu"), self.config)
        self.assertEqual(calls, [(3, 3, 16, 16)])
        loss.backward()

    def test_random_crops_refresh_and_ignore_region_batching(self):
        sampled = []
        for run, (micro, size) in enumerate(((False, 1), (False, 1), (True, 1), (True, 2))):
            if run != 1:
                torch.manual_seed(123)
            self.config["training"] = {"use_region_microbatch": micro, "region_microbatch_size": size}
            with torch.no_grad(), \
                 patch("engine.load_images", side_effect=lambda paths, size: self.lr[paths]), \
                 patch("engine.load_hr_crops", side_effect=lambda paths, box: self.truth[paths]) as gt, \
                 patch.object(self.model.sr, "decode_crop", side_effect=lambda f, box: self.truth[:len(f)]):
                wsi_loss(self.model, self.sample, torch.device("cpu"), self.config)
            sampled.append([(path, call.args[1]) for call in gt.call_args_list for path in call.args[0]])
        self.assertEqual([path for path, _ in sampled[0]], [0, 1, 2])
        self.assertEqual(len({box for _, box in sampled[0]}), 3)
        for (_, first), (_, second) in zip(sampled[0], sampled[1]):
            self.assertNotEqual(first, second)
        self.assertEqual(sampled[0], sampled[2])
        self.assertEqual(sampled[0], sampled[3])
        for run in sampled:
            for _, (y, x, h, w) in run:
                self.assertEqual((h, w), (32, 32))
                self.assertTrue(0 <= y <= 8192-h and 0 <= x <= 8192-w)

    def test_distinct_crops_align_gt_and_preserve_joint_gradients_across_batches(self):
        coordinates = torch.tensor([[0, 0], [17, 31], [480, 480]])
        expected_boxes = [(y, x, 32, 32) for y, x in coordinates.tolist()]
        with torch.no_grad():
            features, _ = self.model.encode_regions(self.lr)
            expected_sr = torch.stack([torch.nn.functional.l1_loss(
                self.model.sr.decode_crop(features[i:i+1], box).float(), self.truth[i:i+1])
                for i, box in enumerate(expected_boxes)]).mean()
        full_result, full_gradients = None, None
        tracked = {name: parameter for name, parameter in self.model.named_parameters()
                   if name in ("sr.SFENet1.weight", "sr.RDBs.15.LFF.weight", "sr.P2W.meta_block.2.weight",
                               "region_encoder.conv1.weight", "mil_head.attention_V.weight", "classifier.weight")}
        for micro, size in ((False, 1), (True, 2)):
            self.model.zero_grad(set_to_none=True)
            self.config["training"] = {"use_region_microbatch": micro, "region_microbatch_size": size}
            with patch("engine.load_images", side_effect=lambda paths, size: self.lr[paths]), \
                 patch("engine.load_hr_crops", side_effect=lambda paths, box: self.truth[paths]) as gt, \
                 patch("engine.torch.randint", return_value=coordinates), \
                 patch.object(self.model.sr, "decode_crop", wraps=self.model.sr.decode_crop) as decode:
                result = wsi_loss(self.model, self.sample, torch.device("cpu"), self.config)
            self.assertEqual([call.args for call in gt.call_args_list],
                             [([i], box) for i, box in enumerate(expected_boxes)])
            self.assertEqual([call.args[1] for call in decode.call_args_list], expected_boxes)
            for i, call in enumerate(decode.call_args_list):
                torch.testing.assert_close(call.args[0], features[i:i+1], atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(result[2], expected_sr, atol=2e-5, rtol=2e-4)
            result[0].backward()
            for parameter in tracked.values():
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.norm().item(), 0)
            if not micro:
                full_result = [value.detach().clone() for value in result]
                full_gradients = {name: p.grad.clone() for name, p in tracked.items()}
            else:
                for actual, expected in zip(result, full_result):
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
                for name, parameter in tracked.items():
                    torch.testing.assert_close(parameter.grad, full_gradients[name], atol=2e-5, rtol=2e-4)

    def test_training_update_and_evaluation_sr_off(self):
        self.config["lambda_sr"] = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        before = self.model.classifier.weight.detach().clone()
        with patch("engine.load_images", side_effect=lambda paths, size: self.lr[paths]), \
             patch("engine.load_hr_crops", side_effect=AssertionError("SR OFF must not load HR")), \
             patch("engine.torch.randint", side_effect=AssertionError("SR OFF must not sample crops")), \
             patch.object(self.model.sr, "decode_crop", side_effect=AssertionError("SR OFF decoder")):
            result = train_wsi(self.model, self.sample, optimizer, torch.device("cpu"), self.config)
            evaluated = evaluate(self.model, [self.sample, dict(self.sample, label=0)],
                                 torch.device("cpu"), self.config)
        self.assertEqual(result["train_loss_sr"], 0)
        self.assertFalse(torch.equal(before, self.model.classifier.weight))
        self.assertTrue(all(p.grad is None for p in self.model.sr.P2W.parameters()))
        self.assertEqual(len(evaluated["predictions"]), 2)

    def test_hr_crop_coordinate_and_memmap_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "gt.png"
            pixels = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
            Image.fromarray(pixels).save(path)
            actual = load_hr_crops([path], (3, 7, 17, 19), size=64)
            expected = torch.from_numpy(pixels[3:20, 7:26].copy()).permute(2, 0, 1).float() / 255
            torch.testing.assert_close(actual[0], expected)
            output = Path(temp) / "sr.npy"
            lr = self.lr[:1, :, :2, :3]
            reconstruct(self.model, lr, output)
            saved = np.load(output, mmap_mode="r")
            self.assertEqual(saved.shape, (1, 3, 64, 96))
            with torch.no_grad():
                crop = self.model.sr(lr, (5, 7, 11, 13))
            torch.testing.assert_close(torch.from_numpy(saved[:, :, 5:16, 7:20].copy()), crop,
                                       atol=3e-6, rtol=3e-5)
            del saved

    def test_groupnorm_train_output_independent_of_region_batching(self):
        self.model.train()
        with torch.no_grad():
            feature = torch.randn(3, 64, 32, 32)
            full = self.model.region_encoder(feature)
            split = torch.cat([self.model.region_encoder(x) for x in feature.split(1)])
        torch.testing.assert_close(full, split, atol=3e-5, rtol=3e-4)
