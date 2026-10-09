import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from engine import train_wsi, wsi_loss
from models.baseline import MetaSRABMIL, SpatialPyramidPooling, model_from_checkpoint
from train_e2e import load_config


class SPPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def setUp(self):
        torch.manual_seed(108)
        self.model = MetaSRABMIL(classification_encoder="spp", rdn_blocks=8)
        self.lr = torch.rand(3, 3, 8, 8)
        self.truth = torch.rand(3, 3, 16, 16)
        self.sample = {"slide_id": "synthetic", "n_regions": 3, "label": 1,
                       "lr_paths": [0, 1, 2], "hr_paths": [0, 1, 2]}
        self.config = {"training": {"use_region_microbatch": False, "region_microbatch_size": 1},
                       "lambda_sr": 0.1, "sr_train_crop": 16, "precision": "fp32"}

    def test_pooling_dimensions_and_parameter_free_gradient(self):
        feature = torch.randn(3, 64, 7, 9, requires_grad=True)
        pooling = SpatialPyramidPooling()
        embedding = pooling(feature)
        self.assertEqual(embedding.shape, (3, 3200))
        self.assertEqual(sum(p.numel() for p in pooling.parameters()), 0)
        start = 0
        for level in (1, 2, 3, 6):
            width = 64 * level * level
            expected = F.adaptive_max_pool2d(feature, level).flatten(1)
            torch.testing.assert_close(embedding[:, start:start + width], expected, atol=0, rtol=0)
            start += width
        embedding.square().mean().backward()
        self.assertTrue(torch.isfinite(feature.grad).all())
        self.assertGreater(feature.grad.abs().sum().item(), 0)
        self.assertIsInstance(self.model.region_encoder, SpatialPyramidPooling)
        self.assertEqual(self.model.classifier.in_features, 3200)

    def test_joint_loss_sr_on_and_off_gradient_connections(self):
        for coefficient in (0.0, 0.1):
            self.model.zero_grad(set_to_none=True)
            if coefficient:
                result = self.model.joint_loss(self.lr, torch.tensor([1]), lambda_sr=coefficient,
                    box=(17, 31, 16, 16), hr_crop=self.truth)
            else:
                with patch.object(self.model.sr, "decode_crop", side_effect=AssertionError("SR OFF")):
                    result = self.model.joint_loss(self.lr, torch.tensor([1]), lambda_sr=0)
            total, cls, sr, logits = result
            self.assertEqual(logits.shape, (1, 2))
            torch.testing.assert_close(total, cls + coefficient * sr)
            total.backward()
            for parameter in (self.model.sr.SFENet1.weight, self.model.classifier.weight,
                              self.model.mil_head.attention_V.weight):
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.norm().item(), 0)
            p2w = self.model.sr.P2W.meta_block[2].weight
            if coefficient:
                self.assertGreater(p2w.grad.norm().item(), 0)
            else:
                self.assertIsNone(p2w.grad)
                self.assertEqual(sr.item(), 0)

    def test_fp32_full_and_microbatch_objective_and_gradients(self):
        reference = None
        for micro in (False, True):
            self.config["training"]["use_region_microbatch"] = micro
            self.config["training"]["region_microbatch_size"] = 2
            self.model.zero_grad(set_to_none=True)
            with patch("engine.load_images", side_effect=lambda paths, size: self.lr[paths]), \
                 patch("engine.load_hr_crops", side_effect=lambda paths, box: self.truth[paths]), \
                 patch("engine.torch.randint", return_value=torch.tensor([[0, 0], [17, 31], [120, 120]])):
                result = wsi_loss(self.model, self.sample, torch.device("cpu"), self.config)
            result[0].backward()
            values = [value.detach().clone() for value in result]
            gradients = {name: p.grad.clone() for name, p in self.model.named_parameters()
                         if name in ("sr.SFENet1.weight", "sr.P2W.meta_block.2.weight",
                                     "classifier.weight", "mil_head.attention_V.weight")}
            if reference is None:
                reference = values, gradients
            else:
                for actual, expected in zip(values, reference[0]):
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
                for name, actual in gradients.items():
                    torch.testing.assert_close(actual, reference[1][name], atol=2e-5, rtol=2e-4)

    def test_one_adam_step_with_parameter_free_encoder(self):
        before = self.model.classifier.weight.detach().clone()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        with patch("engine.load_images", side_effect=lambda paths, size: self.lr[paths]), \
             patch("engine.load_hr_crops", side_effect=lambda paths, box: self.truth[paths]), \
             patch("engine.torch.randint", return_value=torch.tensor([[0, 0], [17, 31], [120, 120]])):
            result = train_wsi(self.model, self.sample, optimizer, torch.device("cpu"), self.config)
        self.assertFalse(torch.equal(before, self.model.classifier.weight))
        self.assertEqual(list(self.model.region_encoder.parameters()), [])
        self.assertAlmostEqual(result["train_loss_total"],
                               result["train_loss_cls"] + 0.1 * result["train_loss_sr"], places=6)

    def test_checkpoint_architecture_and_legacy_loading(self):
        for encoder, version in (("spp", "metasr-abmil-v2"), ("resnet18", "metasr-abmil-v1")):
            model = MetaSRABMIL(classification_encoder=encoder, rdn_blocks=4)
            saved = {"format": version, "config": {"metasr": {"rdn_blocks": 4}},
                     "model_state": model.state_dict()}
            if version.endswith("v2"):
                saved["config"]["classification_encoder"] = encoder
                saved["architecture"] = model.architecture()
            restored = model_from_checkpoint(saved)
            self.assertEqual(restored.architecture(), model.architecture())
            for name, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)
            invalid = copy.deepcopy(saved)
            invalid["model_state"].pop("classifier.weight")
            with self.assertRaises(RuntimeError):
                model_from_checkpoint(invalid)
            if version.endswith("v2"):
                invalid = copy.deepcopy(saved)
                invalid["architecture"]["embedding_dim"] = 512
                with self.assertRaises(ValueError):
                    model_from_checkpoint(invalid)

    def test_spp_d8_config_and_legacy_default(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs/spp_d8.yaml")
        self.assertEqual(config["classification_encoder"], "spp")
        self.assertEqual(config["metasr"]["rdn_blocks"], 8)
        self.assertEqual((config["epochs"], config["early_stopping_patience"]), (100, 15))
        self.assertEqual(load_config(root / "configs/baseline.yaml")["classification_encoder"], "resnet18")


if __name__ == "__main__":
    unittest.main()
