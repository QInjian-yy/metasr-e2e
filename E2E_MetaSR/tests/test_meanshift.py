import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from models.baseline import MetaSRABMIL, model_from_checkpoint
from models.metasr import MemoryEfficientMetaRDN
from train_e2e import load_config


MEANSHIFT_PARAMETERS = {"sub_mean.weight", "sub_mean.bias", "add_mean.weight", "add_mean.bias"}


class MeanShiftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_default_and_explicit_parameter_flags(self):
        for options in ({}, {"freeze_meanshift": False}, {"freeze_meanshift": True}):
            with self.subTest(options=options):
                model = MemoryEfficientMetaRDN(rdn_blocks=1, **options)
                model.train()
                for name, parameter in model.named_parameters():
                    frozen = options.get("freeze_meanshift", False) and name in MEANSHIFT_PARAMETERS
                    self.assertEqual(parameter.requires_grad, not frozen, name)

    def test_same_initialization_and_forward_for_both_encoders(self):
        for encoder in ("spp", "resnet18"):
            with self.subTest(encoder=encoder):
                torch.manual_seed(71)
                legacy = MetaSRABMIL(classification_encoder=encoder, rdn_blocks=1, scale=2).eval()
                legacy_rng = torch.get_rng_state()
                torch.manual_seed(71)
                frozen = MetaSRABMIL(classification_encoder=encoder, rdn_blocks=1, scale=2,
                                    freeze_meanshift=True).eval()
                self.assertTrue(torch.equal(legacy_rng, torch.get_rng_state()))
                self.assertEqual(legacy.architecture(), frozen.architecture())
                self.assertEqual(legacy.state_dict().keys(), frozen.state_dict().keys())
                for name, value in legacy.state_dict().items():
                    torch.testing.assert_close(value, frozen.state_dict()[name], atol=0, rtol=0)
                lr = torch.rand(2, 3, 8, 8)
                with torch.no_grad():
                    torch.testing.assert_close(legacy(lr), frozen(lr), atol=0, rtol=0)
                    box = (1, 2, 5, 6)
                    torch.testing.assert_close(legacy.sr(lr, box), frozen.sr(lr, box), atol=0, rtol=0)

    def test_joint_backward_and_adam_respect_freezing(self):
        for freeze in (False, True):
            with self.subTest(freeze=freeze):
                torch.manual_seed(71)
                model = MetaSRABMIL(classification_encoder="spp", rdn_blocks=1, scale=2,
                                   freeze_meanshift=freeze).train()
                before = {name: p.detach().clone() for name, p in model.sr.named_parameters()
                          if name in MEANSHIFT_PARAMETERS}
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
                lr = torch.rand(2, 3, 8, 8, requires_grad=True)
                loss, _, _, _ = model.joint_loss(lr, torch.tensor([1]), lambda_sr=0.1,
                    box=(1, 2, 5, 6), hr_crop=torch.rand(2, 3, 5, 6))
                loss.backward()
                for name, parameter in model.sr.named_parameters():
                    if name in MEANSHIFT_PARAMETERS:
                        if freeze:
                            self.assertIsNone(parameter.grad, name)
                        else:
                            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                for parameter in (lr, model.sr.SFENet1.weight, model.sr.P2W.meta_block[2].weight,
                                  model.mil_head.attention_V.weight, model.classifier.weight):
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
                optimizer.step()
                for name, parameter in model.sr.named_parameters():
                    if name in before:
                        self.assertEqual(torch.equal(parameter.detach(), before[name]), freeze, name)

    def test_config_default_and_boolean_validation(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / "configs/baseline.yaml").read_text(encoding="utf-8"))
        config["metasr"].pop("freeze_meanshift", None)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            self.assertIs(load_config(path)["metasr"]["freeze_meanshift"], False)
            for value in (False, True, "false", 0, 1, None):
                with self.subTest(value=value):
                    config["metasr"]["freeze_meanshift"] = value
                    path.write_text(yaml.safe_dump(config), encoding="utf-8")
                    if isinstance(value, bool):
                        self.assertIs(load_config(path)["metasr"]["freeze_meanshift"], value)
                    else:
                        with self.assertRaisesRegex(ValueError, "freeze_meanshift must be a boolean"):
                            load_config(path)
                        with self.assertRaisesRegex(ValueError, "freeze_meanshift must be a boolean"):
                            MemoryEfficientMetaRDN(freeze_meanshift=value)

    def test_checkpoint_restores_flags_without_resetting_weights(self):
        model = MetaSRABMIL(classification_encoder="spp", rdn_blocks=1)
        with torch.no_grad():
            model.sr.sub_mean.bias.add_(0.25)
            model.sr.add_mean.weight.mul_(0.9)
        for options in ({}, {"freeze_meanshift": False}, {"freeze_meanshift": True}):
            with self.subTest(options=options):
                saved = {"format": "metasr-abmil-v2", "architecture": model.architecture(),
                         "config": {"classification_encoder": "spp",
                                    "metasr": {"rdn_blocks": 1, **options}},
                         "model_state": model.state_dict()}
                restored = model_from_checkpoint(saved)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
                for name, parameter in restored.sr.named_parameters():
                    frozen = options.get("freeze_meanshift", False) and name in MEANSHIFT_PARAMETERS
                    self.assertEqual(parameter.requires_grad, not frozen, name)


if __name__ == "__main__":
    unittest.main()
