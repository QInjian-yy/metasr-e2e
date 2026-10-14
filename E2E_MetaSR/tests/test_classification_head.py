import copy
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

import eval_e2e
from engine import evaluate, train_wsi
from models.baseline import MetaSRABMIL, model_from_checkpoint
from train_e2e import load_config


class ClassificationHeadTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(83)
        self.config = {
            "classification_encoder": "spp", "classification_head": "mlp",
            "metasr": {"rdn_blocks": 1, "scale": 2}, "precision": "fp32",
            "training": {"use_region_microbatch": False, "region_microbatch_size": 1},
            "lambda_sr": 0, "sr_train_crop": 4,
        }
        self.lr = torch.rand(2, 3, 8, 8)
        self.samples = [
            {"slide_id": "normal", "label": 0, "n_regions": 2,
             "lr_paths": [0, 1], "hr_paths": []},
            {"slide_id": "tumor", "label": 1, "n_regions": 2,
             "lr_paths": [1, 0], "hr_paths": []},
        ]

    def make_model(self):
        return MetaSRABMIL(classification_encoder="spp", classification_head="mlp",
                           **self.config["metasr"])

    def checkpoint(self, model):
        return {"format": "metasr-abmil-v2", "architecture": model.architecture(),
                "config": copy.deepcopy(self.config), "model_state": model.state_dict(),
                "fold": 0, "data_provenance": {"manifest": {"sha256": "synthetic"}}}

    def test_default_linear_preserves_initialization_keys_and_architecture(self):
        torch.manual_seed(83)
        default = MetaSRABMIL(classification_encoder="spp", rdn_blocks=1, scale=2)
        rng = torch.get_rng_state()
        torch.manual_seed(83)
        explicit = MetaSRABMIL(classification_encoder="spp", classification_head="linear",
                              rdn_blocks=1, scale=2)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIsInstance(default.classifier, nn.Linear)
        self.assertEqual(default.architecture(), {
            "classification_encoder": "spp", "embedding_dim": 3200,
            "attention_hidden_dim": 128, "num_classes": 2,
        })
        self.assertEqual(default.state_dict().keys(), explicit.state_dict().keys())
        self.assertIn("classifier.weight", default.state_dict())
        for name, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], atol=0, rtol=0)
        saved = self.checkpoint(default)
        saved["config"].pop("classification_head")
        restored = model_from_checkpoint(saved).eval()
        with torch.no_grad():
            expected = default.eval()(self.lr)
            torch.testing.assert_close(explicit.eval()(self.lr), expected, atol=0, rtol=0)
            torch.testing.assert_close(restored(self.lr), expected, atol=0, rtol=0)

    def test_mlp_structure_and_invalid_selection(self):
        model = self.make_model()
        head = model.classifier
        self.assertIsInstance(head, nn.Sequential)
        self.assertEqual([type(layer) for layer in head], [nn.Linear, nn.ReLU, nn.Dropout, nn.Linear])
        self.assertEqual((head[0].in_features, head[0].out_features), (3200, 128))
        self.assertEqual((head[3].in_features, head[3].out_features), (128, 2))
        self.assertEqual(head[2].p, 0.3)
        self.assertEqual(model.architecture(), {
            "classification_encoder": "spp", "embedding_dim": 3200,
            "attention_hidden_dim": 128, "num_classes": 2,
            "classification_head": "mlp", "classifier_hidden_dim": 128,
            "classifier_dropout": 0.3,
        })
        for encoder, head_name in (("spp", "unknown"), ("resnet18", "mlp")):
            with self.subTest(encoder=encoder, head=head_name), self.assertRaises(ValueError):
                MetaSRABMIL(classification_encoder=encoder, classification_head=head_name,
                            rdn_blocks=1)

    def test_yaml_defaults_new_config_and_invalid_selection(self):
        configs = Path(__file__).resolve().parents[1] / "configs"
        legacy = load_config(configs / "baseline.yaml")
        self.assertEqual(legacy["classification_head"], "linear")
        selected = load_config(configs / "spp_d8_mlp.yaml")
        self.assertEqual(selected["classification_encoder"], "spp")
        self.assertEqual(selected["classification_head"], "mlp")
        for encoder, head in (("spp", "unknown"), ("resnet18", "mlp")):
            invalid = dict(selected, classification_encoder=encoder, classification_head=head)
            with self.subTest(encoder=encoder, head=head), \
                 patch("train_e2e.yaml.safe_load", return_value=invalid), \
                 self.assertRaises(ValueError):
                load_config(configs / "baseline.yaml")

    def test_dropout_masks_scaling_and_eval_determinism(self):
        model = self.make_model()
        embeddings = torch.randn(3, 3200)
        captures = []
        handle = model.classifier[2].register_forward_hook(
            lambda module, args, output: captures.append((args[0].detach().clone(), output.detach().clone())))
        try:
            model.train()
            first = model.forward_embeddings(embeddings)
            second = model.forward_embeddings(embeddings)
            for before, after in captures:
                nonzero = before != 0
                self.assertTrue(((after == 0) & nonzero).any())
                self.assertTrue(((after != 0) & nonzero).any())
                kept = after != 0
                torch.testing.assert_close(after[kept], before[kept] / 0.7)
            self.assertFalse(torch.equal(first, second))
            model.eval()
            with torch.no_grad():
                first = model.forward_embeddings(embeddings)
                second = model.forward_embeddings(embeddings)
            torch.testing.assert_close(first, second, atol=0, rtol=0)
            for before, after in captures[-2:]:
                torch.testing.assert_close(before, after, atol=0, rtol=0)
        finally:
            handle.remove()

    def test_train_evaluate_train_modes_and_no_evaluation_updates(self):
        model = self.make_model().eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        modes = []
        handle = model.classifier[2].register_forward_hook(
            lambda module, args, output: modes.append(
                (model.training, module.training, torch.is_grad_enabled())))
        try:
            with patch("engine.load_images", side_effect=lambda paths, size: self.lr[paths]), \
                 patch.object(optimizer, "step", wraps=optimizer.step) as step:
                train_wsi(model, self.samples[0], optimizer, torch.device("cpu"), self.config)
                before = {name: value.detach().clone() for name, value in model.state_dict().items()}
                gradients = {name: p.grad.clone() if p.grad is not None else None
                             for name, p in model.named_parameters()}
                train_result = evaluate(model, self.samples, torch.device("cpu"), self.config)
                val_result = evaluate(model, self.samples, torch.device("cpu"), self.config)
                self.assertEqual(train_result, val_result)
                self.assertEqual(step.call_count, 1)
                self.assertTrue(torch.is_grad_enabled())
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[name], atol=0, rtol=0)
                for name, parameter in model.named_parameters():
                    if gradients[name] is None:
                        self.assertIsNone(parameter.grad)
                    else:
                        torch.testing.assert_close(parameter.grad, gradients[name], atol=0, rtol=0)
                train_wsi(model, self.samples[1], optimizer, torch.device("cpu"), self.config)
                self.assertEqual(step.call_count, 2)
            self.assertEqual(modes, [(True, True, True)] + [(False, False, False)] * 4
                             + [(True, True, True)])
            self.assertTrue(model.training)
        finally:
            handle.remove()

    def test_joint_gradients_with_and_without_sr(self):
        for coefficient in (0, 0.1):
            with self.subTest(lambda_sr=coefficient):
                model = self.make_model().train()
                lr = self.lr.clone().requires_grad_()
                total, _, sr, _ = model.joint_loss(
                    lr, torch.tensor([1]), lambda_sr=coefficient,
                    box=(1, 2, 5, 6), hr_crop=torch.rand(2, 3, 5, 6))
                total.backward()
                for parameter in (lr, model.sr.SFENet1.weight, model.sr.RDBs[0].LFF.weight,
                                  model.mil_head.attention_V.weight, model.mil_head.attention_U.weight,
                                  model.mil_head.attention_w.weight,
                                  model.classifier[0].weight, model.classifier[3].weight):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
                if coefficient:
                    self.assertGreater(sr.item(), 0)
                    for parameter in model.sr.P2W.parameters():
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(parameter.grad.abs().sum().item(), 0)
                else:
                    self.assertEqual(sr.item(), 0)
                    self.assertTrue(all(p.grad is None for p in model.sr.P2W.parameters()))

    def test_mlp_checkpoint_roundtrip_and_strict_mismatches(self):
        model = self.make_model().eval()
        saved = self.checkpoint(model)
        restored = model_from_checkpoint(saved).eval()
        self.assertEqual(restored.architecture(), model.architecture())
        with torch.no_grad():
            torch.testing.assert_close(restored(self.lr), model(self.lr), atol=0, rtol=0)
        for key, value in (("classifier_hidden_dim", 64), ("classifier_dropout", 0.2)):
            invalid = copy.deepcopy(saved)
            invalid["architecture"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                model_from_checkpoint(invalid)
        invalid = copy.deepcopy(saved)
        invalid["config"].pop("classification_head")
        with self.assertRaises(ValueError):
            model_from_checkpoint(invalid)
        invalid = copy.deepcopy(saved)
        del invalid["model_state"]["classifier.0.weight"]
        with self.assertRaises(RuntimeError):
            model_from_checkpoint(invalid)

    def test_standalone_eval_entrypoint_disables_dropout_and_gradients(self):
        saved = self.checkpoint(self.make_model())
        captured = {}

        def load(saved_state, device):
            model = model_from_checkpoint(saved_state, device)
            captured["model"] = model
            captured["modes"] = []
            captured["handle"] = model.classifier[2].register_forward_hook(
                lambda module, args, output: captured["modes"].append(
                    (model.training, module.training, torch.is_grad_enabled())))
            return model

        try:
            with patch("sys.argv", ["eval_e2e.py", "--checkpoint", "unused.pth",
                                    "--data-root", "unused", "--device", "cpu"]), \
                 patch("eval_e2e.torch.load", return_value=saved), \
                 patch("eval_e2e.model_from_checkpoint", side_effect=load), \
                 patch("eval_e2e.load_fold_datasets", return_value=([], self.samples)), \
                 patch("eval_e2e.provenance", return_value=saved["data_provenance"]), \
                 patch("engine.load_images", side_effect=lambda paths, size: self.lr[paths]), \
                 patch("sys.stdout", new_callable=io.StringIO):
                eval_e2e.main()
            self.assertEqual(captured["modes"], [(False, False, False)] * 2)
            model = captured["model"]
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, saved["model_state"][name], atol=0, rtol=0)
        finally:
            if "handle" in captured:
                captured["handle"].remove()


if __name__ == "__main__":
    unittest.main()
