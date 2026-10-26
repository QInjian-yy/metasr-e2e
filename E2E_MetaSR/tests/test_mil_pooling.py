"""Pooling correctness, legacy compatibility and the opt-in training/eval/SR entrypoints."""

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

import eval_e2e_mil
import infer_sr_mil
import train_e2e_mil
from data_modes import data_spec
from engine import train_wsi
from models.baseline import MetaSRABMIL
from models.meanmil import MeanMIL
from models.mil_model import CHECKPOINT_FORMAT, MetaSRMIL, model_from_checkpoint
from models.region_gate_model import RegionGateMetaSRABMIL
from wsi_data import Camelyon16WSI


ROOT = Path(__file__).resolve().parents[1]


class MILPoolingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def setUp(self):
        torch.manual_seed(271)

    def checkpoint(self, model, config):
        return {"format": CHECKPOINT_FORMAT, "architecture": model.architecture(),
                "config": copy.deepcopy(config), "data_spec": data_spec(config).metadata(),
                "model_state": model.state_dict(), "fold": 0,
                "data_provenance": {"manifest": {"sha256": "synthetic"}}}

    def config(self):
        config = train_e2e_mil.load_config(ROOT / "configs/spp_d1_native512_aug_sr0.yaml")
        config.update(precision="fp32", epochs=1, lambda_sr=0, sr_train_crop=4)
        config["metasr"].update(checkpoint_rdb=True, lr_chunk_size=4)
        config["augmentation"]["enabled"] = False
        return config

    def test_mean_formula_uniform_weights_permutation_and_gradients(self):
        head = MeanMIL()
        self.assertEqual(list(head.parameters()), [])
        for n in (1, 5, 35):
            with self.subTest(n=n):
                embeddings = torch.randn(n, 7, dtype=torch.float64, requires_grad=True)
                pooled, weights = head(embeddings)
                reference = sum(embeddings[i:i+1] for i in range(n)) / n
                torch.testing.assert_close(pooled, reference, atol=1e-14, rtol=1e-14)
                torch.testing.assert_close(weights, torch.full((n,), 1 / n, dtype=torch.float64))
                torch.testing.assert_close(head(embeddings.flip(0))[0], pooled, atol=1e-14, rtol=1e-14)
                pooled.sum().backward()
                torch.testing.assert_close(embeddings.grad, torch.full_like(embeddings, 1 / n))
        for invalid in (torch.empty(0, 7), torch.ones(1, 2, 7)):
            with self.assertRaises(ValueError):
                head(invalid)

    def test_abmil_matches_all_original_models_and_mean_preserves_shared_weights_and_rng(self):
        choices = [(name, "linear") for name in
                   ("spp", "spp_ln", "spp_avg", "spp_1344", "resnet18", "spp_region_gate")]
        choices.append(("spp", "mlp"))
        for encoder, classifier in choices:
            with self.subTest(encoder=encoder, classifier=classifier):
                kwargs = dict(classification_encoder=encoder, classification_head=classifier,
                              rdn_blocks=1, scale=16)
                original_type = RegionGateMetaSRABMIL if encoder == "spp_region_gate" else MetaSRABMIL
                torch.manual_seed(271)
                original = original_type(**kwargs).eval()
                expected_rng = torch.get_rng_state()
                for pooling in ("abmil", "meanmil"):
                    torch.manual_seed(271)
                    model = MetaSRMIL(mil_pooling=pooling, **kwargs).eval()
                    self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                    for name, value in model.state_dict().items():
                        torch.testing.assert_close(value, original.state_dict()[name], atol=0, rtol=0)
                    architecture = dict(original.architecture(), mil_pooling=pooling)
                    if pooling == "meanmil":
                        architecture.pop("attention_hidden_dim")
                        self.assertEqual(list(model.mil_head.parameters()), [])
                        self.assertFalse(any(name.startswith("mil_head.") for name in model.state_dict()))
                    else:
                        self.assertEqual(model.state_dict().keys(), original.state_dict().keys())
                    self.assertEqual(model.architecture(), architecture)
                    embeddings = torch.randn(5, architecture["embedding_dim"])
                    with torch.no_grad():
                        expected = (original.forward_embeddings(embeddings) if pooling == "abmil"
                                    else original.classifier(embeddings.mean(dim=0, keepdim=True)))
                        torch.testing.assert_close(model.forward_embeddings(embeddings), expected, atol=0, rtol=0)

    def test_checkpoint_roundtrip_and_reject_wrong_pooling_or_extra_parameters(self):
        for encoder, classifier, mode, scale in (("spp", "linear", "legacy256", 32),
                ("spp", "mlp", "native512", 16), ("spp_region_gate", "linear", "native512", 16)):
            for pooling in ("abmil", "meanmil"):
                with self.subTest(encoder=encoder, pooling=pooling, mode=mode):
                    config = {"classification_encoder": encoder, "classification_head": classifier,
                              "mil_pooling": pooling, "metasr": {"rdn_blocks": 1, "scale": scale,
                              "rgb_range": 1}, "data_mode": mode}
                    model = MetaSRMIL(classification_encoder=encoder, classification_head=classifier,
                                      mil_pooling=pooling, **config["metasr"]).eval()
                    saved = self.checkpoint(model, config)
                    stream = io.BytesIO()
                    torch.save(saved, stream)
                    stream.seek(0)
                    restored = model_from_checkpoint(torch.load(stream, weights_only=False)).eval()
                    embeddings = torch.randn(3, model.architecture()["embedding_dim"])
                    with torch.no_grad():
                        torch.testing.assert_close(restored.forward_embeddings(embeddings),
                                                   model.forward_embeddings(embeddings), atol=0, rtol=0)
                    invalid = copy.deepcopy(saved)
                    invalid["config"]["mil_pooling"] = "meanmil" if pooling == "abmil" else "abmil"
                    with self.assertRaises(ValueError):
                        model_from_checkpoint(invalid)
                    invalid = copy.deepcopy(saved)
                    invalid["model_state"]["mil_head.unexpected"] = torch.ones(1)
                    with self.assertRaises(RuntimeError):
                        model_from_checkpoint(invalid)
        for encoder, old_format in (("resnet18", "metasr-abmil-v1"), ("spp", "metasr-abmil-v2"),
                                    ("spp_region_gate", "metasr-abmil-region-gate-v1")):
            cls = RegionGateMetaSRABMIL if encoder == "spp_region_gate" else MetaSRABMIL
            original = cls(classification_encoder=encoder, rdn_blocks=1, scale=2)
            saved = {"format": old_format, "architecture": original.architecture(),
                     "config": {"classification_encoder": encoder, "metasr": {"rdn_blocks": 1, "scale": 2}},
                     "model_state": original.state_dict()}
            restored = model_from_checkpoint(saved)
            self.assertIs(type(restored), cls)
            for name, value in original.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)

    def test_engine_full_bag_mean_microbatch_gradients_sr_and_adam(self):
        config = self.config()
        config["mil_pooling"] = "meanmil"
        lr, truth = torch.rand(5, 3, 8, 8), torch.rand(5, 3, 4, 4)
        sample = {"slide_id": "synthetic", "label": 1, "data_mode": "native512",
                  "n_regions": 5, "lr_paths": list(range(5)), "hr_paths": list(range(5))}
        initial = MetaSRMIL(classification_encoder="spp", mil_pooling="meanmil", **config["metasr"]).state_dict()
        for coefficient in (0, 0.1):
            config["lambda_sr"] = coefficient
            reference = None
            for micro in (False, True):
                with self.subTest(sr=coefficient, micro=micro):
                    config["training"] = {"use_region_microbatch": micro, "region_microbatch_size": 2}
                    model = MetaSRMIL(classification_encoder="spp", mil_pooling="meanmil", **config["metasr"])
                    model.load_state_dict(initial)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                    bags = []
                    handle = model.mil_head.register_forward_pre_hook(lambda module, args: bags.append(args[0]))
                    hr_effect = (lambda paths, box: truth[paths]) if coefficient else AssertionError("SR OFF HR")
                    rng_effect = (lambda *args: torch.zeros(5, 2, dtype=torch.long)) if coefficient else AssertionError("SR OFF crops")
                    with patch("engine.load_images", side_effect=lambda paths, size: lr[paths]), \
                         patch("engine.load_hr_crops", side_effect=hr_effect) as hr_load, \
                         patch("engine.torch.randint", side_effect=rng_effect), \
                         patch.object(optimizer, "step", wraps=optimizer.step) as step:
                        result = train_wsi(model, sample, optimizer, torch.device("cpu"), config)
                    handle.remove()
                    self.assertEqual([tuple(bag.shape) for bag in bags], [(5, 3200)])
                    step.assert_called_once()
                    self.assertEqual(hr_load.call_count, 5 if coefficient else 0)
                    self.assertEqual(result["train_loss_sr"] == 0, coefficient == 0)
                    for name in ("sr.SFENet1.weight", "classifier.weight"):
                        parameter = dict(model.named_parameters())[name]
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(parameter.grad.abs().sum().item(), 0)
                        self.assertFalse(torch.equal(parameter.detach(), initial[name]))
                    self.assertEqual(all(p.grad is None for p in model.sr.P2W.parameters()), coefficient == 0)
                    if coefficient:
                        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.sr.P2W.parameters()), 0)
                    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
                    if reference is None:
                        reference = result, gradients
                    else:
                        for key in ("train_loss_cls", "train_loss_sr", "train_loss_total"):
                            self.assertAlmostEqual(result[key], reference[0][key], places=5)
                        for name, gradient in gradients.items():
                            torch.testing.assert_close(gradient, reference[1][name], atol=2e-5, rtol=3e-4)

    def test_config_defaults_existing_region_gate_and_invalid_selection(self):
        for path in (ROOT / "configs/baseline.yaml", ROOT / "configs/native512_d1_region_gate_sr01_noaug_m8_300.yaml"):
            config = train_e2e_mil.load_config(path)
            self.assertEqual(config["mil_pooling"], "abmil")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.yaml"
            config = self.config()
            for pooling in ("meanmil", "unknown"):
                config["mil_pooling"] = pooling
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                if pooling == "meanmil":
                    self.assertEqual(train_e2e_mil.load_config(path)["mil_pooling"], pooling)
                else:
                    with self.assertRaises(ValueError):
                        train_e2e_mil.load_config(path)
        with self.assertRaises(ValueError):
            MetaSRMIL(mil_pooling="unknown", rdn_blocks=1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_meanmil_joint_backward_and_adam_at_both_scales(self):
        device = torch.device("cuda:0")
        precisions = ("fp32", "bf16") if torch.cuda.is_bf16_supported() else ("fp32",)
        lr, truth = torch.rand(2, 3, 8, 8), torch.rand(2, 3, 4, 4)
        for scale in (16, 32):
            for precision in precisions:
                with self.subTest(scale=scale, precision=precision):
                    config = self.config()
                    config.update(data_mode="native512" if scale == 16 else "legacy256",
                                  precision=precision, lambda_sr=0.1)
                    config["metasr"]["scale"] = scale
                    config["training"] = {"use_region_microbatch": True, "region_microbatch_size": 1}
                    model = MetaSRMIL(classification_encoder="spp", mil_pooling="meanmil",
                                      **config["metasr"]).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                    before = model.classifier.weight.detach().clone()
                    sample = {"slide_id": "synthetic_cuda", "label": 1,
                              "data_mode": config["data_mode"], "n_regions": 2,
                              "lr_paths": [0, 1], "hr_paths": [0, 1]}
                    with patch("engine.load_images", side_effect=lambda paths, size: lr[paths]), \
                         patch("engine.load_hr_crops", side_effect=lambda paths, box: truth[paths]), \
                         patch("engine.torch.randint", return_value=torch.zeros(2, 2, dtype=torch.long)):
                        result = train_wsi(model, sample, optimizer, device, config)
                    self.assertGreater(result["train_loss_sr"], 0)
                    self.assertFalse(torch.equal(before, model.classifier.weight))
                    for module in (model.sr, model.classifier):
                        gradients = [p.grad for p in module.parameters() if p.grad is not None]
                        self.assertTrue(gradients)
                        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
                    self.assertTrue(all(p.grad is not None for p in model.sr.P2W.parameters()))

    def test_new_entrypoints_train_save_evaluate_infer_and_preserve_existing_output(self):
        lr = torch.rand(3, 3, 8, 8)
        samples = [{"slide_id": "normal", "label": 0, "data_mode": "native512",
                    "n_regions": 3, "lr_paths": [0, 1, 2], "hr_paths": []},
                   {"slide_id": "tumor", "label": 1, "data_mode": "native512",
                    "n_regions": 2, "lr_paths": [1, 2], "hr_paths": []}]
        dataset = Camelyon16WSI(samples)
        provenance = {"manifest": {"sha256": "synthetic"}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(self.config()), encoding="utf-8")
            original_config = config_path.read_bytes()
            for pooling in ("meanmil", "abmil"):
                with self.subTest(pooling=pooling):
                    output = root / pooling
                    argv = ["train_e2e_mil.py", "--config", str(config_path), "--mil-pooling", pooling,
                            "--data-root", str(root), "--output", str(output), "--device", "cpu"]
                    with patch("sys.argv", argv), patch("sys.stdout", new_callable=io.StringIO), \
                         patch("train_e2e_mil.run_equivalence", return_value={"selected_config": "synthetic"}), \
                         patch("train_e2e_mil.load_fold_datasets", return_value=(dataset, dataset)), \
                         patch("train_e2e_mil.provenance", return_value=provenance), \
                         patch("engine.load_images", side_effect=lambda paths, size: lr[paths]):
                        train_e2e_mil.main()
                    self.assertEqual(config_path.read_bytes(), original_config)
                    saved = torch.load(output / "best.pth", weights_only=False)
                    self.assertEqual(saved["format"], CHECKPOINT_FORMAT)
                    self.assertEqual(saved["architecture"]["mil_pooling"], pooling)
                    info = json.loads((output / "run_info.json").read_text(encoding="utf-8"))
                    self.assertEqual((info["status"], info["mil_pooling"]), ("completed", pooling))
                    last_bytes = (output / "last.pth").read_bytes()
                    with patch("sys.argv", argv), self.assertRaisesRegex(ValueError, "new or empty"):
                        train_e2e_mil.main()
                    self.assertEqual((output / "last.pth").read_bytes(), last_bytes)
                    predictions = output / "evaluation.csv"
                    with patch("sys.argv", ["eval_e2e_mil.py", "--checkpoint", str(output / "best.pth"),
                            "--data-root", str(root), "--device", "cpu", "--predictions-csv", str(predictions)]), \
                         patch("sys.stdout", new_callable=io.StringIO), \
                         patch("eval_e2e_mil.load_fold_datasets", return_value=(dataset, dataset)), \
                         patch("eval_e2e_mil.provenance", return_value=provenance), \
                         patch("engine.load_images", side_effect=lambda paths, size: lr[paths]):
                        eval_e2e_mil.main()
                    self.assertEqual(len(predictions.read_text(encoding="utf-8").splitlines()), 3)
                    sr_path = output / "synthetic_sr.npy"
                    with patch("sys.argv", ["infer_sr_mil.py", "--checkpoint", str(output / "best.pth"),
                            "--lr-image", "synthetic.png", "--output", str(sr_path), "--device", "cpu"]), \
                         patch("sys.stdout", new_callable=io.StringIO), \
                         patch("infer_sr_mil.load_images", return_value=lr[:1]) as image_load:
                        infer_sr_mil.main()
                    image_load.assert_called_once_with([Path("synthetic.png")], 512)
                    self.assertEqual(np.load(sr_path).shape, (1, 3, 128, 128))


if __name__ == "__main__":
    unittest.main()
