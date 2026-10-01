import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from models.baseline import MetaSRABMIL
from models.metasr import MemoryEfficientMetaRDN
from validation import (compare_gradients, gradients, input_matrix_wpn,
                        max_error, official_decode, run_equivalence)


class MetaSRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def setUp(self):
        torch.manual_seed(71)

    def test_official_forward_checkpoint_and_small_scale_equivalence(self):
        result = run_equivalence()
        self.assertTrue(result["passed"])
        print("EQUIVALENCE", result, flush=True)

    def test_official_architecture(self):
        model = MemoryEfficientMetaRDN()
        self.assertEqual((model.D, len(model.RDBs)), (16, 16))
        self.assertEqual(len(model.RDBs[0].convs), 8)
        self.assertEqual(model.RDBs[0].convs[7].conv[0].in_channels, 512)
        self.assertEqual(model.RDBs[0].LFF.in_channels, 576)
        self.assertEqual(model.P2W.meta_block[0].weight.shape, (256, 3))
        self.assertEqual(model.P2W.meta_block[2].weight.shape, (1728, 256))
        self.assertFalse(any(isinstance(m, (torch.nn.BatchNorm2d, torch.nn.Dropout))
                             for m in model.modules()))

    def test_unaligned_crop_and_decoder_gradients(self):
        model = MemoryEfficientMetaRDN(scale=3, lr_chunk_size=2).eval()
        feature = torch.randn(2, 64, 4, 5, requires_grad=True)
        pos, _ = input_matrix_wpn(4, 5, 3)
        full = official_decode(model, feature, pos)
        box = (1, 2, 10, 11)
        expected = full[:, :, 1:11, 2:13]
        actual = model.decode_crop(feature, box)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
        expected.square().mean().backward()
        expected_grad, feature_grad = gradients(model), feature.grad.clone()
        model.zero_grad(set_to_none=True)
        feature.grad = None
        actual.square().mean().backward()
        error = compare_gradients(expected_grad, gradients(model))
        torch.testing.assert_close(feature.grad, feature_grad, atol=3e-6, rtol=3e-5)
        print("CROP", {"forward_error": max_error(actual, expected), "gradient_error": error}, flush=True)

    def test_scale32_n1_n6_sr_forward_backward_and_streaming(self):
        model = MemoryEfficientMetaRDN(scale=32, lr_chunk_size=1).eval()
        self.assertEqual(model.position_table("cpu").shape, (1024, 3))
        pos, _ = input_matrix_wpn(2, 3, 32)
        table = model.position_table("cpu")
        torch.testing.assert_close(pos.reshape(2, 32, 3, 32, 3)[1, :, 2].reshape(-1, 3), table,
                                   atol=0, rtol=0)
        for n in (1, 6):
            lr = torch.rand(n, 3, 2, 3)
            feature = model.extract_features(lr)
            self.assertEqual(feature.shape, (n, 64, 2, 3))
            with patch.object(model, "repeat_x", side_effect=AssertionError("repeat_x forbidden")):
                with patch("models.metasr.F.unfold", wraps=F.unfold) as unfold:
                    crop = model.decode_crop(feature, (3, 5, 37, 52))
                    self.assertEqual(unfold.call_count, 1)
                crop.abs().mean().backward()
            self.assertTrue(torch.isfinite(model.SFENet1.weight.grad).all())
            self.assertGreater(model.P2W.meta_block[2].weight.grad.abs().max().item(), 0)
            output = torch.empty(n, 3, 64, 96)
            for y, x, tile in model.iter_full_sr(feature):
                self.assertFalse(tile.requires_grad)
                output[:, :, y:y + tile.shape[2], x:x + tile.shape[3]] = tile
            torch.testing.assert_close(output[:, :, 3:40, 5:57], crop.detach(), atol=3e-6, rtol=3e-5)
            model.zero_grad(set_to_none=True)

    def test_abmil_joint_backward_and_sr_off(self):
        model = MetaSRABMIL(lr_chunk_size=3).eval()
        self.assertEqual(sum(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()), 0)
        self.assertEqual(sum(isinstance(m, torch.nn.GroupNorm) for m in model.region_encoder.modules()), 20)
        lr = torch.rand(2, 3, 8, 8)
        target = torch.tensor([1])
        for coefficient in (0.0, 0.1, 1.0):
            model.zero_grad(set_to_none=True)
            if coefficient == 0:
                with patch.object(model.sr, "decode_crop", side_effect=AssertionError("SR OFF")):
                    loss, cls, sr, logits = model.joint_loss(lr, target, lambda_sr=0)
            else:
                loss, cls, sr, logits = model.joint_loss(lr, target, lambda_sr=coefficient,
                    box=(3, 7, 32, 33), hr_crop=torch.rand(2, 3, 32, 33))
            self.assertEqual(logits.shape, (1, 2))
            torch.testing.assert_close(loss, cls + coefficient * sr)
            loss.backward()
            for module in (model.sr.SFENet1, model.region_encoder.conv1, model.classifier,
                           model.mil_head.attention_V, model.mil_head.attention_w):
                self.assertTrue(torch.isfinite(module.weight.grad).all())
                self.assertGreater(module.weight.grad.abs().max().item(), 0)
            if coefficient == 0:
                self.assertEqual(sr.item(), 0)
                self.assertTrue(all(p.grad is None for p in model.sr.P2W.parameters()))
                self.assertTrue(all(p.grad is None for p in model.sr.add_mean.parameters()))
            else:
                self.assertTrue(all(p.grad is not None for p in model.sr.P2W.parameters()))

    def test_chunk_size_and_boundary_crops(self):
        model = MemoryEfficientMetaRDN(scale=4).eval()
        feature = torch.randn(1, 64, 3, 5)
        pos, _ = input_matrix_wpn(3, 5, 4)
        expected = official_decode(model, feature, pos)
        for chunk_size in (1, 2, 7):
            model.lr_chunk_size = chunk_size
            for box in ((0, 0, 12, 20), (11, 19, 1, 1), (3, 3, 5, 6), (4, 8, 4, 8)):
                y, x, h, w = box
                actual = model.decode_crop(feature, box)
                torch.testing.assert_close(actual, expected[:, :, y:y+h, x:x+w], atol=3e-6, rtol=3e-5)


if __name__ == "__main__":
    unittest.main()
