import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from models.metasr import MemoryEfficientMetaRDN
from validation import (compare_gradients, gradients, input_matrix_wpn,
                        official_decode)


class Native512MetaSRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def setUp(self):
        torch.manual_seed(71)
        self.model = MemoryEfficientMetaRDN(scale=16, rdn_blocks=1,
                                            lr_chunk_size=2).eval()

    def test_scale16_position_and_weight_tables_match_official_coordinates(self):
        positions, mask = input_matrix_wpn(3, 5, 16)
        self.assertTrue(mask.all())
        expected = torch.tensor([[1 / 16, y / 16, x / 16]
                                 for y in range(16) for x in range(16)])
        actual = self.model.position_table("cpu")
        self.assertEqual(actual.shape, (256, 3))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        official = positions.reshape(3, 16, 5, 16, 3)[2, :, 4].reshape(-1, 3)
        torch.testing.assert_close(actual, official, atol=0, rtol=0)
        feature = torch.zeros(1, 64, 3, 5)
        weights = self.model.weight_table(feature)
        expected_weights = self.model.P2W(official).reshape(256, 576, 3)
        self.assertEqual(weights.shape, (256, 576, 3))
        torch.testing.assert_close(weights, expected_weights, atol=0, rtol=0)

    def test_scale16_chunks_random_unaligned_and_boundary_crops_match_official(self):
        feature = torch.randn(2, 64, 3, 5)
        positions, _ = input_matrix_wpn(3, 5, 16)
        with torch.no_grad():
            expected = official_decode(self.model, feature, positions)
            self.assertEqual(expected.shape, (2, 3, 48, 80))
            boxes = [(0, 0, 48, 80), (47, 79, 1, 1), (3, 5, 37, 52),
                     (16, 16, 16, 16), (0, 79, 48, 1), (47, 0, 1, 80)]
            generator = torch.Generator().manual_seed(109)
            for _ in range(12):
                height = int(torch.randint(1, 49, (), generator=generator))
                width = int(torch.randint(1, 81, (), generator=generator))
                top = int(torch.randint(0, 49 - height, (), generator=generator))
                left = int(torch.randint(0, 81 - width, (), generator=generator))
                boxes.append((top, left, height, width))
            for chunk in (1, 2, 7):
                self.model.lr_chunk_size = chunk
                for box in boxes:
                    top, left, height, width = box
                    with self.subTest(chunk=chunk, box=box), \
                            patch.object(self.model, "repeat_x",
                                         side_effect=AssertionError("Repeated full features")), \
                            patch("models.metasr.F.unfold", wraps=F.unfold) as unfold:
                        actual = self.model.decode_crop(feature, box)
                        self.assertEqual(unfold.call_count, 1)
                        torch.testing.assert_close(
                            actual, expected[:, :, top:top + height, left:left + width],
                            atol=3e-6, rtol=3e-5)

    def test_scale16_decoder_and_feature_gradients_match_official(self):
        feature = torch.randn(2, 64, 3, 5, requires_grad=True)
        positions, _ = input_matrix_wpn(3, 5, 16)
        expected = official_decode(self.model, feature, positions)[:, :, 3:40, 5:57]
        actual = self.model.decode_crop(feature, (3, 5, 37, 52))
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
        expected.square().mean().backward()
        expected_parameters = gradients(self.model)
        expected_feature = feature.grad.clone()
        self.model.zero_grad(set_to_none=True)
        feature.grad = None
        actual.square().mean().backward()
        compare_gradients(expected_parameters, gradients(self.model))
        torch.testing.assert_close(feature.grad, expected_feature, atol=3e-6, rtol=3e-5)
        for value in (feature.grad, self.model.P2W.meta_block[0].weight.grad,
                      self.model.P2W.meta_block[2].weight.grad,
                      self.model.add_mean.weight.grad):
            self.assertTrue(torch.isfinite(value).all())
            self.assertGreater(value.abs().max().item(), 0)

    def test_scale16_rdn_checkpoint_and_crop_backward_for_multiple_regions(self):
        for checkpoint_rdb in (False, True):
            self.model.checkpoint_rdb = checkpoint_rdb
            for count in (1, 3):
                with self.subTest(checkpoint_rdb=checkpoint_rdb, regions=count):
                    self.model.zero_grad(set_to_none=True)
                    lr = torch.rand(count, 3, 3, 5, requires_grad=True)
                    feature = self.model.extract_features(lr)
                    self.assertEqual(feature.shape, (count, 64, 3, 5))
                    crop = self.model.decode_crop(feature, (3, 5, 37, 52))
                    self.assertEqual(crop.shape, (count, 3, 37, 52))
                    crop.square().mean().backward()
                    for value in (lr.grad, self.model.SFENet1.weight.grad,
                                  self.model.RDBs[0].LFF.weight.grad,
                                  self.model.P2W.meta_block[2].weight.grad):
                        self.assertTrue(torch.isfinite(value).all())
                        self.assertGreater(value.abs().max().item(), 0)

    def test_scale16_streaming_matches_official_without_retaining_a_graph(self):
        feature = torch.randn(2, 64, 3, 5, requires_grad=True)
        positions, _ = input_matrix_wpn(3, 5, 16)
        with torch.no_grad():
            expected = official_decode(self.model, feature, positions)
        actual = torch.empty_like(expected)
        covered = torch.zeros(48, 80, dtype=torch.int32)
        for top, left, tile in self.model.iter_full_sr(feature):
            self.assertFalse(tile.requires_grad)
            self.assertIsNone(tile.grad_fn)
            height, width = tile.shape[-2:]
            actual[:, :, top:top + height, left:left + width] = tile
            covered[top:top + height, left:left + width] += 1
        self.assertTrue(torch.all(covered == 1))
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


if __name__ == "__main__":
    unittest.main()
