import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from augmentation import AugmentedPath, TransformParams, apply_transform
from paired_geometry import crop_transformed
from wsi_data import load_images


class Native512GeometryTests(unittest.TestCase):
    def test_scale16_all_transforms_match_virtual_8k_coordinate_chart(self):
        class CoordinateImage:
            size = (8192, 8192)

            def crop(self, bounds):
                left, top, right, bottom = bounds
                rows = np.arange(top, bottom, dtype=np.int32)[:, None] // 16
                columns = np.arange(left, right, dtype=np.int32)[None, :] // 16
                return Image.fromarray(rows * 512 + columns)

        lr = Image.fromarray(np.arange(512 * 512, dtype=np.int32).reshape(512, 512))
        boxes = [(0, 0, 1, 1), (8191, 8191, 1, 1), (15, 17, 33, 71),
                 (7936, 7936, 256, 256), (8173, 0, 19, 25), (0, 8183, 31, 9)]
        generator = np.random.default_rng(109)
        for _ in range(20):
            height, width = generator.integers(1, 257, size=2)
            top = generator.integers(0, 8193 - height)
            left = generator.integers(0, 8193 - width)
            boxes.append(tuple(map(int, (top, left, height, width))))
        for horizontal, vertical, rotation in itertools.product((False, True), (False, True), range(4)):
            params = TransformParams(horizontal, vertical, rotation)
            transformed_lr = np.asarray(apply_transform(lr, params))
            for box in boxes:
                top, left, height, width = box
                expected = transformed_lr[
                    np.arange(top, top + height)[:, None] // 16,
                    np.arange(left, left + width)[None, :] // 16]
                with self.subTest(params=params, box=box), \
                        patch("paired_geometry.apply_transform", wraps=apply_transform) as transform:
                    actual = crop_transformed(CoordinateImage(), box, params)
                    np.testing.assert_array_equal(np.asarray(actual), expected)
                    self.assertEqual(transform.call_count, 1)
                    tile = transform.call_args.args[0]
                    self.assertEqual(tile.width * tile.height, width * height)
                    self.assertLessEqual(tile.width * tile.height, 256 * 256)

    def test_saved_jpeg_crop_matches_full_decoded_image_for_all_transforms(self):
        # JPEG is lossy. The oracle is the saved file's decoded pixels, never the
        # uncompressed source image or an unrelated WSI extraction.
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "region.jpg"
            source = np.random.default_rng(113).integers(0, 256, (64, 96, 3), dtype=np.uint8)
            Image.fromarray(source).save(path, quality=90)
            with Image.open(path) as image:
                decoded = image.convert("RGB")
                for horizontal, vertical, rotation in itertools.product((False, True), (False, True), range(4)):
                    params = TransformParams(horizontal, vertical, rotation)
                    full = apply_transform(decoded, params)
                    width, height = full.size
                    boxes = ((0, 0, 1, 1), (height - 1, width - 1, 1, 1),
                             (3, 5, 37, 52), (height - 19, width - 25, 19, 25))
                    for box in boxes:
                        top, left, crop_height, crop_width = box
                        with self.subTest(params=params, box=box):
                            actual = crop_transformed(image, box, params).convert("RGB")
                            expected = full.crop((left, top, left + crop_width, top + crop_height))
                            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_512_jpeg_rgb_normalization_and_transforms_use_decoded_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "native512.jpg"
            source = np.random.default_rng(127).integers(0, 256, (512, 512), dtype=np.uint8)
            Image.fromarray(source).save(path, quality=90)
            with Image.open(path) as image:
                decoded = image.convert("RGB")
                for horizontal, vertical, rotation in itertools.product((False, True), (False, True), range(4)):
                    params = TransformParams(horizontal, vertical, rotation)
                    actual = load_images([AugmentedPath(path, params)], 512)
                    pixels = np.array(apply_transform(decoded, params), dtype=np.uint8)
                    expected = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float().div(255)
                    with self.subTest(params=params):
                        self.assertEqual(actual.shape, (1, 3, 512, 512))
                        self.assertEqual(actual.dtype, torch.float32)
                        self.assertGreaterEqual(actual.min().item(), 0)
                        self.assertLessEqual(actual.max().item(), 1)
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
