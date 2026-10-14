import itertools
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from augmentation import TransformParams, apply_transform
from paired_geometry import crop_transformed


class PairedGeometryTests(unittest.TestCase):
    def test_all_rectangles_match_full_transform_with_unique_pixel_coordinates(self):
        image = Image.fromarray(np.arange(15, dtype=np.int32).reshape(3, 5))
        for horizontal, vertical, k in itertools.product((False, True), (False, True), range(4)):
            params = TransformParams(horizontal, vertical, k)
            transformed = apply_transform(image, params)
            width, height = transformed.size
            for top in range(height):
                for left in range(width):
                    for bottom in range(top + 1, height + 1):
                        for right in range(left + 1, width + 1):
                            box = (top, left, bottom - top, right - left)
                            with self.subTest(params=params, box=box):
                                actual = crop_transformed(image, box, params)
                                expected = transformed.crop((left, top, right, bottom))
                                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_scale32_absolute_coordinates_match_transformed_lr_at_8k_boundaries(self):
        # A virtual 8K nearest-neighbor coordinate chart materializes only each
        # requested raw crop; the 256x256 LR chart is the independent oracle.
        class CoordinateImage:
            size = (8192, 8192)

            def crop(self, bounds):
                left, top, right, bottom = bounds
                rows = np.arange(top, bottom, dtype=np.int32)[:, None] // 32
                columns = np.arange(left, right, dtype=np.int32)[None, :] // 32
                return Image.fromarray(rows * 256 + columns)

        lr = Image.fromarray(np.arange(256 * 256, dtype=np.int32).reshape(256, 256))
        boxes = ((0, 0, 1, 1), (8191, 8191, 1, 1), (31, 17, 33, 71),
                 (7936, 7936, 256, 256), (8173, 0, 19, 25), (0, 8183, 31, 9))
        for horizontal, vertical, k in itertools.product((False, True), (False, True), range(4)):
            params = TransformParams(horizontal, vertical, k)
            transformed_lr = np.asarray(apply_transform(lr, params))
            for box in boxes:
                top, left, height, width = box
                expected = transformed_lr[
                    np.arange(top, top + height)[:, None] // 32,
                    np.arange(left, left + width)[None, :] // 32,
                ]
                with self.subTest(params=params, box=box), \
                        patch("paired_geometry.apply_transform", wraps=apply_transform) as transform:
                    actual = crop_transformed(CoordinateImage(), box, params)
                    np.testing.assert_array_equal(np.asarray(actual), expected)
                    self.assertEqual(transform.call_count, 1)
                    tile = transform.call_args.args[0]
                    self.assertEqual(tile.width * tile.height, width * height)

    def test_disabled_transform_is_identical_to_raw_crop(self):
        image = Image.fromarray(np.arange(3 * 7 * 11, dtype=np.uint8).reshape(7, 11, 3))
        box = (1, 3, 5, 8)
        actual = crop_transformed(image, box, TransformParams(False, False, 0))
        expected = image.crop((3, 1, 11, 6))
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_rejects_invalid_boxes_in_transformed_coordinate_space(self):
        image = Image.new("RGB", (7, 3))
        for horizontal, vertical, k in itertools.product((False, True), (False, True), range(4)):
            params = TransformParams(horizontal, vertical, k)
            width, height = apply_transform(image, params).size
            boxes = ((-1, 0, 1, 1), (0, -1, 1, 1), (0, 0, 0, 1), (0, 0, 1, 0),
                     (0, 0, -1, 1), (0, 0, 1, -1), (height, 0, 1, 1),
                     (0, width, 1, 1), (0, 0, height + 1, 1), (0, 0, 1, width + 1))
            for box in boxes:
                with self.subTest(params=params, box=box), self.assertRaises(ValueError):
                    crop_transformed(image, box, params)


if __name__ == "__main__":
    unittest.main()
