"""Native export schema/security checks using CPU fixtures, never full 8K tensors."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_data import NATIVE_COLUMNS, load_native_regions, native_manifest_path, reject_native_legacy
from wsi_data import load_images


class Native512DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.addCleanup(self.temp.cleanup)
        self.rows = []
        for index, name in enumerate(("arbitrary_a.jpg", "arbitrary_b.png", "arbitrary_c.jpeg")):
            row = dict(slide_id="slide_{}".format(index // 2), x=str(17 + index * 16384), y="33",
                       hr_level="1", lr_level="5", scale="16",
                       hr_path="images_8192/" + name, lr_path="images_512/" + name)
            self.rows.append(row)
            lr, hr = self.root / row["lr_path"], self.root / row["hr_path"]
            lr.parent.mkdir(exist_ok=True)
            hr.parent.mkdir(exist_ok=True)
            Image.new("L", (512, 512), 128).save(lr)
            hr.write_bytes(b"header mocked without allocating an 8K image")
        geometry = dict(level_downsamples=[1., 2., 4., 8., 16., 32.],
                        level_dimensions=[[131072 // (2 ** level)] * 2 for level in range(6)],
                        patch_size_level0=16384, status="complete")
        self.report = dict(status="complete", hr_level=1, lr_level=5, scale=16, hr_size=8192,
                           lr_size=512, sampling=dict(method="tissue_windows_nonoverlap_v2", overlap_allowed=False),
                           slides=[dict(geometry, slide_id="slide_0", pairs=2),
                                   dict(geometry, slide_id="slide_1", pairs=1),
                                   dict(slide_id="empty", pairs=0)],
                           zero_patch_slides=["empty"], skipped_slides=["excluded"])
        self.report_path = self.root / "verification" / "report.json"
        self.report_path.parent.mkdir()
        self.write_report()
        original_open = Image.open

        def header_open(path, *args, **kwargs):
            if "images_8192" in Path(path).parts:
                image = MagicMock()
                image.size = (8192, 8192)
                image.__enter__.return_value = image
                return image
            return original_open(path, *args, **kwargs)

        self.image_open = patch("native_data.Image.open", side_effect=header_open).start()
        self.addCleanup(patch.stopall)

    def write_report(self):
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")

    def test_detailed_schema_grouping_order_and_rgb_jpeg(self):
        self.assertEqual(tuple(self.rows[0]), NATIVE_COLUMNS)
        regions, report_path = load_native_regions(self.root, self.rows)
        self.assertEqual(report_path, self.report_path)
        self.assertEqual(list(regions), ["slide_0", "slide_1"])
        self.assertEqual([pair[0].name for pair in regions["slide_0"]],
                         ["arbitrary_a.jpg", "arbitrary_b.png"])
        batch = load_images([regions["slide_0"][0][0]], 512)
        self.assertEqual(tuple(batch.shape), (1, 3, 512, 512))
        self.assertEqual(batch.dtype, torch.float32)
        torch.testing.assert_close(batch, torch.full_like(batch, 128 / 255.))
        with self.assertRaisesRegex(ValueError, "no resize allowed"):
            load_images([regions["slide_0"][0][0]], 256)

    def test_sr_off_never_stats_resolves_or_opens_hr(self):
        for row in self.rows:
            (self.root / row["hr_path"]).unlink()
        original_file, original_resolve = Path.is_file, Path.resolve

        def guarded_file(path):
            self.assertNotIn("images_8192", path.parts)
            return original_file(path)

        def guarded_resolve(path, *args, **kwargs):
            self.assertNotIn("images_8192", path.parts)
            return original_resolve(path, *args, **kwargs)

        with patch.object(Path, "is_file", guarded_file), patch.object(Path, "resolve", guarded_resolve):
            regions, _ = load_native_regions(self.root, self.rows, require_hr=False)
        self.assertEqual(sum(map(len, regions.values())), 3)
        self.assertTrue(all("images_8192" not in Path(call.args[0]).parts
                            for call in self.image_open.call_args_list))

    def test_missing_hr_only_rejected_for_sr_on(self):
        (self.root / self.rows[0]["hr_path"]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing native512"):
            load_native_regions(self.root, self.rows)
        load_native_regions(self.root, self.rows, require_hr=False)

    def test_missing_lr_and_wrong_size_fail_without_resize(self):
        path = self.root / self.rows[0]["lr_path"]
        Image.new("RGB", (256, 256)).save(path)
        with self.assertRaisesRegex(ValueError, "no resize allowed"):
            load_native_regions(self.root, self.rows, require_hr=False)
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing native512"):
            load_native_regions(self.root, self.rows, require_hr=False)

    def test_wrong_hr_header_rejected(self):
        self.image_open.side_effect = lambda *args, **kwargs: MagicMock(
            __enter__=lambda _self: type("Header", (), {"size": (512, 512)})())
        with self.assertRaisesRegex(ValueError, "8192"):
            load_native_regions(self.root, self.rows)

    def test_report_contract_and_pyramid_mismatch_rejected(self):
        baseline = copy.deepcopy(self.report)
        for key, value in (("status", "incomplete"), ("hr_level", 0), ("lr_level", 4),
                           ("scale", 32), ("lr_size", 256), ("hr_size", 4096),
                           ("sampling", dict(method="fixed_grid", overlap_allowed=False))):
            with self.subTest(key=key):
                self.report = dict(baseline, **{key: value})
                self.write_report()
                with self.assertRaisesRegex(ValueError, "native512 verification report"):
                    load_native_regions(self.root, self.rows)
        for key, value in (("level_downsamples", [1., 2., 4., 8., 16., 64.]),
                           ("level_dimensions", [[64, 64]] * 6), ("patch_size_level0", 8192)):
            with self.subTest(key=key):
                self.report = copy.deepcopy(baseline)
                self.report["slides"][0][key] = value
                self.write_report()
                with self.assertRaisesRegex(ValueError, "pyramid geometry"):
                    load_native_regions(self.root, self.rows)

    def test_missing_verification_and_derived_manifest_explained(self):
        with self.assertRaisesRegex(ValueError, "detailed export manifest"):
            load_native_regions(self.root, [dict(filename="a.jpg", slide_id="slide_0")])
        self.report_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "original completed export"):
            load_native_regions(self.root, self.rows)

    def test_manifest_geometry_duplicate_coordinates_and_reused_files(self):
        for key, value in (("slide_id", "unknown"), ("x", "-1"), ("y", "not-coordinate"),
                           ("hr_level", "0"), ("lr_level", "4"), ("scale", "32")):
            rows = copy.deepcopy(self.rows)
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_native_regions(self.root, rows)
        rows = self.rows + [dict(self.rows[0])]
        with self.assertRaisesRegex(ValueError, "Duplicate native512 region coordinates"):
            load_native_regions(self.root, rows)
        rows[-1]["x"] = "999"
        with self.assertRaisesRegex(ValueError, "reused native512 region image"):
            load_native_regions(self.root, rows)

    def test_manifest_counts_zero_and_skipped_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "region counts"):
            load_native_regions(self.root, self.rows[:2])
        for key, value in (("zero_patch_slides", []), ("skipped_slides", ["slide_0"])):
            with self.subTest(key=key):
                original = self.report[key]
                self.report[key] = value
                self.write_report()
                with self.assertRaisesRegex(ValueError, "zero-patch or skipped"):
                    load_native_regions(self.root, self.rows)
                self.report[key] = original

    def test_unsafe_paths_wrong_pair_and_file_extension_rejected(self):
        for value in ("../outside.jpg", "/images_512/a.jpg", "C:/images_512/a.jpg",
                      "images_512/../a.jpg", "images_512\\a.jpg", "images_256/a.jpg", "images_512/a.bmp"):
            rows = copy.deepcopy(self.rows)
            rows[0]["lr_path"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "safe relative"):
                load_native_regions(self.root, rows, require_hr=False)
        rows = copy.deepcopy(self.rows)
        rows[0]["hr_path"] = self.rows[1]["hr_path"]
        with self.assertRaisesRegex(ValueError, "filenames differ"):
            load_native_regions(self.root, rows, require_hr=False)

    def test_explicit_manifest_path_has_no_directory_guessing(self):
        expected = self.root / "Manifests" / "patch_manifest.csv"
        self.assertEqual(native_manifest_path(self.root, "Manifests/patch_manifest.csv"), expected)
        for value in ("../patch_manifest.csv", "/a.csv", "C:/a.csv", "Manifests\\a.csv", "a.json"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "safe relative CSV"):
                native_manifest_path(self.root, value)

    def test_legacy_rejects_native_detailed_or_derived_even_if_images256_exist(self):
        legacy_directory = self.root / "images_256"
        legacy_directory.mkdir()
        derived = []
        for row in self.rows:
            filename = Path(row["lr_path"]).name
            Image.new("RGB", (256, 256)).save(legacy_directory / filename)
            derived.append(dict(filename=filename, slide_id=row["slide_id"]))
        with self.assertRaisesRegex(ValueError, "native512 dataset verification"):
            reject_native_legacy(self.root, derived)
        self.report_path.unlink()
        detailed = [dict(row, filename=Path(row["lr_path"]).name) for row in self.rows]
        with self.assertRaisesRegex(ValueError, "native512 manifest"):
            reject_native_legacy(self.root, detailed)
        for evidence in (dict(derived[0], lr_path=self.rows[0]["lr_path"]),
                         dict(derived[0], hr_level="1", lr_level="5", scale="16")):
            with self.subTest(evidence=evidence), self.assertRaisesRegex(ValueError, "native512 manifest"):
                reject_native_legacy(self.root, [evidence])

    def test_legacy_keeps_ignoring_unrelated_or_unreadable_reports_and_extra_columns(self):
        rows = [dict(filename="old.png", slide_id="old", split="test", scale="32",
                     lr_path="images_256/old.png")]
        self.report_path.unlink()
        reject_native_legacy(self.root, rows)
        for report in ({"hr_level": 1, "lr_level": 6, "lr_size": 256, "scale": 32},
                       {"status": "complete", "unrelated": True}, ["unrelated"]):
            with self.subTest(report=report):
                self.report_path.write_text(json.dumps(report), encoding="utf-8")
                reject_native_legacy(self.root, rows)
        self.report_path.write_text("unrelated old non-JSON report", encoding="utf-8")
        reject_native_legacy(self.root, rows)

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            link = self.root / "images_512" / "escape.jpg"
            try:
                link.symlink_to(Path(outside) / "target.jpg")
            except OSError as error:
                self.fail("Cannot exercise symlink validation in this environment: {}".format(error))
            rows = copy.deepcopy(self.rows)
            rows[0]["lr_path"] = "images_512/escape.jpg"
            with self.assertRaisesRegex(ValueError, "escapes images_512"):
                load_native_regions(self.root, rows, require_hr=False)
            csv_link = self.root / "outside.csv"
            csv_link.symlink_to(Path(outside) / "target.csv")
            with self.assertRaisesRegex(ValueError, "escapes data-root"):
                native_manifest_path(self.root, "outside.csv")


if __name__ == "__main__":
    unittest.main()
