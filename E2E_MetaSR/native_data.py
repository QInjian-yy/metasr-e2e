"""Validate the explicit level-1/level-5 native export without resizing images."""

import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath

from PIL import Image


NATIVE_COLUMNS = ("slide_id", "x", "y", "hr_level", "lr_level", "scale", "hr_path", "lr_path")


def reject_native_legacy(root, rows):
    """Reject explicit native512 evidence while leaving historical extra metadata alone."""
    for row in rows:
        lr_path = row.get("lr_path", "")
        parts = PurePosixPath(lr_path.replace("\\", "/")).parts if isinstance(lr_path, str) else ()
        geometry = tuple(str(row.get(key, "")) for key in ("hr_level", "lr_level", "scale"))
        if (parts and parts[0] == "images_512") or geometry == ("1", "5", "16"):
            raise ValueError("legacy256 cannot load a native512 manifest; select data_mode=native512")
    try:
        report = json.loads((Path(root) / "verification" / "report.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return  # Legacy loading historically ignores optional or unrelated verification files.
    if isinstance(report, dict) and all(report.get(key) == value for key, value in
                                        {"hr_level": 1, "lr_level": 5, "lr_size": 512, "scale": 16}.items()):
        raise ValueError("legacy256 cannot load a native512 dataset verification report; "
                         "select data_mode=native512")


def native_manifest_path(root, relative):
    """Resolve only the explicitly configured, case-sensitive native CSV source."""
    root = Path(root).resolve()
    value = str(relative)
    parts = PurePosixPath(value)
    if (not value or "\\" in value or ":" in value or PureWindowsPath(value).drive or parts.is_absolute()
            or ".." in parts.parts or parts.suffix.lower() != ".csv"):
        raise ValueError("native512 data_manifest must be a safe relative CSV path: {}".format(value))
    path = root.joinpath(*parts.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError("native512 data_manifest escapes data-root (including symlinks): {}".format(value))
    return path


def _image_path(root, value, directory, check_file):
    relative = PurePosixPath(value)
    if (not value or "\\" in value or ":" in value or PureWindowsPath(value).drive or relative.is_absolute()
            or ".." in relative.parts or len(relative.parts) < 2
            or relative.parts[0] != directory or relative.suffix.lower() not in {".jpg", ".jpeg", ".png"}):
        raise ValueError("native512 requires a safe relative {} image path: {}".format(directory, value))
    path = root.joinpath(*relative.parts)
    # SR-off must not stat, resolve or open HR paths. Its lexical contract is still checked.
    if check_file:
        resolved = path.resolve()
        if not resolved.is_relative_to(root / directory):
            raise ValueError("native512 image path escapes {} (including symlinks): {}".format(directory, value))
        if not path.is_file():
            raise FileNotFoundError("Missing native512 region image: {}".format(path))
    return path


def _check_size(path, size):
    with Image.open(path) as image:
        if image.size != (size, size):
            raise ValueError("{} has size {}; native512 expected ({}, {}), no resize allowed"
                             .format(path, image.size, size, size))


def _report_slides(report):
    contract = {"status": "complete", "hr_level": 1, "lr_level": 5,
                "hr_size": 8192, "lr_size": 512, "scale": 16}
    for key, expected in contract.items():
        if report.get(key) != expected:
            raise ValueError("native512 verification report requires {}={!r}; got {!r}"
                             .format(key, expected, report.get(key)))
    if report.get("sampling") != {"method": "tissue_windows_nonoverlap_v2", "overlap_allowed": False}:
        raise ValueError("native512 verification report requires tissue_windows_nonoverlap_v2 without overlap")
    entries = report.get("slides")
    if not isinstance(entries, list) or not entries:
        raise ValueError("native512 verification report requires per-slide metadata")
    slides = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("native512 verification report requires per-slide JSON objects")
        slide_id, pairs = entry.get("slide_id"), entry.get("pairs")
        if not isinstance(slide_id, str) or not slide_id or slide_id in slides:
            raise ValueError("Invalid or duplicate slide_id in native512 verification report")
        if type(pairs) is not int or pairs < 0:
            raise ValueError("Invalid native512 report pairs for {}".format(slide_id))
        slides[slide_id] = entry
        if not pairs:
            continue
        downsamples, dimensions = entry.get("level_downsamples"), entry.get("level_dimensions")
        if (not isinstance(downsamples, list) or len(downsamples) <= 5
                or downsamples[1] != 2 or downsamples[5] != 32
                or not isinstance(dimensions, list) or len(dimensions) <= 5
                or any(not isinstance(dimensions[level], list) or len(dimensions[level]) != 2
                       or any(type(value) is not int or value <= 0 for value in dimensions[level])
                       for level in (1, 5))
                or any(high != low * 16 for high, low in zip(dimensions[1], dimensions[5]))
                or entry.get("patch_size_level0") != 16384):
            raise ValueError("native512 report has incompatible native pyramid geometry for {}".format(slide_id))
    skipped = report.get("skipped_slides", [])
    zeros = report.get("zero_patch_slides", [])
    if (not isinstance(skipped, list) or not isinstance(zeros, list)
            or any(not isinstance(value, str) or not value for value in skipped + zeros)
            or len(skipped) != len(set(skipped)) or len(zeros) != len(set(zeros))
            or set(skipped) & set(slides)
            or set(zeros) != {slide_id for slide_id, entry in slides.items() if not entry["pairs"]}):
        raise ValueError("native512 report has inconsistent zero-patch or skipped slides")
    return slides


def load_native_regions(root, rows, require_hr=True):
    """Return complete ordered region pairs and their native verification source.

    ``rows`` must come from the detailed export CSV, with ``NATIVE_COLUMNS``.
    Coordinates are level-0 origins used only for identity checks here; SR crops
    use local HR pixels. JPEG source-pixel identity is intentionally not claimed.
    """
    root = Path(root).resolve()
    verification_path = root / "verification" / "report.json"
    try:
        with verification_path.open(encoding="utf-8-sig") as handle:
            report = json.load(handle)
    except FileNotFoundError as error:
        raise FileNotFoundError("native512 requires the original completed export verification report: {}"
                                .format(verification_path)) from error
    if not isinstance(report, dict):
        raise ValueError("native512 verification report must be a JSON object")
    slides = _report_slides(report)
    regions, coordinates, used_paths = defaultdict(list), set(), set()
    for row in rows:
        if not set(NATIVE_COLUMNS).issubset(row):
            raise ValueError("native512 needs the detailed export manifest columns {}; "
                             "the derived filename manifest omits native paths/geometry".format(NATIVE_COLUMNS))
        slide_id = row["slide_id"]
        try:
            x, y = int(row["x"]), int(row["y"])
            geometry = tuple(int(row[key]) for key in ("hr_level", "lr_level", "scale"))
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid native512 manifest coordinates/levels: {}".format(row)) from error
        if not slide_id or x < 0 or y < 0 or geometry != (1, 5, 16):
            raise ValueError("native512 requires slide_id, nonnegative level-0 coordinates and levels 1/5/scale16")
        if slide_id not in slides or not slides[slide_id]["pairs"]:
            raise ValueError("native512 manifest WSI absent or zero-patch in report: {}".format(slide_id))
        identity = (slide_id, x, y)
        if identity in coordinates:
            raise ValueError("Duplicate native512 region coordinates: {}".format(identity))
        coordinates.add(identity)
        lr = _image_path(root, row["lr_path"], "images_512", True)
        hr = _image_path(root, row["hr_path"], "images_8192", require_hr)
        if lr.name != hr.name:
            raise ValueError("native512 LR/HR paired filenames differ: {} / {}".format(lr, hr))
        for path, checked in ((lr, True), (hr, require_hr)):
            identity_path = path.resolve() if checked else path
            if identity_path in used_paths:
                raise ValueError("Duplicate or reused native512 region image: {}".format(path))
            used_paths.add(identity_path)
        _check_size(lr, 512)
        if require_hr:
            _check_size(hr, 8192)
        regions[slide_id].append((lr, hr))
    expected = Counter({slide_id: entry["pairs"] for slide_id, entry in slides.items() if entry["pairs"]})
    if Counter({slide_id: len(pairs) for slide_id, pairs in regions.items()}) != expected:
        raise ValueError("native512 manifest region counts differ from the completed verification report")
    return regions, verification_path
