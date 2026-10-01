"""WSI bags from CAMELYON16 manifests, labels and downstream cross-validation CSVs.

Train/val assignment uses ONLY ``downstream_train/splits_<fold>.csv`` (case-level).
Legacy SR-stage columns such as ``patch_manifest.csv:split`` or ``slide_manifest.csv``
are ignored here; they must not influence E2E training or validation.
"""

import csv
import copy
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from augmentation import (AugmentationConfig, AugmentedPath, apply_to_pair,
                          apply_transform, generate_transform_params)


DEFAULT_DATA_ROOT = (
    "/media/ub/shidang_plu/MT/py_project/yujian/double_wsi/"
    "CAMELYON16_WSI_DATA/c16_continuoussr_l1_tissue35_nocap_train_biopsy"
)


def _read_csv(path, required):
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError("{} requires CSV columns {}".format(path, required))
        return [
            {key: value.strip() if value is not None else "" for key, value in row.items()}
            for row in reader
        ]


def _label_path(downstream, explicit):
    if explicit is not None:
        return Path(explicit)
    default = downstream / "camelyon16_labels.csv"
    if default.is_file():
        return default
    candidates = []
    for path in sorted(downstream.rglob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            columns = next(csv.reader(handle), [])
        if {"case_id", "slide_id", "label"}.issubset(columns):
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(
            "Expected one labels CSV below {}; found {}. Set --labels-csv explicitly."
            .format(downstream, candidates)
        )
    return candidates[0]


def discover_fold_ids(split_directory):
    """Return sorted fold ids from existing ``splits_<id>.csv`` files."""
    directory = Path(split_directory)
    folds = []
    for path in sorted(directory.glob("splits_*.csv")):
        suffix = path.stem.removeprefix("splits_")
        if suffix.isdigit():
            folds.append(int(suffix))
    if not folds:
        raise FileNotFoundError("No splits_*.csv found in {}".format(directory))
    return folds


def _split_directory(downstream, explicit):
    if explicit is not None:
        directory = Path(explicit)
    elif (downstream / "splits_0.csv").is_file():
        directory = downstream
    else:
        candidates = sorted({path.parent for path in downstream.rglob("splits_0.csv")})
        if len(candidates) != 1:
            raise ValueError(
                "Expected one cross-validation directory below {}; found {}. "
                "Set --split-dir explicitly.".format(downstream, candidates)
            )
        directory = candidates[0]
    discover_fold_ids(directory)
    return directory


class Camelyon16WSI(Dataset):
    """One item holds all region paths for one WSI; no image is eagerly decoded."""

    def __init__(self, samples, augmentation=None):
        self.samples = samples
        self.augmentation = AugmentationConfig(**(augmentation or {}))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if not self.augmentation.enabled:
            return sample
        # Keep only paths/parameters: the engine still decodes one micro-batch.
        lr_paths, hr_paths = [], []
        for lr, hr in zip(sample["lr_paths"], sample["hr_paths"]):
            params = generate_transform_params(self.augmentation)
            lr, hr = apply_to_pair(lr, hr, params)
            lr_paths.append(lr)
            hr_paths.append(hr)
        return dict(sample, lr_paths=lr_paths, hr_paths=hr_paths)

    def evaluation_view(self):
        """Share immutable bag metadata, with no augmentation or random draws."""
        dataset = copy.copy(self)
        dataset.augmentation = AugmentationConfig()
        return dataset


def load_fold_datasets(data_root, fold, labels_csv=None, split_dir=None, *, augmentation=None, require_hr=True):
    """Return (train, val) for one fold.

    Regions come from ``manifests/patch_manifest.csv`` (filename + slide_id only).
    Case train/val comes from ``downstream_train/splits_<fold>.csv`` only.
    Augmentation is opt-in for training use; a train split alone never enables it.
    """
    root = Path(data_root)
    downstream = root / "downstream_train"
    label_path = _label_path(downstream, labels_csv)
    split_directory = _split_directory(downstream, split_dir)
    available_folds = discover_fold_ids(split_directory)
    if fold not in available_folds:
        raise ValueError("fold must be one of {}; got {}".format(available_folds, fold))
    label_rows = _read_csv(label_path, ("case_id", "slide_id", "label"))
    by_slide, by_case = {}, defaultdict(list)
    for row in label_rows:
        slide_id, case_id = row["slide_id"], row["case_id"]
        if not slide_id or not case_id or row["label"] not in ("0", "1"):
            raise ValueError("Invalid C16 label row in {}: {} (Normal=0, Tumor=1)"
                             .format(label_path, row))
        if slide_id in by_slide:
            raise ValueError("Duplicate slide_id in labels: {}".format(slide_id))
        record = {"slide_id": slide_id, "case_id": case_id, "label": int(row["label"])}
        by_slide[slide_id] = record
        by_case[case_id].append(record)
    if not by_slide:
        raise ValueError("No real WSI labels in {}".format(label_path))

    # patch_manifest may carry a legacy SR ``split`` column; E2E ignores it entirely.
    manifest = root / "manifests" / "patch_manifest.csv"
    regions, filenames = defaultdict(list), set()
    for row in _read_csv(manifest, ("filename", "slide_id")):
        slide_id, filename = row["slide_id"], row["filename"]
        if not slide_id or not filename or "/" in filename or "\\" in filename:
            raise ValueError("Invalid manifest slide_id/filename: {}".format(row))
        if filename in filenames:
            raise ValueError("Duplicate region filename in manifest: {}".format(filename))
        filenames.add(filename)
        regions[slide_id].append((root / "images_256" / filename,
                                  root / "images_8192" / filename))
    missing_labels = set(regions) - set(by_slide)
    missing_regions = set(by_slide) - set(regions)
    if missing_labels:
        raise ValueError("Manifest WSIs missing real labels: {}".format(sorted(missing_labels)))
    if missing_regions:
        raise ValueError("Labelled WSIs missing manifest regions: {}".format(sorted(missing_regions)))

    split_path = split_directory / "splits_{}.csv".format(fold)
    split_rows = _read_csv(split_path, ("train", "val"))
    split_cases = {name: [row[name] for row in split_rows if row[name]]
                   for name in ("train", "val")}
    for name, cases in split_cases.items():
        if not cases:
            if name == "val":
                continue
            raise ValueError("{} {} split is empty or has duplicate cases".format(split_path, name))
        if len(cases) != len(set(cases)):
            raise ValueError("{} {} split has duplicate cases".format(split_path, name))
        unknown = set(cases) - set(by_case)
        if unknown:
            raise ValueError("{} references unknown cases: {}".format(split_path, sorted(unknown)))
    overlap = set(split_cases["train"]) & set(split_cases["val"])
    if overlap:
        raise ValueError("Train/val case overlap: {}".format(sorted(overlap)))
    omitted = set(by_case) - set(split_cases["train"]) - set(split_cases["val"])
    if omitted:
        raise ValueError("Fold omits labelled cases: {}".format(sorted(omitted)))

    datasets = []
    for name in ("train", "val"):
        samples = []
        for case_id in split_cases[name]:
            for label in by_case[case_id]:
                pairs = regions[label["slide_id"]]
                for lr_path, hr_path in pairs:
                    for path in ((lr_path, hr_path) if require_hr else (lr_path,)):
                        if not path.is_file():
                            raise FileNotFoundError(
                                "Missing region file for WSI {} (all {} regions required): {}"
                                .format(label["slide_id"], len(pairs), path)
                            )
                samples.append(dict(label, lr_paths=[pair[0] for pair in pairs],
                                    hr_paths=[pair[1] for pair in pairs], n_regions=len(pairs)))
        dataset = Camelyon16WSI(samples, augmentation=augmentation if name == "train" else None)
        dataset.labels_csv = label_path
        dataset.split_csv = split_path
        datasets.append(dataset)
    return tuple(datasets)


def collate_one_wsi(batch):
    if len(batch) != 1:
        raise ValueError("Use DataLoader batch_size=1: each sample is one complete WSI")
    return batch[0]


def load_images(paths, size):
    """Decode only the current micro-batch as CPU float32 RGB [m,3,size,size]."""
    if not paths:
        raise ValueError("Image micro-batch cannot be empty")
    batch = torch.empty((len(paths), 3, size, size), dtype=torch.float32)
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            if image.size != (size, size):
                raise ValueError("{} has size {}; expected ({}, {}), no resize allowed"
                                 .format(path, image.size, size, size))
            if isinstance(path, AugmentedPath):
                image = apply_transform(image, path.params)
            array = np.array(image.convert("RGB"), dtype=np.uint8)
        batch[index].copy_(torch.from_numpy(array).permute(2, 0, 1))
    return batch.div_(255.0)
