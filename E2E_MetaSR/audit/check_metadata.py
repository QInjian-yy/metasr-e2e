"""Real CSV metadata audit; missing image files are explicitly reported, never decoded."""
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wsi_data import load_fold_datasets, collate_one_wsi
from torch.utils.data import DataLoader


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


labels = read(ROOT / "downstream_train/camelyon16_labels.csv")
manifest = read(ROOT / "manifests/patch_manifest.csv")
by_slide = {row["slide_id"]: row for row in labels}
regions = defaultdict(list)
for row in manifest:
    regions[row["slide_id"]].append(row)
assert len({r["filename"] for r in manifest}) == len(manifest)
assert set(by_slide) == set(regions)
prefix_errors, pair_name_errors = [], []
for row in labels:
    expected = 0 if row["slide_id"].startswith("normal_") else 1 if row["slide_id"].startswith("tumor_") else None
    if expected != int(row["label"]):
        prefix_errors.append(row)
for row in manifest:
    if Path(row["filename"]).stem != f"{row['slide_id']}_{row['x']}_{row['y']}":
        pair_name_errors.append(row)
assert not prefix_errors and not pair_name_errors

result = {"real_metadata": True, "images_decoded": 0, "slides": len(labels), "regions": len(manifest),
          "label_counts": dict(Counter(r["label"] for r in labels)),
          "region_count_distribution": dict(sorted(Counter(len(v) for v in regions.values()).items())),
          "prefix_label_errors": prefix_errors, "coordinate_filename_errors": pair_name_errors,
          "copied_metadata_identical_to_E2E": {}, "image_directories": {}, "folds": []}
for folder in ("manifests", "downstream_train"):
    for path in (ROOT / folder).glob("*.csv"):
        other = ROOT.parent / "E2E" / folder / path.name
        equal = path.read_bytes() == other.read_bytes()
        assert equal
        result["copied_metadata_identical_to_E2E"][f"{folder}/{path.name}"] = equal
for base in (ROOT, ROOT.parent / "E2E"):
    for folder in ("images_256", "images_8192"):
        result["image_directories"][str(base / folder)] = (base / folder).is_dir()
try:
    load_fold_datasets(ROOT, 0)
    result["unmodified_loader"] = "passed"
except FileNotFoundError as exc:
    result["unmodified_loader"] = str(exc)

real_is_file = Path.is_file
expected_image_paths = {ROOT / folder / r["filename"] for folder in ("images_256", "images_8192") for r in manifest}
def metadata_only_exists(path):
    return True if path in expected_image_paths else real_is_file(path)

for fold in (0, 1, 2):
    # Only bypass missing image existence for exact manifest-derived paths.
    # This is a metadata-only probe, not a successful real-image loader run.
    with patch.object(Path, "is_file", metadata_only_exists):
        train, val = load_fold_datasets(ROOT, fold)
    samples = train.samples + val.samples
    for sample in samples:
        expected_names = [r["filename"] for r in regions[sample["slide_id"]]]
        assert [p.name for p in sample["lr_paths"]] == [p.name for p in sample["hr_paths"]] == expected_names
        assert sample["n_regions"] == len(expected_names)
        assert sample["label"] == int(by_slide[sample["slide_id"]]["label"])
    train_cases, val_cases = {s["case_id"] for s in train}, {s["case_id"] for s in val}
    assert not train_cases.intersection(val_cases)
    selected = []
    for label in (0, 1):
        group = sorted([s for s in train if s["label"] == label], key=lambda s: s["n_regions"])
        selected += [group[0], group[-1]]
    dry_rows = []
    for sample in DataLoader(selected, batch_size=1, collate_fn=collate_one_wsi):
        dry_rows.append({key: sample[key] for key in ("slide_id", "case_id", "label", "n_regions")}
                        | {"first_lr": str(sample["lr_paths"][0]), "first_hr": str(sample["hr_paths"][0]),
                           "last_filename": sample["lr_paths"][-1].name})
    result["folds"].append({"fold": fold, "train_slides": len(train), "val_slides": len(val),
                            "case_overlap": [], "metadata_only_dry_run": dry_rows})
result["pair_identity_and_order_passed"] = True
result["pixel_content_and_image_pair_alignment_verified"] = False
(ROOT / "audit/metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2), flush=True)
