"""Lazy, paired CPU geometry; parameters are fixed before any GradPool pass."""

import random
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = False
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.5
    rotation90: bool = True

    def __post_init__(self):
        for name in ("enabled", "rotation90"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError("augmentation.{} must be a boolean".format(name))
        for name in ("horizontal_flip", "vertical_flip"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not 0 <= value <= 1):
                raise ValueError("augmentation.{} must be a probability in [0, 1]".format(name))


def load_augmentation_config(path=None):
    if path is None:
        return AugmentationConfig()
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("augmentation"), dict):
        raise ValueError("--augmentation-config requires a top-level 'augmentation' mapping")
    return AugmentationConfig(**config["augmentation"])


@dataclass(frozen=True)
class TransformParams:
    horizontal_flip: bool
    vertical_flip: bool
    rotation_k: int


def generate_transform_params(config):
    """Draw once per pair per WSI access, never while decoding or replaying."""
    if not config.enabled:
        return TransformParams(False, False, 0)
    return TransformParams(
        random.random() < config.horizontal_flip,
        random.random() < config.vertical_flip,
        random.randrange(4) if config.rotation90 else 0,
    )


@dataclass(frozen=True)
class AugmentedPath:
    path: Path
    params: TransformParams

    def __fspath__(self):
        return str(self.path)


def apply_to_pair(lr, hr, params):
    """Bind identical geometry to two paths; decoding stays micro-batch lazy."""
    return AugmentedPath(lr, params), AugmentedPath(hr, params)


def apply_transform(image, params):
    """PIL pixel permutations only, before float conversion or GPU transfer."""
    if params.horizontal_flip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if params.vertical_flip:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if params.rotation_k:
        rotations = {1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180,
                     3: Image.Transpose.ROTATE_270}
        image = image.transpose(rotations[params.rotation_k])
    return image
