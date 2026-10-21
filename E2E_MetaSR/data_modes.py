"""Explicit image contracts; an omitted mode retains the legacy input path."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DataSpec:
    mode: str
    lr_size: int
    hr_size: int
    scale: int

    def metadata(self):
        return asdict(self)


def data_spec(config):
    mode = config.get("data_mode", "legacy256")
    if mode == "legacy256":
        return DataSpec(mode, 256, 8192, 32)
    if mode == "native512":
        return DataSpec(mode, 512, 8192, 16)
    raise ValueError("data_mode must be legacy256 or native512")


def validate_data_config(config):
    spec = data_spec(config)
    if spec.mode == "legacy256" and "data_manifest" in config:
        raise ValueError("data_manifest is only supported by explicit native512 mode")
    if config["metasr"]["scale"] != spec.scale or config["metasr"]["rgb_range"] != 1:
        raise ValueError(f"{spec.mode} requires {spec.lr_size} -> {spec.hr_size}, "
                         f"scale={spec.scale}, rgb_range=1")
    return spec


def checkpoint_data_spec(saved):
    config = saved["config"]
    spec = data_spec(config)
    metadata = saved.get("data_spec")
    if spec.mode == "native512" or "data_mode" in config or metadata is not None:
        validate_data_config(config)
        if metadata != spec.metadata():
            raise ValueError("Checkpoint data_spec is missing or disagrees with its data mode")
    # Historical model-only checkpoints also support small synthetic scales.
    # No new restrictions are imposed on their existing restoration path.
    return spec
