"""Additive region-gated classifier and checkpoint loader; legacy modules stay intact."""

from data_modes import checkpoint_data_spec
from models.baseline import MetaSRABMIL, model_from_checkpoint as legacy_model_from_checkpoint
from models.spp_region_gate import RegionAwareAdaptiveSPP


CHECKPOINT_FORMAT = "metasr-abmil-region-gate-v1"


class RegionGateMetaSRABMIL(MetaSRABMIL):
    def __init__(self, classification_encoder="spp_region_gate", classification_head="linear", **metasr):
        if classification_encoder != "spp_region_gate" or classification_head != "linear":
            raise ValueError("Region-gate V1 requires spp_region_gate and a linear classification head")
        super().__init__(classification_encoder="spp", classification_head="linear", **metasr)
        self.region_encoder = RegionAwareAdaptiveSPP()
        self.classification_encoder = classification_encoder

    def architecture(self):
        return dict(super().architecture(), region_gate_version=1)


def model_from_checkpoint(saved, device="cpu"):
    """Strictly restore V1, or delegate old formats to their unchanged loader."""
    if saved.get("format") != CHECKPOINT_FORMAT:
        return legacy_model_from_checkpoint(saved, device)
    checkpoint_data_spec(saved)
    config = saved["config"]
    model = RegionGateMetaSRABMIL(
        classification_encoder=config["classification_encoder"],
        classification_head=config.get("classification_head", "linear"),
        **config["metasr"],
    )
    if saved["architecture"] != model.architecture():
        raise ValueError("Checkpoint architecture does not match region-gate V1")
    model.load_state_dict(saved["model_state"], strict=True)
    return model.to(device)
