"""Opt-in MeanMIL/ABMIL version; existing models and checkpoint loaders stay intact."""

from data_modes import checkpoint_data_spec
from models.baseline import MetaSRABMIL, model_from_checkpoint as legacy_model_from_checkpoint
from models.meanmil import MeanMIL


CHECKPOINT_FORMAT = "metasr-mil-v1"


class MetaSRMIL(MetaSRABMIL):
    def __init__(self, classification_encoder="resnet18", classification_head="linear",
                 mil_pooling="abmil", **metasr):
        if mil_pooling not in ("abmil", "meanmil"):
            raise ValueError("mil_pooling must be abmil or meanmil")
        region_gate = classification_encoder == "spp_region_gate"
        if region_gate and classification_head != "linear":
            raise ValueError("Region-gate V1 requires classification_head=linear")
        super().__init__(classification_encoder="spp" if region_gate else classification_encoder,
                         classification_head=classification_head, **metasr)
        if region_gate:
            from models.spp_region_gate import RegionAwareAdaptiveSPP

            self.region_encoder = RegionAwareAdaptiveSPP()
            self.classification_encoder = classification_encoder
        self.mil_pooling = mil_pooling
        self._mil_architecture = super().architecture()
        if region_gate:
            self._mil_architecture["region_gate_version"] = 1
        if mil_pooling == "meanmil":
            # Initialize shared weights and consume RNG exactly as in the ABMIL version.
            self.mil_head = MeanMIL()
            self._mil_architecture.pop("attention_hidden_dim")

    def architecture(self):
        return dict(self._mil_architecture, mil_pooling=self.mil_pooling)


def model_from_checkpoint(saved, device="cpu"):
    """Strictly load the selected pooling, or use the original legacy loader."""
    if saved.get("format") != CHECKPOINT_FORMAT:
        if saved.get("format") == "metasr-abmil-region-gate-v1":
            from models.region_gate_model import model_from_checkpoint as load_region_gate

            return load_region_gate(saved, device)
        return legacy_model_from_checkpoint(saved, device)
    checkpoint_data_spec(saved)
    config = saved["config"]
    model = MetaSRMIL(classification_encoder=config["classification_encoder"],
                      classification_head=config.get("classification_head", "linear"),
                      mil_pooling=config["mil_pooling"], **config["metasr"])
    if saved["architecture"] != model.architecture():
        raise ValueError("Checkpoint architecture disagrees with the selected MIL configuration")
    model.load_state_dict(saved["model_state"], strict=True)
    return model.to(device)
