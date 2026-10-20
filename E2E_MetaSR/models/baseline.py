"""Shared official RDN features -> Meta-Upscale and SPP or ResNet18-GN/ABMIL."""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18

from models.abmil import GatedAttentionMIL
from models.metasr import MemoryEfficientMetaRDN
from models.spp_1344 import SpatialPyramidPooling1344
from models.spp_avg import AverageSpatialPyramidPooling
from models.spp_ln import LayerNormalizedSPP


def _replace_bn_with_gn(module, num_groups=32):
    # Verbatim helper from E2E/downstream/shared_model.py, enabled by user request.
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            channels = child.num_features
            groups = min(num_groups, channels)
            while channels % groups:
                groups -= 1
            setattr(module, name, nn.GroupNorm(groups, channels))
        else:
            _replace_bn_with_gn(child, num_groups)


class SpatialPyramidPooling(nn.Module):
    def forward(self, feature):
        return torch.cat([F.adaptive_max_pool2d(feature, (level, level)).flatten(start_dim=1)
                          for level in (1, 2, 3, 6)], dim=1)


class MetaSRABMIL(nn.Module):
    def __init__(self, classification_encoder="resnet18", classification_head="linear", **metasr):
        super().__init__()
        if classification_encoder not in ("resnet18", "spp", "spp_ln", "spp_avg", "spp_1344"):
            raise ValueError(f"Unknown classification_encoder: {classification_encoder}")
        if classification_head not in ("linear", "mlp"):
            raise ValueError(f"Unknown classification_head: {classification_head}")
        if classification_head == "mlp" and classification_encoder != "spp":
            raise ValueError("classification_head=mlp requires classification_encoder=spp")
        self.classification_encoder = classification_encoder
        self.classification_head = classification_head
        self.sr = MemoryEfficientMetaRDN(**metasr)
        if classification_encoder == "spp_1344":
            self.region_encoder = SpatialPyramidPooling1344()
            input_dim = 64 * (1 + 4 + 16)
        elif classification_encoder == "spp_avg":
            self.region_encoder = AverageSpatialPyramidPooling()
            input_dim = 64 * (1 + 4 + 9 + 36)
        elif classification_encoder in ("spp", "spp_ln"):
            self.region_encoder = SpatialPyramidPooling()
            if classification_encoder == "spp_ln":
                self.region_encoder = LayerNormalizedSPP(self.region_encoder)
            input_dim = 64 * (1 + 4 + 9 + 36)
        else:
            # Copied from E2E/downstream/shared_model.py:SharedE2EModel.__init__.
            self.region_encoder = resnet18(weights=None)
            self.region_encoder.conv1 = nn.Conv2d(64, 64, 7, stride=2, padding=3, bias=False)
            self.region_encoder.fc = nn.Identity()
            _replace_bn_with_gn(self.region_encoder)
            input_dim = 512
        if classification_head == "linear":
            self.classifier = nn.Linear(input_dim, 2)
        else:
            self.classifier = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(),
                                            nn.Dropout(0.3), nn.Linear(128, 2))
        self.mil_head = GatedAttentionMIL(input_dim=input_dim, hidden_dim=128)

    def architecture(self):
        output = self.classifier if self.classification_head == "linear" else self.classifier[-1]
        architecture = {"classification_encoder": self.classification_encoder,
                        "embedding_dim": self.mil_head.attention_V.in_features,
                        "attention_hidden_dim": self.mil_head.attention_V.out_features,
                        "num_classes": output.out_features}
        if self.classification_head == "mlp":
            architecture.update(classification_head="mlp", classifier_hidden_dim=128,
                                classifier_dropout=self.classifier[2].p)
        return architecture

    def encode_regions(self, lr):
        feature = self.sr.extract_features(lr)
        return feature, self.region_encoder(feature)

    def forward_embeddings(self, embeddings):
        pooled, _ = self.mil_head(embeddings)
        return self.classifier(pooled)

    def forward(self, lr):
        _, embeddings = self.encode_regions(lr)
        return self.forward_embeddings(embeddings)

    def joint_loss(self, lr, target, *, lambda_sr=0.1, hr_crop=None, box=None):
        if lambda_sr < 0:
            raise ValueError("lambda_sr must be nonnegative")
        feature, embeddings = self.encode_regions(lr)
        logits = self.forward_embeddings(embeddings)
        cls_loss = F.cross_entropy(logits, target)
        sr_loss = cls_loss.new_zeros(())
        if lambda_sr:
            if hr_crop is None or box is None:
                raise ValueError("SR supervision requires an HR crop and its coordinates")
            prediction = self.sr.decode_crop(feature, box)
            if prediction.shape != hr_crop.shape:
                raise ValueError("Predicted SR crop and GT crop must have identical shapes")
            sr_loss = F.l1_loss(prediction.float(), hr_crop.float())
        return cls_loss + lambda_sr * sr_loss, cls_loss, sr_loss, logits


def model_from_checkpoint(saved, device="cpu"):
    """Construct the saved architecture and strictly load every model parameter."""
    config = saved["config"]
    if saved.get("format") == "metasr-abmil-v1":
        classification_encoder = "resnet18"
        if (config.get("classification_encoder", "resnet18") != "resnet18"
                or config.get("classification_head", "linear") != "linear"):
            raise ValueError("Legacy metasr-abmil-v1 checkpoints require ResNet18 with a linear head")
    elif saved.get("format") == "metasr-abmil-v2":
        classification_encoder = saved["architecture"]["classification_encoder"]
        if config["classification_encoder"] != classification_encoder:
            raise ValueError("Checkpoint config and architecture disagree")
    else:
        raise ValueError(f"Unknown MetaSR checkpoint format: {saved.get('format')}")
    model = MetaSRABMIL(classification_encoder=classification_encoder,
                       classification_head=config.get("classification_head", "linear"),
                       **config["metasr"])
    if saved.get("format") == "metasr-abmil-v2" and saved["architecture"] != model.architecture():
        raise ValueError("Checkpoint architecture dimensions do not match the selected model")
    model.load_state_dict(saved["model_state"], strict=True)
    return model.to(device)
