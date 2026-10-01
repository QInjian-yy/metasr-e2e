"""Shared official RDN features -> Meta-Upscale and E2E ResNet18-GN/ABMIL."""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18

from models.abmil import GatedAttentionMIL
from models.metasr import MemoryEfficientMetaRDN


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


class MetaSRABMIL(nn.Module):
    def __init__(self, **metasr):
        super().__init__()
        self.sr = MemoryEfficientMetaRDN(**metasr)
        # Copied from E2E/downstream/shared_model.py:SharedE2EModel.__init__.
        self.region_encoder = resnet18(weights=None)
        self.region_encoder.conv1 = nn.Conv2d(64, 64, 7, stride=2, padding=3, bias=False)
        self.region_encoder.fc = nn.Identity()
        _replace_bn_with_gn(self.region_encoder)
        self.classifier = nn.Linear(512, 2)
        self.mil_head = GatedAttentionMIL(input_dim=512, hidden_dim=128)

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
