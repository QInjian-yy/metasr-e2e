"""Region-aware Adaptive SPP V1: one input-dependent max/average gate per region."""

import torch
import torch.nn.functional as F
from torch import nn

from models.baseline import SpatialPyramidPooling
from models.spp_avg import AverageSpatialPyramidPooling


class RegionAwareAdaptiveSPP(nn.Module):
    """Map [B,64,H,W] to [B,3200], using the existing (1,2,3,6) SPPs."""

    def __init__(self):
        super().__init__()
        self.max_spp = SpatialPyramidPooling()
        self.avg_spp = AverageSpatialPyramidPooling()
        self.gate = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
        # Small nonzero weights start near 0.5 and allow upstream gate gradients
        # on the first backward pass, unlike a zero-initialized final weight.
        nn.init.normal_(self.gate[3].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.gate[3].bias)

    def get_alpha(self, feature):
        """Return differentiable [B,1] gates without storing previous forwards."""
        if (feature.ndim != 4 or feature.shape[1] != 64
                or min(feature.shape[0], feature.shape[2], feature.shape[3]) < 1):
            raise ValueError("Region-aware SPP expects nonempty features [B,64,H,W]")
        maximum = F.adaptive_max_pool2d(feature, 1).flatten(start_dim=1)
        average = F.adaptive_avg_pool2d(feature, 1).flatten(start_dim=1)
        return self.gate(torch.cat((maximum, average), dim=1))

    def forward_with_alpha(self, feature):
        """Optional analysis interface; neither features nor gates are detached."""
        alpha = self.get_alpha(feature)
        fused = alpha * self.max_spp(feature) + (1 - alpha) * self.avg_spp(feature)
        return fused, alpha

    def forward(self, feature):
        return self.forward_with_alpha(feature)[0]
