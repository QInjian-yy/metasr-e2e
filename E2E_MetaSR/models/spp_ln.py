"""The existing SPP followed by per-region LayerNorm, before MIL pooling."""

from torch import nn


class LayerNormalizedSPP(nn.Module):
    def __init__(self, spp):
        super().__init__()
        self.spp = spp
        self.norm = nn.LayerNorm(3200)

    def forward(self, feature):
        return self.norm(self.spp(feature))
