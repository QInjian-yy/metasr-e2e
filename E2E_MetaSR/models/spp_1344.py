"""Independent 1344-dimensional Max SPP region encoder."""

import torch
import torch.nn.functional as F
from torch import nn


class SpatialPyramidPooling1344(nn.Module):
    def forward(self, feature):
        return torch.cat([F.adaptive_max_pool2d(feature, (level, level)).flatten(start_dim=1)
                          for level in (1, 2, 4)], dim=1)
