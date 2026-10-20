"""Parameter-free average SPP with the same levels and layout as the max variant."""

import torch
import torch.nn.functional as F
from torch import nn


class AverageSpatialPyramidPooling(nn.Module):
    def forward(self, feature):
        return torch.cat([F.adaptive_avg_pool2d(feature, (level, level)).flatten(start_dim=1)
                          for level in (1, 2, 3, 6)], dim=1)
