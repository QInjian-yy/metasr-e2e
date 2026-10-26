"""Parameter-free mean pooling over all region embeddings in one WSI."""

from torch import nn


class MeanMIL(nn.Module):
    def forward(self, embeddings):
        if embeddings.ndim != 2 or embeddings.shape[0] < 1:
            raise ValueError("MeanMIL expects a nonempty [N,D] region bag")
        weights = embeddings.new_full((embeddings.shape[0],), 1.0 / embeddings.shape[0])
        return embeddings.mean(dim=0, keepdim=True), weights
