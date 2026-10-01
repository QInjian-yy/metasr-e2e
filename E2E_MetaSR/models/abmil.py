"""Verbatim GatedAttentionMIL from E2E/downstream/attention_resnet/model.py."""

import torch
from torch import nn


class GatedAttentionMIL(nn.Module):
    """Attention over N region embeddings; output is one bag vector and N weights."""

    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        self.attention_V = nn.Linear(input_dim, hidden_dim, bias=True)
        self.attention_U = nn.Linear(input_dim, hidden_dim, bias=True)
        self.attention_w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, embeddings):
        gated = torch.tanh(self.attention_V(embeddings)) * torch.sigmoid(self.attention_U(embeddings))
        scores = self.attention_w(gated).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        pooled = (weights.unsqueeze(-1) * embeddings).sum(dim=0, keepdim=True)
        return pooled, weights
