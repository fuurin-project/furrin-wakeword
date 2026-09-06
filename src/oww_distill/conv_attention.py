from __future__ import annotations

import torch
from torch import nn


class ConvAttentionHead(nn.Module):
    """Small temporal Conv1D + self-attention wake-word classifier.

    The design follows the LiveKit WakeWord classifier while returning logits
    so training can use numerically stable focal/BCE losses.
    """

    def __init__(
        self,
        n_timesteps: int = 16,
        embedding_dim: int = 96,
        layer_dim: int = 16,
        n_blocks: int = 1,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        if n_timesteps < 1 or embedding_dim < 1 or layer_dim < 1 or n_blocks < 0:
            raise ValueError("invalid conv-attention dimensions")
        if n_heads < 1 or layer_dim % n_heads:
            raise ValueError("n_heads must divide layer_dim")

        layers: list[nn.Module] = [
            nn.Conv1d(embedding_dim, layer_dim, kernel_size=3, padding=1),
            nn.LayerNorm([layer_dim, n_timesteps]),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_blocks):
            layers.extend(
                [
                    nn.Conv1d(layer_dim, layer_dim, kernel_size=3, padding=1),
                    nn.LayerNorm([layer_dim, n_timesteps]),
                    nn.ReLU(inplace=True),
                ]
            )
        self.conv = nn.Sequential(*layers)
        self.attention = nn.MultiheadAttention(
            embed_dim=layer_dim, num_heads=n_heads, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(layer_dim)
        self.output = nn.Linear(layer_dim, 1)

        self.n_timesteps = n_timesteps
        self.embedding_dim = embedding_dim
        self.layer_dim = layer_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape (batch, time, feature)")
        hidden = self.conv(embeddings.transpose(1, 2)).transpose(1, 2)
        attended, _ = self.attention(hidden, hidden, hidden, need_weights=False)
        hidden = self.attention_norm(hidden + attended)
        return self.output(hidden.mean(dim=1)).squeeze(1)

    @property
    def deploy_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
