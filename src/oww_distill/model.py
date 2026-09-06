from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


TEACHER_INDICES = tuple(75 + 8 * index for index in range(16))


class CausalDepthwiseBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.left_context = 2 * dilation
        self.depthwise = nn.Conv1d(
            channels, channels, 3, dilation=dilation, groups=channels
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(F.pad(x, (self.left_context, 0)))
        return F.relu(self.pointwise(x) + residual)


class TinyEmbeddingStudent(nn.Module):
    """Causal, MCU-oriented encoder with an openWakeWord-compatible output."""

    def __init__(self, feature_bins: int = 32, channels: int = 64) -> None:
        super().__init__()
        self.feature_bins = feature_bins
        self.channels = channels
        self.input_projection = nn.Conv1d(feature_bins, channels, 1)
        self.blocks = nn.ModuleList(
            CausalDepthwiseBlock(channels, dilation)
            for dilation in (1, 2, 4, 8, 16, 32)
        )
        self.output_projection = nn.Conv1d(channels, 96, 1)

    def frame_embeddings(self, features: torch.Tensor) -> torch.Tensor:
        x = features.transpose(1, 2)
        x = F.relu(self.input_projection(x))
        for block in self.blocks:
            x = block(x)
        return self.output_projection(x).transpose(1, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        frames = self.frame_embeddings(features)
        return frames[:, TEACHER_INDICES, :]

    @property
    def deploy_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
