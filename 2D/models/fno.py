import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from layers.spectral_layer import SpectralConv2d

class FNO2d(nn.Module):
    """
    Fourier Neural Operator (FNO) architecture for 2D inputs.
    Matches the original paper's FNO2d exactly:
    - Padding applied AFTER lifting, removed BEFORE projection
    - All layers use GELU activation (including the last)
    """
    def __init__(self, in_channels: int, out_channels: int, width: int,
                 modes: Tuple[int, int], num_blocks: int = 4, padding: int = 1):
        super().__init__()
        self.width = width
        self.padding = padding
        self.num_layers = num_blocks

        # Lifting: Conv2d(in, width, 1) ≡ Linear(in, width) on last dim
        self.lifting = nn.Conv2d(in_channels, width, kernel_size=1)

        # Spectral + skip layers (registered individually like the original)
        for i in range(self.num_layers):
            self.add_module('conv%d' % i, SpectralConv2d(width, width, modes))
            self.add_module('w%d' % i, nn.Conv2d(width, width, 1))

        # Projection: Conv2d(width,128,1) → GELU → Conv2d(128,out,1)
        self.projection = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.lifting(x)

        # Pad AFTER lifting (matching original)
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        # Spectral blocks with GELU on ALL layers (matching original)
        for i in range(self.num_layers):
            x1 = self._modules['conv%d' % i](x)
            x2 = self._modules['w%d' % i](x)
            x = x1 + x2
            x = F.gelu(x)

        # Remove padding BEFORE projection (matching original)
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]

        x = self.projection(x)
        return x


class FNO2dMultiGoal(nn.Module):
    """
    MultiGoal wrapper for FNO2d.
    Matches the original paper's FNO2dMultiGoal.
    """
    def __init__(self, num_layers, padding, modes1, modes2, width):
        super(FNO2dMultiGoal, self).__init__()
        self.padding = padding

        # The backbone handles its own padding internally now
        self.fno_backbone = FNO2d(
            in_channels=1,
            out_channels=1,
            width=width,
            modes=(modes1, modes2),
            num_blocks=num_layers,
            padding=padding
        )

    def forward(self, x, goal):
        # x shape: (B, 1, H, W)
        x_mod = x.clone()

        # Original repo indicates goal with a -1.0.
        for i in range(x_mod.shape[0]):
            x_mod[i, 0, goal[i][1].long(), goal[i][0].long()] = -1.0

        # Padding is now handled inside the backbone
        out = self.fno_backbone(x_mod)

        return out