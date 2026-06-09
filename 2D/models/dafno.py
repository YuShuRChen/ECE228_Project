import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from layers.spectral_layer import SpectralConv2d
from utilities.grid import get_grid2d

class DAFNOBlock(nn.Module):
    """
    DAFNO block combining masked spectral convolution and skip connection.
    Correctly implements the integral operator for DAFNO.
    """
    def __init__(self, channels: int, modes: Tuple[int, int], use_gelu: bool = True):
        super().__init__()
        self.spectral_conv = SpectralConv2d(channels, channels, modes)
        self.skip_conn = nn.Conv2d(channels, channels, kernel_size=1)
        self.channels = channels
        self.use_gelu = use_gelu
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = mask.expand(-1, self.channels, -1, -1)
        conv_chi = self.spectral_conv(mask_expanded)
        conv_chix = self.spectral_conv(mask_expanded * x)
        xconv_chi = x * conv_chi
        wx = self.skip_conn(x)
        
        out = mask_expanded * (conv_chix - xconv_chi + wx)
        if self.use_gelu:
            out = F.gelu(out)
        return out

class DAFNO2d(nn.Module):
    """
    Domain-Agnostic Fourier Neural Operator (DAFNO) architecture for 2D inputs.
    """
    def __init__(self, in_channels: int, out_channels: int, width: int, modes: Tuple[int, int], num_blocks: int = 4):
        super().__init__()
        
        # Lifting layer
        self.lifting = nn.Conv2d(in_channels, width, kernel_size=1)
        
        # Sequential DAFNO blocks
        self.fno_blocks = nn.ModuleList([
            DAFNOBlock(width, modes, use_gelu=(i < num_blocks - 1)) 
            for i in range(num_blocks)
        ])
        
        # Projection network
        self.projection = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1), 
            nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, Channel, Height, Width)
        x = self.lifting(x)
        
        # Pass through DAFNO blocks with mask
        for block in self.fno_blocks:
            x = block(x, mask)
        
        x = self.projection(x)
        return x

class DAFNO2dMultiGoal(nn.Module):
    """
    MultiGoal wrapper for DAFNO2d.
    """
    def __init__(self, num_layers, modes1, modes2, width):
        super(DAFNO2dMultiGoal, self).__init__()
        # Standard DAFNO backbone, taking grid as input (in_channels=2)
        self.dafno_backbone = DAFNO2d(
            in_channels=2, 
            out_channels=1, 
            width=width, 
            modes=(modes1, modes2), 
            num_blocks=num_layers
        )

    def forward(self, chi, goal):
        # chi shape: (B, 1, H, W)
        chi_mod = chi.clone()
        
        # Original repo indicates goal with a -1.0.
        for i in range(chi_mod.shape[0]):
            chi_mod[i, 0, goal[i][1].long(), goal[i][0].long()] = -1.0
            
        batchsize, _, size_x, size_y = chi.shape
        grid = get_grid2d(batchsize, size_x, size_y, chi.device)
        
        # DAFNO uses chi_mod as mask
        out = self.dafno_backbone(x=grid, mask=chi_mod)
            
        return out
