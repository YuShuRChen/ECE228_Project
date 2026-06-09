import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from layers.spectral_layer import SpectralConv3d
from utilities.grid import get_grid3d

class DAFNOBlock3d(nn.Module):
    """
    DAFNO block for 3D domain-agnostic inputs.
    Correctly implements the integral operator for DAFNO:
    mask * (K(mask*x) - x*K(mask) + W(x))
    """
    def __init__(self, channels: int, modes: Tuple[int, int, int], use_gelu: bool = True):
        super().__init__()
        self.spectral_conv = SpectralConv3d(channels, channels, modes)
        self.skip_conn = nn.Conv3d(channels, channels, kernel_size=1)
        self.channels = channels
        self.use_gelu = use_gelu
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask is [B, 1, X, Y, Z]
        # expand mask to match x channels
        mask_expanded = mask.expand(-1, self.channels, -1, -1, -1)
        
        conv_chi = self.spectral_conv(mask_expanded)
        conv_chix = self.spectral_conv(mask_expanded * x)
        xconv_chi = x * conv_chi
        wx = self.skip_conn(x)
        
        out = mask_expanded * (conv_chix - xconv_chi + wx)
        if self.use_gelu:
            out = F.gelu(out)
        return out

class DAFNO3d(nn.Module):
    """
    Domain-Agnostic Fourier Neural Operator (DAFNO) architecture for 3D inputs.
    """
    def __init__(self, in_channels: int, out_channels: int, width: int, modes: Tuple[int, int, int], num_blocks: int = 4):
        super().__init__()
        
        # Lifting layer
        self.lifting = nn.Conv3d(in_channels, width, kernel_size=1)
        
        # Sequential DAFNO blocks
        self.fno_blocks = nn.ModuleList([
            DAFNOBlock3d(width, modes, use_gelu=(i < num_blocks - 1)) 
            for i in range(num_blocks)
        ])
        
        # Projection network
        self.projection = nn.Sequential(
            nn.Conv3d(width, 128, kernel_size=1), 
            nn.GELU(),
            nn.Conv3d(128, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, Channel, X, Y, Z)
        x = self.lifting(x)
        
        # Pass through DAFNO blocks with mask
        for block in self.fno_blocks:
            x = block(x, mask)
        
        x = self.projection(x)
        return x

class DAFNO3dMultiGoal(nn.Module):
    """
    MultiGoal wrapper for DAFNO3d.
    Injects the goal (-1.0) directly into the distance field mask.
    """
    def __init__(self, num_layers: int, modes1: int, modes2: int, modes3: int, width: int):
        super().__init__()
        # Standard DAFNO backbone, taking grid as input (in_channels=3)
        self.dafno_backbone = DAFNO3d(
            in_channels=3, 
            out_channels=1, 
            width=width, 
            modes=(modes1, modes2, modes3), 
            num_blocks=num_layers
        )

    def forward(self, chi: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        # chi shape: (B, 1, X, Y, Z)
        # goal shape: (B, 3, 1) or similar. The original script expects goals extracted like gs[:,0,0]
        chi_mod = chi.clone()
        
        # Original repo indicates goal with a -1.0.
        for i in range(chi_mod.shape[0]):
            x_idx = goal[i, 0].long().item() if goal.dim() == 2 else goal[i, 0, 0].long().item()
            y_idx = goal[i, 1].long().item() if goal.dim() == 2 else goal[i, 1, 0].long().item()
            z_idx = goal[i, 2].long().item() if goal.dim() == 2 else goal[i, 2, 0].long().item()
            chi_mod[i, 0, x_idx, y_idx, z_idx] = -1.0
            
        batchsize, _, size_x, size_y, size_z = chi.shape
        grid = get_grid3d(batchsize, size_x, size_y, size_z, chi.device)
        
        # DAFNO uses chi_mod as mask
        out = self.dafno_backbone(x=grid, mask=chi_mod)
        return out
