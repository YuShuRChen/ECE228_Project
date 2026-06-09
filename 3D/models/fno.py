import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from layers.spectral_layer import SpectralConv3d
from utilities.grid import get_grid3d

class FNOBlock3d(nn.Module):
    """
    Standard FNO block for 3D inputs.
    """
    def __init__(self, channels: int, modes: Tuple[int, int, int], use_gelu: bool = True):
        super().__init__()
        self.spectral_conv = SpectralConv3d(channels, channels, modes)
        self.skip_conn = nn.Conv3d(channels, channels, kernel_size=1)
        self.use_gelu = use_gelu
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.spectral_conv(x) + self.skip_conn(x)
        if self.use_gelu:
            out = F.gelu(out)
        return out

class FNO3d(nn.Module):
    """
    Standard Fourier Neural Operator architecture for 3D inputs.
    """
    def __init__(self, in_channels: int, out_channels: int, width: int, modes: Tuple[int, int, int], num_blocks: int = 4):
        super().__init__()
        
        # Lifting layer
        self.lifting = nn.Conv3d(in_channels, width, kernel_size=1)
        
        # Sequential FNO blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock3d(width, modes, use_gelu=(i < num_blocks - 1)) 
            for i in range(num_blocks)
        ])
        
        # Projection network
        self.projection = nn.Sequential(
            nn.Conv3d(width, 128, kernel_size=1), 
            nn.GELU(),
            nn.Conv3d(128, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, Channel, X, Y, Z)
        x = self.lifting(x)
        
        for block in self.fno_blocks:
            x = block(x)
        
        x = self.projection(x)
        return x

class FNO3dMultiGoal(nn.Module):
    """
    MultiGoal wrapper for standard FNO3d.
    Passes the mask with the goal embedded inside it directly into FNO.
    """
    def __init__(self, num_layers: int, modes1: int, modes2: int, modes3: int, width: int):
        super().__init__()
        # Standard FNO backbone, taking mask + grid as input (in_channels = 1 + 3 = 4)
        self.fno_backbone = FNO3d(
            in_channels=4, 
            out_channels=1, 
            width=width, 
            modes=(modes1, modes2, modes3), 
            num_blocks=num_layers
        )

    def forward(self, chi: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        # Embed goal into mask
        chi_mod = chi.clone()
        for i in range(chi_mod.shape[0]):
            x_idx = goal[i, 0].long().item() if goal.dim() == 2 else goal[i, 0, 0].long().item()
            y_idx = goal[i, 1].long().item() if goal.dim() == 2 else goal[i, 1, 0].long().item()
            z_idx = goal[i, 2].long().item() if goal.dim() == 2 else goal[i, 2, 0].long().item()
            chi_mod[i, 0, x_idx, y_idx, z_idx] = -1.0
            
        batchsize, _, size_x, size_y, size_z = chi.shape
        grid = get_grid3d(batchsize, size_x, size_y, size_z, chi.device)
        
        # Concat mask and grid
        x = torch.cat((chi_mod, grid), dim=1)
        out = self.fno_backbone(x)
        return out
