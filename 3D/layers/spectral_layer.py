import torch
import torch.nn as nn
from typing import Tuple

class SpectralConv3d(nn.Module):
    """
    3D Spectral Convolution layer using Fourier transforms.
    """
    def __init__(self, in_channels: int, out_channels: int, modes: Tuple[int, int, int]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x, self.modes_y, self.modes_z = modes
        
        # Scale for weight initialization
        scale = 1.0 / (in_channels * out_channels)
        
        # Learnable complex weights for low frequency modes
        self.weights1 = nn.Parameter(scale * torch.rand((in_channels, out_channels, self.modes_x, self.modes_y, self.modes_z), dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand((in_channels, out_channels, self.modes_x, self.modes_y, self.modes_z), dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.rand((in_channels, out_channels, self.modes_x, self.modes_y, self.modes_z), dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.rand((in_channels, out_channels, self.modes_x, self.modes_y, self.modes_z), dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        # Complex multiplication using einsum: (B, in_C, x, y, z), (in_C, out_C, x, y, z) -> (B, out_C, x, y, z)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, dim_x, dim_y, dim_z = x.shape
        
        # Transform to spectral domain (compute along the last 3 spatial dimensions)
        x_freq = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        # Allocate memory for truncated frequency representation
        out_freq = torch.zeros((batch_size, self.out_channels, dim_x, dim_y, dim_z // 2 + 1), dtype=torch.cfloat, device=x.device)
        
        # Multiply low frequency modes
        out_freq[:, :, :self.modes_x, :self.modes_y, :self.modes_z] = \
            self.compl_mul3d(x_freq[:, :, :self.modes_x, :self.modes_y, :self.modes_z], self.weights1)
        
        out_freq[:, :, -self.modes_x:, :self.modes_y, :self.modes_z] = \
            self.compl_mul3d(x_freq[:, :, -self.modes_x:, :self.modes_y, :self.modes_z], self.weights2)
            
        out_freq[:, :, :self.modes_x, -self.modes_y:, :self.modes_z] = \
            self.compl_mul3d(x_freq[:, :, :self.modes_x, -self.modes_y:, :self.modes_z], self.weights3)
            
        out_freq[:, :, -self.modes_x:, -self.modes_y:, :self.modes_z] = \
            self.compl_mul3d(x_freq[:, :, -self.modes_x:, -self.modes_y:, :self.modes_z], self.weights4)
        
        # Transform back to spatial domain
        x_out = torch.fft.irfftn(out_freq, s=(dim_x, dim_y, dim_z))
        return x_out
