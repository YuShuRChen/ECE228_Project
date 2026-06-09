import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution layer using Fourier transforms.
    """
    def __init__(self, in_channels: int, out_channels: int, modes: Tuple[int, int]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x, self.modes_y = modes
        
        # Scale for weight initialization
        scale = 1.0 / (in_channels * out_channels)
        
        # Learnable complex weights for low frequency modes
        self.weight_top = nn.Parameter(scale * torch.rand((in_channels, out_channels, self.modes_x, self.modes_y), dtype=torch.cfloat))
        self.weight_bottom = nn.Parameter(scale * torch.rand((in_channels, out_channels, self.modes_x, self.modes_y), dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # Complex multiplication using einsum: (B, in_C, x, y), (in_C, out_C, x, y) -> (B, out_C, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        
        # Transform to spectral domain
        x_freq = torch.fft.rfft2(x)
        
        # Allocate memory for truncated frequency representation
        out_freq = torch.zeros((batch_size, self.out_channels, height, width // 2 + 1), dtype=torch.cfloat, device=x.device)
        
        # Multiply low frequency modes
        out_freq[:, :, :self.modes_x, :self.modes_y] = self.compl_mul2d(x_freq[:, :, :self.modes_x, :self.modes_y], self.weight_top)
        out_freq[:, :, -self.modes_x:, :self.modes_y] = self.compl_mul2d(x_freq[:, :, -self.modes_x:, :self.modes_y], self.weight_bottom)
        
        # Transform back to spatial domain
        x_out = torch.fft.irfft2(out_freq, s=(height, width))
        return x_out
    
class DAFNOSpectralConv2d(nn.Module):
    """
    Domain-Agnostic Spectral Convolution.
    Takes an additional 'mask' parameter representing the active domain geometry.
    """
    def __init__(self, in_channels: int, out_channels: int, modes: Tuple[int, int]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x, self.modes_y = modes
        
        scale = 1.0 / (in_channels * out_channels)
        self.weight_top = nn.Parameter(scale * torch.rand((in_channels, out_channels, self.modes_x, self.modes_y), dtype=torch.cfloat))
        self.weight_bottom = nn.Parameter(scale * torch.rand((in_channels, out_channels, self.modes_x, self.modes_y), dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Apply mask before frequency transform to ignore obstacles
        x_masked = x * mask
        
        batch_size, _, height, width = x_masked.shape
        
        # Transform to spectral domain
        x_freq = torch.fft.rfft2(x_masked)
        out_freq = torch.zeros((batch_size, self.out_channels, height, width // 2 + 1), dtype=torch.cfloat, device=x.device)
        
        # Truncate and multiply with weights
        out_freq[:, :, :self.modes_x, :self.modes_y] = self.compl_mul2d(x_freq[:, :, :self.modes_x, :self.modes_y], self.weight_top)
        out_freq[:, :, -self.modes_x:, :self.modes_y] = self.compl_mul2d(x_freq[:, :, -self.modes_x:, :self.modes_y], self.weight_bottom)
        
        # Transform back to spatial domain
        x_out = torch.fft.irfft2(out_freq, s=(height, width))
        
        # Apply mask again to remove high-frequency artifacts in obstacle regions
        x_out_masked = x_out * mask
        
        return x_out_masked
