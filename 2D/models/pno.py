import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from .dafno import DAFNOBlock
from .deepnorm import DeepNormMetric, MaxReLUPairwiseActivation
from utilities.grid import get_grid2d

class PNO2d(nn.Module):
    """
    Planning Neural Operator (PNO) Backbone.
    """
    def __init__(self, in_channels: int, width: int, modes: Tuple[int, int], num_blocks: int = 4):
        super().__init__()
        
        self.lifting = nn.Conv2d(in_channels, width, kernel_size=1)
        self.dafno_blocks = nn.ModuleList([
            DAFNOBlock(width, modes, use_gelu=(i < num_blocks - 1)) 
            for i in range(num_blocks)
        ])
        self.width = width
        
        self.deepnorm = DeepNormMetric(
            num_features=width, 
            layers=(128, 128), 
            concave_activation_size=20, 
            activation=lambda: MaxReLUPairwiseActivation(128), 
            symmetric=True
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        # x is the coordinate grid (B, 2, H, W)
        x = self.lifting(x)
        
        # Pass features and smoothed mask through DAFNO blocks
        for block in self.dafno_blocks:
            x = block(x, mask)
            
        # PNO feature extraction at goal
        # x is (B, width, H, W)
        batchsize, width, size_x, size_y = x.shape
        x_permuted = x.permute(0, 2, 3, 1) # (B, H, W, width)
        
        # Extract features exactly at the goal location: shape (B, 1, 1, width)
        goal_x_indices = goal[:, 0].long()
        goal_y_indices = goal[:, 1].long()
        g = x_permuted[torch.arange(batchsize), goal_y_indices, goal_x_indices, :]
        # Original notebook: g.unsqueeze(1).repeat(1, size_x, 1).unsqueeze(1).repeat(1, size_x, 1, 1)
        g = g.unsqueeze(1).repeat(1, size_x, 1).unsqueeze(1).repeat(1, size_x, 1, 1)  # (B, H, W, width)
        
        # Reshape for DeepNorm Metric
        feature1 = g.reshape(-1, width)
        feature2 = x_permuted.reshape(-1, width)
        
        # Predict value function
        output = self.deepnorm(feature1, feature2)
        
        # Reshape back to spatial dimensions (B, 1, H, W)
        # DeepNorm reduces the last dimension to 1 value per pixel, so it's (B*H*W,)
        output = output.reshape(batchsize, 1, size_x, size_y)
        
        return output

class DEEPNORM2dMultiGoal(nn.Module):
    """
    MultiGoal wrapper for PNO2d (DeepNorm).
    Also exported as PNO2dMultiGoal for convenience.
    """
    def __init__(self, num_layers, modes1, modes2, width):
        super(DEEPNORM2dMultiGoal, self).__init__()
        # Standard PNO backbone, taking grid as input (in_channels=2)
        self.pno_backbone = PNO2d(
            in_channels=2, 
            width=width, 
            modes=(modes1, modes2), 
            num_blocks=num_layers
        )

    def forward(self, chi, goal):
        # chi shape: (B, 1, H, W)
        batchsize, _, size_x, size_y = chi.shape
        grid = get_grid2d(batchsize, size_x, size_y, chi.device)
        
        # PNO does NOT inject the goal into chi (unlike DAFNO).
        # It relies on extracting the spatial feature at the goal location.
        out = self.pno_backbone(x=grid, mask=chi, goal=goal)
        return out

# Alias for convenience
PNO2dMultiGoal = DEEPNORM2dMultiGoal
