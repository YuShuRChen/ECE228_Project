import torch
import torch.nn as nn
from typing import Tuple
from models.dafno import DAFNOBlock3d
from models.deepnorm import DeepNormMetric, MaxReLUPairwiseActivation
from utilities.grid import get_grid3d

class PNO3d(nn.Module):
    """
    Planning Neural Operator (PNO) backbone for 3D inputs.
    Uses DAFNO blocks for feature extraction without the projection layer.
    """
    def __init__(self, in_channels: int, width: int, modes: Tuple[int, int, int], num_blocks: int = 4):
        super().__init__()
        
        # Lifting layer
        self.lifting = nn.Conv3d(in_channels, width, kernel_size=1)
        
        # Sequential DAFNO blocks
        self.dafno_blocks = nn.ModuleList([
            DAFNOBlock3d(width, modes, use_gelu=(i < num_blocks - 1)) 
            for i in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, Channel, X, Y, Z)
        x = self.lifting(x)
        
        for block in self.dafno_blocks:
            x = block(x, mask)
            
        return x

class PNO3dMultiGoal(nn.Module):
    """
    MultiGoal wrapper for PNO3d.
    Extracts DeepNorm features at the target goal coordinates using the extracted PNO features.
    """
    def __init__(self, num_layers: int, modes1: int, modes2: int, modes3: int, width: int):
        super().__init__()
        self.width = width
        
        # PNO backbone
        self.pno_backbone = PNO3d(
            in_channels=3, 
            width=width, 
            modes=(modes1, modes2, modes3), 
            num_blocks=num_layers
        )
        
        # DeepNorm metric for goal conditioning
        self.deepnorm = DeepNormMetric(
            num_features=width, 
            layers=(128, 128), 
            concave_activation_size=20, 
            activation=lambda: MaxReLUPairwiseActivation(128), 
            symmetric=True
        )

    def forward(self, chi: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        # chi shape: (B, 1, X, Y, Z)
        batchsize, _, size_x, size_y, size_z = chi.shape
        grid = get_grid3d(batchsize, size_x, size_y, size_z, chi.device)
        
        # PNO does NOT inject the goal into the mask (chi).
        # We pass the pure distance field chi as the mask.
        x = self.pno_backbone(x=grid, mask=chi)
        
        # Extract features at the goal locations
        # goal shape might be [B, 3] or [B, 3, 1]
        batch_indices = torch.arange(batchsize, device=goal.device)
        
        x_indices = goal[:, 0].long() if goal.dim() == 2 else goal[:, 0, 0].long()
        y_indices = goal[:, 1].long() if goal.dim() == 2 else goal[:, 1, 0].long()
        z_indices = goal[:, 2].long() if goal.dim() == 2 else goal[:, 2, 0].long()
        
        # Permute x to [B, X, Y, Z, Channels] to match indexing logic
        x_permuted = x.permute(0, 2, 3, 4, 1)
        
        # g is the feature vector at the goal location for each item in the batch -> [B, Channels]
        g = x_permuted[batch_indices, x_indices, y_indices, z_indices, :]
        
        # Reshape to apply DeepNorm over all spatial locations
        # feature1: Goal features broadcasted over all spatial points -> [B * X * Y * Z, width]
        g_expanded = g.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1, size_x, size_y, size_z, -1)
        
        feature1 = g_expanded.reshape(-1, self.width)
        feature2 = x_permuted.reshape(-1, self.width)
        
        # Project using DeepNorm metric
        output = self.deepnorm(feature1, feature2)
        
        # Reshape back to [B, 1, X, Y, Z]
        output = output.reshape(batchsize, 1, size_x, size_y, size_z)
        
        return output
