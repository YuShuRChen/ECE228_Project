import torch
import numpy as np

def get_grid2d(batchsize: int, size_x: int, size_y: int, device: torch.device) -> torch.Tensor:
    """
    Generates a 2D spatial coordinate grid bounded between [-1, 1].
    
    Args:
        batchsize (int): Batch size.
        size_x (int): Height (or X spatial dimension).
        size_y (int): Width (or Y spatial dimension).
        device (torch.device): Device to allocate the tensor on.
        
    Returns:
        torch.Tensor: A tensor of shape (B, 2, H, W) containing the spatial grid.
    """
    gridx = torch.tensor(np.linspace(-1, 1, size_x), dtype=torch.float32, device=device)
    gridx = gridx.reshape(1, 1, size_x, 1).expand(batchsize, 1, size_x, size_y)
    
    gridy = torch.tensor(np.linspace(-1, 1, size_y), dtype=torch.float32, device=device)
    gridy = gridy.reshape(1, 1, 1, size_y).expand(batchsize, 1, size_x, size_y)
    
    return torch.cat((gridx, gridy), dim=1)
