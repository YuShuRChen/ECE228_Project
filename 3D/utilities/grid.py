import torch
import numpy as np

def get_grid3d(batchsize: int, size_x: int, size_y: int, size_z: int, device: torch.device) -> torch.Tensor:
    """
    Generates a 3D spatial coordinate grid bounded between [-1, 1].
    Matches PyTorch's channel-first convention: returns [B, 3, X, Y, Z].
    """
    gridx = torch.tensor(np.linspace(-1, 1, size_x), dtype=torch.float32, device=device)
    gridx = gridx.reshape(1, 1, size_x, 1, 1).expand(batchsize, 1, size_x, size_y, size_z)
    
    gridy = torch.tensor(np.linspace(-1, 1, size_y), dtype=torch.float32, device=device)
    gridy = gridy.reshape(1, 1, 1, size_y, 1).expand(batchsize, 1, size_x, size_y, size_z)
    
    gridz = torch.tensor(np.linspace(-1, 1, size_z), dtype=torch.float32, device=device)
    gridz = gridz.reshape(1, 1, 1, 1, size_z).expand(batchsize, 1, size_x, size_y, size_z)
    
    return torch.cat((gridx, gridy, gridz), dim=1)
