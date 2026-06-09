import os
import torch
import numpy as np
from torch.utils.data import Dataset

def smooth_chi(mask: torch.Tensor, dist: torch.Tensor, smooth_coef: float = 5.0) -> torch.Tensor:
    """
    Smoothes the distance field inside the masked region.
    """
    return torch.mul(torch.tanh(dist * smooth_coef), (mask - 0.5)) + 0.5

class MotionPlanningDataset3D(Dataset):
    """
    Unified dataset loader for 3D Motion Planning.
    Dynamically infers dimensions from the `.npy` files.
    """
    def __init__(self, data_dir: str, smooth_coef: float = 5.0, max_items: int = None):
        super().__init__()
        
        mask_path = os.path.join(data_dir, 'mask.npy')
        dist_path = os.path.join(data_dir, 'dist_in.npy')
        output_path = os.path.join(data_dir, 'output.npy')
        goals_path = os.path.join(data_dir, 'goals.npy')
        
        # Load arrays
        mask = np.load(mask_path)
        dist_in = np.load(dist_path)
        output = np.load(output_path)
        goals = np.load(goals_path)
        
        if max_items is not None:
            mask = mask[:max_items]
            dist_in = dist_in[:max_items]
            output = output[:max_items]
            goals = goals[:max_items]
            
        N, Sx, Sy, Sz = mask.shape
        
        self.mask = torch.tensor(mask, dtype=torch.float32).reshape(N, 1, Sx, Sy, Sz)
        dist_tensor = torch.tensor(dist_in, dtype=torch.float32)
        
        self.chi = smooth_chi(torch.tensor(mask, dtype=torch.float32), dist_tensor, smooth_coef).reshape(N, 1, Sx, Sy, Sz)
        self.y = torch.tensor(output, dtype=torch.float32).reshape(N, 1, Sx, Sy, Sz)
        self.goals = torch.tensor(goals, dtype=torch.float32).reshape(N, 3, 1)

    def __len__(self):
        return self.mask.shape[0]

    def __getitem__(self, idx):
        return self.mask[idx], self.chi[idx], self.goals[idx], self.y[idx]
