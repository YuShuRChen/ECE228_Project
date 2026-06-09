import torch
import numpy as np
from torch.utils.data import Dataset

class MotionPlanningDataset(Dataset):
    """
    Dataset loader for .npy files.
    """
    def __init__(self, dir_path: str):
        if not dir_path.endswith('/'):
            dir_path += '/'
            
        # (N, 1, C, C).
        self.mask = torch.tensor(np.load(dir_path + 'mask.npy'), dtype=torch.float32).unsqueeze(1)
        self.dist = torch.tensor(np.load(dir_path + 'dist_in.npy'), dtype=torch.float32).unsqueeze(1)
        self.goals = torch.tensor(np.load(dir_path + 'goal.npy'), dtype=torch.long) # (N, 2)
        self.output = torch.tensor(np.load(dir_path + 'output.npy'), dtype=torch.float32).unsqueeze(1)

        self.n_samples = self.mask.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        mask = self.mask[idx]
        dist = self.dist[idx]
        gt_value = self.output[idx]
        
        goal_coords = self.goals[idx]
        # Calculate chi: smoothed distance function
        chi = torch.mul(torch.tanh(dist * 5.0), (mask - 0.5)) + 0.5

        return mask, chi, goal_coords, gt_value, dist