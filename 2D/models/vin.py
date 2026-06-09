import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModule(nn.Module):
    """
    Computes a reward map R from the input occupancy + goal map.
    Three conv layers: 1 → 32 → 32 → 1, all 3×3 with GELU activations.
    """
    def __init__(self, hidden_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x: (B, 1, H, W)
        return self.net(x)


class ValueIterationModule(nn.Module):
    """
    Performs K iterations of a learned Bellman backup.
    Each iteration: Q = f_theta(concat(V, R)) with shared weights across iterations.
    The convolutions approximate max_a Q(s, a) (soft value iteration).
    """
    def __init__(self, k_iterations=20, hidden_channels=32):
        super().__init__()
        self.k_iterations = k_iterations
        # Shared transition model applied at every iteration
        # Input: concat(V, R) → 2 channels
        self.transition = nn.Sequential(
            nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, R):
        # R: (B, 1, H, W) — reward map
        V = R  # Initialize value = reward
        for _ in range(self.k_iterations):
            VR = torch.cat([V, R], dim=1)  # (B, 2, H, W)
            Q = self.transition(VR)         # (B, 1, H, W)
            V = Q
        return V


class OutputProjection(nn.Module):
    """
    Projects the converged value map to the final output.
    Conv2d(1, 32, 3) → GELU → Conv2d(32, 1, 1).
    """
    def __init__(self, hidden_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class VIN2dMultiGoal(nn.Module):
    """
    Value Iteration Network (VIN) for 2D multi-goal motion planning.

    VIN learns to approximate value iteration by using convolutions to simulate
    the Bellman backup operator. Unlike operator-learning approaches (FNO/PNO),
    VIN learns the *algorithm structure* of value iteration.

    Architecture:
        1. Reward Module: maps input (occupancy + goal) to a reward map R
        2. Value Iteration Module: K iterations of learned Bellman backup
        3. Output Projection: refines the converged value map

    Fully convolutional — naturally supports variable resolution
    (e.g., train on 64×64, test on 256×256 or higher).

    Interface matches FNO2dMultiGoal:
        forward(chi, goals) where chi is (B, 1, H, W), goals is (B, 2)
        Returns (B, 1, H, W)
    """
    def __init__(self, k_iterations=20, hidden_channels=32):
        super().__init__()
        self.reward = RewardModule(hidden_channels=hidden_channels)
        self.vi = ValueIterationModule(
            k_iterations=k_iterations, hidden_channels=hidden_channels
        )
        self.projection = OutputProjection(hidden_channels=hidden_channels)

    def forward(self, x, goal):
        # x: (B, 1, H, W) — occupancy grid (chi)
        # goal: (B, 2) — goal coordinates [col, row]
        x_mod = x.clone()

        # Inject goal as -1 at the goal pixel (same convention as FNO)
        for i in range(x_mod.shape[0]):
            x_mod[i, 0, goal[i][1].long(), goal[i][0].long()] = -1.0

        R = self.reward(x_mod)        # (B, 1, H, W)
        V = self.vi(R)                # (B, 1, H, W)
        out = self.projection(V)      # (B, 1, H, W)
        return out
