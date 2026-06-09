import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepONet2dMultiGoal(nn.Module):
    """
    Deep Operator Network (DeepONet) for 2D motion planning.

    Branch-trunk architecture:
      - Branch network encodes the environment map (chi with goal injected)
        through adaptive pooling (to fixed resolution) + MLP → (B, p)
      - Trunk network encodes normalised (x,y) query coordinates
        through MLP → (H*W, p)
      - Output = einsum('bp,np->bn') + bias → reshape (B, 1, H, W)

    Naturally supports variable resolution at test time because the trunk
    network takes continuous coordinate inputs while the branch network
    uses adaptive average pooling to a fixed internal resolution.

    Interface matches FNO2dMultiGoal:
      forward(chi, goals) where chi is (B, 1, H, W) and goals is (B, 2)
      returns (B, 1, H, W)
    """

    def __init__(self, branch_fixed_res=16, branch_hidden=512,
                 trunk_hidden=128, p=64):
        """
        Args:
            branch_fixed_res: Resolution to pool input to before flattening.
                              16x16=256 inputs keeps param count manageable.
            branch_hidden: Hidden dimension in the branch MLP.
            trunk_hidden: Hidden dimension in the trunk MLP.
            p: Feature dimension (rank of the decomposition).
        """
        super(DeepONet2dMultiGoal, self).__init__()
        self.branch_fixed_res = branch_fixed_res
        self.p = p

        branch_input_dim = branch_fixed_res * branch_fixed_res  # 256

        # Branch network: encodes environment + goal
        self.branch_net = nn.Sequential(
            nn.Linear(branch_input_dim, branch_hidden),
            nn.GELU(),
            nn.Linear(branch_hidden, 256),
            nn.GELU(),
            nn.Linear(256, p),
        )

        # Trunk network: encodes query coordinates
        self.trunk_net = nn.Sequential(
            nn.Linear(2, trunk_hidden),
            nn.GELU(),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.GELU(),
            nn.Linear(trunk_hidden, p),
        )

        # Learned bias for the output
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, goal):
        """
        Args:
            x: (B, 1, H, W) environment occupancy map
            goal: (B, 2) goal pixel coordinates

        Returns:
            (B, 1, H, W) predicted value function
        """
        B, _, H, W = x.shape

        # --- Inject goal as -1 at goal pixel (same convention as FNO) ---
        x_mod = x.clone()
        for i in range(B):
            x_mod[i, 0, goal[i][1].long(), goal[i][0].long()] = -1.0

        # --- Branch ---
        # Adaptive pool to fixed resolution so branch MLP has constant input dim
        branch_in = F.adaptive_avg_pool2d(x_mod, (self.branch_fixed_res,
                                                    self.branch_fixed_res))
        branch_in = branch_in.view(B, -1)              # (B, fixed_res^2)
        branch_out = self.branch_net(branch_in)         # (B, p)

        # --- Trunk ---
        # Build normalised coordinate grid for the *actual* resolution
        # coords in [0, 1] matching pixel centres
        gy = torch.linspace(0, 1, H, device=x.device, dtype=x.dtype)
        gx = torch.linspace(0, 1, W, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        coords = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        coords = coords.reshape(-1, 2)                  # (H*W, 2)

        trunk_out = self.trunk_net(coords)               # (H*W, p)

        # --- Combine: dot product + bias ---
        # branch_out: (B, p), trunk_out: (H*W, p)
        out = torch.einsum('bp,np->bn', branch_out, trunk_out)  # (B, H*W)
        out = out + self.bias
        out = out.reshape(B, 1, H, W)

        return out
