import torch
import torch.nn as nn
import math

class LpLoss(object):
    """
    Relative L-p norm loss for discretized d-dimensional functions.
    Standard metric used in Neural Operator evaluations.
    """
    def __init__(self, d=1, p=2, L=2*math.pi, reduce_dims=0, reductions='sum'):
        super().__init__()
        self.d = d
        self.p = p

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims
        
        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions in ['sum', 'mean']
                self.reductions = [reductions] * len(self.reduce_dims)
            else:
                for r in reductions:
                    assert r in ['sum', 'mean']
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L] * self.d
        else:
            self.L = L
    
    @property
    def name(self):
        return f"L{self.p}_{self.d}Dloss"
    
    def uniform_h(self, x):
        h = [0.0] * self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j] / x.size(-j)
        return h

    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)
        return x

    def abs(self, x, y, h=None):
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h] * self.d
        
        const = math.prod(h) ** (1.0 / self.p)
        diff = const * torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), 
                                  p=self.p, dim=-1, keepdim=False)

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff

    def rel(self, x, y):
        diff = torch.norm(torch.flatten(x.contiguous(), start_dim=-self.d) - torch.flatten(y.contiguous(), start_dim=-self.d), 
                          p=self.p, dim=-1, keepdim=False)
        ynorm = torch.norm(torch.flatten(y.contiguous(), start_dim=-self.d), p=self.p, dim=-1, keepdim=False)

        diff = torch.nan_to_num(diff / ynorm, nan=0.0, posinf=0.0, neginf=0.0)

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, y_pred, y, **kwargs):
        return self.rel(y_pred, y)

class PINNLoss(nn.Module):
    """
    Physics-Informed (Eikonal) loss term.
    Enforces |∇V| ≈ 1 in free space (the Eikonal equation).
    
    Matches the original notebook implementation exactly:
    - Gradients computed on (B, H, W) shape (channel dim squeezed)
    - Boundary threshold: >= 1.01 (not > 5)
    - grad_mask is NOT binarized (uses raw GT gradient magnitudes)
    - Weight: 0.05
    """
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, out: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            out: predicted value function, shape (B, 1, H, W) or (B, H, W)
            gt:  ground truth value function, shape (B, 1, H, W) or (B, H, W)
        """
        # Squeeze to (B, H, W) — original notebook does .reshape(B, H, W)
        if out.dim() == 4:
            out_3d = out.squeeze(1)
        else:
            out_3d = out
        if gt.dim() == 4:
            gt_3d = gt.squeeze(1)
        else:
            gt_3d = gt
        
        B, H, W = out_3d.shape
        
        # Compute spatial gradient magnitudes: ||∇f|| over dims [1, 2] (H, W)
        grad_out = torch.linalg.vector_norm(
            torch.stack(list(torch.gradient(out_3d, dim=[1, 2])), dim=0), dim=0)
        grad_gt = torch.linalg.vector_norm(
            torch.stack(list(torch.gradient(gt_3d, dim=[1, 2])), dim=0), dim=0)
        
        # Mask: zero out boundary gradients (>= 1.01), keep free-space gradients (~1.0)
        # NOT binarized — raw gradient magnitudes are used as targets
        grad_mask = grad_gt.clone()
        grad_mask[grad_mask >= 1.01] = 0
        
        # Loss: relative L2 of (predicted_grad * mask) vs mask
        # In free space where grad_mask ≈ 1, this enforces |∇V_pred| ≈ 1
        loss = self.loss_fn(
            (grad_out * grad_mask).view(B, H, W),
            grad_mask.view(B, H, W))
        
        return loss * 0.05

