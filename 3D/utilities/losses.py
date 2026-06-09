import torch
import torch.nn as nn

class LpLoss(object):
    """
    Relative L2 loss metric for performance evaluation and training.
    Works for 2D and 3D depending on the d parameter.
    """
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]
        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class PINNLoss3d(nn.Module):
    """
    Sobolev/PINN Loss for 3D outputs.
    Calculates the spatial gradients of the predictions and the ground truth,
    and computes the L2 loss between them.
    """
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, out: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # Calculate gradients along spatial dims X, Y, Z (dims 2, 3, 4)
        grads_out = torch.gradient(out, dim=[2, 3, 4])
        grads_gt = torch.gradient(gt, dim=[2, 3, 4])
        
        # Stack gradients: [3, B, C, X, Y, Z]
        stacked_grads_out = torch.stack(list(grads_out), dim=0)
        stacked_grads_gt = torch.stack(list(grads_gt), dim=0)
        
        # Calculate vector norm over the 3 spatial gradient components
        # Result shape: [B, C, X, Y, Z]
        norm_grad_out = torch.linalg.vector_norm(stacked_grads_out, dim=0)
        norm_grad_gt = torch.linalg.vector_norm(stacked_grads_gt, dim=0)
        
        # Exact Eikonal mask formulation from original repo
        # Set boundary gradients (> 5) to 0, and non-zero free space to 1.
        grad_mask = norm_grad_gt.clone()
        grad_mask[grad_mask > 5] = 0
        grad_mask[grad_mask != 0] = 1
        
        # Enforce |grad_out| = 1 in the free space by matching grad_out * grad_mask to grad_mask
        loss = self.loss_fn(norm_grad_out * grad_mask, grad_mask)
        return loss * 0.05
