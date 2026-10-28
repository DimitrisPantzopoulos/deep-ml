# --- cell 1 ---
import torch
import numpy as np


def fit_linear_regression(X : torch.Tensor, y : torch.Tensor, lr : float=0.1, steps : int=500) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit y ~= X @ w + b with full-batch GD using only autograd.

    Args:
        X: Float tensor (N, D)
        y: Float tensor (N,) or (N, 1)
        lr: learning rate
        steps: number of gradient descent iterations

    Returns:
        w: Float tensor (D,) learned weights (no grad)
        b: Float tensor scalar learned bias (no grad)
    """
    # TODO: ensure y is shape (N,)
    N, D = X.shape

    assert y.shape == (N,), f"Expected y shape {(N,)}, got {tuple(y.shape)}"
    
    # TODO: initialize w (D,) and b with requires_grad=True
    weights : torch.Tensor = torch.randn(D, requires_grad=True)
    bias    : torch.Tensor = torch.zeros(1, requires_grad=True)
    
    for _ in range(steps):
        # Forward pass
        y_hat : torch.Tensor = X @ weights + bias
        
        # Calculate loss
        loss : torch.Tensor = ((y - y_hat) ** 2).mean()
        
        loss.backward()
        
        with torch.no_grad():
            weights -= lr * weights.grad ; weights.grad.zero_()
            bias    -= lr * bias.grad    ; bias.grad.zero_()
        
    return weights.detach(), bias.detach()