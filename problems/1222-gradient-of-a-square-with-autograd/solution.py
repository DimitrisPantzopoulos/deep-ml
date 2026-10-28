import torch

def grad_of_square(x_val : float) -> float:
    """Return dy/dx for y = x**2 at x = x_val using autograd.

    Args:
        x_val (float): scalar input value.

    Returns:
        float: gradient of x**2 w.r.t. x at x_val.
    """
    # TODO: create tensor, compute y = x**2, backward, return grad
    x : torch.Tensor = torch.tensor([x_val], dtype=torch.float32, requires_grad=True)

    y : torch.Tensor = x ** 2

    y.backward()

    return x.grad.item()
