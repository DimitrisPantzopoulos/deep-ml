import torch

def grad_wss(
    w_list : list[float], 
    x_list : list[float]
    ) -> list[float]:
    """
    Build w (requires_grad) and x from lists, compute
    loss = 0.5 * sum((w * x)**2), backward, return w.grad
    as a list of floats rounded to 4 decimals.
    """
    weights : torch.tensor = torch.tensor(w_list, requires_grad=True)
    x       : torch.tensor = torch.tensor(x_list)

    # Compute the Scalar loss
    L : torch.tensor = (0.5) * torch.sum((weights * x) ** 2)

    L.backward()

    return weights.grad.tolist()
