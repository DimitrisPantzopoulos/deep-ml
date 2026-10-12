import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    # Your code here

    # Compute numerical approx. of grad
    steps : np.ndarray = epsilon * np.eye(x.size)

    numerical_grad = np.array([
        (f(x + step) - f(x - step)) / (2 * epsilon)
        for step in steps
    ])

    # Compute the relative error
    relative_error : np.float32 = np.linalg.norm(numerical_grad - analytical_grad) / (np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad))

    return numerical_grad, relative_error