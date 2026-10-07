import numpy as np

def forward_pass(theta : np.ndarray, X : np.ndarray) -> np.ndarray:
    return X @ theta

def compute_error(y_hat : np.ndarray, y : np.ndarray) -> np.ndarray:
    return y_hat - y

def compute_gradient(X : np.ndarray, error : np.ndarray, m : int) -> np.ndarray:
    return (1 / m) * np.dot(X.transpose(), error)

def get_weight_updates(gradient : np.ndarray, alpha : float) -> np.ndarray:
    return -alpha * gradient


def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D array of shape (n,)
    """
    m, n = X.shape
    y = y.reshape(-1, 1)  # Ensure y is a column vector
    theta = np.zeros((n, 1))  # Initialize weights to zeros

    # Your code here: implement gradient descent
    for _ in range(iterations):
        # Forward pass
        y_hat : np.ndarray = forward_pass(theta, X)

        # Compute error
        error : np.ndarray = compute_error(y_hat, y)

        # compute gradient
        weight_grad : np.ndarray = compute_gradient(X, error, m)

        # Update weights
        theta += get_weight_updates(weight_grad, alpha)

    return theta.flatten()