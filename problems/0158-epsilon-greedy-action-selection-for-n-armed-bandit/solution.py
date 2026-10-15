import numpy as np

def epsilon_greedy(Q, epsilon : float =0.1):
    """
    Selects an action using epsilon-greedy policy.
    Q: np.ndarray of shape (n,) -- estimated action values
    epsilon: float in [0, 1]
    Returns: int, selected action index
    """
    # Your code here

    # Exploitation
    if epsilon < np.random.rand(1):
        return np.argmax(Q, axis=0)
    else:
        return np.random.randint(0, Q.size)