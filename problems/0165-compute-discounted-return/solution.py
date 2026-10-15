import numpy as np

def discounted_return(
        rewards : list[float] | np.ndarray, 
        gamma   : float
    ) -> np.ndarray:
    """
    Compute the total discounted return for a sequence of rewards.
    Args:
        rewards (list or np.ndarray): List or array of rewards [r_0, r_1, ..., r_T-1]
        gamma (float): Discount factor (0 < gamma <= 1)
    Returns:
        float: Total discounted return
    """
    # Your code here
    rewards : np.ndarray = np.array(rewards, dtype=np.float32)

    expo : np.ndarray = np.arange(0, rewards.size, step=1)

    rewards = rewards * (gamma ** expo)

    return np.sum(rewards)

