import numpy as np

def ucb_action(
      counts          : np.ndarray, 
      values          : np.ndarray, 
      timestep        : int, 
      exploration_coef : float
    ) -> int:
    """
    Choose an action using the UCB1 formula.
    Args:
      counts (np.ndarray): Number of times each action has been chosen
      values (np.ndarray): Average reward of each action
      t (int): Current timestep (starts from 1)
      c (float): Exploration coefficient
    Returns:
      int: Index of action to select
    """
    # TODO: Implement the UCB action selection
    # Exploration
    exploration : np.ndarray = exploration_coef * np.sqrt(np.log(timestep) / counts)

    ucb : np.ndarray = values + exploration

    return np.argmax(ucb)