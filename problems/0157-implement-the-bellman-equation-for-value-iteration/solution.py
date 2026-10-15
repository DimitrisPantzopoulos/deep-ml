import numpy as np

def bellman_update(
      state_values : np.ndarray, 
      transitions  : list[dict[int, list[tuple[float, int, float, bool]]]],
      gamma        : float
    ) -> np.ndarray:
    """
    Perform one step of value iteration using the Bellman equation.
    Args:
      transition_values: np.ndarray, state values, shape (n_states,)
      transitions: list of dicts. transitions[s][a] is a list of (prob, next_state, reward, done)
      gamma: float, discount factor
    Returns:
      np.ndarray, updated state values
    """
    # TODO: Implement Bellman update
    new_state_values : np.ndarray = np.array(state_values, copy=True)

    for state in range(len(state_values)): 
      actions : dict[int, list[tuple[float, int, float, bool]]] = transitions[state]

      best_action_value : float = -np.inf

      for future_state in actions.values():
        action_value : float = 0.0

        for probability, next_state, reward, done in future_state:
          future_value : float = state_values[next_state] if not done else 0.0

          action_value += probability * (reward + gamma * future_value)

        best_action_value = max(best_action_value, action_value)
        
      new_state_values[state] = best_action_value

    return new_state_values
      


    