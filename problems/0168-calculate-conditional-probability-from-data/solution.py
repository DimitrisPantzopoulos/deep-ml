import numpy as np

def conditional_probability(data, x, y):
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    
    # Your code here
    data : np.ndarray = np.array(data)

    X : np.ndarray = data[:, 0]
    Y : np.ndarray = data[:, 1]

    P_AB : np.int32 = np.sum((X == x) & (Y == y))
    P_B  : np.int32 = np.sum((X == x))

    if P_B == 0:
      return 0.0
      
    return P_AB / P_B

    