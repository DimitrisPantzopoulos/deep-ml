import numpy as np

ALPHA  : float = 1.67326
LAMBDA : float = 1.0507

def activation(x : np.ndarray) -> np.ndarray:
    '''
    Apply an activation function element-wise.
    
    Args:
        x: numpy array of any shape (raw neuron outputs)
    
    Returns:
        numpy array of same shape (activated outputs)
    
    Requirements:
        - Must be non-linear (not just returning x)
        - Must work on arrays of any shape
        - Must be deterministic
    '''
    # TODO: Implement your activation function
    
    result : np.ndarray = np.where(
        x <= 0, LAMBDA * ALPHA * (np.exp(x) - 1), LAMBDA * x 
    )
    
    return result
