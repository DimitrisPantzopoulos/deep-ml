import numpy as np

def l1_norm(arr : np.ndarray) -> np.ndarray:
    return np.sum((np.abs(arr)))

def l2_norm(arr : np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.square(arr)))

def linf_norm(arr : np.ndarray) -> np.ndarray:
    return np.max(np.abs(arr))

def fronebius_norm(arr : np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError("Expected a 2d matrix")
    
    return np.sqrt(np.sum(np.square(np.abs(arr))))

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D array.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    # Your code here
    if norm_type == 'l1':
        return float(l1_norm(arr))
    elif norm_type == 'l2':
        return float(l2_norm(arr))
    elif norm_type == 'linf':
        return float(linf_norm(arr))
    elif norm_type == 'frobenius':
        return float(fronebius_norm(arr))
    
    raise ValueError("Invalid Input")
