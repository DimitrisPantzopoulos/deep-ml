import numpy as np

def matrix_rank(vectors : list[list[float]], tol : float=1e-9) -> int:
    if len(vectors) == 0: return 0

    matrix : np.ndarray = np.array(vectors, dtype=float, copy=True)

    rows, cols = matrix.shape

    successful_pivots : int = 0

    for j in range(cols):
        if successful_pivots == rows:
            continue
        
        # Find the pivot and the pivot's idx
        pivot_idx : int   = successful_pivots + np.argmax(np.abs(matrix[successful_pivots:, j]))
        pivot     : float = matrix[pivot_idx, j]

        # Check if the pivot is above the given tolerance
        if abs(pivot) <= tol:
            continue
        
        # Swap the pivot rows and current rows
        matrix[[j, pivot_idx]] = matrix[[pivot_idx, j]]

        # Calculate the reduction
        reduction : float = (1 / pivot) * matrix[j, :]

        for i in range(successful_pivots + 1, rows):
            matrix[i, :] -= matrix[i, j] * reduction
        successful_pivots += 1
    return np.count_nonzero(np.any(np.abs(matrix) > 0, axis=1))


def is_linearly_independent(vectors: list[list[float]]) -> bool:
    """
    Check if a set of vectors is linearly independent.
    
    Args:
        vectors: List of vectors, where each vector is a list of floats.
                 All vectors must have the same dimension.
        
    Returns:
        True if vectors are linearly independent, False otherwise.
    """
    
    # Your code here
    return len(vectors) == matrix_rank(vectors)