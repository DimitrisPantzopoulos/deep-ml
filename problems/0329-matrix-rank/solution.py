import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix.
    
    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering values as zero
    
    Returns:
        The rank of the matrix (integer)
    """

    # Your code here
    A = np.array(A, dtype=float, copy=True)

    rows, cols = A.shape

    successful_pivot : int = 0

    for i in range(cols):
        if successful_pivot == rows:
            break

        # Find the pivot and it's index
        pivot_idx : float = successful_pivot + np.argmax(abs(A[successful_pivot:, i]))

        pivot : float = A[pivot_idx, i]

        if abs(pivot) <= tol: 
            continue

        # This is what we will use to reduce the rows it's  (1 / pivot) * pivot's row. Only thing we need to do is multiply this by the value below the pivot.
        reduction : float = (1 / pivot) * A[pivot_idx, :]

        # Switch the pivots rows if needed
        A[[i, pivot_idx]] = A[[pivot_idx, i]]

        # Don't include the current row
        for j in range(successful_pivot + 1, rows):
            A[j, :] -= A[j, i] * reduction

        successful_pivot += 1
        
    return np.count_nonzero(np.any(np.abs(A) > tol, axis=1))




