import numpy as np

def generate_rotation_matrix(theta : np.float32) -> np.ndarray:
    sin_theta : np.float32 = np.sin(theta)
    cos_theta : np.float32 = np.cos(theta)

    return np.array([
        [cos_theta, -sin_theta],
        [sin_theta,  cos_theta],
    ])

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    # Your code here
    B : np.ndarray = np.matmul(A.T, A)

    # Find the angle that makes B a diagonal
    theta : np.float32 = (1 / 2) * np.arctan2(2 * (B[0, 1]), B[0, 0] - B[1, 1])

    # Build the rotation matrix V
    V : np.ndarray = generate_rotation_matrix(theta)

    # Diagonal
    D : np.ndarray = V.T @ B @ V

    S : np.ndarray = np.sqrt(np.diag(D))

    U : np.ndarray = (A @ V) / S[None, :]

    return U, S, V.T
