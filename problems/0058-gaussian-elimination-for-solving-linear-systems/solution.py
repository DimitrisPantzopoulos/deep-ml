import numpy as np

def gaussian_elimination(A : np.ndarray, b : np.ndarray):
	"""
	Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
	:param A: Coefficient matrix
	:param b: Right-hand side vector
	:return: Solution vector x
	"""

	A : np.ndarray = np.array(A, dtype=float, copy=True)
	b : np.ndarray = np.array(b, dtype=float, copy=True)

	rows, cols = A.shape

	augmented_matrix : np.ndarray = np.concatenate([A, b.reshape(-1, 1)], axis=1)

	# Reduction into row echelon form
	for j in range(cols - 1):
		# Find the pivot's point
		pivot_row : int = j + np.argmax(np.abs(augmented_matrix[j:, j]))

		# Find the pivot
		pivot : float = augmented_matrix[pivot_row, j]

		# Swap the rows
		augmented_matrix[[pivot_row, j]] = augmented_matrix[[j, pivot_row]]

		# Calculate the reduction
		factors   : np.ndarray = augmented_matrix[j + 1:, j].copy()
		reduction : np.ndarray = factors[:, None] * augmented_matrix[j, :] * (1 / pivot)

		# Reduce the rows beneath
		augmented_matrix[j + 1:, :] -= reduction

	# Backwards Substitution
	A = augmented_matrix[:, :-1]
	b = augmented_matrix[:, -1]

	x = np.zeros(rows)

	for k in range(rows - 1, -1, -1):
		known_factors : np.ndarray = A[k, k + 1:] @ x[k + 1:]
		x[k] = (b[k] - known_factors) / A[k, k]
	
	return x
