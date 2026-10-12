import numpy as np

def calculate_determinant(matrix : np.ndarray) -> np.float32:
	matrix : np.ndarray = np.array(matrix, dtype=np.float32, copy=True)

	rows, cols = matrix.shape
	
	# Base Case
	if (rows == 2 and cols == 2):
		# Calculate the determinant of a 2x2 matrix
		return np.prod(np.diag(matrix)) - np.prod(np.diag(matrix[:, ::-1]))
	
	det : np.float32 = 0.0

	for j in range(cols):
		# Generate Minor
		minor : np.ndarray = np.delete(np.delete(matrix, 0, axis=0), j, axis=1)

		# Calculate Cofactor recursively
		cofactor : np.float32 = ((-1) ** j) * calculate_determinant(minor)
		
		# Multiply cofactor by a[1, j]
		det += cofactor * matrix[0, j]
	
	return det

def determinant_4x4(matrix: list[list[int|float]]) -> float:
	# Your recursive implementation here
	matrix : np.ndarray = np.array(matrix, dtype=np.float32, copy=True)

	return calculate_determinant(matrix)
