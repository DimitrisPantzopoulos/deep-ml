import numpy as np

def calculate_covariance(z_score_matrix : np.ndarray, i : int, j : int) -> float:
	_, cols = z_score_matrix.shape

	return float((np.sum(z_score_matrix[i, :] * z_score_matrix[j, :])) / (cols - 1))

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
	# Convert the list of vectors into a numpy matrix.
	matrix : np.ndarray = np.array(vectors)

	# Find the averages row by row
	row_avgs : np.ndarray = np.mean(matrix, axis=1)

	# Subtract the row avg's from their given rows
	z_score_matrix : np.ndarray = matrix - row_avgs[:, None]

	# Calculate each number in the covariance matrix
	rows, cols = z_score_matrix.shape

	return [
		[
			calculate_covariance(z_score_matrix, i, j) for j in range(0, rows)
		] for i in range(0, rows)
	]