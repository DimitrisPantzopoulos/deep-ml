def build_M_ij(matrix : list[list[float]], j : int, rows : int, cols : int) -> list[list[float]]:
	cols : int = len(matrix[0])

	M_ij_dim : int = cols - 1

	# This removes the top row and the column k and return all elements in order.
	items : list[float] = [
		matrix[r][c]
		for r in range(1, rows)
		for c in range(cols)
        if (c != j)
	]

	return [
		items[i:i+M_ij_dim]
		for i in range(0, len(items), M_ij_dim)
	]


def calculate_determinant(matrix : list[list[float]]) -> float:
	rows : int = len(matrix)
	cols : int = len(matrix[0])

	# Base Case: Matrix has dimensions (2 x 2)
	if (rows == 2) and(cols == 2):
		return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

	det : float = 0.0
	
	for j in range(cols):
		det += (-1) ** (j) * matrix[0][j] * calculate_determinant(build_M_ij(matrix, j, rows, cols))

	return det

def calculate_trace(matrix : list[list[float]]) -> float:
	rows : int = len(matrix)

	trace : float = sum([matrix[i][i] for i in range(rows)])

	return trace

def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
	"""
	Compute the determinant and trace of a square matrix.
	
	Args:
		matrix: A square matrix (n x n) represented as list of lists
	
	Returns:
		Tuple of (determinant, trace)
	"""
	# Your code here
	det   : float = calculate_determinant(matrix)
	trace : float = calculate_trace(matrix)

	return (det, trace)
