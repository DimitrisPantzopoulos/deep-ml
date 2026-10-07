def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	# Your code here
	for i, row in enumerate(matrix):
		for j, _ in enumerate((row)):
			matrix[i][j] *= scalar

	return matrix