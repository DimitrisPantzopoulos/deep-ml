def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	# Shape of matrix(m x n):
	m, n = len(matrix), len(matrix[0])

	means : list[list[float]] = []

	if mode == 'row':
		means = [(1 / n) * (sum([matrix[i][j] for j in range(n)])) for i in range(m)]
	elif mode == 'column':
		means = [(1 / m) * (sum([matrix[i][j] for i in range(m)])) for j in range(n)]
	return means