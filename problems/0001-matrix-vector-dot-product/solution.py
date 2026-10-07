def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	matrix_n : int = len(a)
	matrix_m : int = len(a[0])

	vector_n : int = len(b)

	if matrix_m != vector_n:
		return -1

	matrix_vec : list[list[int|float]] = [0 for _ in range(matrix_n)]

	# [  0  1  2
	# 	[1, 2, 3] 0, 
	# 	[4, 5, 6] 1,
	# ]

	for i in range(matrix_n):
		matrix_vec[i] = sum([a[i][j] * b[j] for j in range(matrix_m)])
	
	return matrix_vec





