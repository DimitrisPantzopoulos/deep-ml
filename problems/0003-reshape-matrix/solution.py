import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	# Write your code here and return a python list after reshaping by using numpy's tolist() method

	old_rows : int = len(a)
	old_cols : int = len(a[0])

	new_rows : int = new_shape[0]
	new_cols : int = new_shape[1]

	if (old_rows * old_cols) != (new_rows * new_cols):
		return []
	

	items : list[int|float] = [
		item
		for sub_vector in a
    	for item in sub_vector
	]

	reshaped_matrix : list[int|float] = [
		items[i:i + new_cols]
    	for i in range(0, len(items), new_cols)
	]

	return reshaped_matrix