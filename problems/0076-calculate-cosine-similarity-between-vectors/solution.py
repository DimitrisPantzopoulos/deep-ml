import numpy as np

def l2_norm(arr : np.ndarray) -> np.float32:
	return np.sqrt(np.sum(np.square(arr)))

def cosine_similarity(v1 : np.ndarray, v2 : np.ndarray) -> float:
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	# Implement your code here
	if v1.ndim != v2.ndim:
		raise ValueError
	
	return (np.sum(v1 * v2)) / (l2_norm(v1) * l2_norm(v2))