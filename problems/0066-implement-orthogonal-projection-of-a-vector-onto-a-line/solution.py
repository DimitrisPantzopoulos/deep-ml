import numpy as np

def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	v : np.ndarray = np.array(v, dtype=float, copy=True)
	L : np.ndarray = np.array(L, dtype=float, copy=True)

	return (np.dot(v, L) / np.square(np.linalg.norm(L))) * L 
