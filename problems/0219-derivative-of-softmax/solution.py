import numpy as np

def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	# Your code here
	x : np.ndarray = np.array(x, dtype=np.float32, copy=True)

	# Compute Softmax
	softmax : np.ndarray = np.exp(x) / (np.sum(np.exp(x)))

	# Use Jacobian Formula: J = diag(s) - s * s^T
	return np.diag(softmax) - np.outer(softmax, softmax.T)