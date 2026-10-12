import numpy as np

def sigmoid(x : np.float32) -> np.float32:
	return 1 / (1 + np.exp(-x))

def d_sigmoid(x : np.float32) -> np.float32:
	s : np.float32 = sigmoid(x)

	return s * (1 - s)

def d_tanh(x : np.ndarray) -> np.float32:
	t : np.ndarray = np.tanh(x)

	return 1 - np.square(t)

def d_relu(x : np.float32) -> np.float32:
	return x > 0

def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	# Your code here
	return {
		'sigmoid' : d_sigmoid(x),
		'tanh'	  : d_tanh(x),
		'relu'    : d_relu(x),
	}