import numpy as np

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	Compute the derivative of cross-entropy loss with respect to logits.
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
		
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""
	# Your code here
	logits : np.ndarray = np.array(logits, dtype=np.float32, copy=True)

	# Softmax logits
	softmax : np.ndarray = np.exp(logits) / (np.sum(np.exp(logits)))

	# subtract the ohe target vector. In this case we only have one target so we can just subtract 1 from the target index
	softmax[target] -= 1

	return softmax