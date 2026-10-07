import numpy as np

def l2_norm(gradient : np.ndarray) -> np.float32:
	return np.sqrt(np.sum(np.square(gradient)))

def gradient_ascent_direction(gradient : np.ndarray) -> np.ndarray:
	return gradient / ((np.linalg.norm(gradient)) + 1e-9)

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	# Your code here
	greatest_ascent  : np.ndarray = gradient_ascent_direction(gradient)
	greatest_descent : np.ndarray = -greatest_ascent
	magnitude 		 : np.ndarray = l2_norm(gradient)

	return {
		'magnitude' : magnitude,
		'direction' : greatest_ascent,
		'descent_direction' : greatest_descent,
	}