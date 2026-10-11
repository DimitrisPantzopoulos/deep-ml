import numpy as np

def poly2d(point : np.ndarray) -> np.ndarray:
	x, y = point

	return (2 * x * y + y **2), (x ** 2 + 2 * x * y)

def exp_sum(point : np.ndarray) -> np.ndarray:
	x, y = point

	return (np.exp(x + y)), (np.exp(x + y))

def product_sin(point : np.ndarray) -> np.ndarray:
	x, y = point

	return np.sin(y), x * np.cos(y)

def poly_3d(point : np.ndarray) -> np.ndarray:
	x, y, z = point

	return 2 * x * y, x ** 2 + z ** 2, 2 * y * z

def squared_error(point : np.ndarray) -> np.ndarray:
	x, y = point

	return (2 * (x - y), 2 * (x - y) * (-1))

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
	"""
	Compute partial derivatives of multivariable functions.
	
	Args:
		func_name: Function identifier
			'poly2d': f(x,y) = x²y + xy²
			'exp_sum': f(x,y) = e^(x+y)
			'product_sin': f(x,y) = x·sin(y)
			'poly3d': f(x,y,z) = x²y + yz²
			'squared_error': f(x,y) = (x-y)²
		point: Point (x, y) or (x, y, z) at which to evaluate
	
	Returns:
		Tuple of partial derivatives (∂f/∂x, ∂f/∂y, ...) at point
	"""

	# Your code here
	point : np.ndarray = np.array(point, dtype=np.float32, copy=True)

	match func_name:
		case 'poly2d':
			return poly2d(point)
		case 'exp_sum':
			return exp_sum(point)
		case 'product_sin':
			return product_sin(point)
		case 'poly3d':
			return poly_3d(point)
		case 'squared_error':
			return squared_error(point)
		case _:
			raise ValueError(f"Unknown function: {func_name}")
