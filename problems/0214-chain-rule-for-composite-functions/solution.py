import numpy as np

def eval_func(function : str, x : float) -> np.float32:
	match function:
		case 'square':
			return x ** 2
		case 'sin':
			return np.sin(x)
		case 'exp':
			return np.exp(x)
		case 'log':
			return np.log(x)
		case _:
			raise ValueError("Invalid Function")

def eval_derivative(function : str, x : float) -> np.float32:
	match function:
		case 'square':
			return 2 * x
		case 'sin':
			return np.cos(x)
		case 'exp':
			return np.exp(x)
		case 'log':
			return 1 / x
		case _:
			raise ValueError("Invalid Function")

def evaluate_derivative(functions : list[str], x : float) -> float:
	# Chain rule is h = f(g(x)), dh/dx = f'(g(x)) * g'(x)
	value    : float = x
	gradient : float = 1

	for function in reversed(functions):
		gradient *= eval_derivative(function, value)
		value = eval_func(function, value)
	
	return gradient

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""
	# Your code here
	return evaluate_derivative(functions, x)