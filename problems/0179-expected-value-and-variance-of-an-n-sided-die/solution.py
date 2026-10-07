def expected_value_x_square(n : int) -> float:
	return (1 / 6) * ((n + 1) * (2 * n + 1))

def expected_value(n : int) -> float:
	return (1 / 2) * (n + 1)

def dice_statistics(n: int) -> tuple[float, float]:
	"""
	Compute the expected value and variance of a fair n-sided die roll.

	Args:
		n (int): Number of sides of the die

	Returns:
		tuple: (expected_value, variance)
	"""
	# Your code here
	expected_value_n : float = expected_value(n)

	var_n : float = expected_value_x_square(n) - (expected_value_n) ** 2
	
	return (expected_value_n, var_n)