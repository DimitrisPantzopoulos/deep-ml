from typing import Callable

import numpy as np

def jacobian_matrix(f : Callable[[list[float]], list[float]], x: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Jacobian matrix using numerical differentiation.
	
	Args:
		f: Function that takes a list and returns a list
		x: Point at which to evaluate the Jacobian
		h: Step size for finite differences
	
	Returns:
		Jacobian matrix as list of lists
	"""
	x   : np.ndarray = np.array(x, dtype=np.float64, copy=True)
	f_x : np.ndarray = np.array(f(x), dtype=np.float64)

	n : int = x.size
	m : int = f_x.size

	jacobian : np.ndarray = np.empty((m, n))

	for j in range(n):
		x_h = x.copy()
		x_h[j] += h

		f_x_h : np.ndarray = np.asarray(f(x_h), dtype=np.float64)
		jacobian[:, j] = (f_x_h - f_x) / h

	return jacobian.tolist()