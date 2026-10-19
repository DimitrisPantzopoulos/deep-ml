def calculate_huber_loss(
		y_true : float, 
		y_pred : float, 
		delta  : float = 1.0
	) -> float:
	error : float = y_true - y_pred

	abs_error : float = abs(error)

	if abs_error <= delta:
		return 0.5 * error ** 2
	return delta * (abs_error - 0.5 * delta)

def huber_loss(
		y_true : float | list[float], 
		y_pred : float | list[float], 
		delta  : float = 1.0
	):
	"""
	Compute the Huber Loss between true and predicted values.

	Args:
		y_true (float | list[float]): Ground truth values
		y_pred (float | list[float]): Predicted values
		delta (float): Transition threshold between MSE and MAE behavior

	Returns:
		float: Average Huber loss
	"""
	# Your code here
	error : float = 0.0

	if isinstance(y_pred, list):
		for true, pred in zip(y_true, y_pred):
			error += calculate_huber_loss(true, pred, delta)
		
		return error / len(y_pred)
	
	return calculate_huber_loss(y_true, y_pred, delta)