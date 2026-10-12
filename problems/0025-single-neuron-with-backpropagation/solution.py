import numpy as np

def train_neuron(
		features 		: np.ndarray, 
		labels   		: np.ndarray, 
		initial_weights : np.ndarray, 
		initial_bias	: float, 
		learning_rate	: float, 
		epochs			: int
	) -> (np.ndarray, float, list[float]):

	mse_values : np.ndarray = np.zeros((epochs))

	weights : np.ndarray = np.copy(initial_weights)
	bias    : np.float32 = np.float32(initial_bias)

	n : int = len(labels)

	for epoch in range(epochs):
		# Forward pass
		y_pred : np.ndarray = features @ weights + bias

		# Compute Sigmoid Probabilities
		sigmoid : np.ndarray = 1 / (1 + np.exp(-y_pred))

		# Compute MSE
		mse_values[epoch] = (1 / n) * np.sum(np.square(sigmoid - labels))

		# Compute derivatives
		dMSE_wrt_z : np.ndarray = (2 / n) * (sigmoid - labels) * sigmoid * (1 - sigmoid)

		dMSE_wrt_w : np.ndarray = features.T @ dMSE_wrt_z

		dMSE_wrt_b : np.ndarray = np.sum(dMSE_wrt_z)

		weights -= learning_rate * dMSE_wrt_w
		bias    -= learning_rate * dMSE_wrt_b
		
	# Your code here
	return weights, bias, mse_values