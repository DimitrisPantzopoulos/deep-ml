import numpy as np

def calculate_covariance(z_score_matrix : np.ndarray, i : int, j : int) -> float:
	_, n_obs = z_score_matrix.shape

	return float(np.sum(z_score_matrix[i, :] * z_score_matrix[j, :]) / (n_obs - 1))

def calculate_correlation_matrix(X : np.ndarray, Y : np.ndarray | None=None):
	# Your code here
	
	# Calculate Covariance
	matrix : np.ndarray = np.transpose(X if Y is None else np.concatenate([X, Y], axis=1))

	# Calculate row averages
	row_averages : np.ndarray = np.mean(matrix, axis=1)

	# Calculate each elements z-score
	centered : np.ndarray = matrix - row_averages[:, None]

	# Calculate the Covariance of each element in the covariance matrix
	n_features, _ = centered.shape

	# Instiatiate an empty correlation matrix
	correlation : np.ndarray = np.zeros((n_features, n_features))

	stds : np.ndarray = np.std(matrix, axis=1, ddof=1)

	for i in range(0, n_features):
		for j in range(0, n_features):
			conv : np.ndarray = calculate_covariance(centered, i, j)
			correlation[i, j] = conv / (
				(stds[i] * stds[j])
			)

	if Y is not None:
		n_x_features = X.shape[1]
		return correlation[:n_x_features, n_x_features:]

	return correlation