import numpy as np

def train(
        X : np.ndarray, 
        y : np.ndarray, 
        W : np.ndarray, 
        b : float
    ) -> (np.ndarray, np.float32):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # Initialise Weights and Biases
    weights : np.ndarray = np.copy(W)
    bias    : np.float32 = np.float32(b)

    # Initialise Inputs and Outputs
    features : np.ndarray = np.copy(X)
    targets  : np.ndarray = np.copy(y)

    # Initialise Hyperparameters
    EPOCHS : int   = 1000
    LR     : float = 1e-3

    # Start Training
    for epoch in range(EPOCHS):
        # Forward Pass - Get the model's current predictions
        y_pred : np.ndarray = X @ weights + bias

        # Measure the loss wrt to the weights and biases
        dMSE_wrt_z : np.ndarray = (2 / targets.size) * (y_pred - targets)

        dMSE_wrt_weights : np.ndarray = features.T @ dMSE_wrt_z
        dMSE_wrt_bias    : np.float32 = np.sum(dMSE_wrt_z)

        # Update the weights and biases
        weights -= LR * dMSE_wrt_weights
        bias    -= LR * dMSE_wrt_bias

    return weights, bias