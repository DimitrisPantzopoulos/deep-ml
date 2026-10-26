import numpy as np

def standard_scaler(
        X_train : np.ndarray, 
        X_test  : np.ndarray
    ) -> np.ndarray:
    """
    Fit a standard scaler on X_train and transform X_test.
    Returns the standardized X_test as a numpy array.
    """

    train_mean : np.float32 = X_train.mean(axis=0)
    train_std  : np.float32 = X_train.std(axis=0)

    train_std = np.where(train_std != 0, train_std, 1)
    return (X_test - train_mean) / train_std
    
