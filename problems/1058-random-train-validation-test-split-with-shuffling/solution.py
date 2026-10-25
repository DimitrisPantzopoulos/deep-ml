import numpy as np

def random_split(
        data            : np.ndarray, 
        train_frac      : float, 
        validation_frac : float, 
        seed            : int = 123
    ) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    # Your code here
    n, d = data.shape

    rand_idx : np.ndarray = np.random.default_rng(seed).permutation(n)

    train_end      : int = int(n * train_frac)
    validation_end : int = train_end + int(n * validation_frac)

    train      : np.ndarray = data[rand_idx[: train_end]]
    test       : np.ndarray = data[rand_idx[validation_end :]]
    validation : np.ndarray = data[rand_idx[train_end : validation_end]]

    return train, validation, test