import numpy as np

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    """
    Simulate the Central Limit Theorem.

    Args:
        distribution (str): The distribution to sample from ('uniform', 'exponential', 'bernoulli').
        n (int): Sample size.
        runs (int): Number of repeated experiments.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: {'mean': float, 'std': float} of the standardized sample means.
    """
    np.random.seed(seed)
    
    dist  : np.ndarray
    mu    : float
    sigma : float

    match distribution:
        case 'uniform':
            dist  = np.random.uniform(0, 1, size=(runs, n))
            mu    = 0.5
            sigma = np.sqrt(1/12)
        case 'exponential':
            dist = np.random.exponential(1.0, size=(runs, n))
            mu    = 1.0
            sigma = 1.0
        case 'bernoulli':
            dist = np.random.rand(runs, n) < 0.3
            mu    = 0.3
            sigma = np.sqrt(0.3 * 0.7)
        case _:
            raise ValueError

    sample_mean = np.mean(dist, axis=1)
    standardized = (sample_mean - mu) * np.sqrt(n) / sigma

    return {
        'mean': float(standardized.mean()),
        'std': float(standardized.std()),
    }