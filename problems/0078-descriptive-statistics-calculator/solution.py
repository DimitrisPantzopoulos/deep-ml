import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Your code here
    data : np.ndarray = np.array(data)

    q3_percentile : np.float32 = np.percentile(data, 75)
    q1_percentile : np.float32 = np.percentile(data, 25)

    value, counts = np.unique(data, return_counts=True)
    return {
        'mean'     : np.mean(data),
        'median'   : np.median(data),
        'mode'     : value[np.argmax(counts)],
        'variance' : np.var(data),
        'standard_deviation'  : np.std(data),
        '25th_percentile'     : q1_percentile,
        '50th_percentile'     : np.percentile(data, 50),
        '75th_percentile'     : q3_percentile,
        'interquartile_range' : q3_percentile - q1_percentile
    }