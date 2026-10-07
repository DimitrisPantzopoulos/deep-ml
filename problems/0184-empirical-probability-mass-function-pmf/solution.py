from collections import Counter

def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    counter : Counter = Counter(samples)

    total_samples : int = len(samples)

    return sorted([(item, (frequency / total_samples)) for item, frequency in counter.items()], key=lambda x: x[0], reverse=False)
