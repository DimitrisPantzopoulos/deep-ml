import numpy as np

def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
	"""
	Calculate posterior probabilities using Bayes' Theorem.
	
	Args:
		priors: Prior probabilities P(H_i) for each hypothesis
		likelihoods: Likelihoods P(E|H_i) for each hypothesis
		
	Returns:
		Posterior probabilities P(H_i|E) for each hypothesis
	"""
	priors 		: np.ndarray = np.array(priors)
	likelihoods : np.ndarray = np.array(likelihoods)

	LTP : np.float32 = np.sum(priors * likelihoods)

	return (priors * likelihoods) / LTP