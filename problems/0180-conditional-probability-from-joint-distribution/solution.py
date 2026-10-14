def conditional_probability(joint_distribution: dict) -> float:
	"""
	Compute conditional probability P(A|B) from a joint probability distribution.

	Args:
		joint_distribution (dict): dictionary with keys ('A','B'), ('A','`B'), ('`A','B'), ('`A','`B')

	Returns:
		float: Conditional probability P(A|B)
	"""
	# Your code here
	P_AB : float = joint_distribution[('A', 'B')]
	P_B  : float = joint_distribution[('A', 'B')] + joint_distribution[('`A', 'B')]

	return P_AB / P_B