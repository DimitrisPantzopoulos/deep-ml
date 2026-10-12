import numpy as np

def pos_encoding(position : int, d_model: int):
	# Your code here
	# Handle Edge Cases
	if position == 0 or d_model <= 0:
		return -1
	
	# Create pos_encoding matrix (position, d_model)
	pos_encoding : np.ndarray = np.zeros((position, d_model), dtype=np.float16)

	pos, dim = np.indices(pos_encoding.shape)

	# Calculate PE(pos, 2i)=sin(pos / (10000 ** (2i / d_model)))
	pos_encoding[:, 0::2] = np.sin(
		pos[:, 0::2] / (10_000 ** (dim[:, 0::2] / d_model))
	)

	# Calculate PE(pos, 2i + 1)=cos(pos / (10000 ** (2i / d_model)))
	pos_encoding[:, 1::2] = np.cos(
		pos[:, 1::2] / (10_000 ** ((dim[:, 0::2]) / d_model))
	)

	return pos_encoding


