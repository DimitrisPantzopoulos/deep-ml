import jax
import jax.numpy as jnp

def replace_at(
        x       : jnp.array, 
        indices : list[int], 
        values  : list[float]
    ) -> jnp.array:
    """
    Return a copy of 1-D array x with x[indices] replaced by values.
    Must not modify x.
    """
    x = x.at[jnp.array(indices)].set(values)
    return x
