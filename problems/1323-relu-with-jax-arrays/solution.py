import jax
import jax.numpy as jnp

def relu(x : jnp.array) -> jnp.array:
    """Return x with negative entries replaced by 0 (same shape as x)."""
    return jnp.maximum(x, 0)
