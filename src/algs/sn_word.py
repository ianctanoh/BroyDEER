"""
sn_word.py
Symmetric-group S_N word problem dynamical system.

Same structure as s5_word.py but avoids storing all N! permutation matrices.
Each driver is itself a length-N permutation array:

    x_t = P_{u_t} x_{t-1}    <=>    x_t[i] = x_{t-1}[u_t[i]]

Drivers are shape (T, N) int arrays where each row is a permutation of
range(N). Memory scales as O(T N) instead of O(N! N^2).
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class SN_word(eqx.Module):
    N: int

    def __init__(self, N: int):
        self.N = N

    def deer_fxn(self, state, input):
        """
        Args:
            state: (N,) current state
            input: (N,) permutation array in one-line notation:
                   x_t[i] = state[input[i]]
        Returns:
            (N,) permuted state
        """
        return state[input]

    def scan_fxn(self, state, input):
        out = self.deer_fxn(state, input)
        return out, out


def sample_permutations(key, T, N):
    """Sample T independent uniform permutations of range(N). Shape (T, N)."""
    keys = jr.split(key, T)
    base = jnp.arange(N)
    return jax.vmap(lambda k: jr.permutation(k, base))(keys)
