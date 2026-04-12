"""
s5_word.py
S5 symmetric group word problem dynamical system.
"""

import jax.numpy as jnp
import equinox as eqx
from itertools import permutations


class S5_word(eqx.Module):
    perm_mats: jnp.ndarray

    def __init__(self):
        perms = list(permutations(range(5)))
        matrices = []
        for perm in perms:
            mat = jnp.zeros((5, 5), dtype=int)
            mat = mat.at[jnp.arange(5), jnp.array(perm)].set(1)
            matrices.append(mat)
        self.perm_mats = jnp.stack(matrices)  # shape (120, 5, 5)

    def get_transition_matrix(self, input):
        return self.perm_mats[input]

    def deer_fxn(self, state, input):
        transition_matrix = self.get_transition_matrix(input)
        return transition_matrix @ state

    def scan_fxn(self, state, input):
        out = self.deer_fxn(state, input)
        return out, out
