"""
gru.py
GRU dynamical system wrapper.
"""

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import equinox as eqx
from equinox.nn import GRUCell


class GRU(eqx.Module):
    D: int
    gru_cell: eqx.Module

    def __init__(self, D: int, key: jr.PRNGKey, gain: float = 1.0):
        self.D = D
        cell = GRUCell(D, D, key=key)
        if gain != 1.0:
            cell = jtu.tree_map(lambda x: x * gain, cell)
        self.gru_cell = cell

    def deer_fxn(self, state, input):
        """
        Arg:
            state: (D,) x_t
            input: (d_input,) u_t
        Returns:
            out: (D,) x_{t+1}
        """
        out = self.gru_cell(state, input)
        return out

    def scan_fxn(self, state, input):
        out = self.deer_fxn(state, input)
        return out, out
