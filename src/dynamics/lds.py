"""
lds.py
Linear dynamical system.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class LDS(eqx.Module):
    J: jax.Array

    @classmethod
    def from_params(cls, J: jax.Array):
        return cls(J=J)

    def deer_fxn(self, state, input):
        """
        Arg:
            state: (D,) x_t
            input: (d_input,) u_t
        Returns:
            out: (D,) x_{t+1}
        """
        out = self.J @ state
        return out

    def scan_fxn(self, state, input):
        out = self.deer_fxn(state, input)
        return out, out
