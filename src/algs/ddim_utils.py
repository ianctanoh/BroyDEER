"""
ddim_utils.py
Helpers shared by the DDIM-specific solvers.
"""

from typing import Optional
import jax.numpy as jnp


def merit_from_residual(r: jnp.ndarray, Ts: Optional[jnp.ndarray] = None):
    """
    Merit L = (1/2) sum_t sum_{s<=t} ||r_s||^2 from precomputed residuals.

    Same value as deer.merit_fxn, but takes the residuals r (T, D) directly so
    the DDIM solvers can reuse the eps evaluation they already made in the
    current step instead of running the model again.
    """
    Ls = 0.5 * jnp.sum(jnp.cumsum(r**2, axis=0), axis=1)  # (T,)
    Ls = jnp.where(jnp.isnan(Ls), jnp.inf, Ls)
    if Ts is not None:
        return Ls[Ts - 1]
    return Ls[-1]
