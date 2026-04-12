"""
lle.py
Helper functions to estimate the largest Lyapunov exponent (LLE) of a dynamical system.

Note that the functions:
- estimate_lle_from_vector
- estimate_lle_from_jacobians
- wrapper_estimate_lle_from_jacobians
All work for discrete-time systems (i.e. dt=1)

The function estimate_lle_and_overshoot works for both discrete-time and
continuous-time systems, as it takes in a time scale dt as an argument.
"""

import jax
import jax.random as jr
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp


def estimate_lle_from_vector(jacobians, v, Ts=None):
    """
    Estimate the (max) LLE given a sequence of Jacobians and an initial vector v.
    """
    D = jacobians.shape[-1] if jacobians.ndim >= 2 else 1
    v = v / jnp.linalg.norm(v)

    def body_fn(carry, J):
        v, logsum = carry
        v = J[0, 0] * v if D == 1 else J @ v
        n = jnp.linalg.norm(v)

        logsum += lax.cond(
            n == 0.0,
            lambda _: -jnp.inf,
            lambda n_: jnp.log(n_),
            n,
        )
        v = lax.cond(
            n == 0.0,
            lambda _: v,
            lambda n_: v / n_,
            n,
        )
        return (v, logsum), logsum

    (v_final, lyap_sum), step_logs = lax.scan(body_fn, (v, 0.0), jacobians)

    if Ts is not None:
        return step_logs[Ts] / (Ts + 1)
    return lyap_sum / jacobians.shape[0]


def estimate_lle_from_jacobians(jacobians, key, Ts=None):
    """
    Estimate the largest Lyapunov exponent (lambda) from a collection of Jacobian matrices,
    for a specific random key.
    """
    D = jacobians.shape[-1] if jacobians.ndim >= 2 else 1
    v = jr.uniform(key, (D,), minval=-1.0, maxval=1.0)
    return estimate_lle_from_vector(jacobians, v, Ts=Ts)


def wrapper_estimate_lle_from_jacobians(jacobians, key, Ts=None, numkeys=3):
    """
    Wrapper to estimate LLE averaged over numkeys random keys.
    """
    keys = jr.split(key, numkeys)
    lle_fn = lambda k: estimate_lle_from_jacobians(jacobians, k, Ts)
    lles = jax.vmap(lle_fn)(keys)
    return jnp.mean(lles, axis=0)


def estimate_rho_from_jacobians(jacobians, key):
    """Estimate rho_eff from a collection of Jacobian matrices."""
    lle = estimate_lle_from_jacobians(jacobians, key)
    return jnp.exp(lle)


def log_mu_numerically_stable(LLE, T, a=1.0):
    x = LLE * jnp.arange(T)
    return -2 * (jnp.log(1 / a) + logsumexp(x))


def log_mu_from_jacobians(jacobians, key):
    lypaunov_sum = estimate_lle_from_jacobians(jacobians, key)
    return log_mu_numerically_stable(lypaunov_sum, jacobians.shape[0])


def estimate_lle_and_overshoot(jacobians, key, dt, Ts=None):
    """
    Estimate the (max) LLE and overshoot constant 'a' given a sequence of Jacobians.

    Args:
        jacobians: array of shape (T, D, D) or (T,) for scalar case
        key: PRNG key for sampling the initial direction
        dt: time scale of discretization
        Ts: optional array of time indices to return LLE estimates at

    Returns:
        If Ts is None: (lle, a)
        If Ts is not None: (lles, a)
    """
    D = jacobians.shape[-1] if jacobians.ndim >= 2 else 1
    v = jr.normal(key, shape=(D,))
    v = v / jnp.linalg.norm(v)

    def body_fn(carry, J):
        v, logsum = carry
        v = J[0, 0] * v if D == 1 else J @ v
        n = jnp.linalg.norm(v)

        log_n = lax.cond(n == 0.0, lambda _: -jnp.inf, lambda n_: jnp.log(n_), n)
        logsum += log_n

        v = lax.cond(n == 0.0, lambda _: v, lambda n_: v / n_, n)

        return (v, logsum), logsum

    (v_final, lyap_sum), step_logs = lax.scan(body_fn, (v, 0.0), jacobians)

    steps = jnp.arange(len(step_logs)) + 1
    lles = step_logs / steps
    lle_final = lyap_sum / jacobians.shape[0]

    log_scaled = step_logs - lle_final * steps
    log_a = jnp.max(log_scaled)
    a = jnp.exp(log_a)

    if Ts is not None:
        return lles[Ts], a
    return lle_final / dt, a


def get_spectral_norm(x, u, f):
    """Returns spectral norm (largest singular value) of Jacobian of dynamics fxn at x."""
    J = jax.jacobian(f)(x, u)
    return jnp.linalg.norm(J, ord=2)
