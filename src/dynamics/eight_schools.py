"""
eight_schools.py
Eight Schools Gibbs sampler as a dynamical system for DEER.

The eight schools problem is a classic Bayesian hierarchical model.
The Gibbs sampler is treated as a discrete-time dynamical system:
    state = (thetas, sigmasqs, mu, tausq)
    driver = PRNG key (used to sample noise)

State dimension: D = 2*S + 2 = 18 (S=8 schools).
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

# --- Default data and hyperparameters ---

S   = 8
N_s = 20

DEFAULT_X_BARS     = jnp.array([28., 8., -3., 7., -1., 1., 18., 12.])
DEFAULT_SIGMA_BARS = jnp.array([15., 10., 16., 11., 9., 11., 10., 18.])

DEFAULT_MU_0      = 0.0
DEFAULT_KAPPA_0   = 0.1
DEFAULT_NU_0      = 0.1
DEFAULT_TAUSQ_0   = 100.0
DEFAULT_ALPHA_0   = 0.1
DEFAULT_SIGMASQ_0 = 10.0


def make_gibbs_step(S, N_s, xs, mu_0=DEFAULT_MU_0, kappa_0=DEFAULT_KAPPA_0,
                    nu_0=DEFAULT_NU_0, tausq_0=DEFAULT_TAUSQ_0,
                    alpha_0=DEFAULT_ALPHA_0, sigmasq_0=DEFAULT_SIGMASQ_0):
    """
    Gibbs-sampler step (state, driver) -> new_state for observed data xs (N_s, S).

    xs may be a traced array, so one jitted function can serve every dataset
    of a given (S, N_s) instead of recompiling per dataset.
    """
    nu_N    = nu_0 + S + 1
    alpha_N = alpha_0 + N_s

    def gibbs_sample_tausq(mu, thetas, zs_):
        tausq_N = (1.0 / nu_N) * (nu_0 * tausq_0 + kappa_0 * (mu - mu_0)**2
                                   + jnp.sum((thetas - mu)**2))
        return 0.5 * nu_N * tausq_N * zs_

    def gibbs_sample_mu(tausq, thetas, zs_):
        v_mu   = tausq / (kappa_0 + S)
        mu_hat = (mu_0 * kappa_0 + thetas.sum()) / (kappa_0 + S)
        return mu_hat + jnp.sqrt(v_mu) * zs_

    def gibbs_sample_thetas(tausq, mu, sigmasqs, zs_):
        v_theta   = 1.0 / (N_s / sigmasqs + 1.0 / tausq)
        theta_hat = v_theta * (xs.sum(axis=0) / sigmasqs + mu / tausq)
        return theta_hat + jnp.sqrt(v_theta) * zs_

    def gibbs_sample_sigmasq(thetas, zs_):
        sigmasq_N = (1.0 / alpha_N) * (alpha_0 * sigmasq_0
                                       + jnp.sum((xs - thetas)**2, axis=0))
        return 0.5 * alpha_N * sigmasq_N * zs_

    def deer_fxn(state, driver):
        thetas   = state[:S]
        sigmasqs = state[S:2*S]
        mu       = state[2*S:2*S+1]
        tausq    = state[2*S+1:]

        key_, *skeys = jr.split(driver, 5)
        zs_thetas  = jr.normal(skeys[0], (S,))
        zs_sigmasq = tfd.InverseGamma(0.5 * alpha_N, 1).sample((S,), seed=skeys[1])
        zs_mu      = jr.normal(skeys[2], (1,))
        zs_tausq   = tfd.InverseGamma(0.5 * nu_N, 1).sample((1,), seed=skeys[3])

        tausq    = gibbs_sample_tausq(mu, thetas, zs_tausq)
        mu       = gibbs_sample_mu(tausq, thetas, zs_mu)
        thetas   = gibbs_sample_thetas(tausq, mu, sigmasqs, zs_thetas)
        sigmasqs = gibbs_sample_sigmasq(thetas, zs_sigmasq)

        return jnp.concatenate((thetas, sigmasqs, mu, tausq))

    return deer_fxn


def make_eight_schools(S=S, N_s=N_s, x_bars=None, sigma_bars=None, seed=13,
                       mu_0=DEFAULT_MU_0, kappa_0=DEFAULT_KAPPA_0,
                       nu_0=DEFAULT_NU_0, tausq_0=DEFAULT_TAUSQ_0,
                       alpha_0=DEFAULT_ALPHA_0, sigmasq_0=DEFAULT_SIGMASQ_0):
    """
    Create the eight schools Gibbs sampler dynamics function.

    Args:
        S: number of schools (default 8)
        N_s: number of students per school (default 20)

    Returns:
        deer_fxn: function (state, driver) -> new_state
        params: dict of hyperparameters
        xs: observed data (N_s, S)
        D: state dimension
    """
    key = jr.PRNGKey(seed)
    if x_bars is None:
        if S == 8:
            x_bars = DEFAULT_X_BARS
        else:
            key, k1 = jr.split(key)
            x_bars = jr.normal(k1, (S,)) * 10.0
    if sigma_bars is None:
        if S == 8:
            sigma_bars = DEFAULT_SIGMA_BARS
        else:
            key, k2 = jr.split(key)
            sigma_bars = jnp.abs(jr.normal(k2, (S,)) * 5.0) + 5.0
    xs_raw = tfd.Normal(x_bars, jnp.sqrt(N_s) * sigma_bars).sample((N_s,), seed=key)
    zs     = (xs_raw - xs_raw.mean(axis=0)) / xs_raw.std(axis=0)
    xs     = x_bars + jnp.sqrt(N_s) * sigma_bars * zs

    deer_fxn = make_gibbs_step(S, N_s, xs, mu_0=mu_0, kappa_0=kappa_0, nu_0=nu_0,
                               tausq_0=tausq_0, alpha_0=alpha_0, sigmasq_0=sigmasq_0)
    nu_N    = nu_0 + S + 1
    alpha_N = alpha_0 + N_s
    D       = 2 * S + 2

    params = {
        "S": S, "N_s": N_s,
        "mu_0": mu_0, "kappa_0": kappa_0, "nu_0": nu_0,
        "tausq_0": tausq_0, "alpha_0": alpha_0, "sigmasq_0": sigmasq_0,
        "nu_N": nu_N, "alpha_N": alpha_N,
    }

    return deer_fxn, params, xs, D
