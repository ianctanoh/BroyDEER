"""
deer_ddim.py
DDIM-specific quasi-DEER with optimized eps evaluation.

Same decomposition as broyden_ddim.py: vmaps only eps_model over T positions,
then applies DDIM coefficient math outside the vmap. The Hutchinson diagonal
Jacobian estimate exploits DDIM structure:

    f(x, t) = a[t]*x + b[t]*eps(x, t) + n[t]*z
    J_f(x, t) = a[t]*I + b[t]*J_eps(x, t)

so diag(J_f) = a[t] + b[t] * diag(J_eps), where diag(J_eps) is estimated
via Hutchinson trace (JVP with Rademacher vectors through eps_model only).

Per step: 1 eps vmap (forward) + 1 JVP vmap (Hutchinson) + 1 eps vmap (merit)
= 2 eps vmaps + 1 JVP vmap, matching the cost structure of broyden_ddim.

API matches deer_alg / broyden_ddim_alg: returns the standard 7-tuple.
"""

from typing import Optional
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jax.lax import scan

from .deer import merit_fxn, diag_mat_operator


def deer_ddim_alg(
    dynamics,
    initial_state: jnp.ndarray,
    states_guess: jnp.ndarray,
    drivers: jnp.ndarray,
    num_iters: int,
    num_samples: int = 1,
    k: float = 0.0,
    clip: bool = False,
    full_trace: bool = False,
    Ts: Optional[jnp.ndarray] = None,
    reset: bool = True,
    tol: float = 5e-4,
    key: Optional[jnp.ndarray] = None,
):
    """
    DDIM-specific quasi-DEER with decomposed eps evaluation.

    Hutchinson diagonal Jacobian estimated via JVP through eps_model only,
    then combined with DDIM coefficients: diag(J_f) = a[t] + b[t]*diag(J_eps).

    Args:
        dynamics      : DDIMDynamics instance
        initial_state : x_T (D,)
        states_guess  : initial trajectory guess (T, D)
        drivers       : inputs (T, D+1)
        num_iters     : maximum iterations
        num_samples   : Rademacher samples for Hutchinson estimate (default 1)
        k             : damping on Jacobian diagonal (default 0.0)
        clip          : if True, clip diagonal to [-1, 1]
        full_trace    : if True, return all intermediate states
        Ts            : optional time indices for multi-checkpoint convergence
        reset         : if True, replace NaN states with states_guess
        tol           : convergence tolerance (on merit function)
        key           : PRNG key for Rademacher vectors

    Returns:
        7-tuple matching deer_alg:
        (all_states, final_state, newton_steps, is_nan, nan_mask,
         all_newtons, mf_val)
    """
    T, D_dim = states_guess.shape

    if key is None:
        key = jr.PRNGKey(0)

    eps_model_local  = dynamics.eps_model
    coef_x_local     = dynamics.coef_x
    coef_eps_local   = dynamics.coef_eps
    coef_noise_local = dynamics.coef_noise

    # Canonical single-step DDIM closure (for merit_fxn)
    def f_local(state, driver):
        z = driver[:D_dim]
        t = driver[D_dim].astype(jnp.int32)
        eps = eps_model_local(state, t)
        return (coef_x_local[t] * state
                + coef_eps_local[t] * eps
                + coef_noise_local[t] * z)

    # Precompute per-step DDIM coefficients in BroDeer index order
    s_idx   = jnp.arange(T)
    t_per_s = T - s_idx                            # (T,) DDIM timesteps
    a_per_s = coef_x_local[t_per_s]                # (T,)
    b_per_s = coef_eps_local[t_per_s]              # (T,)
    n_per_s = coef_noise_local[t_per_s]            # (T,)
    z_per_s = drivers[:, :D_dim]                   # (T, D)

    # Rademacher keys for Hutchinson estimator (one per position, reused)
    keys = jr.split(key, T - 1)

    # JVP through eps_model only
    def eps_jvp(state, t, z_rad):
        """JVP of eps_model w.r.t. state in direction z_rad."""
        return jax.jvp(lambda x: eps_model_local(x, t), (state,), (z_rad,))[1]

    def hutchinson_diag_eps(state, t, key_t):
        """Estimate diag(J_eps) via Hutchinson with Rademacher vectors."""
        z_rad = jr.rademacher(key_t, (num_samples, D_dim)).astype(state.dtype)
        jvps = vmap(eps_jvp, in_axes=(None, None, 0))(state, t, z_rad)  # (num_samples, D)
        return jnp.mean(z_rad * jvps, axis=0)  # (D,)

    @jax.jit
    def _step(carry, args):
        states, is_nan = carry

        # Build full trajectory
        X = jnp.concatenate([initial_state[None], states], axis=0)  # (T+1, D)

        # ONE eps vmap: forward evaluation
        eps_arr = vmap(eps_model_local)(X[:T], t_per_s)              # (T, D)
        f_vals = (a_per_s[:, None] * X[:T]
                  + b_per_s[:, None] * eps_arr
                  + n_per_s[:, None] * z_per_s)                     # (T, D)

        # Hutchinson diagonal of J_eps via JVP (positions 1..T-1)
        diag_J_eps = vmap(hutchinson_diag_eps)(
            X[1:T], t_per_s[1:], keys
        )                                                            # (T-1, D)

        # Full DDIM Jacobian diagonal: diag(J_f) = a + b * diag(J_eps)
        As = a_per_s[1:, None] + b_per_s[1:, None] * diag_J_eps    # (T-1, D)
        As = (1 - k) * As
        if clip:
            As = jnp.clip(As, -1, 1)

        # b = f - A * x_prev
        bs = f_vals[1:] - As * X[1:T]                               # (T-1, D)

        # Position 0: no Jacobian, just f(x_0, u_0)
        A0 = jnp.zeros((1, D_dim))
        b0 = f_vals[0:1]                                            # (1, D)
        A = jnp.concatenate([A0, As], axis=0)                       # (T, D)
        b = jnp.concatenate([b0, bs], axis=0)                       # (T, D)

        # Solve linearized system via associative scan (diagonal case)
        _, new_states = jax.lax.associative_scan(diag_mat_operator, (A, b))

        is_nan = jnp.logical_or(jnp.isnan(new_states).any(), is_nan)
        nan_mask = jnp.isnan(new_states)
        if reset:
            new_states = jnp.where(nan_mask, states_guess, new_states)

        # Convergence check — ONE more eps vmap (inside merit_fxn).
        # Evaluated on pre-update states (same convention as broyden_ddim).
        mf_val = merit_fxn(f_local, initial_state, states, drivers)

        return (new_states, is_nan), (new_states, nan_mask, mf_val)

    # --- convergence loop ---
    def cond_func(iter_inp):
        iter_idx, _, err, *_ = iter_inp
        return jnp.logical_and(iter_idx < num_iters, err > tol)

    def body_func_single(iter_inp):
        iter_idx, states, _, is_nan_, _ = iter_inp
        step_output, step_aux = _step((states, is_nan_), None)
        new_states, is_nan = step_output
        _, _, mf_val = step_aux
        return iter_idx + 1, new_states, mf_val, is_nan, None

    def body_func_multiple(iter_inp):
        iter_idx, states, _, is_nan_, iters_below = iter_inp
        step_output, _ = _step((states, is_nan_), None)
        new_states, is_nan = step_output
        new_errs = merit_fxn(f_local, initial_state, states, drivers, Ts=Ts)
        iters_below = iters_below + (new_errs < tol).astype(int)
        return iter_idx + 1, new_states, new_errs[-1], is_nan, iters_below

    if full_trace:
        last_output, all_outputs = scan(
            _step, (states_guess, False), None, length=num_iters
        )
        final_state, is_nan = last_output
        all_states, nan_mask, mf_val = all_outputs
        newton_steps, iters_below = None, None
        all_states = jnp.concatenate([states_guess[None, ...], all_states])
    elif Ts is not None:
        result = jax.lax.while_loop(
            cond_func,
            body_func_multiple,
            (
                0,
                states_guess,
                merit_fxn(f_local, initial_state, states_guess, drivers),
                False,
                jnp.zeros_like(Ts),
            ),
        )
        newton_steps, final_state = result[0], result[1]
        is_nan, iters_below = result[3], result[4]
        all_states, nan_mask, mf_val = None, None, None
    else:
        result = jax.lax.while_loop(
            cond_func,
            body_func_single,
            (
                0,
                states_guess,
                merit_fxn(f_local, initial_state, states_guess, drivers),
                False,
                None,
            ),
        )
        newton_steps, final_state = result[0], result[1]
        is_nan, iters_below = result[3], result[4]
        all_states, nan_mask, mf_val = None, None, None

    if iters_below is not None:
        all_newtons = newton_steps - iters_below + 1
    else:
        all_newtons = None
    return (
        all_states,
        final_state,
        newton_steps,
        is_nan,
        nan_mask,
        all_newtons,
        mf_val,
    )
