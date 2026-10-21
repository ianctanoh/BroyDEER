"""
picard_ddim.py
DDIM-specific Picard (Jacobi) iteration with optimized eps evaluation.

Same decomposition as broyden_ddim.py and taa_ddim_alg: vmaps only eps_model
over T positions, then applies DDIM coefficient math outside the vmap.
This avoids vmapping the full dynamics closure which OOMs on large models.

The update is pure Jacobi/Picard: new_states[s] = f(prev_states[s-1], driver[s]).
No Broyden blocks, no Anderson acceleration — just parallel fixed-point iteration.

API matches deer_alg / broyden_ddim_alg: returns the standard 7-tuple.
"""

from typing import Optional
import jax
import jax.numpy as jnp
from jax import vmap
from jax.lax import scan

from .deer import merit_fxn
from .ddim_utils import merit_from_residual


def picard_ddim_alg(
    dynamics,
    initial_state: jnp.ndarray,
    states_guess: jnp.ndarray,
    drivers: jnp.ndarray,
    num_iters: int,
    full_trace: bool = False,
    Ts: Optional[jnp.ndarray] = None,
    reset: bool = True,
    tol: float = 5e-4,
):
    """
    DDIM-specific Picard iteration with decomposed eps evaluation.

    Each iteration computes:
        new_states[s] = a[s] * X[s] + b[s] * eps(X[s], t_s) + n[s] * z[s]
    where X = (initial_state, states[0], ..., states[T-2]).

    Only eps_model is vmapped over T positions; coefficient math is element-wise.

    Args:
        dynamics      : DDIMDynamics instance
        initial_state : x_T (D,)
        states_guess  : initial trajectory guess (T, D)
        drivers       : inputs (T, D+1)
        num_iters     : maximum iterations
        full_trace    : if True, return all intermediate states
        Ts            : optional time indices for multi-checkpoint convergence
        reset         : if True, replace NaN states with states_guess
        tol           : convergence tolerance (on merit function)

    Returns:
        7-tuple matching deer_alg:
        (all_states, final_state, newton_steps, is_nan, nan_mask,
         all_newtons, mf_val)
    """
    T, D_dim = states_guess.shape

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

    @jax.jit
    def _step(carry, args):
        states, is_nan = carry

        # Build full trajectory: X[0]=initial_state, X[1..T]=states[0..T-1]
        X = jnp.concatenate([initial_state[None], states], axis=0)  # (T+1, D)

        # ONE eps vmap over all T positions
        eps_arr = vmap(eps_model_local)(X[:T], t_per_s)              # (T, D)

        # f(X[s], t_s) for each position
        f_vals = (a_per_s[:, None] * X[:T]
                  + b_per_s[:, None] * eps_arr
                  + n_per_s[:, None] * z_per_s)                     # (T, D)

        # Picard: compute increments b[s] = f(X[s]) - X[s], then prefix sum
        # b[0] = f(x_0, u_0) (no subtraction — initial state is fixed)
        # b[s] = f(X[s], u_s) - X[s] for s >= 1
        increments = f_vals - X[:T]                                  # (T, D)
        increments = increments.at[0].set(f_vals[0])                 # b[0] = f(x_0, u_0)
        new_states = jax.lax.associative_scan(
            lambda a, b: a + b, increments
        )                                                            # (T, D)

        is_nan = jnp.logical_or(jnp.isnan(new_states).any(), is_nan)
        nan_mask = jnp.isnan(new_states)
        if reset:
            new_states = jnp.where(nan_mask, states_guess, new_states)

        # Convergence check: merit reuses the eps evaluation already made in this step (no extra eps vmap).
        # Evaluated on pre-update states (same convention as broyden_ddim).
        mf_val = merit_from_residual(states - f_vals)

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
        new_errs = merit_fxn(f_local, initial_state, new_states, drivers, Ts=Ts)
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
