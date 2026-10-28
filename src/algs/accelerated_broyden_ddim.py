"""
accelerated_broyden_ddim.py
DDIM-specific Accelerated Broyden-DEER (fresh diagonal + rank-m secant correction).

Same memory-efficient DDIM decomposition as broyden_ddim.py: vmaps only eps_model,
carries r_old forward, reuses merit from _step.

Difference from broyden_ddim: instead of incrementally updating U,V via rank-1
Broyden corrections, this recomputes U,V each iteration from a sliding window of
the last m Newton steps and their secant targets. The diagonal can be identity or
rescaled_identity (no jacfwd).
"""

from typing import Optional
import jax
import jax.numpy as jnp
from jax import vmap
from jax.lax import scan

from .deer import merit_fxn
from .ddim_utils import merit_from_residual
from .broyden_deer import (
    broyden_solve,
    diagonal_solve,
    sequential_broyden_solve,
    sequential_diagonal_solve,
)


def accelerated_broyden_ddim_alg(
    dynamics,
    initial_state: jnp.ndarray,
    states_guess: jnp.ndarray,
    drivers: jnp.ndarray,
    num_iters: int,
    m: int = 3,
    init: str = "rescaled_identity",
    gamma: float = 1.0,
    sequential_solve: bool = False,
    compress: str = "rand",
    k: float = 0.0,
    eps_gram: float = 1e-10,
    full_trace: bool = False,
    Ts: Optional[jnp.ndarray] = None,
    reset: bool = True,
    tol: float = 1e-4,
):
    """
    DDIM-specific Accelerated Broyden-DEER.

    Each iteration recomputes U,V from a sliding window of m previous Newton
    steps via secant conditions, instead of incremental rank-1 updates.

    Args:
        dynamics       : DDIMDynamics instance
        initial_state  : x_T (D,)
        states_guess   : (T, D)
        drivers        : (T, D+1)
        num_iters      : max iterations
        m              : secant window size / rank (default 3)
        init           : "identity" or "rescaled_identity"
        gamma          : scaling for identity init
        sequential_solve : if True, use sequential forward substitution
        compress       : "rand" (GPU-friendly), "rand_rr" (rand with rank-revealing
                         Gram-Schmidt) or "qr_svd" for parallel solve
        k              : damping (default 0.0)
        eps_gram       : regularization for Gram matrix (default 1e-10)
        full_trace     : if True, return all states via scan
        Ts             : optional multi-checkpoint indices
        reset          : if True, replace NaN with states_guess
        tol            : convergence tolerance

    Returns:
        7-tuple: (all_states, final_state, newton_steps, is_nan, nan_mask,
                  all_newtons, mf_val)
    """
    if init not in ("identity", "rescaled_identity"):
        raise ValueError(
            f"init must be 'identity' or 'rescaled_identity', got {init!r}"
        )

    T, D_dim = states_guess.shape

    eps_model_local  = dynamics.eps_model
    coef_x_local     = dynamics.coef_x
    coef_eps_local   = dynamics.coef_eps
    coef_noise_local = dynamics.coef_noise

    def f_local(state, driver):
        z = driver[:D_dim]
        t = driver[D_dim].astype(jnp.int32)
        eps = eps_model_local(state, t)
        return (coef_x_local[t] * state
                + coef_eps_local[t] * eps
                + coef_noise_local[t] * z)

    s_idx   = jnp.arange(T)
    t_per_s = T - s_idx
    a_per_s = coef_x_local[t_per_s]
    b_per_s = coef_eps_local[t_per_s]
    n_per_s = coef_noise_local[t_per_s]
    z_per_s = drivers[:, :D_dim]

    # --- init ---
    p_window_init      = jnp.zeros((T, m, D_dim))
    target_window_init = jnp.zeros((T, m, D_dim))
    gamma_init         = -jnp.ones(T) if init == "rescaled_identity" else -gamma * jnp.ones(T)

    def _ddim_residual(states):
        X = jnp.concatenate([initial_state[None], states], axis=0)
        eps_arr = vmap(eps_model_local)(X[:T], t_per_s)
        f_vals = (a_per_s[:, None] * X[:T]
                  + b_per_s[:, None] * eps_arr
                  + n_per_s[:, None] * z_per_s)
        return states - f_vals

    r_init = _ddim_residual(states_guess)

    # --- U,V computation from secant window ---
    if m == 1:
        def _compute_UV_block(d_t, steps_t, targets_t):
            v_t = steps_t[0]
            r_t = targets_t[0] - d_t * v_t
            denom = jnp.dot(v_t, v_t) + eps_gram
            u_t = (1 - k) * r_t / denom
            return u_t[:, None], v_t[:, None]
    elif m == 2:
        def _compute_UV_block(d_t, steps_t, targets_t):
            V_t = steps_t.T
            R_t = targets_t.T - d_t[:, None] * V_t
            C_t = V_t.T @ V_t + eps_gram * jnp.eye(2)
            det = C_t[0, 0] * C_t[1, 1] - C_t[0, 1] * C_t[1, 0]
            inv_det = 1.0 / jnp.maximum(det, 1e-30)
            U_t = (1 - k) * inv_det * jnp.stack([
                R_t[:, 0] * C_t[1, 1] - R_t[:, 1] * C_t[0, 1],
                R_t[:, 1] * C_t[0, 0] - R_t[:, 0] * C_t[1, 0],
            ], axis=1)
            return U_t, V_t
    else:
        def _compute_UV_block(d_t, steps_t, targets_t):
            V_t = steps_t.T
            R_t = targets_t.T - d_t[:, None] * V_t
            C_t = V_t.T @ V_t + eps_gram * jnp.eye(m)
            U_t = (1 - k) * jnp.linalg.solve(C_t, R_t.T).T
            return U_t, V_t

    @jax.jit
    def _step(carry, args):
        (states, p_window, target_window, gamma_arr,
         r_old, is_nan, iteration) = carry

        d_init = gamma_arr[:, None] * jnp.ones((T, D_dim))

        # compute U,V from secant history
        U_inner, V_inner = vmap(_compute_UV_block)(
            d_init[:-1], p_window[:-1], target_window[:-1]
        )
        U_blocks = jnp.concatenate([U_inner, jnp.zeros((1, D_dim, m))], axis=0)
        V_blocks = jnp.concatenate([V_inner, jnp.zeros((1, D_dim, m))], axis=0)

        # zero out U on first iteration (no secant pairs yet)
        U_blocks = jnp.where(iteration == 0, jnp.zeros_like(U_blocks), U_blocks)

        # solve
        if m > 0:
            delta_s = (sequential_broyden_solve(d_init, U_blocks, V_blocks, r_old)
                       if sequential_solve else
                       broyden_solve(d_init, U_blocks, V_blocks, r_old,
                                     compress=compress, iteration=iteration))
        else:
            delta_s = (sequential_diagonal_solve(d_init, r_old)
                       if sequential_solve else
                       diagonal_solve(d_init, r_old))

        # update trajectory
        new_states = states + delta_s
        is_nan = jnp.logical_or(jnp.isnan(new_states).any(), is_nan)
        nan_mask = jnp.isnan(new_states)
        if reset:
            new_states = jnp.where(nan_mask, states_guess, new_states)

        # new residuals -- ONE eps vmap
        r_new = _ddim_residual(new_states)

        # update secant windows
        y = r_new - r_old
        new_target = jnp.concatenate(
            [y[1:] - delta_s[1:], jnp.zeros((1, D_dim))], axis=0
        )
        p_window_new = jnp.concatenate(
            [delta_s[:, None, :], p_window[:, :-1, :]], axis=1
        )
        target_window_new = jnp.concatenate(
            [new_target[:, None, :], target_window[:, :-1, :]], axis=1
        )

        # gamma update (rescaled_identity)
        p_last = p_window_new[:, 0, :]
        tgt_last = target_window_new[:, 0, :]
        gamma_num = jnp.sum(p_last * tgt_last, axis=-1)
        gamma_den = jnp.sum(p_last * p_last, axis=-1) + 1e-10
        gamma_new = jnp.where(iteration == 0, gamma_arr, gamma_num / gamma_den)

        # convergence on pre-update states: r_old is their residual
        # (computed last step), so no extra eps vmap
        mf_val = merit_from_residual(r_old)

        return (new_states, p_window_new, target_window_new, gamma_new,
                r_new, is_nan, iteration + 1), \
               (new_states, nan_mask, mf_val)

    # --- convergence loop ---
    def cond_func(iter_inp):
        iter_idx, _, _, _, _, _, err, *_ = iter_inp
        return jnp.logical_and(iter_idx < num_iters, err > tol)

    def body_func_single(iter_inp):
        (iter_idx, states, pw, tw, gamma_arr,
         r_old, _, is_nan_, _) = iter_inp
        step_output, step_aux = _step(
            (states, pw, tw, gamma_arr, r_old, is_nan_, iter_idx), None
        )
        (new_states, pw_new, tw_new, gamma_new,
         r_new, is_nan, _) = step_output
        _, _, mf_val = step_aux
        return (iter_idx + 1, new_states, pw_new, tw_new, gamma_new,
                r_new, mf_val, is_nan, None)

    def body_func_multiple(iter_inp):
        (iter_idx, states, pw, tw, gamma_arr,
         r_old, _, is_nan_, iters_below) = iter_inp
        step_output, step_aux = _step(
            (states, pw, tw, gamma_arr, r_old, is_nan_, iter_idx), None
        )
        (new_states, pw_new, tw_new, gamma_new,
         r_new, is_nan, _) = step_output
        new_errs = merit_fxn(f_local, initial_state, states, drivers, Ts=Ts)
        iters_below = iters_below + (new_errs < tol).astype(int)
        return (iter_idx + 1, new_states, pw_new, tw_new, gamma_new,
                r_new, new_errs[-1], is_nan, iters_below)

    init_carry = (states_guess, p_window_init, target_window_init, gamma_init,
                  r_init, False, 0)

    if full_trace:
        last_output, all_outputs = scan(
            _step, init_carry, None, length=num_iters
        )
        final_state = last_output[0]
        is_nan = last_output[5]
        all_states, nan_mask, mf_val = all_outputs
        newton_steps, iters_below = None, None
        all_states = jnp.concatenate([states_guess[None, ...], all_states])
    elif Ts is not None:
        result = jax.lax.while_loop(
            cond_func,
            body_func_multiple,
            (0, states_guess, p_window_init, target_window_init, gamma_init,
             r_init,
             merit_fxn(f_local, initial_state, states_guess, drivers),
             False, jnp.zeros_like(Ts)),
        )
        newton_steps, final_state = result[0], result[1]
        is_nan, iters_below = result[7], result[8]
        all_states, nan_mask, mf_val = None, None, None
    else:
        result = jax.lax.while_loop(
            cond_func,
            body_func_single,
            (0, states_guess, p_window_init, target_window_init, gamma_init,
             r_init,
             merit_fxn(f_local, initial_state, states_guess, drivers),
             False, None),
        )
        newton_steps, final_state = result[0], result[1]
        is_nan, iters_below = result[7], result[8]
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
