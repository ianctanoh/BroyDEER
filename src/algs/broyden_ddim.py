"""
broyden_ddim.py
DDIM-specific Broyden-DEER that avoids vmapping the full dynamics function.

The standard `broyden_deer_alg` uses `get_residual(f, ...)` which vmaps the
full dynamics function `f` (containing the neural network) over all T timesteps.
On large models (e.g., DiT-XL/2), XLA tries to compile this into one fused
kernel requiring prohibitive GPU memory.

This module decomposes the DDIM step: vmap only `eps_model` over T positions,
then do coefficient math (coef_x, coef_eps, coef_noise) outside the vmap.
This is the same decomposition used by `taa_ddim_alg`.

Memory optimization: `r_old` is carried forward from the previous iteration
(instead of recomputed), and `merit_fxn` is computed once per step (not
duplicated in the body function).  This keeps the while_loop body at 2 eps
vmaps (1 for r_new + 1 for merit), well within the GPU budget that TAA
demonstrates is safe (3 vmaps).

Only "identity" and "rescaled_identity" init modes are supported, since
"jac", "low_rank", and "fresh_jac" all require Jacobian computation via
`jacfwd(f)` which vmaps the full dynamics and OOMs.
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
    update_all_blocks,
    compute_gamma_rescale,
    update_gamma,
)


def broyden_ddim_alg(
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
    full_trace: bool = False,
    Ts: Optional[jnp.ndarray] = None,
    reset: bool = True,
    tol: float = 1e-4,
):
    """
    DDIM-specific Broyden-DEER: parallel Newton solver with limited-memory
    Broyden Jacobian blocks, using decomposed DDIM residual evaluation.

    Instead of vmapping the full dynamics function (which OOMs on large models),
    this vmaps only the eps_model and performs DDIM coefficient math outside
    the vmap.

    Args:
        dynamics       : DDIMDynamics instance providing eps_model and the
                         per-timestep coefficient arrays
        initial_state  : x_T (D,)
        states_guess   : initial trajectory guess (T, D)
        drivers        : inputs (T, D+1)
        num_iters      : maximum number of outer iterations
        m              : rank / window size for Broyden blocks (default 3).
                         m=0: pure diagonal solve, no rank-1 updates
        init           : initialization mode for diagonal d_t:
                           "identity"          -- d_t = -gamma * I (fixed)
                           "rescaled_identity" -- d_t = -I; gamma rescaled
        gamma          : scaling for identity init (default 1.0)
        sequential_solve : if True, use sequential forward substitution
        compress       : compression for parallel solve: "rand" (GPU-friendly,
                         default), "rand_rr" (rand with rank-revealing
                         Gram-Schmidt) or "qr_svd" (LAPACK, may OOM on large D)
        k              : damping parameter for rank-1 corrections (default 0.0)
        full_trace     : if True, return all intermediate states via scan
        Ts             : optional time indices for multi-checkpoint convergence
        reset          : if True, replace NaN states with states_guess
        tol            : convergence tolerance (on merit function)

    Returns:
        7-tuple matching deer_alg:
        (all_states, final_state, newton_steps, is_nan, nan_mask,
         all_newtons, mf_val)
    """
    if init not in ("identity", "rescaled_identity"):
        raise ValueError(
            f"init must be 'identity' or 'rescaled_identity', got {init!r}. "
            f"Other modes require Jacobian computation that OOMs on large models."
        )

    T, D_dim = states_guess.shape
    m = min(m, D_dim)
    use_lrmat = (m > 0)

    # Extract dynamics fields into local variables.  Closures over these
    # hash by Python identity (not by recursion into arrays), which avoids
    # lax.scan / while_loop caching issues with equinox modules.
    eps_model_local  = dynamics.eps_model
    coef_x_local     = dynamics.coef_x
    coef_eps_local   = dynamics.coef_eps
    coef_noise_local = dynamics.coef_noise

    # Canonical single-step DDIM closure (used by merit_fxn for convergence).
    # merit_fxn -> get_residual vmaps this closure, which is lightweight:
    # coefficient math + one eps_model call per position.
    def f_local(state, driver):
        z = driver[:D_dim]
        t = driver[D_dim].astype(jnp.int32)
        eps = eps_model_local(state, t)
        return (coef_x_local[t] * state
                + coef_eps_local[t] * eps
                + coef_noise_local[t] * z)

    # Precompute per-step DDIM coefficients in BroDeer index order.
    s_idx   = jnp.arange(T)
    t_per_s = T - s_idx                            # (T,) DDIM timesteps
    a_per_s = coef_x_local[t_per_s]                # (T,)
    b_per_s = coef_eps_local[t_per_s]              # (T,)
    n_per_s = coef_noise_local[t_per_s]            # (T,)
    z_per_s = drivers[:, :D_dim]                   # (T, D)

    # --- initialize d_init, U_init, V_init ---
    if init == "identity":
        d_init = -gamma * jnp.ones((T, D_dim))
    else:  # "rescaled_identity"
        d_init = -jnp.ones((T, D_dim))

    U_init = jnp.zeros((T, D_dim, max(m, 1)))
    V_init = jnp.zeros((T, D_dim, max(m, 1)))
    gamma_init = jnp.ones(T)

    # --- helper: compute inlined DDIM residual ---
    def _ddim_residual(states):
        """
        Compute r = states - f(prev) using decomposed DDIM evaluation.

        Vmaps only eps_model over T positions, then applies DDIM coefficient
        math element-wise.  Same result as get_residual(f_local, ...) but
        avoids vmapping the full dynamics closure.
        """
        X = jnp.concatenate([initial_state[None], states], axis=0)  # (T+1, D)
        eps_arr = vmap(eps_model_local)(X[:T], t_per_s)              # (T, D)
        f_vals = (a_per_s[:, None] * X[:T]
                  + b_per_s[:, None] * eps_arr
                  + n_per_s[:, None] * z_per_s)                     # (T, D)
        return states - f_vals                                       # (T, D)

    # Precompute initial residual BEFORE the loop (not inside the compiled
    # while_loop body).  This is carried forward: each iteration's r_new
    # becomes the next iteration's r_old, saving one eps vmap per step.
    r_init = _ddim_residual(states_guess)

    # --- step function ---
    # Carry includes r_old so we avoid recomputing _ddim_residual(states).
    # Per step: 1 eps vmap (_ddim_residual for r_new) + 1 eps vmap (merit_fxn)
    # = 2 total, well within GPU budget.
    @jax.jit
    def _step(carry, args):
        states, U, V, gamma_vec, r_old, is_nan, iteration = carry

        # effective diagonal
        d_eff = gamma_vec[:, None] * d_init

        # r_old is already in carry -- NO eps vmap here

        # solve block-bidiagonal system
        if use_lrmat:
            delta_s = (sequential_broyden_solve(d_eff, U, V, r_old)
                       if sequential_solve else
                       broyden_solve(d_eff, U, V, r_old,
                                     compress=compress, iteration=iteration))
        else:
            delta_s = (sequential_diagonal_solve(d_eff, r_old)
                       if sequential_solve else
                       diagonal_solve(d_eff, r_old))

        # update trajectory
        new_states = states + delta_s
        is_nan = jnp.logical_or(jnp.isnan(new_states).any(), is_nan)
        nan_mask = jnp.isnan(new_states)
        if reset:
            new_states = jnp.where(nan_mask, states_guess, new_states)

        # new residuals for Broyden update -- ONE eps vmap
        r_new = _ddim_residual(new_states)

        # update Broyden blocks
        if init == "rescaled_identity":
            gamma_new = compute_gamma_rescale(delta_s, r_old, r_new, d=d_init)
            if use_lrmat:
                d_eff_update = gamma_new[:, None] * d_init
                U_new, V_new = update_all_blocks(
                    d_eff_update, U, V, delta_s, r_old, r_new, k
                )
            else:
                U_new, V_new = U, V
        else:  # "identity"
            if use_lrmat:
                U_new, V_new = update_all_blocks(
                    d_eff, U, V, delta_s, r_old, r_new, k
                )
            else:
                U_new, V_new = U, V
            gamma_new = update_gamma(
                gamma_vec, delta_s, r_old, r_new, m, iteration, d_init
            )

        # convergence check on pre-update states: r_old is their residual
        # (computed last step), so no extra eps vmap
        mf_val = merit_from_residual(r_old)

        return (new_states, U_new, V_new, gamma_new, r_new,
                is_nan, iteration + 1), \
               (new_states, nan_mask, mf_val)

    # --- convergence loop ---
    def cond_func(iter_inp):
        iter_idx, _, _, _, _, _, err, *_ = iter_inp
        return jnp.logical_and(iter_idx < num_iters, err > tol)

    def body_func_single(iter_inp):
        iter_idx, states, U, V, gamma_vec, r_old, _, is_nan_, _ = iter_inp
        step_output, step_aux = _step(
            (states, U, V, gamma_vec, r_old, is_nan_, iter_idx), None
        )
        new_states, U_new, V_new, gamma_new, r_new, is_nan, _ = step_output
        _, _, mf_val = step_aux
        return (iter_idx + 1, new_states, U_new, V_new, gamma_new,
                r_new, mf_val, is_nan, None)

    def body_func_multiple(iter_inp):
        iter_idx, states, U, V, gamma_vec, r_old, _, is_nan_, iters_below = iter_inp
        step_output, step_aux = _step(
            (states, U, V, gamma_vec, r_old, is_nan_, iter_idx), None
        )
        new_states, U_new, V_new, gamma_new, r_new, is_nan, _ = step_output
        new_errs = merit_fxn(f_local, initial_state, states, drivers, Ts=Ts)
        iters_below = iters_below + (new_errs < tol).astype(int)
        return (iter_idx + 1, new_states, U_new, V_new, gamma_new,
                r_new, new_errs[-1], is_nan, iters_below)

    init_carry = (states_guess, U_init, V_init, gamma_init, r_init, False, 0)

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
            (0, states_guess, U_init, V_init, gamma_init,
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
            (0, states_guess, U_init, V_init, gamma_init,
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
