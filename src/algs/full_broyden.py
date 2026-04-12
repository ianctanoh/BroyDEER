"""
full_broyden.py
Global Broyden solver for discrete-time dynamical systems.

Treats the full TD x TD Jacobian as a single structured-plus-low-rank matrix:

  B = D_init(gamma) + U V^T,   U, V in R^{TD x k}

where k is the rank budget (sliding window of secant pairs).

D(gamma) depends on the init scheme:
  "identity"          -- D(gamma) = I_{TD}, gamma = 1 fixed
  "rescaled_identity" -- D(gamma) = gamma * I_{TD}, gamma updated each iteration
  "jac"               -- D(gamma) = I + gamma * L_sub  (block lower-bidiagonal)
  "fresh_jac"         -- D recomputed via jacfwd at current X every iteration

Newton step solved via Woodbury identity:
  (D + UV^T)^{-1} r = D^{-1}r - D^{-1}U (I_k + V^T D^{-1}U)^{-1} V^T D^{-1}r
"""

from typing import Callable, Optional
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, lax, jacfwd
from jax.lax import scan

from .deer import get_residual, merit_fxn, quasi_diag_estimator


# --- block lower-bidiagonal D operations ("jac" / "fresh_jac" init) ---

def bidiag_solve(d_sub, gamma, r):
    """
    Forward substitution for  D @ w = r  where D is block lower-bidiagonal:
        diagonal blocks   = I_D
        subdiagonal block t = gamma * diag(d_sub[t])

    Implemented as a first-order linear associative scan.
    """
    T, D_dim = r.shape

    def combine(elem_a, elem_b):
        a_a, b_a = elem_a
        a_b, b_b = elem_b
        return a_b * a_a, a_b * b_a + b_b

    a = jnp.concatenate([jnp.zeros((1, D_dim)),
                         -gamma * d_sub[:-1]], axis=0)
    _, w = lax.associative_scan(combine, (a, r))
    return w


def bidiag_matvec(d_sub, gamma, x):
    """
    Matrix-vector product  y = D @ x  for block lower-bidiagonal D.
    """
    T, D_dim = x.shape
    sub = jnp.concatenate([jnp.zeros((1, D_dim)),
                           gamma * d_sub[:-1] * x[:-1]], axis=0)
    return x + sub


# --- rank-1 Broyden update ---

def rank1_update(D_new_delta, U, V, delta_s, y, damping=1.0):
    """
    Sliding-window rank-1 Broyden update (global, flattened).

    D_new_delta = D_new @ delta_s  (diagonal/bidiagonal part applied to delta_s).
    Appends the new secant pair and drops the oldest.
    """
    b_p   = D_new_delta + U @ (V.T @ delta_s)
    denom = jnp.dot(delta_s, delta_s) + 1e-10
    new_u = damping * (y - b_p) / denom
    new_v = delta_s
    U_new = jnp.concatenate([U[:, 1:], new_u[:, None]], axis=1)
    V_new = jnp.concatenate([V[:, 1:], new_v[:, None]], axis=1)
    return U_new, V_new


# --- gamma update ---

def update_gamma_from_secant(D0_delta, y):
    """
    Find scalar gamma such that  gamma * D0 * delta_s = y  in least squares.
    """
    return jnp.dot(D0_delta, y) / (jnp.dot(D0_delta, D0_delta) + 1e-10)


# --- main algorithm ---

def full_broyden_deer_alg(
    f: Callable,
    initial_state: jnp.ndarray,
    states_guess: jnp.ndarray,
    drivers: jnp.ndarray,
    num_iters: int,
    k: int = 10,
    init: str = "identity",
    gamma: float = 1.0,
    qmem_efficient: bool = False,
    damping: float = 1.0,
    line_search: bool = True,
    max_ls: int = 10,
    full_trace: bool = False,
    Ts=None,
    reset: bool = True,
    tol: float = 5e-4,
    key=None,
):
    """
    Full-Broyden DEER: global (D0 + UV^T) Newton solver with Woodbury solve.

    Each iteration forms a TD x TD structured matrix D + UV^T where D is either
    identity-like or block lower-bidiagonal. The Newton step is solved via the
    Woodbury identity in O(TD * k) time.

    Args:
        f              : dynamics function (state, driver) -> new_state
        initial_state  : x_0 (D,)
        states_guess   : initial trajectory guess (T, D)
        drivers        : inputs (T, d_in)
        num_iters      : maximum number of outer iterations
        k              : rank / sliding-window size (default 10)
        init           : initialization mode for D:
                           "identity"          -- D = gamma * I, fixed
                           "rescaled_identity" -- D = I, gamma updated each iter
                           "jac"               -- D block lower-bidiagonal with diag(-df/dx)
                           "fresh_jac"         -- D recomputed at current X every iteration
        gamma          : scaling for identity init (default 1.0)
        qmem_efficient : if True, use Hutchinson estimator for Jacobian diagonal
        damping        : scaling for rank-1 corrections (default 1.0)
        line_search    : if True, backtracking line search (default True)
        max_ls         : maximum number of backtracking halvings (default 10)
        full_trace     : if True, return all intermediate states via scan
        Ts             : optional time indices for multi-checkpoint convergence
        reset          : if True, replace NaN states with states_guess
        tol            : convergence tolerance (on merit function)
        key            : PRNG key for Hutchinson estimator

    Returns:
        7-tuple matching deer_alg:
        (all_states, final_state, newton_steps, is_nan, nan_mask, all_newtons, mf_val)
    """
    if init not in ("identity", "rescaled_identity", "jac", "fresh_jac"):
        raise ValueError(
            f"init must be 'identity', 'rescaled_identity', 'jac', or 'fresh_jac', "
            f"got {init!r}"
        )

    T, D = states_guess.shape
    TD = T * D

    if key is None:
        key = jr.PRNGKey(0)

    # --- Hutchinson setup ---
    if qmem_efficient:
        def deer_jvp(state, inp, z):
            return jax.jvp(lambda x: f(x, inp), (state,), (z,))[1]
        keys = jr.split(key, T - 1)

    def compute_jac_diag(states_at, drivers_at):
        """Compute diag(-df/dx) using Hutchinson or exact jacfwd."""
        if qmem_efficient:
            return -vmap(quasi_diag_estimator, in_axes=(0, 0, None, 0))(
                states_at, drivers_at, deer_jvp, keys)
        else:
            jacs = vmap(jacfwd(f))(states_at, drivers_at)
            return -vmap(jnp.diag)(jacs)

    # --- initialize D0 ---
    if init == "jac":
        d_sub_init = compute_jac_diag(states_guess[:-1], drivers[1:])
        d_sub_init = jnp.concatenate([d_sub_init, jnp.zeros((1, D))], axis=0)
    else:
        d_sub_init = jnp.zeros((T, D))

    # --- helpers ---
    ig_flat = states_guess.reshape(-1)

    def res_flat(states):
        """Compute residuals as flat TD vector."""
        return get_residual(f, initial_state, states, drivers).reshape(-1)

    def apply_Dinv(gamma, v_flat, d_sub):
        """Apply D^{-1} to a flat vector."""
        if init in ("jac", "fresh_jac"):
            return bidiag_solve(d_sub, gamma,
                                v_flat.reshape(T, D)).reshape(-1)
        else:
            return v_flat / (gamma + 1e-30)

    def apply_D0(delta_2d, d_sub):
        """Apply D0 (at gamma=1) to a (T, D) array, return flat."""
        if init in ("jac", "fresh_jac"):
            return bidiag_matvec(d_sub, 1.0, delta_2d).reshape(-1)
        else:
            return delta_2d.reshape(-1)

    # --- initial carry ---
    U_init     = jnp.zeros((TD, k))
    V_init     = jnp.zeros((TD, k))
    gamma_init = jnp.ones(()) if init != "identity" else gamma * jnp.ones(())

    # --- step function ---
    @jax.jit
    def _step(carry, args):
        states, U, V, gamma, is_nan, iteration = carry

        # compute d_sub for this iteration
        if init == "fresh_jac":
            d_sub = compute_jac_diag(states[:-1], drivers[1:])
            d_sub = jnp.concatenate([d_sub, jnp.zeros((1, D))], axis=0)
            _gamma = jnp.ones(())
        else:
            d_sub = d_sub_init
            _gamma = gamma

        r_old_flat = res_flat(states)

        # Woodbury solve: (D + UV^T) delta_s = -r
        def _Dinv(v):
            return apply_Dinv(_gamma, v, d_sub)

        w     = -_Dinv(r_old_flat)
        DinvU = vmap(lambda col: _Dinv(col), in_axes=1, out_axes=1)(U)
        M     = jnp.eye(k) + V.T @ DinvU
        b     = V.T @ w
        z     = jnp.linalg.solve(M, b)
        delta_s = w - DinvU @ z

        # --- backtracking line search ---
        states_flat = states.reshape(-1)
        norm_old = jnp.dot(r_old_flat, r_old_flat)

        if line_search:
            def ls_cond(ls_val):
                alpha, _ = ls_val
                cand = states_flat + alpha * delta_s
                cand_safe = jnp.where(jnp.isnan(cand), ig_flat, cand)
                r_try = res_flat(cand_safe.reshape(T, D))
                return jnp.logical_and(
                    jnp.dot(r_try, r_try) >= norm_old,
                    alpha > (0.5 ** max_ls)
                )

            def ls_body(ls_val):
                alpha, _ = ls_val
                alpha_new = alpha * 0.5
                cand = states_flat + alpha_new * delta_s
                cand_safe = jnp.where(jnp.isnan(cand), ig_flat, cand)
                r_try = res_flat(cand_safe.reshape(T, D))
                return alpha_new, r_try

            cand0 = states_flat + delta_s
            cand0_safe = jnp.where(jnp.isnan(cand0), ig_flat, cand0)
            r_try0 = res_flat(cand0_safe.reshape(T, D))
            alpha_final, r_new_flat = lax.while_loop(
                ls_cond, ls_body, (1.0, r_try0)
            )
            delta_s = alpha_final * delta_s

        # apply step and handle NaN
        raw_flat = states_flat + delta_s
        new_flat = jnp.where(jnp.isnan(raw_flat), ig_flat, raw_flat)
        new_states = new_flat.reshape(T, D)

        if not line_search:
            r_new_flat = res_flat(new_states)

        is_nan = jnp.logical_or(jnp.isnan(raw_flat).any(), is_nan)
        nan_mask = jnp.isnan(raw_flat).reshape(T, D)
        if reset:
            new_states = jnp.where(nan_mask, states_guess, new_states)

        # --- Broyden update and gamma update ---
        y = r_new_flat - r_old_flat

        if init == "fresh_jac":
            gamma_new   = gamma
            D_new_delta = bidiag_matvec(d_sub, 1.0,
                                        delta_s.reshape(T, D)).reshape(-1)
        elif init in ("rescaled_identity", "jac"):
            D0_delta = apply_D0(delta_s.reshape(T, D), d_sub)
            if init == "jac":
                L_sub_delta = D0_delta - delta_s
                gamma_new   = (jnp.dot(L_sub_delta, y - delta_s) /
                               (jnp.dot(L_sub_delta, L_sub_delta) + 1e-10))
                D_new_delta = bidiag_matvec(d_sub, gamma_new,
                                            delta_s.reshape(T, D)).reshape(-1)
            else:
                gamma_new   = update_gamma_from_secant(D0_delta, y)
                D_new_delta = gamma_new * delta_s
        else:  # identity
            gamma_new   = gamma
            D_new_delta = gamma_new * delta_s

        U_new, V_new = rank1_update(D_new_delta, U, V, delta_s, y, damping)

        mf_val = merit_fxn(f, initial_state, states, drivers)
        return (new_states, U_new, V_new, gamma_new, is_nan, iteration + 1), \
               (new_states, nan_mask, mf_val)

    # --- convergence loop ---
    def cond_func(iter_inp):
        iter_idx, _, _, _, _, err, *_ = iter_inp
        return jnp.logical_and(iter_idx < num_iters, err > tol)

    def body_func_single(iter_inp):
        iter_idx, states, U, V, gamma, _, is_nan_, _ = iter_inp
        step_output, _ = _step((states, U, V, gamma, is_nan_, iter_idx), None)
        new_states, U_new, V_new, gamma_new, is_nan, _ = step_output
        new_err = merit_fxn(f, initial_state, states, drivers)
        return iter_idx + 1, new_states, U_new, V_new, gamma_new, new_err, is_nan, None

    def body_func_multiple(iter_inp):
        iter_idx, states, U, V, gamma, _, is_nan_, iters_below = iter_inp
        step_output, _ = _step((states, U, V, gamma, is_nan_, iter_idx), None)
        new_states, U_new, V_new, gamma_new, is_nan, _ = step_output
        new_errs = merit_fxn(f, initial_state, states, drivers, Ts=Ts)
        iters_below = iters_below + (new_errs < tol).astype(int)
        return iter_idx + 1, new_states, U_new, V_new, gamma_new, new_errs[-1], is_nan, iters_below

    init_carry = (states_guess, U_init, V_init, gamma_init, False, 0)

    if full_trace:
        last_output, all_outputs = scan(
            _step, init_carry, None, length=num_iters
        )
        final_state = last_output[0]
        is_nan = last_output[4]
        all_states, nan_mask, mf_val = all_outputs
        newton_steps, iters_below = None, None
        all_states = jnp.concatenate([states_guess[None, ...], all_states])
    elif Ts is not None:
        result = jax.lax.while_loop(
            cond_func,
            body_func_multiple,
            (0, states_guess, U_init, V_init, gamma_init,
             merit_fxn(f, initial_state, states_guess, drivers),
             False, jnp.zeros_like(Ts)),
        )
        newton_steps, final_state = result[0], result[1]
        is_nan, iters_below = result[6], result[7]
        all_states, nan_mask, mf_val = None, None, None
    else:
        result = jax.lax.while_loop(
            cond_func,
            body_func_single,
            (0, states_guess, U_init, V_init, gamma_init,
             merit_fxn(f, initial_state, states_guess, drivers),
             False, None),
        )
        newton_steps, final_state = result[0], result[1]
        is_nan, iters_below = result[6], result[7]
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
