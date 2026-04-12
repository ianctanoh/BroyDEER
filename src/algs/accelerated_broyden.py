"""
accelerated_broyden.py
DEER variant: fresh true diagonal + rank-k secant correction per block.

At each iteration i, the subdiagonal block b_t is:

    b_t^(i) = diag(d_t^(i)) + U_t^(i) V_t^(i)^T

where:
  - d_t^(i) = -diag(df/dx)|_{X^(i)[t]}   recomputed fresh each iteration
  - V_t^(i) = [p_t^(i-1), ..., p_t^(i-k)]  the last k Newton steps (sliding window)
  - U_t^(i) is chosen so that all k secant conditions hold simultaneously:
        b_t^(i) @ p_t^(i-j) = target_t^(i-j),  j = 1, ..., k

    Rearranging:  U_t C_t = R_t
    where  C_t = V_t^T V_t  (k x k Gram matrix of stored steps)
           R_t = [target_t^(i-j) - d_t^(i) * p_t^(i-j)]_{j=1}^k  (D x k)
    Solved as:  U_t = R_t @ inv(C_t + eps I)

The block-bidiagonal system is solved via lax.associative_scan with rank-k
QR+SVD compression, or optionally via sequential forward substitution.
"""

from typing import Callable, Optional
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, lax, jacfwd
from jax.lax import scan

from .deer import get_residual, merit_fxn, quasi_diag_estimator


# --- low-rank matrix-vector product ---

def lrmat_apply(d, U, V, x):
    """Apply (diag(d) + U V^T) to vector x.  d, x: (D,), U, V: (D, k)."""
    return d * x + U @ (V.T @ x)


# --- rank-k composition via QR + SVD ---

def lrmat_compose(d_a, U_a, V_a, d_b, U_b, V_b, k):
    """
    Rank-k approximation of (diag(d_b) + U_b V_b^T)(diag(d_a) + U_a V_a^T).

    Exact rank-2k factorization:
        Uc = [ D_b U_a  |  U_b ]
        Vc = [ V_a      |  D_a V_b + V_a M^T ]   where M = V_b^T U_a

    Compressed to rank-k via thin QR + truncated SVD on the (2k x 2k) core.
    O(D k + k^3) cost.
    """
    M  = V_b.T @ U_a
    Uc = jnp.concatenate([-d_b[:, None] * U_a, -U_b],       axis=1)
    Vc = jnp.concatenate([V_a, d_a[:, None] * V_b + V_a @ M.T], axis=1)

    Qu, Ru = jnp.linalg.qr(Uc)
    Qv, Rv = jnp.linalg.qr(Vc)
    K = Ru @ Rv.T

    Uk, sk, Vtk = jnp.linalg.svd(K, full_matrices=False)
    sh    = jnp.sqrt(jnp.abs(sk[:k]))
    d_new = -(d_b * d_a)
    U_new = (Qu @ Uk[:, :k]) * sh
    V_new = (Qv @ Vtk[:k, :].T) * sh
    return d_new, U_new, V_new


# --- combination for associative scan ---

def make_combine(k):
    """Return a combine function for lax.associative_scan with rank-k blocks."""
    def combine(elem_a, elem_b):
        alpha_a, d_a, U_a, V_a, dvec_a = elem_a
        alpha_b, d_b, U_b, V_b, dvec_b = elem_b
        alpha_new            = alpha_a * alpha_b
        d_new, U_new, V_new  = lrmat_compose(d_a, U_a, V_a, d_b, U_b, V_b, k)
        dvec_new             = -alpha_b * lrmat_apply(d_b, U_b, V_b, dvec_a) + dvec_b
        return alpha_new, d_new, U_new, V_new, dvec_new
    return combine


# --- block-bidiagonal solve (parallel) ---

def solve_rankk_bidiag(d_blocks, U_blocks, V_blocks, r, k):
    """Solve  B @ delta_s = -r  via lax.associative_scan."""
    T, D = d_blocks.shape
    alphas = jnp.ones(T).at[0].set(0.0)
    d_scan = jnp.concatenate([jnp.ones((1, D)),     d_blocks[:-1]], axis=0)
    U_scan = jnp.concatenate([jnp.zeros((1, D, k)), U_blocks[:-1]], axis=0)
    V_scan = jnp.concatenate([jnp.zeros((1, D, k)), V_blocks[:-1]], axis=0)
    dvec   = -r

    _, _, _, _, delta_s = lax.associative_scan(
        vmap(make_combine(k)), (alphas, d_scan, U_scan, V_scan, dvec), reverse=False
    )
    return delta_s


# --- block-bidiagonal solve (sequential) ---

def solve_rankk_bidiag_sequential(d_blocks, U_blocks, V_blocks, r):
    """Solve  B @ delta_s = -r  via lax.scan (sequential forward substitution)."""
    def step(delta_prev, inputs):
        r_t, d_t, U_t, V_t = inputs
        delta_t = -r_t - lrmat_apply(d_t, U_t, V_t, delta_prev)
        return delta_t, delta_t

    delta_0 = -r[0]
    _, deltas_rest = lax.scan(
        step, delta_0, (r[1:], d_blocks[:-1], U_blocks[:-1], V_blocks[:-1])
    )
    return jnp.concatenate([delta_0[None], deltas_rest], axis=0)


# --- orthonormal basis helpers ---

def _update_basis_ortho(V_old, Gamma_old, p_new, q_new, n_valid, k, delta=1e-12):
    """
    Incrementally update an orthonormal basis V and paired targets Gamma
    when a new secant pair (p_new, q_new) arrives.
    """
    col_mask = (jnp.arange(k) >= (k - n_valid)).astype(V_old.dtype)

    alpha = (V_old.T @ p_new) * col_mask
    p_tilde = p_new - V_old @ alpha
    norm_p = jnp.linalg.norm(p_tilde)

    safe_norm = jnp.where(norm_p < delta, 1.0, norm_p)
    v_new = p_tilde / safe_norm
    q_tilde = (q_new - Gamma_old @ alpha) / safe_norm

    V_slid = jnp.concatenate([V_old[:, 1:], v_new[:, None]], axis=1)
    Gamma_slid = jnp.concatenate([Gamma_old[:, 1:], q_tilde[:, None]], axis=1)

    insert_idx = k - n_valid - 1
    V_grown = V_old.at[:, insert_idx].set(v_new)
    Gamma_grown = Gamma_old.at[:, insert_idx].set(q_tilde)

    basis_full = n_valid >= k
    V_updated = jnp.where(basis_full, V_slid, V_grown)
    Gamma_updated = jnp.where(basis_full, Gamma_slid, Gamma_grown)

    skip = norm_p < delta
    V_new = jnp.where(skip, V_old, V_updated)
    Gamma_new = jnp.where(skip, Gamma_old, Gamma_updated)
    return V_new, Gamma_new


def _compute_U_ortho(d_t, V_t, Gamma_t, n_valid, k):
    """Closed-form U from orthonormal V:  U = Gamma - diag(d) V."""
    U = Gamma_t - d_t[:, None] * V_t
    col_mask = (jnp.arange(k) >= (k - n_valid)).astype(V_t.dtype)
    return U * col_mask[None, :]


# --- main algorithm ---

def accelerated_broyden_deer_alg(
    f: Callable,
    initial_state: jnp.ndarray,
    states_guess: jnp.ndarray,
    drivers: jnp.ndarray,
    num_iters: int,
    k: int = 1,
    init: str = "autodiff",
    gamma: float = 1.0,
    qmem_efficient: bool = False,
    sequential_solve: bool = False,
    damping: float = 1.0,
    fd_h: float = 1e-5,
    eps_gram: float = 1e-10,
    full_trace: bool = False,
    Ts=None,
    reset: bool = True,
    tol: float = 5e-4,
    key=None,
):
    """
    Accelerated Broyden DEER: fresh diagonal + rank-k secant correction.

    Each iteration recomputes the diagonal d_t and fits rank-k secant corrections
    U_t, V_t from the sliding window of k previous Newton steps.

    Args:
        f              : dynamics function (state, driver) -> new_state
        initial_state  : x_0 (D,)
        states_guess   : initial trajectory guess (T, D)
        drivers        : inputs (T, d_in)
        num_iters      : maximum number of outer iterations
        k              : rank of the secant correction (default 1)
        init           : diagonal scheme:
                           "autodiff"          -- d_t = -diag(df/dx) via jacfwd (exact)
                           "identity"          -- d_t = -gamma * I (fixed)
                           "manual"            -- d_t via per-dimension central differences
                           "rescaled_identity" -- d_t = gamma_t * ones, scalar per block
        gamma          : scaling for identity init (default 1.0)
        qmem_efficient : if True, use Hutchinson estimator for Jacobian diagonal
                         (only applies to init="autodiff")
        sequential_solve : if True, use sequential forward substitution
        damping        : scaling factor for the rank-k secant correction (default 1.0)
        fd_h           : step size for init="manual" (default 1e-5)
        eps_gram       : regularization for the Gram matrix V^T V (default 1e-10)
        full_trace     : if True, return all intermediate states via scan
        Ts             : optional time indices for multi-checkpoint convergence
        reset          : if True, replace NaN states with states_guess
        tol            : convergence tolerance (on merit function)
        key            : PRNG key for Hutchinson estimator

    Returns:
        7-tuple matching deer_alg:
        (all_states, final_state, newton_steps, is_nan, nan_mask, all_newtons, mf_val)
    """
    if init not in ("autodiff", "identity", "manual", "rescaled_identity"):
        raise ValueError(
            f"init must be 'autodiff', 'identity', 'manual', or 'rescaled_identity', "
            f"got {init!r}"
        )

    T, D = states_guess.shape

    if key is None:
        key = jr.PRNGKey(0)

    # --- Hutchinson setup ---
    if qmem_efficient:
        def deer_jvp(state, inp, z):
            return jax.jvp(lambda x: f(x, inp), (state,), (z,))[1]
        keys = jr.split(key, T - 1)

    def compute_jac_diag(states_at, drivers_at):
        """Compute diag(-df/dx) using the chosen scheme."""
        if init == "autodiff":
            if qmem_efficient:
                return -vmap(quasi_diag_estimator, in_axes=(0, 0, None, 0))(
                    states_at, drivers_at, deer_jvp, keys)
            else:
                jacs = vmap(jacfwd(f))(states_at, drivers_at)
                return -vmap(jnp.diag)(jacs)
        elif init == "manual":
            def compute_diag_fd(x_t, u_t):
                def cd_diag_i(i):
                    x_p = x_t.at[i].add(fd_h)
                    x_m = x_t.at[i].add(-fd_h)
                    return -(f(x_p, u_t)[i] - f(x_m, u_t)[i]) / (2 * fd_h)
                return vmap(cd_diag_i)(jnp.arange(D))
            return vmap(compute_diag_fd)(states_at, drivers_at)
        else:  # identity
            return -gamma * jnp.ones((states_at.shape[0], D))

    # --- initial carry ---
    p_window_init      = jnp.zeros((k, T, D))
    target_window_init = jnp.zeros((k, T, D))
    gamma_init         = -jnp.ones(T)

    # --- step function ---
    @jax.jit
    def _step(carry, args):
        states, p_window, target_window, gamma, is_nan, iteration = carry

        # residuals
        r_old = get_residual(f, initial_state, states, drivers)

        # compute diagonal
        if init == "rescaled_identity":
            d_init = gamma[:, None] * jnp.ones((T, D))
        else:
            d_diag = compute_jac_diag(states[:-1], drivers[1:])
            d_init = jnp.concatenate([d_diag, jnp.ones((1, D))], axis=0)

        # compute U, V from secant history
        p_blocks      = p_window[:, :-1, :].transpose(1, 0, 2)       # (T-1, k, D)
        target_blocks = target_window[:, :-1, :].transpose(1, 0, 2)  # (T-1, k, D)

        def compute_UV(d_t, steps_t, targets_t):
            V_t = steps_t.T                                          # (D, k)
            C_t = V_t.T @ V_t + eps_gram * jnp.eye(k)
            R_t = targets_t.T - d_t[:, None] * V_t
            U_t = damping * R_t @ jnp.linalg.inv(C_t)
            return U_t, V_t

        U_inner, V_inner = vmap(compute_UV)(d_init[:-1], p_blocks, target_blocks)
        U_blocks = jnp.concatenate([U_inner, jnp.zeros((1, D, k))], axis=0)
        V_blocks = jnp.concatenate([V_inner, jnp.zeros((1, D, k))], axis=0)

        # zero out U on first iteration (no secant pairs yet)
        U_blocks = jnp.where(iteration == 0, jnp.zeros_like(U_blocks), U_blocks)

        # solve block-bidiagonal system
        if sequential_solve:
            delta_s = solve_rankk_bidiag_sequential(d_init, U_blocks, V_blocks, r_old)
        else:
            delta_s = solve_rankk_bidiag(d_init, U_blocks, V_blocks, r_old, k)

        # update states
        new_states = states + delta_s
        is_nan = jnp.logical_or(jnp.isnan(new_states).any(), is_nan)
        nan_mask = jnp.isnan(new_states)
        if reset:
            new_states = jnp.where(nan_mask, states_guess, new_states)

        # new residuals for secant update
        r_new = get_residual(f, initial_state, new_states, drivers)

        # update secant windows
        y          = r_new - r_old
        new_target = jnp.concatenate(
            [y[1:] - delta_s[1:], jnp.zeros((1, D))], axis=0
        )

        p_window_new      = jnp.concatenate([delta_s[None], p_window[:-1]], axis=0)
        target_window_new = jnp.concatenate([new_target[None], target_window[:-1]], axis=0)

        # gamma update (for rescaled_identity)
        p_last   = p_window_new[0]
        tgt_last = target_window_new[0]
        gamma_num = jnp.sum(p_last * tgt_last, axis=-1)
        gamma_den = jnp.sum(p_last * p_last,   axis=-1) + 1e-10
        gamma_new = jnp.where(iteration == 0, gamma, gamma_num / gamma_den)

        mf_val = merit_fxn(f, initial_state, states, drivers)
        return (new_states, p_window_new, target_window_new, gamma_new, is_nan, iteration + 1), \
               (new_states, nan_mask, mf_val)

    # --- convergence loop ---
    def cond_func(iter_inp):
        iter_idx, _, _, _, _, err, *_ = iter_inp
        return jnp.logical_and(iter_idx < num_iters, err > tol)

    def body_func_single(iter_inp):
        iter_idx, states, pw, tw, gamma, _, is_nan_, _ = iter_inp
        step_output, _ = _step((states, pw, tw, gamma, is_nan_, iter_idx), None)
        new_states, pw_new, tw_new, gamma_new, is_nan, _ = step_output
        new_err = merit_fxn(f, initial_state, states, drivers)
        return iter_idx + 1, new_states, pw_new, tw_new, gamma_new, new_err, is_nan, None

    def body_func_multiple(iter_inp):
        iter_idx, states, pw, tw, gamma, _, is_nan_, iters_below = iter_inp
        step_output, _ = _step((states, pw, tw, gamma, is_nan_, iter_idx), None)
        new_states, pw_new, tw_new, gamma_new, is_nan, _ = step_output
        new_errs = merit_fxn(f, initial_state, states, drivers, Ts=Ts)
        iters_below = iters_below + (new_errs < tol).astype(int)
        return iter_idx + 1, new_states, pw_new, tw_new, gamma_new, new_errs[-1], is_nan, iters_below

    init_carry = (states_guess, p_window_init, target_window_init, gamma_init, False, 0)

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
            (0, states_guess, p_window_init, target_window_init, gamma_init,
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
            (0, states_guess, p_window_init, target_window_init, gamma_init,
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


# --- orthonormal variant ---

def accelerated_broyden_deer_alg_ortho(
    f: Callable,
    initial_state: jnp.ndarray,
    states_guess: jnp.ndarray,
    drivers: jnp.ndarray,
    num_iters: int,
    k: int = 1,
    init: str = "autodiff",
    gamma: float = 1.0,
    qmem_efficient: bool = False,
    sequential_solve: bool = False,
    damping: float = 1.0,
    fd_h: float = 1e-5,
    delta_ortho: float = 1e-12,
    full_trace: bool = False,
    Ts=None,
    reset: bool = True,
    tol: float = 5e-4,
    key=None,
):
    """
    Orthonormal Accelerated Broyden DEER.

    Same as accelerated_broyden_deer_alg but maintains V_t with orthonormal columns
    via incremental Gram-Schmidt.  U_t = Gamma_t - diag(d_t) V_t  (closed-form, no
    linear solve).

    Args:
        f              : dynamics function (state, driver) -> new_state
        initial_state  : x_0 (D,)
        states_guess   : initial trajectory guess (T, D)
        drivers        : inputs (T, d_in)
        num_iters      : maximum number of outer iterations
        k              : rank of the secant correction (default 1)
        init           : diagonal scheme:
                           "autodiff"          -- d_t = -diag(df/dx) via jacfwd (exact)
                           "identity"          -- d_t = -gamma * I (fixed)
                           "manual"            -- d_t via per-dimension central differences
                           "rescaled_identity" -- d_t = gamma_t * ones, scalar per block
        gamma          : scaling for identity init (default 1.0)
        qmem_efficient : if True, use Hutchinson estimator for Jacobian diagonal
        sequential_solve : if True, use sequential forward substitution
        damping        : scaling factor for the UV correction (default 1.0)
        fd_h           : step size for init="manual" (default 1e-5)
        delta_ortho    : threshold for redundancy check in basis update (default 1e-12)
        full_trace     : if True, return all intermediate states via scan
        Ts             : optional time indices for multi-checkpoint convergence
        reset          : if True, replace NaN states with states_guess
        tol            : convergence tolerance (on merit function)
        key            : PRNG key for Hutchinson estimator

    Returns:
        7-tuple matching deer_alg:
        (all_states, final_state, newton_steps, is_nan, nan_mask, all_newtons, mf_val)
    """
    if init not in ("autodiff", "identity", "manual", "rescaled_identity"):
        raise ValueError(
            f"init must be 'autodiff', 'identity', 'manual', or 'rescaled_identity', "
            f"got {init!r}"
        )

    T, D = states_guess.shape

    if key is None:
        key = jr.PRNGKey(0)

    # --- Hutchinson setup ---
    if qmem_efficient:
        def deer_jvp(state, inp, z):
            return jax.jvp(lambda x: f(x, inp), (state,), (z,))[1]
        keys = jr.split(key, T - 1)

    def compute_jac_diag(states_at, drivers_at):
        """Compute diag(-df/dx) using the chosen scheme."""
        if init == "autodiff":
            if qmem_efficient:
                return -vmap(quasi_diag_estimator, in_axes=(0, 0, None, 0))(
                    states_at, drivers_at, deer_jvp, keys)
            else:
                jacs = vmap(jacfwd(f))(states_at, drivers_at)
                return -vmap(jnp.diag)(jacs)
        elif init == "manual":
            def compute_diag_fd(x_t, u_t):
                def cd_diag_i(i):
                    x_p = x_t.at[i].add(fd_h)
                    x_m = x_t.at[i].add(-fd_h)
                    return -(f(x_p, u_t)[i] - f(x_m, u_t)[i]) / (2 * fd_h)
                return vmap(cd_diag_i)(jnp.arange(D))
            return vmap(compute_diag_fd)(states_at, drivers_at)
        else:  # identity
            return -gamma * jnp.ones((states_at.shape[0], D))

    # --- initial carry ---
    V_basis_init     = jnp.zeros((T, D, k))
    Gamma_basis_init = jnp.zeros((T, D, k))
    gamma_init       = -jnp.ones(T)

    # --- step function ---
    @jax.jit
    def _step(carry, args):
        states, V_basis, Gamma_basis, gamma, is_nan, iteration = carry

        n_valid = jnp.minimum(iteration, k)

        # residuals
        r_old = get_residual(f, initial_state, states, drivers)

        # compute diagonal
        if init == "rescaled_identity":
            d_init = gamma[:, None] * jnp.ones((T, D))
        else:
            d_diag = compute_jac_diag(states[:-1], drivers[1:])
            d_init = jnp.concatenate([d_diag, jnp.ones((1, D))], axis=0)

        # compute U from orthonormal V and Gamma
        U_blocks = damping * vmap(
            lambda d, V, G: _compute_U_ortho(d, V, G, n_valid, k)
        )(d_init, V_basis, Gamma_basis)

        # zero out U on first iteration (no secant pairs yet)
        U_blocks = jnp.where(iteration == 0, jnp.zeros_like(U_blocks), U_blocks)

        # solve block-bidiagonal system
        if sequential_solve:
            delta_s = solve_rankk_bidiag_sequential(d_init, U_blocks, V_basis, r_old)
        else:
            delta_s = solve_rankk_bidiag(d_init, U_blocks, V_basis, r_old, k)

        # update states
        new_states = states + delta_s
        is_nan = jnp.logical_or(jnp.isnan(new_states).any(), is_nan)
        nan_mask = jnp.isnan(new_states)
        if reset:
            new_states = jnp.where(nan_mask, states_guess, new_states)

        # new residuals for secant update
        r_new = get_residual(f, initial_state, new_states, drivers)

        # update orthonormal basis
        y          = r_new - r_old
        new_target = jnp.concatenate(
            [y[1:] - delta_s[1:], jnp.zeros((1, D))], axis=0
        )

        V_new, Gamma_new = vmap(
            lambda V, G, p, q: _update_basis_ortho(V, G, p, q, n_valid, k, delta_ortho)
        )(V_basis, Gamma_basis, delta_s, new_target)

        # gamma update (for rescaled_identity)
        gamma_num = jnp.sum(delta_s * new_target, axis=-1)
        gamma_den = jnp.sum(delta_s * delta_s,    axis=-1) + 1e-10
        gamma_new = jnp.where(iteration == 0, gamma, gamma_num / gamma_den)

        mf_val = merit_fxn(f, initial_state, states, drivers)
        return (new_states, V_new, Gamma_new, gamma_new, is_nan, iteration + 1), \
               (new_states, nan_mask, mf_val)

    # --- convergence loop ---
    def cond_func(iter_inp):
        iter_idx, _, _, _, _, err, *_ = iter_inp
        return jnp.logical_and(iter_idx < num_iters, err > tol)

    def body_func_single(iter_inp):
        iter_idx, states, vb, gb, gamma, _, is_nan_, _ = iter_inp
        step_output, _ = _step((states, vb, gb, gamma, is_nan_, iter_idx), None)
        new_states, vb_new, gb_new, gamma_new, is_nan, _ = step_output
        new_err = merit_fxn(f, initial_state, states, drivers)
        return iter_idx + 1, new_states, vb_new, gb_new, gamma_new, new_err, is_nan, None

    def body_func_multiple(iter_inp):
        iter_idx, states, vb, gb, gamma, _, is_nan_, iters_below = iter_inp
        step_output, _ = _step((states, vb, gb, gamma, is_nan_, iter_idx), None)
        new_states, vb_new, gb_new, gamma_new, is_nan, _ = step_output
        new_errs = merit_fxn(f, initial_state, states, drivers, Ts=Ts)
        iters_below = iters_below + (new_errs < tol).astype(int)
        return iter_idx + 1, new_states, vb_new, gb_new, gamma_new, new_errs[-1], is_nan, iters_below

    init_carry = (states_guess, V_basis_init, Gamma_basis_init, gamma_init, False, 0)

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
            (0, states_guess, V_basis_init, Gamma_basis_init, gamma_init,
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
            (0, states_guess, V_basis_init, Gamma_basis_init, gamma_init,
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
