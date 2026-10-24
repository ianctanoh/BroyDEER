"""
broyden_deer.py
DEER with limited-memory Broyden Jacobian approximations.

Each subdiagonal block b_t is initialized as  b_t^0 = diag(d_t), where d_t is
either the true Jacobian diagonal  d_t = diag(-df/dx)  ("jac" mode) or the
identity  d_t = -I  ("identity" / "rescaled_identity" modes, no jacfwd cost).
Subsequent iterates add rank-1 Broyden corrections:
    b_t^(i) = gamma_{i,t} * diag(d_t) + U_t V_t^T,   U_t, V_t in R^{D x m}.

The resulting block-bidiagonal system is solved via associative_scan
with low-rank compressed composition (QR + truncated SVD).

References:
  - Original parallel ODE solver: https://github.com/casiacob/parallel-ode/
  - Broyden / L-BFGS: Nocedal & Wright, "Numerical Optimization"
"""

from typing import Callable, Optional
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, lax, jacfwd
from jax.lax import scan

from .deer import get_residual, merit_fxn, quasi_diag_estimator
from .accelerated_broyden import lrmat_compose_rand, lrmat_compose_zero_diag, lrmat_compose_k1
from .scan_trees import get_scan_fn


# --- low-rank matrix operations ---

def lrmat_apply(d, U, V, x):
    """Apply (diag(d) + U V^T) to vector x.  d, x: (D,), U, V: (D, m)."""
    return d * x + U @ (V.T @ x)


# --- low-rank composition via QR + SVD ---

def lrmat_compose(d_a, U_a, V_a, d_b, U_b, V_b, m):
    """
    Rank-m approximation of (diag(d_b) + U_b V_b^T)(diag(d_a) + U_a V_a^T).

    Exact expansion:
        = diag(d_b * d_a) + (D_b U_a) V_a^T + U_b [D_a V_b + V_a M^T]^T

    where D_a = diag(d_a), D_b = diag(d_b), M = V_b^T U_a.

    Compressed to rank m via thin QR + truncated SVD on the (2m x 2m) core,
    costing O(D m + m^3).
    """
    M  = V_b.T @ U_a                                              # (m, m)
    Uc = jnp.concatenate([-d_b[:, None] * U_a, -U_b],        axis=1)  # (D, 2m)
    Vc = jnp.concatenate([V_a, d_a[:, None] * V_b + V_a @ M.T], axis=1)  # (D, 2m)

    Qu, Ru = jnp.linalg.qr(Uc)                                    # Qu: (D, 2m)
    Qv, Rv = jnp.linalg.qr(Vc)
    K = Ru @ Rv.T                                                  # (2m, 2m)

    Uk, sk, Vtk = jnp.linalg.svd(K, full_matrices=False)          # sk: (2m,)
    Uk  = Uk[:,  :m]
    sk  = sk[    :m]
    Vtk = Vtk[  :m, :]

    sh = jnp.sqrt(jnp.abs(sk))                                    # (m,)
    d_new = -(d_b * d_a)                                          # (D,)
    U_new = Qu @ (Uk * sh)                                        # (D, m)
    V_new = Qv @ (Vtk.T * sh)                                     # (D, m)
    return d_new, U_new, V_new


# --- combination for associative scan ---

def make_combine(m, compress="qr_svd"):
    """Return a combine function for lax.associative_scan.

    compress: "qr_svd"         — LAPACK QR + SVD on (D, 2m) (exact, reference)
              "rand"           — randomized range finder (no SVD, GPU-friendly);
                                 requires m >= 2 and a PRNG key carried per element.
              "low_rank_exact" — exact rank-m composition when d_b = 0 (no QR/SVD)
    """
    if m >= 2 and compress == "rand":
        def combine(elem_a, elem_b):
            alpha_a, d_a, U_a, V_a, dvec_a, key_a = elem_a
            alpha_b, d_b, U_b, V_b, dvec_b, key_b = elem_b
            alpha_new           = alpha_a * alpha_b
            subkey, new_key     = jr.split(key_a)
            d_new, U_new, V_new = lrmat_compose_rand(
                d_a, U_a, V_a, d_b, U_b, V_b, m, subkey)
            dvec_new            = -alpha_b * lrmat_apply(d_b, U_b, V_b, dvec_a) + dvec_b
            return alpha_new, d_new, U_new, V_new, dvec_new, new_key
    else:
        if m == 1:
            _compose = lrmat_compose_k1
        elif compress == "low_rank_exact":
            _compose = lrmat_compose_zero_diag
        else:
            _compose = lambda da, Ua, Va, db, Ub, Vb: lrmat_compose(da, Ua, Va, db, Ub, Vb, m)
        def combine(elem_a, elem_b):
            alpha_a, d_a, U_a, V_a, dvec_a = elem_a
            alpha_b, d_b, U_b, V_b, dvec_b = elem_b
            alpha_new            = alpha_a * alpha_b
            d_new, U_new, V_new  = _compose(d_a, U_a, V_a, d_b, U_b, V_b)
            dvec_new             = -alpha_b * lrmat_apply(d_b, U_b, V_b, dvec_a) + dvec_b
            return alpha_new, d_new, U_new, V_new, dvec_new
    return combine


# --- diagonal-only solve (m=0) ---

def diagonal_solve(d_blocks, r):
    """
    Solve  B @ delta_s = -r  for a pure-diagonal block-bidiagonal system (m=0).
    """
    T, D = d_blocks.shape

    def combine_diag(elem_a, elem_b):
        alpha_a, d_a, dvec_a = elem_a
        alpha_b, d_b, dvec_b = elem_b
        alpha_new = alpha_a * alpha_b
        d_new     = -(d_b * d_a)
        dvec_new  = -alpha_b * d_b * dvec_a + dvec_b
        return alpha_new, d_new, dvec_new

    alphas    = jnp.ones(T).at[0].set(0.0)
    d_scan    = jnp.concatenate([jnp.ones_like(d_blocks[:1]), d_blocks[:-1]], axis=0)
    dvec_scan = -r

    _, _, delta_s = lax.associative_scan(
        vmap(combine_diag), (alphas, d_scan, dvec_scan), reverse=False
    )
    return delta_s


# --- solve one Newton step ---

def broyden_solve(d_blocks, U_blocks, V_blocks, r, compress="qr_svd", iteration=0, scan_tree="blelloch"):
    """
    Solve  B @ delta_s = -r  where B is block-bidiagonal with subdiagonal blocks
        b_t = diag(d_blocks[t]) + U_blocks[t] V_blocks[t]^T.

    compress: "qr_svd" (default) or "rand". The "rand" path uses a randomized
    range finder (requires m >= 2) and draws fresh Gaussian sketches per
    outer iteration via fold_in(PRNGKey(42), iteration).
    scan_tree: "blelloch" (default, matches jax.lax.associative_scan) or
    "hillis_steele" — different tree topology for combining the same
    (approximately associative, under compression) blocks.
    """
    T, _, m = U_blocks.shape

    alphas = jnp.ones(T).at[0].set(0.0)
    d_scan    = jnp.concatenate([jnp.ones_like(d_blocks[:1]),  d_blocks[:-1]], axis=0)
    U_scan    = jnp.concatenate([jnp.zeros_like(U_blocks[:1]), U_blocks[:-1]], axis=0)
    V_scan    = jnp.concatenate([jnp.zeros_like(V_blocks[:1]), V_blocks[:-1]], axis=0)
    dvec_scan = -r

    combine = make_combine(m, compress=compress)
    scan_fn = get_scan_fn(scan_tree)
    if m >= 2 and compress == "rand":
        keys = jr.split(jr.fold_in(jr.PRNGKey(42), iteration), T)
        _, _, _, _, delta_s, _ = scan_fn(
            vmap(combine),
            (alphas, d_scan, U_scan, V_scan, dvec_scan, keys),
            reverse=False,
        )
    else:
        _, _, _, _, delta_s = scan_fn(
            vmap(combine), (alphas, d_scan, U_scan, V_scan, dvec_scan), reverse=False
        )
    return delta_s


# --- Broyden block update ---

def broyden_rank1_update(d_t, U_t, V_t, p_t, target_t, k=0.0):
    """
    Rank-1 Broyden update for a single subdiagonal block.
    """
    b_p   = lrmat_apply(d_t, U_t, V_t, p_t)
    denom = jnp.dot(p_t, p_t) + 1e-10
    new_u = (1 - k) * (target_t - b_p) / denom
    new_v = p_t

    U_new = jnp.concatenate([U_t[:, 1:], new_u[:, None]], axis=1)
    V_new = jnp.concatenate([V_t[:, 1:], new_v[:, None]], axis=1)
    return U_new, V_new


def update_all_blocks(d_eff, U, V, delta_s, r_old, r_new, k=0.0):
    """
    Update all T-1 subdiagonal Broyden blocks in parallel.
    """
    y      = r_new - r_old
    p      = delta_s[:-1]
    target = y[1:] - delta_s[1:]

    U_inner, V_inner = vmap(lambda d, u, v, p_, t_: broyden_rank1_update(d, u, v, p_, t_, k))(
        d_eff[:-1], U[:-1], V[:-1], p, target
    )

    U_new = jnp.concatenate([U_inner, U[-1:]], axis=0)
    V_new = jnp.concatenate([V_inner, V[-1:]], axis=0)
    return U_new, V_new


# --- secant-history replay for rescaled_identity ---

def update_secant_history(P_hist, G_hist, delta_s, r_old, r_new):
    """
    Store the latest block secant pair for each subdiagonal block.

    P_hist[t] stores recent secant directions p_t.
    G_hist[t] stores recent secant targets g_t.

    For the residual block system, the Broyden target for subdiagonal block t is
        g_t = (r_new - r_old)_{t+1} - delta_s_{t+1},
    while
        p_t = delta_s_t.
    """
    y = r_new - r_old
    p = delta_s[:-1]
    target = y[1:] - delta_s[1:]

    P_inner = jnp.concatenate([P_hist[:-1, :, 1:], p[:, :, None]], axis=2)
    G_inner = jnp.concatenate([G_hist[:-1, :, 1:], target[:, :, None]], axis=2)

    P_new = jnp.concatenate([P_inner, P_hist[-1:]], axis=0)
    G_new = jnp.concatenate([G_inner, G_hist[-1:]], axis=0)
    return P_new, G_new


def replay_broyden_block(d_t, P_t, G_t, gamma_t, k=0.0):
    """
    Rebuild one limited-memory Broyden block from the rescaled diagonal baseline
    gamma_t * diag(d_t), replaying the stored secant pairs in chronological order.
    """
    D, m = P_t.shape
    d_eff_t = gamma_t * d_t

    U0 = jnp.zeros((D, m), dtype=P_t.dtype)
    V0 = jnp.zeros((D, m), dtype=P_t.dtype)

    def replay_one(carry, inputs):
        U_cur, V_cur = carry
        p_j, g_j = inputs

        b_p = lrmat_apply(d_eff_t, U_cur, V_cur, p_j)
        denom = jnp.dot(p_j, p_j) + 1e-10

        new_u = (1.0 - k) * (g_j - b_p) / denom
        new_v = p_j

        U_next = jnp.concatenate([U_cur[:, 1:], new_u[:, None]], axis=1)
        V_next = jnp.concatenate([V_cur[:, 1:], new_v[:, None]], axis=1)
        return (U_next, V_next), None

    (U_final, V_final), _ = lax.scan(
        replay_one,
        (U0, V0),
        (P_t.T, G_t.T),
    )
    return U_final, V_final


def replay_all_blocks(d_init, P_hist, G_hist, gamma, k=0.0):
    """
    Rebuild all subdiagonal Broyden blocks by replaying stored secant pairs from
    gamma_t * diag(d_t).
    """
    U_inner, V_inner = vmap(
        lambda d, P, G, gam: replay_broyden_block(d, P, G, gam, k)
    )(
        d_init[:-1],
        P_hist[:-1],
        G_hist[:-1],
        gamma[:-1],
    )

    U_new = jnp.concatenate([U_inner, jnp.zeros_like(U_inner[:1])], axis=0)
    V_new = jnp.concatenate([V_inner, jnp.zeros_like(V_inner[:1])], axis=0)
    return U_new, V_new


# --- gamma update (L-BFGS style scalar scaling) ---

def update_gamma(gamma, delta_s, r_old, r_new, m, iteration, d):
    """
    Compute the per-block scalar gamma for the next iteration.
    """
    y   = r_new - r_old
    p   = delta_s[:-1]
    u   = y[1:] - delta_s[1:]
    dp  = d[:-1] * p

    def gamma_t(dp_t, u_t):
        return jnp.dot(dp_t, u_t) / (jnp.dot(dp_t, dp_t) + 1e-10)

    gamma_inner     = vmap(gamma_t)(dp, u)
    gamma_candidate = jnp.concatenate(
        [gamma_inner, jnp.ones_like(gamma[:1])], axis=0
    )

    return jnp.where(iteration >= m, gamma_candidate, jnp.ones_like(gamma))


# --- one-time gamma rescale (rescaled_identity) ---

def compute_gamma_rescale(delta_s, r_old, r_new, d=None):
    """
    Compute per-block gamma so that gamma * diag(d) satisfies the secant condition.
    """
    y      = r_new - r_old
    p      = delta_s[:-1]
    target = y[1:] - delta_s[1:]

    if d is not None:
        dp = d[:-1] * p
    else:
        dp = p

    def gamma_t(dp_t, tgt_t):
        return jnp.dot(dp_t, tgt_t) / (jnp.dot(dp_t, dp_t) + 1e-10)

    gamma_inner = vmap(gamma_t)(dp, target)
    return jnp.concatenate([gamma_inner, jnp.ones(1)], axis=0)


# --- sequential forward-substitution solves ---

def sequential_broyden_solve(d_blocks, U_blocks, V_blocks, r):
    """
    Sequential forward substitution for the block-bidiagonal system B @ delta = -r.
    Produces the exact solution with no QR/SVD compression errors.
    """
    def step(delta_prev, inputs):
        r_t, d_t, U_t, V_t = inputs
        delta_t = -r_t - lrmat_apply(d_t, U_t, V_t, delta_prev)
        return delta_t, delta_t

    delta_0 = -r[0]
    _, deltas_rest = lax.scan(
        step, delta_0, (r[1:], d_blocks[:-1], U_blocks[:-1], V_blocks[:-1])
    )
    return jnp.concatenate([delta_0[None], deltas_rest], axis=0)


def sequential_diagonal_solve(d_blocks, r):
    """Sequential forward substitution for the pure-diagonal (m=0) case."""
    def step(delta_prev, inputs):
        r_t, d_t = inputs
        delta_t = -r_t - d_t * delta_prev
        return delta_t, delta_t

    delta_0 = -r[0]
    _, deltas_rest = lax.scan(step, delta_0, (r[1:], d_blocks[:-1]))
    return jnp.concatenate([delta_0[None], deltas_rest], axis=0)


# --- main algorithm ---

def broyden_deer_alg(
    f: Callable,
    initial_state: jnp.ndarray,
    states_guess: jnp.ndarray,
    drivers: jnp.ndarray,
    num_iters: int,
    m: int = 10,
    init: str = "jac",
    gamma: float = 1.0,
    qmem_efficient: bool = False,
    sequential_solve: bool = False,
    k: float = 0.0,
    full_trace: bool = False,
    Ts=None,
    reset: bool = True,
    tol: float = 1e-4,
    key=None,
    lower_bounds=None,
    compress: str = "qr_svd",
    line_search: bool = False,
    ls_c: float = 1e-4,
    scan_tree: str = "blelloch",
):
    """
    Broyden-DEER: parallel Newton solver with limited-memory Broyden Jacobian blocks.

    Each subdiagonal block is initialized as b_t^0 = diag(d_t) and updated
    with rank-1 Broyden corrections:
        b_t^(i) = gamma_{i,t} * diag(d_t) + U_t V_t^T

    The resulting block-bidiagonal system is solved via associative_scan
    with low-rank compressed composition.

    Args:
        f              : dynamics function (state, driver) -> new_state
        initial_state  : x_0 (D,)
        states_guess   : initial trajectory guess (T, D)
        drivers        : inputs (T, d_in)
        num_iters      : maximum number of outer iterations
        m              : rank / window size for Broyden blocks (default 10).
                         m=0: pure diagonal solve, no rank-1 updates
        init           : initialization mode for diagonal d_t:
                           "jac"               -- d_t = -diag(df/dx) at initial guess;
                                                  initial gamma scaling = `gamma` arg
                           "identity"          -- d_t = -I, gamma fixed at `gamma` arg
                           "rescaled_identity" -- d_t = -I; gamma starts at `gamma` arg,
                                                  then rescaled from the newest secant
                                                  pair each iteration; the last m Broyden
                                                  updates are replayed from gamma_t*d_t
                           "fresh_jac"         -- d_t recomputed at current X every iteration
                                                  (gamma argument ignored)
        gamma          : scalar applied as the *initial* per-block gamma for all modes
                         except fresh_jac. For "identity" it stays fixed; for "jac"
                         and "rescaled_identity" it is overwritten by secant rescaling
                         after the first iteration. Default 1.0.
        qmem_efficient : if True, use Hutchinson estimator for Jacobian diagonal
        sequential_solve : if True, use sequential forward substitution
        k              : damping parameter for rank-1 corrections (default 0.0, no damping)
        full_trace     : if True, return all intermediate states via scan
        Ts             : optional time indices for multi-checkpoint convergence
        reset          : if True, replace NaN states with states_guess
        tol            : convergence tolerance (on merit function)
        key            : PRNG key for Hutchinson estimator
        lower_bounds   : optional per-dim lower clip applied after each step
        compress       : "qr_svd" (default) or "rand". "rand" uses a randomized
                         range finder in the associative_scan composition
                         (requires m >= 2), matching accelerated_broyden.
        line_search    : if True, damp the Newton step with ls_c when the update
                         is non-finite or raises the merit more than 10x
        ls_c           : damping factor used when the line-search guard triggers

    Returns:
        7-tuple matching deer_alg:
        (all_states, final_state, newton_steps, is_nan, nan_mask, all_newtons, mf_val)
    """
    if init not in ("jac", "identity", "rescaled_identity", "fresh_jac"):
        raise ValueError(
            f"init must be 'jac', 'identity', 'rescaled_identity', "
            f"or 'fresh_jac', got {init!r}"
        )

    T, D = states_guess.shape
    m = min(m, D)
    use_lrmat = (m > 0)

    if key is None:
        key = jr.PRNGKey(0)

    # --- Hutchinson setup ---
    if qmem_efficient:
        def deer_jvp(state, inp, z):
            return jax.jvp(lambda x: f(x, inp), (state,), (z,))[1]
        keys = jr.split(key, T - 1)

    def compute_jac_diag(states_at, drivers_at):
        """Compute -diag(df/dx) using Hutchinson or exact jacfwd."""
        if qmem_efficient:
            return -vmap(quasi_diag_estimator, in_axes=(0, 0, None, 0))(
                states_at, drivers_at, deer_jvp, keys)
        else:
            jacs = vmap(jacfwd(f))(states_at, drivers_at)
            return -vmap(jnp.diag)(jacs)

    # --- initialize d_init, U_init, V_init ---
    # gamma is now carried entirely in `gamma_init` for every mode (no longer
    # baked into d_init for "identity"). The user-supplied `gamma` argument
    # therefore acts as the initial per-block scalar uniformly.
    if init == "jac":
        d_diag = compute_jac_diag(states_guess[:-1], drivers[1:])
        d_init = jnp.concatenate([d_diag, jnp.ones((1, D))], axis=0)
    elif init == "fresh_jac":
        d_init = jnp.ones((T, D))
    elif init == "identity":
        d_init = -jnp.ones((T, D))
    else:  # "rescaled_identity"
        d_init = -jnp.ones((T, D))

    U_init = jnp.zeros((T, D, max(m, 1)))
    V_init = jnp.zeros((T, D, max(m, 1)))
    gamma_init = gamma * jnp.ones(T)
    P_hist_init = jnp.zeros((T, D, max(m, 1)))
    G_hist_init = jnp.zeros((T, D, max(m, 1)))

    # --- step function ---
    @jax.jit
    def _step(carry, args):
        states, U, V, P_hist, G_hist, gamma, is_nan, iteration = carry

        # effective diagonal
        if init == "fresh_jac":
            d_diag_fresh = compute_jac_diag(states[:-1], drivers[1:])
            d_eff = jnp.concatenate([d_diag_fresh, jnp.ones((1, D))], axis=0)
        else:
            d_eff = gamma[:, None] * d_init

        # compute residual (1 vmap of f)
        r_old = get_residual(f, initial_state, states, drivers)

        # solve block-bidiagonal system
        if use_lrmat:
            delta_s = (sequential_broyden_solve(d_eff, U, V, r_old)
                       if sequential_solve else
                       broyden_solve(d_eff, U, V, r_old,
                                     compress=compress, iteration=iteration, scan_tree=scan_tree))
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
        if lower_bounds is not None:
            new_states = jnp.maximum(new_states, lower_bounds)

        # new residuals for Broyden update
        r_new = get_residual(f, initial_state, new_states, drivers)

        # backtracking line search: damp if non-finite OR merit jumped >10x
        if line_search:
            def _merit_from_r(r):
                r2 = jnp.where(jnp.isfinite(r), r**2, jnp.inf)
                return 0.5 * jnp.sum(jnp.cumsum(r2, axis=0), axis=1)[-1]

            merit_old = _merit_from_r(r_old)
            merit_new = _merit_from_r(r_new)

            bad = jnp.logical_or(
                jnp.logical_or(
                    (~jnp.isfinite(new_states)).any(),
                    (~jnp.isfinite(r_new)).any(),
                ),
                merit_new > 10.0 * merit_old,
            )

            def _damp(_):
                cand = states + ls_c * delta_s
                cand = jnp.where(~jnp.isfinite(cand), states_guess, cand)
                if lower_bounds is not None:
                    cand = jnp.maximum(cand, lower_bounds)
                r_cand = get_residual(f, initial_state, cand, drivers)
                return cand, r_cand

            def _keep(_):
                return new_states, r_new

            new_states, r_new = lax.cond(bad, _damp, _keep, None)
            delta_s = new_states - states
            bad_mask = ~jnp.isfinite(new_states)
            is_nan = jnp.logical_or(bad_mask.any(), is_nan)
            nan_mask = bad_mask

        # update Broyden blocks
        P_hist_new, G_hist_new = P_hist, G_hist

        if init == "fresh_jac":
            if use_lrmat:
                U_new, V_new = update_all_blocks(d_eff, U, V, delta_s, r_old, r_new, k)
            else:
                U_new, V_new = U, V
            gamma_new = gamma

        elif init == "rescaled_identity":
            # L-BFGS-style scalar baseline: compute gamma from the newest
            # secant pair, then rebuild the limited-memory Broyden factors by
            # replaying the stored secants from gamma_t * diag(d_t).
            gamma_new = compute_gamma_rescale(delta_s, r_old, r_new, d=d_init)

            if use_lrmat:
                P_hist_new, G_hist_new = update_secant_history(
                    P_hist, G_hist, delta_s, r_old, r_new
                )
                U_new, V_new = replay_all_blocks(
                    d_init, P_hist_new, G_hist_new, gamma_new, k
                )
            else:
                U_new, V_new = U, V

        elif init == "jac":
            gamma_new = compute_gamma_rescale(delta_s, r_old, r_new, d=d_init)
            if use_lrmat:
                d_eff_update = gamma_new[:, None] * d_init
                U_new, V_new = update_all_blocks(d_eff_update, U, V, delta_s, r_old, r_new, k)
            else:
                U_new, V_new = U, V

        else:  # "identity"
            if use_lrmat:
                U_new, V_new = update_all_blocks(d_eff, U, V, delta_s, r_old, r_new, k)
            else:
                U_new, V_new = U, V
            # keep the user-supplied gamma constant — no secant rescaling
            gamma_new = gamma

        # stopping criterion: ||r||_2 / sqrt(T*D) of the updated states
        # (r_new, after any line-search damping), as in deer/picard/jacobi
        mf_val = jnp.sqrt(jnp.mean(r_new ** 2))
        mf_val = jnp.where(jnp.isnan(mf_val), jnp.inf, mf_val)

        return (new_states, U_new, V_new, P_hist_new, G_hist_new,
                gamma_new, is_nan, iteration + 1), \
               (new_states, nan_mask, mf_val)

    # --- convergence loop ---
    def cond_func(iter_inp):
        iter_idx, _, _, _, _, _, _, err, *_ = iter_inp
        return jnp.logical_and(iter_idx < num_iters, err > tol)

    def body_func_single(iter_inp):
        iter_idx, states, U, V, P_hist, G_hist, gamma, _, is_nan_, _ = iter_inp
        step_output, step_aux = _step(
            (states, U, V, P_hist, G_hist, gamma, is_nan_, iter_idx), None
        )
        new_states, U_new, V_new, P_hist_new, G_hist_new, gamma_new, is_nan, _ = step_output
        _, _, mf_val = step_aux
        return (iter_idx + 1, new_states, U_new, V_new, P_hist_new, G_hist_new,
                gamma_new, mf_val, is_nan, None)

    def body_func_multiple(iter_inp):
        iter_idx, states, U, V, P_hist, G_hist, gamma, _, is_nan_, iters_below = iter_inp
        step_output, step_aux = _step(
            (states, U, V, P_hist, G_hist, gamma, is_nan_, iter_idx), None
        )
        new_states, U_new, V_new, P_hist_new, G_hist_new, gamma_new, is_nan, _ = step_output
        new_errs = merit_fxn(f, initial_state, states, drivers, Ts=Ts)
        iters_below = iters_below + (new_errs < tol).astype(int)
        return (iter_idx + 1, new_states, U_new, V_new, P_hist_new, G_hist_new,
                gamma_new, new_errs[-1], is_nan, iters_below)

    init_carry = (
        states_guess,
        U_init,
        V_init,
        P_hist_init,
        G_hist_init,
        gamma_init,
        False,
        0,
    )

    # initial stopping criterion: ||r||_2 / sqrt(T*D)
    r_init = get_residual(f, initial_state, states_guess, drivers)
    merit_init = jnp.sqrt(jnp.mean(r_init ** 2))
    merit_init = jnp.where(jnp.isnan(merit_init), jnp.inf, merit_init)

    if full_trace:
        last_output, all_outputs = scan(
            _step, init_carry, None, length=num_iters
        )
        final_state = last_output[0]
        is_nan = last_output[6]
        all_states, nan_mask, mf_val = all_outputs
        newton_steps, iters_below = None, None
        all_states = jnp.concatenate([states_guess[None, ...], all_states])
    elif Ts is not None:
        result = jax.lax.while_loop(
            cond_func,
            body_func_multiple,
            (0, states_guess, U_init, V_init, P_hist_init, G_hist_init, gamma_init,
             merit_init,
             False, jnp.zeros_like(Ts)),
        )
        newton_steps, final_state = result[0], result[1]
        is_nan, iters_below = result[8], result[9]
        all_states, nan_mask, mf_val = None, None, None
    else:
        result = jax.lax.while_loop(
            cond_func,
            body_func_single,
            (0, states_guess, U_init, V_init, P_hist_init, G_hist_init, gamma_init,
             merit_init,
             False, None),
        )
        newton_steps, final_state = result[0], result[1]
        is_nan, iters_below = result[8], result[9]
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