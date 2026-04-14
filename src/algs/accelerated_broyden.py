"""
accelerated_broyden.py
DEER variant: fresh true diagonal + rank-m secant correction per block.

At each iteration i, the subdiagonal block b_t is:

    b_t^(i) = diag(d_t^(i)) + U_t^(i) V_t^(i)^T

where:
  - d_t^(i) = -diag(df/dx)|_{X^(i)[t]}   recomputed fresh each iteration
  - V_t^(i) = [p_t^(i-1), ..., p_t^(i-m)]  the last m Newton steps (sliding window)
  - U_t^(i) is chosen so that all m secant conditions hold simultaneously:
        b_t^(i) @ p_t^(i-j) = target_t^(i-j),  j = 1, ..., m

    Rearranging:  U_t C_t = R_t
    where  C_t = V_t^T V_t  (m x m Gram matrix of stored steps)
           R_t = [target_t^(i-j) - d_t^(i) * p_t^(i-j)]_{j=1}^m  (D x m)
    Solved as:  U_t = solve(C_t + eps I, R_t^T)^T

The block-bidiagonal system is solved via lax.associative_scan with rank-m
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
    """Apply (diag(d) + U V^T) to vector x.  d, x: (D,), U, V: (D, m)."""
    return d * x + U @ (V.T @ x)


# --- rank-m composition via QR + SVD ---

def lrmat_compose(d_a, U_a, V_a, d_b, U_b, V_b, m):
    """
    Rank-m approximation of (diag(d_b) + U_b V_b^T)(diag(d_a) + U_a V_a^T).

    Exact rank-2m factorization:
        Uc = [ D_b U_a  |  U_b ]
        Vc = [ V_a      |  D_a V_b + V_a M^T ]   where M = V_b^T U_a

    Compressed to rank-m via thin QR + truncated SVD on the (2m x 2m) core.
    O(D m + m^3) cost.
    """
    M  = V_b.T @ U_a
    Uc = jnp.concatenate([-d_b[:, None] * U_a, -U_b],       axis=1)
    Vc = jnp.concatenate([V_a, d_a[:, None] * V_b + V_a @ M.T], axis=1)

    Qu, Ru = jnp.linalg.qr(Uc)
    Qv, Rv = jnp.linalg.qr(Vc)
    K = Ru @ Rv.T

    Uk, sk, Vtk = jnp.linalg.svd(K, full_matrices=False)
    sh    = jnp.sqrt(jnp.abs(sk[:m]))
    d_new = -(d_b * d_a)
    U_new = (Qu @ Uk[:, :m]) * sh
    V_new = (Qv @ Vtk[:m, :].T) * sh
    return d_new, U_new, V_new


# --- GPU-friendly composition (avoids QR/SVD LAPACK kernel launches) ---

def _gram_schmidt_2(a, b):
    """Manual QR of [a, b] (D, 2) via Gram-Schmidt. Returns (q1, q2, r11, r12, r22)."""
    r11 = jnp.linalg.norm(a)
    r11_safe = jnp.maximum(r11, 1e-30)
    q1 = a / r11_safe
    r12 = jnp.dot(q1, b)
    b_perp = b - r12 * q1
    r22 = jnp.linalg.norm(b_perp)
    r22_safe = jnp.maximum(r22, 1e-30)
    q2 = b_perp / r22_safe
    return q1, q2, r11, r12, r22


def _svd2_top1(k00, k01, k10, k11):
    """Top singular triplet of 2x2 matrix [[k00,k01],[k10,k11]].
    Returns (sigma1, ul0, ul1, vr0, vr1)."""
    a = k00 * k00 + k10 * k10
    b = k00 * k01 + k10 * k11
    c = k01 * k01 + k11 * k11
    half_tr = (a + c) / 2
    disc = jnp.sqrt(jnp.maximum(half_tr * half_tr - (a * c - b * b), 0.0))
    lam1 = half_tr + disc
    sigma1 = jnp.sqrt(jnp.maximum(lam1, 0.0))
    vr0 = b
    vr1 = lam1 - a
    vr_norm = jnp.sqrt(vr0 * vr0 + vr1 * vr1)
    safe = vr_norm > 1e-30
    vr_norm_safe = jnp.maximum(vr_norm, 1e-30)
    vr0 = jnp.where(safe, vr0 / vr_norm_safe, jnp.where(a >= c, 1.0, 0.0))
    vr1 = jnp.where(safe, vr1 / vr_norm_safe, jnp.where(a >= c, 0.0, 1.0))
    sigma1_safe = jnp.maximum(sigma1, 1e-30)
    ul0 = (k00 * vr0 + k01 * vr1) / sigma1_safe
    ul1 = (k10 * vr0 + k11 * vr1) / sigma1_safe
    return sigma1, ul0, ul1, vr0, vr1


def lrmat_compose_k1(d_a, U_a, V_a, d_b, U_b, V_b):
    """Rank-1 composition: manual Gram-Schmidt + analytic 2x2 SVD."""
    u_a, v_a = U_a[:, 0], V_a[:, 0]
    u_b, v_b = U_b[:, 0], V_b[:, 0]
    m = jnp.dot(v_b, u_a)
    uc1 = -d_b * u_a
    uc2 = -u_b
    vc1 = v_a
    vc2 = d_a * v_b + m * v_a
    qu1, qu2, ru11, ru12, ru22 = _gram_schmidt_2(uc1, uc2)
    qv1, qv2, rv11, rv12, rv22 = _gram_schmidt_2(vc1, vc2)
    k00 = ru11 * rv11 + ru12 * rv12
    k01 = ru12 * rv22
    k10 = ru22 * rv12
    k11 = ru22 * rv22
    sigma1, ul0, ul1, vr0, vr1 = _svd2_top1(k00, k01, k10, k11)
    sh = jnp.sqrt(sigma1)
    d_new = -(d_b * d_a)
    U_new = ((qu1 * ul0 + qu2 * ul1) * sh)[:, None]
    V_new = ((qv1 * vr0 + qv2 * vr1) * sh)[:, None]
    return d_new, U_new, V_new


def lrmat_compose_rand(d_a, U_a, V_a, d_b, U_b, V_b, m, key):
    """
    GPU-friendly rank-m composition via randomized range finder.
    No QR on (D,2m), no SVD. Just sketch + GS on m columns + matmuls.
    Uses ℓ = m (no oversampling).
    """
    D = d_a.shape[0]

    M = V_b.T @ U_a
    Uc = jnp.concatenate([-d_b[:, None] * U_a, -U_b], axis=1)
    Vc = jnp.concatenate([V_a, d_a[:, None] * V_b + V_a @ M.T], axis=1)

    # Gaussian sketch matrix (different key per composition)
    G = jr.normal(key, (D, m))

    # Sketch: H = (Uc Vc^T) G = Uc (Vc^T G)
    H = Uc @ (Vc.T @ G)                       # (D, m)

    # Orthonormalize H via Modified Gram-Schmidt (m columns)
    qs = []
    for i in range(m):
        v = H[:, i]
        for j in range(i):
            v = v - jnp.dot(qs[j], v) * qs[j]
        v = v / jnp.maximum(jnp.linalg.norm(v), 1e-30)
        qs.append(v)
    Q = jnp.stack(qs, axis=1)                  # (D, m)

    # Factored form: A ≈ Q (Q^T A) = Q (Q^T Uc) Vc^T = U_new V_new^T
    P = Q.T @ Uc                               # (m, 2m)
    V_new = Vc @ P.T                           # (D, m)

    d_new = -(d_b * d_a)
    return d_new, Q, V_new


def lrmat_compose_gpu(d_a, U_a, V_a, d_b, U_b, V_b, m):
    """
    GPU-friendly rank-m composition for general m.
    Replaces jnp.linalg.qr on (D, 2k) matrices with unrolled Modified
    Gram-Schmidt (pure vector ops, XLA-fuseable).  Keeps jnp.linalg.svd
    on the tiny (2k, 2k) core for exact truncation.
    """
    n = 2 * m

    M = V_b.T @ U_a
    Uc = jnp.concatenate([-d_b[:, None] * U_a, -U_b], axis=1)
    Vc = jnp.concatenate([V_a, d_a[:, None] * V_b + V_a @ M.T], axis=1)

    # --- Manual Modified Gram-Schmidt (replaces jnp.linalg.qr) ---
    # Loops unroll at trace time since n is a Python constant.
    qs_u, qs_v = [], []
    R_u = jnp.zeros((n, n))
    R_v = jnp.zeros((n, n))

    for i in range(n):
        vu = Uc[:, i]
        vv = Vc[:, i]
        for j in range(i):
            ru_ji = jnp.dot(qs_u[j], vu)
            vu = vu - ru_ji * qs_u[j]
            R_u = R_u.at[j, i].set(ru_ji)
            rv_ji = jnp.dot(qs_v[j], vv)
            vv = vv - rv_ji * qs_v[j]
            R_v = R_v.at[j, i].set(rv_ji)
        ru_ii = jnp.linalg.norm(vu)
        vu = vu / jnp.maximum(ru_ii, 1e-30)
        qs_u.append(vu)
        R_u = R_u.at[i, i].set(ru_ii)
        rv_ii = jnp.linalg.norm(vv)
        vv = vv / jnp.maximum(rv_ii, 1e-30)
        qs_v.append(vv)
        R_v = R_v.at[i, i].set(rv_ii)

    Qu = jnp.stack(qs_u, axis=1)
    Qv = jnp.stack(qs_v, axis=1)

    # --- Core matrix ---
    K_core = R_u @ R_v.T                    # (2k, 2k)

    # --- Exact truncated SVD on tiny core ---
    Uk, sk, Vtk = jnp.linalg.svd(K_core, full_matrices=False)

    sh    = jnp.sqrt(jnp.abs(sk[:m]))
    d_new = -(d_b * d_a)
    U_new = (Qu @ Uk[:, :m]) * sh
    V_new = (Qv @ Vtk[:m, :].T) * sh
    return d_new, U_new, V_new


# --- combination for associative scan ---

def make_combine(m, compress="rand"):
    """Return a combine function for lax.associative_scan with rank-m blocks.

    compress: "rand" — randomized range finder (no SVD, fastest on GPU)
              "gs_svd" — manual Gram-Schmidt + LAPACK SVD on tiny core
              "qr_svd" — original LAPACK QR + SVD (reference)

    When compress="rand" and m >= 2, the scan state carries a PRNG key per
    element so each composition draws a fresh Gaussian sketch.
    """
    if m >= 2 and compress == "rand":
        # --- rand path: 6-element state with PRNG key ---
        def combine(elem_a, elem_b):
            alpha_a, d_a, U_a, V_a, dvec_a, key_a = elem_a
            alpha_b, d_b, U_b, V_b, dvec_b, key_b = elem_b
            alpha_new            = alpha_a * alpha_b
            subkey, new_key      = jr.split(key_a)
            d_new, U_new, V_new  = lrmat_compose_rand(
                d_a, U_a, V_a, d_b, U_b, V_b, m, subkey)
            dvec_new             = -alpha_b * lrmat_apply(d_b, U_b, V_b, dvec_a) + dvec_b
            return alpha_new, d_new, U_new, V_new, dvec_new, new_key
    else:
        # --- all other paths: 5-element state, no key ---
        if m == 1:
            _compose = lrmat_compose_k1
        elif compress == "gs_svd":
            _compose = lambda da, Ua, Va, db, Ub, Vb: lrmat_compose_gpu(da, Ua, Va, db, Ub, Vb, m)
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


# --- block-bidiagonal solve (parallel) ---

def solve_rankk_bidiag(d_blocks, U_blocks, V_blocks, r, m, compress="rand"):
    """Solve  B @ delta_s = -r  via lax.associative_scan."""
    T, D = d_blocks.shape
    alphas = jnp.ones(T).at[0].set(0.0)
    d_scan = jnp.concatenate([jnp.ones((1, D)),     d_blocks[:-1]], axis=0)
    U_scan = jnp.concatenate([jnp.zeros((1, D, m)), U_blocks[:-1]], axis=0)
    V_scan = jnp.concatenate([jnp.zeros((1, D, m)), V_blocks[:-1]], axis=0)
    dvec   = -r

    if m >= 2 and compress == "rand":
        keys = jr.split(jr.PRNGKey(42), T)
        _, _, _, _, delta_s, _ = lax.associative_scan(
            vmap(make_combine(m, compress=compress)),
            (alphas, d_scan, U_scan, V_scan, dvec, keys), reverse=False,
        )
    else:
        _, _, _, _, delta_s = lax.associative_scan(
            vmap(make_combine(m, compress=compress)),
            (alphas, d_scan, U_scan, V_scan, dvec), reverse=False,
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

def _update_basis_ortho(V_old, Gamma_old, p_new, q_new, n_valid, m, delta=1e-12):
    """
    Incrementally update an orthonormal basis V and paired targets Gamma
    when a new secant pair (p_new, q_new) arrives.
    """
    col_mask = (jnp.arange(m) >= (m - n_valid)).astype(V_old.dtype)

    alpha = (V_old.T @ p_new) * col_mask
    p_tilde = p_new - V_old @ alpha
    norm_p = jnp.linalg.norm(p_tilde)

    safe_norm = jnp.where(norm_p < delta, 1.0, norm_p)
    v_new = p_tilde / safe_norm
    q_tilde = (q_new - Gamma_old @ alpha) / safe_norm

    V_slid = jnp.concatenate([V_old[:, 1:], v_new[:, None]], axis=1)
    Gamma_slid = jnp.concatenate([Gamma_old[:, 1:], q_tilde[:, None]], axis=1)

    insert_idx = m - n_valid - 1
    V_grown = V_old.at[:, insert_idx].set(v_new)
    Gamma_grown = Gamma_old.at[:, insert_idx].set(q_tilde)

    basis_full = n_valid >= m
    V_updated = jnp.where(basis_full, V_slid, V_grown)
    Gamma_updated = jnp.where(basis_full, Gamma_slid, Gamma_grown)

    skip = norm_p < delta
    V_new = jnp.where(skip, V_old, V_updated)
    Gamma_new = jnp.where(skip, Gamma_old, Gamma_updated)
    return V_new, Gamma_new


def _compute_U_ortho(d_t, V_t, Gamma_t, n_valid, m):
    """Closed-form U from orthonormal V:  U = Gamma - diag(d) V."""
    U = Gamma_t - d_t[:, None] * V_t
    col_mask = (jnp.arange(m) >= (m - n_valid)).astype(V_t.dtype)
    return U * col_mask[None, :]


# --- main algorithm ---

def accelerated_broyden_deer_alg(
    f: Callable,
    initial_state: jnp.ndarray,
    states_guess: jnp.ndarray,
    drivers: jnp.ndarray,
    num_iters: int,
    m: int = 1,
    init: str = "autodiff",
    gamma: float = 1.0,
    qmem_efficient: bool = False,
    sequential_solve: bool = False,
    compress: str = "rand",
    k: float = 0.0,
    fd_h: float = 1e-5,
    eps_gram: float = 1e-10,
    full_trace: bool = False,
    Ts=None,
    reset: bool = True,
    tol: float = 5e-4,
    key=None,
):
    """
    Accelerated Broyden DEER: fresh diagonal + rank-m secant correction.

    Each iteration recomputes the diagonal d_t and fits rank-m secant corrections
    U_t, V_t from the sliding window of m previous Newton steps.

    Args:
        f              : dynamics function (state, driver) -> new_state
        initial_state  : x_0 (D,)
        states_guess   : initial trajectory guess (T, D)
        drivers        : inputs (T, d_in)
        num_iters      : maximum number of outer iterations
        m              : rank of the secant correction (default 1)
        init           : diagonal scheme:
                           "autodiff"          -- d_t = -diag(df/dx) via jacfwd (exact)
                           "identity"          -- d_t = -gamma * I (fixed)
                           "manual"            -- d_t via per-dimension central differences
                           "rescaled_identity" -- d_t = gamma_t * ones, scalar per block
        gamma          : scaling for identity init (default 1.0)
        qmem_efficient : if True, use Hutchinson estimator for Jacobian diagonal
                         (only applies to init="autodiff")
        sequential_solve : if True, use sequential forward substitution
        k              : damping parameter for the rank-m secant correction (default 0.0, no damping)
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
    p_window_init      = jnp.zeros((T, m, D))
    target_window_init = jnp.zeros((T, m, D))
    gamma_init         = -jnp.ones(T)

    # --- step function ---
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
        if m == 1:
            p_blocks      = p_window[:-1, 0, :]       # (T-1, D)
            target_blocks = target_window[:-1, 0, :]   # (T-1, D)

            def compute_UV(d_t, step_t, target_t):
                v_t = step_t                                          # (D,)
                r_t = target_t - d_t * v_t                            # (D,)
                denom = jnp.dot(v_t, v_t) + eps_gram
                u_t = (1 - k) * r_t / denom
                return u_t[:, None], v_t[:, None]                     # (D, 1), (D, 1)

            U_inner, V_inner = vmap(compute_UV)(d_init[:-1], p_blocks, target_blocks)
        elif m == 2:
            p_blocks      = p_window[:-1]              # (T-1, 2, D)
            target_blocks = target_window[:-1]         # (T-1, 2, D)

            def compute_UV(d_t, steps_t, targets_t):
                V_t = steps_t.T                                       # (D, 2)
                R_t = targets_t.T - d_t[:, None] * V_t                # (D, 2)
                C_t = V_t.T @ V_t + eps_gram * jnp.eye(2)             # (2, 2)
                # Manual 2x2 solve via Cramer's rule (avoids cuSOLVER)
                det = C_t[0, 0] * C_t[1, 1] - C_t[0, 1] * C_t[1, 0]
                inv_det = 1.0 / jnp.maximum(det, 1e-30)
                # U_t = (1 - k) * R_t @ C_t^{-1}  (C_t is symmetric)
                U_t = (1 - k) * inv_det * jnp.stack([
                    R_t[:, 0] * C_t[1, 1] - R_t[:, 1] * C_t[0, 1],
                    R_t[:, 1] * C_t[0, 0] - R_t[:, 0] * C_t[1, 0],
                ], axis=1)
                return U_t, V_t

            U_inner, V_inner = vmap(compute_UV)(d_init[:-1], p_blocks, target_blocks)
        else:
            p_blocks      = p_window[:-1]              # (T-1, m, D)
            target_blocks = target_window[:-1]         # (T-1, m, D)

            def compute_UV(d_t, steps_t, targets_t):
                V_t = steps_t.T                                       # (D, m)
                R_t = targets_t.T - d_t[:, None] * V_t                # (D, m)
                C_t = V_t.T @ V_t + eps_gram * jnp.eye(m)             # (m, m)
                U_t = (1 - k) * jnp.linalg.solve(C_t, R_t.T).T        # (D, m)
                return U_t, V_t

            U_inner, V_inner = vmap(compute_UV)(d_init[:-1], p_blocks, target_blocks)
        U_blocks = jnp.concatenate([U_inner, jnp.zeros((1, D, m))], axis=0)
        V_blocks = jnp.concatenate([V_inner, jnp.zeros((1, D, m))], axis=0)

        # zero out U on first iteration (no secant pairs yet)
        U_blocks = jnp.where(iteration == 0, jnp.zeros_like(U_blocks), U_blocks)

        # solve block-bidiagonal system
        if sequential_solve:
            delta_s = solve_rankk_bidiag_sequential(d_init, U_blocks, V_blocks, r_old)
        else:
            delta_s = solve_rankk_bidiag(d_init, U_blocks, V_blocks, r_old, m, compress=compress)

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

        p_window_new      = jnp.concatenate([delta_s[:, None, :], p_window[:, :-1, :]], axis=1)
        target_window_new = jnp.concatenate([new_target[:, None, :], target_window[:, :-1, :]], axis=1)

        # gamma update (for rescaled_identity)
        p_last   = p_window_new[:, 0, :]
        tgt_last = target_window_new[:, 0, :]
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
    m: int = 1,
    init: str = "autodiff",
    gamma: float = 1.0,
    qmem_efficient: bool = False,
    sequential_solve: bool = False,
    k: float = 0.0,
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
        m              : rank of the secant correction (default 1)
        init           : diagonal scheme:
                           "autodiff"          -- d_t = -diag(df/dx) via jacfwd (exact)
                           "identity"          -- d_t = -gamma * I (fixed)
                           "manual"            -- d_t via per-dimension central differences
                           "rescaled_identity" -- d_t = gamma_t * ones, scalar per block
        gamma          : scaling for identity init (default 1.0)
        qmem_efficient : if True, use Hutchinson estimator for Jacobian diagonal
        sequential_solve : if True, use sequential forward substitution
        k              : damping parameter for the UV correction (default 0.0, no damping)
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
    V_basis_init     = jnp.zeros((T, D, m))
    Gamma_basis_init = jnp.zeros((T, D, m))
    gamma_init       = -jnp.ones(T)

    # --- step function ---
    @jax.jit
    def _step(carry, args):
        states, V_basis, Gamma_basis, gamma, is_nan, iteration = carry

        n_valid = jnp.minimum(iteration, m)

        # residuals
        r_old = get_residual(f, initial_state, states, drivers)

        # compute diagonal
        if init == "rescaled_identity":
            d_init = gamma[:, None] * jnp.ones((T, D))
        else:
            d_diag = compute_jac_diag(states[:-1], drivers[1:])
            d_init = jnp.concatenate([d_diag, jnp.ones((1, D))], axis=0)

        # compute U from orthonormal V and Gamma
        U_blocks = (1 - k) * vmap(
            lambda d, V, G: _compute_U_ortho(d, V, G, n_valid, m)
        )(d_init, V_basis, Gamma_basis)

        # zero out U on first iteration (no secant pairs yet)
        U_blocks = jnp.where(iteration == 0, jnp.zeros_like(U_blocks), U_blocks)

        # solve block-bidiagonal system
        if sequential_solve:
            delta_s = solve_rankk_bidiag_sequential(d_init, U_blocks, V_basis, r_old)
        else:
            delta_s = solve_rankk_bidiag(d_init, U_blocks, V_basis, r_old, m)

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
            lambda V, G, p, q: _update_basis_ortho(V, G, p, q, n_valid, m, delta_ortho)
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
