"""
taa.py
Triangular Anderson Acceleration (TAA) for fixed-point trajectory solvers.

Adapted from ParaTAA-Diffusion (parallel_sd.py: ParallelDDIM.taa_step). Implements
per-timestep Anderson acceleration with a *triangular cumulative Gram matrix*: at
timestep t, the normal equations use information aggregated over all timesteps
t' >= t. This couples later (more refined) timesteps to drive corrections at
earlier ones.

Differences from ParaTAA:
  - Generic single-step Picard residual r(x_t) = f(x_{t-1}, u_t) - x_t. We do
    NOT use the DDIM-specific higher-order residual (compute_ho_residual), which
    only works because Stable Diffusion's DDIM step is *linear* in the noise
    prediction direction, allowing many parallel "future" sample updates from a
    single eps evaluation.
  - No sliding window over timesteps: all T timesteps are updated at every
    iteration, matching the BroDeer convention used by deer_alg, picard_alg, etc.
  - All T timesteps share the same valid-memory depth (no per-timestep
    memory_indexes book-keeping), since they advance in lock-step.

API matches deer_alg / picard_alg: returns the standard 7-tuple
  (all_states, final_state, newton_steps, is_nan, nan_mask, all_newtons, mf_val)

Setting `memory_size = 1` recovers plain Picard iteration (no secant pairs are
ever formed, so the Anderson correction Gf is identically zero).
"""

from typing import Callable, Optional
import jax
import jax.numpy as jnp
from jax import vmap
from jax.lax import scan

from .deer import get_residual, merit_fxn
from .ddim_utils import merit_from_residual


def taa_alg(
    f: Callable,
    initial_state: jnp.ndarray,
    states_guess: jnp.ndarray,
    drivers: jnp.ndarray,
    num_iters: int,
    memory_size: int = 5,
    lmd: float = 1e-3,
    full_trace: bool = False,
    Ts: Optional[jnp.ndarray] = None,
    reset: bool = False,
    tol: float = 5e-4,
):
    """
    Triangular Anderson Acceleration for x = f(x) trajectory fixed-points.

    At each iteration k:
      1. Compute the Picard residual per timestep
            r_old(t) = f(x_{t-1}, u_t) - x_t
      2. Push (states, r_old) into a sliding memory of length `memory_size`
         (newest at the end, oldest dropped).
      3. Form per-timestep sample/residual differences across the memory:
            dS_i, dR_i  for i = 0..memory_size-2
      4. Build the per-timestep Gram and RHS:
            B[i, p, t] = <dR_i[t], dR_p[t]>
            d[i, t]    = <dR_i[t], r_old[t]>
      5. Cumulate them *triangularly* over timesteps (sum from t to T-1):
            B_cum[i, p, t] = sum_{t' >= t} B[i, p, t']
            d_cum[i, t]    = sum_{t' >= t} d[i, t']
      6. Solve the regularized normal equations independently at every t:
            (B_cum[t] + lmd * I) c_t = d_cum[t]
      7. Form the Anderson correction
            Gf[t] = sum_i c_t[i] * (dS_i[t] + dR_i[t])
         and update
            x_new[t] = x[t] + r_old[t] - Gf[t].

    Notes:
      - The sign of `r_old` here is the Picard *direction* (f(x) - x), which is
        the negative of `get_residual` (which returns x - f(x)).
      - When `memory_size = 1`, no secant pair is ever available, Gf == 0, and
        the update reduces to plain Picard: x_new = x + (f(x) - x) = f(x).
      - During warm-up (iter < memory_size - 1), invalid memory slots are
        masked out of the Gram and RHS so the Tikhonov-regularized solve simply
        returns zero coefficients for those columns.

    Args:
        f             : dynamics function (state, driver) -> next_state
        initial_state : x_0 of shape (D,)
        states_guess  : initial trajectory guess of shape (T, D)
        drivers       : input sequence of shape (T, ...)
        num_iters     : maximum number of TAA iterations
        memory_size   : number of past iterates to remember (default 5,
                        matching ParaTAA's `memory_size`). Set to 1 to recover
                        Picard iteration.
        lmd           : Tikhonov regularization on the Gram matrix
                        (default 1e-3, matching ParaTAA's `lmd`).
        full_trace    : if True, return the full trajectory trace via scan
        Ts            : optional checkpoints for multi-T convergence reporting
        reset         : if True, replace NaN entries by states_guess
        tol           : convergence tolerance on the merit function

    Returns:
        7-tuple matching deer_alg:
        (all_states, final_state, newton_steps, is_nan, nan_mask,
         all_newtons, mf_val)
    """
    if memory_size < 1:
        raise ValueError(f"memory_size must be >= 1, got {memory_size}")

    T, D = states_guess.shape
    m = memory_size

    # initial memory: zeros (slots are warmed up via the validity mask)
    samples_mem_init  = jnp.zeros((m, T, D))
    residual_mem_init = jnp.zeros((m, T, D))

    @jax.jit
    def _step(carry, args):
        states, samples_mem, residual_mem, is_nan, iteration = carry

        # Picard residual (direction): r_old = f(x) - x = -get_residual
        r_old = -get_residual(f, initial_state, states, drivers)  # (T, D)

        # push (states, r_old) into memory; newest at the end
        samples_mem  = jnp.concatenate(
            [samples_mem[1:],  states[None]], axis=0
        )
        residual_mem = jnp.concatenate(
            [residual_mem[1:], r_old[None]],  axis=0
        )

        if m > 1:
            # secant differences across memory dimension
            dS = samples_mem[1:]  - samples_mem[:-1]   # (m-1, T, D)
            dR = residual_mem[1:] - residual_mem[:-1]  # (m-1, T, D)

            # per-timestep Gram and RHS
            gram_per_t = jnp.einsum("itd,ptd->ipt", dR, dR)   # (m-1, m-1, T)
            d_per_t    = jnp.einsum("itd,td->it",  dR, r_old) # (m-1, T)

            # triangular cumulative sum over t (sum from t to T-1)
            gram_cum = jnp.flip(
                jnp.cumsum(jnp.flip(gram_per_t, axis=2), axis=2), axis=2
            )
            d_cum = jnp.flip(
                jnp.cumsum(jnp.flip(d_per_t, axis=1), axis=1), axis=1
            )

            # mask invalid memory slots (warm-up); valid diffs sit at the END
            n_valid_diffs = jnp.minimum(iteration, m - 1)              # 0..m-1
            valid_mask = (
                jnp.arange(m - 1) >= ((m - 1) - n_valid_diffs)
            ).astype(gram_cum.dtype)                                   # (m-1,)

            gram_cum = (gram_cum
                        * valid_mask[None, :, None]
                        * valid_mask[:, None, None])
            d_cum = d_cum * valid_mask[:, None]

            # Tikhonov regularization (broadcast across t)
            eye = jnp.eye(m - 1, dtype=gram_cum.dtype)
            gram_reg = gram_cum + lmd * eye[:, :, None]                # (m-1, m-1, T)

            # per-timestep solve: vmap over the trailing time axis
            gram_t = jnp.transpose(gram_reg, (2, 0, 1))                # (T, m-1, m-1)
            d_t    = jnp.transpose(d_cum,    (1, 0))                   # (T, m-1)
            c = vmap(jnp.linalg.solve)(gram_t, d_t)                    # (T, m-1)

            # Anderson correction Gf[t] = sum_i c[t,i] * (dS_i[t] + dR_i[t])
            A_blocks = dS + dR                                         # (m-1, T, D)
            Gf = jnp.einsum("tj,jtd->td", c, A_blocks)                 # (T, D)
        else:
            # memory_size == 1: no secant pairs ever, Anderson correction is 0
            Gf = jnp.zeros_like(states)

        # update step
        new_states = states + r_old - Gf

        is_nan = jnp.logical_or(jnp.isnan(new_states).any(), is_nan)
        nan_mask = jnp.isnan(new_states)
        if reset:
            new_states = jnp.where(nan_mask, states_guess, new_states)

        mf_val = merit_fxn(f, initial_state, states, drivers)
        return (new_states, samples_mem, residual_mem, is_nan, iteration + 1), \
               (new_states, nan_mask, mf_val)

    # --- convergence loop ---
    def cond_func(iter_inp):
        iter_idx, _, _, _, err, *_ = iter_inp
        return jnp.logical_and(iter_idx < num_iters, err > tol)

    def body_func_single(iter_inp):
        iter_idx, states, sm, rm, _, is_nan_, _ = iter_inp
        step_output, step_aux = _step((states, sm, rm, is_nan_, iter_idx), None)
        new_states, sm_new, rm_new, is_nan, _ = step_output
        _, _, mf_val = step_aux
        return iter_idx + 1, new_states, sm_new, rm_new, mf_val, is_nan, None

    def body_func_multiple(iter_inp):
        iter_idx, states, sm, rm, _, is_nan_, iters_below = iter_inp
        step_output, _ = _step((states, sm, rm, is_nan_, iter_idx), None)
        new_states, sm_new, rm_new, is_nan, _ = step_output
        new_errs = merit_fxn(f, initial_state, new_states, drivers, Ts=Ts)
        iters_below = iters_below + (new_errs < tol).astype(int)
        return iter_idx + 1, new_states, sm_new, rm_new, new_errs[-1], is_nan, iters_below

    init_carry = (states_guess, samples_mem_init, residual_mem_init, False, 0)

    if full_trace:
        last_output, all_outputs = scan(
            _step, init_carry, None, length=num_iters
        )
        final_state = last_output[0]
        is_nan = last_output[3]
        all_states, nan_mask, mf_val = all_outputs
        newton_steps, iters_below = None, None
        all_states = jnp.concatenate([states_guess[None, ...], all_states])
    elif Ts is not None:
        result = jax.lax.while_loop(
            cond_func,
            body_func_multiple,
            (0, states_guess, samples_mem_init, residual_mem_init,
             merit_fxn(f, initial_state, states_guess, drivers),
             False, jnp.zeros_like(Ts)),
        )
        newton_steps, final_state = result[0], result[1]
        is_nan, iters_below = result[5], result[6]
        all_states, nan_mask, mf_val = None, None, None
    else:
        result = jax.lax.while_loop(
            cond_func,
            body_func_single,
            (0, states_guess, samples_mem_init, residual_mem_init,
             merit_fxn(f, initial_state, states_guess, drivers),
             False, None),
        )
        newton_steps, final_state = result[0], result[1]
        is_nan, iters_below = result[5], result[6]
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


def taa_ddim_alg(
    dynamics,
    initial_state: jnp.ndarray,
    states_guess: jnp.ndarray,
    drivers: jnp.ndarray,
    num_iters: int,
    order: int = 1,
    memory_size: int = 5,
    lmd: float = 1e-3,
    window_size: Optional[int] = None,
    window_tol1: float = 1e-2,
    window_tol2: float = 5e-2,
    full_trace: bool = False,
    Ts: Optional[jnp.ndarray] = None,
    reset: bool = False,
    tol: float = 5e-4,
    gram_dtype=None,
):
    """
    DDIM-specific Triangular Anderson Acceleration with multi-step lookahead.

    This is the third ParaTAA hyperparameter: `order` is the depth of the
    higher-order Picard residual, exploiting the fact that DDIM's update is
    *linear* in the noise prediction direction. K = `order` consecutive DDIM
    steps unroll into a closed-form linear combination of K eps evaluations
    on the *current* iterate -- which we already pay for once per outer step,
    so order > 1 is essentially free.

    BroDeer index convention:
        X[0] = initial_state = x_T
        X[s] = states[s - 1] = x_{T - s}    for s = 1, ..., T
        a[s] = coef_x[T - s], b[s] = coef_eps[T - s], n[s] = coef_noise[T - s]

    Single-step DDIM:
        X[s + 1] = a[s] X[s] + b[s] eps(X[s], T - s) + n[s] z[s]

    K-step unroll from X[r] -> X[s + 1] with r = max(0, s + 1 - K):
        X[s + 1] = (prod_{k=r}^{s} a[k]) X[r]
                   + sum_{j=r}^{s} (prod_{k=j+1}^{s} a[k])
                                   * (b[j] eps(X[j], T - j) + n[j] z[j])

    With Pq[i] = prod_{j=0}^{i-1} a[j] (Pq[0] = 1):
        prod_{k=r}^{s}     a[k] = Pq[s + 1] / Pq[r]
        prod_{k=j+1}^{s}   a[k] = Pq[s + 1] / Pq[j + 1]

    The eps evaluations are reused from the single-step pass over the current
    iterate -- we evaluate eps(X[s], T - s) once per s and combine them via
    precomputed coefficient matrices `B_mat` (eps weights) and `N_mat` (noise
    weights), banded to enforce the K-step lookahead window.

    At `order = 1`, the K-step unroll collapses to the canonical single-step
    DDIM update, so the residual matches `taa_alg` exactly on the same
    dynamics (and the merit/convergence behavior is identical).

    The Anderson kernel (sliding memory, triangular cumulative Gram, per-t
    Tikhonov solve, x_new = x + r_old - Gf) is identical to `taa_alg`.
    Convergence is checked against the canonical single-step fixed point via
    `merit_fxn(f_local, ...)`, NOT the higher-order residual -- so a converged
    state with `order > 1` still satisfies the same DDIM equation.

    Args:
        dynamics      : a `DDIMDynamics` instance providing eps_model and the
                        per-timestep coefficient arrays
        initial_state : x_T of shape (D,)
        states_guess  : initial trajectory guess of shape (T, D)
        drivers       : input sequence of shape (T, D + 1) where drivers[s, :D]
                        is the per-step injected noise z_s and drivers[s, D] is
                        the DDIM timestep T - s
        num_iters     : maximum number of TAA iterations
        order         : depth of the multi-step lookahead residual (>= 1).
                        Set to 1 to recover plain TAA (single-step Picard).
        memory_size   : Anderson memory depth. Set to 1 to disable Anderson
                        and recover (higher-order) Picard.
        lmd           : Tikhonov regularization on the per-t Gram
        window_size   : if not None, enables ParaTAA's sliding-front mode.
                        Only positions in [front, front + window_size) are
                        updated each outer iteration; `front` advances past
                        every leading position whose per-iter higher-order
                        residual RMS norm drops below the (tol1, tol2)
                        threshold pair. Termination is `front == T` OR
                        global `merit < tol`, whichever comes first. Setting
                        this to None (default) recovers full-trajectory
                        updates with scalar-merit termination, identical to
                        taa_alg. Combined with `order > 1`, this is the
                        mechanism ParaTAA uses to break the T-iteration
                        causality bound on DDIM (Anderson-driven front
                        advances by multiple positions per outer step).
        window_tol1   : absolute floor on per-position residual RMS norm
                        in ParaTAA's freeze criterion. A position is
                        considered settled only when
                            sqrt(mean(r_old[s]**2)) < tol1
                        AND
                            sqrt(mean(r_old[s]**2)) < tol2 * noise_thr[t]
                        where noise_thr[t] is the eta-free natural DDIM
                        noise scale. Matches ParaTAA's `tol1` (default 1e-2).
        window_tol2   : noise-relative coefficient in the same criterion.
                        Matches ParaTAA's `tol2` (default 5e-2).
        full_trace    : if True, return the full trajectory trace via scan
        Ts            : optional checkpoints for multi-T convergence reporting
        reset         : if True, replace NaN entries by states_guess
        tol           : convergence tolerance on the merit function
        gram_dtype    : dtype for accumulating and solving the Gram system in
                        the no-window path (e.g. jnp.float64, as ParaTAA;
                        needs JAX x64). None (default) uses the states' dtype.

    Returns:
        7-tuple matching deer_alg / taa_alg:
        (all_states, final_state, newton_steps, is_nan, nan_mask,
         all_newtons, mf_val)
    """
    if memory_size < 1:
        raise ValueError(f"memory_size must be >= 1, got {memory_size}")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    T, D_dim = states_guess.shape
    m = memory_size
    K = order

    if hasattr(dynamics, "T") and dynamics.T != T:
        raise ValueError(
            f"states_guess length {T} does not match dynamics.T={dynamics.T}"
        )
    if hasattr(dynamics, "D") and dynamics.D != D_dim:
        raise ValueError(
            f"states_guess dim {D_dim} does not match dynamics.D={dynamics.D}"
        )

    # Extract dynamics fields into local cells.
    # Bound methods of equinox modules trip lax.scan caching when the module
    # has jnp.ndarray fields (see diffusion.py). Closures over the fields are
    # safe -- they hash by Python identity, not by recursion into arrays.
    eps_model_local  = dynamics.eps_model
    coef_x_local     = dynamics.coef_x
    coef_eps_local   = dynamics.coef_eps
    coef_noise_local = dynamics.coef_noise
    noise_thr_local  = getattr(dynamics, "noise_thr", None)

    # canonical single-step DDIM closure (used by merit_fxn for convergence)
    def f_local(state, driver):
        z = driver[:D_dim]
        t = driver[D_dim].astype(jnp.int32)
        eps = eps_model_local(state, t)
        return (coef_x_local[t] * state
                + coef_eps_local[t] * eps
                + coef_noise_local[t] * z)

    # --- precompute lookahead coefficient matrices (BroDeer index) ---
    s_idx     = jnp.arange(T)
    t_per_s   = T - s_idx                                        # (T,) DDIM ts
    a_per_s   = coef_x_local[t_per_s]                            # (T,)
    b_per_s   = coef_eps_local[t_per_s]                          # (T,)
    n_per_s   = coef_noise_local[t_per_s]                        # (T,)

    # Pq[i] = prod_{j=0}^{i-1} a[j], length T + 1, all strictly positive
    Pq = jnp.concatenate([jnp.ones((1,), dtype=a_per_s.dtype),
                          jnp.cumprod(a_per_s)])                 # (T+1,)

    # K-band lower-triangular mask: (s, j) with max(0, s - K + 1) <= j <= s
    ones_T = jnp.ones((T, T), dtype=Pq.dtype)
    order_mask = jnp.tril(ones_T) * jnp.triu(ones_T, k=-(K - 1)) # (T, T)

    # M_mat[s, j] = Pq[s + 1] / Pq[j + 1] inside the band, 0 outside
    M_mat = (Pq[1:][:, None] / Pq[1:][None, :]) * order_mask     # (T, T)
    B_mat = M_mat * b_per_s[None, :]                              # eps weights
    N_mat = M_mat * n_per_s[None, :]                              # z weights

    # ref_idx[s] = max(0, s + 1 - K), ref_coef[s] = Pq[s + 1] / Pq[ref_idx[s]]
    ref_idx  = jnp.maximum(0, s_idx + 1 - K)                      # (T,)
    ref_coef = Pq[s_idx + 1] / Pq[ref_idx]                        # (T,)

    # per-step injected noise vectors
    z_per_s = drivers[:, :D_dim]                                  # (T, D)

    # initial Anderson memory (warm-up via valid_mask, same as taa_alg)
    samples_mem_init  = jnp.zeros((m, T, D_dim), dtype=states_guess.dtype)
    residual_mem_init = jnp.zeros((m, T, D_dim), dtype=states_guess.dtype)

    # ===== sliding-window code path (ParaTAA front advancement) =====
    if window_size is not None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if Ts is not None:
            raise NotImplementedError(
                "taa_ddim_alg with both window_size and Ts (multi-checkpoint "
                "reporting) is not yet supported"
            )
        if noise_thr_local is None:
            raise ValueError(
                "taa_ddim_alg with window_size requires dynamics.noise_thr "
                "for ParaTAA's noise-relative freeze criterion. Either pass "
                "a DDIMDynamics built via the updated factories, or attach a "
                "noise_thr field manually."
            )
        W = int(window_size)
        wtol1 = float(window_tol1)
        wtol2 = float(window_tol2)
        front_init = jnp.int32(0)
        s_idx_full = jnp.arange(T, dtype=jnp.int32)
        # noise_thr in BroDeer's per-s indexing (s = 0..T-1, t_per_s[s] = T - s)
        noise_thr_per_s = noise_thr_local[t_per_s]                    # (T,)

        # mem_count tracks per-position rolling-memory pushes (matches
        # ParaTAA's `memory_indexes[t]`). Used to mask cross-boundary
        # spurious secants for positions that have just become active.
        mem_count_init = jnp.zeros((T,), dtype=jnp.int32)

        @jax.jit
        def _step_win(carry, args):
            (states, samples_mem, residual_mem, mem_count,
             front, is_nan, iteration) = carry

            # ----- 1. window mask for THIS iter -----
            window_end = jnp.minimum(front + W, T)
            window_mask_b = (s_idx_full >= front) & (s_idx_full < window_end)
            window_mask = window_mask_b.astype(states.dtype)

            # ----- 2. eps eval ONLY on window positions -----
            # x_window[i] = X[front + i] for i = 0..W-1, where X is
            # (initial_state, states[0], ..., states[T-1]).
            X = jnp.concatenate([initial_state[None], states], axis=0)  # (T+1, D)
            front_i = front.astype(jnp.int32)
            zero_i = jnp.int32(0)
            x_window = jax.lax.dynamic_slice(X, (front_i, zero_i), (W, D_dim))
            t_window = jax.lax.dynamic_slice(t_per_s, (front_i,), (W,))
            eps_window = vmap(eps_model_local)(x_window, t_window)      # (W, D)
            # scatter into a (T, D) eps_arr at start index `front`. Entries
            # outside [front, front + W) are zero; the K-band j_mask below
            # zeros out their contribution to new_pred anyway.
            eps_arr = jax.lax.dynamic_update_slice(
                jnp.zeros((T, D_dim), dtype=eps_window.dtype),
                eps_window,
                (front_i, zero_i),
            )

            # ----- 3. front-clipped K-step lookahead -----
            # ref_idx_clip[s] = max(s + 1 - K, front)  (anchor inside window
            # boundary). For frozen-region s < front the math is dead since
            # window_mask[s] = 0, but we still keep the formula well-defined.
            ref_idx_clip = jnp.maximum(s_idx + 1 - K, front)           # (T,)
            ref_coef_clip = Pq[s_idx + 1] / Pq[ref_idx_clip]           # (T,)
            ref_states = X[ref_idx_clip]                                # (T, D)

            # Mask the j-axis of B_mat / N_mat so the K-band sum over
            # eps[j] / z[j] cannot reach back into the frozen region
            # (matches ParaTAA's window-restricted compute_ho_residual).
            j_mask = (s_idx >= front).astype(B_mat.dtype)              # (T,)
            B_eff = B_mat * j_mask[None, :]
            N_eff = N_mat * j_mask[None, :]

            new_pred = (ref_coef_clip[:, None] * ref_states
                        + B_eff @ eps_arr
                        + N_eff @ z_per_s)                              # (T, D)
            r_old = (new_pred - states) * window_mask[:, None]

            # ----- single-step residual for freeze criterion -----
            # ParaTAA (parallel_sd.py:172-176) checks single-step DDIM
            # residuals for the freeze decision, not the K-step lookahead.
            r_single = (a_per_s[:, None] * X[:-1]
                        + b_per_s[:, None] * eps_arr
                        + n_per_s[:, None] * z_per_s
                        - states) * window_mask[:, None]

            # ----- 4. push memory + per-position memory count -----
            samples_mem  = jnp.concatenate(
                [samples_mem[1:],  states[None]], axis=0
            )
            residual_mem = jnp.concatenate(
                [residual_mem[1:], r_old[None]],  axis=0
            )
            mem_count_new = mem_count + window_mask_b.astype(mem_count.dtype)

            if m > 1:
                dS = samples_mem[1:]  - samples_mem[:-1]                # (m-1, T, D)
                dR = residual_mem[1:] - residual_mem[:-1]               # (m-1, T, D)

                gram_per_t = jnp.einsum("itd,ptd->ipt", dR, dR)         # (m-1, m-1, T)
                d_per_t    = jnp.einsum("itd,td->it",  dR, r_old)       # (m-1, T)

                gram_cum = jnp.flip(
                    jnp.cumsum(jnp.flip(gram_per_t, axis=2), axis=2), axis=2
                )
                d_cum = jnp.flip(
                    jnp.cumsum(jnp.flip(d_per_t, axis=1), axis=1), axis=1
                )

                # Per-position valid_mask (matches ParaTAA's `memory_indexes`
                # masking, but we keep the LAST n_valid secants instead of
                # the FIRST -- ParaTAA's [:use_memory_max] slicing reaches
                # into the stale-zero pre-window entries when m > 2 and the
                # position has just become active. For m = 2 (their default
                # and our toy comparison setting) the two are identical.
                n_valid_diffs_per_t = jnp.clip(
                    mem_count_new - 1, 0, m - 1
                )                                                      # (T,)
                i_idx = jnp.arange(m - 1)
                valid_mask = (
                    i_idx[:, None]
                    >= ((m - 1) - n_valid_diffs_per_t[None, :])
                ).astype(gram_cum.dtype)                               # (m-1, T)

                gram_cum = (gram_cum
                            * valid_mask[:, None, :]
                            * valid_mask[None, :, :])
                d_cum = d_cum * valid_mask

                # Float64 Gram solve (matches ParaTAA's torch.float64
                # accumulator). Honored when JAX is run with X64 enabled;
                # falls back silently to the native dtype otherwise.
                solve_dtype = (jnp.float64
                               if jax.config.read("jax_enable_x64")
                               else gram_cum.dtype)
                eye = jnp.eye(m - 1, dtype=solve_dtype)
                gram_reg = (gram_cum.astype(solve_dtype)
                            + lmd * eye[:, :, None])
                gram_t = jnp.transpose(gram_reg, (2, 0, 1))
                d_t    = jnp.transpose(d_cum.astype(solve_dtype), (1, 0))
                c = vmap(jnp.linalg.solve)(gram_t, d_t).astype(states.dtype)

                A_blocks = dS + dR
                Gf = jnp.einsum("tj,jtd->td", c, A_blocks)
                # frozen / deferred positions must not be touched
                Gf = Gf * window_mask[:, None]
            else:
                Gf = jnp.zeros_like(states)

            new_states = states + r_old - Gf

            is_nan = jnp.logical_or(jnp.isnan(new_states).any(), is_nan)
            nan_mask = jnp.isnan(new_states)
            if reset:
                new_states = jnp.where(nan_mask, states_guess, new_states)

            # ----- 5. ParaTAA's residual-based freeze criterion -----
            # Per-position residual RMS norm over the D-dim coordinate axis
            # (matches ParaTAA's sqrt(mean(residual**2, dim=(1,2,3)))).
            # Uses single-step residual, not K-step (parallel_sd.py:175).
            res_norm = jnp.sqrt(jnp.mean(r_single * r_single, axis=-1))  # (T,)
            # Position is "unfinished" iff
            #     res > tol1   AND   res > tol2 * noise_thr[t]
            # so it is "settled" iff res <= tol1 OR res <= tol2*noise_thr[t].
            unfinished = (res_norm > wtol1) & (
                res_norm > wtol2 * noise_thr_per_s
            )
            settled_b = ~unfinished                                    # (T,)
            rel = jnp.arange(W, dtype=jnp.int32)
            abs_pos = front_i + rel
            abs_clipped = jnp.clip(abs_pos, jnp.int32(0), jnp.int32(T - 1))
            in_bounds = abs_pos < jnp.int32(T)
            settled = settled_b[abs_clipped] & in_bounds
            prefix = jnp.cumprod(settled.astype(jnp.int32))
            front_new = front_i + jnp.sum(prefix).astype(jnp.int32)

            # ----- 6. next-frame warm-start for newly-active positions -----
            # When the front advances by `delta = front_new - front`, the
            # `delta` positions [front + W, front_new + W) are about to
            # enter the window for the first time. ParaTAA's
            # initialize_samples copies _samples[uninitialized_index + 1]
            # (the just-settled adjacent position) into them; the analogue
            # in our s-indexing is the back of the OLD window,
            # new_states[front + W - 1].
            src_idx = jnp.minimum(front_i + jnp.int32(W - 1),
                                   jnp.int32(T - 1))
            src = new_states[src_idx]
            newly_active = (
                (s_idx_full >= front_i + jnp.int32(W))
                & (s_idx_full < front_new + jnp.int32(W))
            )
            new_states = jnp.where(
                newly_active[:, None], src[None, :], new_states
            )

            # Merit reported on the PRE-update states for consistency with
            # the no-window _step trace (one-iter lag in both paths).
            mf_val = merit_fxn(f_local, initial_state, states, drivers)
            return (new_states, samples_mem, residual_mem, mem_count_new,
                    front_new, is_nan, iteration + 1), \
                   (new_states, nan_mask, mf_val)

        if full_trace:
            init_carry_win = (
                states_guess, samples_mem_init, residual_mem_init,
                mem_count_init, front_init, False, 0,
            )
            last_output, all_outputs = scan(
                _step_win, init_carry_win, None, length=num_iters
            )
            final_state = last_output[0]
            is_nan = last_output[5]
            all_states, nan_mask, mf_val = all_outputs
            newton_steps = None
            all_states = jnp.concatenate([states_guess[None, ...], all_states])
            all_newtons = None
        else:
            def cond_func_win(state):
                iter_idx, _, _, _, _, front, err, _ = state
                # keep iterating while: budget not exhausted AND front
                # still inside the trajectory AND global merit is above
                # tolerance. Either of the latter two clearing the bar
                # signals convergence.
                budget = iter_idx < num_iters
                front_open = front < T
                err_open = err > tol
                return jnp.logical_and(budget,
                                       jnp.logical_and(front_open, err_open))

            def body_func_win(state):
                (iter_idx, states, sm, rm, mc, front, _, is_nan_) = state
                step_out, _ = _step_win(
                    (states, sm, rm, mc, front, is_nan_, iter_idx), None
                )
                (new_states, sm_new, rm_new, mc_new,
                 front_new, is_nan_new, _) = step_out
                new_err = merit_fxn(f_local, initial_state, new_states, drivers)
                return (iter_idx + 1, new_states, sm_new, rm_new, mc_new,
                        front_new, new_err, is_nan_new)

            init_loop = (
                0, states_guess, samples_mem_init, residual_mem_init,
                mem_count_init, front_init,
                merit_fxn(f_local, initial_state, states_guess, drivers),
                False,
            )
            result = jax.lax.while_loop(cond_func_win, body_func_win, init_loop)
            newton_steps = result[0]
            final_state = result[1]
            is_nan = result[7]
            all_states, nan_mask, mf_val = None, None, None
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

    # ===== existing code path (no windowing) =====
    @jax.jit
    def _step(carry, args):
        states, samples_mem, residual_mem, is_nan, iteration = carry

        # full trajectory including initial_state at index 0
        X = jnp.concatenate([initial_state[None], states], axis=0)  # (T+1, D)

        # eps(X[s], T - s) for s = 0..T-1 -- one eval per position
        def eval_eps_at(x_in, t_in):
            return eps_model_local(x_in, t_in)
        eps_arr = vmap(eval_eps_at)(X[:T], t_per_s)                 # (T, D)

        # K-step lookahead prediction of X[s + 1]:
        # new_pred[s] = ref_coef[s] * X[ref_idx[s]]
        #             + sum_j B_mat[s, j] * eps_arr[j]
        #             + sum_j N_mat[s, j] * z_per_s[j]
        ref_states = X[ref_idx]                                     # (T, D)
        new_pred = (ref_coef[:, None] * ref_states
                    + B_mat @ eps_arr
                    + N_mat @ z_per_s)                              # (T, D)

        # Picard direction higher-order residual: F_K(x) - x
        r_old = new_pred - states                                   # (T, D)

        # --- Anderson kernel (mirrors taa_alg) ---
        samples_mem  = jnp.concatenate(
            [samples_mem[1:],  states[None]], axis=0
        )
        residual_mem = jnp.concatenate(
            [residual_mem[1:], r_old[None]],  axis=0
        )

        if m > 1:
            dS = samples_mem[1:]  - samples_mem[:-1]                # (m-1, T, D)
            dR = residual_mem[1:] - residual_mem[:-1]               # (m-1, T, D)

            # Optionally accumulate the Gram system and solve it in a wider
            # dtype (e.g. float64, as ParaTAA does); None keeps states' dtype.
            dR_g    = dR    if gram_dtype is None else dR.astype(gram_dtype)
            r_old_g = r_old if gram_dtype is None else r_old.astype(gram_dtype)
            gram_per_t = jnp.einsum("itd,ptd->ipt", dR_g, dR_g)     # (m-1, m-1, T)
            d_per_t    = jnp.einsum("itd,td->it",  dR_g, r_old_g)   # (m-1, T)

            gram_cum = jnp.flip(
                jnp.cumsum(jnp.flip(gram_per_t, axis=2), axis=2), axis=2
            )
            d_cum = jnp.flip(
                jnp.cumsum(jnp.flip(d_per_t, axis=1), axis=1), axis=1
            )

            n_valid_diffs = jnp.minimum(iteration, m - 1)
            valid_mask = (
                jnp.arange(m - 1) >= ((m - 1) - n_valid_diffs)
            ).astype(gram_cum.dtype)

            gram_cum = (gram_cum
                        * valid_mask[None, :, None]
                        * valid_mask[:, None, None])
            d_cum = d_cum * valid_mask[:, None]

            eye = jnp.eye(m - 1, dtype=gram_cum.dtype)
            gram_reg = gram_cum + lmd * eye[:, :, None]             # (m-1, m-1, T)

            gram_t = jnp.transpose(gram_reg, (2, 0, 1))             # (T, m-1, m-1)
            d_t    = jnp.transpose(d_cum,    (1, 0))                # (T, m-1)
            c = vmap(jnp.linalg.solve)(gram_t, d_t)                 # (T, m-1)
            c = c.astype(states.dtype)

            A_blocks = dS + dR                                      # (m-1, T, D)
            Gf = jnp.einsum("tj,jtd->td", c, A_blocks)              # (T, D)
        else:
            Gf = jnp.zeros_like(states)

        new_states = states + r_old - Gf

        is_nan = jnp.logical_or(jnp.isnan(new_states).any(), is_nan)
        nan_mask = jnp.isnan(new_states)
        if reset:
            new_states = jnp.where(nan_mask, states_guess, new_states)

        # Convergence is judged on the canonical single-step fixed point,
        # using the eps evaluation already made above (no extra eps vmap).
        f_vals = (a_per_s[:, None] * X[:T]
                  + b_per_s[:, None] * eps_arr
                  + n_per_s[:, None] * z_per_s)
        mf_val = merit_from_residual(states - f_vals)
        return (new_states, samples_mem, residual_mem, is_nan, iteration + 1), \
               (new_states, nan_mask, mf_val)

    # --- convergence loop (identical layout to taa_alg) ---
    def cond_func(iter_inp):
        iter_idx, _, _, _, err, *_ = iter_inp
        return jnp.logical_and(iter_idx < num_iters, err > tol)

    def body_func_single(iter_inp):
        iter_idx, states, sm, rm, _, is_nan_, _ = iter_inp
        step_output, step_aux = _step((states, sm, rm, is_nan_, iter_idx), None)
        new_states, sm_new, rm_new, is_nan, _ = step_output
        _, _, mf_val = step_aux
        return iter_idx + 1, new_states, sm_new, rm_new, mf_val, is_nan, None

    def body_func_multiple(iter_inp):
        iter_idx, states, sm, rm, _, is_nan_, iters_below = iter_inp
        step_output, _ = _step((states, sm, rm, is_nan_, iter_idx), None)
        new_states, sm_new, rm_new, is_nan, _ = step_output
        new_errs = merit_fxn(f_local, initial_state, states, drivers, Ts=Ts)
        iters_below = iters_below + (new_errs < tol).astype(int)
        return iter_idx + 1, new_states, sm_new, rm_new, new_errs[-1], is_nan, iters_below

    init_carry = (states_guess, samples_mem_init, residual_mem_init, False, 0)

    if full_trace:
        last_output, all_outputs = scan(
            _step, init_carry, None, length=num_iters
        )
        final_state = last_output[0]
        is_nan = last_output[3]
        all_states, nan_mask, mf_val = all_outputs
        newton_steps, iters_below = None, None
        all_states = jnp.concatenate([states_guess[None, ...], all_states])
    elif Ts is not None:
        result = jax.lax.while_loop(
            cond_func,
            body_func_multiple,
            (0, states_guess, samples_mem_init, residual_mem_init,
             merit_fxn(f_local, initial_state, states_guess, drivers),
             False, jnp.zeros_like(Ts)),
        )
        newton_steps, final_state = result[0], result[1]
        is_nan, iters_below = result[5], result[6]
        all_states, nan_mask, mf_val = None, None, None
    else:
        result = jax.lax.while_loop(
            cond_func,
            body_func_single,
            (0, states_guess, samples_mem_init, residual_mem_init,
             merit_fxn(f_local, initial_state, states_guess, drivers),
             False, None),
        )
        newton_steps, final_state = result[0], result[1]
        is_nan, iters_below = result[5], result[6]
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
