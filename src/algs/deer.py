"""
Fixed-point solvers for linear dynamical systems using JAX.

This module implements Picard iteration and DEER (Differential Equation as
fixed-point itERation), a Newton-style fixed-point solver obtained by
linearizing the recurrence and evaluating the resulting LDS with a parallel scan.

Both methods solve fixed-point equations of the form x = f(x) by iteratively
updating the entire trajectory in parallel.
"""

from typing import Callable, Optional, Tuple, Union
import jax
from jax import vmap
from jax.lax import scan
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg as jsl

from functools import partial


@jax.vmap
def full_mat_operator(q_i: Tuple[jnp.ndarray, jnp.ndarray],
                     q_j: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Associative binary operator for parallel scan over full-matrix LDS transitions.

    For an LDS step:
        x_k = A_k x_{k-1} + b_k
    define the affine map:
        f_k(x) = A_k x + b_k.

    Composition of affine maps satisfies:
        f_j o f_i (x) = A_j A_i x + (A_j b_i + b_j)

    Thus the scan operator is:
        (A_i, b_i) \\otimes (A_j, b_j) = (A_j @ A_i, A_j @ b_i + b_j)

    Args:
        q_i: (A_i, b_i) for step i, with shapes (P, P) and (P,)
        q_j: (A_j, b_j) for step j, with shapes (P, P) and (P,)

    Returns:
        (A_out, b_out): combined affine parameters.
    """

    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j @ A_i, A_j @ b_i + b_j


@jax.vmap
def diag_mat_operator(q_i: Tuple[jnp.ndarray, jnp.ndarray],
                      q_j: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Associative binary operator for parallel scan over diagonal LDS transitions.

    For diagonal dynamics A_i = diag(a_i), affine maps take the form:
        f_i(x) = A_i x + b_i
    and composition reduces to element-wise operations:
        f_j \\circ f_i (x) = diag(a_j * a_i) x + (a_j * b_i + b_j)

    Thus the scan operator becomes:
        (a_i, b_i) \\otimes (a_j, b_j) = (a_j * a_i, a_j * b_i + b_j)

    Args:
        q_i: (a_i, b_i) for step i, where a_i and b_i have shape (P,)
        q_j: (a_j, b_j) for step j, same shapes as above

    Returns:
        (a_out, b_out): element-wise combined diagonal parameters.
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j

@jax.vmap
def add_operator(q_i: jnp.ndarray, q_j: jnp.ndarray) -> jnp.ndarray:
    """
    Associative operator for Picard iteration (identity-Jacobian case).

    In Picard iteration, the linearization uses A_t = I for all t, so each
    update contributes only a residual increment. The LDS composition therefore
    collapses to simple element-wise addition:

        q_i + q_j = q_i + q_j

    This operator is associative and suitable for use with
    `jax.lax.associative_scan`.

    Args:
        q_i: First operand, shape (...).
        q_j: Second operand, same shape.

    Returns:
        Element-wise sum q_i + q_j.
    """
    return q_i + q_j


def get_residual(f: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                 initial_state: jnp.ndarray,
                 states: jnp.ndarray,
                 drivers: jnp.ndarray) -> jnp.ndarray:
    """
    Compute residuals r_t = x_t - f(x_{t-1}, u_t) for a proposed trajectory.

    states[t]  = x_{t+1}   for t = 0..T-1
    drivers[t] = u_{t+1}   for t = 0..T-1

    The first step uses the provided initial state:
        r_0 = x_1 - f(x_0, u_1)

    For subsequent t >= 1:
        r_t = x_{t+1} - f(x_t, u_{t+1})

    Args:
        f: Function mapping (state, driver) -> next state.
        initial_state: x_0, shape (D,).
        states: Proposed trajectory {x_1, ..., x_T} of shape (T, D).
        drivers: Inputs {u_1, ..., u_T} of shape (T, ...).

    Returns:
        Residuals r = {r_0, ..., r_{T-1}} of shape (T, D).
    """
    # Evaluate f(x_t, u_{t+1}) for t = 1..T-1
    fs = vmap(f)(states[:-1], drivers[1:])  # (T-1, D)
    # prepend f(x_0, u_1)
    f0 = f(initial_state, drivers[0])[None, :]     # (1, D)
    fs = jnp.concatenate([f0, fs], axis=0)         # (T, D)
    # compute residuals
    r = states - fs                                # (T, D)
    return r


def merit_fxn(f: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
              initial_state: jnp.ndarray,
              states: jnp.ndarray,
              drivers: jnp.ndarray,
              Ts: Optional[jnp.ndarray] = None) -> Union[float, jnp.ndarray]:
    """Compute merit function for convergence assessment.

    Mathematical formulation:
    Merit function: L = (1/2) sum_{t=1}^T sum_{s=1}^t ||r_s||^2
    where r_s = x_s - f(x_{s-1}, u_s) is the residual at time s.

    This cumulative squared residual provides a measure of how well
    the current trajectory satisfies the fixed-point constraints.
    Convergence is achieved when L -> 0.

    Args:
        f: Forward dynamics function f(x_{t-1}, u_t) -> x_t
        initial_state: Initial condition x_0 of shape (D,)
        states: Current trajectory estimate {x_t}_{t=1}^T of shape (T, D)
        drivers: Input sequence {u_t}_{t=1}^T of shape (T, N_noise)
        Ts: Optional array of time indices to evaluate merit at

    Returns:
        Merit function value(s): scalar if Ts is None, array if Ts provided
    """
    r = get_residual(f, initial_state, states, drivers)  # (T,D)
    T, D = r.shape
    Ls = 0.5 * jnp.sum(
        jnp.cumsum(r**2, axis=0), axis=1
    )  # (T,)
    Ls = jnp.where(jnp.isnan(Ls), jnp.inf, Ls)
    if Ts is not None:
        return Ls[Ts - 1]
    else:
        return Ls[-1]


def quasi_diag_estimator(state, inputs, deer_jvp, key, num_samples=1):
    z_rad = jr.rademacher(key, (num_samples, state.shape[0])).astype(float)
    vmap_jvp = jax.vmap(deer_jvp, in_axes=(None, None, 0))
    jac_diag = jnp.mean(z_rad * vmap_jvp(state, inputs, z_rad), axis=0)
    return jac_diag


def compute_sequence(scan_fxn, initial_state, inputs):
    """Helper for gauss_seidel_alg: runs scan over a chunk."""
    _, states = jax.lax.scan(scan_fxn, initial_state, inputs)
    return states[-1]


def gauss_seidel_alg(
    f,
    initial_state,
    states_guess,
    drivers,
    num_iters,
    full_trace=False,
    reset=True,
    tol=5e-4,
    chunk_size=1,
):
    """
    GS iteration.
    Chunk size controls how many timesteps per chunk.
    """

    def _scan_fxn(carry, arg):
        out = f(carry, arg)
        return out, out

    @jax.jit
    def _step(carry, args):
        states, is_nan = carry
        T = len(states_guess)
        inputs_tiled = jnp.tile(drivers[None, ...], (T - chunk_size, 1, 1))
        shifted = jax.vmap(partial(jnp.roll, axis=0), in_axes=(0, 0))(
            inputs_tiled, -(1 + jnp.arange(T - chunk_size))
        )[:, :chunk_size, :]
        new_states = states
        new_states = new_states.at[chunk_size:].set(
            jax.vmap(compute_sequence, in_axes=(None, 0, 0))(
                _scan_fxn, states[:-chunk_size], shifted
            )
        )
        is_nan = jnp.logical_or(jnp.isnan(new_states).any(), is_nan)
        nan_mask = jnp.isnan(new_states)
        if reset:
            new_states = jnp.where(nan_mask, states_guess, new_states)
        mf_val = merit_fxn(f, initial_state, states, drivers)
        return (new_states, is_nan), (new_states, nan_mask, mf_val)

    def body_func_single(iter_inp):
        iter_idx, states, _, is_nan_, _ = iter_inp
        step_output, _ = _step((states, is_nan_), None)
        new_states, is_nan = step_output
        new_err = merit_fxn(f, initial_state, states, drivers)
        return iter_idx + 1, new_states, new_err, is_nan, None

    def cond_func(iter_inp):
        iter_idx, _, err, *_ = iter_inp
        return jnp.logical_and(iter_idx < num_iters, err > tol)

    _, starter_states = jax.lax.scan(_scan_fxn, initial_state, drivers[:chunk_size])
    states_guess = states_guess.at[:chunk_size].set(starter_states)
    if full_trace:
        last_output, all_outputs = scan(
            _step, (states_guess, False), None, length=num_iters
        )
        final_state, is_nan = last_output
        all_states, nan_mask, mf_val = all_outputs
        newton_steps, iters_below = None, None
        all_states = jnp.concatenate([states_guess[None, ...], all_states])
    else:
        newton_steps, final_state, _, is_nan, iters_below = jax.lax.while_loop(
            cond_func,
            body_func_single,
            (
                0,
                states_guess,
                merit_fxn(f, initial_state, states_guess, drivers),
                False,
                None,
            ),
        )
        all_states = None
        mf_val = None
    return (all_states, final_state, newton_steps, is_nan, None, None, mf_val)


def jacobi_alg(
    f,
    initial_state,
    states_guess,
    drivers,
    num_iters,
    full_trace=False,
    reset=True,
    tol=5e-4,
):
    """
    Jacobi iteration.
    """

    @jax.jit
    def _step(carry, args):
        states, is_nan = carry
        fs = vmap(f)(states[:-1], drivers[1:])  # (T-1, D)
        b0 = f(initial_state, drivers[0])  # h1=f(h0, e1)
        new_states = jnp.concatenate([b0[jnp.newaxis, :], fs])  # (T, D)
        is_nan = jnp.logical_or(jnp.isnan(new_states).any(), is_nan)
        nan_mask = jnp.isnan(new_states)
        if reset:
            new_states = jnp.where(nan_mask, states_guess, new_states)
        mf_val = merit_fxn(f, initial_state, states, drivers)
        return (new_states, is_nan), (new_states, nan_mask, mf_val)

    def body_func_single(iter_inp):
        iter_idx, states, _, is_nan_, _ = iter_inp
        step_output, _ = _step((states, is_nan_), None)
        new_states, is_nan = step_output
        new_err = merit_fxn(f, initial_state, states, drivers)
        return iter_idx + 1, new_states, new_err, is_nan, None

    def cond_func(iter_inp):
        iter_idx, _, err, *_ = iter_inp
        return jnp.logical_and(iter_idx < num_iters, err > tol)

    if full_trace:
        last_output, all_outputs = scan(
            _step, (states_guess, False), None, length=num_iters
        )
        final_state, is_nan = last_output
        all_states, nan_mask, mf_val = all_outputs
        newton_steps, iters_below = None, None
        all_states = jnp.concatenate([states_guess[None, ...], all_states])
    else:
        newton_steps, final_state, _, is_nan, iters_below = jax.lax.while_loop(
            cond_func,
            body_func_single,
            (
                0,
                states_guess,
                merit_fxn(f, initial_state, states_guess, drivers),
                False,
                None,
            ),
        )
        all_states = None
        mf_val = None
    return (all_states, final_state, newton_steps, is_nan, None, None, mf_val)


def guo_alg(
    f,
    initial_state,
    states_guess,
    drivers,
    num_iters,
    full_trace=False,
    reset=True,
    tol=5e-4,
):
    """
    Guo iteration: Jacobi + JVP correction.
    """

    @jax.jit
    def _step(carry, args):
        states, is_nan = carry
        fs = vmap(f)(states[:-1], drivers[1:])  # (T-1, D)
        b0 = f(initial_state, drivers[0])  # h1=f(h0, e1)
        new_states = jnp.concatenate([b0[jnp.newaxis, :], fs])  # (T, D)
        errs = new_states - states # (T,D)
        _, guo_corrections = vmap(jax.jvp, in_axes=(None, 0, 0))(
            f, (states[:-1], drivers[1:]), (errs[:-1], jnp.zeros_like(drivers[1:]))
        ) # guo corrections are (T-1, D)
        new_states = new_states.at[1:].add(guo_corrections)
        is_nan = jnp.logical_or(jnp.isnan(new_states).any(), is_nan)
        nan_mask = jnp.isnan(new_states)
        if reset:
            new_states = jnp.where(nan_mask, states_guess, new_states)
        mf_val = merit_fxn(f, initial_state, states, drivers)
        return (new_states, is_nan), (new_states, nan_mask, mf_val)

    def body_func_single(iter_inp):
        iter_idx, states, _, is_nan_, _ = iter_inp
        step_output, _ = _step((states, is_nan_), None)
        new_states, is_nan = step_output
        new_err = merit_fxn(f, initial_state, states, drivers)
        return iter_idx + 1, new_states, new_err, is_nan, None

    def cond_func(iter_inp):
        iter_idx, _, err, *_ = iter_inp
        return jnp.logical_and(iter_idx < num_iters, err > tol)

    if full_trace:
        last_output, all_outputs = scan(
            _step, (states_guess, False), None, length=num_iters
        )
        final_state, is_nan = last_output
        all_states, nan_mask, mf_val = all_outputs
        newton_steps, iters_below = None, None
        all_states = jnp.concatenate([states_guess[None, ...], all_states])
    else:
        newton_steps, final_state, _, is_nan, iters_below = jax.lax.while_loop(
            cond_func,
            body_func_single,
            (
                0,
                states_guess,
                merit_fxn(f, initial_state, states_guess, drivers),
                False,
                None,
            ),
        )
        all_states = None
        mf_val = None
    return (all_states, final_state, newton_steps, is_nan, None, None, mf_val)


def picard_alg(f,
               initial_state,
               states_guess,
               drivers,
               num_iters,
               full_trace=False,
               reset=True,
               tol=5e-4):
    """
    Picard iteration.
    """
    @jax.jit
    def _step(carry, args):
        states, is_nan = carry
        fs = vmap(f)(states[:-1], drivers[1:])    # (T-1, D)
        bs = fs - states[:-1]
        b0 = f(initial_state, drivers[0])  # h1=f(h0, e1)
        b = jnp.concatenate([b0[jnp.newaxis, :], bs])  # (T, D)
        binary_op = add_operator
        new_states = jax.lax.associative_scan(
            binary_op, b
        )
        is_nan = jnp.logical_or(jnp.isnan(new_states).any(), is_nan)
        nan_mask = jnp.isnan(new_states)
        if reset:
            new_states = jnp.where(nan_mask, states_guess, new_states)
        mf_val = merit_fxn(f, initial_state, states, drivers)
        return (new_states, is_nan), (new_states, nan_mask, mf_val)

    def body_func_single(iter_inp):
        iter_idx, states, _, is_nan_, _ = iter_inp
        step_output, _ = _step((states, is_nan_), None)
        new_states, is_nan = step_output
        new_err = merit_fxn(f, initial_state, states, drivers)
        return iter_idx + 1, new_states, new_err, is_nan, None

    def cond_func(iter_inp):
        iter_idx, _, err, *_ = iter_inp
        return jnp.logical_and(iter_idx < num_iters, err > tol)

    if full_trace:
        last_output, all_outputs = scan(
            _step, (states_guess, False), None, length=num_iters
        )
        final_state, is_nan = last_output
        all_states, nan_mask, mf_val = all_outputs
        newton_steps, iters_below = None, None
        all_states = jnp.concatenate([states_guess[None, ...], all_states])
    else:
        newton_steps, final_state, _, is_nan, iters_below = jax.lax.while_loop(
            cond_func,
            body_func_single,
            (
                0,
                states_guess,
                merit_fxn(f, initial_state, states_guess, drivers),
                False,
                None,
            ),
        )
        all_states = None
        mf_val = None
    return (all_states, final_state, newton_steps, is_nan, None, None, mf_val)


def deer_alg(
    f: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    initial_state: jnp.ndarray,
    states_guess: jnp.ndarray,
    drivers: jnp.ndarray,
    num_iters: int,
    quasi: bool = False,
    diagonal_func: Optional[Callable] = None,
    qmem_efficient: bool = False,
    k: Optional[float] = 0.0,
    clip: bool = False,
    full_trace: bool = False,
    Ts: Optional[jnp.ndarray] = None,
    reset: bool = False,
    tol: Optional[float] = 5e-4,
    key: Optional[jnp.ndarray] = None,
):
    """
    DEER iteration: fixed-point solver for trajectories x_{1:T}.

    Each step linearizes the dynamics:
        x_{t+1} ~ f(x_t, u_{t+1}) + J_f(x_t,u_{t+1}) (x_t - x_t^old)

    This defines an LDS (A_t, b_t) solved in parallel via associative_scan.
    """

    T, D = states_guess.shape

    if key is None:
        key = jr.PRNGKey(0)
    # ------------------------------------------------------------
    # Helper: compute diagonal of Jacobian using Hutchinson
    # ------------------------------------------------------------
    if qmem_efficient:
        def deer_jvp(state, inp, z):
            return jax.jvp(lambda x: f(x, inp), (state,), (z,))[1]
        keys = jr.split(key, T - 1)

    @jax.jit
    def _step(carry, args):
        states, is_nan = carry
        fs = vmap(f)(
            states[:-1], drivers[1:]
        )
        nan_jac_idx = False
        if quasi:
            if qmem_efficient:
                As = vmap(quasi_diag_estimator, in_axes=(0, 0, None, 0))(states[:-1], drivers[1:], deer_jvp, keys)
            elif diagonal_func:
                As = vmap(diagonal_func)(states[:-1], drivers[1:])
            else:
                Jfs = vmap(jax.jacrev(f, argnums=0))(states[:-1], drivers[1:])
                As = vmap(lambda Jf: jnp.diag(Jf))(Jfs)
            As = (1 - k) * As  # damping
            if clip:
                As = jnp.clip(As, -1, 1)
            bs = fs - As * states[:-1]
        else:
            Jfs = vmap(jax.jacrev(f, argnums=0))(states[:-1], drivers[1:])  # (T, D, D)
            if reset:
                nan_mask_jacobians = jnp.isnan(Jfs).any(
                    axis=(1, 2), keepdims=True
                )
                nan_jac_idx = jnp.isnan(Jfs).any()
                I = jnp.eye(Jfs.shape[1])
                Jfs = jnp.where(nan_mask_jacobians, I, Jfs)
            As = Jfs
            As = (1 - k) * As  # damping
            bs = fs - jnp.einsum("tij,tj->ti", As, states[:-1])

        b0 = f(initial_state, drivers[0])  # h1=f(h0, e1)
        A0 = jnp.zeros_like(As[0])
        A = jnp.concatenate(
            [A0[jnp.newaxis, :], As]
        )
        b = jnp.concatenate([b0[jnp.newaxis, :], bs])  # (T, D)
        if quasi:
            binary_op = diag_mat_operator
        else:
            binary_op = full_mat_operator
        _, new_states = jax.lax.associative_scan(
            binary_op, (A, b)
        )
        is_nan = jnp.logical_or(
            jnp.logical_or(jnp.isnan(new_states).any(), is_nan), nan_jac_idx
        )
        nan_mask = jnp.isnan(new_states)
        if reset:
            new_states = jnp.where(nan_mask, states_guess, new_states)
        mf_val = merit_fxn(f, initial_state, states, drivers)
        return (new_states, is_nan), (new_states, nan_mask, mf_val)

    def cond_func(iter_inp):
        iter_idx, _, err, *_ = iter_inp
        return jnp.logical_and(iter_idx < num_iters, err > tol)

    def body_func_single(iter_inp):
        iter_idx, states, _, is_nan_, _ = iter_inp
        step_output, _ = _step((states, is_nan_), None)
        new_states, is_nan = step_output
        new_err = merit_fxn(f, initial_state, states, drivers)
        return iter_idx + 1, new_states, new_err, is_nan, None

    def body_func_multiple(iter_inp):
        iter_idx, states, _, is_nan_, iters_below = iter_inp
        step_output, _ = _step((states, is_nan_), None)
        new_states, is_nan = step_output
        new_errs = merit_fxn(f, initial_state, states, drivers, Ts=Ts)
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
        newton_steps, final_state, _, is_nan, iters_below = jax.lax.while_loop(
            cond_func,
            body_func_multiple,
            (
                0,
                states_guess,
                merit_fxn(f, initial_state, states_guess, drivers),
                False,
                jnp.zeros_like(Ts),
            ),
        )
        all_states, nan_mask, mf_val = None, None, None
    else:
        newton_steps, final_state, _, is_nan, iters_below = jax.lax.while_loop(
            cond_func,
            body_func_single,
            (
                0,
                states_guess,
                merit_fxn(f, initial_state, states_guess, drivers),
                False,
                None,
            ),
        )
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
