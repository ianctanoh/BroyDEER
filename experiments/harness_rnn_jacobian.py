"""
harness_rnn_jacobian.py
Jacobian recovery analysis for Broyden and Accelerated Broyden on low-rank RNN.

For each healthy chain, runs the solver with full_trace=True, then at each
iteration reconstructs the Broyden block approximation B_t and compares to
the true analytical Jacobian J_t of the low-rank RNN:

    J_t = a * I + (dt/tau) * M @ N^T @ diag(phi'(x_t))

Reports per-iteration mean relative Frobenius error across timesteps:
    err(iter) = mean_t( ||B_t + J_t||_F / ||J_t||_F )

(B_t approximates -J_t in the Newton system, so B_t + J_t is the error.)

Usage:
  python harness_rnn_jacobian.py alg=broyden N=100 T=1000
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger

from dynamics.low_rank_rnn import LowRankRNN
from algs.deer import get_residual
from algs.broyden_deer import (
    broyden_solve, diagonal_solve, sequential_broyden_solve,
    sequential_diagonal_solve, broyden_rank1_update, update_all_blocks,
    compute_gamma_rescale, lrmat_apply,
)
from algs.accelerated_broyden import (
    solve_rankk_bidiag, solve_rankk_bidiag_sequential,
)


def true_jacobian_batch(M, Nt, a, dt_over_tau, states):
    """Compute true Jacobian J_t at each state. Returns (T, N, N)."""
    def jac_at(x):
        phi_prime = 1.0 - jnp.tanh(x) ** 2
        return a * jnp.eye(x.shape[0]) + dt_over_tau * M @ (Nt.T * phi_prime[None, :])
    return vmap(jac_at)(states)


def block_to_dense(d_t, U_t, V_t):
    """Convert diag(d) + U V^T to dense (N, N) matrix."""
    return jnp.diag(d_t) + U_t @ V_t.T


def relative_spectral_error(B_t, neg_J_t):
    """||B_t - (-J_t)||_2 / ||J_t||_2  (operator 2-norm = largest singular value)"""
    err = jnp.linalg.norm(B_t - neg_J_t, 2)
    ref = jnp.linalg.norm(neg_J_t, 2)
    return jnp.where(ref > 1e-30, err / ref, 0.0)


def true_newton_step(f, initial_state, states, drivers):
    """Compute the exact Newton step by sequential forward substitution
    with true dense Jacobian blocks. O(T*N^2) work, O(N^2) memory."""
    T, D = states.shape
    r = get_residual(f, initial_state, states, drivers)  # (T, D)

    # First block: delta_0 = r_0 (no sub-diagonal)
    # For t >= 1: delta_t = r_t + J_t @ delta_{t-1}
    # where J_t = jacfwd(f)(states[t-1], drivers[t])
    def step(delta_prev, t_idx):
        J_t = jax.jacfwd(f)(states[t_idx - 1], drivers[t_idx])
        delta_t = r[t_idx] + J_t @ delta_prev
        return delta_t, delta_t

    delta_0 = r[0]
    _, deltas_rest = jax.lax.scan(step, delta_0, jnp.arange(1, T))
    return jnp.concatenate([delta_0[None], deltas_rest], axis=0)


@hydra.main(config_path="configs", config_name="harness_rnn_sweep", version_base=None)
def main(cfg: DictConfig):
    seed = cfg.seed
    N = cfg.N
    R = cfg.R
    T = cfg.T
    tau = cfg.tau
    dt = cfg.dt
    g = cfg.g
    sigma_noise = cfg.sigma_noise
    phi = cfg.phi
    num_iters = cfg.num_iters
    tol = cfg.tol
    batch_size = cfg.batch_size
    mode = cfg.mode
    alg = cfg.alg

    if alg not in ("broyden", "accelerated_broyden", "quasi"):
        print(f"Jacobian recovery only for broyden/accelerated_broyden/quasi, got {alg}. Skipping.")
        wandb.init(project="broydeer_rnn_jacobian", mode=mode)
        wandb.finish()
        return

    logger = WandbLogger(project="broydeer_rnn_jacobian", mode=mode)
    logger.log_hyperparams(OmegaConf.to_container(cfg))
    wandb.config.update(OmegaConf.to_container(cfg), allow_val_change=True)

    try:
        _run(cfg, logger, seed, N, R, T, tau, dt, g, sigma_noise, phi,
             num_iters, tol, batch_size, alg)
    except Exception as e:
        print(f"ERROR: {alg} | N={N}, T={T} failed: {e}")
        wandb.log({"error": str(e), "N": N, "T": T, "alg": alg})
    finally:
        wandb.finish()


def _run(cfg, logger, seed, N, R, T, tau, dt, g, sigma_noise, phi,
         num_iters, tol, batch_size, alg):

    a = 1.0 - dt / tau
    dt_over_tau = dt / tau
    max_attempts = batch_size * 10

    def setup_chain(chain_seed):
        chain_key = jr.PRNGKey(chain_seed)
        k_rnn, k_init, k_drv, k_guess = jr.split(chain_key, 4)
        rnn = LowRankRNN(N, R, key=k_rnn, tau=tau, dt=dt, g=g, phi=phi)
        initial_state = jr.normal(k_init, (N,)) * 0.1
        chain_drivers = sigma_noise * jr.normal(k_drv, (T, N))
        states_guess = jr.normal(k_guess, (T, N)) * 0.1
        return rnn.M, rnn.Nt, initial_state, chain_drivers, states_guess

    def _phi(x):
        if phi == "relu":
            return jnp.maximum(x, 0.0)
        else:
            return jnp.tanh(x)

    def make_deer_fxn(M, Nt):
        def f(state, inp):
            recurrent = M @ (Nt.T @ _phi(state))
            return a * state + dt_over_tau * recurrent + inp
        return f

    # --- saturation filter ---
    sat_threshold = 0.8
    dead_threshold = 0.1

    def seq_trajectory(M_c, Nt_c, init_c, drv_c):
        f = make_deer_fxn(M_c, Nt_c)
        def step(state, driver):
            new = f(state, driver)
            return new, new
        _, traj = jax.lax.scan(step, init_c, drv_c)
        return traj

    def check_saturation(traj):
        burn = T // 4
        phi_vals = jnp.tanh(traj[burn:])
        abs_mean = jnp.mean(jnp.abs(phi_vals), axis=0)
        phi_prime_mean = jnp.mean(1.0 - phi_vals ** 2, axis=0)
        n_sat = jnp.sum(abs_mean > sat_threshold)
        n_dead = jnp.sum(phi_prime_mean < dead_threshold)
        return n_sat, n_dead

    healthy_data = []
    total_tried = 0
    chain_idx = 0
    print(f"Collecting {batch_size} healthy chains...")
    while len(healthy_data) < batch_size and chain_idx < max_attempts:
        cd = setup_chain(seed + chain_idx)
        M_i, Nt_i, init_i, drv_i, _ = cd
        traj_i = seq_trajectory(M_i, Nt_i, init_i, drv_i)
        n_sat, n_dead = check_saturation(traj_i)
        total_tried += 1
        if int(n_sat) == 0 and int(n_dead) == 0:
            healthy_data.append(cd)
            print(f"  seed {seed + chain_idx}: healthy [{len(healthy_data)}/{batch_size}]")
        chain_idx += 1

    n_healthy = len(healthy_data)
    print(f"Saturation filter: {n_healthy}/{total_tried} healthy.")
    if n_healthy == 0:
        print("ERROR: no healthy chains.")
        return

    # --- Iteration with block tracking ---
    # We manually replicate the iteration logic, recording the
    # Jacobian approximation at each iteration for comparison.

    if alg == "quasi":
        m = 0
        k_damp = cfg.deer_k
        gamma_val = 1.0
        init_scheme = "autodiff"
        eps_gram = 0.0
    elif alg == "broyden":
        m = cfg.bd_m
        k_damp = cfg.bd_k
        gamma_val = cfg.bd_gamma
        init_scheme = cfg.bd_init
        eps_gram = cfg.acc_eps_gram
    else:
        m = cfg.acc_m
        k_damp = cfg.acc_k
        gamma_val = cfg.acc_gamma
        init_scheme = cfg.acc_init
        eps_gram = cfg.acc_eps_gram

    # --- Pre-run: identify converged seeds using actual solver (num_iters=1000) ---
    # This matches runtime experiment seeds exactly.
    from algs.broyden_deer import broyden_deer_alg
    from algs.accelerated_broyden import accelerated_broyden_deer_alg
    from algs.deer import deer_alg

    if alg == "quasi":
        # Quasi never converges at T>=10k, just use first 8 healthy seeds
        converged_data = healthy_data[:batch_size]
        n_analyze = len(converged_data)
        print(f"Quasi: using first {n_analyze} healthy seeds (no convergence check).")
    else:
        print(f"Pre-run: identifying converged seeds with num_iters={num_iters}...")
        converged_data = []
        for i, cd in enumerate(healthy_data):
            M_c, Nt_c, init_c, drv_c, sg_c = cd
            f_check = make_deer_fxn(M_c, Nt_c)

            if alg == "broyden":
                _, _, steps, *_ = broyden_deer_alg(
                    f_check, init_c, sg_c, drv_c, num_iters,
                    m=m, init=init_scheme, k=k_damp, gamma=gamma_val,
                    compress=cfg.acc_compress, full_trace=False, tol=tol,
                    line_search=True, ls_c=1e-3)
            else:  # accelerated_broyden
                _, _, steps, *_ = accelerated_broyden_deer_alg(
                    f_check, init_c, sg_c, drv_c, num_iters,
                    m=m, init=init_scheme, k=k_damp, gamma=gamma_val,
                    eps_gram=eps_gram, compress=cfg.acc_compress,
                    qmem_efficient=True, full_trace=False, tol=tol,
                    line_search=True, ls_c=1e-3)

            steps = int(steps)
            if steps < num_iters:
                converged_data.append(cd)
                print(f"  healthy chain {i}: CONVERGED in {steps} iters")
            else:
                print(f"  healthy chain {i}: did not converge ({steps} iters)")

            if len(converged_data) >= batch_size:
                break

        n_analyze = len(converged_data)
        print(f"Found {n_analyze} converged seeds for Jacobian analysis.")
        if n_analyze == 0:
            print("No converged seeds, nothing to analyze.")
            wandb.log({"n_converged_jac": 0, "N": N, "T": T, "alg": alg})
            return

    # Fixed random subset of timesteps for Jacobian comparison.
    # Same subset for all chains and all methods (seeded deterministically).
    n_jac_samples = min(100, T - 1)
    jac_key = jr.PRNGKey(999)
    jac_indices = jr.choice(jac_key, T - 1, shape=(n_jac_samples,), replace=False)
    jac_indices = jnp.sort(jac_indices)
    print(f"Jacobian comparison on {n_jac_samples}/{T-1} timesteps (fixed subset).")

    all_error_curves = []

    for chain_idx in range(n_analyze):
        M_c, Nt_c, init_c, drv_c, sg_c = converged_data[chain_idx]
        f = make_deer_fxn(M_c, Nt_c)

        # Initialize
        D = N
        use_lrmat = (m > 0)

        if init_scheme == "identity":
            d_init = -gamma_val * jnp.ones((T, D))
        elif init_scheme == "rescaled_identity":
            d_init = -jnp.ones((T, D))
        else:
            # "jac" or "autodiff": compute true diagonal
            jacs = vmap(jax.jacfwd(f))(sg_c[:-1], drv_c[1:])
            d_diag = -vmap(jnp.diag)(jacs)
            d_init = jnp.concatenate([d_diag, jnp.ones((1, D))], axis=0)

        U = jnp.zeros((T, D, max(m, 1)))
        V = jnp.zeros((T, D, max(m, 1)))
        gamma_arr = jnp.ones(T)
        states = sg_c.copy()

        # For accelerated broyden: sliding window of Newton steps
        p_window = jnp.zeros((T, m, D))
        target_window = jnp.zeros((T, m, D))

        error_per_iter = []

        max_jac_iters = min(num_iters, 30)
        for iteration in range(max_jac_iters):
            # Effective diagonal
            if alg == "accelerated_broyden" and init_scheme in ("identity", "rescaled_identity"):
                d_eff = -gamma_val * jnp.ones((T, D))
                if init_scheme == "rescaled_identity" and iteration > 0:
                    d_eff = gamma_arr[:, None] * jnp.ones((T, D))
            else:
                d_eff = gamma_arr[:, None] * d_init

            # Compute residual
            r_old = get_residual(f, init_c, states, drv_c)

            # Check convergence
            rms = float(jnp.sqrt(jnp.mean(r_old ** 2)))
            if rms < tol:
                print(f"  chain {chain_idx}: converged at iter {iteration}, rms={rms:.2e}")
                break

            if alg == "quasi":
                # Quasi-DEER: compute exact diagonal analytically (no jacrev, no OOM)
                # diag(J_t) = a + (dt/tau) * diag(M @ Nt^T) * phi'(x_t)
                phi_prime = 1.0 - jnp.tanh(states[:-1]) ** 2              # (T-1, N)
                diag_MNt = jnp.sum(M_c * Nt_c, axis=1)                    # (N,)
                d_diag_iter = -(1 - k_damp) * (a + dt_over_tau * diag_MNt[None, :] * phi_prime)
                d_quasi = jnp.concatenate([d_diag_iter, jnp.ones((1, D))], axis=0)

                # Solve diagonal bidiagonal system
                delta_s = diagonal_solve(d_quasi, r_old)
                new_states = states + delta_s

                # Store for Jacobian comparison (no U, V)
                d_eff = d_quasi
                U = jnp.zeros((T, D, 1))
                V = jnp.zeros((T, D, 1))

            elif alg == "broyden":
                # Broyden solve
                if use_lrmat:
                    delta_s = broyden_solve(d_eff, U, V, r_old, compress=cfg.acc_compress, iteration=iteration)
                else:
                    delta_s = diagonal_solve(d_eff, r_old)

                new_states = states + delta_s
                r_new = get_residual(f, init_c, new_states, drv_c)

                # Gamma rescale
                if init_scheme in ("rescaled_identity", "jac"):
                    gamma_arr = compute_gamma_rescale(delta_s, r_old, r_new, d=d_init)
                    d_eff_update = gamma_arr[:, None] * d_init
                else:
                    d_eff_update = d_eff

                # Update Broyden blocks
                if use_lrmat:
                    U, V = update_all_blocks(d_eff_update, U, V, delta_s, r_old, r_new, k_damp)

            else:  # accelerated_broyden
                # Build U, V from sliding window
                if iteration == 0:
                    U_blocks = jnp.zeros((T, D, max(m, 1)))
                    V_blocks = jnp.zeros((T, D, max(m, 1)))
                else:
                    p_blocks = p_window[:-1]
                    target_blocks = target_window[:-1]

                    def compute_UV(d_t, steps_t, targets_t):
                        V_t = steps_t.T
                        R_t = targets_t.T - d_t[:, None] * V_t
                        C_t = V_t.T @ V_t + eps_gram * jnp.eye(m)
                        U_t = (1 - k_damp) * jnp.linalg.solve(C_t, R_t.T).T
                        return U_t, V_t

                    U_inner, V_inner = vmap(compute_UV)(d_eff[:-1], p_blocks, target_blocks)
                    U_blocks = jnp.concatenate([U_inner, jnp.zeros((1, D, m))], axis=0)
                    V_blocks = jnp.concatenate([V_inner, jnp.zeros((1, D, m))], axis=0)

                # Solve (uses solve_rankk_bidiag, matching accelerated_broyden_deer_alg)
                if use_lrmat:
                    delta_s = solve_rankk_bidiag(d_eff, U_blocks, V_blocks, r_old,
                                                 m, compress=cfg.acc_compress, iteration=iteration)
                else:
                    delta_s = diagonal_solve(d_eff, r_old)

                new_states = states + delta_s
                r_new = get_residual(f, init_c, new_states, drv_c)

                # Update sliding window
                y = r_new - r_old
                new_target = jnp.concatenate(
                    [y[1:] - delta_s[1:], jnp.zeros((1, D))], axis=0)
                p_window = jnp.concatenate([delta_s[:, None, :], p_window[:, :-1, :]], axis=1)
                target_window = jnp.concatenate([new_target[:, None, :], target_window[:, :-1, :]], axis=1)

                # Gamma update for rescaled_identity
                if init_scheme == "rescaled_identity" and iteration > 0:
                    p_last = p_window[:, 0, :]
                    tgt_last = target_window[:, 0, :]
                    gamma_num = jnp.sum(p_last * tgt_last, axis=-1)
                    gamma_den = jnp.sum(p_last * p_last, axis=-1) + 1e-10
                    gamma_arr = gamma_num / gamma_den

                U = U_blocks if iteration > 0 else jnp.zeros_like(U)
                V = V_blocks if iteration > 0 else jnp.zeros_like(V)

            # --- Jacobian comparison at this iteration (on sampled timesteps) ---
            # Compare block to true J at the OLD states (where the block was computed)
            old_states_sub = states[:-1][jac_indices]      # (n_jac_samples, N)
            true_J_sub = true_jacobian_batch(M_c, Nt_c, a, dt_over_tau, old_states_sub)
            neg_J_sub = -true_J_sub

            # Current blocks at sampled timesteps
            if alg == "quasi":
                d_cur = d_eff[:-1][jac_indices]
                U_cur = jnp.zeros((n_jac_samples, D, 1))
                V_cur = jnp.zeros((n_jac_samples, D, 1))
            elif alg == "broyden":
                d_cur = d_eff[:-1][jac_indices] if init_scheme not in ("rescaled_identity", "jac") else (gamma_arr[:, None] * d_init)[:-1][jac_indices]
                U_cur = U[:-1][jac_indices]
                V_cur = V[:-1][jac_indices]
            else:  # accelerated_broyden
                d_cur = d_eff[:-1][jac_indices]
                U_cur = U[:-1][jac_indices] if iteration > 0 else jnp.zeros((n_jac_samples, D, max(m, 1)))
                V_cur = V[:-1][jac_indices] if iteration > 0 else jnp.zeros((n_jac_samples, D, max(m, 1)))

            # Block error (per-block spectral norm)
            B_dense = vmap(block_to_dense)(d_cur, U_cur, V_cur)
            errors = vmap(relative_spectral_error)(B_dense, neg_J_sub)
            mean_err = float(jnp.mean(errors))
            max_err = float(jnp.max(errors))

            # NOW update states
            states = new_states

            error_per_iter.append((iteration, rms, mean_err, max_err))

            print(f"  chain {chain_idx}, iter {iteration}: rms={rms:.2e}, "
                  f"jac_err={mean_err:.4f}")

        all_error_curves.append(error_per_iter)

    # --- log per-chain data at each iteration ---
    max_len = max(len(c) for c in all_error_curves)
    for i in range(max_len):
        log_dict = {"iteration": i}
        for chain_j, curve in enumerate(all_error_curves):
            if i < len(curve):
                _, _, me, _ = curve[i]
                log_dict[f"jac_err_chain_{chain_j}"] = float(me)
        # Also log aggregated for convenience
        errs_at_i = [curve[i][2] for curve in all_error_curves if i < len(curve)]
        if errs_at_i:
            log_dict["jac_err_mean_mean"] = float(jnp.mean(jnp.array(errs_at_i)))
            log_dict["jac_err_mean_std"] = float(jnp.std(jnp.array(errs_at_i)))
        wandb.log(log_dict)

    # Final summary
    final_jac_errs = [c[-1][2] for c in all_error_curves if len(c) > 0]
    if final_jac_errs:
        fj = jnp.array(final_jac_errs)
        results = {
            "final_jac_err_mean": float(jnp.mean(fj)),
            "final_jac_err_std": float(jnp.std(fj)),
            "n_converged_jac": n_analyze,
            "N": N, "R": R, "T": T, "dt": dt, "g": g,
            "alg": alg,
        }
        logger.log_metrics(results)
        print(f"\n{alg} | N={N}, T={T} | jac_err: "
              f"{results['final_jac_err_mean']:.4f} +/- {results['final_jac_err_std']:.4f}")


if __name__ == "__main__":
    main()
