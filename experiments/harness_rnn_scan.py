"""
harness_rnn_scan.py
Same as harness_rnn_compress.py but compares associative-scan tree topology
(Blelloch vs Hillis-Steele) instead of compression scheme, for both
Broyden-DEER and Accelerated Broyden.

scan_tree options: "blelloch", "hillis_steele"
acc_compress is held fixed (not swept) -- see sweep_rnn_A_scan_tree.yaml.

Usage:
  python harness_rnn_scan.py alg=accelerated_broyden scan_tree=hillis_steele N=100 T=100000
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import jax
import jax.numpy as jnp
import jax.random as jr
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger
import time

from dynamics.low_rank_rnn import LowRankRNN
from algs.deer import deer_alg
from algs.broyden_deer import broyden_deer_alg
from algs.accelerated_broyden import accelerated_broyden_deer_alg


@hydra.main(config_path="configs", config_name="harness_rnn_scan", version_base=None)
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
    nwarmups = cfg.nwarmups
    nreps = cfg.nreps
    mode = cfg.mode
    alg = cfg.alg

    logger = WandbLogger(project="broydeer_rnn_sweep", mode=mode)
    logger.log_hyperparams(OmegaConf.to_container(cfg))
    wandb.config.update(OmegaConf.to_container(cfg), allow_val_change=True)

    try:
        _run(cfg, logger, seed, N, R, T, tau, dt, g, sigma_noise, phi,
             num_iters, tol, batch_size, nwarmups, nreps, alg)
    except Exception as e:
        print(f"ERROR: {alg} | N={N}, T={T}, dt={dt}, scan_tree={cfg.scan_tree} failed: {e}")
        wandb.log({"error": str(e), "N": N, "T": T, "dt": dt, "alg": alg,
                    "scan_tree": cfg.scan_tree,
                    "iters_mean": -1, "time_conv_mean": -1, "n_converged": -1})
    finally:
        wandb.finish()


def _run(cfg, logger, seed, N, R, T, tau, dt, g, sigma_noise, phi,
         num_iters, tol, batch_size, nwarmups, nreps, alg):

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

    def deer_fxn_with_weights(M, Nt, state, inp):
        recurrent = M @ (Nt.T @ _phi(state))
        return a * state + dt_over_tau * recurrent + inp

    def make_deer_fxn(M, Nt):
        def f(state, inp):
            return deer_fxn_with_weights(M, Nt, state, inp)
        return f

    # --- saturation check ---
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
        return n_sat, n_dead, jnp.max(abs_mean), jnp.min(phi_prime_mean)

    # collect healthy chains
    healthy_data = []
    total_saturated_chains = 0
    total_tried = 0
    chain_idx = 0

    print(f"Collecting {batch_size} healthy chains (sat_thr={sat_threshold}, "
          f"dead_thr={dead_threshold})...")
    while len(healthy_data) < batch_size and chain_idx < max_attempts:
        chain_seed_i = seed + chain_idx
        cd = setup_chain(chain_seed_i)
        M_i, Nt_i, init_i, drv_i, _ = cd
        traj_i = seq_trajectory(M_i, Nt_i, init_i, drv_i)
        n_sat, n_dead, worst_abs, worst_phi = check_saturation(traj_i)
        n_sat, n_dead = int(n_sat), int(n_dead)
        total_tried += 1

        if n_sat > 0 or n_dead > 0:
            total_saturated_chains += 1
            print(f"  seed {chain_seed_i}: SATURATED "
                  f"({n_sat} neurons |tanh|>{sat_threshold}, "
                  f"{n_dead} neurons phi'<{dead_threshold}, "
                  f"worst |tanh|={float(worst_abs):.3f}, "
                  f"min phi'={float(worst_phi):.3f}) — skipping")
        else:
            healthy_data.append(cd)
            print(f"  seed {chain_seed_i}: healthy [{len(healthy_data)}/{batch_size}] "
                  f"(worst |tanh|={float(worst_abs):.3f}, "
                  f"min phi'={float(worst_phi):.3f})")
        chain_idx += 1

    n_healthy = len(healthy_data)
    print(f"\nSaturation filter: {n_healthy}/{total_tried} seeds healthy, "
          f"{total_saturated_chains} saturated and skipped.")

    if n_healthy == 0:
        print("ERROR: no healthy chains found, nothing to benchmark.")
        wandb.log({"n_healthy": 0, "n_saturated_chains": total_saturated_chains,
                    "total_tried": total_tried})
        return

    # --- eval functions (single chain) ---
    def seq_eval(M, Nt, initial_state, states_guess, drivers):
        f = make_deer_fxn(M, Nt)
        def step(state, driver):
            new = f(state, driver)
            return new, new
        _, samples = jax.lax.scan(step, initial_state, drivers)
        return samples[-1]

    def deer_eval(M, Nt, initial_state, states_guess, drivers):
        f = make_deer_fxn(M, Nt)
        _, _, newton_steps, *_ = deer_alg(
            f, initial_state, states_guess, drivers,
            num_iters, k=cfg.deer_k,
            full_trace=False, tol=tol, stopping="rms",
        )
        return newton_steps

    def quasi_eval(M, Nt, initial_state, states_guess, drivers):
        f = make_deer_fxn(M, Nt)
        _, _, newton_steps, *_ = deer_alg(
            f, initial_state, states_guess, drivers,
            num_iters, quasi=True, qmem_efficient=False, k=cfg.deer_k,
            full_trace=False, tol=tol, stopping="rms",
        )
        return newton_steps

    def broyden_eval(M, Nt, initial_state, states_guess, drivers):
        f = make_deer_fxn(M, Nt)
        _, _, newton_steps, *_ = broyden_deer_alg(
            f, initial_state, states_guess, drivers,
            num_iters, m=cfg.bd_m, init=cfg.bd_init,
            k=cfg.bd_k, gamma=cfg.bd_gamma,
            compress=cfg.acc_compress,
            scan_tree=cfg.scan_tree,
            full_trace=False, tol=tol,
            line_search=True, ls_c=1e-3,
        )
        return newton_steps

    def accelerated_broyden_eval(M, Nt, initial_state, states_guess, drivers):
        f = make_deer_fxn(M, Nt)
        _, _, newton_steps, *_ = accelerated_broyden_deer_alg(
            f, initial_state, states_guess, drivers,
            num_iters, m=cfg.acc_m, init=cfg.acc_init,
            k=cfg.acc_k, gamma=cfg.acc_gamma,
            eps_gram=cfg.acc_eps_gram,
            compress=cfg.acc_compress,
            scan_tree=cfg.scan_tree,
            qmem_efficient=True,
            full_trace=False, tol=tol,
            line_search=True, ls_c=1e-3,
        )
        return newton_steps

    eval_fns = {
        "seq": seq_eval,
        "deer": deer_eval,
        "quasi": quasi_eval,
        "broyden": broyden_eval,
        "accelerated_broyden": accelerated_broyden_eval,
    }
    if alg not in eval_fns:
        raise ValueError(f"Invalid algorithm: {alg}. Choose from {list(eval_fns.keys())}")

    use_highest = alg in ("deer", "quasi", "broyden", "accelerated_broyden")

    # compile single-chain jit with first healthy chain
    eval_fn = eval_fns[alg]
    if use_highest:
        with jax.default_matmul_precision("highest"):
            fxn = jax.jit(eval_fn)
            M0, Nt0, init0, drv0, sg0 = healthy_data[0]
            fxn.lower(M0, Nt0, init0, sg0, drv0).compile()
    else:
        fxn = jax.jit(eval_fn)
        M0, Nt0, init0, drv0, sg0 = healthy_data[0]
        fxn.lower(M0, Nt0, init0, sg0, drv0).compile()

    # --- run all chains sequentially, collect iters + timing ---
    all_steps = []
    all_times = []

    for idx in range(n_healthy):
        M_i, Nt_i, init_i, drv_i, sg_i = healthy_data[idx]
        # warmup
        if use_highest:
            with jax.default_matmul_precision("highest"):
                for _ in range(nwarmups):
                    r = fxn(M_i, Nt_i, init_i, sg_i, drv_i)
                    jax.block_until_ready(r)
                t0 = time.time()
                for _ in range(nreps):
                    r = fxn(M_i, Nt_i, init_i, sg_i, drv_i)
                    jax.block_until_ready(r)
                t1 = time.time()
        else:
            for _ in range(nwarmups):
                r = fxn(M_i, Nt_i, init_i, sg_i, drv_i)
                jax.block_until_ready(r)
            t0 = time.time()
            for _ in range(nreps):
                r = fxn(M_i, Nt_i, init_i, sg_i, drv_i)
                jax.block_until_ready(r)
            t1 = time.time()

        chain_time = (t1 - t0) / nreps
        if alg == "seq":
            steps = T
        else:
            steps = int(r)
        all_steps.append(steps)
        all_times.append(chain_time)
        print(f"  chain {idx}: steps={steps}, time={chain_time:.3e}s")

    all_steps = jnp.array(all_steps)
    all_times = jnp.array(all_times)

    # convergence stats (converged only)
    if alg == "seq":
        n_converged = n_healthy
        iters_converged = all_steps
        times_converged = all_times
    else:
        converged_mask = all_steps < num_iters
        n_converged = int(jnp.sum(converged_mask))
        iters_converged = all_steps[converged_mask] if n_converged > 0 else jnp.array([])
        times_converged = all_times[converged_mask] if n_converged > 0 else jnp.array([])

    if n_converged > 0:
        iters_mean = float(jnp.mean(iters_converged))
        iters_std = float(jnp.std(iters_converged))
        time_conv_mean = float(jnp.mean(times_converged))
        time_conv_std = float(jnp.std(times_converged))
    else:
        iters_mean = float(num_iters)
        iters_std = 0.0
        time_conv_mean = -1.0
        time_conv_std = 0.0

    print(f"{alg} | N={N}, R={R}, T={T}, dt={dt}, g={g}, scan_tree={cfg.scan_tree} | "
          f"converged: {n_converged}/{n_healthy} | "
          f"iters: {iters_mean:.1f} +/- {iters_std:.1f} | "
          f"time(conv): {time_conv_mean:.3e} +/- {time_conv_std:.3e} s | "
          f"healthy: {n_healthy}/{total_tried}")
    if alg != "seq":
        print(f"  per-chain newton_steps: {all_steps.tolist()}")
        print(f"  per-chain times:        {[f'{t:.3e}' for t in all_times.tolist()]}")

    results = {
        "iters_mean": iters_mean,
        "iters_std": iters_std,
        "time_conv_mean": time_conv_mean,
        "time_conv_std": time_conv_std,
        "n_converged": n_converged,
        "frac_converged": n_converged / n_healthy,
        "n_healthy": n_healthy,
        "n_saturated_chains": total_saturated_chains,
        "total_tried": total_tried,
        "N": N,
        "R": R,
        "T": T,
        "dt": dt,
        "g": g,
        "scan_tree": cfg.scan_tree,
    }
    # Per-chain data
    for i in range(n_healthy):
        results[f"n_iters_chain_{i}"] = int(all_steps[i])
        results[f"time_chain_{i}"] = float(all_times[i])
    logger.log_metrics(results)


if __name__ == "__main__":
    main()
