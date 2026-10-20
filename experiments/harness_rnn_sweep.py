"""
harness_rnn_sweep.py
Low-rank RNN sweep over (N, T) for two regimes (Script A: dt=3, Script B: dt=7).

Algorithms:
  - deer           : full Newton DEER
  - quasi          : diagonal-Jacobian DEER (exact diagonal, no Hutchinson)
  - broyden        : Broyden-DEER (identity init, rand compress, line search)
  - accelerated_broyden : Accelerated Broyden-DEER (same, + eps_gram)
  - seq            : sequential scan baseline

All Broyden-type methods use:
  compress="rand", init="identity", gamma=1-dt/tau, m=R,
  line_search=True, ls_c=1e-3, k=0 (no damping).

Per-seed batching (like N_schools): each chain element gets a different
global seed that controls everything — RNN weights (M, Nt),
initial_state, drivers, and states_guess. Mini-batching controls GPU
memory.

Usage (via wandb sweep):
  wandb sweep sweep_rnn_A.yaml   # creates sweep, prints sweep ID
  wandb agent <entity>/<project>/<sweep_id>

Direct usage:
  python harness_rnn_sweep.py alg=deer N=100 T=1000 dt=3.0 g=2.0
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
        print(f"ERROR: {alg} | N={N}, T={T}, dt={dt} failed: {e}")
        wandb.log({"error": str(e), "N": N, "T": T, "dt": dt, "alg": alg,
                    "iters_mean": -1, "time_wallclock": -1, "n_converged": -1})
    finally:
        wandb.finish()


def _run(cfg, logger, seed, N, R, T, tau, dt, g, sigma_noise, phi,
         num_iters, tol, batch_size, nwarmups, nreps, alg):

    # --- per-seed chain setup ---
    # Each chain gets a different global seed (seed, seed+1, ...) that
    # controls everything: RNN weights, initial_state, drivers, states_guess.
    a = 1.0 - dt / tau
    dt_over_tau = dt / tau

    def setup_chain(chain_seed):
        chain_key = jr.PRNGKey(chain_seed)
        k_rnn, k_init, k_drv, k_guess = jr.split(chain_key, 4)
        rnn = LowRankRNN(N, R, key=k_rnn, tau=tau, dt=dt, g=g, phi=phi)
        initial_state = jr.normal(k_init, (N,)) * 0.1
        chain_drivers = sigma_noise * jr.normal(k_drv, (T, N))
        states_guess = jr.normal(k_guess, (T, N)) * 0.1
        return rnn.M, rnn.Nt, initial_state, chain_drivers, states_guess

    mini_batch = 4
    max_attempts = batch_size * 10  # safety cap to avoid infinite loop

    # --- deer_fxn that takes M, Nt as arguments (for vmap over weights) ---
    def _phi(x):
        if phi == "relu":
            return jnp.maximum(x, 0.0)
        else:
            return jnp.tanh(x)

    def deer_fxn_with_weights(M, Nt, state, inp):
        recurrent = M @ (Nt.T @ _phi(state))
        return a * state + dt_over_tau * recurrent + inp

    def make_deer_fxn(M, Nt):
        """Close over per-chain weights to get the standard (state, input) -> state signature."""
        def f(state, inp):
            return deer_fxn_with_weights(M, Nt, state, inp)
        return f

    # --- saturation pre-check via sequential trajectory ---
    # Run sequential scan per chain, check for tanh saturation, and
    # discard seeds where neurons are persistently saturated.
    # Keep drawing seeds until we have exactly batch_size healthy chains.
    sat_threshold = 0.8     # |tanh(x_i)| above this = saturated
    dead_threshold = 0.1    # mean phi'(x_i) below this = dead neuron

    def seq_trajectory(M_c, Nt_c, init_c, drv_c):
        f = make_deer_fxn(M_c, Nt_c)
        def step(state, driver):
            new = f(state, driver)
            return new, new
        _, traj = jax.lax.scan(step, init_c, drv_c)
        return traj

    def check_saturation(traj):
        """Returns (n_saturated, n_dead, worst_abs_mean, worst_phi_prime)."""
        burn = T // 4
        phi_vals = jnp.tanh(traj[burn:])
        abs_mean = jnp.mean(jnp.abs(phi_vals), axis=0)
        phi_prime_mean = jnp.mean(1.0 - phi_vals ** 2, axis=0)
        n_sat = jnp.sum(abs_mean > sat_threshold)
        n_dead = jnp.sum(phi_prime_mean < dead_threshold)
        return n_sat, n_dead, jnp.max(abs_mean), jnp.min(phi_prime_mean)

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

    if n_healthy < batch_size:
        print(f"WARNING: only found {n_healthy}/{batch_size} healthy chains "
              f"after {total_tried} attempts.")
        if n_healthy == 0:
            print("ERROR: no healthy chains found, nothing to benchmark.")
            wandb.log({"n_healthy": 0, "n_saturated_chains": total_saturated_chains,
                        "total_tried": total_tried})
            return

    # Batch into mini-batch groups
    n_groups = max(n_healthy // mini_batch, 1)
    # Trim to a clean multiple of mini_batch
    n_used = n_groups * mini_batch
    healthy_data = healthy_data[:n_used]
    n_healthy = n_used

    group_Ms = []
    group_Nts = []
    group_initial_states = []
    group_drivers = []
    group_states_guesses = []
    for grp in range(n_groups):
        cd_group = healthy_data[grp * mini_batch : (grp + 1) * mini_batch]
        group_Ms.append(jnp.stack([cd[0] for cd in cd_group]))
        group_Nts.append(jnp.stack([cd[1] for cd in cd_group]))
        group_initial_states.append(jnp.stack([cd[2] for cd in cd_group]))
        group_drivers.append(jnp.stack([cd[3] for cd in cd_group]))
        group_states_guesses.append(jnp.stack([cd[4] for cd in cd_group]))

    # --- evaluation functions ---
    # Each takes (M, Nt, initial_state, states_guess, drivers) so we can
    # vmap over the batch dim (each element has its own RNN weights).

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
            full_trace=False, tol=tol,
            stopping="rms",
        )
        return newton_steps

    def quasi_eval(M, Nt, initial_state, states_guess, drivers):
        f = make_deer_fxn(M, Nt)
        _, _, newton_steps, *_ = deer_alg(
            f, initial_state, states_guess, drivers,
            num_iters, quasi=True, qmem_efficient=False, k=cfg.deer_k,
            full_trace=False, tol=tol,
            stopping="rms",
        )
        return newton_steps

    def broyden_eval(M, Nt, initial_state, states_guess, drivers):
        f = make_deer_fxn(M, Nt)
        _, _, newton_steps, *_ = broyden_deer_alg(
            f, initial_state, states_guess, drivers,
            num_iters, m=cfg.bd_m, init=cfg.bd_init,
            k=cfg.bd_k, gamma=cfg.bd_gamma,
            compress=cfg.acc_compress,
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
            eps_gram=cfg.acc_eps_gram, compress=cfg.acc_compress,
            qmem_efficient=True,
            full_trace=False, tol=tol,
            line_search=True, ls_c=1e-3,
        )
        return newton_steps

    # select and compile
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

    # for compilation / warmup, use the first group
    Ms = group_Ms[0]
    Nts = group_Nts[0]
    initial_states = group_initial_states[0]
    states_guesses = group_states_guesses[0]
    drivers = group_drivers[0]

    if use_highest:
        with jax.default_matmul_precision("highest"):
            fxn = jax.jit(jax.vmap(eval_fns[alg]))
            fxn.lower(Ms, Nts, initial_states, states_guesses, drivers).compile()
    else:
        fxn = jax.jit(jax.vmap(eval_fns[alg]))
        fxn.lower(Ms, Nts, initial_states, states_guesses, drivers).compile()

    # warmup with first group
    if use_highest:
        with jax.default_matmul_precision("highest"):
            for _ in range(nwarmups):
                result = fxn(Ms, Nts, initial_states, states_guesses, drivers)
                jax.block_until_ready(result)
    else:
        for _ in range(nwarmups):
            result = fxn(Ms, Nts, initial_states, states_guesses, drivers)
            jax.block_until_ready(result)

    # timed runs: run all groups, accumulate results and time
    all_steps_list = []
    total_time = 0.0

    for grp in range(n_groups):
        gM = group_Ms[grp]
        gNt = group_Nts[grp]
        gi = group_initial_states[grp]
        gg = group_states_guesses[grp]
        gd = group_drivers[grp]
        if use_highest:
            with jax.default_matmul_precision("highest"):
                t0 = time.time()
                for _ in range(nreps):
                    result_g = fxn(gM, gNt, gi, gg, gd)
                    jax.block_until_ready(result_g)
                t1 = time.time()
        else:
            t0 = time.time()
            for _ in range(nreps):
                result_g = fxn(gM, gNt, gi, gg, gd)
                jax.block_until_ready(result_g)
            t1 = time.time()
        group_time = (t1 - t0) / nreps
        total_time += group_time

        if alg == "seq":
            all_steps_list.append(jnp.array([T] * mini_batch))
        else:
            all_steps_list.append(result_g)

    time_avg = total_time / n_groups
    all_steps = jnp.concatenate(all_steps_list)[:n_healthy]  # trim padding

    # convergence stats (over healthy chains only)
    if alg == "seq":
        n_converged = n_healthy
        iters_converged = all_steps
    else:
        converged_mask = all_steps < num_iters
        n_converged = int(jnp.sum(converged_mask))
        iters_converged = all_steps[converged_mask] if n_converged > 0 else jnp.array([])

    if n_converged > 0:
        iters_mean = float(jnp.mean(iters_converged))
        iters_std = float(jnp.std(iters_converged))
    else:
        iters_mean = float(num_iters)
        iters_std = 0.0

    # --- second pass: per-chain timing for converged chains only ---
    # Run each converged chain individually (no vmap) to get honest wallclock.
    eval_fn_single = eval_fns[alg]

    if use_highest:
        with jax.default_matmul_precision("highest"):
            fxn_single = jax.jit(eval_fn_single)
            # compile with first healthy chain
            M0, Nt0, init0, drv0, sg0 = healthy_data[0]
            fxn_single.lower(M0, Nt0, init0, sg0, drv0).compile()
    else:
        fxn_single = jax.jit(eval_fn_single)
        M0, Nt0, init0, drv0, sg0 = healthy_data[0]
        fxn_single.lower(M0, Nt0, init0, sg0, drv0).compile()

    # identify converged chain indices
    if alg == "seq":
        converged_indices = list(range(n_healthy))
    else:
        converged_indices = [i for i in range(n_healthy) if int(all_steps[i]) < num_iters]

    converged_times = []
    for idx in converged_indices:
        M_i, Nt_i, init_i, drv_i, sg_i = healthy_data[idx]
        # warmup
        if use_highest:
            with jax.default_matmul_precision("highest"):
                for _ in range(nwarmups):
                    r = fxn_single(M_i, Nt_i, init_i, sg_i, drv_i)
                    jax.block_until_ready(r)
                t0 = time.time()
                for _ in range(nreps):
                    r = fxn_single(M_i, Nt_i, init_i, sg_i, drv_i)
                    jax.block_until_ready(r)
                t1 = time.time()
        else:
            for _ in range(nwarmups):
                r = fxn_single(M_i, Nt_i, init_i, sg_i, drv_i)
                jax.block_until_ready(r)
            t0 = time.time()
            for _ in range(nreps):
                r = fxn_single(M_i, Nt_i, init_i, sg_i, drv_i)
                jax.block_until_ready(r)
            t1 = time.time()
        converged_times.append((t1 - t0) / nreps)

    if len(converged_times) > 0:
        ct = jnp.array(converged_times)
        time_conv_mean = float(jnp.mean(ct))
        time_conv_std = float(jnp.std(ct))
    else:
        time_conv_mean = -1.0
        time_conv_std = 0.0

    print(f"{alg} | N={N}, R={R}, T={T}, dt={dt}, g={g} | "
          f"converged: {n_converged}/{n_healthy} | "
          f"iters: {iters_mean:.1f} +/- {iters_std:.1f} | "
          f"time(conv): {time_conv_mean:.3e} +/- {time_conv_std:.3e} s | "
          f"healthy: {n_healthy}/{total_tried}")
    if alg != "seq":
        print(f"  per-chain newton_steps: {all_steps.tolist()}")
        print(f"  per-chain times (conv): {[f'{t:.3e}' for t in converged_times]}")

    results = {
        "iters_mean": iters_mean,
        "iters_std": iters_std,
        "time_conv_mean": time_conv_mean,
        "time_conv_std": time_conv_std,
        "time_wallclock_vmap": time_avg,
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
    }
    logger.log_metrics(results)


if __name__ == "__main__":
    main()
