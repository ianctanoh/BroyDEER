"""
harness_gru.py
Runs GRU dynamical-system experiments using hydra + wandb.

Same structure as harness_sn_word.py: the update function (deer_fxn) is
determined solely by the GRU weights, which are drawn once from `seed`/`gain`
and shared across all chains -- so we jit the eval function once and loop
over `batch_size` chains re-using that same compiled function, timing each
chain individually. What varies per chain is the driver sequence, initial
state, and Newton initial guess.

Options for alg are:
- seq
- deer
- quasi
- broyden
- accelerated_broyden

Each run also reports the RMS residual (get_residual + stopping_criterion,
same metric used internally by the "rms" stopping mode) evaluated at the
trajectory the algorithm stopped on -- useful for seeing how close a
non-converged run got, not just that it hit num_iters.

Sweepable parameters:
  D              : GRU hidden/state dim
  T              : sequence length
  gain           : GRU weight gain (spectral scale)
  sigma_noise    : driver (input) noise scale
  deer_k         : DEER / quasi-DEER damping (0 = no damping)
  bd_m           : Broyden-DEER rank
  bd_init        : Broyden-DEER init scheme (jac, identity, rescaled_identity, low_rank, fresh_jac)
  bd_k           : Broyden-DEER damping (0 = no damping)
  acc_m          : Accelerated Broyden rank
  acc_init       : Accelerated Broyden init scheme (autodiff, identity, rescaled_identity)
  acc_k          : Accelerated Broyden damping (0 = no damping)
  acc_eps_gram   : Accelerated Broyden Gram regularization

Example usage:
  python harness_gru.py alg=deer D=32 T=1000
  python harness_gru.py -m alg=deer,quasi,broyden,accelerated_broyden D=32,64,128 T=1000
  python harness_gru.py alg=accelerated_broyden D=32 acc_m=1,2,3,5,10
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import jax
# assert jax.devices()[0].platform == "gpu", f"No GPU found. Devices: {jax.devices()}"
import jax.numpy as jnp
import jax.random as jr
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger
import time

from dynamics.gru import GRU
from algs.deer import deer_alg, get_residual, stopping_criterion
from algs.broyden_deer import broyden_deer_alg
from algs.accelerated_broyden import accelerated_broyden_deer_alg


@hydra.main(config_path="configs", config_name="harness_gru")
def main(cfg: DictConfig):
    seed = cfg.seed
    D = cfg.D
    T = cfg.T
    gain = cfg.gain
    sigma_noise = cfg.sigma_noise
    num_iters = cfg.num_iters
    tol = cfg.tol
    batch_size = cfg.batch_size
    nwarmups = cfg.nwarmups
    nreps = cfg.nreps
    mode = cfg.mode
    alg = cfg.alg

    logger = WandbLogger(project="broydeer_gru", mode=mode)
    logger.log_hyperparams(OmegaConf.to_container(cfg))
    wandb.config.update(OmegaConf.to_container(cfg), allow_val_change=True)

    # GRU weights are fixed for the whole sweep -- drawn once from `seed`
    key = jr.PRNGKey(seed)
    key, k_gru = jr.split(key)
    gru = GRU(D, key=k_gru, gain=gain)
    deer_fxn = gru.deer_fxn

    # per-chain data: same deer_fxn, different drivers/initial_state/states_guess
    def setup_chain(chain_idx):
        chain_key = jr.fold_in(key, chain_idx)
        k1, k2 = jr.split(chain_key, 2)
        drivers_i = sigma_noise * jr.normal(k1, (T, D))
        initial_state_i = jnp.zeros(D)
        states_guess_i = jr.normal(k2, (T, D)) * 0.1
        return drivers_i, initial_state_i, states_guess_i

    all_chain_data = [setup_chain(i) for i in range(batch_size)]

    # final RMS residual at the trajectory an algorithm stopped on --
    # matches the stopping_criterion used internally by deer/quasi (stopping="rms")
    # and by broyden/accelerated_broyden, computed post-hoc since the algs
    # discard their internal mf_val when full_trace=False.
    def final_rms(final_states, initial_state, drivers):
        r = get_residual(deer_fxn, initial_state, final_states, drivers)
        return stopping_criterion(r)

    # define evaluation function once -- deer_fxn is shared across chains
    def seq_eval(initial_state, states_guess, drivers):
        def step(state, driver):
            new = deer_fxn(state, driver)
            return new, new
        _, samples = jax.lax.scan(step, initial_state, drivers)
        return samples[-1], final_rms(samples, initial_state, drivers)

    def deer_eval(initial_state, states_guess, drivers):
        _, final_states, newton_steps, *_ = deer_alg(
            deer_fxn, initial_state, states_guess, drivers,
            num_iters, k=cfg.deer_k,
            full_trace=False, tol=tol, stopping="rms",
        )
        return newton_steps, final_rms(final_states, initial_state, drivers)

    def quasi_eval(initial_state, states_guess, drivers):
        _, final_states, newton_steps, *_ = deer_alg(
            deer_fxn, initial_state, states_guess, drivers,
            num_iters, quasi=True, k=cfg.deer_k,
            full_trace=False, tol=tol, stopping="rms",
        )
        return newton_steps, final_rms(final_states, initial_state, drivers)

    def broyden_eval(initial_state, states_guess, drivers):
        _, final_states, newton_steps, *_ = broyden_deer_alg(
            deer_fxn, initial_state, states_guess, drivers,
            num_iters, m=cfg.bd_m, init=cfg.bd_init,
            k=cfg.bd_k, gamma=cfg.bd_gamma,
            compress=cfg.bd_compress,
            full_trace=False, tol=tol,
            line_search=True, ls_c=1e-3,
        )
        return newton_steps, final_rms(final_states, initial_state, drivers)

    def accelerated_broyden_eval(initial_state, states_guess, drivers):
        _, final_states, newton_steps, *_ = accelerated_broyden_deer_alg(
            deer_fxn, initial_state, states_guess, drivers,
            num_iters, m=cfg.acc_m, init=cfg.acc_init,
            k=cfg.acc_k, gamma=cfg.acc_gamma,
            eps_gram=cfg.acc_eps_gram, compress=cfg.acc_compress,
            full_trace=False, tol=tol,
            line_search=True, ls_c=1e-3,
        )
        return newton_steps, final_rms(final_states, initial_state, drivers)

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

    # jit once -- deer_fxn (and thus eval_fn) is identical across chains,
    # so re-using this compiled function across the loop below does not
    # trigger recompilation.
    fxn = jax.jit(eval_fns[alg])
    drivers_0, init_0, guess_0 = all_chain_data[0]

    if use_highest:
        with jax.default_matmul_precision("highest"):
            fxn.lower(init_0, guess_0, drivers_0).compile()
            for _ in range(nwarmups):
                jax.block_until_ready(fxn(init_0, guess_0, drivers_0))
    else:
        fxn.lower(init_0, guess_0, drivers_0).compile()
        for _ in range(nwarmups):
            jax.block_until_ready(fxn(init_0, guess_0, drivers_0))

    # timed runs: loop over each chain individually, reusing the same
    # compiled fxn -- gives per-chain n_iters/time like harness_sn_word.py
    all_steps_list = []
    all_times_list = []
    all_final_rms_list = []

    for i in range(batch_size):
        drivers_i, init_i, guess_i = all_chain_data[i]

        if use_highest:
            with jax.default_matmul_precision("highest"):
                t0 = time.time()
                for _ in range(nreps):
                    result_i = fxn(init_i, guess_i, drivers_i)
                    jax.block_until_ready(result_i)
                t1 = time.time()
        else:
            t0 = time.time()
            for _ in range(nreps):
                result_i = fxn(init_i, guess_i, drivers_i)
                jax.block_until_ready(result_i)
            t1 = time.time()

        chain_time = (t1 - t0) / nreps
        all_times_list.append(chain_time)

        steps_or_state, final_err_i = result_i
        all_final_rms_list.append(float(final_err_i))
        if alg == "seq":
            all_steps_list.append(T)
        else:
            all_steps_list.append(int(steps_or_state))

    all_steps = jnp.array(all_steps_list)
    all_times = jnp.array(all_times_list)

    if alg == "seq":
        n_converged = batch_size
        converged_steps = all_steps
        converged_times = all_times
    else:
        converged_mask = all_steps < num_iters
        n_converged = int(jnp.sum(converged_mask))
        converged_steps = all_steps[converged_mask]
        converged_times = all_times[converged_mask]

    n_iters_mean = float(jnp.mean(converged_steps)) if n_converged > 0 else float(num_iters)
    n_iters_sd = float(jnp.std(converged_steps)) if n_converged > 0 else 0.0
    time_mean = float(jnp.mean(converged_times)) if n_converged > 0 else float(jnp.mean(all_times))
    time_sd = float(jnp.std(converged_times)) if n_converged > 0 else float(jnp.std(all_times))

    all_final_rms = jnp.array(all_final_rms_list)
    final_rms_mean = float(jnp.mean(all_final_rms))
    final_rms_sd = float(jnp.std(all_final_rms))

    print(f"{alg} | D={D}, T={T}, gain={gain} | "
          f"converged: {n_converged}/{batch_size} | "
          f"n_iters: {n_iters_mean:.1f} ± {n_iters_sd:.1f} | "
          f"time: {time_mean:.3e} ± {time_sd:.3e}s | "
          f"final_rms: {final_rms_mean:.3e} ± {final_rms_sd:.3e}")
    print(f"  per-chain n_iters: {all_steps.tolist()}")
    print(f"  per-chain times: {[f'{t:.4e}' for t in all_times_list]}")
    print(f"  per-chain final_rms: {[f'{e:.4e}' for e in all_final_rms_list]}")

    results = {
        "per_seed_n_iters": all_steps.tolist(),
        "per_seed_times": all_times_list,
        "per_seed_final_rms": all_final_rms_list,
        "n_iters_mean": n_iters_mean,
        "n_iters_sd": n_iters_sd,
        "time_mean": time_mean,
        "time_sd": time_sd,
        "final_rms_mean": final_rms_mean,
        "final_rms_sd": final_rms_sd,
        "n_converged": n_converged,
        "frac_converged": n_converged / batch_size,
        "D": D,
    }
    logger.log_metrics(results)
    wandb.finish()


if __name__ == "__main__":
    main()
