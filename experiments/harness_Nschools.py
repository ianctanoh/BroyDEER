"""
harness_Nschools.py
Runs N-schools Gibbs-sampler experiments using hydra + wandb.

Options for alg are:
- seq
- deer
- quasi
- quasi_qmem
- picard
- jacobi
- broyden
- accelerated_broyden

Stopping (all parallel solvers): ||r(x_{1:T})||_2 / sqrt(T*D) < tol, so the
criterion scales with the chain length T and state dimension D = 2S + 2.

Sweepable parameters:
  S              : number of schools (8, 16, 32, ...)
  T              : chain length / number of Gibbs samples
  deer_k         : DEER (full Newton) damping (0 = no damping)
  quasi_k        : quasi-DEER damping, also used by quasi_qmem (0 = no damping)
  bd_m           : Broyden-DEER rank
  bd_init        : Broyden-DEER init scheme (jac, identity, rescaled_identity, fresh_jac)
  bd_k           : Broyden-DEER damping (0 = no damping)
  bd_line_search : Broyden-DEER backtracking safeguard on/off (default on)
  acc_m          : Accelerated Broyden rank
  acc_init       : Accelerated Broyden init scheme (autodiff, identity, rescaled_identity)
  acc_k          : Accelerated Broyden damping (0 = no damping)
  acc_eps_gram   : Accelerated Broyden Gram regularization
  acc_line_search: Accelerated Broyden backtracking safeguard on/off (default on)
  acc_ls_c       : Accelerated Broyden safeguard step fraction for bad steps (default 0.01)

Example usage:
  python harness_Nschools.py alg=deer S=8 T=10000
  python harness_Nschools.py -m alg=deer,quasi,picard,jacobi S=8,16,32 T=10000
  python harness_Nschools.py alg=broyden bd_m=1,2,3,4 bd_init=jac,identity bd_k=0.0,0.5
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import jax
# float64 unless the sweep command sets JAX_ENABLE_X64=False (float32 ablation)
X64 = os.environ.get("JAX_ENABLE_X64", "True").strip().lower() not in ("0", "false", "no")
jax.config.update("jax_enable_x64", X64)
# assert jax.devices()[0].platform == "gpu", f"No GPU found. Devices: {jax.devices()}"
import jax.numpy as jnp
import jax.random as jr
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger
import time
from functools import partial

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from dynamics.eight_schools import make_eight_schools, make_gibbs_step
from algs.deer import deer_alg, picard_alg, jacobi_alg, get_residual, stopping_criterion
from algs.broyden_deer import broyden_deer_alg
from algs.accelerated_broyden import accelerated_broyden_deer_alg


@hydra.main(config_path="configs", config_name="harness_Nschools")
def main(cfg: DictConfig):
    seed = cfg.seed
    S = cfg.S
    T = cfg.T
    num_iters = cfg.num_iters
    tol = cfg.tol
    batch_size = cfg.batch_size
    nwarmups = cfg.nwarmups
    nreps = cfg.nreps
    mode = cfg.mode
    alg = cfg.alg

    logger = WandbLogger(project="broydeer", mode=mode)
    logger.log_hyperparams(OmegaConf.to_container(cfg))
    wandb.config.update(OmegaConf.to_container(cfg), allow_val_change=True)

    # Per-element chains: each gets a different seed for data generation
    # and a different PRNG key for chain initialization/drivers
    total_chains = batch_size

    def setup_chain(chain_idx):
        """Set up one chain with its own dataset and initialization."""
        chain_seed = seed + chain_idx
        deer_fxn_i, params_i, xs_i, D_i = make_eight_schools(
            S=S, N_s=cfg.N_s, seed=chain_seed,
            tausq_0=cfg.tausq_0, sigmasq_0=cfg.sigmasq_0)

        nu_N      = params_i["nu_N"]
        alpha_N   = params_i["alpha_N"]
        tausq_0_i = params_i["tausq_0"]
        kappa_0   = params_i["kappa_0"]
        mu_0      = params_i["mu_0"]
        sigmasq_0_i = params_i.get("sigmasq_0", 10.0)

        def sample_prior(key):
            key, *skeys = jr.split(key, 5)
            tausq    = tfd.InverseGamma(0.5 * nu_N, nu_N * tausq_0_i * 0.5).sample(seed=skeys[0])
            mu       = tfd.Normal(mu_0, jnp.sqrt(tausq / kappa_0)).sample(seed=skeys[1])
            thetas   = tfd.Normal(mu, jnp.sqrt(tausq)).sample((S,), seed=skeys[2])
            sigmasqs = tfd.InverseGamma(0.5 * alpha_N, 0.5 * alpha_N * sigmasq_0_i).sample((S,), seed=skeys[3])
            return jnp.concatenate((thetas, sigmasqs, jnp.array([mu]), jnp.array([tausq])))

        master_key = jr.PRNGKey(chain_idx)
        key, init_key = jr.split(master_key)
        init_state = sample_prior(init_key)
        key, step_key = jr.split(key)
        init_state = deer_fxn_i(init_state, step_key)
        key, drivers_key = jr.split(key)
        drivers = jr.split(drivers_key, T)

        return deer_fxn_i, init_state, drivers, D_i, xs_i

    all_chain_data = [setup_chain(i) for i in range(total_chains)]
    deer_fxn_0, _, _, D, _ = all_chain_data[0]

    # lower bounds: sigmasqs and tausq must be positive
    lower_bounds = jnp.concatenate([
        jnp.full(S, -jnp.inf),      # thetas
        jnp.full(S, 1e-6),          # sigmasqs
        jnp.array([-jnp.inf]),      # mu
        jnp.array([1e-6]),           # tausq
    ])

    # Use first chain for print info
    _, init_state_0, drivers_0, _, _ = all_chain_data[0]
    states_guess_0 = jnp.broadcast_to(init_state_0, (T, D))

    print(f"x64 enabled: {jax.config.jax_enable_x64}")
    print(f"states_guesses dtype: {states_guess_0.dtype}")
    print(f"drivers dtype: {drivers_0.dtype}")

    # define evaluation functions — each takes its own deer_fxn
    def make_seq_eval(deer_fxn_i):
        def seq_eval(initial_state, states_guess, drivers):
            def step(state, driver):
                new = deer_fxn_i(state, driver)
                return new, new
            _, samples = jax.lax.scan(step, initial_state, drivers)
            return samples[-1]
        return seq_eval

    def make_deer_eval(deer_fxn_i):
        def deer_eval(initial_state, states_guess, drivers):
            _, final_state, newton_steps, is_nan, *_ = deer_alg(
                deer_fxn_i, initial_state, states_guess, drivers,
                num_iters, k=cfg.deer_k,
                full_trace=False, tol=tol, stopping="rms",
            )
            return newton_steps, final_state, is_nan
        return deer_eval

    def make_quasi_eval(deer_fxn_i):
        def quasi_eval(initial_state, states_guess, drivers):
            _, final_state, newton_steps, is_nan, *_ = deer_alg(
                deer_fxn_i, initial_state, states_guess, drivers,
                num_iters, quasi=True, k=cfg.quasi_k,
                full_trace=False, tol=tol, stopping="rms",
            )
            return newton_steps, final_state, is_nan
        return quasi_eval

    def make_quasi_qmem_eval(deer_fxn_i):
        def quasi_qmem_eval(initial_state, states_guess, drivers):
            _, final_state, newton_steps, is_nan, *_ = deer_alg(
                deer_fxn_i, initial_state, states_guess, drivers,
                num_iters, quasi=True, k=cfg.quasi_k,
                qmem_efficient=True, num_samples=cfg.num_samples,
                full_trace=False, tol=tol, stopping="rms",
            )
            return newton_steps, final_state, is_nan
        return quasi_qmem_eval

    def make_picard_eval(deer_fxn_i):
        def picard_eval(initial_state, states_guess, drivers):
            _, final_state, newton_steps, is_nan, *_ = picard_alg(
                deer_fxn_i, initial_state, states_guess, drivers,
                num_iters, full_trace=False, tol=tol, stopping="rms",
            )
            return newton_steps, final_state, is_nan
        return picard_eval

    def make_jacobi_eval(deer_fxn_i):
        def jacobi_eval(initial_state, states_guess, drivers):
            _, final_state, newton_steps, is_nan, *_ = jacobi_alg(
                deer_fxn_i, initial_state, states_guess, drivers,
                num_iters, full_trace=False, tol=tol, stopping="rms",
            )
            return newton_steps, final_state, is_nan
        return jacobi_eval

    def make_broyden_eval(deer_fxn_i):
        def broyden_eval(initial_state, states_guess, drivers):
            _, final_state, newton_steps, is_nan, *_ = broyden_deer_alg(
                deer_fxn_i, initial_state, states_guess, drivers,
                num_iters, m=cfg.bd_m, init=cfg.bd_init,
                k=cfg.bd_k, gamma=cfg.bd_gamma,
                sequential_solve=False,
                compress=cfg.acc_compress,
                full_trace=False, tol=tol,
                lower_bounds=lower_bounds,
                line_search=cfg.bd_line_search, ls_c=1e-3,
            )
            return newton_steps, final_state, is_nan
        return broyden_eval

    def make_accelerated_broyden_eval(deer_fxn_i):
        def accelerated_broyden_eval(initial_state, states_guess, drivers):
            _, final_state, newton_steps, is_nan, *_ = accelerated_broyden_deer_alg(
                deer_fxn_i, initial_state, states_guess, drivers,
                num_iters, m=cfg.acc_m, init=cfg.acc_init,
                k=cfg.acc_k, gamma=cfg.acc_gamma,
                eps_gram=cfg.acc_eps_gram, compress=cfg.acc_compress,
                full_trace=False, tol=tol,
                lower_bounds=lower_bounds, line_search=cfg.acc_line_search, ls_c=cfg.acc_ls_c,
            )
            return newton_steps, final_state, is_nan
        return accelerated_broyden_eval

    make_eval_fns = {
        "seq": make_seq_eval,
        "deer": make_deer_eval,
        "quasi": make_quasi_eval,
        "quasi_qmem": make_quasi_qmem_eval,
        "picard": make_picard_eval,
        "jacobi": make_jacobi_eval,
        "broyden": make_broyden_eval,
        "accelerated_broyden": make_accelerated_broyden_eval,
    }
    if alg not in make_eval_fns:
        raise ValueError(f"Invalid algorithm: {alg}. Choose from {list(make_eval_fns.keys())}")

    use_highest = alg in ("deer", "quasi", "quasi_qmem", "broyden", "accelerated_broyden")

    # Compile once per run: the chain's observed data xs is a jit argument (not a
    # baked-in constant), so all chains share one compiled solver.
    def gibbs_step(xs):
        return make_gibbs_step(S, cfg.N_s, xs, tausq_0=cfg.tausq_0, sigmasq_0=cfg.sigmasq_0)

    make_eval = make_eval_fns[alg]
    fxn = jax.jit(lambda xs, *args: make_eval(gibbs_step(xs))(*args))
    final_rms_fxn = jax.jit(lambda xs, x0, states, drivers: stopping_criterion(
        get_residual(gibbs_step(xs), x0, states, drivers)))

    # timed runs: loop over each chain individually
    all_steps_list = []
    all_times_list = []
    final_rms_list = []   # per-chain RMS residual of the returned trajectory
    is_nan_list = []      # per-chain flag: a NaN appeared during the solve

    for i in range(total_chains):
        _, init_state_i, drivers_i, _, xs_i = all_chain_data[i]
        states_guess_i = jnp.broadcast_to(init_state_i, (T, D))
        fxn_i = partial(fxn, xs_i)

        # warmup: trigger compilation before timing
        if use_highest:
            with jax.default_matmul_precision("highest"):
                for _ in range(nwarmups):
                    _warmup = fxn_i(init_state_i, states_guess_i, drivers_i)
                    jax.block_until_ready(_warmup)
        else:
            for _ in range(nwarmups):
                _warmup = fxn_i(init_state_i, states_guess_i, drivers_i)
                jax.block_until_ready(_warmup)

        if use_highest:
            with jax.default_matmul_precision("highest"):
                t0 = time.time()
                for _ in range(nreps):
                    result_i = fxn_i(init_state_i, states_guess_i, drivers_i)
                    jax.block_until_ready(result_i)
                t1 = time.time()
        else:
            t0 = time.time()
            for _ in range(nreps):
                result_i = fxn_i(init_state_i, states_guess_i, drivers_i)
                jax.block_until_ready(result_i)
            t1 = time.time()
        chain_time = (t1 - t0) / nreps
        all_times_list.append(chain_time)

        if alg == "seq":
            all_steps_list.append(T)
            final_rms_list.append(0.0)      # exact by construction
            is_nan_list.append(False)
        else:
            newton_steps_i, final_state_i, is_nan_i = result_i
            all_steps_list.append(int(newton_steps_i))
            # untimed: RMS residual ||r||_2 / sqrt(T*D) of the returned trajectory
            with jax.default_matmul_precision("highest"):
                final_rms_i = final_rms_fxn(xs_i, init_state_i, final_state_i, drivers_i)
            final_rms_list.append(float(final_rms_i))
            is_nan_list.append(bool(is_nan_i))

    all_steps = jnp.array(all_steps_list)
    all_times = jnp.array(all_times_list)

    if alg == "seq":
        n_converged = total_chains
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

    print(f"{alg} | S={S}, T={T}, D={D} | "
          f"converged: {n_converged}/{total_chains} | "
          f"n_iters: {n_iters_mean:.1f} ± {n_iters_sd:.1f} | "
          f"time: {time_mean:.3e} ± {time_sd:.3e}s")
    print(f"  per-chain n_iters: {all_steps.tolist()}")
    print(f"  per-chain times: {[f'{t:.4e}' for t in all_times_list]}")
    print(f"  per-chain final RMS residual: {[f'{r:.2e}' for r in final_rms_list]}")
    print(f"  per-chain NaN: {is_nan_list}")

    results = {
        "per_chain_n_iters": all_steps.tolist(),
        "per_chain_times": all_times_list,
        "per_chain_final_rms": final_rms_list,
        "per_chain_is_nan": is_nan_list,
        "n_nan": int(sum(is_nan_list)),
        "n_iters_mean": n_iters_mean,
        "n_iters_sd": n_iters_sd,
        "time_mean": time_mean,
        "time_sd": time_sd,
        "n_converged": n_converged,
        "frac_converged": n_converged / total_chains,
        "D": D,
        "x64": X64,
    }
    logger.log_metrics(results)
    wandb.finish()


if __name__ == "__main__":
    main()
