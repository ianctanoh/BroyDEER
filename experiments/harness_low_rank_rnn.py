"""
harness_low_rank_rnn.py
Runs low-rank RNN experiments using hydra + wandb.

Options for alg are:
- deer
- quasi
- picard
- jacobi
- broyden
- accelerated_broyden
- seq

Sweepable parameters:
  N              : number of neurons
  R              : rank of weight matrix
  T              : sequence length
  g              : gain
  sigma_noise    : noise scale
  deer_k         : DEER / quasi-DEER damping (0 = no damping)
  bd_m           : Broyden-DEER rank
  bd_init        : Broyden-DEER init scheme (jac, identity, rescaled_identity, low_rank, fresh_jac)
  bd_k           : Broyden-DEER damping (0 = no damping)
  acc_m          : Accelerated Broyden rank
  acc_init       : Accelerated Broyden init scheme (autodiff, identity, rescaled_identity)
  acc_k          : Accelerated Broyden damping (0 = no damping)
  acc_eps_gram   : Accelerated Broyden Gram regularization

Example usage:
  python harness_low_rank_rnn.py alg=deer N=500 R=3 T=1000
  python harness_low_rank_rnn.py -m alg=deer,quasi,accelerated_broyden N=500 R=3 T=1000
  python harness_low_rank_rnn.py alg=broyden bd_m=1,2,3 bd_init=rescaled_identity,low_rank
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

from dynamics.low_rank_rnn import LowRankRNN
from algs.deer import deer_alg, picard_alg, jacobi_alg
from algs.broyden_deer import broyden_deer_alg
from algs.accelerated_broyden import accelerated_broyden_deer_alg


@hydra.main(config_path="configs", config_name="harness_low_rank_rnn")
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

    logger = WandbLogger(project="broydeer_rnn", mode=mode)
    logger.log_hyperparams(OmegaConf.to_container(cfg))
    wandb.config.update(OmegaConf.to_container(cfg), allow_val_change=True)

    # set up dynamics
    key = jr.PRNGKey(seed)
    key, k1 = jr.split(key)
    rnn = LowRankRNN(N, R, key=k1, tau=tau, dt=dt, g=g, phi=phi)
    deer_fxn = rnn.deer_fxn

    # set up initial state and drivers
    initial_state = jnp.zeros(N)

    key, drivers_key = jr.split(key)
    driver_keys = jr.split(drivers_key, batch_size)
    drivers = jax.vmap(
        lambda k: sigma_noise * jr.normal(k, (T, N))
    )(driver_keys)  # (batch_size, T, N)

    key, guess_key = jr.split(key)
    states_guess = jr.normal(guess_key, (T, N)) * 0.1

    # define evaluation functions — pass drivers as argument (not closed over)
    # so JAX traces with abstract shapes rather than concrete arrays
    def seq_eval(drivers):
        def step(state, driver):
            new = deer_fxn(state, driver)
            return new, new
        _, samples = jax.lax.scan(step, initial_state, drivers)
        return samples[-1]

    def deer_eval(drivers):
        _, _, newton_steps, *_ = deer_alg(
            deer_fxn, initial_state, states_guess, drivers,
            num_iters, k=cfg.deer_k,
            full_trace=False, tol=tol,
        )
        return newton_steps

    def quasi_eval(drivers):
        _, _, newton_steps, *_ = deer_alg(
            deer_fxn, initial_state, states_guess, drivers,
            num_iters, quasi=True, k=cfg.deer_k,
            full_trace=False, tol=tol,
        )
        return newton_steps

    def picard_eval(drivers):
        _, _, newton_steps, *_ = picard_alg(
            deer_fxn, initial_state, states_guess, drivers,
            num_iters, full_trace=False, tol=tol,
        )
        return newton_steps

    def jacobi_eval(drivers):
        _, _, newton_steps, *_ = jacobi_alg(
            deer_fxn, initial_state, states_guess, drivers,
            num_iters, full_trace=False, tol=tol,
        )
        return newton_steps

    def broyden_eval(drivers):
        _, _, newton_steps, *_ = broyden_deer_alg(
            deer_fxn, initial_state, states_guess, drivers,
            num_iters, m=cfg.bd_m, init=cfg.bd_init,
            k=cfg.bd_k, gamma=cfg.bd_gamma,
            compress=cfg.acc_compress,
            full_trace=False, tol=tol,
            line_search=True, ls_c=1e-3,
        )
        return newton_steps

    def accelerated_broyden_eval(drivers):
        _, _, newton_steps, *_ = accelerated_broyden_deer_alg(
            deer_fxn, initial_state, states_guess, drivers,
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
        "picard": picard_eval,
        "jacobi": jacobi_eval,
        "broyden": broyden_eval,
        "accelerated_broyden": accelerated_broyden_eval,
    }
    if alg not in eval_fns:
        raise ValueError(f"Invalid algorithm: {alg}. Choose from {list(eval_fns.keys())}")

    use_highest = alg in ("deer", "quasi", "broyden", "accelerated_broyden")

    if use_highest:
        with jax.default_matmul_precision("highest"):
            fxn = jax.jit(jax.vmap(eval_fns[alg]))
            fxn.lower(drivers).compile()
    else:
        fxn = jax.jit(jax.vmap(eval_fns[alg]))
        fxn.lower(drivers).compile()

    # warmup (triggers compilation) + timed runs
    if use_highest:
        with jax.default_matmul_precision("highest"):
            for _ in range(nwarmups):
                result = fxn(drivers)
                jax.block_until_ready(result)
            t0 = time.time()
            for _ in range(nreps):
                result = fxn(drivers)
                jax.block_until_ready(result)
            t1 = time.time()
    else:
        for _ in range(nwarmups):
            result = fxn(drivers)
            jax.block_until_ready(result)
        t0 = time.time()
        for _ in range(nreps):
            result = fxn(drivers)
            jax.block_until_ready(result)
        t1 = time.time()

    time_avg = (t1 - t0) / nreps

    if alg == "seq":
        n_iters = T
    else:
        n_iters = float(jnp.mean(result))

    print(f"{alg} | N={N}, R={R}, T={T} | time: {time_avg:.3e}s | n_iters: {n_iters}")

    results = {
        "time": time_avg,
        "n_iters": n_iters,
        "N": N,
        "R": R,
    }
    logger.log_metrics(results)
    wandb.finish()


if __name__ == "__main__":
    main()
