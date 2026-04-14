"""
harness.py
Runs eight-schools experiments using hydra + wandb.

Options for alg are:
- deer
- quasi
- picard
- jacobi
- broyden
- accelerated_broyden

Sweepable parameters:
  S              : number of schools (8, 16, 32, ...)
  T              : chain length / number of Gibbs samples
  deer_k         : DEER / quasi-DEER damping (0 = no damping)
  bd_m           : Broyden-DEER rank
  bd_init        : Broyden-DEER init scheme (jac, identity, rescaled_identity, fresh_jac)
  bd_k           : Broyden-DEER damping (0 = no damping)
  acc_m          : Accelerated Broyden rank
  acc_init       : Accelerated Broyden init scheme (autodiff, identity, rescaled_identity)
  acc_k          : Accelerated Broyden damping (0 = no damping)
  acc_eps_gram   : Accelerated Broyden Gram regularization

Example usage:
  python harness.py alg=deer S=8 T=10000
  python harness.py -m alg=deer,quasi,picard,jacobi S=8,16,32 T=10000
  python harness.py alg=broyden bd_m=1,2,3,4 bd_init=jac,identity bd_k=0.0,0.5
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

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from dynamics.eight_schools import make_eight_schools
from algs.deer import deer_alg, picard_alg, jacobi_alg
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

    # set up dynamics
    deer_fxn, params, xs, D = make_eight_schools(S=S, seed=seed)

    # set up initial state and drivers — matches notebook exactly
    nu_N      = params["nu_N"]
    alpha_N   = params["alpha_N"]
    tausq_0   = params["tausq_0"]
    kappa_0   = params["kappa_0"]
    mu_0      = params["mu_0"]
    sigmasq_0 = params.get("sigmasq_0", 10.0)

    key = jr.PRNGKey(1313)
    key, init_key = jr.split(key)

    def sample_prior(key):
        key, *skeys = jr.split(key, 5)
        tausq    = tfd.InverseGamma(0.5 * nu_N, nu_N * tausq_0 * 0.5).sample(seed=skeys[0])
        mu       = tfd.Normal(mu_0, jnp.sqrt(tausq / kappa_0)).sample(seed=skeys[1])
        thetas   = tfd.Normal(mu, jnp.sqrt(tausq)).sample((S,), seed=skeys[2])
        sigmasqs = tfd.InverseGamma(0.5 * alpha_N, 0.5 * alpha_N * sigmasq_0).sample((S,), seed=skeys[3])
        return jnp.concatenate((thetas, sigmasqs, jnp.array([mu]), jnp.array([tausq])))

    initial_state = sample_prior(init_key)
    key, step_key = jr.split(key)
    initial_state = deer_fxn(initial_state, step_key)

    key, drivers_key = jr.split(key)
    driver_keys = jr.split(drivers_key, batch_size)
    drivers = jax.vmap(lambda k: jr.split(k, T))(driver_keys)  # (batch_size, T, 2)
    states_guess = jnp.broadcast_to(initial_state, (T, D))

    # define evaluation functions — pass drivers as argument (not closed over)
    # so JAX traces with abstract shapes rather than concrete T=10000 arrays
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
            full_trace=False, tol=tol,
        )
        return newton_steps

    def accelerated_broyden_eval(drivers):
        _, _, newton_steps, *_ = accelerated_broyden_deer_alg(
            deer_fxn, initial_state, states_guess, drivers,
            num_iters, m=cfg.acc_m, init=cfg.acc_init,
            k=cfg.acc_k, gamma=cfg.acc_gamma,
            eps_gram=cfg.acc_eps_gram, compress=cfg.acc_compress,
            full_trace=False, tol=tol,
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

    print(f"{alg} | S={S}, T={T}, D={D} | time: {time_avg:.3e}s | n_iters: {n_iters}")

    results = {
        "time": time_avg,
        "n_iters": n_iters,
        "D": D,
    }
    logger.log_metrics(results)
    wandb.finish()


if __name__ == "__main__":
    main()
