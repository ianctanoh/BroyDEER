"""
nschools_tol_study.py

Is tol (RMS residual ||r||_2 / sqrt(T*D)) loose enough to matter for N-schools in
float32? For each parallel solver and tol, the returned trajectory is compared
with the float32 sequential Gibbs chain (same data, initial state and noise):

  - trajectory error per variable, relative to that variable's posterior sd
  - posterior summaries (mean, sd of every state coordinate after burn-in):
    mean difference / Monte Carlo standard error (batch means), and relative
    sd difference

The float32 sequential chain is also compared with the float64 sequential chain
(alg "seq_f32_vs_f64"). jax.random draws different noise in float64 from the same
keys, so this compares essentially independent chains: a seed-to-seed baseline for
the metrics, not float32 rounding error.

Chain setup and solver settings are those of harness_Nschools.py with
wandb_sweep_configs/Nschools_float32_tol1e-3.yaml (deer_k=0, quasi_k=0,
bd_k=0.3, acc_k=0.1, m=2, identity init, gamma=0, low_rank_exact,
lower bounds, num_iters=1000; Broyden without line search, Accelerated Broyden
with line search and safeguard step acc_ls_c=0.001),
run in float32.

Usage:
    JAX_PLATFORMS=cuda python nschools_tol_study.py [--S 10 50 100] [--T 10000 100000]
        [--chains 8] [--tols 1e-2 3e-3 1e-3] [--out PATH]
        [--save_chain 7 --traj_dir DIR]   # keep full trajectories of one chain for trace plots
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import argparse
import pickle
import time

import jax
jax.config.update("jax_enable_x64", False)          # float32 study
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.experimental import enable_x64
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from dynamics.eight_schools import make_eight_schools, make_gibbs_step
from algs.deer import deer_alg, get_residual, stopping_criterion
from algs.broyden_deer import broyden_deer_alg
from algs.accelerated_broyden import accelerated_broyden_deer_alg

parser = argparse.ArgumentParser()
parser.add_argument("--S", type=int, nargs="+", default=[10, 50, 100])
parser.add_argument("--T", type=int, nargs="+", default=[10000, 100000])
parser.add_argument("--chains", type=int, default=8)
parser.add_argument("--tols", type=float, nargs="+", default=[1e-2, 3e-3, 1e-3])
parser.add_argument("--algs", nargs="+",
                    default=["deer", "quasi", "broyden", "accelerated_broyden"])
parser.add_argument("--save_chain", type=int, default=None,
                    help="also save full float32 trajectories (sequential + each solver/tol) of this chain")
parser.add_argument("--traj_dir", default=None, help="directory for --save_chain .npz files")
parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "saved_files",
                                                  "nschools_tol_study_float32.pkl"))
args = parser.parse_args()

# ── settings of harness_Nschools.py / Nschools_float32_tol1e-3.yaml ──
SEED, N_S, TAUSQ_0, SIGMASQ_0 = 13, 20, 100.0, 10.0
NUM_ITERS = 1000
DEER_K, QUASI_K, BD_K, ACC_K = 0.0, 0.0, 0.3, 0.1
M, INIT, GAMMA, COMPRESS, EPS_GRAM = 2, "identity", 0.0, "low_rank_exact", 1e-6
BURNIN_FRAC = 0.1


def setup_chain(S, T, chain_idx):
    """Exactly harness_Nschools.setup_chain."""
    deer_fxn_i, params_i, xs_i, D_i = make_eight_schools(
        S=S, N_s=N_S, seed=SEED + chain_idx, tausq_0=TAUSQ_0, sigmasq_0=SIGMASQ_0)
    nu_N, alpha_N = params_i["nu_N"], params_i["alpha_N"]
    tausq_0_i, kappa_0, mu_0 = params_i["tausq_0"], params_i["kappa_0"], params_i["mu_0"]
    sigmasq_0_i = params_i.get("sigmasq_0", 10.0)

    def sample_prior(key):
        key, *skeys = jr.split(key, 5)
        tausq    = tfd.InverseGamma(0.5 * nu_N, nu_N * tausq_0_i * 0.5).sample(seed=skeys[0])
        mu       = tfd.Normal(mu_0, jnp.sqrt(tausq / kappa_0)).sample(seed=skeys[1])
        thetas   = tfd.Normal(mu, jnp.sqrt(tausq)).sample((S,), seed=skeys[2])
        sigmasqs = tfd.InverseGamma(0.5 * alpha_N, 0.5 * alpha_N * sigmasq_0_i).sample((S,), seed=skeys[3])
        return jnp.concatenate((thetas, sigmasqs, jnp.array([mu]), jnp.array([tausq])))

    key, init_key = jr.split(jr.PRNGKey(chain_idx))
    init_state = sample_prior(init_key)
    key, step_key = jr.split(key)
    init_state = deer_fxn_i(init_state, step_key)
    key, drivers_key = jr.split(key)
    drivers = jr.split(drivers_key, T)
    return xs_i, init_state, drivers, D_i


def gibbs_step(S, xs):
    return make_gibbs_step(S, N_S, xs, tausq_0=TAUSQ_0, sigmasq_0=SIGMASQ_0)


def make_solver(alg, S, tol, lower_bounds):
    """Same solver calls as harness_Nschools.py; returns (traj, iters, is_nan)."""
    def solve(xs, x0, guess, drivers):
        f = gibbs_step(S, xs)
        if alg == "deer":
            out = deer_alg(f, x0, guess, drivers, NUM_ITERS, k=DEER_K,
                           full_trace=False, tol=tol, stopping="rms")
        elif alg == "quasi":
            out = deer_alg(f, x0, guess, drivers, NUM_ITERS, quasi=True, k=QUASI_K,
                           full_trace=False, tol=tol, stopping="rms")
        elif alg == "broyden":
            out = broyden_deer_alg(f, x0, guess, drivers, NUM_ITERS, m=M, init=INIT,
                                   k=BD_K, gamma=GAMMA, sequential_solve=False,
                                   compress=COMPRESS, full_trace=False, tol=tol,
                                   lower_bounds=lower_bounds, line_search=False)
        elif alg == "accelerated_broyden":
            out = accelerated_broyden_deer_alg(f, x0, guess, drivers, NUM_ITERS, m=M, init=INIT,
                                               k=ACC_K, gamma=GAMMA, eps_gram=EPS_GRAM,
                                               compress=COMPRESS, full_trace=False, tol=tol,
                                               lower_bounds=lower_bounds, line_search=True,
                                               ls_c=1e-3)
        else:
            raise ValueError(alg)
        _, traj, iters, is_nan, *_ = out
        return traj, iters, is_nan
    return jax.jit(solve)


def sequential(S):
    def run(xs, x0, drivers):
        f = gibbs_step(S, xs)
        _, traj = jax.lax.scan(lambda s, d: (f(s, d), f(s, d)), x0, drivers)
        return traj
    return jax.jit(run)


def batch_means_mcse(x):
    """MCSE of the mean of each column of x (n, D) by batch means, sqrt(n) batches."""
    n = x.shape[0]
    B = int(np.sqrt(n))
    L = n // B
    bm = x[:B * L].reshape(B, L, -1).mean(axis=1)
    return bm.std(axis=0, ddof=1) / np.sqrt(B)


def compare(traj, ref):
    """Accuracy of traj (T, D) against the reference chain ref (T, D), both numpy."""
    b = int(BURNIN_FRAC * ref.shape[0])
    r, x = ref[b:].astype(np.float64), traj[b:].astype(np.float64)
    sd = r.std(axis=0) + 1e-30
    err = np.abs(x - r) / sd                            # per entry, in posterior sds
    mcse = batch_means_mcse(r) + 1e-30
    mean_z = np.abs(x.mean(axis=0) - r.mean(axis=0)) / mcse
    sd_rel = np.abs(x.std(axis=0) - r.std(axis=0)) / sd
    return {
        "traj_err_max_sd": float(np.nanmax(err)),        # worst entry, in posterior sds
        "traj_err_rms_sd": float(np.sqrt(np.nanmean(err ** 2))),
        "mean_err_max_mcse": float(np.nanmax(mean_z)),   # worst coordinate, in MCSEs
        "mean_err_med_mcse": float(np.nanmedian(mean_z)),
        "sd_rel_err_max": float(np.nanmax(sd_rel)),
    }


rows = []
t_start = time.time()
print(f"Device: {jax.devices()} | float32 | S={args.S} T={args.T} chains={args.chains} "
      f"tols={args.tols} algs={args.algs}", flush=True)

for S in args.S:
    lower_bounds = jnp.concatenate([jnp.full(S, -jnp.inf), jnp.full(S, 1e-6),
                                    jnp.array([-jnp.inf]), jnp.array([1e-6])])
    seq32 = sequential(S)
    with enable_x64():
        seq64 = sequential(S)
    for T in args.T:
        chains = [setup_chain(S, T, i) for i in range(args.chains)]
        D = chains[0][3]
        refs = []
        for i, (xs, x0, drivers, _) in enumerate(chains):
            ref32 = np.asarray(seq32(xs, x0, drivers))
            with enable_x64():
                ref64 = np.asarray(seq64(jnp.asarray(xs, jnp.float64),
                                         jnp.asarray(x0, jnp.float64), drivers))
            refs.append(ref32)
            if args.save_chain == i:
                os.makedirs(args.traj_dir, exist_ok=True)
                np.savez_compressed(os.path.join(args.traj_dir, f"traj_S{S}_T{T}_chain{i}_seq.npz"),
                                    traj=ref32)
            rows.append({"S": S, "T": T, "D": D, "chain": i, "alg": "seq_f32_vs_f64",
                         "tol": np.nan, "n_iters": T, "converged": True, "is_nan": False,
                         "final_rms": 0.0, **compare(ref32, ref64)})
        for alg in args.algs:
            for tol in args.tols:
                solver = make_solver(alg, S, tol, lower_bounds)
                with jax.default_matmul_precision("highest"):
                    for i, (xs, x0, drivers, _) in enumerate(chains):
                        guess = jnp.broadcast_to(x0, (T, D))
                        traj, iters, is_nan = solver(xs, x0, guess, drivers)
                        traj = np.asarray(traj)
                        if args.save_chain == i:
                            np.savez_compressed(os.path.join(
                                args.traj_dir, f"traj_S{S}_T{T}_chain{i}_{alg}_tol{tol:g}.npz"), traj=traj)
                        rms = float(stopping_criterion(get_residual(
                            gibbs_step(S, xs), x0, jnp.asarray(traj), drivers)))
                        rows.append({"S": S, "T": T, "D": D, "chain": i, "alg": alg, "tol": tol,
                                     "n_iters": int(iters), "converged": int(iters) < NUM_ITERS,
                                     "is_nan": bool(is_nan), "final_rms": rms,
                                     **compare(traj, refs[i])})
                last = rows[-args.chains:]
                print(f"S={S:3d} T={T:6d} {alg:20s} tol={tol:.0e} | conv "
                      f"{sum(r['converged'] for r in last)}/{len(last)} | iters med "
                      f"{np.median([r['n_iters'] for r in last]):6.1f} | traj err max "
                      f"{max(r['traj_err_max_sd'] for r in last):.2e} sd | mean err max "
                      f"{max(r['mean_err_max_mcse'] for r in last):.2e} MCSE | "
                      f"[{time.time() - t_start:.0f}s]", flush=True)
        ref_rows = [r for r in rows if r["alg"] == "seq_f32_vs_f64" and r["S"] == S and r["T"] == T]
        print(f"S={S:3d} T={T:6d} {'seq f32 vs f64':20s}            | traj err max "
              f"{max(r['traj_err_max_sd'] for r in ref_rows):.2e} sd | mean err max "
              f"{max(r['mean_err_max_mcse'] for r in ref_rows):.2e} MCSE", flush=True)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "wb") as fh:
    pickle.dump({"config": vars(args), "rows": rows}, fh)
print(f"Saved {len(rows)} rows to {args.out} [{time.time() - t_start:.0f}s]")
