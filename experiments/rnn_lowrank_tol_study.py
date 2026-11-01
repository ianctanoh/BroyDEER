"""
rnn_lowrank_tol_study.py

Is the stopping tolerance (RMS residual ||r||_2 / sqrt(T*D)) tight enough for the low-rank RNN
experiments (harness_rnn_lowrank.py)? For each parallel solver and tol, the returned trajectory
is compared with the float32 sequential rollout of the same chain (same weights, initial state
and noise):

  - trajectory error per entry, relative to that neuron's sd over time (after T/4 burn-in):
    worst entry and RMS
  - final RMS residual of the returned trajectory

Baseline: the float32 sequential rollout vs the float64 sequential rollout of the *same* inputs
cast to float64 (alg "seq_f32_vs_f64"): pure float32 rounding error of the reference.

Models, chain construction, solver settings and the damping table are those of
harness_rnn_lowrank.py (float32, highest matmul precision, compress="rand_rr", line search
c=1e-3 for Accelerated Broyden only,
gamma = 1 - dt/tau, max 1000 iterations).

Usage:
    JAX_PLATFORMS=cuda python rnn_lowrank_tol_study.py [--configs std:100:0 mlp:100:8]
        [--dt 3.0 0.5] [--T 10000] [--chains 8] [--tols 1e-2 1e-3 1e-4] [--out PATH]
        [--save_chain 0 --traj_dir DIR]   # keep full trajectories of one chain for trace plots
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import argparse
import pickle
import time

import jax
jax.config.update("jax_enable_x64", False)                   # float32 study
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.experimental import enable_x64

from dynamics.low_rank_rnn import LowRankRNN
from algs.deer import deer_alg, jacobi_alg, get_residual, stopping_criterion
from algs.broyden_deer import broyden_deer_alg
from algs.accelerated_broyden import accelerated_broyden_deer_alg

# Same as harness_rnn_lowrank.py
DAMPING = {
    3.0: dict(newton=0.0, quasi=0.8, broyden=0.4, accel=0.1),
    0.5: dict(newton=0.0, quasi=0.1, broyden=0.2, accel=0.2),
}
SEED, R, TAU, G, SIG, BETA = 42, 3, 10.0, 2.0, 0.1, 0.5
NUM_ITERS, M_HIST, EPS_GRAM, LS_C = 1000, 3, 1e-6, 1e-3
BURNIN_FRAC = 0.25
f32 = jnp.float32

parser = argparse.ArgumentParser()
parser.add_argument("--configs", nargs="+", default=["std:100:0", "mlp:100:8"], help="model:D:L")
parser.add_argument("--dt", type=float, nargs="+", default=[3.0, 0.5])
parser.add_argument("--T", type=int, nargs="+", default=[10000])
parser.add_argument("--chains", type=int, default=8)
parser.add_argument("--tols", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4])
parser.add_argument("--methods", nargs="+", default=["newton", "quasi", "jacobi", "broyden", "accel"])
parser.add_argument("--save_chain", type=int, default=None,
                    help="also save full trajectories (sequential + each method/tol) of this chain")
parser.add_argument("--traj_dir", default=None, help="directory for --save_chain .npz files")
parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "saved_files",
                                                  "rnn_lowrank_tol_study.pkl"))
args = parser.parse_args()


def make_params(D, L, seed):
    k_rnn, k_init, k_drv, k_guess = jr.split(jr.PRNGKey(seed), 4)
    rnn = LowRankRNN(D, R, key=k_rnn, tau=TAU, g=G, phi="tanh")
    Ws = []
    if L > 0:
        km = jr.split(jr.fold_in(jr.PRNGKey(seed), 1000), L + 1)
        Ws = [jr.normal(km[0], (D, R), dtype=f32) / np.sqrt(R)]
        Ws += [jr.normal(km[i], (D, D), dtype=f32) / np.sqrt(D) for i in range(1, L)]
        Ws += [jr.normal(km[L], (R, D), dtype=f32) / np.sqrt(D)]
    return dict(M=rnn.M, Nt=rnn.Nt, Ws=Ws), k_init, k_drv, k_guess


def make_f(P, dt):
    a, dtt = 1.0 - dt / TAU, dt / TAU
    def g(z):
        if not P["Ws"]:
            return z
        h = z
        for Wl in P["Ws"][:-1]:
            h = jnp.tanh(Wl @ h)
        return z + BETA * (P["Ws"][-1] @ h)
    def f(x, eps):
        return a * x + dtt * (P["M"] @ g(P["Nt"].T @ jnp.tanh(x))) + eps
    return f


def sequential(dt):
    def run(P, x0, d):
        f = make_f(P, dt)
        return jax.lax.scan(lambda s, u: (f(s, u),) * 2, x0, d)[1]
    return jax.jit(run)


def make_solver(method, dt, k, tol):
    gamma = 1.0 - dt / TAU
    def solve(P, x0, guess, d):
        f = make_f(P, dt)
        if method == "newton":
            out = deer_alg(f, x0, guess, d, NUM_ITERS, k=k["newton"], full_trace=False,
                           tol=tol, stopping="rms")
        elif method == "quasi":
            out = deer_alg(f, x0, guess, d, NUM_ITERS, quasi=True, qmem_efficient=False,
                           k=k["quasi"], full_trace=False, tol=tol, stopping="rms")
        elif method == "jacobi":
            out = jacobi_alg(f, x0, guess, d, NUM_ITERS, full_trace=False, tol=tol, stopping="rms")
        elif method == "broyden":
            out = broyden_deer_alg(f, x0, guess, d, NUM_ITERS, m=M_HIST, init="identity",
                                   k=k["broyden"], gamma=gamma, compress="rand_rr",
                                   full_trace=False, tol=tol, line_search=False)
        else:
            out = accelerated_broyden_deer_alg(f, x0, guess, d, NUM_ITERS, m=M_HIST,
                                               init="identity", k=k["accel"], gamma=gamma,
                                               eps_gram=EPS_GRAM, compress="rand_rr",
                                               qmem_efficient=True, full_trace=False, tol=tol,
                                               line_search=True, ls_c=LS_C)
        traj = out[1]
        return traj, out[2], stopping_criterion(get_residual(f, x0, traj, d))
    return jax.jit(solve)


def compare(traj, ref):
    """Error of traj (T, D) against ref (T, D), relative to each neuron's sd in ref (after burn-in)."""
    b = int(BURNIN_FRAC * ref.shape[0])
    r, x = ref[b:].astype(np.float64), traj[b:].astype(np.float64)
    sd = r.std(axis=0) + 1e-30
    err = np.abs(x - r) / sd
    return {"err_max_sd": float(np.nanmax(err)), "err_rms_sd": float(np.sqrt(np.nanmean(err ** 2)))}


rows = []
t_start = time.time()
print(f"{jax.devices()} | float32 | configs {args.configs} dt={args.dt} T={args.T} "
      f"chains={args.chains} tols={args.tols} methods={args.methods}", flush=True)
for cfg in args.configs:
    model, D, L = cfg.split(":")
    D, L = int(D), int(L) if model == "mlp" else 0
    for dt in args.dt:
        k = DAMPING[float(dt)]
        seq32 = sequential(dt)
        with enable_x64():
            seq64 = sequential(dt)
        for T in args.T:
            chains, refs, s = [], [], 0
            while len(chains) < args.chains and s < 10 * args.chains:
                P, k_init, k_drv, k_guess = make_params(D, L, SEED + s)
                s += 1
                x0 = jr.normal(k_init, (D,), dtype=f32) * 0.1
                d = SIG * jr.normal(k_drv, (T, D), dtype=f32)
                guess = jr.normal(k_guess, (T, D), dtype=f32) * 0.1
                ref32 = seq32(P, x0, d)
                ph = jnp.tanh(ref32[T // 4:])
                if int(jnp.sum(jnp.mean(jnp.abs(ph), 0) > 0.8)) or \
                   int(jnp.sum(jnp.mean(1 - ph ** 2, 0) < 0.1)):
                    continue
                ref32 = np.asarray(ref32)
                with enable_x64():
                    c64 = lambda z: jnp.asarray(np.asarray(z), jnp.float64)
                    P64 = jax.tree_util.tree_map(c64, P)
                    ref64 = np.asarray(seq64(P64, c64(x0), c64(d)))
                i = len(chains)
                chains.append((SEED + s - 1, P, x0, guess, d))
                refs.append(ref32)
                rows.append(dict(model=model, D=D, L=L, dt=dt, T=T, chain=i, seed=SEED + s - 1,
                                 method="seq_f32_vs_f64", tol=np.nan, n_iters=T, converged=True,
                                 final_rms=0.0, **compare(ref32, ref64)))
                if args.save_chain == i:
                    os.makedirs(args.traj_dir, exist_ok=True)
                    np.savez_compressed(os.path.join(
                        args.traj_dir, f"traj_{model}_D{D}_dt{dt:g}_T{T}_chain{i}_seq.npz"), traj=ref32)
            base = [r for r in rows[-len(chains):]]
            print(f"\n=== {model} D={D} L={L} dt={dt} T={T}: {len(chains)} healthy of {s} | damping {k} | "
                  f"seq f32 vs f64: err max {max(r['err_max_sd'] for r in base):.2e} sd, "
                  f"rms {max(r['err_rms_sd'] for r in base):.2e} sd [{time.time() - t_start:.0f}s] ===",
                  flush=True)
            for method in args.methods:
                for tol in args.tols:
                    solver = make_solver(method, dt, k, tol)
                    out = []
                    for i, (seed, P, x0, guess, d) in enumerate(chains):
                        traj, it, rms = solver(P, x0, guess, d)
                        traj = np.asarray(traj)
                        it = int(it)
                        row = dict(model=model, D=D, L=L, dt=dt, T=T, chain=i, seed=seed, method=method,
                                   k=k.get(method), tol=tol, n_iters=it, converged=it < NUM_ITERS,
                                   final_rms=float(rms), **compare(traj, refs[i]))
                        rows.append(row)
                        out.append(row)
                        if args.save_chain == i:
                            np.savez_compressed(os.path.join(
                                args.traj_dir, f"traj_{model}_D{D}_dt{dt:g}_T{T}_chain{i}_{method}_tol{tol:g}.npz"),
                                traj=traj)
                    conv = [r for r in out if r["converged"]]
                    em = max(r["err_max_sd"] for r in conv) if conv else float("nan")
                    er = max(r["err_rms_sd"] for r in conv) if conv else float("nan")
                    print(f"  {method:8s} tol={tol:.0e} | conv {len(conv)}/{len(out)} | iters med "
                          f"{np.median([r['n_iters'] for r in out]):6.1f} | traj err (converged) max "
                          f"{em:.2e} sd, rms {er:.2e} sd [{time.time() - t_start:.0f}s]", flush=True)
                    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                    with open(args.out, "wb") as fh:
                        pickle.dump({"config": vars(args), "damping": DAMPING, "rows": rows}, fh)
print(f"\nSaved {len(rows)} rows to {args.out} [{time.time() - t_start:.0f}s]")
