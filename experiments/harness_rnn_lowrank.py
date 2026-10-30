"""
harness_rnn_lowrank.py

Low-rank RNN experiments (proof of concept: exact diagonal + low-rank Jacobian, systems of
decreasing predictability, cheap and expensive steps). Sequential vs parallel solvers.

Models (D neurons, rank R=3, tanh, tau=10, gain g=2, noise sigma=0.1 per step):
  std : x_{t+1} = a x_t + (dt/tau) M N^T tanh(x_t) + eps_t
  mlp : x_{t+1} = a x_t + (dt/tau) M g(N^T tanh x_t) + eps_t,   g(z) = z + beta * MLP(z)
        MLP: R -> D -> ... -> D -> R (L hidden layers of width D, tanh), beta = 0.5
with a = 1 - dt/tau. The Jacobian is exactly a*I + rank R for both models; the MLP makes each
step expensive (L dependent D x D layers). Smaller dt = longer memory (less predictable).

Chains: seeds 42, 43, ... ; float32 draws of M, N, the initial state (0.1 N(0,1)), the noise
and the initial guess (0.1 N(0,1)); a seed is skipped if its sequential trajectory saturates
(after T/4 burn-in, some neuron with mean |tanh| > 0.8 or mean tanh' < 0.1).

Methods (float32, highest matmul precision, RMS stopping ||r||_2/sqrt(T*D) < tol = 1e-4,
at most 1000 iterations):
  seq      : sequential rollout (lax.scan)
  newton   : DEER (full Jacobian),                        damping k_newton
  quasi    : quasi-DEER (exact diagonal of the Jacobian), damping k_quasi
  jacobi   : Jacobi / fixed-point iteration (no Jacobian; = quasi-DEER with k=1)
  broyden  : Parallel Broyden,            m=3, diagonal init gamma = a (exact diagonal),
             compress="rand_rr", line search ls_c=1e-3,   damping k_broyden
  accel    : Accelerated Parallel Broyden, same + eps_gram=1e-6, damping k_accel
Damping defaults depend on dt (table DAMPING below): tuned per setting by minimizing the median
number of iterations over all chains (non-converged chains counted at the cap), summed over
std D in {50,100,200} and mlp (L=8) D in {100,200}, T in {1e4,1e5} (quasi-DEER: T=1e4), 32 chains.

Timing: each method is compiled and warmed up on the first chain, then every chain is solved
twice and the mean wall-clock time is recorded. Per-chain iterations and times are saved.

Usage (default arguments = the full grid):
    JAX_PLATFORMS=cuda python harness_rnn_lowrank.py
    JAX_PLATFORMS=cuda python harness_rnn_lowrank.py --configs std:100:0 mlp:200:8 --dt 0.5 \
        --T 10000 --methods seq accel newton --chains 8 --out PATH
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import argparse
import pickle
import time

import jax
jax.config.update("jax_enable_x64", False)                   # float32
jax.config.update("jax_default_matmul_precision", "highest")  # no TF32 on the GPU
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from dynamics.low_rank_rnn import LowRankRNN
from algs.deer import deer_alg, jacobi_alg
from algs.broyden_deer import broyden_deer_alg
from algs.accelerated_broyden import accelerated_broyden_deer_alg

# Damping k per time step dt (tau = 10), tuned as described in the module docstring.
DAMPING = {
    3.0: dict(newton=0.0, quasi=0.8, broyden=0.4, accel=0.1),
    0.5: dict(newton=0.0, quasi=0.4, broyden=0.2, accel=0.2),
}

parser = argparse.ArgumentParser()
parser.add_argument("--configs", nargs="+",
                    default=["std:50:0", "std:100:0", "std:200:0", "mlp:100:8", "mlp:200:8"],
                    help="model:D:L (L = MLP depth, ignored for std)")
parser.add_argument("--dt", type=float, nargs="+", default=[3.0, 0.5])
parser.add_argument("--T", type=int, nargs="+", default=[10000, 100000])
parser.add_argument("--methods", nargs="+",
                    default=["seq", "newton", "quasi", "jacobi", "broyden", "accel"])
parser.add_argument("--chains", type=int, default=32)
parser.add_argument("--k_newton", type=float, default=None, help="override DAMPING[dt]")
parser.add_argument("--k_quasi", type=float, default=None)
parser.add_argument("--k_broyden", type=float, default=None)
parser.add_argument("--k_accel", type=float, default=None)
parser.add_argument("--no_timing", action="store_true", help="one untimed solve per chain")
parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "saved_files",
                                                  "rnn_lowrank_results.pkl"))
args = parser.parse_args()

SEED, R, TAU, G, SIG, BETA = 42, 3, 10.0, 2.0, 0.1, 0.5
TOL, NUM_ITERS, M_HIST, EPS_GRAM, LS_C = 1e-4, 1000, 3, 1e-6, 1e-3
f32 = jnp.float32


def damping(dt):
    """Damping per method for this dt (table DAMPING, command-line overrides take priority)."""
    k = dict(DAMPING.get(float(dt), {}))
    for name in ("newton", "quasi", "broyden", "accel"):
        override = getattr(args, f"k_{name}")
        if override is not None:
            k[name] = override
    needed = {"newton", "quasi", "broyden", "accel"} & set(args.methods)
    missing = needed - set(k)
    if missing:
        raise ValueError(f"no tuned damping for dt={dt}: pass --k_{' --k_'.join(sorted(missing))}")
    return k


def make_params(D, L, seed):
    """Weights of one chain. M, N do not depend on dt; the MLP is drawn from a separate key."""
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
            return z                                    # standard low-rank RNN
        h = z
        for Wl in P["Ws"][:-1]:
            h = jnp.tanh(Wl @ h)
        return z + BETA * (P["Ws"][-1] @ h)
    def f(x, eps):
        return a * x + dtt * (P["M"] @ g(P["Nt"].T @ jnp.tanh(x))) + eps
    return f


def make_chains(D, L, T, dt, n):
    seq = jax.jit(lambda P, x0, d: jax.lax.scan(
        lambda s, u: (make_f(P, dt)(s, u),) * 2, x0, d)[1])
    chains, s = [], 0
    while len(chains) < n and s < 10 * n:
        P, k_init, k_drv, k_guess = make_params(D, L, SEED + s)
        s += 1
        x0 = jr.normal(k_init, (D,), dtype=f32) * 0.1
        d = SIG * jr.normal(k_drv, (T, D), dtype=f32)
        guess = jr.normal(k_guess, (T, D), dtype=f32) * 0.1
        ph = jnp.tanh(seq(P, x0, d)[T // 4:])
        if int(jnp.sum(jnp.mean(jnp.abs(ph), 0) > 0.8)) or \
           int(jnp.sum(jnp.mean(1 - ph ** 2, 0) < 0.1)):
            continue
        chains.append((SEED + s - 1, P, x0, guess, d))
    return chains, s


def make_solver(method, dt, k):
    gamma = 1.0 - dt / TAU                                # exact diagonal part of the Jacobian
    def solve(P, x0, guess, d):
        f = make_f(P, dt)
        if method == "seq":
            traj = jax.lax.scan(lambda s, u: (f(s, u),) * 2, x0, d)[1]
            return jnp.int32(d.shape[0]), traj[-1]
        if method == "newton":
            out = deer_alg(f, x0, guess, d, NUM_ITERS, k=k["newton"], full_trace=False,
                           tol=TOL, stopping="rms")
        elif method == "quasi":
            out = deer_alg(f, x0, guess, d, NUM_ITERS, quasi=True, qmem_efficient=False,
                           k=k["quasi"], full_trace=False, tol=TOL, stopping="rms")
        elif method == "jacobi":
            out = jacobi_alg(f, x0, guess, d, NUM_ITERS, full_trace=False, tol=TOL, stopping="rms")
        elif method == "broyden":
            out = broyden_deer_alg(f, x0, guess, d, NUM_ITERS, m=M_HIST, init="identity",
                                   k=k["broyden"], gamma=gamma, compress="rand_rr",
                                   full_trace=False, tol=TOL, line_search=True, ls_c=LS_C)
        elif method == "accel":
            out = accelerated_broyden_deer_alg(f, x0, guess, d, NUM_ITERS, m=M_HIST,
                                               init="identity", k=k["accel"], gamma=gamma,
                                               eps_gram=EPS_GRAM, compress="rand_rr",
                                               qmem_efficient=True, full_trace=False, tol=TOL,
                                               line_search=True, ls_c=LS_C)
        else:
            raise ValueError(method)
        return out[2], out[1][-1]                       # iterations, last state
    return jax.jit(solve)


rows = []
t_start = time.time()
print(f"{jax.devices()} | float32, matmul {jax.config.jax_default_matmul_precision} | "
      f"configs {args.configs} dt={args.dt} T={args.T} chains={args.chains} "
      f"methods={args.methods} timing={not args.no_timing}", flush=True)
for cfg in args.configs:
    model, D, L = cfg.split(":")
    D, L = int(D), int(L) if model == "mlp" else 0
    for dt in args.dt:
        k = damping(dt)
        for T in args.T:
            chains, tried = make_chains(D, L, T, dt, args.chains)
            print(f"\n=== {model} D={D} L={L} dt={dt} (a=gamma={1 - dt / TAU:.2f}) T={T}: "
                  f"{len(chains)} healthy chains of {tried} seeds | damping {k} "
                  f"[{time.time() - t_start:.0f}s] ===", flush=True)
            for method in args.methods:
                solver = make_solver(method, dt, k)
                its, times, err = [], [], None
                for ci, (seed, P, x0, guess, d) in enumerate(chains):
                    try:
                        if ci == 0 and not args.no_timing:     # compile + warm up once
                            jax.block_until_ready(solver(P, x0, guess, d))
                        reps = 1 if args.no_timing else 2
                        t0 = time.time()
                        for _ in range(reps):
                            it, last = solver(P, x0, guess, d)
                            jax.block_until_ready(last)
                        times.append((time.time() - t0) / reps)
                        its.append(int(it))
                    except Exception as e:                      # e.g. Newton out of memory
                        err = str(e).split("\n")[0][:200]
                        break
                row = dict(model=model, D=D, L=L, dt=dt, T=T, method=method,
                           k=k.get(method), seeds=[c[0] for c in chains], seeds_tried=tried)
                if err:
                    row["error"] = err
                    print(f"  {method:8s} ERROR: {err[:100]}", flush=True)
                else:
                    its, times = np.array(its), np.array(times)
                    # sequential always "converges" (its count is T, not an iteration count)
                    c = np.ones(len(its), bool) if method == "seq" else its < NUM_ITERS
                    row.update(n_iters=its, times=None if args.no_timing else times)
                    tc = times[c] if c.any() else np.array([np.nan])
                    print(f"  {method:8s} k={str(k.get(method, '-')):4s} | failed {(~c).sum():2d}/"
                          f"{len(its)} | median iters (all) {np.median(its):7.1f} | median time "
                          f"(converged) {np.median(tc):.4f} s [{time.time() - t_start:.0f}s]",
                          flush=True)
                rows.append(row)
                os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                with open(args.out, "wb") as fh:
                    pickle.dump({"args": vars(args), "damping": DAMPING, "tol": TOL,
                                 "num_iters": NUM_ITERS, "rows": rows}, fh)
print(f"\nSaved {len(rows)} rows to {args.out} [{time.time() - t_start:.0f}s]")
