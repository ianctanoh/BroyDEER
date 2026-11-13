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
  quasi    : quasi-DEER with the exact diagonal (full Jacobian by reverse-mode autodiff), damping
             k_quasi
  quasi_hutch : quasi-DEER with a Hutchinson estimate of the diagonal (--quasi_probes Rademacher
             JVPs per step, default 1), damping k_quasi (not run by default; appendix). Accurate
             here because the leak a dominates the diagonal: 5-15% more iterations than exact.
  jacobi   : Jacobi / fixed-point iteration (no Jacobian; = quasi-DEER with k=1; not run by default)
  broyden  : Parallel Broyden,            m=3, diagonal init gamma = a (exact diagonal),
             compress="rand_rr", no line search (--ls_broyden 1: on), damping k_broyden
  accel    : Accelerated Parallel Broyden, same + eps_gram=1e-6, line search with c = ls_c = 1e-3
             (--ls_c; --ls_accel 0: off), damping k_accel
Line search: off for Broyden (same failures and median iterations with or without it; mean
iterations ~4% lower with it), on for Accelerated Broyden (without it ~47% of chains fail;
c tested in {1e-4, 1e-3, 1e-2, 0.1, 0.5}: 1e-4 equivalent to 1e-3, >= 1e-2 much worse).
Damping defaults depend on dt (table DAMPING below): tuned per setting by minimizing the median
number of iterations over all chains (non-converged chains counted at the cap), summed over
std D in {50,100,200} and mlp (L=8) D in {100,200}, T in {1e4,1e5} (quasi-DEER: T=1e4, exact
diagonal; the same k is used for quasi_hutch), 32 chains. T=1e3 uses the same values (not tuned).

Timing (as harness_Nschools.py): one compiled solver per (setting, method), shared by all
chains; per chain, --nwarmups untimed solves then --nreps timed solves (default 3 and 3), and the
mean wall-clock time is recorded (a chain whose first solve does not converge is not solved
again and gets time NaN: runtimes are over converged chains). Per-chain iterations and times are
saved, with the GPU type.
A method that fails on a setting (e.g. out of memory) is saved with its error. Chain inputs are
regenerated from their seeds when needed (same values; one chain on the GPU at a time). The output is rewritten after each (setting, method);
--resume skips the (setting, method) pairs already in --out (for preemptible / requeued jobs).

Usage (default arguments = the full grid of the paper):
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
    0.5: dict(newton=0.0, quasi=0.1, broyden=0.2, accel=0.2),
}

parser = argparse.ArgumentParser()
parser.add_argument("--configs", nargs="+",
                    default=["std:100:0", "std:200:0", "mlp:100:8", "mlp:200:8"],
                    help="model:D:L (L = MLP depth, ignored for std)")
parser.add_argument("--dt", type=float, nargs="+", default=[3.0, 0.5])
parser.add_argument("--T", type=int, nargs="+", default=[1000, 10000, 100000])
parser.add_argument("--methods", nargs="+",
                    default=["seq", "newton", "quasi", "broyden", "accel"])
parser.add_argument("--chains", type=int, default=32)
parser.add_argument("--k_newton", type=float, default=None, help="override DAMPING[dt]")
parser.add_argument("--k_quasi", type=float, default=None)
parser.add_argument("--k_broyden", type=float, default=None)
parser.add_argument("--k_accel", type=float, default=None)
parser.add_argument("--ls_broyden", type=int, default=0, help="line search for Parallel Broyden (1 on, 0 off; default off)")
parser.add_argument("--ls_accel", type=int, default=1, help="line search for Accelerated Broyden (1 on, 0 off)")
parser.add_argument("--ls_c", type=float, default=1e-3, help="line-search sufficient-decrease constant c")
parser.add_argument("--quasi_probes", type=int, default=1, help="Rademacher probes per step for quasi_hutch")
parser.add_argument("--resume", action="store_true", help="skip (setting, method) pairs already saved in --out")
parser.add_argument("--nwarmups", type=int, default=3, help="untimed solves per chain before timing")
parser.add_argument("--nreps", type=int, default=3, help="timed solves per chain (mean is recorded)")
parser.add_argument("--x64", action="store_true", help="float64 throughout (ablations; paper runs use float32)")
parser.add_argument("--scan_tree", default="blelloch", choices=["blelloch", "hillis_steele"],
                    help="prefix-scan tree of the Broyden solvers (ablation)")
parser.add_argument("--compress", default="rand_rr", choices=["rand_rr", "rand", "qr_svd"],
                    help="rank compression of the Broyden solvers (ablation)")
parser.add_argument("--sequential_solve", action="store_true",
                    help="Broyden solvers: solve the linearized recurrence by sequential forward substitution (ablation)")
parser.add_argument("--no_timing", action="store_true", help="one untimed solve per chain")
parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "saved_files",
                                                  "rnn_lowrank_results.pkl"))
args = parser.parse_args()
jax.config.update("jax_enable_x64", bool(args.x64))           # float32 unless --x64
jax.config.update("jax_default_matmul_precision", "highest")  # no TF32 on the GPU

SEED, R, TAU, G, SIG, BETA = 42, 3, 10.0, 2.0, 0.1, 0.5
TOL, NUM_ITERS, M_HIST, EPS_GRAM = 1e-4, 1000, 3, 1e-6
LS_C = args.ls_c                                              # line search c (default 1e-3)
f32 = jnp.float64 if args.x64 else jnp.float32                # dtype of all draws
BASE = dict(quasi_hutch="quasi")                              # damping-table entry per method


def damping(dt):
    """Damping per method for this dt (table DAMPING, command-line overrides take priority)."""
    k = dict(DAMPING.get(float(dt), {}))
    for name in ("newton", "quasi", "broyden", "accel"):
        override = getattr(args, f"k_{name}")
        if override is not None:
            k[name] = override
    needed = {"newton", "quasi", "broyden", "accel"} & {BASE.get(m, m) for m in args.methods}
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


def chain_inputs(keys, D, T):
    """Initial state, initial guess and noise of a chain, from its keys (regenerated when needed)."""
    k_init, k_drv, k_guess = keys
    x0 = jr.normal(k_init, (D,), dtype=f32) * 0.1
    guess = jr.normal(k_guess, (T, D), dtype=f32) * 0.1
    d = SIG * jr.normal(k_drv, (T, D), dtype=f32)
    return x0, guess, d


def make_chains(D, L, T, dt, n):
    """First n seeds whose sequential trajectory does not saturate: (seed, P, keys) per chain."""
    seq = jax.jit(lambda P, x0, d: jax.lax.scan(
        lambda s, u: (make_f(P, dt)(s, u),) * 2, x0, d)[1])
    chains, s = [], 0
    while len(chains) < n and s < 10 * n:
        P, k_init, k_drv, k_guess = make_params(D, L, SEED + s)
        s += 1
        x0, _, d = chain_inputs((k_init, k_drv, k_guess), D, T)
        ph = jnp.tanh(seq(P, x0, d)[T // 4:])
        if int(jnp.sum(jnp.mean(jnp.abs(ph), 0) > 0.8)) or \
           int(jnp.sum(jnp.mean(1 - ph ** 2, 0) < 0.1)):
            continue
        chains.append((SEED + s - 1, P, (k_init, k_drv, k_guess)))
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
        elif method == "quasi_hutch":
            out = deer_alg(f, x0, guess, d, NUM_ITERS, quasi=True, qmem_efficient=True,
                           num_samples=args.quasi_probes, k=k["quasi"], full_trace=False,
                           tol=TOL, stopping="rms")
        elif method == "jacobi":
            out = jacobi_alg(f, x0, guess, d, NUM_ITERS, full_trace=False, tol=TOL, stopping="rms")
        elif method == "broyden":
            out = broyden_deer_alg(f, x0, guess, d, NUM_ITERS, m=M_HIST, init="identity",
                                   k=k["broyden"], gamma=gamma, compress=args.compress,
                                   sequential_solve=args.sequential_solve, scan_tree=args.scan_tree,
                                   full_trace=False, tol=TOL, line_search=bool(args.ls_broyden),
                                   ls_c=LS_C)
        elif method == "accel":
            out = accelerated_broyden_deer_alg(f, x0, guess, d, NUM_ITERS, m=M_HIST,
                                               init="identity", k=k["accel"], gamma=gamma,
                                               eps_gram=EPS_GRAM, compress=args.compress,
                                               sequential_solve=args.sequential_solve,
                                               scan_tree=args.scan_tree,
                                               qmem_efficient=True, full_trace=False, tol=TOL,
                                               line_search=bool(args.ls_accel), ls_c=LS_C)
        else:
            raise ValueError(method)
        return out[2], out[1][-1]                       # iterations, last state
    return jax.jit(solve)


GPU = jax.devices()[0].device_kind
rows, done = [], set()
if args.resume and os.path.exists(args.out):
    with open(args.out, "rb") as fh:
        rows = pickle.load(fh)["rows"]
    done = {(r["model"], r["D"], r["L"], r["dt"], r["T"], r["method"]) for r in rows}
    print(f"resume: {len(rows)} rows already in {args.out}", flush=True)
t_start = time.time()
print(f"{jax.devices()} ({GPU}) | {'float64' if args.x64 else 'float32'}, scan_tree={args.scan_tree}, "
      f"compress={args.compress}, sequential_solve={args.sequential_solve} | matmul {jax.config.jax_default_matmul_precision} | "
      f"configs {args.configs} dt={args.dt} T={args.T} chains={args.chains} "
      f"methods={args.methods} timing={not args.no_timing}", flush=True)
for cfg in args.configs:
    model, D, L = cfg.split(":")
    D, L = int(D), int(L) if model == "mlp" else 0
    for dt in args.dt:
        k = damping(dt)
        for T in args.T:
            todo = [m for m in args.methods if (model, D, L, dt, T, m) not in done]
            if not todo:
                continue
            chains, tried = make_chains(D, L, T, dt, args.chains)
            print(f"\n=== {model} D={D} L={L} dt={dt} (a=gamma={1 - dt / TAU:.2f}) T={T}: "
                  f"{len(chains)} healthy chains of {tried} seeds | damping {k} "
                  f"[{time.time() - t_start:.0f}s] ===", flush=True)
            for method in todo:
                solver = make_solver(method, dt, k)
                its, times, err = [], [], None
                for seed, P, keys in chains:
                    try:
                        x0, guess, d = chain_inputs(keys, D, T)
                        if not args.no_timing:                  # first call also compiles
                            it, last = solver(P, x0, guess, d)
                            jax.block_until_ready(last)
                            if method != "seq" and int(it) >= NUM_ITERS:
                                # not converged: no further solves (runtimes are reported
                                # over converged chains only), time not recorded
                                times.append(np.nan)
                                its.append(int(it))
                                continue
                            for _ in range(args.nwarmups - 1):
                                jax.block_until_ready(solver(P, x0, guess, d))
                        reps = 1 if args.no_timing else args.nreps
                        t0 = time.time()
                        for _ in range(reps):
                            it, last = solver(P, x0, guess, d)
                            jax.block_until_ready(last)
                        times.append((time.time() - t0) / reps)
                        its.append(int(it))
                        del x0, guess, d
                    except Exception as e:                      # e.g. Newton out of memory
                        err = str(e).split("\n")[0][:200]
                        break
                row = dict(model=model, D=D, L=L, dt=dt, T=T, method=method,
                           k=k.get(BASE.get(method, method)), gpu=GPU, x64=bool(args.x64),
                           scan_tree=args.scan_tree, compress=args.compress,
                           sequential_solve=bool(args.sequential_solve),
                           seeds=[c[0] for c in chains], seeds_tried=tried,
                           line_search={"broyden": bool(args.ls_broyden),
                                        "accel": bool(args.ls_accel)}.get(method),
                           ls_c=LS_C if method in ("broyden", "accel") else None)
                if err:
                    row["error"] = err
                    print(f"  {method:8s} ERROR: {err[:100]}", flush=True)
                else:
                    its, times = np.array(its), np.array(times)
                    # sequential always "converges" (its count is T, not an iteration count)
                    c = np.ones(len(its), bool) if method == "seq" else its < NUM_ITERS
                    row.update(n_iters=its, times=None if args.no_timing else times)
                    tc = times[c] if c.any() else np.array([np.nan])
                    kk = k.get(BASE.get(method, method), "-")
                    print(f"  {method:11s} k={str(kk):4s} | failed {(~c).sum():2d}/"
                          f"{len(its)} | median iters (all) {np.median(its):7.1f} | median time "
                          f"(converged) {np.median(tc):.4f} s [{time.time() - t_start:.0f}s]",
                          flush=True)
                rows.append(row)
                os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                with open(args.out + ".tmp", "wb") as fh:          # atomic: safe under preemption
                    pickle.dump({"args": vars(args), "damping": DAMPING, "tol": TOL, "gpu": GPU,
                                 "num_iters": NUM_ITERS, "rows": rows}, fh)
                os.replace(args.out + ".tmp", args.out)
print(f"\nSaved {len(rows)} rows to {args.out} [{time.time() - t_start:.0f}s]")
