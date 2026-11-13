"""
wallclock_breakdown.py

Where does the per-iteration time go? For each benchmark, at the paper settings, we time the
components of one solver iteration separately (jitted, median over --nrep runs after warm-up):
  f_eval        : one evaluation of the transition function at all T positions (vmap f; for DiT,
                  the batched eps_theta call of the DDIM residual)
  jac_diag      : diagonal of df/dx by reverse-mode autodiff (vmap jacrev, then diag), as in quasi-DEER
  jac_full      : full Jacobians df/dx (vmap jacrev), as in DEER
  hutch         : 1-probe Hutchinson diagonal estimate (one JVP per position), as in quasi-DEER (DiT)
                  and quasi_hutch (low-rank RNN)
  scan_diag     : parallel scan of the diagonal linear recurrence (quasi-DEER)
  scan_full     : parallel scan of the dense linear recurrence (DEER)
  broyden_update: rank-1 Broyden update of all blocks (Parallel Broyden)
  broyden_scan  : parallel scan with DPLR blocks + rank compression (Parallel Broyden; also
                  Accelerated Broyden on DiT, which uses the same solve)
  accel_fit     : refit of U_t from the window (Gram solve) (Accelerated Broyden)
  accel_scan    : parallel scan with DPLR blocks + rank compression (Accelerated Broyden)
Per method, the component times are combined with the number of calls per iteration read off the
solver code (COMPOSITION), and compared with the measured end-to-end time per iteration of the
actual solver run for a fixed number of iterations (--e2e_iters, tol < 0 so it never stops early).
The difference ("other") is elementwise work, the stopping criterion, line search, window updates.

Inputs: iterates = the sequential trajectory (values do not affect timing), secant windows =
random steps of size 1e-2. float32; matmul precision 'highest' (as the RNN / Gibbs harnesses) unless
--precision default (as run_dit_32images.py; used for the DiT rows).

Benchmarks:
  lowrank_std : low-rank RNN std D=200 dt=0.5 T=1e5 (harness_rnn_lowrank.py settings, chain seed 42)
  lowrank_mlp : low-rank RNN mlp D=200 L=8 dt=3 T=1e5
  nschools    : N-schools Gibbs S=100 T=1e5 (and S=200 with --S 100 200)
  dit         : DiT-XL/2 256, 50 DDIM steps, cfg 5, one image (orangutan)
Output: saved_files/wallclock_breakdown.pkl (and .csv): one row per (benchmark, method, component).

Usage:
    JAX_PLATFORMS=cuda python speedup/wallclock_breakdown.py [--benchmarks lowrank_std lowrank_mlp nschools dit]
        [--T 100000] [--S 100 200] [--nrep 10] [--e2e_iters 10]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

import argparse
import pickle
import time
import traceback

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import vmap

p = argparse.ArgumentParser()
p.add_argument("--benchmarks", nargs="+", default=["lowrank_std", "lowrank_mlp", "nschools", "dit"])
p.add_argument("--T", type=int, default=100000, help="length for the low-rank RNN and N-schools")
p.add_argument("--D_lowrank", type=int, default=200)
p.add_argument("--S", type=int, nargs="+", default=[100, 200])
p.add_argument("--nrep", type=int, default=10)
p.add_argument("--e2e_iters", type=int, default=10)
p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experiments", "saved_files",
                                             "wallclock_breakdown.pkl"))
p.add_argument("--precision", default="highest", choices=["highest", "default"],
               help="float32 matmul precision: 'highest' as in the RNN / Gibbs harnesses; 'default' (TF32 on the GPU) "
                    "as in run_dit_32images.py, which sets no precision")
args = p.parse_args()
jax.config.update("jax_enable_x64", False)
if args.precision == "highest":
    jax.config.update("jax_default_matmul_precision", "highest")

from algs.deer import get_residual, diag_mat_operator, full_mat_operator, deer_alg
from algs.broyden_deer import broyden_solve, update_all_blocks, broyden_deer_alg
from algs.accelerated_broyden import solve_rankk_bidiag, accelerated_broyden_deer_alg

GPU = jax.devices()[0].device_kind
print(f"{jax.devices()} ({GPU}) | benchmarks {args.benchmarks} | precision {args.precision} | nrep {args.nrep} | e2e iters {args.e2e_iters}",
      flush=True)

# calls per iteration of each component, read off the solver code
COMPOSITION = {
    "quasi":       {"f_eval": 2, "jac_diag": 1, "scan_diag": 1},       # deer_alg(quasi, stopping="rms")
    "quasi_hutch": {"f_eval": 2, "hutch": 1, "scan_diag": 1},
    "newton":      {"f_eval": 2, "jac_full": 1, "scan_full": 1},
    "broyden":     {"f_eval": 2, "broyden_update": 1, "broyden_scan": 1},
    "accel":       {"f_eval": 2, "accel_fit": 1, "accel_scan": 1},     # + 1 f_eval when line search backtracks
    # DDIM solvers carry r_old: one eps call per iteration
    "dit_quasi":   {"f_eval": 1, "hutch": 1, "scan_diag": 1},
    "dit_broyden": {"f_eval": 1, "broyden_update": 1, "broyden_scan": 1},
    "dit_accel":   {"f_eval": 1, "accel_fit": 1, "broyden_scan": 1},
}
rows = []


def bench(fn, *a):
    """Median wall-clock seconds of jitted fn(*a) (compile + 2 warm-ups excluded); NaN on error."""
    try:
        jf = jax.jit(fn)
        jax.block_until_ready(jf(*a))
        for _ in range(2):
            jax.block_until_ready(jf(*a))
        ts = []
        for _ in range(args.nrep):
            t0 = time.perf_counter()
            jax.block_until_ready(jf(*a))
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts)), None
    except Exception as e:                                       # e.g. out of memory
        return float("nan"), f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"


def save():
    import pandas as pd
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_pickle(args.out)
    df.to_csv(args.out.replace(".pkl", ".csv"), index=False)


def fit_U(d, P, G, k, eps_gram):
    """Accelerated Broyden U_t from the window (as in accelerated_broyden_deer_alg, generic m)."""
    m = P.shape[1]
    def one(d_t, steps_t, targets_t):
        V_t = steps_t.T
        R_t = targets_t.T - d_t[:, None] * V_t
        C_t = V_t.T @ V_t + eps_gram * jnp.eye(m)
        if m == 2:                                           # Cramer's rule, as in the solver
            det = C_t[0, 0] * C_t[1, 1] - C_t[0, 1] * C_t[1, 0]
            inv_det = 1.0 / jnp.maximum(det, 1e-30)
            U_t = (1 - k) * inv_det * jnp.stack([R_t[:, 0] * C_t[1, 1] - R_t[:, 1] * C_t[0, 1],
                                                 R_t[:, 1] * C_t[0, 0] - R_t[:, 0] * C_t[1, 0]], axis=1)
        else:
            U_t = (1 - k) * jnp.linalg.solve(C_t, R_t.T).T
        return U_t, V_t
    return vmap(one)(d[:-1], P[:-1], G[:-1])


def report(bench_name, setting, comp, e2e, notes):
    """Add rows (components and per-method totals) and print the breakdown."""
    for c, (t, err) in comp.items():
        rows.append(dict(benchmark=bench_name, setting=setting, method=None, component=c, calls=None,
                         time_s=t, error=err, gpu=GPU))
    print(f"\n--- {bench_name} | {setting} ---")
    for c, (t, err) in comp.items():
        print(f"  {c:15s} {t * 1e3:10.3f} ms" + (f"   [{err}]" if err else ""))
    for meth, (t_e2e, err) in e2e.items():
        parts = {c: n * comp[c][0] for c, n in COMPOSITION[meth].items()}
        total = sum(parts.values())
        for c, v in parts.items():
            rows.append(dict(benchmark=bench_name, setting=setting, method=meth, component=c,
                             calls=COMPOSITION[meth][c], time_s=v, error=comp[c][1], gpu=GPU))
        rows.append(dict(benchmark=bench_name, setting=setting, method=meth, component="sum_components",
                         calls=None, time_s=total, error=None, gpu=GPU))
        rows.append(dict(benchmark=bench_name, setting=setting, method=meth, component="end_to_end",
                         calls=None, time_s=t_e2e, error=err, gpu=GPU))
        rows.append(dict(benchmark=bench_name, setting=setting, method=meth, component="other",
                         calls=None, time_s=t_e2e - total, error=None, gpu=GPU))
        share = "  ".join(f"{c} {v / t_e2e:5.1%}" for c, v in parts.items())
        print(f"  {meth:12s} e2e/iter {t_e2e * 1e3:10.3f} ms | sum {total * 1e3:10.3f} ms | {share} | "
              f"other {(t_e2e - total) / t_e2e:5.1%}" + (f"   [{err}]" if err else ""))
    if notes:
        print("  " + notes)
    save()


def e2e_time(solve, *a):
    """End-to-end seconds per iteration: solve(*a) -> number of iterations, with tol < 0."""
    t, err = bench(solve, *a)
    if err is None:
        n = int(jax.jit(solve)(*a))
        if n != args.e2e_iters:
            err = f"ran {n} iterations, expected {args.e2e_iters}"
        t = t / max(n, 1)
    return t, err


def generic_components(f, x0, X, drivers, m, k_accel, k_broy, gamma, eps_gram, compress, newton=True):
    """Component timings for a (state, driver) -> state map over T positions (RNN / N-schools)."""
    T, D = X.shape
    key = jr.PRNGKey(0)
    kP, kG, kz = jr.split(key, 3)
    P = 1e-2 * jr.normal(kP, (T, m, D))
    G = 1e-2 * jr.normal(kG, (T, m, D))
    d = -gamma * jnp.ones((T, D))
    U = 1e-2 * jr.normal(kG, (T, D, m))
    V = 1e-2 * jr.normal(kP, (T, D, m))
    r = 1e-2 * jr.normal(kz, (T, D))
    ds = 1e-2 * jr.normal(kz, (T, D))
    A_diag = 0.5 * jnp.ones((T, D))
    comp = {}
    comp["f_eval"] = bench(lambda X: get_residual(f, x0, X, drivers), X)
    comp["jac_diag"] = bench(lambda X: vmap(jnp.diag)(vmap(jax.jacrev(f))(X[:-1], drivers[1:])), X)
    def hutch(X):
        zs = jr.rademacher(kz, (T - 1, D)).astype(X.dtype)
        jv = vmap(lambda x, u, z: jax.jvp(lambda y: f(y, u), (x,), (z,))[1])(X[:-1], drivers[1:], zs)
        return zs * jv
    comp["hutch"] = bench(hutch, X)
    comp["scan_diag"] = bench(lambda A, b: jax.lax.associative_scan(diag_mat_operator, (A, b))[1], A_diag, r)
    if newton:
        comp["jac_full"] = bench(lambda X: vmap(jax.jacrev(f))(X[:-1], drivers[1:]), X)
        A_full = jnp.broadcast_to(0.5 * jnp.eye(D) / D, (T, D, D))
        comp["scan_full"] = bench(lambda A, b: jax.lax.associative_scan(full_mat_operator, (A, b))[1], A_full, r)
        del A_full
    else:
        comp["jac_full"] = comp["scan_full"] = (float("nan"), "skipped (dense Jacobians do not fit)")
    comp["broyden_update"] = bench(lambda d, U, V, ds, r0, r1: update_all_blocks(d, U, V, ds, r0, r1, k_broy),
                                   d, U, V, ds, r, r + ds)
    comp["broyden_scan"] = bench(lambda d, U, V, r: broyden_solve(d, U, V, r, compress=compress, iteration=1),
                                 d, U, V, r)
    comp["accel_fit"] = bench(lambda d, P, G: fit_U(d, P, G, k_accel, eps_gram), d, P, G)
    comp["accel_scan"] = bench(lambda d, U, V, r: solve_rankk_bidiag(d, U, V, r, m, compress=compress, iteration=1),
                               d, U, V, r)
    return comp


# ---------------------------------------------------------------- low-rank RNN
def lowrank(model):
    from dynamics.low_rank_rnn import LowRankRNN
    R, TAU, G_, SIG, BETA, M_HIST, EPS_GRAM = 3, 10.0, 2.0, 0.1, 0.5, 3, 1e-6
    D, T = args.D_lowrank, args.T
    L, dt = (0, 0.5) if model == "std" else (8, 3.0)
    k = {0.5: dict(newton=0.0, quasi=0.1, broyden=0.2, accel=0.2),
         3.0: dict(newton=0.0, quasi=0.8, broyden=0.4, accel=0.1)}[dt]
    seed = 42
    k_rnn, k_init, k_drv, k_guess = jr.split(jr.PRNGKey(seed), 4)
    rnn = LowRankRNN(D, R, key=k_rnn, tau=TAU, g=G_, phi="tanh")
    Ws = []
    if L > 0:
        km = jr.split(jr.fold_in(jr.PRNGKey(seed), 1000), L + 1)
        Ws = [jr.normal(km[0], (D, R)) / np.sqrt(R)]
        Ws += [jr.normal(km[i], (D, D)) / np.sqrt(D) for i in range(1, L)]
        Ws += [jr.normal(km[L], (R, D)) / np.sqrt(D)]
    M, Nt = rnn.M, rnn.Nt
    a, dtt = 1.0 - dt / TAU, dt / TAU

    def g(z):
        if not Ws:
            return z
        h = z
        for Wl in Ws[:-1]:
            h = jnp.tanh(Wl @ h)
        return z + BETA * (Ws[-1] @ h)

    def f(x, eps):
        return a * x + dtt * (M @ g(Nt.T @ jnp.tanh(x))) + eps

    x0 = jr.normal(k_init, (D,)) * 0.1
    guess = jr.normal(k_guess, (T, D)) * 0.1
    drv = SIG * jr.normal(k_drv, (T, D))
    X = jax.jit(lambda x0, d: jax.lax.scan(lambda s, u: (f(s, u),) * 2, x0, d)[1])(x0, drv)
    comp = generic_components(f, x0, X, drv, M_HIST, k["accel"], k["broyden"], a, EPS_GRAM, "rand_rr")

    n, tol = args.e2e_iters, -1.0
    solvers = {
        "newton": lambda x0, g_, d: deer_alg(f, x0, g_, d, n, k=k["newton"], tol=tol, stopping="rms")[2],
        "quasi": lambda x0, g_, d: deer_alg(f, x0, g_, d, n, quasi=True, k=k["quasi"], tol=tol,
                                            stopping="rms")[2],
        "quasi_hutch": lambda x0, g_, d: deer_alg(f, x0, g_, d, n, quasi=True, qmem_efficient=True,
                                                  num_samples=1, k=k["quasi"], tol=tol, stopping="rms")[2],
        "broyden": lambda x0, g_, d: broyden_deer_alg(f, x0, g_, d, n, m=M_HIST, init="identity", k=k["broyden"],
                                                      gamma=a, compress="rand_rr", tol=tol,
                                                      line_search=False)[2],
        "accel": lambda x0, g_, d: accelerated_broyden_deer_alg(f, x0, g_, d, n, m=M_HIST, init="identity",
                                                                k=k["accel"], gamma=a, eps_gram=EPS_GRAM,
                                                                compress="rand_rr", qmem_efficient=True,
                                                                tol=tol, line_search=True, ls_c=1e-3)[2],
    }
    e2e = {meth: e2e_time(s, x0, guess, drv) for meth, s in solvers.items()}
    report("low-rank RNN", f"{model} D={D} L={L} dt={dt} T={T}", comp, e2e,
           "Broyden/Accel use the fixed exact diagonal a I (identity init): no autodiff.")


# ---------------------------------------------------------------- N-schools
def nschools(S):
    from tensorflow_probability.substrates import jax as tfp
    tfd = tfp.distributions
    from dynamics.eight_schools import make_eight_schools
    T, N_s, seed = args.T, 20, 13
    f, params, _, D = make_eight_schools(S=S, N_s=N_s, seed=seed, tausq_0=100.0, sigmasq_0=10.0)
    nu_N, alpha_N, tausq_0, kappa_0, mu_0 = (params[k] for k in ("nu_N", "alpha_N", "tausq_0", "kappa_0", "mu_0"))
    sigmasq_0 = params.get("sigmasq_0", 10.0)
    key, init_key = jr.split(jr.PRNGKey(0))
    sk = jr.split(init_key, 4)
    tausq = tfd.InverseGamma(0.5 * nu_N, nu_N * tausq_0 * 0.5).sample(seed=sk[0])
    mu = tfd.Normal(mu_0, jnp.sqrt(tausq / kappa_0)).sample(seed=sk[1])
    thetas = tfd.Normal(mu, jnp.sqrt(tausq)).sample((S,), seed=sk[2])
    sigmasqs = tfd.InverseGamma(0.5 * alpha_N, 0.5 * alpha_N * sigmasq_0).sample((S,), seed=sk[3])
    x0 = jnp.concatenate((thetas, sigmasqs, jnp.array([mu]), jnp.array([tausq])))
    key, step_key = jr.split(key)
    x0 = f(x0, step_key)
    key, drivers_key = jr.split(key)
    drv = jr.split(drivers_key, T)
    guess = jnp.broadcast_to(x0, (T, D))
    lb = jnp.concatenate([jnp.full(S, -jnp.inf), jnp.full(S, 1e-6), jnp.array([-jnp.inf]), jnp.array([1e-6])])
    X = jax.jit(lambda x0, d: jax.lax.scan(lambda s, u: (f(s, u),) * 2, x0, d)[1])(x0, drv)
    # dense T x D x D Jacobians: 4 T D^2 bytes (16 GB at S=100, 65 GB at S=200; the official S=200 DEER run failed)
    newton = 4.0 * T * D * D < 30e9
    comp = generic_components(f, x0, X, drv, 2, 0.1, 0.3, 0.0, 1e-6, "low_rank_exact", newton=newton)

    n, tol = args.e2e_iters, -1.0
    solvers = {
        "quasi": lambda x0, g_, d: deer_alg(f, x0, g_, d, n, quasi=True, k=0.0, tol=tol, stopping="rms")[2],
        "broyden": lambda x0, g_, d: broyden_deer_alg(f, x0, g_, d, n, m=2, init="identity", k=0.3, gamma=0.0,
                                                      sequential_solve=False, compress="low_rank_exact",
                                                      tol=tol, lower_bounds=lb, line_search=False,
                                                      ls_c=1e-3)[2],
        "accel": lambda x0, g_, d: accelerated_broyden_deer_alg(f, x0, g_, d, n, m=2, init="identity", k=0.1,
                                                                gamma=0.0, eps_gram=1e-6,
                                                                compress="low_rank_exact", tol=tol,
                                                                lower_bounds=lb, line_search=True,
                                                                ls_c=1e-3)[2],
    }
    if newton:
        solvers["newton"] = lambda x0, g_, d: deer_alg(f, x0, g_, d, n, k=0.0, tol=tol, stopping="rms")[2]
    e2e = {meth: e2e_time(s, x0, guess, drv) for meth, s in solvers.items()}
    report("N-schools", f"S={S} D={D} T={T}", comp, e2e,
           "Broyden/Accel: identity init with gamma=0 (b_t = U_t V_t^T), compress=low_rank_exact.")


# ---------------------------------------------------------------- DiT
def dit():
    from models.dit import make_dit_diffusion_reusable
    from algs.deer_ddim import deer_ddim_alg
    from algs.broyden_ddim import broyden_ddim_alg
    from algs.accelerated_broyden_ddim import accelerated_broyden_ddim_alg
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ckpt = os.path.join(root, "pretrained_models", "DiT-XL-2-256x256.pt")
    T, m = 50, 2
    set_class, _ = make_dit_diffusion_reusable(ckpt, T=T, cfg_scale=5.0, eta=0.0, seed=0)
    dyn, x0, drivers = set_class(365)                                   # orangutan
    D = x0.shape[0]
    X = jax.jit(lambda dyn, x0, d: jax.lax.scan(lambda s, u: (dyn.deer_fxn(s, u),) * 2, x0, d)[1])(dyn, x0, drivers)
    t_per_s = T - jnp.arange(T)
    kP, kG, kz = jr.split(jr.PRNGKey(0), 3)
    P = 1e-2 * jr.normal(kP, (T, m, D))
    Gw = 1e-2 * jr.normal(kG, (T, m, D))
    d = -jnp.ones((T, D))
    U = 1e-2 * jr.normal(kG, (T, D, m))
    V = 1e-2 * jr.normal(kP, (T, D, m))
    r = 1e-2 * jr.normal(kz, (T, D))

    def resid(dyn, X):                                                  # as _ddim_residual
        Xf = jnp.concatenate([x0[None], X], axis=0)
        eps = vmap(dyn.eps_model)(Xf[:T], t_per_s)
        f = (dyn.coef_x[t_per_s][:, None] * Xf[:T] + dyn.coef_eps[t_per_s][:, None] * eps
             + dyn.coef_noise[t_per_s][:, None] * drivers[:, :D])
        return X - f

    def hutch(dyn, X):                                                  # as deer_ddim_alg, num_samples=1
        Xf = jnp.concatenate([x0[None], X], axis=0)
        zs = jr.rademacher(kz, (T - 1, D)).astype(X.dtype)
        jv = vmap(lambda x, t, z: jax.jvp(lambda y: dyn.eps_model(y, t), (x,), (z,))[1])(Xf[1:T], t_per_s[1:], zs)
        return zs * jv

    comp = {
        "f_eval": bench(resid, dyn, X),
        "hutch": bench(hutch, dyn, X),
        "scan_diag": bench(lambda A, b: jax.lax.associative_scan(diag_mat_operator, (A, b))[1],
                           0.5 * jnp.ones((T, D)), r),
        "broyden_update": bench(lambda d, U, V, ds, r0, r1: update_all_blocks(d, U, V, ds, r0, r1, 0.0),
                                d, U, V, r, r, 2 * r),
        "broyden_scan": bench(lambda d, U, V, r: broyden_solve(d, U, V, r, compress="rand", iteration=1),
                              d, U, V, r),
        "accel_fit": bench(lambda d, P, G: fit_U(d, P, G, 0.0, 1e-10), d, P, Gw),
    }
    n, tol = args.e2e_iters, -1.0
    guess = jnp.broadcast_to(x0, (T, D))
    solvers = {
        "dit_quasi": lambda dyn, x0, g_, d: deer_ddim_alg(dyn, x0, g_, d, n, num_samples=1, k=0.0, tol=tol)[2],
        "dit_broyden": lambda dyn, x0, g_, d: broyden_ddim_alg(dyn, x0, g_, d, n, m=2, init="identity", gamma=1.0,
                                                               k=0.0, tol=tol, sequential_solve=False,
                                                               compress="rand")[2],
        "dit_accel": lambda dyn, x0, g_, d: accelerated_broyden_ddim_alg(dyn, x0, g_, d, n, m=2, init="identity",
                                                                         gamma=1.0, k=0.0, tol=tol,
                                                                         sequential_solve=False,
                                                                         compress="rand")[2],
    }
    e2e = {meth: e2e_time(s, dyn, x0, guess, drivers) for meth, s in solvers.items()}
    report("DiT", f"DiT-XL/2 T={T} D={D} cfg=5 (orangutan)", comp, e2e,
           "DDIM solvers carry the residual: one eps call per iteration (+ one JVP for quasi-DEER).")


t0 = time.time()
for b in args.benchmarks:
    jobs = [(b, S) for S in args.S] if b == "nschools" else [(b, None)]
    for name, S in jobs:
        try:
            if name == "lowrank_std":
                lowrank("std")
            elif name == "lowrank_mlp":
                lowrank("mlp")
            elif name == "nschools":
                nschools(S)
            elif name == "dit":
                dit()
            else:
                raise ValueError(name)
        except Exception:
            print(f"\n!!! {name} {S or ''} failed:\n{traceback.format_exc()}", flush=True)
        print(f"[{time.time() - t0:.0f}s]", flush=True)
save()
print(f"saved {len(rows)} rows -> {args.out}")
