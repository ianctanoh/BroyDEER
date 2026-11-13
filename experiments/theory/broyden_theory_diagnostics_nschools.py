"""
broyden_theory_diagnostics_nschools.py

N-schools version of broyden_theory_diagnostics.py (same quantities, same block rebuild): Gibbs sampler of
harness_Nschools.py with the official settings (float32, m=2, identity init with gamma=0, i.e. b_t = U_t V_t^T,
compress="low_rank_exact", lower bounds on the variances, tol 1e-3 on the RMS residual, Broyden k=0.3 without
line search, Accelerated k=0.1, eps_gram=1e-6, line search c=1e-3). Chains as in harness_Nschools.py (seed 13+i,
PRNGKey(i)). The merge tree is replayed with lrmat_compose_zero_diag (the composition the solver uses).

Original description:
Measure, along actual solver runs, the quantities the convergence theorem (Appendix "Convergence analysis
of quasi-Newton updates with compressed scan") depends on, for Parallel Broyden and Accelerated Parallel
Broyden on the low-rank RNN of harness_rnn_lowrank.py (same models, chains, damping, line search, rand_rr):

  per iteration i (blocks as used by the solver, i.e. damped, and the same secant pairs undamped):
    sigma_i      = max_t ||A~_t - A_t||_2            (+ median over t, and relative to ||A_t||_2)
    rho_i        = max_t ||(A~_t - A_t) e_{t-1}||_2 / ||e||_inf   (error direction; rho_i <= sigma_i,
                   and Proposition (one-step bound) holds with rho_i in place of sigma_i)
    dm_i         = ||(A~_t - A_t) s_{t-1}||_2 / ||s_{t-1}||_2 along the step s = x^(i+1) - x^(i)
                   (Dennis-More type; median and 90th percentile over t)
    rms residual and ||e||_inf (e = iterate - sequential trajectory)
  at a few iterations, replaying a balanced binary merge tree (Blelloch up-sweep) over the blocks with the
  solver's compression (lrmat_compose_rand, rank-revealing):
    eta_nu       = ||Comp(X_nu) - X_nu||_2 / ||Pi_nu||_2 for every merge node nu
    ||Pi_nu||_2 (exact product of the approximate blocks) and ||A_k...A_{j+1}||_2 (exact Jacobians)
    per segment length, to compare with the bound a exp(lambda~ n)

Blocks are rebuilt offline from a full-trace run (iterates), exactly as the solvers build them:
  Accelerated: A~_{j+1} = gamma I - (1-k) U_j V_j^T, U_j = solve(V^T V + eps I, R^T)^T over the last m pairs
  Broyden:     A~_{j+1} = gamma I - U_j V_j^T, sliding window of rank-1 updates u = (1-k)(g - b p)/|p|^2
(b = -A~ is the solver's subdiagonal block; d = -gamma for the identity init).

Usage (from experiments/):
    JAX_PLATFORMS=cuda python theory/broyden_theory_diagnostics_nschools.py [--S 10 50 100] [--methods accel broyden quasi]
        [--T 1000] [--chains 8] [--num_iters 300] [--rho_only] [--out PATH]
        [--T 1000] [--chains 8] [--eta_chains 2] [--num_iters 100] [--out PATH]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

import argparse
import pickle
import time

import jax
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from tensorflow_probability.substrates import jax as tfp
from dynamics.eight_schools import make_eight_schools, make_gibbs_step
from algs.deer import deer_alg
from algs.broyden_deer import broyden_deer_alg, broyden_solve
from algs.accelerated_broyden import accelerated_broyden_deer_alg, lrmat_compose_zero_diag, solve_rankk_bidiag
tfd = tfp.distributions

# --- official N-schools settings (harness_Nschools.py, 2026-10-27 final float32 run) ---
DAMPING = dict(broyden=0.3, accel=0.1, quasi=0.0)
SEED, N_S, TAUSQ_0, SIGMASQ_0 = 13, 20, 100.0, 10.0
TOL, M_HIST, EPS_GRAM, LS_C, GAMMA, COMPRESS = 1e-3, 2, 1e-6, 1e-3, 0.0, "low_rank_exact"
f32 = jnp.float32

p = argparse.ArgumentParser()
p.add_argument("--S", type=int, nargs="+", default=[10, 50, 100], help="number of schools (D = 2S+2)")
p.add_argument("--methods", nargs="+", default=["accel", "broyden"])
p.add_argument("--T", type=int, default=1000)
p.add_argument("--chains", type=int, default=8)
p.add_argument("--eta_chains", type=int, default=2, help="chains on which the merge tree is replayed")
p.add_argument("--rho_only", action="store_true", help="skip the spectral norms (sigma, SVDs): rho and Dennis-More only")
p.add_argument("--num_iters", type=int, default=300)
p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experiments",
                                             "saved_files", "broyden_theory_diagnostics_nschools.pkl"))
args = p.parse_args()


def nschools_chain(S, T, chain_idx):
    """Chain chain_idx exactly as setup_chain in harness_Nschools.py: (data xs, x0, guess, drivers, lower bounds, D)."""
    f, params, xs, D = make_eight_schools(S=S, N_s=N_S, seed=SEED + chain_idx, tausq_0=TAUSQ_0, sigmasq_0=SIGMASQ_0)
    nu_N, alpha_N, tausq_0, kappa_0, mu_0 = (params[k] for k in ("nu_N", "alpha_N", "tausq_0", "kappa_0", "mu_0"))
    sigmasq_0 = params.get("sigmasq_0", 10.0)
    key, init_key = jr.split(jr.PRNGKey(chain_idx))
    sk = jr.split(init_key, 4)
    tausq = tfd.InverseGamma(0.5 * nu_N, nu_N * tausq_0 * 0.5).sample(seed=sk[0])
    mu = tfd.Normal(mu_0, jnp.sqrt(tausq / kappa_0)).sample(seed=sk[1])
    thetas = tfd.Normal(mu, jnp.sqrt(tausq)).sample((S,), seed=sk[2])
    sigmasqs = tfd.InverseGamma(0.5 * alpha_N, 0.5 * alpha_N * sigmasq_0).sample((S,), seed=sk[3])
    x0 = jnp.concatenate((thetas, sigmasqs, jnp.array([mu]), jnp.array([tausq])))
    key, step_key = jr.split(key)
    x0 = f(x0, step_key)
    key, drivers_key = jr.split(key)
    d = jr.split(drivers_key, T)
    lb = jnp.concatenate([jnp.full(S, -jnp.inf), jnp.full(S, 1e-6), jnp.array([-jnp.inf]), jnp.array([1e-6])])
    return xs, x0, jnp.broadcast_to(x0, (T, D)), d, lb, D


def residual(f, x0, X, d):
    prev = jnp.concatenate([x0[None], X[:-1]], axis=0)
    return X - jax.vmap(f)(prev, d)


def spec(Ms):
    return jnp.linalg.norm(Ms, ord=2, axis=(-2, -1))


def block_errors(At, Ab, e, s):
    """At, Ab: (T-1, D, D) approximate / exact blocks for transitions j -> j+1 (acting on state j)."""
    Dl = At - Ab
    if args.rho_only:
        sig = rel = jnp.full((Dl.shape[0],), jnp.nan)
    else:
        sig = spec(Dl)
        rel = sig / spec(Ab)
    De = jnp.einsum("tij,tj->ti", Dl, e[:-1])
    rho = jnp.max(jnp.linalg.norm(De, axis=1)) / jnp.max(jnp.linalg.norm(e, axis=1))
    Ds = jnp.einsum("tij,tj->ti", Dl, s[:-1])
    dm = jnp.linalg.norm(Ds, axis=1) / jnp.maximum(jnp.linalg.norm(s[:-1], axis=1), 1e-30)
    return dict(sigma_max=float(sig.max()), sigma_med=float(jnp.median(sig)),
                sigma_rel_med=float(jnp.median(rel)), sigma_rel_max=float(rel.max()), rho=float(rho),
                dm_med=float(jnp.median(dm)), dm_p90=float(jnp.percentile(dm, 90)))


compose = jax.jit(lambda da, Ua, Va, db, Ub, Vb, key: lrmat_compose_zero_diag(da, Ua, Va, db, Ub, Vb))


def replay_tree(d, U, V, A_exact):
    """Balanced binary up-sweep over blocks 1..T-1 in the solver's b-form (b = diag(d) + U V^T, A~ = -b).
    Returns per node: segment length, eta = ||Comp(X)-X|| / ||Pi||, ||Pi|| (approx blocks), ||exact Jacobian product||."""
    n0 = d.shape[0]
    to_A = lambda dd, UU, VV: -(np.diag(np.asarray(dd, np.float64)) + np.asarray(UU, np.float64) @ np.asarray(VV, np.float64).T)
    # leaf summaries: stored factors, exact approximate product, exact Jacobian product
    nodes = [dict(fac=(d[t], U[t], V[t]), Pi=to_A(d[t], U[t], V[t]), PA=np.asarray(A_exact[t], np.float64), n=1)
             for t in range(n0)]
    recs, key = [], jr.PRNGKey(0)
    while len(nodes) > 1:
        nxt = []
        for i in range(0, len(nodes) - 1, 2):
            a, b = nodes[i], nodes[i + 1]                       # a earlier, b later
            key, sub = jr.split(key)
            dn, Un, Vn = compose(*a["fac"], *b["fac"], sub)
            stored = to_A(dn, Un, Vn)
            X = to_A(*b["fac"]) @ to_A(*a["fac"])               # product of the children's stored matrices
            Pi = b["Pi"] @ a["Pi"]
            PA = b["PA"] @ a["PA"]
            nPi = np.linalg.norm(Pi, 2)
            recs.append((a["n"] + b["n"], np.linalg.norm(stored - X, 2) / max(nPi, 1e-300), nPi,
                         np.linalg.norm(PA, 2)))
            nxt.append(dict(fac=(dn, Un, Vn), Pi=Pi, PA=PA, n=a["n"] + b["n"]))
        if len(nodes) % 2:
            nxt.append(nodes[-1])
        nodes = nxt
    return np.array(recs)                                       # columns: n, eta, ||Pi||, ||A-product||


solve_acc_j = jax.jit(lambda d_, U_, V_, r_, i_: solve_rankk_bidiag(d_, U_, V_, r_, M_HIST, compress=COMPRESS,
                                                                    iteration=i_))
solve_broy_j = jax.jit(lambda d_, U_, V_, r_, i_: broyden_solve(d_, U_, V_, r_, compress=COMPRESS, iteration=i_))
@jax.jit
def diag_solve_j(a, r):
    """Step of diagonal quasi-Newton: delta_0 = -r_0, delta_t = -r_t + a_{t-1} * delta_{t-1} (a: (T-1, D))."""
    def stepf(prev, inp):
        cur = -inp[1] + inp[0] * prev
        return cur, cur
    _, rest = jax.lax.scan(stepf, -r[0], (a, r[1:]))
    return jnp.concatenate([-r[:1], rest], axis=0)


records = []
t_start = time.time()
print(f"{jax.devices()} | float32 highest | T={args.T} chains={args.chains} S {args.S}", flush=True)
for S in args.S:
    T = args.T
    gamma = GAMMA
    # compiled once per S: the chain's data xs is a jit argument (as in harness_Nschools.py)
    gibbs = lambda xs: make_gibbs_step(S, N_S, xs, tausq_0=TAUSQ_0, sigmasq_0=SIGMASQ_0)
    run = {
        "accel": jax.jit(lambda xs, x0, guess, d, lb: accelerated_broyden_deer_alg(
            gibbs(xs), x0, guess, d, args.num_iters, m=M_HIST, init="identity", k=DAMPING["accel"], gamma=gamma,
            eps_gram=EPS_GRAM, compress=COMPRESS, full_trace=True, tol=TOL, lower_bounds=lb, line_search=True,
            ls_c=LS_C)[0]),
        "broyden": jax.jit(lambda xs, x0, guess, d, lb: broyden_deer_alg(
            gibbs(xs), x0, guess, d, args.num_iters, m=M_HIST, init="identity", k=DAMPING["broyden"], gamma=gamma,
            compress=COMPRESS, full_trace=True, tol=TOL, lower_bounds=lb, line_search=False, ls_c=LS_C)[0]),
        # diagonal quasi-Newton as in harness_Nschools.py (exact diagonal, no bounds)
        "quasi": jax.jit(lambda xs, x0, guess, d, lb: deer_alg(
            gibbs(xs), x0, guess, d, args.num_iters, quasi=True, k=DAMPING["quasi"], full_trace=True, tol=TOL,
            stopping="rms")[0]),
    }
    jac_j = jax.jit(lambda xs, X, d: jax.vmap(jax.jacrev(gibbs(xs)))(X, d))
    res_j = jax.jit(lambda xs, x0, X, d: residual(gibbs(xs), x0, X, d))
    seq_j = jax.jit(lambda xs, x0, d: jax.lax.scan(lambda st, u: (gibbs(xs)(st, u),) * 2, x0, d)[1])
    for ci in range(args.chains):
        xs, x0, guess, d, lb, D = nschools_chain(S, T, ci)
        seed, model, L, dt = SEED + ci, f"nschools_S{S}", 0, None
        xstar = seq_j(xs, x0, d)
        jac = lambda X_, d_: jac_j(xs, X_, d_)
        resid = lambda X_: res_j(xs, x0, X_, d)
        for method in args.methods:
            k = DAMPING[method]
            X = run[method](xs, x0, guess, d, lb)
            X = jnp.asarray(X)                                   # (I+1, T, D): X[0] = guess
            I = X.shape[0] - 1
            rms = [float(jnp.sqrt(jnp.mean(resid(X[i]) ** 2))) for i in range(I + 1)]
            conv = next((i for i, v in enumerate(rms) if v < TOL), None)
            last = conv if conv is not None else I
            eta_iters = sorted({M_HIST, max(M_HIST, last // 2), max(M_HIST, last - 1)}) if (ci < args.eta_chains and method != "quasi") else []
            # secant history (newest first) and Broyden factors, replayed as the solvers do
            Pw = jnp.zeros((M_HIST, T, D)); Gw = jnp.zeros((M_HIST, T, D))
            Ub = {kk: jnp.zeros((T, D, M_HIST)) for kk in ("damped", "undamped")}
            Vb = jnp.zeros((T, D, M_HIST))
            per_iter = []
            for i in range(last):
                dvec = -gamma * jnp.ones((T, D))                    # b-form diagonal, d = -gamma
                if method == "quasi":
                    facs = None                                     # blocks (1-k) diag(A_t), built below
                elif method == "accel":
                    Vt = jnp.transpose(Pw, (1, 2, 0))               # (T, D, m)
                    Rt = jnp.transpose(Gw, (1, 2, 0)) - dvec[:, :, None] * Vt
                    C = jnp.einsum("tdi,tdj->tij", Vt, Vt) + EPS_GRAM * jnp.eye(M_HIST)
                    Usol = jnp.swapaxes(jnp.linalg.solve(C, jnp.swapaxes(Rt, 1, 2)), 1, 2)
                    zero = 0.0 if i == 0 else 1.0
                    facs = {"damped": ((1 - k) * zero * Usol, Vt), "undamped": (zero * Usol, Vt)}
                else:
                    facs = {"damped": (Ub["damped"], Vb), "undamped": (Ub["undamped"], Vb)}
                Aex = jac(X[i][:-1], d[1:])                         # exact blocks at iterate i (T-1, D, D)
                e = X[i] - xstar
                s = X[i + 1] - X[i]
                row = dict(i=i, rms=rms[i], e_inf=float(jnp.max(jnp.linalg.norm(e, axis=1))))
                # self-check: the rebuilt (damped) blocks must reproduce the solver's step through its own
                # scan (same PRNG keys); large values only where the line search replaced the step
                r_i = resid(X[i])
                if method == "quasi":                               # blocks used: (1-k) diag(A_t)
                    dA = jax.vmap(jnp.diag)(Aex)
                    s_rec = diag_solve_j((1 - k) * dA, r_i)
                    row["recon_err"] = float(jnp.linalg.norm(s_rec - s) / jnp.maximum(jnp.linalg.norm(s), 1e-30))
                    for kind, kk in (("damped", k), ("undamped", 0.0)):
                        At = jax.vmap(jnp.diag)((1 - kk) * dA)
                        row.update({f"{q}_{kind}": v for q, v in block_errors(At, Aex, e, s).items()})
                else:
                    d_full = dvec.at[-1].set(1.0)
                    Uk_, Vk_ = facs["damped"]
                    Uk_, Vk_ = Uk_.at[-1].set(0.0), Vk_.at[-1].set(0.0)
                    if method == "accel":
                        s_rec = solve_acc_j(d_full, Uk_, Vk_, r_i, i)
                    else:
                        s_rec = solve_broy_j(d_full, Uk_, Vk_, r_i, i)
                    row["recon_err"] = float(jnp.linalg.norm(s_rec - s) / jnp.maximum(jnp.linalg.norm(s), 1e-30))
                    for kind, (Uk, Vk) in facs.items():
                        At = gamma * jnp.eye(D)[None] - jnp.einsum("tdi,tei->tde", Uk[:-1], Vk[:-1])
                        row.update({f"{kk}_{kind}": v for kk, v in block_errors(At, Aex, e, s).items()})
                if i in eta_iters:
                    Uk, Vk = facs["damped"]
                    row["tree"] = replay_tree(dvec[:-1], Uk[:-1], Vk[:-1], Aex)
                per_iter.append(row)
                if method != "quasi":
                    # --- update secant information with the step actually taken at iteration i ---
                    r_old, r_new = resid(X[i]), resid(X[i + 1])
                    y = r_new - r_old
                    tgt = jnp.concatenate([y[1:] - s[1:], jnp.zeros((1, D))], axis=0)   # target for block j
                    pj = s                                                              # step at state j
                    if method == "accel":
                        Pw = jnp.concatenate([pj[None], Pw[:-1]]); Gw = jnp.concatenate([tgt[None], Gw[:-1]])
                    else:
                        for kind, kk in (("damped", k), ("undamped", 0.0)):
                            bp = dvec * pj + jnp.einsum("tdi,ti->td", Ub[kind], jnp.einsum("tdi,td->ti", Vb, pj))
                            u_new = (1 - kk) * (tgt - bp) / (jnp.sum(pj ** 2, axis=1, keepdims=True) + 1e-10)
                            Ub[kind] = jnp.concatenate([Ub[kind][:, :, 1:], u_new[:, :, None]], axis=2)
                        Vb = jnp.concatenate([Vb[:, :, 1:], pj[:, :, None]], axis=2)
            records.append(dict(model=model, S=S, D=D, L=L, dt=dt, T=T, chain=ci, seed=seed, method=method, k=k,
                                gamma=gamma, converged_at=conv, rms=np.array(rms), per_iter=per_iter))
            fin = per_iter[-1] if per_iter else {}
            print(f"{model} D={D} chain {ci} {method:7s} k={k}: conv@{conv} | last iter: sigma_max "
                  f"{fin.get('sigma_max_damped', np.nan):.3f} (undamped {fin.get('sigma_max_undamped', np.nan):.3f}), "
                  f"rho {fin.get('rho_damped', np.nan):.3f}, dm_med {fin.get('dm_med_damped', np.nan):.3f} | "
                  f"recon_err median {np.median([r['recon_err'] for r in per_iter]) if per_iter else np.nan:.1e} "
                  f"[{time.time() - t_start:.0f}s]", flush=True)
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "wb") as fh:
                pickle.dump({"config": vars(args), "damping": DAMPING, "records": records}, fh)
print(f"saved {len(records)} records -> {args.out} [{time.time() - t_start:.0f}s]")
