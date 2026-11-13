"""
broyden_theory_diagnostics.py

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

Usage:
    JAX_PLATFORMS=cuda python theory/broyden_theory_diagnostics.py [--settings std:200:0:0.5 mlp:200:8:3.0]
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

from dynamics.low_rank_rnn import LowRankRNN
from algs.deer import deer_alg
from algs.broyden_deer import broyden_deer_alg, broyden_solve
from algs.accelerated_broyden import accelerated_broyden_deer_alg, lrmat_compose_rand, solve_rankk_bidiag

# --- same constants / damping as harness_rnn_lowrank.py ---
DAMPING = {3.0: dict(broyden=0.4, accel=0.1, quasi=0.8), 0.5: dict(broyden=0.2, accel=0.2, quasi=0.1)}
SEED, R, TAU, G, SIG, BETA = 42, 3, 10.0, 2.0, 0.1, 0.5
TOL, M_HIST, EPS_GRAM, LS_C = 1e-4, 3, 1e-6, 1e-3
f32 = jnp.float32

p = argparse.ArgumentParser()
p.add_argument("--settings", nargs="+", default=["std:200:0:0.5", "mlp:200:8:3.0", "std:200:0:3.0", "mlp:200:8:0.5"],
               help="model:D:L:dt")
p.add_argument("--methods", nargs="+", default=["accel", "broyden"])
p.add_argument("--T", type=int, default=1000)
p.add_argument("--chains", type=int, default=8)
p.add_argument("--eta_chains", type=int, default=2, help="chains on which the merge tree is replayed")
p.add_argument("--rho_only", action="store_true", help="skip the spectral norms (sigma, SVDs): rho and Dennis-More only")
p.add_argument("--sigma_every", type=int, default=1,
               help="compute sigma (exact SVDs) only every N iterations and at the last one (NaN elsewhere)")
p.add_argument("--num_iters", type=int, default=100)
p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experiments",
                                             "saved_files", "broyden_theory_diagnostics.pkl"))
args = p.parse_args()


def make_params(D, L, seed):
    k_rnn, k_init, k_drv, k_guess = jr.split(jr.PRNGKey(seed), 4)
    rnn = LowRankRNN(D, R, key=k_rnn, tau=TAU, g=G, phi="tanh")
    Ws = []
    if L > 0:
        km = jr.split(jr.fold_in(jr.PRNGKey(seed), 1000), L + 1)
        Ws = [jr.normal(km[0], (D, R), dtype=f32) / np.sqrt(R)]
        Ws += [jr.normal(km[i], (D, D), dtype=f32) / np.sqrt(D) for i in range(1, L)]
        Ws += [jr.normal(km[L], (R, D), dtype=f32) / np.sqrt(D)]
    return dict(M=rnn.M, Nt=rnn.Nt, Ws=Ws), (k_init, k_drv, k_guess)


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


def chain_inputs(keys, D, T):
    k_init, k_drv, k_guess = keys
    return (jr.normal(k_init, (D,), dtype=f32) * 0.1, jr.normal(k_guess, (T, D), dtype=f32) * 0.1,
            SIG * jr.normal(k_drv, (T, D), dtype=f32))


def healthy_chains(D, L, T, dt, n):
    """First n non-saturating seeds (same filter as harness_rnn_lowrank.py)."""
    out, s = [], 0
    while len(out) < n and s < 10 * n:
        P, keys = make_params(D, L, SEED + s)
        s += 1
        x0, _, d = chain_inputs(keys, D, T)
        f = make_f(P, dt)
        ph = jnp.tanh(jax.lax.scan(lambda st, u: (f(st, u),) * 2, x0, d)[1][T // 4:])
        if int(jnp.sum(jnp.mean(jnp.abs(ph), 0) > 0.8)) or int(jnp.sum(jnp.mean(1 - ph ** 2, 0) < 0.1)):
            continue
        out.append((SEED + s - 1, P, keys))
    return out


def residual(f, x0, X, d):
    prev = jnp.concatenate([x0[None], X[:-1]], axis=0)
    return X - jax.vmap(f)(prev, d)


def spec(Ms):
    return jnp.linalg.norm(Ms, ord=2, axis=(-2, -1))


def block_errors(At, Ab, e, s, sigma=True):
    """At, Ab: (T-1, D, D) approximate / exact blocks for transitions j -> j+1 (acting on state j)."""
    Dl = At - Ab
    if args.rho_only or not sigma:
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


compose = jax.jit(lambda da, Ua, Va, db, Ub, Vb, key: lrmat_compose_rand(da, Ua, Va, db, Ub, Vb, M_HIST, key,
                                                                          rank_revealing=True))


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
print(f"{jax.devices()} | float32 highest | T={args.T} chains={args.chains} settings {args.settings}", flush=True)
for setting in args.settings:
    model, D, L, dt = setting.split(":")
    D, L, dt = int(D), int(L), float(dt)
    gamma = 1.0 - dt / TAU
    T = args.T
    for ci, (seed, P, keys) in enumerate(healthy_chains(D, L, T, dt, args.chains)):
        f = make_f(P, dt)
        x0, guess, d = chain_inputs(keys, D, T)
        xstar = jax.lax.scan(lambda st, u: (f(st, u),) * 2, x0, d)[1]
        jac = jax.jit(jax.vmap(jax.jacrev(f)))
        for method in args.methods:
            k = DAMPING[dt][method]
            if method == "accel":
                X = accelerated_broyden_deer_alg(f, x0, guess, d, args.num_iters, m=M_HIST, init="identity", k=k,
                                                 gamma=gamma, eps_gram=EPS_GRAM, compress="rand_rr",
                                                 full_trace=True, tol=TOL, line_search=True, ls_c=LS_C)[0]
            elif method == "quasi":                              # diagonal quasi-Newton, exact diagonal
                X = deer_alg(f, x0, guess, d, args.num_iters, quasi=True, k=k, full_trace=True, tol=TOL,
                             stopping="rms")[0]
            else:
                X = broyden_deer_alg(f, x0, guess, d, args.num_iters, m=M_HIST, init="identity", k=k,
                                     gamma=gamma, compress="rand_rr", full_trace=True, tol=TOL, line_search=False)[0]
            X = jnp.asarray(X)                                   # (I+1, T, D): X[0] = guess
            I = X.shape[0] - 1
            rms = [float(jnp.sqrt(jnp.mean(residual(f, x0, X[i], d) ** 2))) for i in range(I + 1)]
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
                want_sigma = (i % args.sigma_every == 0) or (i == last - 1)
                # self-check: the rebuilt (damped) blocks must reproduce the solver's step through its own
                # scan (same PRNG keys); large values only where the line search replaced the step
                r_i = residual(f, x0, X[i], d)
                if method == "quasi":                               # blocks used: (1-k) diag(A_t)
                    dA = jax.vmap(jnp.diag)(Aex)
                    s_rec = diag_solve_j((1 - k) * dA, r_i)
                    row["recon_err"] = float(jnp.linalg.norm(s_rec - s) / jnp.maximum(jnp.linalg.norm(s), 1e-30))
                    for kind, kk in (("damped", k), ("undamped", 0.0)):
                        At = jax.vmap(jnp.diag)((1 - kk) * dA)
                        row.update({f"{q}_{kind}": v for q, v in block_errors(At, Aex, e, s, sigma=want_sigma).items()})
                else:
                    d_full = dvec.at[-1].set(1.0)
                    Uk_, Vk_ = facs["damped"]
                    Uk_, Vk_ = Uk_.at[-1].set(0.0), Vk_.at[-1].set(0.0)
                    if method == "accel":
                        s_rec = solve_rankk_bidiag(d_full, Uk_, Vk_, r_i, M_HIST, compress="rand_rr", iteration=i)
                    else:
                        s_rec = broyden_solve(d_full, Uk_, Vk_, r_i, compress="rand_rr", iteration=i)
                    row["recon_err"] = float(jnp.linalg.norm(s_rec - s) / jnp.maximum(jnp.linalg.norm(s), 1e-30))
                    for kind, (Uk, Vk) in facs.items():
                        At = gamma * jnp.eye(D)[None] - jnp.einsum("tdi,tei->tde", Uk[:-1], Vk[:-1])
                        row.update({f"{kk}_{kind}": v for kk, v in block_errors(At, Aex, e, s, sigma=want_sigma).items()})
                if i in eta_iters:
                    Uk, Vk = facs["damped"]
                    row["tree"] = replay_tree(dvec[:-1], Uk[:-1], Vk[:-1], Aex)
                per_iter.append(row)
                if method != "quasi":
                    # --- update secant information with the step actually taken at iteration i ---
                    r_old, r_new = residual(f, x0, X[i], d), residual(f, x0, X[i + 1], d)
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
            records.append(dict(model=model, D=D, L=L, dt=dt, T=T, chain=ci, seed=seed, method=method, k=k,
                                gamma=gamma, converged_at=conv, rms=np.array(rms), per_iter=per_iter))
            fin = per_iter[-1] if per_iter else {}
            print(f"{model} D={D} dt={dt} chain {ci} {method:7s} k={k}: conv@{conv} | last iter: sigma_max "
                  f"{fin.get('sigma_max_damped', np.nan):.3f} (undamped {fin.get('sigma_max_undamped', np.nan):.3f}), "
                  f"rho {fin.get('rho_damped', np.nan):.3f}, dm_med {fin.get('dm_med_damped', np.nan):.3f} | "
                  f"recon_err median {np.median([r['recon_err'] for r in per_iter]) if per_iter else np.nan:.1e} "
                  f"[{time.time() - t_start:.0f}s]", flush=True)
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "wb") as fh:
                pickle.dump({"config": vars(args), "damping": DAMPING, "records": records}, fh)
print(f"saved {len(records)} records -> {args.out} [{time.time() - t_start:.0f}s]")
