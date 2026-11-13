"""
make_theory_summary.py

Summary of the convergence-theory diagnostics (broyden_theory_diagnostics*.py), one row per (benchmark,
setting, method), written to results/appendix/theory_summary.csv. Inputs (experiments/saved_files/):
  broyden_theory_diagnostics.pkl                 low-rank RNN, Parallel / Accelerated Parallel Broyden
  broyden_theory_diagnostics_quasi.pkl           low-rank RNN, diagonal quasi-Newton (rho only)
  broyden_theory_diagnostics_quasi_sigma.pkl     same runs with sigma every 5 iterations (used first, if present)
  broyden_theory_diagnostics_quasi_sigma_last.pkl  same runs with sigma at the first and last iteration only
  broyden_theory_diagnostics_nschools.pkl        N-schools, Parallel / Accelerated Parallel Broyden
Columns (medians over chains unless noted; "last" = last iteration before convergence):
  n_converged, iters_median (converged chains), sigma_max_last (sigma_i = max_t ||A~_t - A_t||_2),
  sigma_med_last (median over blocks t), sigma_rel_last / sigma_rel_max_last (||A~_t - A_t||_2 / ||A_t||_2, median /
  max over blocks), sigma_max_last_undamped, rho_last / rho_last_undamped, frac_rho_gt1 / frac_rho_gt1_undamped (over all iterations),
  dm_last (Dennis-More ratio, median over blocks), err_ratio_last3 (||e^(i+1)||/||e^(i)|| over the last 3
  iterations), frac_err_increase, and, from the merge-tree replays (PB / Accel only): lam_tilde range,
  eta range (relative to a~ e^(lam~ n)), frac_lam_c_neg, c0 (rho term) range where lam_c < 0.

Usage (from experiments/): python theory/make_theory_summary.py
"""

import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SF = os.path.join(ROOT, "experiments", "saved_files")
OUT = os.path.join(ROOT, "results", "appendix")
LAMS = np.linspace(-0.6, -1e-4, 2400)


def load(name):
    p = os.path.join(SF, name)
    return pickle.load(open(p, "rb"))["records"] if os.path.exists(p) else []


quasi = (load("broyden_theory_diagnostics_quasi_sigma.pkl") or load("broyden_theory_diagnostics_quasi_sigma_last.pkl")
         or load("broyden_theory_diagnostics_quasi.pkl"))
groups = defaultdict(list)
for r in load("broyden_theory_diagnostics.pkl") + quasi:
    groups[("low-rank RNN", f"{r['model']} D={r['D']} dt={r['dt']}", r["method"])].append(r)
for r in load("broyden_theory_diagnostics_nschools.pkl"):
    groups[("N-schools", f"S={r['S']} D={r['D']}", r["method"])].append(r)

rows = []
for (bench, setting, method), rs in sorted(groups.items()):
    conv = [r["converged_at"] for r in rs]
    last = [r["per_iter"][-1] for r in rs if r["per_iter"]]
    P = [p for r in rs for p in r["per_iter"][1:]]
    rd = np.array([p["rho_damped"] for p in P], float)
    ru = np.array([p["rho_undamped"] for p in P], float)
    q, ql = [], []
    for r in rs:
        e = np.array([p["e_inf"] for p in r["per_iter"]])
        if len(e) > 2:
            qq = e[1:] / e[:-1]
            q += list(qq[1:])
            ql += list(qq[-3:])
    lt, en, lcn, c0 = [], [], [], []
    for r in rs:
        for p in r["per_iter"]:
            if "tree" not in p:
                continue
            n, eta, nPi, _ = p["tree"].T
            ok = np.isfinite(nPi) & np.isfinite(eta)
            n, eta, nPi = n[ok], eta[ok], nPi[ok]
            S, l, a = min((max(1.0, np.max(nPi * np.exp(-l * n))) / (1 - np.exp(l)), l,
                           max(1.0, np.max(nPi * np.exp(-l * n)))) for l in LAMS)
            e_ = np.max(eta * nPi / (a * np.exp(l * n)))
            lc = l + np.log1p(a * a * e_)
            lt.append(l); en.append(e_); lcn.append(lc < 0)
            if lc < 0 and np.isfinite(p["rho_damped"]):
                c0.append(a * p["rho_damped"] / (1 - np.exp(lc)))
    fin = lambda x: x[np.isfinite(x)]
    rng = lambda v: f"{min(v):.3g} .. {max(v):.3g}" if v else ""
    rows.append(dict(
        benchmark=bench, setting=setting, method=method, k=rs[0]["k"], T=rs[0]["T"], chains=len(rs),
        n_converged=sum(c is not None for c in conv),
        iters_median=float(np.median([c for c in conv if c is not None])) if any(c is not None for c in conv) else np.nan,
        sigma_max_last=float(np.nanmedian([p["sigma_max_damped"] for p in last])),
        sigma_med_last=float(np.nanmedian([p["sigma_med_damped"] for p in last])),
        sigma_rel_last=float(np.nanmedian([p["sigma_rel_med_damped"] for p in last])),
        sigma_rel_max_last=float(np.nanmedian([p["sigma_rel_max_damped"] for p in last])),
        sigma_max_last_undamped=float(np.nanmedian([p["sigma_max_undamped"] for p in last])),
        rho_last=float(np.nanmedian([p["rho_damped"] for p in last])),
        rho_last_undamped=float(np.nanmedian([p["rho_undamped"] for p in last])),
        frac_rho_gt1=float(np.mean(fin(rd) > 1)), frac_rho_gt1_undamped=float(np.mean(fin(ru) > 1)),
        dm_last=float(np.nanmedian([p["dm_med_damped"] for p in last])),
        err_ratio_last3=float(np.nanmedian(ql)), frac_err_increase=float(np.mean(fin(np.array(q)) > 1)),
        recon_err_median=float(np.nanmedian([p["recon_err"] for p in P])),
        lam_tilde=rng(lt), eta=rng(en), frac_lam_c_neg=float(np.mean(lcn)) if lcn else np.nan, c0_rho_term=rng(c0)))

df = pd.DataFrame(rows)
os.makedirs(OUT, exist_ok=True)
df.to_csv(os.path.join(OUT, "theory_summary.csv"), index=False)
pd.set_option("display.width", 250)
print(df[["benchmark", "setting", "method", "n_converged", "iters_median", "sigma_rel_last", "rho_last",
          "rho_last_undamped", "frac_rho_gt1", "frac_rho_gt1_undamped", "dm_last", "err_ratio_last3",
          "c0_rho_term"]].round(3).to_string(index=False))
print(f"-> {OUT}/theory_summary.csv")
