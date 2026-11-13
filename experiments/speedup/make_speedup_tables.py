"""
make_speedup_tables.py

Appendix tables "where the speedup comes from", from saved_files/speedup_decomposition.csv
(speedup_decomposition.py) and saved_files/wallclock_breakdown.csv (wallclock_breakdown.py).
Writes, in results/appendix/: both input CSVs, wallclock_breakdown_table.csv, and the LaTeX tables
table_decomposition.tex and table_wallclock.tex (one representative setting per benchmark).

Wall-clock shares: components are timed in isolation, so inside the solver XLA fuses part of them and
their sum differs from the measured iteration time. Shares are therefore given as fractions of the
summed component time (they add to 100%), together with the ratio (summed components) / (measured
iteration). The measured iteration time excludes the work done once before the loop (the DDIM solvers
evaluate the denoiser twice (Broyden variants) or once (diagonal quasi-Newton) before iterating; the
end-to-end runs had 10 iterations, so this setup is subtracted as n_setup * f_eval / 10).
DiT rows come from wallclock_breakdown_dit_default_precision.csv (run with --precision default, as the
official DiT run, which sets no matmul precision); the other rows use 'highest', as their harnesses.

Usage (from experiments/): python speedup/make_speedup_tables.py
"""

import os
import shutil

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SF = os.path.join(ROOT, "experiments", "saved_files")
OUT = os.path.join(ROOT, "results", "appendix")
os.makedirs(OUT, exist_ok=True)
for f in ("speedup_decomposition.csv", "wallclock_breakdown.csv", "wallclock_breakdown_dit_default_precision.csv"):
    shutil.copy(os.path.join(SF, f), os.path.join(OUT, f))

NAME = {"quasi": "Diag.\\ QN", "quasi_qmem": "Diag.\\ QN (Hutchinson)", "quasi_hutchinson": "Diag.\\ QN (Hutchinson)",
        "deer": "Newton", "broyden": "PB", "accelerated_broyden": "Accel.\\ PB"}

# ---------------------------------------------------------------- decomposition
dec = pd.read_csv(os.path.join(OUT, "speedup_decomposition.csv"))
ROWS = [  # (benchmark, setting in csv, label, baselines)
    ("low-rank RNN", "mlp D=200 dt=3.0 T=100000", "Low-rank MLP-RNN", ["quasi", "quasi_qmem", "deer"]),
    ("low-rank RNN", "std D=200 dt=0.5 T=100000", "Standard low-rank RNN", ["quasi", "deer"]),
    ("N-schools", "S=100 T=100000", "Gibbs, $S=100$", ["quasi", "deer"]),
    ("N-schools", "S=200 T=100000", "Gibbs, $S=200$", ["quasi"]),
    ("DiT", "32 images", "DiT", ["quasi_hutchinson"]),
]
lines = []
for bench, setting, label, bases in ROWS:
    g = dec[(dec.benchmark == bench) & (dec.setting == setting)]
    first = True
    for b in bases:
        for m in ("broyden", "accelerated_broyden"):
            r = g[(g.baseline == b) & (g.method == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(f"{label if first else '':22s} & {NAME[b]} & {NAME[m]} & "
                         f"{r.iters_baseline:.4g} / {r.iters_method:.4g} & {r.iteration_ratio:.2f} & "
                         f"{r.per_iteration_cost_ratio:.2f} & {r.runtime_ratio:.2f} \\\\")
            first = False
    lines.append("\\midrule")
lines = lines[:-1]
tex = r"""\begin{table}[h]
    \centering
    \caption{\textbf{Decomposition of the runtime ratio.} Runtime ratio $=$ iteration ratio $\times$ per-iteration
    cost ratio, baseline over Broyden variant ($>1$: the Broyden variant is faster). $T=10^5$ for the RNNs and the
    Gibbs sampler ($D=200$; low-rank MLP-RNN: $\Delta t=3$, standard low-rank RNN: $\Delta t=0.5$). Iterations:
    baseline / method. For the RNNs, medians are over converged chains, so that iteration and runtime ratios refer
    to the same chains; Figure~\ref{fig:fig1} instead counts non-converged chains at the iteration cap. For the
    Gibbs sampler, medians are over all chains, and for DiT, means are over the $32$ images, as in the corresponding
    figures.}
    \label{tab:decomposition}
    \small
    \begin{tabular}{lllrrrr}
        \toprule
        Benchmark & Baseline & Method & Iterations & Iteration ratio & Per-iteration ratio & Runtime ratio \\
        \midrule
""" + "\n".join("        " + l for l in lines) + r"""
        \bottomrule
    \end{tabular}
\end{table}
"""
open(os.path.join(OUT, "table_decomposition.tex"), "w").write(tex)

# ---------------------------------------------------------------- wall-clock
wc = pd.read_csv(os.path.join(OUT, "wallclock_breakdown.csv"))
wc = pd.concat([wc[wc.benchmark != "DiT"], pd.read_csv(os.path.join(OUT, "wallclock_breakdown_dit_default_precision.csv"))])
GROUP = {"f_eval": "f", "jac_diag": "autodiff", "jac_full": "autodiff", "hutch": "autodiff",
         "broyden_update": "update", "accel_fit": "update",
         "scan_diag": "scan", "scan_full": "scan", "broyden_scan": "scan", "accel_scan": "scan"}
METH = {"quasi": "Diag.\\ QN", "quasi_hutch": "Diag.\\ QN (Hutchinson)", "newton": "Newton", "broyden": "PB",
        "accel": "Accel.\\ PB", "dit_quasi": "Diag.\\ QN (Hutchinson)", "dit_broyden": "PB", "dit_accel": "Accel.\\ PB"}
SETUP_EVALS = {"dit_quasi": 1, "dit_broyden": 2, "dit_accel": 2}            # denoiser calls before the loop
E2E_ITERS = 10
SEL = [("low-rank RNN", "mlp D=200 L=8 dt=3.0 T=100000", "Low-rank MLP-RNN", ["quasi", "quasi_hutch", "newton", "broyden", "accel"]),
       ("low-rank RNN", "std D=200 L=0 dt=0.5 T=100000", "Standard low-rank RNN", ["quasi", "newton", "broyden", "accel"]),
       ("N-schools", "S=100 D=202 T=100000", "Gibbs, $S=100$", ["quasi", "newton", "broyden", "accel"]),
       ("DiT", "DiT-XL/2 T=50 D=4096 cfg=5 (orangutan)", "DiT", ["dit_quasi", "dit_broyden", "dit_accel"])]
rows, lines = [], []
for bench, setting, label, meths in SEL:
    g = wc[(wc.benchmark == bench) & (wc.setting == setting)]
    feval = g[(g.method.isna()) & (g.component == "f_eval")].time_s.iloc[0]
    first = True
    for m in meths:
        gm = g[g.method == m]
        e2e = gm[gm.component == "end_to_end"].time_s.iloc[0] - SETUP_EVALS.get(m, 0) * feval / E2E_ITERS
        parts = gm[gm.component.isin(GROUP)]
        tot = parts.time_s.sum()
        share = {k: 0.0 for k in ("f", "autodiff", "update", "scan")}
        for r in parts.itertuples():
            share[GROUP[r.component]] += r.time_s / tot
        rows.append(dict(benchmark=label, method=METH[m], ms_per_iter=1e3 * e2e, **{f"share_{k}": v for k, v in share.items()},
                         sum_components_over_measured=tot / e2e))
        pct = lambda v: "--" if v == 0 else f"{100 * v:.0f}" if v >= 0.01 else "$<1$"
        ms = f"{1e3 * e2e:.0f}" if e2e >= 0.1 else f"{1e3 * e2e:.3g}"
        lines.append(f"{label if first else '':22s} & {METH[m]} & {ms} & {pct(share['f'])} & "
                     f"{pct(share['autodiff'])} & {pct(share['update'])} & {pct(share['scan'])} & {tot / e2e:.2f} \\\\")
        first = False
    lines.append("\\midrule")
lines = lines[:-1]
pd.DataFrame(rows).to_csv(os.path.join(OUT, "wallclock_breakdown_table.csv"), index=False)
tex = r"""\begin{table}[h]
    \centering
    \caption{\textbf{Wall-clock breakdown of one iteration} (H100, float32, $T=10^5$ except DiT). Measured time per
    iteration, and share of each component: evaluations of the transition $f$ (for DiT, the denoiser),
    automatic differentiation (Jacobian or its diagonal by reverse mode; Hutchinson: Jacobian--vector products),
    Broyden update (rank-one updates or refit of $U_t$), and parallel scan (with rank compression for the
    Broyden variants). Components are timed in isolation; shares are fractions of their summed time, and the last
    column compares this sum with the measured iteration, in which XLA fuses part of the work (on the Gibbs
    sampler, the isolated components of PB and Accel.\ PB take the same time, but the measured Accel.\ PB iteration
    is faster, also in the sweep: $5.4$ vs.\ $8.2$\,ms).
    Matmul precision as in the corresponding experiments (DiT: TF32), so the measured times match the sweeps.}
    \label{tab:wallclock}
    \small
    \begin{tabular}{llrrrrrr}
        \toprule
        Benchmark & Method & ms / iter. & $f$ (\%) & Autodiff (\%) & Update (\%) & Scan (\%) & Sum / measured \\
        \midrule
""" + "\n".join("        " + l for l in lines) + r"""
        \bottomrule
    \end{tabular}
\end{table}
"""
open(os.path.join(OUT, "table_wallclock.tex"), "w").write(tex)
print(open(os.path.join(OUT, "table_decomposition.tex")).read())
print(open(os.path.join(OUT, "table_wallclock.tex")).read())
print(f"-> {OUT}")
