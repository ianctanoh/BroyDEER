"""
speedup_decomposition.py

Split the runtime advantage of the Broyden variants over each baseline into
    runtime ratio = iteration ratio x per-iteration cost ratio,
where the iteration ratio reflects approximation quality (the DPLR correction) and the per-iteration
cost ratio reflects the cost of obtaining the approximation (e.g. automatic differentiation).
No new runs: computed from the final result pickles, with each figure's own convention:
  low-rank RNN (Fig. 1): median iterations and median runtime over converged instances
  N-schools   (Fig. 2): median iterations and median runtime over all instances
  DiT         (Fig. 3): means over the 32 images
Ratios are baseline / Broyden variant (> 1: the Broyden variant is better).

Output: saved_files/speedup_decomposition.csv (and .pkl)
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SF = os.path.join(ROOT, "experiments", "saved_files")
p = argparse.ArgumentParser()
p.add_argument("--lowrank", default=os.path.join(SF, "2026-11-04_rnn_lowrank_final.pkl"),
               help="low-rank RNN results (rnn_lowrank_to_dataframe.py)")
p.add_argument("--nschools", default=os.path.join(SF, "2026-10-27_Nschools_final_float32.pkl"),
               help="N-schools results (wandb_sweep_to_pickle.py on Nschools_final_float32.yaml)")
p.add_argument("--dit", default=os.path.join(ROOT, "figs", "dit", "dit_32images_results.pkl"),
               help="DiT results (run_dit_32images.py)")
args = p.parse_args()
BROYDEN = {"broyden": "Parallel Broyden", "accelerated_broyden": "Accelerated Parallel Broyden"}
rows = []


def add(benchmark, setting, base, var, it_b, it_v, t_b, t_v):
    it_ratio, t_ratio = it_b / it_v, t_b / t_v
    rows.append(dict(benchmark=benchmark, setting=setting, baseline=base, method=var,
                     iters_baseline=it_b, iters_method=it_v, time_baseline=t_b, time_method=t_v,
                     iteration_ratio=it_ratio, per_iteration_cost_ratio=t_ratio / it_ratio,
                     runtime_ratio=t_ratio))


# --- low-rank RNN: converged instances ---
lr = pd.read_pickle(args.lowrank)
lr = lr[lr.error.isna()]
for (model, D, dt, T), g in lr.groupby(["model", "D", "dt", "T"]):
    by = {r.alg: r for r in g.itertuples()}
    setting = f"{model} D={D} dt={dt} T={T}"
    for base in ("quasi", "quasi_qmem", "deer"):
        for var in BROYDEN:
            if base in by and var in by:
                b, v = by[base], by[var]
                add("low-rank RNN", setting, base, var, b.n_iters_median_conv, v.n_iters_median_conv,
                    b.time_median, v.time_median)

# --- N-schools: all instances ---
ns = pd.read_pickle(args.nschools)
for (S, T), g in ns.groupby(["S", "T"]):
    by = {r.alg: r for r in g.itertuples()}
    setting = f"S={S} T={T}"
    for base in ("quasi", "deer"):
        for var in BROYDEN:
            if base in by and var in by:
                b, v = by[base], by[var]
                add("N-schools", setting, base, var, float(np.median(b.per_chain_n_iters)),
                    float(np.median(v.per_chain_n_iters)), float(np.median(b.per_chain_times)),
                    float(np.median(v.per_chain_times)))

# --- DiT: means over images ---
dit = pickle.load(open(args.dit, "rb"))
mean = lambda meth, f: float(np.mean(list(dit[meth][f].values())))
for base, bname in (("QuasiDEER", "quasi_hutchinson"), ("Picard", "picard"), ("TAA", "paratAA")):
    for var, vname in (("Broyden", "broyden"), ("AccBroyden", "accelerated_broyden")):
        add("DiT", "32 images", bname, vname, mean(base, "iters"), mean(var, "iters"),
            mean(base, "runtime"), mean(var, "runtime"))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(SF, "speedup_decomposition.csv"), index=False)
df.to_pickle(os.path.join(SF, "speedup_decomposition.pkl"))
pd.set_option("display.width", 220)
show = df[df.baseline.isin(["quasi", "quasi_hutchinson"])]
print(show[["benchmark", "setting", "method", "iters_baseline", "iters_method", "iteration_ratio",
            "per_iteration_cost_ratio", "runtime_ratio"]].round(2).to_string(index=False))
print(f"\n{len(df)} rows -> {SF}/speedup_decomposition.csv")
