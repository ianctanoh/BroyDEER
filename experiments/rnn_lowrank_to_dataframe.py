"""
rnn_lowrank_to_dataframe.py

Collect the harness_rnn_lowrank.py outputs (one .pkl per job) into one pandas DataFrame, in the
layout of the N-schools sweep pickles (saved_files/*_Nschools_*.pkl): one row per
(model, D, L, dt, T, alg) with per-chain lists and summary columns.

Algorithm names follow the N-schools pickles: seq, deer (Newton), quasi (exact diagonal),
quasi_qmem (Hutchinson, num_samples probes), broyden, accelerated_broyden.

Per-chain times: mean of the timed solves; NaN for a chain that did not converge (it is solved
once and not timed). Summary times (time_mean, time_sd, time_median, time_q25, time_q75) are over
converged chains; iteration summaries are over all chains (non-converged at num_iters) unless
the column name says "_conv".

Usage:
    python rnn_lowrank_to_dataframe.py IN_DIR OUT.pkl
"""

import sys, os, glob, pickle

import numpy as np
import pandas as pd

ALG = dict(seq="seq", newton="deer", quasi="quasi", quasi_hutch="quasi_qmem",
           broyden="broyden", accel="accelerated_broyden", jacobi="jacobi")

in_dir, out = sys.argv[1], sys.argv[2]
records = []
for path in sorted(glob.glob(os.path.join(in_dir, "rnn_lowrank_*.pkl"))):
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    a, num_iters, tol = d["args"], d["num_iters"], d["tol"]
    for r in d["rows"]:
        rec = dict(model=r["model"], D=r["D"], L=r["L"], dt=r["dt"], T=r["T"],
                   alg=ALG.get(r["method"], r["method"]), method=r["method"], k=r["k"],
                   gamma=1.0 - r["dt"] / 10.0, a=1.0 - r["dt"] / 10.0,
                   line_search=r.get("line_search"), ls_c=r.get("ls_c"),
                   num_samples=a["quasi_probes"] if r["method"] == "quasi_hutch" else None,
                   m=3 if r["method"] in ("broyden", "accel") else None,
                   eps_gram=1e-6 if r["method"] == "accel" else None,
                   compress="rand_rr" if r["method"] in ("broyden", "accel") else None,
                   tol=tol, num_iters=num_iters, nwarmups=a["nwarmups"], nreps=a["nreps"],
                   batch_size=a["chains"], R=3, tau=10.0, g=2.0, sigma=0.1, beta=0.5,
                   gpu=r.get("gpu"), seeds=r["seeds"], seeds_tried=r["seeds_tried"],
                   source_file=os.path.basename(path), error=r.get("error"))
        if "error" not in r:
            it = np.asarray(r["n_iters"])
            t = np.asarray(r["times"], dtype=float)
            conv = np.ones(len(it), bool) if r["method"] == "seq" else it < num_iters
            tc = t[conv]
            rec.update(per_chain_n_iters=it.tolist(), per_chain_times=t.tolist(),
                       per_chain_converged=conv.tolist(),
                       n_converged=int(conv.sum()), frac_converged=float(conv.mean()),
                       n_iters_mean=float(it.mean()), n_iters_sd=float(it.std()),
                       n_iters_median=float(np.median(it)),
                       n_iters_q25=float(np.percentile(it, 25)),
                       n_iters_q75=float(np.percentile(it, 75)),
                       n_iters_median_conv=float(np.median(it[conv])) if conv.any() else np.nan,
                       time_mean=float(tc.mean()) if conv.any() else np.nan,
                       time_sd=float(tc.std()) if conv.any() else np.nan,
                       time_median=float(np.median(tc)) if conv.any() else np.nan,
                       time_q25=float(np.percentile(tc, 25)) if conv.any() else np.nan,
                       time_q75=float(np.percentile(tc, 75)) if conv.any() else np.nan)
        records.append(rec)

df = pd.DataFrame(records)
# speedup vs sequential (median times over converged chains) for convenience
seq = df[df.alg == "seq"].set_index(["model", "D", "L", "dt", "T"])["time_median"]
df["seq_time_median"] = [seq.get((r.model, r.D, r.L, r.dt, r.T), np.nan) for r in df.itertuples()]
df["speedup_vs_seq"] = df["seq_time_median"] / df["time_median"]
df.to_pickle(out)
print(f"{len(df)} rows -> {out}")
print(df.groupby(["model", "D", "dt", "T"]).size().to_string())
