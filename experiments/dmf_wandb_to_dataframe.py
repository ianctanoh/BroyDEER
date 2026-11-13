"""
dmf_wandb_to_dataframe.py

Collect harness_dmf_rnn.py runs from the local wandb/ run folders into one pandas DataFrame
(layout of the N-schools / low-rank RNN pickles): one row per run, with the run config, the
per-chain lists (per_chain_n_iters, per_chain_times, per_chain_final_rms; the harness logs them
as per_seed_*, but they are chains: one W, batch_size chains of inputs / initial guesses) and
median / IQR summaries. Times are per chain (mean of nreps timed solves); summary times are over
converged chains, iteration summaries over all chains (non-converged at num_iters).

Usage:
    python dmf_wandb_to_dataframe.py OUT.pkl [--since 20261104_210000] [--T 100000] [--batch_size 32]
"""

import argparse, glob, json, os

import numpy as np
import pandas as pd
import yaml

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--wandb_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "wandb"))
p.add_argument("--since", default=None, help="only runs started at or after YYYYMMDD_HHMMSS")
p.add_argument("--T", type=int, default=None)
p.add_argument("--batch_size", type=int, default=None)
args = p.parse_args()

records = []
for d in sorted(glob.glob(os.path.join(args.wandb_dir, "run-*"))):
    stamp, run_id = os.path.basename(d)[4:19], os.path.basename(d).split("-")[-1]
    if args.since and stamp < args.since:
        continue
    cf, sf = os.path.join(d, "files", "config.yaml"), os.path.join(d, "files", "wandb-summary.json")
    if not (os.path.exists(cf) and os.path.exists(sf)):
        continue
    c = {k: (v.get("value") if isinstance(v, dict) else v) for k, v in yaml.safe_load(open(cf)).items()
         if not k.startswith("_")}
    if "g" not in c or "self_couple" not in c:                     # not a DMF-RNN run
        continue
    if (args.T and c["T"] != args.T) or (args.batch_size and c["batch_size"] != args.batch_size):
        continue
    s = json.load(open(sf))
    it = np.asarray(s.get("per_seed_n_iters", []), dtype=float)
    if it.size == 0:                                               # run did not finish
        continue
    t = np.asarray(s.get("per_seed_times", []), dtype=float)
    rms = np.asarray(s.get("per_seed_final_rms", []), dtype=float)
    alg = c["alg"]
    conv = np.ones(it.size, bool) if alg == "seq" else it < c["num_iters"]
    rec = dict(run_id=run_id, started=stamp, **c)
    rec.update(m=c["acc_m"] if alg == "accelerated_broyden" else c["bd_m"] if alg == "broyden" else None,
               gamma=c["acc_gamma"] if alg == "accelerated_broyden" else c["bd_gamma"] if alg == "broyden" else None,
               init=c["acc_init"] if alg == "accelerated_broyden" else c["bd_init"] if alg == "broyden" else None,
               compress=c["acc_compress"] if alg == "accelerated_broyden" else c["bd_compress"] if alg == "broyden" else None,
               damping=c["deer_k"] if alg in ("deer", "quasi") else c["acc_k"] if alg == "accelerated_broyden"
               else c["bd_k"] if alg == "broyden" else None,
               per_chain_n_iters=it.tolist(), per_chain_times=t.tolist(), per_chain_final_rms=rms.tolist(),
               per_chain_converged=conv.tolist(), n_chains=int(it.size),
               n_converged=int(conv.sum()), frac_converged=float(conv.mean()),
               n_iters_median=float(np.median(it)), n_iters_q25=float(np.percentile(it, 25)),
               n_iters_q75=float(np.percentile(it, 75)), n_iters_mean=float(it.mean()),
               n_iters_sd=float(it.std()))
    tc = t[conv]
    rec.update(time_median=float(np.median(tc)) if tc.size else np.nan,
               time_q25=float(np.percentile(tc, 25)) if tc.size else np.nan,
               time_q75=float(np.percentile(tc, 75)) if tc.size else np.nan,
               time_mean=float(tc.mean()) if tc.size else np.nan)
    records.append(rec)

df = pd.DataFrame(records)
key = ["D", "T", "g", "self_couple", "batch_size", "tol", "num_iters"]
seq = df[df.alg == "seq"].groupby(key)["time_median"].first()
df["seq_time_median"] = [seq.get(tuple(r[k] for k in key), np.nan) for _, r in df.iterrows()]
df["speedup_vs_seq"] = df["seq_time_median"] / df["time_median"]
df.to_pickle(args.out)
cols = ["started", "alg", "m", "gamma", "compress", "damping", "n_converged", "n_chains",
        "n_iters_median", "n_iters_q25", "n_iters_q75", "time_median", "speedup_vs_seq"]
print(df[cols].to_string(index=False))
print(f"\n{len(df)} runs -> {args.out}")
