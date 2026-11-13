"""
nschools_wandb_to_dataframe.py

Collect harness_Nschools.py runs from the local wandb/ run folders into one pandas DataFrame, for
runs launched as a Hydra multirun (python harness_Nschools.py -m ...) rather than as a wandb sweep
(for sweeps, use wandb_sweep_to_pickle.py). One row per run: the run config, the logged summary
(per_chain_n_iters, per_chain_times, per_chain_final_rms, per_chain_is_nan, n_converged, ...) and
the medians over all chains, time_median_all and n_iters_median (non-converged chains at num_iters
and at their actual runtime, as in the N-schools figure).

Usage:
    python nschools_wandb_to_dataframe.py OUT.pkl [--since YYYYMMDD_HHMMSS] [--until YYYYMMDD_HHMMSS] [--alg quasi_qmem]
e.g. the Hutchinson table (Appendix):
    python nschools_wandb_to_dataframe.py saved_files/<date>_Nschools_hutchinson_float32.pkl --since <start> --alg quasi_qmem
"""

import argparse, glob, json, os

import numpy as np
import pandas as pd
import yaml

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--wandb_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "wandb"))
p.add_argument("--since", default=None, help="only runs started at or after YYYYMMDD_HHMMSS")
p.add_argument("--until", default=None, help="only runs started at or before YYYYMMDD_HHMMSS")
p.add_argument("--alg", nargs="+", default=None, help="keep only these algorithms")
args = p.parse_args()

SUMMARY = ["per_chain_n_iters", "per_chain_times", "per_chain_final_rms", "per_chain_is_nan", "n_converged",
           "frac_converged", "n_iters_mean", "n_iters_sd", "time_mean", "time_sd", "n_nan", "D"]
records = []
for d in sorted(glob.glob(os.path.join(args.wandb_dir, "run-*"))):
    stamp, run_id = os.path.basename(d)[4:19], os.path.basename(d).split("-")[-1]
    if (args.since and stamp < args.since) or (args.until and stamp > args.until):
        continue
    cf, sf = os.path.join(d, "files", "config.yaml"), os.path.join(d, "files", "wandb-summary.json")
    if not (os.path.exists(cf) and os.path.exists(sf)):
        continue
    c = {k: (v.get("value") if isinstance(v, dict) else v) for k, v in yaml.safe_load(open(cf)).items()
         if not k.startswith("_")}
    if "S" not in c or "N_s" not in c:                             # not an N-schools run
        continue
    if args.alg and c["alg"] not in args.alg:
        continue
    s = json.load(open(sf))
    if not s.get("per_chain_n_iters"):                             # run did not finish
        continue
    rec = dict(c, run_id=run_id, started=stamp, **{k: s.get(k) for k in SUMMARY})
    rec["time_median_all"] = float(np.median(s["per_chain_times"]))
    rec["n_iters_median"] = float(np.median(s["per_chain_n_iters"]))
    records.append(rec)

df = pd.DataFrame(records)
df.to_pickle(args.out)
print(df[["started", "alg", "S", "T", "num_samples", "n_converged", "n_iters_median", "time_median_all"]].to_string(index=False))
print(f"\n{len(df)} runs -> {args.out}")
