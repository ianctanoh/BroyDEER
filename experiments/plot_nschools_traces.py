"""
plot_nschools_traces.py

Trace plots: sequential Gibbs chain vs each parallel solver's trajectory for one
N-schools chain, from the .npz files written by
    nschools_tol_study.py --save_chain <i> --traj_dir <DIR>

One figure per solver. Rows are variables: theta_1, sigma^2_1, mu, tau^2 and the
coordinate where the solver's trajectory differs most from sequential. Columns:
full chain (sequential and solver overlaid), a zoom on the steps around that
variable's largest difference, and the difference in units of the variable's
posterior sd (after 10% burn-in).

Usage:
    python plot_nschools_traces.py --traj_dir DIR --S 100 --T 100000 --chain 7 --tol 1e-3
        [--out_dir ../figs/nschools_traces]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALGS = {"deer": "DEER", "quasi": "quasi-DEER", "broyden": "Parallel Broyden",
        "accelerated_broyden": "Accel. Parallel Broyden"}
BURNIN_FRAC = 0.1
ZOOM = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--traj_dir", required=True)
parser.add_argument("--S", type=int, default=100)
parser.add_argument("--T", type=int, default=100000)
parser.add_argument("--chain", type=int, default=7)
parser.add_argument("--tol", type=float, default=1e-3)
parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "figs",
                                                      "nschools_traces"))
args = parser.parse_args()

S, T = args.S, args.T
stem = f"traj_S{S}_T{T}_chain{args.chain}"
seq = np.load(os.path.join(args.traj_dir, f"{stem}_seq.npz"))["traj"].astype(np.float64)
b = int(BURNIN_FRAC * T)
sd = seq[b:].std(axis=0) + 1e-30


def name(j):
    if j < S:
        return rf"$\theta_{{{j + 1}}}$"
    if j < 2 * S:
        return rf"$\sigma^2_{{{j - S + 1}}}$"
    return r"$\mu$" if j == 2 * S else r"$\tau^2$"


os.makedirs(args.out_dir, exist_ok=True)
t = np.arange(1, T + 1)
for alg, label in ALGS.items():
    path = os.path.join(args.traj_dir, f"{stem}_{alg}_tol{args.tol:g}.npz")
    if not os.path.exists(path):
        print(f"skip {alg}: {path} not found")
        continue
    par = np.load(path)["traj"].astype(np.float64)
    diff = (par - seq) / sd                              # in posterior sds
    if not np.isfinite(diff[b:]).any(axis=0).any():
        print(f"skip {alg}: trajectory is not finite (solver did not converge on this chain)")
        continue
    worst = int(np.nanargmax(np.nanmax(np.abs(diff[b:]), axis=0)))
    coords = [0, S, 2 * S, 2 * S + 1]
    if worst not in coords:
        coords.append(worst)

    fig, axes = plt.subplots(len(coords), 3, figsize=(13, 2.1 * len(coords)),
                             gridspec_kw={"width_ratios": [2.2, 1, 1.4]})
    for row, j in enumerate(coords):
        tpk = b + int(np.nanargmax(np.abs(diff[b:, j])))
        lo, hi = max(0, tpk - ZOOM // 2), min(T, tpk + ZOOM // 2)
        ax = axes[row, 0]
        ax.plot(t, seq[:, j], lw=0.4, color="k", label="sequential")
        ax.plot(t, par[:, j], lw=0.4, color="tab:red", alpha=0.7, label=label)
        ax.axvspan(lo, hi, color="tab:blue", alpha=0.15)
        ax.set_ylabel(name(j) + (" (worst)" if j == worst else ""))
        ax = axes[row, 1]
        ax.plot(t[lo:hi], seq[lo:hi, j], lw=1.2, color="k")
        ax.plot(t[lo:hi], par[lo:hi, j], lw=0.8, color="tab:red", ls="--")
        ax = axes[row, 2]
        ax.plot(t, diff[:, j], lw=0.4, color="tab:red")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("diff / post. sd")
        ax.text(0.02, 0.85, f"max |diff| = {np.nanmax(np.abs(diff[b:, j])):.1e} sd",
                transform=ax.transAxes, fontsize=8)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[0, 0].set_title("full chain")
    axes[0, 1].set_title(f"zoom: {ZOOM} steps around largest difference")
    axes[0, 2].set_title("parallel − sequential (posterior sds)")
    for ax in axes[-1]:
        ax.set_xlabel("Gibbs step t")
    fig.suptitle(f"N-schools S={S}, T={T}, chain {args.chain}, float32, tol={args.tol:g}: "
                 f"sequential vs {label}", fontsize=11)
    fig.tight_layout()
    out = os.path.join(args.out_dir, f"traces_S{S}_T{T}_chain{args.chain}_{alg}_tol{args.tol:g}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}  (worst coordinate {name(worst)}: {np.nanmax(np.abs(diff[b:, worst])):.2e} sd)")
