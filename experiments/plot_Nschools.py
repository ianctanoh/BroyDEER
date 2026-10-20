"""
plot_Nschools.py

Generate N-schools iteration count and wallclock plots from a wandb sweep pickle.

Usage:
    python plot_Nschools.py --pickle saved_files/2026-04-17_Nschools_alg_comparison_mg52khwf.pkl
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---- style ----
plt.rcParams.update({"text.usetex": False, "font.family": "serif"})
SMALL_SIZE = 8
MEDIUM_SIZE = 9
BIGGER_SIZE = 11
plt.rc("font", size=SMALL_SIZE)
plt.rc("axes", titlesize=SMALL_SIZE)
plt.rc("axes", labelsize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=SMALL_SIZE)
plt.rc("ytick", labelsize=SMALL_SIZE)
plt.rc("legend", fontsize=MEDIUM_SIZE)
plt.rc("figure", titlesize=BIGGER_SIZE)

# ---- per-algorithm aesthetics ----
ALGS = {
    "DEER":           {"tag": "deer",                "col": "tab:red",    "ls": "-",  "marker": "+"},
    "quasi-DEER":     {"tag": "quasi",               "col": "tab:orange", "ls": "-",  "marker": "x"},
    "Jacobi":         {"tag": "jacobi",              "col": "tab:gray",   "ls": ":",  "marker": "s"},
    "Accel. Broyden": {"tag": "accelerated_broyden", "col": "tab:purple", "ls": "--", "marker": "v"},
}


def make_plot(data, save_dir="../figs/"):
    Ss = np.sort(data["S"].dropna().unique())

    # filter to algorithms present in data
    labs = [name for name, cfg in ALGS.items() if cfg["tag"] in data["alg"].unique()]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(4, 4))

    for label in labs:
        cfg = ALGS[label]
        tag = cfg["tag"]
        iters_vals, time_vals = [], []
        Ss_valid = []
        for S in Ss:
            row = data[(data["S"] == S) & (data["alg"] == tag)]
            if row.empty or row["n_iters"].isna().all():
                continue
            iters_vals.append(row["n_iters"].values[0])
            time_vals.append(row["time"].values[0])
            Ss_valid.append(S)

        axes[0].plot(Ss_valid, iters_vals, c=cfg["col"], marker=cfg["marker"],
                     linestyle=cfg["ls"], markersize=5, linewidth=1, label=label)
        axes[1].plot(Ss_valid, time_vals, c=cfg["col"], marker=cfg["marker"],
                     linestyle=cfg["ls"], markersize=5, linewidth=1, label=label)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Number of\nFixed-Point Iters")
    axes[1].set_ylabel("Wallclock (s)")
    axes[1].set_xlabel("Number of Schools ($S$)")

    axes[0].legend(ncols=2, handletextpad=0.5, loc="upper left")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "Nschools.pdf")
    plt.savefig(out_path)
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle", type=str, required=True, help="Path to pickle file")
    parser.add_argument("--save_dir", type=str, default="../figs/", help="Output directory")
    args = parser.parse_args()

    with open(args.pickle, "rb") as f:
        data = pickle.load(f)

    print(data[["alg", "S", "T", "time", "n_iters"]].to_string())
    make_plot(data, save_dir=args.save_dir)
