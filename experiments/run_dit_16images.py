"""
run_dit_16images.py

Run 3 best methods on 16 different ImageNet classes (animals).
Save per-image runtime, LPIPS, and iteration count as pkl.

Methods:
  1. Broyden:      m=2, gamma=1.0, identity
  2. AccBroyden:   m=2, gamma=1.0, identity
  3. TAA:          k=6, m=2

Usage:
    sbatch run_dit_16images.sh
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import time
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algs.taa import taa_ddim_alg
from algs.broyden_ddim import broyden_ddim_alg
from algs.accelerated_broyden_ddim import accelerated_broyden_ddim_alg
from algs.deer import deer_alg
from models.dit import make_dit_diffusion_reusable

# ── config ──
dit_ckpt  = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "pretrained_models", "DiT-XL-2-256x256.pt")
dit_T     = 50
dit_cfg   = 5.0
dit_seed  = 0
max_n     = 50
tol       = 1.0
nwarmups  = 2
nreps     = 3
save_dir  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figs", "dit")

# 16 ImageNet animal classes
class_labels = {
    "goldfish": 1,
    "tiger": 292,
    "zebra": 340,
    "giant_panda": 388,
    "golden_retriever": 207,
    "tabby_cat": 281,
    "hummingbird": 94,
    "african_elephant": 386,
    "gorilla": 366,
    "red_fox": 277,
    "timber_wolf": 269,
    "flamingo": 130,
    "peacock": 84,
    "jellyfish": 107,
    "koala": 105,
    "arctic_fox": 279,
}

print(f"Device: {jax.devices()}")
print(f"T={dit_T}, cfg={dit_cfg}, max_n={max_n}, tol={tol}")
print(f"Classes: {list(class_labels.keys())}")
sys.stdout.flush()

# latent shape (same for all classes)
lat_C, lat_H, lat_W = 4, 32, 32
D = lat_C * lat_H * lat_W
print(f"D={D}, T={dit_T}")
sys.stdout.flush()

# ── run all classes ──
print(f"\n--- Running 16 classes x 5 methods ---")

# results storage
results = {
    "config": {
        "T": dit_T, "cfg": dit_cfg, "tol": tol, "max_n": max_n,
        "seed": dit_seed, "classes": class_labels,
    },
    "Sequential": {"runtime": {}, "iters": {}, "lpips": {}, "latent": {}},
    "QuasiDEER": {"runtime": {}, "iters": {}, "lpips": {}, "latent": {}},
    "Broyden": {"runtime": {}, "iters": {}, "lpips": {}, "latent": {}},
    "AccBroyden": {"runtime": {}, "iters": {}, "lpips": {}, "latent": {}},
    "TAA": {"runtime": {}, "iters": {}, "lpips": {}, "latent": {}},
}

# ── build reusable dynamics (compile once) ──
# The class label is a traced leaf of `dynamics`, so the jitted functions
# below take `dynamics` as an argument and compile once for all classes.
print("\nBuilding reusable DiT dynamics ...")
sys.stdout.flush()
t0 = time.time()
set_class, _ = make_dit_diffusion_reusable(
    dit_ckpt, T=dit_T, cfg_scale=dit_cfg, eta=0.0, seed=dit_seed,
)
print(f"Done in {time.time()-t0:.1f}s")

# x0 and drivers are class-independent
dynamics_0, x0, drivers = set_class(list(class_labels.values())[0])
guess = jnp.broadcast_to(x0, (dit_T, D))

@jax.jit
def run_seq(dynamics, x0, drivers):
    _, traj = jax.lax.scan(
        lambda s, d: (dynamics.deer_fxn(s, d), dynamics.deer_fxn(s, d)), x0, drivers
    )
    return traj

@jax.jit
def run_broyden(dynamics, x0, guess, drivers):
    res = broyden_ddim_alg(dynamics, x0, guess, drivers, max_n,
                           m=2, init="identity", gamma=1.0, k=0.0,
                           full_trace=False, tol=tol,
                           sequential_solve=False, compress="rand")
    return res[1], res[2]

@jax.jit
def run_accbroy(dynamics, x0, guess, drivers):
    res = accelerated_broyden_ddim_alg(dynamics, x0, guess, drivers, max_n,
                                       m=2, init="identity", gamma=1.0, k=0.0,
                                       full_trace=False, tol=tol,
                                       sequential_solve=True)
    return res[1], res[2]

@jax.jit
def run_quasi_deer(dynamics, x0, guess, drivers):
    res = deer_alg(lambda s, d: dynamics.deer_fxn(s, d), x0, guess, drivers, max_n,
                   quasi=True, qmem_efficient=True, full_trace=False, tol=tol)
    return res[1], res[2]

@jax.jit
def run_taa(dynamics, x0, guess, drivers):
    res = taa_ddim_alg(dynamics, x0, guess, drivers, max_n,
                       order=6, memory_size=2, lmd=1e-3,
                       full_trace=False, tol=tol)
    return res[1], res[2]

methods = {
    "QuasiDEER": run_quasi_deer,
    "Broyden": run_broyden,
    "AccBroyden": run_accbroy,
    "TAA": run_taa,
}

# ── compile once ──
print("\nCompiling (one-time) ...", end=" ", flush=True)
run_seq.lower(dynamics_0, x0, drivers).compile()
for mfn in methods.values():
    mfn.lower(dynamics_0, x0, guess, drivers).compile()
print("done.")
sys.stdout.flush()

for cls_name, cls_id in class_labels.items():
    print(f"\n  == {cls_name} (class {cls_id}) ==")
    sys.stdout.flush()

    dynamics_cls, _, _ = set_class(cls_id)

    # warmup
    _ = run_seq(dynamics_cls, x0, drivers)
    for mfn in methods.values():
        _ = mfn(dynamics_cls, x0, guess, drivers)
    jax.block_until_ready(_)

    # --- Sequential ---
    t0 = time.time()
    for _ in range(nreps):
        seq_result = run_seq(dynamics_cls, x0, drivers)
        jax.block_until_ready(seq_result)
    t_seq = (time.time() - t0) / nreps
    seq_traj = np.array(seq_result)
    results["Sequential"]["runtime"][cls_name] = t_seq
    results["Sequential"]["iters"][cls_name] = dit_T
    results["Sequential"]["latent"][cls_name] = np.array(seq_traj[-1])
    print(f"    Sequential: {dit_T} steps, {t_seq:.3f}s")

    # --- Parallel methods ---
    for mname, mfn in methods.items():
        t0 = time.time()
        for _ in range(nreps):
            res = mfn(dynamics_cls, x0, guess, drivers)
            jax.block_until_ready(res)
        t_avg = (time.time() - t0) / nreps
        final, steps = np.array(res[0]), int(res[1])
        results[mname]["runtime"][cls_name] = t_avg
        results[mname]["iters"][cls_name] = steps
        results[mname]["latent"][cls_name] = np.array(final[-1])
        print(f"    {mname}: {steps} iters, {t_avg:.3f}s")
    sys.stdout.flush()

# ── save pkl (before loading PyTorch to avoid OOM) ──
os.makedirs(save_dir, exist_ok=True)
pkl_path = os.path.join(save_dir, "dit_16images_results.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(results, f)
print(f"\nSaved results to {pkl_path}")

# ── print summary (iters + runtime only, LPIPS computed separately) ──
print("\n" + "=" * 60)
print(f"  {'Class':<20s}  {'Method':<12s}  {'Iters':>5s}  {'Time':>8s}")
print("-" * 60)
for cls_name in class_labels:
    for mname in ["Sequential", "QuasiDEER", "Broyden", "AccBroyden", "TAA"]:
        iters = results[mname]["iters"][cls_name]
        rt = results[mname]["runtime"][cls_name]
        print(f"  {cls_name:<20s}  {mname:<12s}  {iters:>5d}  {rt:>7.3f}s")
print("=" * 60)

# averages
print(f"\n  {'AVERAGE':<20s}  {'Method':<12s}  {'Iters':>5s}  {'Time':>8s}")
print("-" * 60)
for mname in ["Sequential", "QuasiDEER", "Broyden", "AccBroyden", "TAA"]:
    avg_iters = np.mean(list(results[mname]["iters"].values()))
    avg_rt = np.mean(list(results[mname]["runtime"].values()))
    print(f"  {'':>20s}  {mname:<12s}  {avg_iters:>5.1f}  {avg_rt:>7.3f}s")
print("=" * 60)

print("\nDone! Run compute_lpips.py to decode images and compute LPIPS.")
