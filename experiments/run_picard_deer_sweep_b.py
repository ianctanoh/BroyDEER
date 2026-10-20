"""
run_picard_deer_sweep.py

Run Picard DDIM and DEER DDIM on 16 ImageNet (batch B) classes.
Sequential reference included for LPIPS baseline.

Usage:
    sbatch run_picard_deer_sweep.sh
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import time
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")

from algs.picard_ddim import picard_ddim_alg
from algs.deer_ddim import deer_ddim_alg
from models.dit import make_dit_diffusion

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

class_labels = {
    "king_penguin": 145,
    "toucan": 96,
    "tree_frog": 31,
    "chameleon": 47,
    "coral_reef_fish": 392,
    "hammerhead_shark": 4,
    "sea_turtle": 33,
    "dalmatian": 251,
    "siamese_cat": 284,
    "cheetah": 293,
    "brown_bear": 294,
    "bison": 347,
    "hippopotamus": 344,
    "orangutan": 365,
    "macaw": 88,
    "ladybug": 301,
}

print(f"Device: {jax.devices()}")
print(f"T={dit_T}, cfg={dit_cfg}, max_n={max_n}, tol={tol}")
print(f"Classes: {list(class_labels.keys())}")
sys.stdout.flush()

lat_C, lat_H, lat_W = 4, 32, 32
D = lat_C * lat_H * lat_W
print(f"D={D}, T={dit_T}")
sys.stdout.flush()

method_names = ["Sequential", "Picard", "QuasiDEER"]

results = {
    "config": {
        "T": dit_T, "cfg": dit_cfg, "tol": tol, "max_n": max_n,
        "seed": dit_seed, "classes": class_labels,
    },
}
for mname in method_names:
    results[mname] = {"runtime": {}, "iters": {}, "lpips": {}, "latent": {}}

print(f"\n--- Running {len(class_labels)} classes x {len(method_names)} methods ---")

for cls_name, cls_id in class_labels.items():
    print(f"\n  == {cls_name} (class {cls_id}) ==")
    sys.stdout.flush()

    deer_fxn_cls, dynamics_cls, x0, drivers, _ = make_dit_diffusion(
        dit_ckpt, T=dit_T, class_label=cls_id, cfg_scale=dit_cfg,
        eta=0.0, seed=dit_seed,
    )
    guess = jnp.broadcast_to(x0, (dit_T, D))

    @jax.jit
    def run_seq_cls(x0, drivers):
        _, traj = jax.lax.scan(
            lambda s, d: (deer_fxn_cls(s, d), deer_fxn_cls(s, d)), x0, drivers
        )
        return traj

    @jax.jit
    def run_picard_cls(x0, guess, drivers):
        res = picard_ddim_alg(dynamics_cls, x0, guess, drivers, max_n,
                              full_trace=False, tol=tol)
        return res[1], res[2]

    @jax.jit
    def run_deer_cls(x0, guess, drivers):
        res = deer_ddim_alg(dynamics_cls, x0, guess, drivers, max_n,
                            num_samples=1, k=0.0, full_trace=False, tol=tol)
        return res[1], res[2]

    cls_methods = {
        "Picard": run_picard_cls,
        "QuasiDEER": run_deer_cls,
    }

    # compile + warmup
    print("    compiling ...", end=" ", flush=True)
    run_seq_cls.lower(x0, drivers).compile()
    for mfn in cls_methods.values():
        mfn.lower(x0, guess, drivers).compile()
    _ = run_seq_cls(x0, drivers)
    for mfn in cls_methods.values():
        _ = mfn(x0, guess, drivers)
    jax.block_until_ready(_)
    print("done.")

    # sequential
    t0 = time.time()
    for _ in range(nreps):
        seq_result = run_seq_cls(x0, drivers)
        jax.block_until_ready(seq_result)
    t_seq = (time.time() - t0) / nreps
    seq_traj = np.array(seq_result)
    results["Sequential"]["runtime"][cls_name] = t_seq
    results["Sequential"]["iters"][cls_name] = dit_T
    results["Sequential"]["latent"][cls_name] = np.array(seq_traj[-1])
    print(f"    Sequential: {dit_T} steps, {t_seq:.3f}s")

    # parallel methods
    for mname, mfn in cls_methods.items():
        t0 = time.time()
        for _ in range(nreps):
            res = mfn(x0, guess, drivers)
            jax.block_until_ready(res)
        t_avg = (time.time() - t0) / nreps
        final, steps = np.array(res[0]), int(res[1])
        results[mname]["runtime"][cls_name] = t_avg
        results[mname]["iters"][cls_name] = steps
        results[mname]["latent"][cls_name] = np.array(final[-1])
        print(f"    {mname}: {steps} iters, {t_avg:.3f}s")

    jax.clear_caches()
    sys.stdout.flush()

# ── save pkl ──
os.makedirs(save_dir, exist_ok=True)
pkl_path = os.path.join(save_dir, "picard_deer_sweep_b.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(results, f)
print(f"\nSaved results to {pkl_path}")

# ── summary ──
print("\n" + "=" * 60)
print(f"  {'Class':<20s}  {'Method':<12s}  {'Iters':>5s}  {'Time':>8s}")
print("-" * 60)
for cls_name in class_labels:
    for mname in method_names:
        iters = results[mname]["iters"][cls_name]
        rt = results[mname]["runtime"][cls_name]
        print(f"  {cls_name:<20s}  {mname:<12s}  {iters:>5d}  {rt:>7.3f}s")
print("=" * 60)

print(f"\n  {'AVERAGE':<20s}  {'Method':<12s}  {'Iters':>5s}  {'Time':>8s}")
print("-" * 60)
for mname in method_names:
    avg_iters = np.mean(list(results[mname]["iters"].values()))
    avg_rt = np.mean(list(results[mname]["runtime"].values()))
    print(f"  {'':>20s}  {mname:<12s}  {avg_iters:>5.1f}  {avg_rt:>7.3f}s")
print("=" * 60)

print("\nDone!")
