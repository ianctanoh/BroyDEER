"""
run_dit_taa_gram64.py

TAA (k=10, m=2) on the same 32 ImageNet classes as run_dit_32images.py, with
the Anderson Gram system accumulated and solved in float64 (as ParaTAA does)
while everything else -- DiT, DDIM dynamics, trajectory -- stays float32.
Standard float32 TAA is run alongside as a control.

JAX needs x64 enabled to represent float64 at all. The DiT dynamics, initial
noise and drivers are therefore built with x64 OFF -- bit-identical to the
float32 runs (run_dit_32images.py) -- and x64 is switched on only afterwards,
before the TAA functions are traced. A dtype audit of each compiled TAA
function is printed to confirm that float64 only appears in the Gram/solve.

Stopping: tol = 0.1, max_n = 100 (same as run_dit_32images.py).

Usage:
    python run_dit_taa_gram64.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import jax
jax.config.update("jax_enable_x64", False)  # build inputs exactly as the float32 runs

import re
import time
import pickle
import collections
import numpy as np
import jax.numpy as jnp
import equinox as eqx

from algs.taa import taa_ddim_alg
from models.dit import make_dit_diffusion_reusable

# ── config (same as run_dit_32images.py) ──
dit_ckpt  = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "pretrained_models", "DiT-XL-2-256x256.pt")
dit_T     = 50
dit_cfg   = 5.0
dit_seed  = 0
max_n     = 100
tol       = 0.1
nwarmups  = 2
nreps     = 3
save_dir  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figs", "dit")

# 32 ImageNet animal classes (same order as run_dit_32images.py)
class_labels = {
    # batch A
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
    # batch B
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
sys.stdout.flush()

lat_C, lat_H, lat_W = 4, 32, 32
D = lat_C * lat_H * lat_W


# ── build reusable dynamics with x64 OFF (identical to the float32 runs) ──
set_class, _ = make_dit_diffusion_reusable(
    dit_ckpt, T=dit_T, cfg_scale=dit_cfg, eta=0.0, seed=dit_seed,
)
dynamics_0, x0, drivers = set_class(list(class_labels.values())[0])
guess = jnp.broadcast_to(x0, (dit_T, D))

# ── now enable x64 so the Gram/solve can use float64 ──
# Arrays created above stay float32; set_class only swaps an int32 label.
jax.config.update("jax_enable_x64", True)
float_dtypes = sorted({x.dtype.name for x in jax.tree.leaves((dynamics_0, x0, drivers))
                       if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating)})
print(f"x64={jax.config.read('jax_enable_x64')}, input float dtypes: {float_dtypes}")


@jax.jit
def run_seq(dynamics, x0, drivers):
    _, traj = jax.lax.scan(
        lambda s, d: (dynamics.deer_fxn(s, d), dynamics.deer_fxn(s, d)), x0, drivers
    )
    return traj


def make_run_taa(gram_dtype):
    @jax.jit
    def run(dynamics, x0, guess, drivers):
        res = taa_ddim_alg(dynamics, x0, guess, drivers, max_n,
                           order=10, memory_size=2, lmd=1e-3,
                           full_trace=False, tol=tol, gram_dtype=gram_dtype)
        return res[1], res[2]
    return run


methods = {
    "TAA": make_run_taa(None),               # float32 Gram/solve (control)
    "TAA_gram64": make_run_taa(jnp.float64), # float64 Gram/solve
}

# ── dtype audit: where does float64 appear? ──
print("\nDtype audit (equations producing float64 values):")
for mname, mfn in methods.items():
    jaxpr_txt = str(jax.make_jaxpr(mfn)(dynamics_0, x0, guess, drivers))
    prims = collections.Counter(
        m.group(1) for line in jaxpr_txt.splitlines() if ":f64[" in line
        for m in [re.search(r"=\s*([a-z_]+)", line)] if m
    )
    out = jax.eval_shape(mfn, dynamics_0, x0, guess, drivers)
    print(f"  {mname:11s} output dtypes {[o.dtype.name for o in jax.tree.leaves(out)]}, "
          f"f64 equations: {dict(prims) if prims else 'none'}")
sys.stdout.flush()

# ── compile once ──
print("\nCompiling (one-time) ...", end=" ", flush=True)
run_seq.lower(dynamics_0, x0, drivers).compile()
for mfn in methods.values():
    mfn.lower(dynamics_0, x0, guess, drivers).compile()
print("done.")
sys.stdout.flush()

results = {
    "config": {
        "T": dit_T, "cfg": dit_cfg, "tol": tol, "max_n": max_n,
        "seed": dit_seed, "classes": class_labels,
        "taa": {"order": 10, "memory_size": 2, "lmd": 1e-3},
    },
}
for mname in ["Sequential", *methods]:
    results[mname] = {"runtime": {}, "iters": {}, "lpips": {}, "latent": {}}

print(f"\n--- Running {len(class_labels)} classes x {len(methods) + 1} methods ---")

for cls_name, cls_id in class_labels.items():
    print(f"\n  == {cls_name} (class {cls_id}) ==")
    sys.stdout.flush()

    dynamics_cls = set_class(cls_id)[0]

    # warmup
    for _ in range(nwarmups):
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
    results["Sequential"]["runtime"][cls_name] = t_seq
    results["Sequential"]["iters"][cls_name] = dit_T
    results["Sequential"]["latent"][cls_name] = np.array(seq_result[-1])
    print(f"    Sequential: {dit_T} steps, {t_seq:.3f}s")

    # --- TAA variants ---
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

# ── save pkl ──
os.makedirs(save_dir, exist_ok=True)
pkl_path = os.path.join(save_dir, "dit_taa_gram64_results.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(results, f)
print(f"\nSaved results to {pkl_path}")

# ── summary ──
print(f"\n  {'Method':<12s}  {'median iters [IQR]':>22s}  {'median time':>11s}")
print("-" * 52)
for mname in ["Sequential", *methods]:
    it = np.array(list(results[mname]["iters"].values()), float)
    rt = np.array(list(results[mname]["runtime"].values()))
    print(f"  {mname:<12s}  {np.median(it):6.1f} [{np.percentile(it,25):.2f}, {np.percentile(it,75):.2f}]"
          f"  {np.median(rt):10.3f}s")

print("\nDone! Run compute_lpips.py --pkl <pkl> to decode images and compute LPIPS.")
