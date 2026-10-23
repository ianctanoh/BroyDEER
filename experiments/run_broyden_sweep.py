"""
run_broyden_sweep.py

Sweep Broyden gamma={0.2,0.4,0.6,0.8,1.0} and rank m={2,4,6} on 32 ImageNet classes.
Identity init, rand compression, no damping (k=0).
Uses make_dit_diffusion_reusable to compile once.
Stopping: tol = 0.1, max_n = 100 (same as run_dit_32images.py).

Usage:
    python run_broyden_sweep.py
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

from algs.broyden_ddim import broyden_ddim_alg
from models.dit import make_dit_diffusion_reusable

# ── config ──
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

# Broyden sweep parameters
broy_gammas = [0.2, 0.4, 0.6, 0.8, 1.0]
broy_ranks  = [2, 4, 6]

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
print(f"Broyden sweep: gamma={broy_gammas}, m={broy_ranks}")
print(f"Classes: {list(class_labels.keys())}")
sys.stdout.flush()

D = 4 * 32 * 32
print(f"D={D}, T={dit_T}")
sys.stdout.flush()

# ── build reusable dynamics ──
print("\nBuilding reusable DiT dynamics ...")
sys.stdout.flush()
t0 = time.time()
set_class, params = make_dit_diffusion_reusable(
    dit_ckpt, T=dit_T, cfg_scale=dit_cfg, eta=0.0, seed=dit_seed,
)
print(f"Done in {time.time()-t0:.1f}s")
sys.stdout.flush()

dynamics_0, x0, drivers = set_class(list(class_labels.values())[0])
guess = jnp.broadcast_to(x0, (dit_T, D))

# ── define jitted functions ONCE ──
@jax.jit
def run_seq(dynamics, x0, drivers):
    eps_model = dynamics.eps_model
    coef_x = dynamics.coef_x
    coef_eps = dynamics.coef_eps
    coef_noise = dynamics.coef_noise
    s_idx = jnp.arange(dit_T)
    t_per_s = dit_T - s_idx

    def scan_fn(state, s):
        t = t_per_s[s]
        z = drivers[s, :D]
        eps = eps_model(state, t)
        next_state = coef_x[t] * state + coef_eps[t] * eps + coef_noise[t] * z
        return next_state, next_state

    _, traj = jax.lax.scan(scan_fn, x0, jnp.arange(dit_T))
    return traj


def make_run_broyden(gamma, m):
    @jax.jit
    def run(dynamics, x0, guess, drivers):
        res = broyden_ddim_alg(dynamics, x0, guess, drivers, max_n,
                               m=m, init="identity", gamma=gamma, k=0.0,
                               full_trace=False, tol=tol,
                               sequential_solve=False, compress="rand")
        return res[1], res[2]
    return run

broy_fns = {}
for g in broy_gammas:
    for m in broy_ranks:
        broy_fns[f"Broy_g{g}_m{m}"] = make_run_broyden(g, m)

# ── compile once ──
print("\nCompiling (one-time) ...", end=" ", flush=True)
run_seq.lower(dynamics_0, x0, drivers).compile()
_ = run_seq(dynamics_0, x0, drivers)
jax.block_until_ready(_)
for fn in broy_fns.values():
    fn.lower(dynamics_0, x0, guess, drivers).compile()
    _ = fn(dynamics_0, x0, guess, drivers)
    jax.block_until_ready(_)
print("done.")
sys.stdout.flush()

# build method names
method_names = ["Sequential"]
for g in broy_gammas:
    for m in broy_ranks:
        method_names.append(f"Broy_g{g}_m{m}")

results = {
    "config": {
        "T": dit_T, "cfg": dit_cfg, "tol": tol, "max_n": max_n,
        "seed": dit_seed, "classes": class_labels,
        "broy_gammas": broy_gammas, "broy_ranks": broy_ranks,
    },
}
for mname in method_names:
    results[mname] = {"runtime": {}, "iters": {}, "lpips": {}, "latent": {}}

# ── run all classes ──
print(f"\n--- Running {len(class_labels)} classes x {len(method_names)} methods ---")

for cls_name, cls_id in class_labels.items():
    print(f"\n  == {cls_name} (class {cls_id}) ==")
    sys.stdout.flush()

    dynamics_cls, _, _ = set_class(cls_id)

    _ = run_seq(dynamics_cls, x0, drivers)
    jax.block_until_ready(_)

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

    for mname, fn in broy_fns.items():
        _ = fn(dynamics_cls, x0, guess, drivers)
        jax.block_until_ready(_)

        t0 = time.time()
        for _ in range(nreps):
            res = fn(dynamics_cls, x0, guess, drivers)
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
pkl_path = os.path.join(save_dir, "broyden_sweep_32images.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(results, f)
print(f"\nSaved results to {pkl_path}")

# ── summary ──
print("\n" + "=" * 70)
print(f"  {'Class':<20s}  {'Method':<18s}  {'Iters':>5s}  {'Time':>8s}")
print("-" * 70)
for cls_name in class_labels:
    for mname in method_names:
        iters = results[mname]["iters"][cls_name]
        rt = results[mname]["runtime"][cls_name]
        print(f"  {cls_name:<20s}  {mname:<18s}  {iters:>5d}  {rt:>7.3f}s")
print("=" * 70)

print(f"\n  {'AVERAGE':<20s}  {'Method':<18s}  {'Iters':>5s}  {'Time':>8s}")
print("-" * 70)
for mname in method_names:
    avg_iters = np.mean(list(results[mname]["iters"].values()))
    avg_rt = np.mean(list(results[mname]["runtime"].values()))
    print(f"  {'':>20s}  {mname:<18s}  {avg_iters:>5.1f}  {avg_rt:>7.3f}s")
print("=" * 70)

print("\nDone!")
