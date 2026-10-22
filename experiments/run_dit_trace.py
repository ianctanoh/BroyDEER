"""
run_dit_trace.py

Generate per-iteration images for one ImageNet class (default gorilla, 366) to visualize
convergence of each method. Saves one decoded image per iteration
per method. Methods and settings match run_dit_32images.py: Picard,
QuasiDEER, Broyden (m=2), Accelerated Broyden (m=2), TAA (k=6, m=2) with
its Gram system accumulated and solved in float64 (as ParaTAA). JAX x64 is
enabled only after the other methods have run, just before TAA is traced; the
DiT dynamics and inputs were built with x64 off and stay float32.

Usage:
    python run_dit_trace.py                                  # gorilla (366)
    python run_dit_trace.py --class_id 388 --name giant_panda   # any other class
"""

import sys, os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algs.taa import taa_ddim_alg
from algs.picard_ddim import picard_ddim_alg
from algs.deer_ddim import deer_ddim_alg
from algs.broyden_ddim import broyden_ddim_alg
from algs.accelerated_broyden_ddim import accelerated_broyden_ddim_alg
from models.dit import make_dit_diffusion, load_vae, decode_latent

# ── config ──
dit_ckpt  = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "pretrained_models", "DiT-XL-2-256x256.pt")
dit_T     = 50
parser = argparse.ArgumentParser()
parser.add_argument("--class_id", type=int, default=366, help="ImageNet class (default 366, gorilla)")
parser.add_argument("--name", type=str, default="gorilla", help="label for titles and the output folder")
args = parser.parse_args()
dit_class = args.class_id
dit_cfg   = 5.0
dit_seed  = 0
num_iters = 30     # fixed number of iterations (full trace)

save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "figs", "dit", f"{args.name}_trace")
os.makedirs(save_dir, exist_ok=True)

print(f"Device: {jax.devices()}")
print(f"T={dit_T}, class={dit_class}, cfg={dit_cfg}, num_iters={num_iters}")
sys.stdout.flush()

# ── build dynamics ──
print("\nBuilding DiT-XL/2 dynamics ...")
sys.stdout.flush()
t0 = time.time()
dit_deer_fxn, dit_dynamics, dit_x0, dit_drivers, dit_params = make_dit_diffusion(
    dit_ckpt, T=dit_T, class_label=dit_class, cfg_scale=dit_cfg,
    eta=0.0, seed=dit_seed,
)
print(f"Done in {time.time()-t0:.1f}s")

dit_D = dit_x0.shape[0]
dit_guess = jnp.broadcast_to(dit_x0, (dit_T, dit_D))
lat_C, lat_H, lat_W = dit_params["latent_shape"]
print(f"D={dit_D}, T={dit_T}, latent=({lat_C},{lat_H},{lat_W})")
sys.stdout.flush()

# ── sequential reference ──
print("\nRunning sequential ...")
sys.stdout.flush()

@jax.jit
def run_seq(x0, drivers):
    _, traj = jax.lax.scan(
        lambda s, d: (dit_deer_fxn(s, d), dit_deer_fxn(s, d)), x0, drivers
    )
    return traj

run_seq.lower(dit_x0, dit_drivers).compile()
seq_traj = np.array(run_seq(dit_x0, dit_drivers))
print("  Sequential done.")

# ── methods with full_trace ──

# Picard
@jax.jit
def run_picard_trace(x0, guess, drivers):
    res = picard_ddim_alg(dit_dynamics, x0, guess, drivers, num_iters,
                          full_trace=True)
    return res[0]  # all_states: (num_iters+1, T, D)

# QuasiDEER (Hutchinson diagonal, 1 sample)
@jax.jit
def run_quasi_deer_trace(x0, guess, drivers):
    res = deer_ddim_alg(dit_dynamics, x0, guess, drivers, num_iters,
                        num_samples=1, k=0.0, full_trace=True)
    return res[0]  # all_states: (num_iters+1, T, D)

# TAA (order=6, m=2), float64 Gram/solve -- traced only after x64 is enabled
@jax.jit
def run_taa_trace(x0, guess, drivers):
    res = taa_ddim_alg(dit_dynamics, x0, guess, drivers, num_iters,
                       order=6, memory_size=2, lmd=1e-3, full_trace=True,
                       gram_dtype=jnp.float64)
    return res[0]  # all_states: (num_iters+1, T, D)

# Broyden (m=2)
@jax.jit
def run_broyden_trace(x0, guess, drivers):
    res = broyden_ddim_alg(dit_dynamics, x0, guess, drivers, num_iters,
                           m=2, init="identity", gamma=1.0, k=0.0,
                           full_trace=True, sequential_solve=False, compress="rand")
    return res[0]  # all_states: (num_iters+1, T, D)

# Accelerated Broyden (m=2)
@jax.jit
def run_accel_trace(x0, guess, drivers):
    res = accelerated_broyden_ddim_alg(dit_dynamics, x0, guess, drivers, num_iters,
                                       m=2, init="identity", gamma=1.0, k=0.0,
                                       full_trace=True, sequential_solve=True)
    return res[0]  # all_states: (num_iters+1, T, D)

methods = [
    ("Picard", run_picard_trace),
    ("QuasiDEER", run_quasi_deer_trace),
    ("Broyden", run_broyden_trace),
    ("AccBroyden", run_accel_trace),
    ("TAA", run_taa_trace),
]

traces = {}
for name, fn in methods:
    if name == "TAA":
        # float64 needs x64; everything traced so far stays float32
        jax.config.update("jax_enable_x64", True)
    print(f"\n{name}: compiling ...", end=" ", flush=True)
    fn.lower(dit_x0, dit_guess, dit_drivers).compile()
    print("running ...", end=" ", flush=True)
    all_states = np.array(fn(dit_x0, dit_guess, dit_drivers))
    traces[name] = all_states
    print(f"done. Shape: {all_states.shape}, dtype: {all_states.dtype}")
    sys.stdout.flush()

# ── decode + save images ──
print("\nLoading VAE ...")
sys.stdout.flush()
vae = load_vae(device="cpu")
print("VAE loaded.")

# save sequential reference
seq_latent = seq_traj[-1].reshape(lat_C, lat_H, lat_W)
seq_img = decode_latent(vae, seq_latent)
plt.imsave(os.path.join(save_dir, "sequential.png"), seq_img)
print("Saved sequential.png")

# save per-iteration images for each method
for name, all_states in traces.items():
    method_dir = os.path.join(save_dir, name.lower())
    os.makedirs(method_dir, exist_ok=True)
    n_iters_total = all_states.shape[0]  # num_iters + 1 (includes initial guess)
    print(f"\nDecoding {name} ({n_iters_total} frames) ...")
    for i in range(n_iters_total):
        latent = all_states[i, -1].reshape(lat_C, lat_H, lat_W)
        img = decode_latent(vae, latent)
        plt.imsave(os.path.join(method_dir, f"iter_{i:03d}.png"), img)
    print(f"  Saved {n_iters_total} images to {method_dir}/")

# ── summary grid: iterations 0, 1, 3, 5, 10, 15, 20, 25, 30 ──
grid_iters = [i for i in [0, 1, 3, 5, 10, 15, 20, 25, 30] if i <= num_iters]
n_methods = len(methods)
n_cols = len(grid_iters) + 1  # +1 for sequential reference

fig, axes = plt.subplots(n_methods, n_cols, figsize=(2.5 * n_cols, 2.8 * n_methods))

for row, (name, all_states) in enumerate(traces.items()):
    # sequential reference in first column
    axes[row, 0].imshow(seq_img)
    axes[row, 0].set_ylabel(name, fontsize=10, fontweight="bold")
    if row == 0:
        axes[row, 0].set_title("Sequential", fontsize=8)
    axes[row, 0].set_xticks([])
    axes[row, 0].set_yticks([])

    for col, it in enumerate(grid_iters):
        ax = axes[row, col + 1]
        latent = all_states[it, -1].reshape(lat_C, lat_H, lat_W)
        img = decode_latent(vae, latent)
        ax.imshow(img)
        if row == 0:
            ax.set_title(f"iter {it}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

fig.suptitle(
    f"{args.name.replace('_', ' ').capitalize()} (class {dit_class})  |  T={dit_T}, cfg={dit_cfg}  |  {num_iters} iters",
    fontsize=12, y=1.01,
)
plt.tight_layout()
grid_path = os.path.join(save_dir, "convergence_grid.png")
plt.savefig(grid_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved convergence grid: {grid_path}")

print("\nDone!")
