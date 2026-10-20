"""
run_dit_taa.py

Compare parallel solvers on DiT-XL/2 diffusion.
All methods are wrapped in @jax.jit for proper compilation caching.
Timing excludes compilation (warmup + .lower().compile()).

Usage:
    sbatch run_dit_taa.sh
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algs.taa import taa_ddim_alg
from algs.broyden_ddim import broyden_ddim_alg
from algs.accelerated_broyden_ddim import accelerated_broyden_ddim_alg
from algs.deer import jacobi_alg, deer_alg
from models.dit import make_dit_diffusion, load_vae, decode_latent

# ── config ──
dit_ckpt  = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "pretrained_models", "DiT-XL-2-256x256.pt")
dit_T     = 50
dit_class = 388    # giant panda
dit_cfg   = 5.0
dit_seed  = 0
max_n     = 50
tol       = 5.0
nwarmups  = 3
nreps     = 3
save_dir  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figs", "dit")

print(f"Device: {jax.devices()}")
print(f"T={dit_T}, class={dit_class}, cfg={dit_cfg}, max_n={max_n}, tol={tol}")
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
print(f"D={dit_D}, T={dit_T}")
sys.stdout.flush()

# ── helper: compile, warmup, time ──
def bench(name, jitted_fn, *args):
    """Compile explicitly, warmup, then time nreps runs."""
    print(f"\n{name}: compiling ...", end=" ", flush=True)
    t_compile_start = time.time()
    jitted_fn.lower(*args).compile()
    t_compile = time.time() - t_compile_start
    print(f"({t_compile:.1f}s)")

    # warmup
    for _ in range(nwarmups):
        result = jitted_fn(*args)
        jax.block_until_ready(result)

    # timed
    t0 = time.time()
    for _ in range(nreps):
        result = jitted_fn(*args)
        jax.block_until_ready(result)
    t1 = time.time()
    time_avg = (t1 - t0) / nreps
    return result, time_avg


# ── define jitted eval functions ──

# Sequential
@jax.jit
def run_seq(x0, drivers):
    def _scan(state, driver):
        return dit_deer_fxn(state, driver), dit_deer_fxn(state, driver)
    _, traj = jax.lax.scan(lambda s, d: (dit_deer_fxn(s, d), dit_deer_fxn(s, d)), x0, drivers)
    return traj

# Jacobi
@jax.jit
def run_jacobi(x0, guess, drivers):
    res = jacobi_alg(dit_deer_fxn, x0, guess, drivers, max_n,
                     full_trace=False, tol=tol)
    return res[1], res[2]  # final_state, newton_steps

# Quasi-DEER (Hutchinson)
@jax.jit
def run_quasi_deer(x0, guess, drivers):
    res = deer_alg(dit_deer_fxn, x0, guess, drivers, max_n,
                   quasi=True, qmem_efficient=True, full_trace=False, tol=tol)
    return res[1], res[2]

# TAA order=1
@jax.jit
def run_taa1(x0, guess, drivers):
    res = taa_ddim_alg(dit_dynamics, x0, guess, drivers, max_n,
                       order=1, memory_size=2, lmd=1e-3, full_trace=False, tol=tol)
    return res[1], res[2]

# TAA order=4
@jax.jit
def run_taa4(x0, guess, drivers):
    res = taa_ddim_alg(dit_dynamics, x0, guess, drivers, max_n,
                       order=4, memory_size=2, lmd=1e-3, full_trace=False, tol=tol)
    return res[1], res[2]

# Broyden parallel (rand)
def make_run_broyden_par(rank):
    @jax.jit
    def run(x0, guess, drivers):
        res = broyden_ddim_alg(dit_dynamics, x0, guess, drivers, max_n,
                               m=rank, init="identity", gamma=1.0, k=0.0,
                               full_trace=False, tol=tol,
                               sequential_solve=False, compress="rand")
        return res[1], res[2]
    return run

# Broyden sequential
def make_run_broyden_seq(rank):
    @jax.jit
    def run(x0, guess, drivers):
        res = broyden_ddim_alg(dit_dynamics, x0, guess, drivers, max_n,
                               m=rank, init="identity", gamma=1.0, k=0.0,
                               full_trace=False, tol=tol,
                               sequential_solve=True)
        return res[1], res[2]
    return run

# Accelerated Broyden (sequential solve)
def make_run_accel_broyden(rank):
    @jax.jit
    def run(x0, guess, drivers):
        res = accelerated_broyden_ddim_alg(dit_dynamics, x0, guess, drivers, max_n,
                                           m=rank, init="identity", gamma=1.0, k=0.0,
                                           full_trace=False, tol=tol,
                                           sequential_solve=True)
        return res[1], res[2]
    return run


# ── run all methods ──
all_results = []  # list of (name, final_state_or_traj, n_iters, time)

# Sequential
result, t = bench("Sequential", run_seq, dit_x0, dit_drivers)
seq_traj = np.array(result)
all_results.append(("Sequential", seq_traj, dit_T, t))
print(f"  Sequential: {dit_T} steps, {t:.3f}s")

# TAA order=4
result, t = bench("TAA(4)", run_taa4, dit_x0, dit_guess, dit_drivers)
final, steps = np.array(result[0]), int(result[1])
all_results.append(("TAA_order4", final, steps, t))
print(f"  TAA(4): {steps} iters, {t:.3f}s")

# Broyden parallel m=2
fn = make_run_broyden_par(2)
result, t = bench("Broyden_par(m=2)", fn, dit_x0, dit_guess, dit_drivers)
final, steps = np.array(result[0]), int(result[1])
all_results.append(("Broy_par_m2", final, steps, t))
print(f"  Broyden_par(m=2): {steps} iters, {t:.3f}s")


# ── summary ──
print("\n" + "=" * 55)
print(f"  {'Method':<25s}  {'Iters':>5s}  {'Time':>8s}")
print("-" * 55)
for name, _, steps, t in all_results:
    print(f"  {name:<25s}  {steps:>5d}  {t:>7.3f}s")
print("=" * 55)
sys.stdout.flush()

# ── decode + save images + LPIPS ──
print("\nLoading VAE ...")
sys.stdout.flush()
vae = load_vae(device="cpu")
print("VAE loaded.")

import torch
import lpips
lpips_fn = lpips.LPIPS(net='vgg').eval()
print("LPIPS (VGG) loaded.")

C, H, W = dit_params["latent_shape"]
os.makedirs(save_dir, exist_ok=True)

# decode all images
decoded_images = {}
for name, sol, _, _ in all_results:
    latent = sol[-1].reshape(C, H, W)
    img = decode_latent(vae, latent)
    decoded_images[name] = img

# compute LPIPS vs sequential reference
seq_img = decoded_images["Sequential"]
seq_tensor = torch.from_numpy(seq_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
seq_tensor = seq_tensor * 2 - 1  # scale to [-1, 1]

lpips_scores = {}
print("\n" + "=" * 70)
print(f"  {'Method':<25s}  {'Iters':>5s}  {'Time':>8s}  {'LPIPS':>8s}")
print("-" * 70)
for name, _, steps, t in all_results:
    if name == "Sequential":
        lpips_scores[name] = 0.0
        print(f"  {name:<25s}  {steps:>5d}  {t:>7.3f}s  {'---':>8s}")
    else:
        img = decoded_images[name]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor * 2 - 1
        with torch.no_grad():
            score = lpips_fn(seq_tensor, img_tensor).item()
        lpips_scores[name] = score
        print(f"  {name:<25s}  {steps:>5d}  {t:>7.3f}s  {score:>8.4f}")
print("=" * 70)
sys.stdout.flush()

# save grid with LPIPS in labels
methods = all_results
plt.rcParams.update({"text.usetex": False, "font.family": "serif"})
fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 5))
if len(methods) == 1:
    axes = [axes]
for ax, (name, sol, steps, runtime) in zip(axes, methods):
    ax.imshow(decoded_images[name])
    ax.set_title(name, fontsize=9)
    if name == "Sequential":
        ax.set_xlabel(f"{steps} steps | {runtime:.3f}s", fontsize=8)
    else:
        ax.set_xlabel(f"{steps} iters | {runtime:.3f}s | LPIPS={lpips_scores[name]:.4f}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(
    f'DiT-XL/2  |  class {dit_class}  |  T={dit_T}, cfg={dit_cfg}, tol={tol}',
    fontsize=12, y=1.02,
)
plt.tight_layout()
grid_path = os.path.join(save_dir, "dit_all_methods.png")
plt.savefig(grid_path, dpi=150, bbox_inches='tight')
print(f"Saved grid to {grid_path}")
plt.close()

for name, sol, _, _ in methods:
    img_path = os.path.join(save_dir, f"dit_{name.lower()}.png")
    plt.imsave(img_path, decoded_images[name])
    print(f"Saved {img_path}")

print("\nDone!")
