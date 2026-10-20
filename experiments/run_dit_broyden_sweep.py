"""
run_dit_broyden_sweep.py

Broyden sweep on DiT-XL/2: gamma x init x rank.
Reports iters, runtime, and LPIPS vs sequential reference.

Usage:
    sbatch run_dit_broyden_sweep.sh
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

from algs.broyden_ddim import broyden_ddim_alg
from models.dit import make_dit_diffusion, load_vae, decode_latent

# ── config ──
dit_ckpt  = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "pretrained_models", "DiT-XL-2-256x256.pt")
dit_T     = 50
dit_class = 388    # giant panda
dit_cfg   = 5.0
dit_seed  = 0
max_n     = 50
tol       = 1.0
nwarmups  = 3
nreps     = 3
save_dir  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figs", "dit")

# sweep grid
broy_gammas = [0.2, 0.5, 0.7, 1.0]
broy_inits  = ["identity", "rescaled_identity"]
broy_ranks  = [2, 3, 4]

print(f"Device: {jax.devices()}")
print(f"T={dit_T}, class={dit_class}, cfg={dit_cfg}, max_n={max_n}, tol={tol}")
print(f"Sweep: gamma={broy_gammas}, init={broy_inits}, m={broy_ranks}")
print(f"Total combos: {len(broy_gammas) * len(broy_inits) * len(broy_ranks)}")
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
print(f"D={dit_D}, T={dit_T}")
sys.stdout.flush()

# ── helper: compile, warmup, time ──
def bench(name, jitted_fn, *args):
    print(f"\n{name}: compiling ...", end=" ", flush=True)
    t_compile_start = time.time()
    jitted_fn.lower(*args).compile()
    t_compile = time.time() - t_compile_start
    print(f"({t_compile:.1f}s)")

    for _ in range(nwarmups):
        result = jitted_fn(*args)
        jax.block_until_ready(result)

    t0 = time.time()
    for _ in range(nreps):
        result = jitted_fn(*args)
        jax.block_until_ready(result)
    t1 = time.time()
    time_avg = (t1 - t0) / nreps
    return result, time_avg

# ── sequential reference ──
@jax.jit
def run_seq(x0, drivers):
    _, traj = jax.lax.scan(
        lambda s, d: (dit_deer_fxn(s, d), dit_deer_fxn(s, d)), x0, drivers
    )
    return traj

print("\n--- Sequential reference ---")
result, t_seq = bench("Sequential", run_seq, dit_x0, dit_drivers)
seq_traj = np.array(result)
print(f"  Sequential: {dit_T} steps, {t_seq:.3f}s")
sys.stdout.flush()

# ── broyden sweep ──
all_results = [("Sequential", seq_traj, dit_T, t_seq)]

def make_run_broyden(rank, gamma, init):
    @jax.jit
    def run(x0, guess, drivers):
        res = broyden_ddim_alg(dit_dynamics, x0, guess, drivers, max_n,
                               m=rank, init=init, gamma=gamma, k=0.0,
                               full_trace=False, tol=tol,
                               sequential_solve=False, compress="rand")
        return res[1], res[2]
    return run

print(f"\n--- Broyden sweep ({len(broy_gammas)*len(broy_inits)*len(broy_ranks)} combos) ---")
sweep_results = []  # (init, gamma, rank, iters, time)

for init in broy_inits:
    init_short = "id" if init == "identity" else "rid"
    print(f"\n  == init={init} ==")
    for gamma in broy_gammas:
        for rank in broy_ranks:
            name = f"Broy(m={rank},g={gamma},{init_short})"
            fn = make_run_broyden(rank, gamma, init)
            try:
                result, t_avg = bench(name, fn, dit_x0, dit_guess, dit_drivers)
                final, steps = np.array(result[0]), int(result[1])
                sweep_results.append((init, gamma, rank, steps, t_avg))
                all_results.append((name, final, steps, t_avg))
                print(f"  {name}: {steps} iters, {t_avg:.3f}s")
            except Exception as e:
                print(f"  {name} FAILED: {e}")
                sweep_results.append((init, gamma, rank, -1, -1.0))
            sys.stdout.flush()

# ── summary table ──
print("\n" + "=" * 70)
print(f"  {'Init':<10s}  {'gamma':>5s}  {'m':>3s}  {'Iters':>5s}  {'Time':>8s}")
print("-" * 70)
for init, gamma, rank, steps, t_avg in sweep_results:
    init_short = "id" if init == "identity" else "rid"
    s_str = f"{steps:>5d}" if steps >= 0 else " FAIL"
    t_str = f"{t_avg:>7.3f}s" if t_avg >= 0 else "    N/A"
    print(f"  {init_short:<10s}  {gamma:>5.1f}  {rank:>3d}  {s_str}  {t_str}")
print("=" * 70)
sys.stdout.flush()

# ── decode + LPIPS ──
print("\nLoading VAE ...")
sys.stdout.flush()
vae = load_vae(device="cpu")
print("VAE loaded.")

import torch
import lpips
lpips_fn = lpips.LPIPS(net='vgg').eval()
print("LPIPS (VGG) loaded.")

os.makedirs(save_dir, exist_ok=True)

# decode sequential reference
seq_latent = seq_traj[-1].reshape(lat_C, lat_H, lat_W)
seq_img = decode_latent(vae, seq_latent)
seq_tensor = torch.from_numpy(seq_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
seq_tensor = seq_tensor * 2 - 1

# decode all and compute LPIPS
print("\n" + "=" * 80)
print(f"  {'Method':<35s}  {'Iters':>5s}  {'Time':>8s}  {'LPIPS':>8s}")
print("-" * 80)
print(f"  {'Sequential':<35s}  {dit_T:>5d}  {t_seq:>7.3f}s  {'---':>8s}")

for name, sol, steps, t_avg in all_results[1:]:
    latent = sol[-1].reshape(lat_C, lat_H, lat_W)
    img = decode_latent(vae, latent)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor * 2 - 1
    with torch.no_grad():
        score = lpips_fn(seq_tensor, img_tensor).item()
    s_str = f"{steps:>5d}" if steps >= 0 else " FAIL"
    print(f"  {name:<35s}  {s_str}  {t_avg:>7.3f}s  {score:>8.4f}")
print("=" * 80)
sys.stdout.flush()

print("\nDone!")
