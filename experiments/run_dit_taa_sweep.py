"""
run_dit_taa_sweep.py

TAA sweep on DiT-XL/2: order k x memory m (no windowing).
Reports iters, runtime, and LPIPS vs sequential reference.

Usage:
    sbatch run_dit_taa_sweep.sh
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
taa_ks = [1, 2, 4, 6, 8]
taa_ms = [2, 4, 6, 8]

print(f"Device: {jax.devices()}")
print(f"T={dit_T}, class={dit_class}, cfg={dit_cfg}, max_n={max_n}, tol={tol}")
print(f"Sweep: k={taa_ks}, m={taa_ms}")
print(f"Total combos: {len(taa_ks) * len(taa_ms)}")
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

# ── TAA sweep ──
all_results = [("Sequential", seq_traj, dit_T, t_seq)]

def make_run_taa(k, m):
    @jax.jit
    def run(x0, guess, drivers):
        res = taa_ddim_alg(dit_dynamics, x0, guess, drivers, max_n,
                           order=k, memory_size=m, lmd=1e-3,
                           full_trace=False, tol=tol)
        return res[1], res[2]
    return run

print(f"\n--- TAA sweep ({len(taa_ks)*len(taa_ms)} combos) ---")
sweep_results = []  # (k, m, iters, time)

for k_val in taa_ks:
    print(f"\n  == order k={k_val} ==")
    for m_val in taa_ms:
        name = f"TAA(k={k_val},m={m_val})"
        fn = make_run_taa(k_val, m_val)
        try:
            result, t_avg = bench(name, fn, dit_x0, dit_guess, dit_drivers)
            final, steps = np.array(result[0]), int(result[1])
            sweep_results.append((k_val, m_val, steps, t_avg))
            all_results.append((name, final, steps, t_avg))
            print(f"  {name}: {steps} iters, {t_avg:.3f}s")
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            sweep_results.append((k_val, m_val, -1, -1.0))
        sys.stdout.flush()

# ── summary table ──
print("\n" + "=" * 55)
print(f"  {'k':>3s}  {'m':>3s}  {'Iters':>5s}  {'Time':>8s}")
print("-" * 55)
for k_val, m_val, steps, t_avg in sweep_results:
    s_str = f"{steps:>5d}" if steps >= 0 else " FAIL"
    t_str = f"{t_avg:>7.3f}s" if t_avg >= 0 else "    N/A"
    print(f"  {k_val:>3d}  {m_val:>3d}  {s_str}  {t_str}")
print("=" * 55)
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
print("\n" + "=" * 70)
print(f"  {'Method':<25s}  {'Iters':>5s}  {'Time':>8s}  {'LPIPS':>8s}")
print("-" * 70)
print(f"  {'Sequential':<25s}  {dit_T:>5d}  {t_seq:>7.3f}s  {'---':>8s}")

for name, sol, steps, t_avg in all_results[1:]:
    latent = sol[-1].reshape(lat_C, lat_H, lat_W)
    img = decode_latent(vae, latent)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor * 2 - 1
    with torch.no_grad():
        score = lpips_fn(seq_tensor, img_tensor).item()
    s_str = f"{steps:>5d}" if steps >= 0 else " FAIL"
    print(f"  {name:<25s}  {s_str}  {t_avg:>7.3f}s  {score:>8.4f}")
print("=" * 70)
sys.stdout.flush()

# ── heatmap: iters (k vs m) ──
print("\nGenerating heatmap ...")
grid = np.full((len(taa_ks), len(taa_ms)), np.nan)
for k_val, m_val, iters, _ in sweep_results:
    ki = taa_ks.index(k_val)
    mi = taa_ms.index(m_val)
    grid[ki, mi] = iters if iters >= 0 else np.nan

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(len(taa_ms)))
ax.set_xticklabels(taa_ms)
ax.set_yticks(range(len(taa_ks)))
ax.set_yticklabels(taa_ks)
ax.set_xlabel("memory m")
ax.set_ylabel("order k")
ax.set_title(f"TAA iterations to convergence  (T={dit_T}, tol={tol})")
for ki in range(len(taa_ks)):
    for mi in range(len(taa_ms)):
        val = grid[ki, mi]
        if np.isnan(val):
            ax.text(mi, ki, "FAIL", ha="center", va="center", fontsize=8, color="red")
        else:
            ax.text(mi, ki, f"{int(val)}", ha="center", va="center", fontsize=10,
                    fontweight="bold")
fig.colorbar(im, ax=ax, label="iterations")
plt.tight_layout()
hm_path = os.path.join(save_dir, "taa_sweep_heatmap.png")
plt.savefig(hm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {hm_path}")

print("\nDone!")
