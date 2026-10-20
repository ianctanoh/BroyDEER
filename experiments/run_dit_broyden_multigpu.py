"""
run_dit_broyden_multigpu.py

Compare single-GPU vs multi-GPU Broyden-DEER on DiT-XL/2.
Multi-GPU distributes the T-position vmap(eps_model) across devices
via JAX GSPMD sharding.

Usage:
    sbatch run_dit_broyden_multigpu.sh
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algs.broyden_ddim import broyden_ddim_alg
from algs.deer import jacobi_alg
from models.dit import make_dit_diffusion, load_vae, decode_latent

# ── config ──
dit_ckpt  = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "pretrained_models", "DiT-XL-2-256x256.pt")
dit_T     = 10
dit_class = 207    # golden retriever
dit_cfg   = 4.0
dit_seed  = 0
max_n     = 50
tol       = 10.0
save_dir  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figs", "dit")

num_gpus = len(jax.devices())
print(f"Devices: {jax.devices()}")
print(f"Using {num_gpus} GPUs")
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

# ── sequential baseline ──
print("\nRunning sequential DDIM ...")
sys.stdout.flush()

def _scan(state, driver):
    out = dit_deer_fxn(state, driver)
    return out, out

@jax.jit
def run_seq(x0, drivers):
    _, traj = jax.lax.scan(_scan, x0, drivers)
    return traj

seq_traj = run_seq(dit_x0, dit_drivers)
jax.block_until_ready(seq_traj)
t0 = time.time()
seq_traj = run_seq(dit_x0, dit_drivers)
jax.block_until_ready(seq_traj)
seq_time = time.time() - t0
seq_traj = np.array(seq_traj)
print(f"Sequential done: {dit_T} steps, {seq_time:.3f}s")
sys.stdout.flush()

# ── Jacobi (single-GPU) ──
print(f"\nRunning Jacobi ({max_n} iters) ...")
sys.stdout.flush()

_, jac_final, jac_steps, *_ = jacobi_alg(
    dit_deer_fxn, dit_x0, dit_guess, dit_drivers, max_n,
    full_trace=False, tol=tol,
)
jax.block_until_ready(jac_final)
t0 = time.time()
_, jac_final, jac_steps, *_ = jacobi_alg(
    dit_deer_fxn, dit_x0, dit_guess, dit_drivers, max_n,
    full_trace=False, tol=tol,
)
jax.block_until_ready(jac_final)
jac_time = time.time() - t0
jac_steps = int(jac_steps)
jac_final = np.array(jac_final)
print(f"Jacobi done: {jac_steps} iters, {jac_time:.3f}s")
sys.stdout.flush()

# ── single-GPU Broyden sweep ──
broyden_ranks = [2, 3, 4, 5]
sg_results = {}

for rank in broyden_ranks:
    print(f"\nRunning 1-GPU Broyden (m={rank}) ...")
    sys.stdout.flush()

    _, final, steps, *_ = broyden_ddim_alg(
        dit_dynamics, dit_x0, dit_guess, dit_drivers, max_n,
        m=rank, init="rescaled_identity", k=0.0,
        full_trace=False, tol=tol, sequential_solve=True,
    )
    jax.block_until_ready(final)
    t0 = time.time()
    _, final, steps, *_ = broyden_ddim_alg(
        dit_dynamics, dit_x0, dit_guess, dit_drivers, max_n,
        m=rank, init="rescaled_identity", k=0.0,
        full_trace=False, tol=tol, sequential_solve=True,
    )
    jax.block_until_ready(final)
    sg_time = time.time() - t0
    steps = int(steps)
    final = np.array(final)
    print(f"1-GPU Broyden(m={rank}): {steps} iters, {sg_time:.3f}s")
    sys.stdout.flush()
    sg_results[rank] = (final, steps, sg_time)

# ── multi-GPU Broyden sweep ──
mesh = Mesh(np.array(jax.devices()), axis_names=('pos',))
pos_d = NamedSharding(mesh, P('pos', None))
rep = NamedSharding(mesh, P())

mg_results = {}

def make_solve_mg(rank):
    """Build a jit-compiled multi-GPU solver for a given Broyden rank."""
    @jax.jit
    def solve_mg(x0, guess, drivers):
        res = broyden_ddim_alg(
            dit_dynamics, x0, guess, drivers, max_n,
            m=rank, init="rescaled_identity", k=0.0,
            full_trace=False, tol=tol, sequential_solve=True,
        )
        return res[1], res[2]  # final_state, newton_steps
    return solve_mg

for rank in broyden_ranks:
    print(f"\nRunning {num_gpus}-GPU Broyden (m={rank}) ...")
    sys.stdout.flush()

    solve_mg = make_solve_mg(rank)

    with mesh:
        x0_s = jax.device_put(dit_x0, rep)
        guess_s = jax.device_put(dit_guess, pos_d)
        drivers_s = jax.device_put(dit_drivers, pos_d)

        # warmup (JIT compile with GSPMD)
        final_mg, steps_mg = solve_mg(x0_s, guess_s, drivers_s)
        jax.block_until_ready(final_mg)
        # timed
        t0 = time.time()
        final_mg, steps_mg = solve_mg(x0_s, guess_s, drivers_s)
        jax.block_until_ready(final_mg)
        mg_time = time.time() - t0

    steps_mg = int(steps_mg)
    final_mg = np.array(final_mg)
    print(f"{num_gpus}-GPU Broyden(m={rank}): {steps_mg} iters, {mg_time:.3f}s")
    sys.stdout.flush()
    mg_results[rank] = (final_mg, steps_mg, mg_time)

# ── summary ──
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  {'Method':<30s}  {'Iters':>5s}  {'Time':>8s}  {'Speedup':>7s}")
print("-" * 60)
print(f"  {'Sequential':<30s}  {dit_T:>5d}  {seq_time:>7.3f}s")
print(f"  {'Jacobi':<30s}  {jac_steps:>5d}  {jac_time:>7.3f}s")

for rank in broyden_ranks:
    _, sg_steps, sg_time = sg_results[rank]
    _, mg_steps, mg_time = mg_results[rank]
    speedup_vs_sg = sg_time / mg_time if mg_time > 0 else float('inf')
    print(f"  {'Broyden(m=%d) 1-GPU' % rank:<30s}  {sg_steps:>5d}  {sg_time:>7.3f}s")
    print(f"  {'Broyden(m=%d) %d-GPU' % (rank, num_gpus):<30s}  {mg_steps:>5d}  {mg_time:>7.3f}s  {speedup_vs_sg:>6.2f}x")

sys.stdout.flush()

# ── decode + save images ──
print("\nLoading VAE ...")
sys.stdout.flush()
vae = load_vae(device="cpu")
print("VAE loaded.")

C, H, W = dit_params["latent_shape"]
os.makedirs(save_dir, exist_ok=True)

# Build method list: sequential, Jacobi, then 1-GPU and N-GPU for each rank
methods = [
    ("Sequential", seq_traj, dit_T, seq_time),
    ("Jacobi", jac_final, jac_steps, jac_time),
]
for rank in broyden_ranks:
    sol_sg, steps_sg, t_sg = sg_results[rank]
    sol_mg, steps_mg, t_mg = mg_results[rank]
    methods.append((f"Broy_m{rank}_1gpu", sol_sg, steps_sg, t_sg))
    methods.append((f"Broy_m{rank}_{num_gpus}gpu", sol_mg, steps_mg, t_mg))

plt.rcParams.update({"text.usetex": False, "font.family": "serif"})
fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 5))
for ax, (name, sol, steps, runtime) in zip(axes, methods):
    latent = sol[-1].reshape(C, H, W)
    img = decode_latent(vae, latent)
    ax.imshow(img)
    ax.set_title(name, fontsize=9)
    ax.set_xlabel(f"{steps} iters | {runtime:.3f}s", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(
    f'DiT-XL/2  |  class {dit_class}  |  T={dit_T}, cfg={dit_cfg}  |  1-GPU vs {num_gpus}-GPU',
    fontsize=12, y=1.02,
)
plt.tight_layout()
grid_path = os.path.join(save_dir, "dit_multigpu_comparison.png")
plt.savefig(grid_path, dpi=150, bbox_inches='tight')
print(f"Saved grid to {grid_path}")
plt.close()

print("\nDone!")
