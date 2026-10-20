"""
run_dit_accel_broyden.py
Just the accelerated Broyden runs (sequential_solve to avoid compilation OOM).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import time
import numpy as np
import jax
import jax.numpy as jnp

from algs.accelerated_broyden_ddim import accelerated_broyden_ddim_alg
from models.dit import make_dit_diffusion

dit_ckpt  = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "pretrained_models", "DiT-XL-2-256x256.pt")
dit_T     = 50
dit_class = 207
dit_cfg   = 4.0
dit_seed  = 0
max_n     = 50
tol       = 1.0

print(f"Device: {jax.devices()}")
print(f"T={dit_T}, tol={tol}, max_n={max_n}")
sys.stdout.flush()

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

accel_ranks = [2, 3, 4]

for rank in accel_ranks:
    print(f"\nRunning Accel-Broyden (m={rank}, {max_n} iters, sequential_solve) ...")
    sys.stdout.flush()

    _, abd_final, abd_steps, *_ = accelerated_broyden_ddim_alg(
        dit_dynamics, dit_x0, dit_guess, dit_drivers, max_n,
        m=rank, init="rescaled_identity", k=0.0, full_trace=False, tol=tol,
        sequential_solve=True,
    )
    jax.block_until_ready(abd_final)
    t0 = time.time()
    _, abd_final, abd_steps, *_ = accelerated_broyden_ddim_alg(
        dit_dynamics, dit_x0, dit_guess, dit_drivers, max_n,
        m=rank, init="rescaled_identity", k=0.0, full_trace=False, tol=tol,
        sequential_solve=True,
    )
    jax.block_until_ready(abd_final)
    abd_time = time.time() - t0
    abd_steps = int(abd_steps)
    print(f"Accel-Broyden(m={rank}) done: {abd_steps} iters, {abd_time:.3f}s (after JIT)")
    sys.stdout.flush()

print("\nDone!")
