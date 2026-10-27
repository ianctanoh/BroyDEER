"""
run_dit_tol_curves.py

Tolerance / quality trade-off for the DiT solvers. Runs each method of
run_dit_32images.py (same 32 classes and settings) for a fixed max_n
iterations with full_trace=True and records, at every iteration, the merit and
the final-step latent. Iterations-to-tolerance for ANY tol, and LPIPS vs the
sequential sample at any iteration, can then be computed after the fact.

Iteration convention (same as the early-stopping solvers): each step reports
the merit of the states *before* its update, so the solver run with tolerance
tol stops after  min{j : merit[j] < tol} + 1  steps. At tol = 0.1 this must
reproduce the iteration counts of run_dit_32images.py.

TAA uses k=10, m=2 with its Gram system accumulated/solved in float64 (as in
run_dit_32images.py); only TAA is traced and run under
jax.experimental.enable_x64, so DiT and the trajectory stay float32.

Two phases, run as separate processes (JAX, then PyTorch):
    python run_dit_tol_curves.py                     # JAX: traces -> pkl
    python run_dit_tol_curves.py --lpips [PKL]       # PyTorch: LPIPS -> same pkl
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import time
import pickle
import argparse
import numpy as np

# ── config (same as run_dit_32images.py) ──
dit_ckpt  = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "pretrained_models", "DiT-XL-2-256x256.pt")
dit_T     = 50
dit_cfg   = 5.0
dit_seed  = 0
max_n     = 50
save_dir  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figs", "dit")
pkl_default = os.path.join(save_dir, "dit_tol_curves.pkl")

# iterations at which LPIPS vs sequential is computed in the --lpips phase
lpips_iters = [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 25, 30, 40, 50]

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

lat_C, lat_H, lat_W = 4, 32, 32
D = lat_C * lat_H * lat_W


def run_traces(pkl_path):
    import jax
    import jax.numpy as jnp
    from algs.picard_ddim import picard_ddim_alg
    from algs.deer_ddim import deer_ddim_alg
    from algs.broyden_ddim import broyden_ddim_alg
    from algs.accelerated_broyden_ddim import accelerated_broyden_ddim_alg
    from algs.taa import taa_ddim_alg
    from models.dit import make_dit_diffusion_reusable
    from jax.experimental import enable_x64

    print(f"Device: {jax.devices()}")
    print(f"T={dit_T}, cfg={dit_cfg}, max_n={max_n} (fixed, full trace)")
    sys.stdout.flush()

    set_class, _ = make_dit_diffusion_reusable(
        dit_ckpt, T=dit_T, cfg_scale=dit_cfg, eta=0.0, seed=dit_seed,
    )
    dynamics_0, x0, drivers = set_class(list(class_labels.values())[0])
    guess = jnp.broadcast_to(x0, (dit_T, D))

    @jax.jit
    def run_seq(dynamics, x0, drivers):
        _, traj = jax.lax.scan(
            lambda s, d: (dynamics.deer_fxn(s, d), dynamics.deer_fxn(s, d)), x0, drivers
        )
        return traj[-1]

    # Each returns (final-step latent at every iteration (max_n+1, D),
    #               merit of the pre-update states at every step (max_n,))
    def trace(res):
        return res[0][:, -1], res[6]

    @jax.jit
    def run_picard(dynamics, x0, guess, drivers):
        return trace(picard_ddim_alg(dynamics, x0, guess, drivers, max_n,
                                     full_trace=True))

    @jax.jit
    def run_quasi_deer(dynamics, x0, guess, drivers):
        return trace(deer_ddim_alg(dynamics, x0, guess, drivers, max_n,
                                   num_samples=1, k=0.0, full_trace=True))

    @jax.jit
    def run_broyden(dynamics, x0, guess, drivers):
        return trace(broyden_ddim_alg(dynamics, x0, guess, drivers, max_n,
                                      m=2, init="identity", gamma=1.0, k=0.0,
                                      full_trace=True,
                                      sequential_solve=False, compress="rand"))

    @jax.jit
    def run_accbroy(dynamics, x0, guess, drivers):
        return trace(accelerated_broyden_ddim_alg(dynamics, x0, guess, drivers, max_n,
                                                  m=2, init="identity", gamma=1.0, k=0.0,
                                                  full_trace=True, sequential_solve=False,
                                                  compress="rand"))

    @jax.jit
    def _run_taa(dynamics, x0, guess, drivers):
        return trace(taa_ddim_alg(dynamics, x0, guess, drivers, max_n,
                                  order=10, memory_size=2, lmd=1e-3,
                                  full_trace=True, gram_dtype=jnp.float64))

    def run_taa(*args):                      # float64 Gram: trace/run under x64 only
        with enable_x64():
            return _run_taa(*args)
    def _lower(*args):
        with enable_x64():
            return _run_taa.lower(*args)
    run_taa.lower = _lower

    methods = {
        "Picard": run_picard,
        "QuasiDEER": run_quasi_deer,
        "Broyden": run_broyden,
        "AccBroyden": run_accbroy,
        "TAA": run_taa,
    }

    print("\nCompiling (one-time) ...", end=" ", flush=True)
    run_seq.lower(dynamics_0, x0, drivers).compile()
    for mfn in methods.values():
        mfn.lower(dynamics_0, x0, guess, drivers).compile()
    print("done.")
    sys.stdout.flush()

    results = {
        "config": {
            "T": dit_T, "cfg": dit_cfg, "max_n": max_n, "seed": dit_seed,
            "classes": class_labels, "lpips_iters": lpips_iters,
        },
        "Sequential": {"latent": {}},
    }
    for mname in methods:
        results[mname] = {"merit": {}, "latents": {}, "time_per_iter": {}, "lpips": {}}

    for cls_name, cls_id in class_labels.items():
        print(f"\n  == {cls_name} (class {cls_id}) ==")
        sys.stdout.flush()
        dynamics_cls = set_class(cls_id)[0]

        seq_last = run_seq(dynamics_cls, x0, drivers)
        results["Sequential"]["latent"][cls_name] = np.array(seq_last)

        for mname, mfn in methods.items():
            t0 = time.time()
            lat, merit = jax.block_until_ready(mfn(dynamics_cls, x0, guess, drivers))
            dt = time.time() - t0
            lat, merit = np.array(lat), np.array(merit)
            results[mname]["latents"][cls_name] = lat
            results[mname]["merit"][cls_name] = merit
            results[mname]["time_per_iter"][cls_name] = dt / max_n
            below = np.nonzero(merit < 0.1)[0]
            it01 = int(below[0]) + 1 if below.size else None
            print(f"    {mname}: iters@tol=0.1 {it01}, final merit {merit[-1]:.3e}, "
                  f"{dt / max_n:.3f}s/iter")
        sys.stdout.flush()

    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved traces to {pkl_path}")


def run_lpips(pkl_path):
    import torch
    import lpips
    from models.dit import load_vae

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"LPIPS phase on {device}")
    vae = load_vae(device=device)
    lpips_fn = lpips.LPIPS(net="vgg").eval().to(device)

    def decode(latent):
        """Same math as models.dit.decode_latent, on `device`."""
        z = torch.tensor(np.asarray(latent).reshape(lat_C, lat_H, lat_W), device=device)
        with torch.no_grad():
            img = vae.decode(z.unsqueeze(0).float() / 0.18215).sample
            img = torch.clamp(127.5 * img + 128.0, 0, 255).to(torch.uint8)
        return img.float() / 255.0 * 2 - 1        # (1, 3, H, W) in [-1, 1]

    with open(pkl_path, "rb") as f:
        results = pickle.load(f)
    methods = [k for k in results if k not in ("config", "Sequential")]

    for cls_name in results["config"]["classes"]:
        ref = decode(results["Sequential"]["latent"][cls_name])
        parts = []
        for mname in methods:
            lats = results[mname]["latents"][cls_name]
            scores = {}
            for it in lpips_iters:
                with torch.no_grad():
                    scores[it] = lpips_fn(ref, decode(lats[it])).item()
            results[mname]["lpips"][cls_name] = scores
            parts.append(f"{mname}@15={scores[15]:.4f}")
        print(f"  {cls_name}: {', '.join(parts)}")
        sys.stdout.flush()

    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nUpdated {pkl_path} with LPIPS at iterations {lpips_iters}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lpips", nargs="?", const=pkl_default, default=None,
                        help="run the LPIPS phase on this pkl (default: %(const)s)")
    args = parser.parse_args()
    if args.lpips:
        run_lpips(args.lpips)
    else:
        run_traces(pkl_default)
