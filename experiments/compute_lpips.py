"""
compute_lpips.py

Post-processing script: loads pkl files from run_dit_16images.py,
decodes latents via VAE, computes LPIPS, saves images, and updates pkl.

No JAX needed — pure PyTorch + numpy.

Usage:
    python compute_lpips.py [--pkl PATH] [--pkl2 PATH]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import lpips

from models.dit import load_vae, decode_latent

# ── args ──
parser = argparse.ArgumentParser()
parser.add_argument("--pkl", nargs="+", default=[
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "figs", "dit", "dit_16images_results.pkl"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "figs", "dit", "dit_16images_b_results.pkl"),
])
args = parser.parse_args()

# ── load models ──
print("Loading VAE ...")
vae = load_vae(device="cpu")
print("Loading LPIPS (VGG) ...")
lpips_fn = lpips.LPIPS(net='vgg').eval()
print("Loaded.")

lat_C, lat_H, lat_W = 4, 32, 32

for pkl_path in args.pkl:
    if not os.path.exists(pkl_path):
        print(f"\nSkipping {pkl_path} (not found)")
        continue

    print(f"\n{'='*70}")
    print(f"Processing: {pkl_path}")
    print(f"{'='*70}")

    with open(pkl_path, "rb") as f:
        results = pickle.load(f)

    class_labels = results["config"]["classes"]
    save_dir = os.path.dirname(pkl_path)
    img_dir = os.path.join(save_dir, "16images")
    os.makedirs(img_dir, exist_ok=True)

    method_names = [k for k in results if k != "config"]

    for cls_name in class_labels:
        # decode sequential reference
        seq_latent = results["Sequential"]["latent"][cls_name].reshape(lat_C, lat_H, lat_W)
        seq_img = decode_latent(vae, seq_latent)
        seq_tensor = torch.from_numpy(seq_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        seq_tensor = seq_tensor * 2 - 1
        results["Sequential"]["lpips"][cls_name] = 0.0
        plt.imsave(os.path.join(img_dir, f"{cls_name}_sequential.png"), seq_img)

        # parallel methods
        lpips_strs = []
        for mname in method_names:
            if mname == "Sequential":
                continue
            latent = results[mname]["latent"][cls_name].reshape(lat_C, lat_H, lat_W)
            img = decode_latent(vae, latent)
            plt.imsave(os.path.join(img_dir, f"{cls_name}_{mname.lower()}.png"), img)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img_tensor = img_tensor * 2 - 1
            with torch.no_grad():
                score = lpips_fn(seq_tensor, img_tensor).item()
            results[mname]["lpips"][cls_name] = score
            lpips_strs.append(f"{mname}={score:.4f}")

        print(f"  {cls_name}: {', '.join(lpips_strs)}")

    # save updated pkl (now includes LPIPS)
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Updated {pkl_path}")

    # print summary with LPIPS
    print(f"\n{'='*80}")
    print(f"  {'Class':<20s}  {'Method':<12s}  {'Iters':>5s}  {'Time':>8s}  {'LPIPS':>8s}")
    print(f"{'-'*80}")
    for cls_name in class_labels:
        for mname in method_names:
            iters = results[mname]["iters"][cls_name]
            rt = results[mname]["runtime"][cls_name]
            lp = results[mname]["lpips"][cls_name]
            lp_str = "---" if mname == "Sequential" else f"{lp:.4f}"
            print(f"  {cls_name:<20s}  {mname:<12s}  {iters:>5d}  {rt:>7.3f}s  {lp_str:>8s}")
    print(f"{'='*80}")

    # averages
    print(f"\n  {'AVERAGE':<20s}  {'Method':<12s}  {'Iters':>5s}  {'Time':>8s}  {'LPIPS':>8s}")
    print(f"{'-'*80}")
    for mname in method_names:
        avg_iters = np.mean(list(results[mname]["iters"].values()))
        avg_rt = np.mean(list(results[mname]["runtime"].values()))
        avg_lp = np.mean(list(results[mname]["lpips"].values()))
        lp_str = "---" if mname == "Sequential" else f"{avg_lp:.4f}"
        print(f"  {'':>20s}  {mname:<12s}  {avg_iters:>5.1f}  {avg_rt:>7.3f}s  {lp_str:>8s}")
    print(f"{'='*80}")

print("\nDone!")
