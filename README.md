# BroyDEER

Code for **Parallel Broyden Methods for Efficiently Evaluating Nonlinear State Space Models**
(Ian C. Tanoh and Scott W. Linderman, NeurIPS 2026, [arXiv:2611.13523](https://arxiv.org/abs/2611.13523)).

Quasi-Newton (Broyden) fixed-point solvers for parallelizing the evaluation of
discrete-time dynamical systems, written in JAX. The package (`broydeer`) contains:

| Solver | Module |
| --- | --- |
| Picard, Newton (DEER) and diagonal quasi-Newton (quasi-DEER) | `src/algs/deer.py` |
| Parallel Broyden (`alg=broyden`) | `src/algs/broyden_deer.py` |
| Accelerated Parallel Broyden (`alg=accelerated_broyden`) | `src/algs/accelerated_broyden.py` |
| Global Broyden over the full TD x TD Jacobian | `src/algs/full_broyden.py` |
| Triangular Anderson Acceleration (TAA, adapted from ParaTAA) | `src/algs/taa.py` |
| Associative-scan tree topologies (Blelloch / Hillis-Steele) | `src/algs/scan_trees.py` |
| DDIM-specialized variants (Picard, DEER, Broyden, Accelerated Broyden) | `src/algs/*_ddim.py` |

Dynamical systems live in `src/dynamics/` (N-schools Gibbs sampler, low-rank RNN,
discretized mean-field RNN, DDIM diffusion), and the
JAX port of DiT-XL/2 is in `src/models/dit.py`.

## Installation

Requires Python >= 3.10 and a CUDA 12 GPU for the timing experiments. The paper's runtimes were measured on a
single NVIDIA H100 (80 GB); the largest Newton runs (T=1e5) need most of its memory. Iteration counts are
reproducible across GPUs (fixed seeds), runtimes are not.

```bash
python -m venv broydeer_venv
source broydeer_venv/bin/activate
pip install -e .            # core solvers + RNN / N-schools experiments
pip install -e ".[dit]"     # additionally: PyTorch, diffusers, lpips for DiT experiments
```

The paper's experiments run in float32. The RNN and N-schools experiments use highest-precision
matrix multiplications (no TF32), which the harnesses set themselves (`jax_default_matmul_precision="highest"`);
the DiT experiments use JAX's default precision (TF32 on recent GPUs). Do not set `JAX_ENABLE_X64=True` for them;
the N-schools commands below set `JAX_ENABLE_X64=False` explicitly, since `harness_Nschools.py` defaults to float64
otherwise.

### DiT weights

The diffusion experiments use the pretrained DiT-XL/2 256x256 checkpoint from
[facebookresearch/DiT](https://github.com/facebookresearch/DiT):

```bash
mkdir -p pretrained_models
wget -P pretrained_models https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt
```

The VAE decoder (`stabilityai/sd-vae-ft-mse`) is downloaded automatically by
`diffusers` on first use.

## Reproducing the paper

All commands are run from `experiments/`. Harnesses use
[Hydra](https://hydra.cc) for configuration (`experiments/configs/`) and log to
[Weights & Biases](https://wandb.ai). Grid sweeps are wandb sweep files; launch
them with

```bash
wandb sweep <sweep_file>.yaml        # prints a sweep id
wandb agent <entity>/<project>/<sweep_id>
```

(this needs a wandb account and `wandb login`). The Hydra harnesses also run without an account with
`mode=offline`; the DMF and N-schools converters below read the local `wandb/` run folders, so offline runs
work with them. Export finished sweeps to pandas pickles with

```bash
python wandb_sweep_to_pickle.py --project <entity>/<project> --sweep_id <id> \
    --pickle_name <name> --date <YYYY-MM-DD>      # -> experiments/saved_files/
```

| Paper | Experiment | Command(s) |
| --- | --- | --- |
| Fig. 1 (left, middle), Tables 8, 11 | Low-rank RNNs (standard and MLP, D in {100,200}, dt in {0.5,3}, T in {1e3,1e4,1e5}, 32 chains): sequential, Newton, diagonal quasi-Newton, Parallel Broyden, Accelerated Parallel Broyden | `python harness_rnn_lowrank.py --out <dir>/<name>.pkl` (default arguments = the full grid; one job per setting: `--configs <model:D:L> --dt <dt>`), then `python rnn_lowrank_to_dataframe.py <dir> saved_files/<name>.pkl` |
| Fig. 1 (right) | Discretized mean-field RNN (D=128, g=1.7, T=1e4, 32 chains): Parallel Broyden m in {1,5,...,80}, Accelerated Parallel Broyden m in {1,5,10,15,20} | `bash run_sweep_dmf_rnn.sh` (commands inside), then `python dmf_wandb_to_dataframe.py saved_files/<name>.pkl --since <start> --T 10000 --batch_size 32` (latest run per setting) |
| Fig. 2 | N-schools Gibbs sampler: S in {10,50,100,200}, T in {1e3,1e4,5e4,1e5} | `wandb sweep wandb_sweep_configs/Nschools_final_float32.yaml` |
| App. G (damping) | N-schools: damping search for each solver | `wandb sweep wandb_sweep_configs/Nschools_float32_tol1e-3_damping_{broyden,accel,deer,quasi}.yaml` |
| Table 9 | N-schools: diagonal quasi-Newton with a Hutchinson diagonal (1, 8, 64 probes) | `JAX_ENABLE_X64=False python harness_Nschools.py -m alg=quasi_qmem S=10,50,100,200 T=1000,10000 num_samples=1,8,64 seed=13 N_s=20 batch_size=32 num_iters=1000 tol=1e-3 nwarmups=3 nreps=3 quasi_k=0.0 tausq_0=100.0 sigmasq_0=10.0`, then `python nschools_wandb_to_dataframe.py saved_files/<name>.pkl --since <start> --alg quasi_qmem` |
| Fig. 3, Table 10 | DiT-XL/2: Picard, diagonal quasi-Newton, Parallel Broyden, Accelerated Parallel Broyden and ParaTAA on 32 ImageNet classes | `python run_dit_32images.py`, then `python compute_lpips.py` |
| Fig. 3A | DiT-XL/2: per-iteration convergence images | `python run_dit_trace.py --class_id <id> --name <name>` (default: orangutan, 365) |
| Figs. 12, 13 | DiT-XL/2: ParaTAA and Parallel Broyden hyperparameter sweeps | `python run_{taa_sweep,broyden_sweep}.py`, then `python compute_lpips.py --pkl <results>.pkl` |
| App. H.6 | DiT-XL/2: tolerance study (residual and LPIPS vs. sequential at every iteration) | `python run_dit_tol_curves.py`, then `python run_dit_tol_curves.py --lpips` |
| App. H.6 | DiT-XL/2: ParaTAA with a float64 Gram system | `python run_dit_taa_gram64.py` |
| Table 2 | Low-rank RNNs: scan tree (Blelloch vs. Hillis-Steele, float64) | `python harness_rnn_lowrank.py --configs std:200:0 mlp:200:8 --methods broyden accel --no_timing --x64 --scan_tree {blelloch,hillis_steele} --out <name>.pkl` |
| Fig. 5 | Low-rank RNN (earlier configuration: dt=3, N up to 200, T up to 2e5): rank compression (sequential solve vs. QR+SVD vs. randomized) | `wandb sweep sweep_rnn_A_compress.yaml` |
| Fig. 6 | Low-rank RNN (earlier configuration): truncation vs. L-BFGS-style identity rescaling | `wandb sweep sweep_rnn_A_broyden_rescaled.yaml` |
| Fig. 7 | Low-rank RNN (earlier configuration): damping of Parallel Broyden | `wandb sweep sweep_rnn_A_broyden_damping.yaml` |
| App. D (Tables 3-6) | Convergence diagnostics (sigma, rho, Dennis-More, compression error) along solver runs | `python theory/broyden_theory_diagnostics.py`, `python theory/broyden_theory_diagnostics.py --methods quasi --sigma_every 100000 --num_iters 200 --out saved_files/broyden_theory_diagnostics_quasi_sigma_last.pkl`, `python theory/broyden_theory_diagnostics_nschools.py --S 10 50 100`, then `python theory/make_theory_summary.py` |
| Tables 11, 12 | Runtime decomposition and wall-clock breakdown | `python speedup/speedup_decomposition.py --lowrank <low-rank pkl> --nschools <N-schools pkl> --dit <DiT pkl>` (results of the Fig. 1-3 rows), `python speedup/wallclock_breakdown.py`, `python speedup/wallclock_breakdown.py --benchmarks dit --precision default --out saved_files/wallclock_breakdown_dit_default_precision.pkl`, then `python speedup/make_speedup_tables.py` |

The DiT scripts write their results to `figs/dit/`. Exploratory notebooks are
in `nbs/`.

## Citation

```bibtex
@inproceedings{tanoh2026parallel,
  title     = {Parallel Broyden Methods for Efficiently Evaluating Nonlinear State Space Models},
  author    = {Tanoh, Ian Christopher and Linderman, Scott W.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  eprint    = {2611.13523},
  archivePrefix = {arXiv}
}
```

## License

MIT, see [LICENSE](LICENSE).
