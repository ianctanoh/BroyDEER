# BroyDEER

Quasi-Newton (Broyden) fixed-point solvers for parallelizing the evaluation of
discrete-time dynamical systems, written in JAX. The package (`brodeer`) contains:

| Solver | Module |
| --- | --- |
| Picard, DEER (full Newton) and quasi-DEER (diagonal Jacobian) | `src/algs/deer.py` |
| Broyden-DEER | `src/algs/broyden_deer.py` |
| Accelerated Broyden-DEER | `src/algs/accelerated_broyden.py` |
| Global Broyden over the full TD x TD Jacobian | `src/algs/full_broyden.py` |
| Triangular Anderson Acceleration (TAA, adapted from ParaTAA) | `src/algs/taa.py` |
| Associative-scan tree topologies (Blelloch / Hillis-Steele) | `src/algs/scan_trees.py` |
| DDIM-specialized variants (Picard, DEER, Broyden, Accelerated Broyden) | `src/algs/*_ddim.py` |

Dynamical systems live in `src/dynamics/` (N-schools Gibbs sampler, low-rank RNN,
discretized mean-field RNN, GRU, S_N word problem, DDIM diffusion), and the
JAX port of DiT-XL/2 is in `src/models/dit.py`.

## Installation

Requires Python >= 3.10 and a CUDA 12 GPU for the timing experiments.

```bash
python -m venv broydeer_venv
source broydeer_venv/bin/activate
pip install -e .            # core solvers + RNN / N-schools experiments
pip install -e ".[dit]"     # additionally: PyTorch, diffusers, lpips for DiT experiments
```

Most experiments need 64-bit floats:

```bash
export JAX_ENABLE_X64=True
```

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

and export finished sweeps to pandas pickles with

```bash
python wandb_sweep_to_pickle.py --project <entity>/<project> --sweep_id <id> \
    --pickle_name <name> --date <YYYY-MM-DD>      # -> experiments/saved_files/
```

| Paper | Experiment | Command(s) |
| --- | --- | --- |
| TODO | N-schools: iterations and wall-clock vs. solver (sequential, DEER, quasi-DEER, Parallel Broyden, Accelerated Parallel Broyden; S in {10,50,100,200}, T in {1k,10k,50k,100k}; float32, tol=1e-3) | `wandb sweep wandb_sweep_configs/Nschools_final_float32.yaml` |
| TODO | N-schools: damping search for each solver | `wandb sweep wandb_sweep_configs/Nschools_float32_tol1e-3_damping_{broyden,accel,deer,quasi}.yaml` |
| TODO | N-schools: line-search ablation (Broyden solvers without the safeguard; Accelerated Broyden safeguard step) | `wandb sweep wandb_sweep_configs/Nschools_float32_tol1e-3_no_linesearch.yaml`, `wandb sweep wandb_sweep_configs/Nschools_float32_tol1e-3_accel_ls_c.yaml` |
| TODO | N-schools: float32 vs. float64 at tol=1e-3 | `wandb sweep wandb_sweep_configs/Nschools_float32_tol1e-3.yaml`, `wandb sweep wandb_sweep_configs/Nschools_tol1e-3.yaml` (ran with damping 0.2 for quasi-DEER and both Broyden solvers; `Nschools_float32_tol1e-3.yaml` now holds the final defaults) |
| TODO | N-schools: diagonal quasi-Newton with a Hutchinson diagonal (1, 8, 64 Rademacher probes; final settings) | `JAX_ENABLE_X64=False python harness_Nschools.py -m alg=quasi_qmem S=10,50,100,200 T=1000,10000 num_samples=1,8,64 seed=13 N_s=20 batch_size=32 num_iters=1000 tol=1e-3 nwarmups=3 nreps=3 quasi_k=0.0 tausq_0=100.0 sigmasq_0=10.0`, then `python nschools_wandb_to_dataframe.py saved_files/<name>.pkl --since <start> --alg quasi_qmem` |
| TODO | N-schools: tolerance study (accuracy vs. sequential Gibbs sampling, float32) | `python nschools_tol_study.py` |
| TODO | N-schools: trace plots (sequential vs. parallel solvers) | `python nschools_tol_study.py --S 100 --T 100000 --tols 1e-3 --save_chain 7 --traj_dir <DIR>`, then `python plot_nschools_traces.py --traj_dir <DIR>` |
| TODO | Low-rank RNNs (standard and MLP, D in {100,200}, dt in {0.5,3}, T in {1e3,1e4,1e5}, 32 chains, float32): sequential, Newton, diagonal quasi-Newton, Parallel Broyden, Accelerated Parallel Broyden | `python harness_rnn_lowrank.py --out <dir>/<name>.pkl` (default arguments = the full grid; one job per setting: `--configs <model:D:L> --dt <dt>`), then `python rnn_lowrank_to_dataframe.py <dir> saved_files/<name>.pkl` |
| TODO | Low-rank RNNs: diagonal quasi-Newton with a Hutchinson diagonal | `python harness_rnn_lowrank.py --methods quasi_hutch` |
| TODO | Low-rank RNNs: tolerance study | `python rnn_lowrank_tol_study.py` |
| TODO | Low-rank RNNs: scan tree (Blelloch vs. Hillis-Steele, float64) | `python harness_rnn_lowrank.py --configs std:200:0 mlp:200:8 --methods broyden accel --no_timing --x64 --scan_tree {blelloch,hillis_steele} --out <name>.pkl` |
| TODO | GRU | `python harness_gru.py -m alg=seq,deer,quasi D=32 tol=1e-5`<br>`python harness_gru.py -m alg=broyden D=32 bd_m=1,2,3,5,10,15,20 tol=1e-5`<br>`python harness_gru.py -m alg=accelerated_broyden D=32 acc_m=1,2,3,5,10,15,20 tol=1e-5` |
| TODO | Discretized mean-field RNN (D=128, g=1.7, T=1e4, 32 chains, tol=1e-4, float32): Newton, quasi-DEER, Parallel Broyden m in {1,5,...,80}, Accelerated Parallel Broyden m in {1,5,10,15,20} | `sbatch run_sweep_dmf_rnn.sh` (commands inside), then `python dmf_wandb_to_dataframe.py saved_files/<name>.pkl --since <start> --T 10000 --batch_size 32` (latest run per setting) |
| TODO | S_N word problem | `python harness_sn_word.py -m alg=seq,deer,quasi N=5,15,30`<br>`python harness_sn_word.py -m alg=accelerated_broyden N=<N> acc_m=<m list>` (N=5: `1,3,4,5`; N=15: `5,10,12,13,15`; N=30: `1,3,5,10,15,20,30`) |
| TODO | DiT-XL/2: Picard, QuasiDEER, Broyden, Accelerated Broyden and TAA on 32 ImageNet classes | `python run_dit_32images.py`, then `python compute_lpips.py` |
| TODO | DiT-XL/2: TAA with float64 Gram accumulation/solve (ParaTAA precision) vs float32, 32 classes | `python run_dit_taa_gram64.py` |
| TODO | DiT-XL/2: Broyden and TAA hyperparameter sweeps (32 ImageNet classes) | `python run_{broyden_sweep,taa_sweep}.py`, then `python compute_lpips.py --pkl <results>.pkl` |
| TODO | DiT-XL/2: tolerance study (residual and LPIPS vs sequential at every iteration) | `python run_dit_tol_curves.py`, then `python run_dit_tol_curves.py --lpips` |
| TODO | DiT-XL/2: per-iteration convergence images | `python run_dit_trace.py --class_id <id> --name <name>` (default: orangutan, 365) |

The DiT scripts write their results to `figs/dit/`. Exploratory notebooks are
in `nbs/`.

## Citation

```bibtex
TODO
```

## License

MIT, see [LICENSE](LICENSE).
