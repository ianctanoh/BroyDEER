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
| TODO | N-schools: iterations and wall-clock vs. solver | `wandb sweep wandb_sweep_configs/Nschools.yaml`, then `python plot_Nschools.py --pickle saved_files/<pickle>.pkl` |
| TODO | N-schools: damping k=0.2 | `wandb sweep wandb_sweep_configs/Nschools_k02.yaml` |
| TODO | N-schools: rescaled | `wandb sweep wandb_sweep_configs/Nschools_rescaled.yaml` |
| TODO | N-schools: Broyden memory (S=10/50/100) | `wandb sweep wandb_sweep_configs/Nschools_qmem_S{10,50,100}.yaml` |
| TODO | N-schools: S=100, T=100k, sequential vs. Accelerated Broyden | `wandb sweep wandb_sweep_configs/Nschools_S100_T100k.yaml` |
| TODO | Low-rank RNN: runtime vs. (N, T), regime A (dt=3) | `wandb sweep sweep_rnn_A.yaml` (also `sweep_rnn_A_full.yaml`, `_N1000`, `_T500k`, `_rescaled`) |
| TODO | Low-rank RNN: regime B (dt=7) | `wandb sweep sweep_rnn_B.yaml` |
| TODO | Low-rank RNN: Broyden damping / rescaling | `wandb sweep sweep_rnn_A_broyden_damping.yaml`, `sweep_rnn_A_broyden_rescaled.yaml` |
| TODO | Low-rank RNN: Jacobian recovery | `wandb sweep sweep_rnn_A_jacobian.yaml` (also `_jacobian_nodamp`, `_jac_perchain`, `_jac_perchain_N200`, `_jac_patch`) |
| TODO | Low-rank RNN: compression scheme | `wandb sweep sweep_rnn_A_compress.yaml` |
| TODO | Low-rank RNN: scan tree topology | `wandb sweep sweep_rnn_A_scan_tree.yaml` |
| TODO | Low-rank RNN: single runs | `python harness_low_rank_rnn.py alg=<alg> ...` (sweep: `wandb_sweep_configs/low_rank_rnn.yaml`) |
| TODO | GRU | `python harness_gru.py -m alg=seq,deer,quasi D=32 tol=1e-5`<br>`python harness_gru.py -m alg=broyden D=32 bd_m=1,2,3,5,10,15,20 tol=1e-5`<br>`python harness_gru.py -m alg=accelerated_broyden D=32 acc_m=1,2,3,5,10,15,20 tol=1e-5` |
| TODO | Discretized mean-field RNN | `python harness_dmf_rnn.py -m alg=seq,deer,quasi D=128 T=5000 g=1.7`<br>`python harness_dmf_rnn.py -m alg=broyden D=128 T=5000 g=1.7 bd_m=1,2,3,5,10,15,20,25 bd_init=identity bd_gamma=0.1`<br>`python harness_dmf_rnn.py -m alg=accelerated_broyden D=128 T=5000 g=1.7 acc_m=1,2,3,5,10,15,20,25 acc_init=identity acc_gamma=0.1` |
| TODO | S_N word problem | `python harness_sn_word.py -m alg=seq,deer,quasi N=5,15,30`<br>`python harness_sn_word.py -m alg=accelerated_broyden N=<N> acc_m=<m list>` (N=5: `1,3,4,5`; N=15: `5,10,12,13,15`; N=30: `1,3,5,10,15,20,30`) |
| TODO | DiT-XL/2 DDIM: solver comparison | `python run_dit_taa.py` |
| TODO | DiT-XL/2: Broyden / Accelerated Broyden / TAA hyperparameter sweeps | `python run_dit_broyden_sweep.py`, `python run_dit_accel_broyden_sweep.py`, `python run_dit_taa_sweep.py` |
| TODO | DiT-XL/2: Picard, QuasiDEER, Broyden, Accelerated Broyden and TAA on 32 ImageNet classes | `python run_dit_32images.py`, then `python compute_lpips.py` |
| TODO | DiT-XL/2: TAA with float64 Gram accumulation/solve (ParaTAA precision) vs float32, 32 classes | `python run_dit_taa_gram64.py` |
| TODO | DiT-XL/2: Broyden, rank and TAA hyperparameter sweeps (32 ImageNet classes) | `python run_{broyden_sweep,broyden_rank_sweep,taa_sweep}.py`, then `python compute_lpips.py --pkl <results>.pkl` |
| TODO | DiT-XL/2: multi-GPU Broyden-DEER | `python run_dit_broyden_multigpu.py` |
| TODO | DiT-XL/2: per-iteration convergence images | `python run_gorilla_trace.py` |

The DiT scripts write their results to `figs/dit/`. Exploratory notebooks are
in `nbs/`.

## Citation

```bibtex
TODO
```

## License

MIT, see [LICENSE](LICENSE).
