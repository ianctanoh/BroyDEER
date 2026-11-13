#!/bin/bash
#SBATCH --job-name=dmf_rnn_sweep
#SBATCH --output=dmf_rnn_sweep_%j.out
#SBATCH --time=24:00:00
#SBATCH -G 1
#SBATCH -C GPU_SKU:H100_SXM5
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --partition=swl1

# DMF RNN sweep of the paper: D=128, g=1.7, self-couplings, one W (seed 0) shared by 32 chains, T=1e4,
# RMS tol 1e-4, at most 1000 iterations, float32 with highest-precision matmuls. Newton damped (k=0.2),
# other methods undamped; Broyden variants: identity init with gamma=0.1, rand_rr compression, line search.
# Then: python dmf_wandb_to_dataframe.py saved_files/<name>.pkl --since <start> --T 10000 --batch_size 32

module load python/3.12.1 openblas/0.3.28
source /home/groups/swl1/ict/broydeer_venv/bin/activate
cd /home/users/ictanoh/BroyDEER/experiments

export JAX_PLATFORMS=cuda
export JAX_DEFAULT_MATMUL_PRECISION=highest

C="D=128 g=1.7 self_couple=true tol=1e-4 num_iters=1000 T=10000 batch_size=32"
python harness_dmf_rnn.py -m alg=seq,quasi $C
python harness_dmf_rnn.py alg=deer $C deer_k=0.2
python harness_dmf_rnn.py -m alg=accelerated_broyden $C acc_m=1,5,10,15,20 acc_init=identity acc_gamma=0.1 acc_compress=rand_rr
python harness_dmf_rnn.py -m alg=broyden $C bd_m=1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80 \
    bd_init=identity bd_gamma=0.1 bd_compress=rand_rr
