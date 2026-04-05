#!/bin/bash
#SBATCH --job-name=ecg_hubert_ssl_v2
#SBATCH --output="logs/%x.%j.out"
#SBATCH --partition=gpuA100x8,gpuA100x4,gpuA40x4
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=4   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus-per-node=1
#SBATCH --account=bebr-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH -t 20:00:00
#SBATCH -e /work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/slurm_outputs/slurm-%j.err
#SBATCH -o /work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/slurm_outputs/slurm-%j.out

# =============================================
# LOAD MODULES (edit for your cluster)
# =============================================
module purge
module load cuda/12.8
module load ffmpeg/7.1

export CUDA_LAUNCH_BLOCKING=1
# Activate environment
source ../venv/bin/activate
# python test_dataloader.py

export WANDB__SERVICE_WAIT=300
export WANDB_API_KEY="83325179ad3537a8c7a1b3a0c8daa4ea71866fae"




torchrun --nproc_per_node=1 main.py
# python main.py 


echo "========== DONE =========="