#!/bin/bash
#SBATCH --job-name=ecg_token_graph_ssl
#SBATCH --output="logs/%x.%j.out"
#SBATCH --partition=gpuA100x8,gpuA100x4,gpuA40x4
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --account=bebr-delta-gpu
#SBATCH -t 20:00:00
#SBATCH -e /work/nvme/bebr/mkhan14/ecg_foundation_model/hybrid_graph_token_generation/slurm_outputs/%x.%j.err
#SBATCH -o /work/nvme/bebr/mkhan14/ecg_foundation_model/hybrid_graph_token_generation/slurm_outputs/%x.%j.out

# =============================================
# SETUP
# =============================================
echo "========== JOB START =========="
date

# Go to project directory (IMPORTANT)
cd $SLURM_SUBMIT_DIR

# Create logs dir if not exists
mkdir -p logs
mkdir -p outputs

# =============================================
# LOAD MODULES
# =============================================
module purge
module load cuda/12.8
module load ffmpeg/7.1

# =============================================
# ENVIRONMENT
# =============================================
source ../venv/bin/activate

export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# DDP
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# WANDB
export WANDB__SERVICE_WAIT=300
export WANDB_API_KEY="83325179ad3537a8c7a1b3a0c8daa4ea71866fae"

# Optional: avoid tokenizer deadlock
export TOKENIZERS_PARALLELISM=false

# =============================================
# PRINT GPU INFO
# =============================================
echo "========== GPU INFO =========="
nvidia-smi

# =============================================
# STAGE 1: TRAIN CODEBOOK (ONLY IF NOT EXISTS)
# =============================================
# python train_motif_encoder.py
CODEBOOK_PATH="./codebook.npy"

if [ ! -f "$CODEBOOK_PATH" ]; then
    echo "🔵 Training codebook..."
    python train_codebook.py
    echo "✅ Codebook created"
else
    echo "🟢 Codebook already exists, skipping..."
fi

# =============================================
# STAGE 2: TRAIN MODEL (DDP)
# =============================================
echo "🚀 Starting training..."

torchrun \
    --nproc_per_node=1 \
    --master_port=29501 \
    main.py

# =============================================
# DONE
# =============================================
echo "========== DONE =========="
date