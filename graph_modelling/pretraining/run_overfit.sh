#!/bin/bash
#SBATCH --job-name=ecg_graph_overfit
#SBATCH --partition=gpuA100x8,gpuA100x4,gpuA40x4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --account=bebr-delta-gpu
#SBATCH -t 2:00:00
#SBATCH -e /work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/pretraining/slurm_outputs/slurm-%j.err
#SBATCH -o /work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/pretraining/slurm_outputs/slurm-%j.out

set -euo pipefail

module purge
module load cuda/12.8
module load ffmpeg/7.1

source /work/nvme/bebr/mkhan14/ecg_foundation_model/venv/bin/activate

export WANDB__SERVICE_WAIT=300
export WANDB_API_KEY="83325179ad3537a8c7a1b3a0c8daa4ea71866fae"
export PYTHONPATH=${PYTHONPATH:-}:/work/nvme/bebr/mkhan14/ecg_foundation_model

RUN_NAME="graph_byol_overfit_check"
OUTPUT_DIR="/work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/experiments/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

cd /work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/pretraining

torchrun --nproc_per_node=1 main.py \
    --output_dir "${OUTPUT_DIR}" \
    --run_name   "${RUN_NAME}" \
    --overfit

echo "========== DONE =========="
