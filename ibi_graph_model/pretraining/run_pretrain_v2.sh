#!/bin/bash
#SBATCH --job-name=ibi_graph_v2
#SBATCH --partition=gpuA100x8,gpuA100x4,gpuA40x4
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --account=bebr-delta-gpu
#SBATCH -t 24:00:00
#SBATCH -e /work/nvme/bebr/mkhan14/ecg_foundation_model/ibi_graph_model/pretraining/slurm_outputs/slurm-%j.err
#SBATCH -o /work/nvme/bebr/mkhan14/ecg_foundation_model/ibi_graph_model/pretraining/slurm_outputs/slurm-%j.out

set -euo pipefail
module purge
module load cuda/12.8
module load ffmpeg/7.1

source /work/nvme/bebr/mkhan14/ecg_foundation_model/venv/bin/activate

export WANDB__SERVICE_WAIT=300
export WANDB_API_KEY="83325179ad3537a8c7a1b3a0c8daa4ea71866fae"
export PYTHONPATH=${PYTHONPATH:-}:/work/nvme/bebr/mkhan14/ecg_foundation_model

cd /work/nvme/bebr/mkhan14/ecg_foundation_model/ibi_graph_model/pretraining

torchrun --nproc_per_node=4 main.py \
    --output_dir "/work/nvme/bebr/mkhan14/ecg_foundation_model/ibi_graph_model/experiments/ibi_graph_v2" \
    --run_name   "ibi_graph_v2"

echo "========== DONE =========="
