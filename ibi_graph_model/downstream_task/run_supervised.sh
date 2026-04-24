#!/bin/bash
#SBATCH --job-name=ibi_graph_sup
#SBATCH --output="logs/%x.%j.out"
#SBATCH --partition=gpuA100x8,gpuA100x4,gpuA40x4
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --account=bebr-delta-gpu
#SBATCH -t 12:00:00
#SBATCH -e /work/nvme/bebr/mkhan14/ecg_foundation_model/ibi_graph_model/downstream_task/slurm_outputs/slurm-%j.err
#SBATCH -o /work/nvme/bebr/mkhan14/ecg_foundation_model/ibi_graph_model/downstream_task/slurm_outputs/slurm-%j.out

export PYTHONPATH=$PYTHONPATH:/work/nvme/bebr/mkhan14/ecg_foundation_model
module purge
module load cuda/12.8
module load ffmpeg/7.1
source ../../venv/bin/activate

python main_supervised.py
