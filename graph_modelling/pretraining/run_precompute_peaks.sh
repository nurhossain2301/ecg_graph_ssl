#!/bin/bash
#SBATCH --job-name=precompute_peaks
#SBATCH --partition=cpu
#SBATCH --account=bebr-delta-cpu
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 4:00:00
#SBATCH -e /work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/pretraining/slurm_outputs/slurm-%j.err
#SBATCH -o /work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/pretraining/slurm_outputs/slurm-%j.out

module purge

source /work/nvme/bebr/mkhan14/ecg_foundation_model/venv/bin/activate

cd /work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/pretraining

python precompute_peaks.py

echo "========== DONE =========="
