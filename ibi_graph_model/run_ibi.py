#!/bin/bash
#SBATCH --job-name=ibi_extraction
#SBATCH --output="test.out.%j.%N.out"
#SBATCH --partition=cpu
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=20   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --account=bebr-delta-cpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH -t 20:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

source ../venv/bin/activate

python3 save_ibi.py



