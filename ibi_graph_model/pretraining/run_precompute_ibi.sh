#!/bin/bash
#SBATCH --job-name=ibi_precompute
#SBATCH --partition=cpu
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --account=bebr-delta-cpu
#SBATCH -t 14:00:00
#SBATCH -e /work/nvme/bebr/mkhan14/ecg_foundation_model/ibi_graph_model/pretraining/slurm_outputs/slurm-%j.err
#SBATCH -o /work/nvme/bebr/mkhan14/ecg_foundation_model/ibi_graph_model/pretraining/slurm_outputs/slurm-%j.out

set -euo pipefail

source /work/nvme/bebr/mkhan14/ecg_foundation_model/venv/bin/activate
export PYTHONPATH=${PYTHONPATH:-}:/work/nvme/bebr/mkhan14/ecg_foundation_model

IBI_DIR=/work/hdd/bebr/Projects/ecg_foundational_model/ibi_precomputed
mkdir -p "${IBI_DIR}/train" "${IBI_DIR}/val"

cd /work/nvme/bebr/mkhan14/ecg_foundation_model/ibi_graph_model/pretraining

echo "===== Precomputing TRAIN set (865 files) ====="
python precompute_ibi.py \
    --input_csv   /work/hdd/bebr/Projects/ecg_foundational_model/ECG_train_files.csv \
    --ecg_col     filename \
    --output_dir  "${IBI_DIR}/train" \
    --output_csv  "${IBI_DIR}/IBI_train_files.csv" \
    --num_workers 20 \
    --skip_done

echo "===== Precomputing VAL set (136 files) ====="
python precompute_ibi.py \
    --input_csv   /work/hdd/bebr/Projects/ecg_foundational_model/ECG_val_files.csv \
    --ecg_col     filename \
    --output_dir  "${IBI_DIR}/val" \
    --output_csv  "${IBI_DIR}/IBI_val_files.csv" \
    --num_workers 20 \
    --skip_done

echo "========== DONE =========="
echo "Update config.py train_csv / test_csv to point at the generated CSVs:"
echo "  train_csv: ${IBI_DIR}/IBI_train_files.csv"
echo "  test_csv:  ${IBI_DIR}/IBI_val_files.csv"
