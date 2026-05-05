#!/bin/bash
# Submit one pretraining job per loss function.
# Each job writes to its own output directory — no shared state.
# Usage: bash run_ablation.sh

set -euo pipefail

PROJ=/work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling
EXPERIMENTS=${PROJ}/experiments

LOSSES=(baseline huber vicreg barlow spectral)

for LOSS in "${LOSSES[@]}"; do

    RUN_NAME="graph_byol_${LOSS}"
    OUTPUT_DIR="${EXPERIMENTS}/${RUN_NAME}"
    mkdir -p "${OUTPUT_DIR}"

    sbatch <<SLURM
#!/bin/bash
#SBATCH --job-name=ecg_pretrain_${LOSS}
#SBATCH --partition=gpuA100x8,gpuA100x4,gpuA40x4
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --account=bebr-delta-gpu
#SBATCH -t 10:00:00
#SBATCH -e ${PROJ}/pretraining/slurm_outputs/slurm-%j.err
#SBATCH -o ${PROJ}/pretraining/slurm_outputs/slurm-%j.out

set -euo pipefail
module purge
module load cuda/12.8
module load ffmpeg/7.1

source /work/nvme/bebr/mkhan14/ecg_foundation_model/venv/bin/activate

export WANDB__SERVICE_WAIT=300
export WANDB_API_KEY="83325179ad3537a8c7a1b3a0c8daa4ea71866fae"
export PYTHONPATH=\${PYTHONPATH:-}:/work/nvme/bebr/mkhan14/ecg_foundation_model

cd ${PROJ}/pretraining

torchrun --nproc_per_node=1 main.py \
    --output_dir "${OUTPUT_DIR}" \
    --run_name   "${RUN_NAME}" \
    --loss_type  "${LOSS}"

echo "========== DONE: ${LOSS} =========="
SLURM

    echo "Submitted: ${RUN_NAME} → ${OUTPUT_DIR}"
done
