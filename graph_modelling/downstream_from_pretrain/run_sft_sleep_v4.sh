#!/bin/bash
#SBATCH --job-name=sft_sleep_v4
#SBATCH --partition=gpuA100x8,gpuA100x4,gpuA40x4
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --account=bebr-delta-gpu
#SBATCH -t 05:00:00
#SBATCH -e /work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/downstream_from_pretrain/slurm_outputs/slurm-%j.err
#SBATCH -o /work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/downstream_from_pretrain/slurm_outputs/slurm-%j.out

module purge
module load cuda/12.8
module load ffmpeg/7.1
export PYTHONPATH=$PYTHONPATH:/work/nvme/bebr/mkhan14/ecg_foundation_model
export WANDB__SERVICE_WAIT=300
export WANDB_API_KEY="83325179ad3537a8c7a1b3a0c8daa4ea71866fae"
export CUDA_LAUNCH_BLOCKING=1

source /work/nvme/bebr/mkhan14/ecg_foundation_model/venv/bin/activate

cd /work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/downstream_from_pretrain

# Using last_checkpoint.pt (epoch 100) — not best_model.pt which was saved at
# epoch 1 during LR warmup and has near-random weights.
ENCODER_CKPT="/work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/experiments/graph_byol_v4_mse_nodrop/last_checkpoint.pt"
EXPERIMENT_DIR="$PWD/brp_sft_experiments_sleep/graph_byol_v4_mse_nodrop"
mkdir -p "$EXPERIMENT_DIR"

python main.py \
    --output_dir     "$EXPERIMENT_DIR" \
    --train_csv      "../../BRP_train.csv" \
    --test_csv       "../../BRP_test.csv" \
    --encoder_ckpt   "$ENCODER_CKPT" \
    --dataset_type   sleep \
    --num_classes    2 \
    --batch_size     32 \
    --window_sec     10 \
    --epochs         30 \
    --lr             5e-5 \
    --freeze_encoder 0 \
    --seed           42 \
    --run_name       "sft_sleep_graph_byol_v4_mse_nodrop"

echo "========== DONE =========="
