#!/bin/bash
#SBATCH --job-name=sft_care_v2_30s
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

ENCODER_CKPT="/work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/experiments/graph_byol_v4_mse_nodrop/last_checkpoint.pt"
EXPERIMENT_DIR="$PWD/brp_sft_experiments_caregiver/graph_byol_v4_mse_nodrop_v2_30s"
mkdir -p "$EXPERIMENT_DIR"

# Dataset v2: session-level normalization + dual-window temporal context
# window_sec=30: gives ANS time to respond to caregiving touch (~15-30s lag)
# max_beats=80: per-window budget at ~150 bpm infant HR; dual-window
#               produces 160-beat sequences that the model handles via masking
python main.py \
    --output_dir     "$EXPERIMENT_DIR" \
    --train_csv      "../../BRP_train.csv" \
    --test_csv       "../../BRP_test.csv" \
    --encoder_ckpt   "$ENCODER_CKPT" \
    --dataset_type   caregiver_v2 \
    --num_classes    2 \
    --batch_size     32 \
    --window_sec     30 \
    --max_beats      80 \
    --epochs         30 \
    --lr             5e-5 \
    --freeze_encoder 0 \
    --head_type      residual_mlp \
    --seed           42 \
    --run_name       "sft_caregiver_v2_30s_session_norm_dual_window_lastckpt"

echo "========== DONE =========="
