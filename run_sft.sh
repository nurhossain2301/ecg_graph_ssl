#!/bin/bash
#SBATCH --job-name=ecg_hubert_ssl_v2
#SBATCH --output="test.out.%j.%N.out"
#SBATCH --partition=gpuA100x8,gpuA100x4,gpuA40x4
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=4   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus-per-node=1
#SBATCH --account=bebr-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH -t 14:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

# =============================================
# LOAD MODULES (edit for your cluster)
# =============================================
module purge
module load cuda/12.8
module load ffmpeg/7.1

export CUDA_LAUNCH_BLOCKING=1
# Activate environment
source venv/bin/activate
# python test_dataloader.py

export WANDB__SERVICE_WAIT=300
export WANDB_API_KEY="83325179ad3537a8c7a1b3a0c8daa4ea71866fae"



# =============================================
# SET PATHS
# =============================================
PROJECT_DIR=$PWD
EXPERIMENT_DIR="$PROJECT_DIR/brp_sft_experiments"
ENCODER_CKPT="/work/hdd/bebr/Projects/ecg_foundational_model/experiments/stage1_model/best_model.pt"
TRAIN_CSV="BRP_train.csv"
TEST_CSV="BRP_test.csv"
# CKPT_DIR="/work/nvme/bebr/mkhan14/ecg_foundation_model/brp_sft_experiments/sft_out"
mkdir -p $EXPERIMENT_DIR
mkdir -p logs_sft


python eval.py \
    --ckpt "$EXPERIMENT_DIR/sft_out/best.pt" \
    --test_csv $TEST_CSV \
    --output_dir eval_results

# python main.py \
#     --output_dir "$EXPERIMENT_DIR/sft_out" \
#     --train_csv $TRAIN_CSV \
#     --test_csv $TEST_CSV \
#     --encoder_ckpt $ENCODER_CKPT \
#     --batch_size 16 \
#     --window_sec 10 \
#     --epochs 50 \
#     --lr 1e-4 \
#     --freeze_encoder 1 \
#     --seed 42 \
#     --run_name "sft_brp_v1.1_window_sec_10"


echo "========== DONE =========="