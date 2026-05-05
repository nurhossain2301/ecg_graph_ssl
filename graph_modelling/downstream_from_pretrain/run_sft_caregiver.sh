#!/bin/bash
#SBATCH --job-name=ecg_graph_caregiver
#SBATCH --partition=gpuA100x8,gpuA100x4,gpuA40x4
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --account=bebr-delta-gpu
#SBATCH -t 5:00:00
#SBATCH -e /work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/downstream_from_pretrain/slurm_outputs/slurm-%j.err
#SBATCH -o /work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/downstream_from_pretrain/slurm_outputs/slurm-%j.out

module purge
module load cuda/12.8
module load ffmpeg/7.1
export PYTHONPATH=$PYTHONPATH:/work/nvme/bebr/mkhan14/ecg_foundation_model

export CUDA_LAUNCH_BLOCKING=1
source /work/nvme/bebr/mkhan14/ecg_foundation_model/venv/bin/activate

export WANDB__SERVICE_WAIT=300
export WANDB_API_KEY="83325179ad3537a8c7a1b3a0c8daa4ea71866fae"

# =============================================
# SET PATHS
# =============================================
PROJECT_DIR=/work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/downstream_from_pretrain
EXPERIMENT_DIR="$PROJECT_DIR/brp_sft_experiments_caregiver"
ENCODER_CKPT="/work/nvme/bebr/mkhan14/ecg_foundation_model/graph_modelling/pretraining/best_model.pt"
TRAIN_CSV="/work/nvme/bebr/mkhan14/ecg_foundation_model/BRP_train.csv"
TEST_CSV="/work/nvme/bebr/mkhan14/ecg_foundation_model/BRP_test.csv"
mkdir -p $EXPERIMENT_DIR
mkdir -p $PROJECT_DIR/logs_sft

cd $PROJECT_DIR

python main.py \
    --output_dir "$EXPERIMENT_DIR/sft_out" \
    --train_csv $TRAIN_CSV \
    --test_csv $TEST_CSV \
    --encoder_ckpt $ENCODER_CKPT \
    --dataset_type caregiver \
    --batch_size 32 \
    --window_sec 10 \
    --epochs 20 \
    --lr 5e-5 \
    --freeze_encoder 0 \
    --seed 42 \
    --run_name "sft_brp-caregiver_v1.0_window_sec_10_2class" \
    --num_classes 2

echo "========== DONE =========="
