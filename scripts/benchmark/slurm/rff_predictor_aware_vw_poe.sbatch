#!/bin/bash
#SBATCH --job-name=rff_pa_poe
#SBATCH --output=logs/rff_predictor_aware_vw_poe_%A_%a.out
#SBATCH --error=logs/rff_predictor_aware_vw_poe_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=atl1-1-01-005-17-0,atl1-1-01-005-19-0
#SBATCH --constraint=nvidia-gpu
#SBATCH --mem=48G
#SBATCH --partition=ice-gpu
#SBATCH --array=0-2

# RFF (predictor_aware, variance_weighted) + CLIP — Product of Experts (log-odds) combination
# Array task 0=CIFAR-10, 1=CIFAR-100, 2=ImageNet-200

mkdir -p logs

module load anaconda3/2023.03
eval "$(conda shell.bash hook)"
conda activate ~/scratch/conda_venvs/openood

if ! nvidia-smi &>/dev/null; then
    echo "ERROR: No NVIDIA GPU on $(hostname). Requeuing job $SLURM_JOB_ID."
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi

cd ~/scratch/openood/src/OpenOOD

DATASETS=("cifar10" "cifar100" "imagenet200")
ROOTS=(
    "./results/cifar10_resnet18_32x32_base_e100_lr0.1_default"
    "./results/cifar100_resnet18_32x32_base_e100_lr0.1_default"
    "./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default"
)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
ROOT=${ROOTS[$SLURM_ARRAY_TASK_ID]}

echo "=== rff_predictor_aware_vw_poe | $DATASET | $(date) ==="

python scripts/eval_ood.py \
    --root "$ROOT" \
    --id-data "$DATASET" \
    --postprocessor rff_predictor_aware_vw_poe \
    --save-csv \
    --save-score

echo "=== Done | $(date) ==="
