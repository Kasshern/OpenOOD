#!/bin/bash
#SBATCH --job-name=rff_margin_novw
#SBATCH --output=logs/rff_margin_novw_%A_%a.out
#SBATCH --error=logs/rff_margin_novw_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --partition=inferno
#SBATCH --array=0-2

# RFF grid search — score_mode=margin, variance_weighted=False
# Array task 0=CIFAR-10, 1=CIFAR-100, 2=ImageNet-200

mkdir -p logs

module load anaconda3/2023.03
conda activate ~/scratch/conda_venvs/openood

cd ~/scratch/openood/src/OpenOOD

DATASETS=("cifar10" "cifar100" "imagenet200")
ROOTS=(
    "./results/cifar10_resnet18_32x32_base_e100_lr0.1_default"
    "./results/cifar100_resnet18_32x32_base_e100_lr0.1_default"
    "./results/imagenet200_res18_v1.5"
)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
ROOT=${ROOTS[$SLURM_ARRAY_TASK_ID]}

echo "=== rff_margin_novw | $DATASET | $(date) ==="

python scripts/eval_ood.py \
    --root "$ROOT" \
    --id-data "$DATASET" \
    --postprocessor rff_margin_novw \
    --save-csv \
    --save-score

echo "=== Done | $(date) ==="
