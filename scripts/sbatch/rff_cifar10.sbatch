#!/bin/bash
#SBATCH --job-name=rff_c10
#SBATCH --output=logs/rff_cifar10_%A_%a.out
#SBATCH --error=logs/rff_cifar10_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --partition=inferno
#SBATCH --array=0-3

set -e
mkdir -p logs

module load anaconda3/2023.03
conda activate ~/scratch/conda_venvs/openood

cd ~/scratch/openood/src/OpenOOD

VARIANTS=("rff_max_vw" "rff_max_novw" "rff_margin_vw" "rff_margin_novw")
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
CHECKPOINT_ROOT="./results/cifar10_resnet18_32x32_base_e100_lr0.1_default"

echo "=== RFF CIFAR-10 | variant=$VARIANT | $(date) ==="

python scripts/eval_ood.py \
    --root $CHECKPOINT_ROOT \
    --id-data cifar10 \
    --postprocessor $VARIANT \
    --save-csv \
    --save-score

echo "=== Done: $VARIANT | $(date) ==="
