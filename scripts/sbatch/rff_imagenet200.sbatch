#!/bin/bash
#SBATCH --job-name=rff_in200
#SBATCH --output=logs/rff_imagenet200_%A_%a.out
#SBATCH --error=logs/rff_imagenet200_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --partition=inferno
#SBATCH --array=0-3

set -e
mkdir -p logs

module load anaconda3/2023.03
conda activate ~/scratch/conda_venvs/openood

cd ~/scratch/openood/src/OpenOOD

VARIANTS=("rff_max_vw" "rff_max_novw" "rff_margin_vw" "rff_margin_novw")
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
CHECKPOINT_ROOT="./results/imagenet200_res18_v1.5"

echo "=== RFF ImageNet-200 | variant=$VARIANT | $(date) ==="

python scripts/eval_ood.py \
    --root $CHECKPOINT_ROOT \
    --id-data imagenet200 \
    --postprocessor $VARIANT \
    --save-csv \
    --save-score

echo "=== Done: $VARIANT | $(date) ==="
