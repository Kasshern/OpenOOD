#!/bin/bash
#SBATCH --job-name=rff_margin_vw_vit_b16
#SBATCH --output=logs/rff_margin_vw_vit_b16_%j.out
#SBATCH --error=logs/rff_margin_vw_vit_b16_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=atl1-1-01-005-17-0,atl1-1-01-005-19-0
#SBATCH --constraint=nvidia-gpu
#SBATCH --mem=32G
#SBATCH --partition=ice-gpu

# RFF on ViT-B/16 (IMAGENET1K_V1 torchvision pretrained) — ImageNet-1K only

mkdir -p logs

module load anaconda3/2023.03
eval "$(conda shell.bash hook)"
conda activate ~/scratch/conda_venvs/openood

# Abort if assigned a non-NVIDIA (e.g. AMD) node
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: No NVIDIA GPU on $(hostname). Requeuing job $SLURM_JOB_ID."
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi

cd ~/scratch/openood/src/OpenOOD

echo "=== rff_margin_vw | imagenet | vit-b-16 | $(date) ==="

python scripts/eval_ood_imagenet_vit.py \
    --arch vit-b-16 \
    --tvs-version 1 \
    --postprocessor rff_margin_vw \
    --save-csv \
    --save-score

echo "=== Done | $(date) ==="
