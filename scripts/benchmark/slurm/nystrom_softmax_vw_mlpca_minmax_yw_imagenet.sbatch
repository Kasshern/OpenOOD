#!/bin/bash
#SBATCH --job-name=nystrom_softmax_vw_mlpca_minmax_yw_imagenet
#SBATCH --output=logs/nystrom_softmax_vw_mlpca_minmax_yw_imagenet_%j.out
#SBATCH --error=logs/nystrom_softmax_vw_mlpca_minmax_yw_imagenet_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=atl1-1-01-005-17-0,atl1-1-01-005-19-0
#SBATCH --constraint=nvidia-gpu
#SBATCH --mem=64G
#SBATCH --partition=ice-gpu

# Nyström softmax with multilayer PCA features + minmax pooling + Yosinski 3-layer weights (MRL) — ImageNet-1K only

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

echo "=== nystrom_softmax_vw_mlpca_minmax_yw | imagenet | $(date) ==="

python scripts/eval_ood.py \
    --root "./results/imagenet_resnet50_pretrained_v1" \
    --id-data imagenet \
    --postprocessor nystrom_softmax_vw_mlpca_minmax_yw \
    --save-csv \
    --save-score

echo "=== Done | $(date) ==="
