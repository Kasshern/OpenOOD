#!/bin/bash
#SBATCH --job-name=rff_max_mlmx_imgnet
#SBATCH --output=logs/rff_max_vw_mlminmax_imagenet_%j.out
#SBATCH --error=logs/rff_max_vw_mlminmax_imagenet_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=atl1-1-01-005-17-0,atl1-1-01-005-19-0
#SBATCH --constraint=nvidia-gpu
#SBATCH --mem=128G
#SBATCH --partition=ice-gpu

# RFF (max, variance_weighted, multilayer minmax concat 1920-d) — ImageNet-1K only

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

echo "=== rff_max_vw_mlminmax | imagenet | $(date) ==="

python scripts/eval_ood.py \
    --root "./results/imagenet_resnet50_pretrained_v1" \
    --id-data imagenet \
    --postprocessor rff_max_vw_mlminmax \
    --save-csv \
    --save-score

echo "=== Done | $(date) ==="
