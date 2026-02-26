#!/bin/bash
# RFF grid search — score_mode=max, variance_weighted=False
# Runs APS (36 combos: 4σ × 3D × 3α) on each dataset and picks the best.

set -e
mkdir -p logs

VARIANT="rff_max_novw"
DATASETS=(
    "cifar10   ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default"
    "cifar100  ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default"
    "imagenet200 ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default"
)

echo "=== RFF grid search: $VARIANT | $(date) ==="

for entry in "${DATASETS[@]}"; do
    DATASET=$(echo $entry | awk '{print $1}')
    ROOT=$(echo $entry | awk '{print $2}')
    LOG="logs/${VARIANT}_${DATASET}.txt"

    echo ""
    echo ">>> $DATASET | root=$ROOT"
    python scripts/eval_ood.py \
        --root "$ROOT" \
        --id-data "$DATASET" \
        --postprocessor "$VARIANT" \
        --save-csv \
        --save-score \
        2>&1 | tee "$LOG"
    echo "    Log: $LOG"
done

echo ""
echo "=== Done: $VARIANT | $(date) ==="
