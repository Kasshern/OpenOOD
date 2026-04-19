#!/bin/bash
# Run McNemar's tests for top RFF variants vs all baselines.
# Top methods chosen from logs/average_results.txt (best Far AUROC per dataset,
# with the best Near AUROC noted separately where it differs).
#
# Prerequisites: baseline score pkls must exist (run_cifar10/100/imagenet200/imagenet1k.sbatch
# must have completed with --save-score).
#
# Usage:
#   bash scripts/run_mcnemar_top.sh
#
# Outputs go to: results/mcnemar/

set -e
cd "$(dirname "$0")/.."

OUTDIR="results/mcnemar"
mkdir -p "$OUTDIR"

echo "============================================================"
echo " McNemar Top-Method Comparison"
echo " Output: $OUTDIR"
echo "============================================================"

# ---------------------------------------------------------------------------
# CIFAR-10
# Best Far AUROC:  rff_centroid_vw_mlpca_minmax (94.23)
# Best Near AUROC: rff_max_vw                   (89.20)
# ---------------------------------------------------------------------------
CIFAR10_ROOT="./results/cifar10_resnet18_32x32_base_e100_lr0.1_default"

echo ""
echo "[CIFAR-10] rff_centroid_vw_mlpca_minmax (best far) vs all baselines..."
python scripts/run_mcnemar.py \
    --root "$CIFAR10_ROOT" \
    --method-a rff_centroid_vw_mlpca_minmax \
    --batch --task ood \
    --out "$OUTDIR/cifar10_centroid_vw_mlpca_minmax_vs_all.json"

echo "[CIFAR-10] rff_max_vw (best near) vs all baselines..."
python scripts/run_mcnemar.py \
    --root "$CIFAR10_ROOT" \
    --method-a rff_max_vw \
    --batch --task ood \
    --out "$OUTDIR/cifar10_max_vw_vs_all.json"

# ---------------------------------------------------------------------------
# CIFAR-100
# Best Far AUROC:  rff_max_vw                      (86.10)
# Best Near AUROC: rff_predictor_aware_vw_allpca   (80.86)
# ---------------------------------------------------------------------------
CIFAR100_ROOT="./results/cifar100_resnet18_32x32_base_e100_lr0.1_default"

echo ""
echo "[CIFAR-100] rff_max_vw (best far) vs all baselines..."
python scripts/run_mcnemar.py \
    --root "$CIFAR100_ROOT" \
    --method-a rff_max_vw \
    --batch --task ood \
    --out "$OUTDIR/cifar100_max_vw_vs_all.json"

echo "[CIFAR-100] rff_predictor_aware_vw_allpca (best near) vs all baselines..."
python scripts/run_mcnemar.py \
    --root "$CIFAR100_ROOT" \
    --method-a rff_predictor_aware_vw_allpca \
    --batch --task ood \
    --out "$OUTDIR/cifar100_predictor_aware_allpca_vs_all.json"

# ---------------------------------------------------------------------------
# ImageNet-200
# Best Far AUROC:  rff_max_novw             (93.57)
# Best Near AUROC: rff_predictor_aware_vw_allpca (83.01)
# ---------------------------------------------------------------------------
IMAGENET200_ROOT="./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default"

echo ""
echo "[ImageNet-200] rff_max_novw (best far) vs all baselines..."
python scripts/run_mcnemar.py \
    --root "$IMAGENET200_ROOT" \
    --method-a rff_max_novw \
    --batch --task ood \
    --out "$OUTDIR/imagenet200_max_novw_vs_all.json"

echo "[ImageNet-200] rff_predictor_aware_vw_allpca (best near) vs all baselines..."
python scripts/run_mcnemar.py \
    --root "$IMAGENET200_ROOT" \
    --method-a rff_predictor_aware_vw_allpca \
    --batch --task ood \
    --out "$OUTDIR/imagenet200_predictor_aware_allpca_vs_all.json"

# ---------------------------------------------------------------------------
# Pretty-print summaries
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Summaries"
echo "============================================================"

for f in "$OUTDIR"/*.json; do
    echo ""
    echo "--- $(basename $f) ---"
    python scripts/summarize_mcnemar.py "$f"
done

echo ""
echo "Done. Results in $OUTDIR/"
echo "For LaTeX tables: python scripts/summarize_mcnemar.py <file.json> --format latex"
