"""
Layer Probe Diagnostic — per-layer binary AUROC (ID vs OOD) using logistic regression.

Ported from tkasturi's layer_probe_diagnostic.py with OpenOOD conventions.

Usage:
    python scripts/layer_probe_diagnostic.py \
        --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
        --id-data cifar10 --seed 0

Output:
  - Full per-layer AUROC table (rows = OOD datasets, columns = layers)
  - near/far average rows
  - Raw product weights and normalized weights
  - Ready-to-paste YAML block for yosinski_weights
"""

import os
import sys

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)

import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from openood.networks import ResNet18_32x32, ResNet18_224x224
from openood.evaluation_api import Evaluator
from openood.evaluation_api.postprocessor import get_postprocessor

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Per-layer binary AUROC probe')
parser.add_argument('--root', required=True,
                    help='Results dir, e.g. ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default')
parser.add_argument('--id-data', required=True,
                    choices=['cifar10', 'cifar100', 'imagenet200'],
                    help='ID dataset name')
parser.add_argument('--seed', type=int, default=0,
                    help='Seed index (0, 1, or 2)')
parser.add_argument('--data-root', type=str,
                    default=os.path.join(ROOT_DIR, 'data'),
                    help='Dataset root directory')
parser.add_argument('--config-root', type=str,
                    default=os.path.join(ROOT_DIR, 'configs'),
                    help='Configs root directory')
parser.add_argument('--batch-size', type=int, default=200)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--layers', type=int, nargs='+', default=[1, 2, 3, 4],
                    help='feature_list indices to probe (default: 1 2 3 4)')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LAYER_NAMES = {
    0: 'conv1',
    1: 'layer1',
    2: 'layer2',
    3: 'layer3',
    4: 'penultimate',
}

NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'imagenet200': 200}
MODEL = {
    'cifar10': ResNet18_32x32,
    'cifar100': ResNet18_32x32,
    'imagenet200': ResNet18_224x224,
}

# OOD split membership for CIFAR-10, CIFAR-100, ImageNet-200
NEAR_OOD = {
    'cifar10':     ['cifar100', 'tin'],
    'cifar100':    ['cifar10', 'tin'],
    'imagenet200': ['ssb_hard', 'ninco'],
}
FAR_OOD = {
    'cifar10':     ['mnist', 'svhn', 'texture', 'place365'],
    'cifar100':    ['mnist', 'svhn', 'texture', 'place365'],
    'imagenet200': ['inaturalist', 'textures', 'openimage_o'],
}


def extract_layer_features(net, loader, layer_indices, device):
    """Extract and max-pool spatial layer features from a dataloader.

    Args:
        net: network with return_feature_list=True support
        loader: dataloader
        layer_indices: list of int — indices into feature_list to extract
        device: torch device

    Returns:
        dict: {layer_name: np.ndarray of shape (N, C)}
    """
    accum = {LAYER_NAMES.get(i, f'layer{i}'): [] for i in layer_indices}
    net.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc='Extracting', leave=False):
            data = batch['data'].to(device).float()
            _, feature_list = net(data, return_feature_list=True)
            for i in layer_indices:
                name = LAYER_NAMES.get(i, f'layer{i}')
                f = feature_list[i]
                if f.dim() > 2:                          # spatial [B, C, H, W]
                    f = f.view(f.size(0), f.size(1), -1).max(dim=2).values  # [B, C]
                accum[name].append(f.cpu().numpy())

    return {name: np.concatenate(arrs, axis=0) for name, arrs in accum.items()}


def probe_auroc(id_feats, ood_feats):
    """Fit logistic regression (70/30 split) and return binary AUROC.

    Args:
        id_feats:  np.ndarray (N_id, C)
        ood_feats: np.ndarray (N_ood, C)

    Returns:
        float: AUROC score (higher = layer discriminates ID from OOD better)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    X = np.concatenate([id_feats, ood_feats], axis=0).astype(np.float32)
    y = np.array([1] * len(id_feats) + [0] * len(ood_feats), dtype=np.int32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0,
                              multi_class='ovr', n_jobs=1)
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_te, prob))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = NUM_CLASSES[args.id_data]
    model_arch = MODEL[args.id_data]

    # Load network
    ckpt_path = os.path.join(args.root, f's{args.seed}', 'best.ckpt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    net = model_arch(num_classes=num_classes)
    net.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    net.to(device)
    net.eval()
    print(f'Loaded checkpoint: {ckpt_path}')

    # Use MSP (BasePostprocessor) as a dummy to get dataloaders via Evaluator
    postprocessor = get_postprocessor(args.config_root, 'msp', args.id_data)
    postprocessor.APS_mode = False

    evaluator = Evaluator(
        net,
        id_name=args.id_data,
        data_root=args.data_root,
        config_root=args.config_root,
        preprocessor=None,
        postprocessor_name='msp',
        postprocessor=postprocessor,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    dataloader_dict = evaluator.dataloader_dict

    layer_names = [LAYER_NAMES.get(i, f'layer{i}') for i in args.layers]
    near_ood_sets = NEAR_OOD[args.id_data]
    far_ood_sets  = FAR_OOD[args.id_data]

    # Extract ID test features once
    print('\nExtracting ID test features...')
    id_feats = extract_layer_features(
        net, dataloader_dict['id']['test'], args.layers, device)

    # Collect all available OOD dataset names
    all_ood_names = []
    for split in ['near', 'far']:
        if split in dataloader_dict['ood']:
            all_ood_names.extend(dataloader_dict['ood'][split].keys())

    # Per OOD dataset: extract features + probe each layer
    results = {}   # {ood_name: {layer_name: auroc}}
    for ood_name in all_ood_names:
        split = 'near' if ood_name in dataloader_dict['ood'].get('near', {}) else 'far'
        loader = dataloader_dict['ood'][split][ood_name]
        print(f'\nProbing layers for OOD dataset: {ood_name} ({split})')
        ood_feats = extract_layer_features(net, loader, args.layers, device)
        aurocs = {}
        for name in layer_names:
            auroc = probe_auroc(id_feats[name], ood_feats[name])
            aurocs[name] = auroc
            print(f'  {name}: AUROC={auroc:.4f}')
        results[ood_name] = aurocs

    # ---------------------------------------------------------------------------
    # Build AUROC table
    # ---------------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('Per-Layer AUROC Table')
    print('=' * 70)

    # Header
    col_w = 14
    header = f'{"OOD Dataset":<22}' + ''.join(f'{n:>{col_w}}' for n in layer_names)
    print(header)
    print('-' * len(header))

    def _avg_row(label, names):
        row_vals = {}
        for name in layer_names:
            vals = [results[d][name] for d in names if d in results]
            row_vals[name] = np.mean(vals) if vals else float('nan')
        line = f'{label:<22}' + ''.join(f'{row_vals[n]:>{col_w}.4f}' for n in layer_names)
        print(line)
        return row_vals

    near_in_results = [d for d in near_ood_sets if d in results]
    far_in_results  = [d for d in far_ood_sets  if d in results]

    for ood_name in all_ood_names:
        split_tag = '(near)' if ood_name in near_in_results else '(far)'
        label = f'{ood_name} {split_tag}'
        line = f'{label:<22}' + ''.join(
            f'{results[ood_name][n]:>{col_w}.4f}' for n in layer_names)
        print(line)

    print('-' * len(header))
    near_avgs = _avg_row('near avg', near_in_results)
    far_avgs  = _avg_row('far avg',  far_in_results)

    # ---------------------------------------------------------------------------
    # Compute weights
    # ---------------------------------------------------------------------------
    raw_weights = np.array([
        near_avgs[n] * far_avgs[n] for n in layer_names
    ])
    norm_weights = raw_weights / raw_weights.sum()

    print('\n' + '=' * 70)
    print('Layer Weight Computation')
    print('=' * 70)
    print(f'{"Layer":<20}  {"near_avg":>10}  {"far_avg":>10}  {"raw (product)":>14}  {"normalized":>12}')
    print('-' * 72)
    for i, name in enumerate(layer_names):
        print(f'{name:<20}  {near_avgs[name]:>10.4f}  {far_avgs[name]:>10.4f}  '
              f'{raw_weights[i]:>14.6f}  {norm_weights[i]:>12.6f}')

    # ---------------------------------------------------------------------------
    # YAML suggestion
    # ---------------------------------------------------------------------------
    weights_str = ', '.join(f'{w:.4f}' for w in norm_weights)
    print('\n' + '=' * 70)
    print('Suggested config addition (normalized):')
    print('=' * 70)
    print(f'  yosinski_weights: [{weights_str}]')
    print()
    print('(Add this under postprocessor_args in your variant yml.)')
    print('(Indices correspond to --layers:', args.layers, ')')


if __name__ == '__main__':
    main()
