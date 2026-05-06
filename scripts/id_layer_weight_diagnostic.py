"""
ID-only layer-weight diagnostic for benchmark-fair RFF/Nyström selection.

This script computes per-layer Fisher class-separability weights using only
ID train or ID validation data. It does not read OOD loaders and can be used
to produce benchmark-comparable layer weights.

Usage:
    python scripts/id_layer_weight_diagnostic.py \
        --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
        --id-data cifar10 --seed 0
"""

import argparse
import os
import sys

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)

import torch
from tqdm import tqdm

from openood.evaluation_api import Evaluator
from openood.evaluation_api.postprocessor import get_postprocessor
from openood.networks import ResNet18_32x32, ResNet18_224x224
from openood.postprocessors.feature_selection import (
    compute_id_layer_weights,
    format_layer_scores,
    format_layer_weights,
    select_topk_layers,
)


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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute ID-only layer weights for multilayer RFF/Nyström.')
    parser.add_argument('--root', required=True)
    parser.add_argument('--id-data', required=True,
                        choices=['cifar10', 'cifar100', 'imagenet200'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-root', type=str,
                        default=os.path.join(ROOT_DIR, 'data'))
    parser.add_argument('--config-root', type=str,
                        default=os.path.join(ROOT_DIR, 'configs'))
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--layers', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--source', choices=['train', 'val'], default='train')
    parser.add_argument('--pool-mode', choices=['avg', 'minmax'], default='avg')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Optionally print ID-only top-k selected layers')
    return parser.parse_args()


def extract_layer_features(net, loader, layer_indices, pool_mode, device):
    accum = [[] for _ in layer_indices]
    labels = []
    net.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc='Extracting ID features'):
            data = batch['data'].to(device).float()
            _, feature_list = net(data, return_feature_list=True)
            for j, layer_idx in enumerate(layer_indices):
                features = feature_list[layer_idx]
                if features.dim() > 2:
                    flat = features.view(features.size(0), features.size(1), -1)
                    if pool_mode == 'minmax':
                        features = torch.cat([
                            flat.min(dim=2).values,
                            flat.max(dim=2).values,
                        ], dim=1)
                    else:
                        features = features.mean(dim=[2, 3])
                accum[j].append(features.cpu())
            labels.append(batch['label'].cpu())
    return [torch.cat(items, dim=0) for items in accum], torch.cat(labels, dim=0)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = os.path.join(args.root, f's{args.seed}', 'best.ckpt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    net = MODEL[args.id_data](num_classes=NUM_CLASSES[args.id_data])
    net.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    net.to(device)
    net.eval()
    print(f'Loaded checkpoint: {ckpt_path}')

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

    loader = evaluator.dataloader_dict['id'][args.source]
    layer_features, labels = extract_layer_features(
        net, loader, args.layers, args.pool_mode, device)
    weights = compute_id_layer_weights(layer_features, labels, metric='id_fisher')

    print('\n' + '=' * 70)
    print(f'ID-only Fisher layer weights ({args.id_data}, source={args.source})')
    print('=' * 70)
    for layer_idx, weight in zip(args.layers, weights.tolist()):
        name = LAYER_NAMES.get(layer_idx, f'layer{layer_idx}')
        print(f'{name:<20} {weight:.6f}')

    weights_str = ', '.join(f'{weight:.4f}' for weight in weights.tolist())
    print('\nSuggested config block:')
    print('  layer_weighting: id_fisher')
    print(f'  layer_weight_source: {args.source}')
    print('  layer_weight_normalize: sum')
    print(f'  # Equivalent static weights for layers {args.layers}:')
    print(f'  yosinski_weights: [{weights_str}]')
    print('\nLog label:')
    print('  ' + format_layer_weights(args.layers, weights))

    if args.top_k is not None:
        selected_layers, _, scores = select_topk_layers(
            args.layers,
            layer_features,
            labels,
            k=args.top_k,
            metric='id_fisher_topk',
        )
        print('\nTop-k selection:')
        print('  scores: ' + format_layer_scores(args.layers, scores))
        print('  selected: [' + ', '.join(str(layer) for layer in selected_layers) + ']')
        print('  config:')
        print('    layer_selection: id_fisher_topk')
        print(f'    layer_selection_k: {args.top_k}')
        print(f'    layer_selection_source: {args.source}')


if __name__ == '__main__':
    main()
