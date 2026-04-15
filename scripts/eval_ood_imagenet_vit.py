import os, sys, time
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
import numpy as np
import pandas as pd
import argparse
import hashlib
import pickle

import torch
from torch.hub import load_state_dict_from_url
from torchvision.models import ViT_B_16_Weights

from openood.evaluation_api import Evaluator
from openood.networks import ViT_B_16


def _get_postprocessor_config_hash(config_root, postprocessor_name):
    """Hash the postprocessor yml so config changes bust the pkl cache."""
    config_path = os.path.join(config_root, 'postprocessors',
                               f'{postprocessor_name}.yml')
    if os.path.exists(config_path):
        with open(config_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    return 'default'


parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='vit-b-16', choices=['vit-b-16'])
parser.add_argument('--tvs-version', default=1, type=int, choices=[1, 2])
parser.add_argument('--postprocessor', default='msp')
parser.add_argument('--batch-size', type=int, default=200)
parser.add_argument('--save-csv', action='store_true')
parser.add_argument('--save-score', action='store_true')
parser.add_argument('--fsood', action='store_true')
args = parser.parse_args()

postprocessor_name = args.postprocessor
root = os.path.join(
    ROOT_DIR, 'results',
    f'imagenet_{args.arch}_tvsv{args.tvs_version}_base_default')
if not os.path.exists(root):
    os.makedirs(root)

_cfg_hash = _get_postprocessor_config_hash(
    os.path.join(ROOT_DIR, 'configs'), postprocessor_name)
pp_pkl_name = f'{postprocessor_name}_{_cfg_hash}.pkl'
scores_pkl_name = f'{postprocessor_name}_{_cfg_hash}_scores.pkl'

# Load pre-setup postprocessor if exists
if os.path.isfile(os.path.join(root, 'postprocessors', pp_pkl_name)):
    with open(os.path.join(root, 'postprocessors', pp_pkl_name), 'rb') as f:
        postprocessor = pickle.load(f)
else:
    postprocessor = None

# Load ViT-B/16 with torchvision pretrained weights (IMAGENET1K_V1 or V2)
net = ViT_B_16()
weights = eval(f'ViT_B_16_Weights.IMAGENET1K_V{args.tvs_version}')
net.load_state_dict(load_state_dict_from_url(weights.url))
preprocessor = weights.transforms()

net.cuda()
net.eval()

t_setup_start = time.time()
evaluator = Evaluator(
    net,
    id_name='imagenet',
    data_root=os.path.join(ROOT_DIR, 'data'),
    config_root=os.path.join(ROOT_DIR, 'configs'),
    preprocessor=preprocessor,
    postprocessor_name=postprocessor_name,
    postprocessor=postprocessor,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=8)
t_setup = time.time() - t_setup_start
print(f'[Timing] Setup: {t_setup:.1f}s')

# Load pre-computed scores if exist
if os.path.isfile(os.path.join(root, 'scores', scores_pkl_name)):
    with open(os.path.join(root, 'scores', scores_pkl_name), 'rb') as f:
        scores = pickle.load(f)
    # merge scores into evaluator
    import collections.abc

    def _update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = _update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    _update(evaluator.scores, scores)
    print('Loaded pre-computed scores from file.')

# Save the postprocessor for future reuse
if hasattr(evaluator.postprocessor, 'setup_flag') or \
        evaluator.postprocessor.hyperparam_search_done is True:
    pp_save_root = os.path.join(root, 'postprocessors')
    os.makedirs(pp_save_root, exist_ok=True)
    if not os.path.isfile(os.path.join(pp_save_root, pp_pkl_name)):
        with open(os.path.join(pp_save_root, pp_pkl_name), 'wb') as f:
            pickle.dump(evaluator.postprocessor, f, pickle.HIGHEST_PROTOCOL)

t_eval_start = time.time()
metrics = evaluator.eval_ood(fsood=args.fsood)
t_eval = time.time() - t_eval_start
print(f'[Timing] Eval:  {t_eval:.1f}s')
print(f'[Timing] Total: {t_setup + t_eval:.1f}s')

# Save computed scores
if args.save_score:
    score_save_root = os.path.join(root, 'scores')
    os.makedirs(score_save_root, exist_ok=True)
    with open(os.path.join(root, 'scores', scores_pkl_name), 'wb') as f:
        pickle.dump(evaluator.scores, f, pickle.HIGHEST_PROTOCOL)

# Save CSV
if args.save_csv:
    saving_root = os.path.join(root, 'ood' if not args.fsood else 'fsood')
    os.makedirs(saving_root, exist_ok=True)
    metrics.to_csv(os.path.join(saving_root, f'{postprocessor_name}.csv'),
                   float_format='{:.2f}'.format)
else:
    print(metrics)
