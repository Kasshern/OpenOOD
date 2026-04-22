"""
RFFCLIPPostprocessor: combines RFF OOD score with CLIP zero-shot prior.

Score-level combination (both are scalar confidence scores, higher = more ID):
    final_score = (1 - clip_weight) * rff_score + clip_weight * clip_score

clip_score  = max(softmax(CLIP_logits * logit_scale, dim=1))
rff_score   = standard RFF score (any score_mode: max, centroid, softmax, etc.)

rff_score is min-max normalized per batch to [0,1] before mixing, since RFF
scores are not naturally bounded while CLIP softmax is in (0,1).

APS sweeps clip_weight in [0.0 ... 1.0]. All RFF hyperparams (sigma, D, alpha)
are fixed at best-known values in the config — no re-sweep needed.

clip_weight=0.0 → pure RFF (equivalent to the base RFF variant)
clip_weight=1.0 → pure CLIP zero-shot
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from openood.preprocessors.transform import normalization_dict
from openood.networks.clip import CLIPZeroshot

from .rff_postprocessor import RFFPostprocessor
from .clip_prior_postprocessor import (
    _CLIP_MEAN, _CLIP_STD, IMAGENET_TEMPLATES,
    CIFAR10_CLASSES, CIFAR100_CLASSES, IMAGENET_CLASSES,
    _resolve_imagenet200_classnames,
)


class RFFCLIPPostprocessor(RFFPostprocessor):
    """RFF OOD scorer augmented with a CLIP zero-shot prior."""

    def __init__(self, config):
        super().__init__(config)
        self.clip_weight   = float(getattr(self.args, 'clip_weight',  None) or 0.3)
        self.clip_backbone = str(getattr(self.args,  'clip_backbone', None) or 'ViT-B/16')
        self.data_root     = str(getattr(self.args,  'data_root',     None) or './data')
        self.dataset_name  = self.config.dataset.name

        self.clip_model    = None
        self._renorm_scale = None
        self._renorm_shift = None

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        # Full RFF setup: feature extraction + embedding computation + threshold
        super().setup(net, id_loader_dict, ood_loader_dict)

        # CLIP initialization — skip if already done (e.g. after pkl load)
        if self.clip_model is None:
            classnames = self._resolve_classnames()
            print(f'[RFFClip] Loading CLIP {self.clip_backbone} '
                  f'with {len(classnames)} classes ({self.dataset_name})...')
            self.clip_model = CLIPZeroshot(
                classnames=classnames,
                templates=IMAGENET_TEMPLATES,
                backbone=self.clip_backbone,
            )
            self.clip_model.cuda().eval()

            # Precompute per-channel renorm: source dataset stats → CLIP stats
            src_mean_v, src_std_v = normalization_dict[self.dataset_name]
            src_mean = torch.tensor(src_mean_v, dtype=torch.float32).view(1,3,1,1).cuda()
            src_std  = torch.tensor(src_std_v,  dtype=torch.float32).view(1,3,1,1).cuda()
            cl_mean  = torch.tensor(_CLIP_MEAN,  dtype=torch.float32).view(1,3,1,1).cuda()
            cl_std   = torch.tensor(_CLIP_STD,   dtype=torch.float32).view(1,3,1,1).cuda()
            self._renorm_scale = src_std / cl_std
            self._renorm_shift = (src_mean - cl_mean) / cl_std

    # ── Postprocess ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        # 1. RFF score — pred is argmax from RFF class scores
        rff_pred, rff_conf = super().postprocess(net, data)

        # 2. CLIP zero-shot score
        clip_data = data.float() * self._renorm_scale + self._renorm_shift
        if clip_data.shape[-1] != 224:
            clip_data = F.interpolate(
                clip_data, size=224, mode='bicubic', align_corners=False)
        clip_logits = self.clip_model(clip_data)
        logit_scale = self.clip_model.model.logit_scale.exp().item()
        clip_conf   = torch.softmax(clip_logits * logit_scale, dim=1).max(dim=1).values

        # 3. Min-max normalize rff_conf to [0, 1] for stable mixing
        #    (CLIP softmax is naturally in (0,1); RFF scale varies by score_mode)
        rff_min = rff_conf.min()
        rff_max = rff_conf.max()
        if rff_max > rff_min:
            rff_conf_norm = (rff_conf - rff_min) / (rff_max - rff_min)
        else:
            rff_conf_norm = rff_conf

        # 4. Weighted combination
        conf = (1.0 - self.clip_weight) * rff_conf_norm + self.clip_weight * clip_conf
        return rff_pred, conf

    # ── APS interface — only clip_weight is swept; RFF params are fixed ───────

    def set_hyperparam(self, hyperparam: list):
        self.clip_weight = float(hyperparam[0])

    def get_hyperparam(self):
        return [self.clip_weight]

    # ── Pickle: exclude CLIP model (~340MB) from saved state ─────────────────

    def __getstate__(self):
        state = self.__dict__.copy()
        state['clip_model']    = None
        state['_renorm_scale'] = None
        state['_renorm_shift'] = None
        return state

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_classnames(self) -> list:
        name = self.dataset_name
        if name == 'cifar10':
            return CIFAR10_CLASSES
        elif name == 'cifar100':
            return CIFAR100_CLASSES
        elif name == 'imagenet':
            return IMAGENET_CLASSES
        elif name == 'imagenet200':
            imglist_path = (f'{self.data_root}/benchmark_imglist/'
                            f'imagenet200/train_imagenet200.txt')
            return _resolve_imagenet200_classnames(imglist_path, self.data_root)
        else:
            raise ValueError(
                f'[RFFClip] Unsupported dataset "{name}". '
                f'Supported: cifar10, cifar100, imagenet200, imagenet.')
