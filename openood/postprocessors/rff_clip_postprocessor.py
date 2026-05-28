"""
RFFCLIPPostprocessor: CLIP infrastructure base class for feature-level fusion variants.

This class provides shared CLIP loading, image renormalization, and pickle handling
for subclasses that concatenate CLIP features with ResNet features before the kernel
method (RFFCLIPConcatPostprocessor, RFFCLIPMlMinmaxPostprocessor).

It does NOT implement OOD scoring itself — subclasses must define postprocess().
"""

import torch
import torch.nn as nn

from openood.preprocessors.transform import normalization_dict
from openood.networks.clip import CLIPZeroshot

from .rff_postprocessor import RFFPostprocessor
from .clip_prior_postprocessor import (
    _CLIP_MEAN, _CLIP_STD, IMAGENET_TEMPLATES,
    CIFAR10_CLASSES, CIFAR100_CLASSES, IMAGENET_CLASSES,
    _resolve_imagenet200_classnames,
)


class RFFCLIPPostprocessor(RFFPostprocessor):
    """CLIP infrastructure base for RFF+CLIP feature-concat postprocessors.

    Subclasses (RFFCLIPConcatPostprocessor, RFFCLIPMlMinmaxPostprocessor) must
    implement postprocess(). This class only provides CLIP loading and renorm.
    """

    def __init__(self, config):
        super().__init__(config)
        self.clip_backbone = str(getattr(self.args, 'clip_backbone', None) or 'ViT-B/16')
        self.data_root     = str(getattr(self.args, 'data_root',     None) or './data')
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
