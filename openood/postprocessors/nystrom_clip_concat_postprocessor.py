"""
NystromCLIPConcatPostprocessor: feature-level fusion of ResNet + CLIP for
Nyström OOD detection.

Algorithm 2 (Nyström anchors) on ResNet-penultimate + CLIP-ViT-B/16-image
concat features. Parallel to RFFCLIPConcatPostprocessor but uses Nyström
kernel approximation instead of RFF projection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from openood.preprocessors.transform import normalization_dict
from openood.networks.clip import CLIPZeroshot

from .nystrom_postprocessor import NystromOODPostprocessor
from .clip_prior_postprocessor import (
    _CLIP_MEAN, _CLIP_STD, IMAGENET_TEMPLATES,
    CIFAR10_CLASSES, CIFAR100_CLASSES, IMAGENET_CLASSES,
    _resolve_imagenet200_classnames,
)


class NystromCLIPConcatPostprocessor(NystromOODPostprocessor):
    """Nyström kernel OOD on ResNet-penultimate + CLIP image embedding concat."""

    def __init__(self, config):
        super().__init__(config)
        self.clip_backbone = str(getattr(self.args, 'clip_backbone', None) or 'ViT-B/16')
        self.data_root     = str(getattr(self.args, 'data_root',     None) or './data')
        self.dataset_name  = self.config.dataset.name

        self.clip_model    = None
        self._renorm_scale = None
        self._renorm_shift = None
        self.feature_space = 'resnet_clip_concat'

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return

        print('\n' + '=' * 50)
        print('Setting up NystromCLIPConcat OOD detector (ResNet + CLIP feature concat)...')
        print(f'  sigma={self.sigma}  n_anchors={self.n_anchors}  alpha={self.alpha}')
        print(f'  clip_backbone={self.clip_backbone}  score_mode={self.score_mode}')
        print(f'  variance_weighted={self.variance_weighted}')
        print('=' * 50)

        net.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Load CLIP first (needed by _extract_concat_features)
        if self.clip_model is None:
            classnames = self._resolve_classnames()
            print(f'[CLIPConcat] Loading CLIP {self.clip_backbone} '
                  f'with {len(classnames)} classes ({self.dataset_name})...')
            self.clip_model = CLIPZeroshot(
                classnames=classnames,
                templates=IMAGENET_TEMPLATES,
                backbone=self.clip_backbone,
            )
            self.clip_model.cuda().eval()
            self._setup_renorm(device)

        # 2. Extract training features (ResNet + CLIP concat)
        train_features, train_labels = [], []
        with torch.no_grad():
            for batch in tqdm(id_loader_dict['train'],
                              desc='Extracting train features (nystrom resnet+clip)'):
                data   = batch['data'].to(device).float()
                labels = batch['label'].to(device)
                feats  = self._extract_concat_features(net, data)
                train_features.append(feats.cpu())
                train_labels.append(labels.cpu())

        self.X_train_raw = torch.cat(train_features, dim=0).to(device)
        self.y_train     = torch.cat(train_labels,   dim=0).to(device)
        self.feature_dim = self.X_train_raw.shape[1]
        self.num_classes = int(self.y_train.max().item()) + 1
        print(f'Train: {self.X_train_raw.shape[0]} samples, feature_dim={self.feature_dim}')

        # 3. Extract val features (also softmax for softmax/predictor_aware)
        val_features, val_labels = [], []
        _need_softmax = self.score_mode in ('softmax', 'predictor_aware')
        val_softmax   = [] if _need_softmax else None

        with torch.no_grad():
            for batch in tqdm(id_loader_dict['val'],
                              desc='Extracting val features (nystrom resnet+clip)'):
                data   = batch['data'].to(device).float()
                labels = batch['label'].to(device)
                if _need_softmax:
                    output, _ = net(data, return_feature=True)
                    val_softmax.append(torch.softmax(output, dim=1).cpu())
                feats = self._extract_concat_features(net, data)
                val_features.append(feats.cpu())
                val_labels.append(labels.cpu())

        self.X_val_raw = torch.cat(val_features, dim=0).to(device)
        self.y_val     = torch.cat(val_labels,   dim=0).to(device)
        if val_softmax is not None:
            self.softmax_val = torch.cat(val_softmax, dim=0)
        print(f'Val:   {self.X_val_raw.shape[0]} samples')

        # 4. Fit Nyström anchors + calibrate threshold
        self._fit_nystrom(device)
        self._compute_nystrom_threshold(device)

        self.setup_flag = True
        print('NystromCLIPConcat setup complete.\n')

    # ── Postprocess ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data):
        device = data.device

        # 1. ResNet predictions
        output, _ = net(data, return_feature=True)
        _, pred   = torch.max(output, dim=1)

        # 2. Concat features (per-source L2 norm applied in _extract_concat_features)
        features = self._extract_concat_features(net, data)
        if self.normalize:
            features = F.normalize(features.float(), p=2, dim=1)

        # 3. Nyström kernel scores
        class_scores = self._nystrom_scores(features, device)  # [B, C]
        B = features.shape[0]

        # 4. Score-mode dispatch (mirrors NystromOODPostprocessor.postprocess)
        if self.use_entropy:
            pos   = torch.relu(class_scores)
            denom = pos.sum(dim=1, keepdim=True).clamp(min=1e-10)
            w     = pos / denom
            conf  = -(w * torch.log(w + 1e-10)).sum(dim=1)
        elif self.score_mode == 'softmax':
            sm   = torch.softmax(output, dim=1)
            conf = (class_scores * sm).sum(dim=1)
        elif self.score_mode == 'predictor_aware':
            sm       = torch.softmax(output, dim=1)
            entropy  = -(sm * torch.log(sm + 1e-8)).sum(dim=1)
            pred_sim = class_scores[torch.arange(B, device=device), pred]
            conf     = pred_sim - entropy
        else:  # 'max' (default)
            conf = class_scores.max(dim=1).values

        return pred, conf

    # ── Pickle: exclude CLIP model from saved state ──────────────────────────

    def __getstate__(self):
        state = self.__dict__.copy()
        state['clip_model']    = None
        state['_renorm_scale'] = None
        state['_renorm_shift'] = None
        return state

    # ── Helpers ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _extract_concat_features(self, net: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """ResNet penultimate + CLIP image embedding, both L2-normed and concat."""
        _, resnet_feats = net(data, return_feature=True)
        resnet_feats    = F.normalize(resnet_feats.float(), p=2, dim=1)

        clip_data = data.float() * self._renorm_scale + self._renorm_shift
        if clip_data.shape[-1] != 224:
            clip_data = F.interpolate(
                clip_data, size=224, mode='bicubic', align_corners=False)
        clip_feats = self.clip_model.encode_image(clip_data)

        return torch.cat([resnet_feats, clip_feats], dim=1)

    def _setup_renorm(self, device: torch.device):
        """Precompute per-channel renorm: source dataset stats → CLIP stats."""
        src_mean_v, src_std_v = normalization_dict[self.dataset_name]
        src_mean = torch.tensor(src_mean_v, dtype=torch.float32).view(1,3,1,1).to(device)
        src_std  = torch.tensor(src_std_v,  dtype=torch.float32).view(1,3,1,1).to(device)
        cl_mean  = torch.tensor(_CLIP_MEAN,  dtype=torch.float32).view(1,3,1,1).to(device)
        cl_std   = torch.tensor(_CLIP_STD,   dtype=torch.float32).view(1,3,1,1).to(device)
        self._renorm_scale = src_std / cl_std
        self._renorm_shift = (src_mean - cl_mean) / cl_std

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
                f'[NystromCLIPConcat] Unsupported dataset "{name}". '
                f'Supported: cifar10, cifar100, imagenet200, imagenet.')
