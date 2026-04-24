"""
RFFCLIPConcatPostprocessor: feature-level fusion of ResNet + CLIP for RFF OOD detection.

Unlike score-level fusion (RFFCLIPPostprocessor / RFFPOEPostprocessor), this method
combines representations *before* the kernel method runs:

  1. Extract ResNet18 penultimate features     [B, 512]  — L2 normalized
  2. Extract CLIP ViT-B/16 image embeddings   [B, 512]  — L2 normalized
  3. Concatenate                               [B, 1024]
  4. Run Algorithm 1 (RFF kernel attention) on the 1024-d combined space

CLIP's pretrained semantic knowledge enters the kernel mean embedding μ̂ directly,
rather than as a post-hoc score correction. This is theoretically cleaner and stays
fully within Algorithm 1's framework — only the input feature space changes.

No APS sweep: σ/D/α are fixed at best-known values; no clip_weight to tune.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from openood.preprocessors.transform import normalization_dict
from openood.networks.clip import CLIPZeroshot

from .rff_postprocessor import RFFPostprocessor
from .rff_clip_postprocessor import RFFCLIPPostprocessor
from .clip_prior_postprocessor import (
    _CLIP_MEAN, _CLIP_STD, IMAGENET_TEMPLATES,
    _resolve_imagenet200_classnames,
    CIFAR10_CLASSES, CIFAR100_CLASSES, IMAGENET_CLASSES,
)


class RFFCLIPConcatPostprocessor(RFFCLIPPostprocessor):
    """RFF kernel OOD detector on ResNet-penultimate + CLIP image embedding concat."""

    def __init__(self, config):
        super().__init__(config)
        # feature_space is ignored for this class — always uses resnet+clip concat
        self.feature_space = 'resnet_clip_concat'

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return

        print('\n' + '=' * 50)
        print('Setting up RFFCLIPConcat OOD detector (ResNet + CLIP feature concat)...')
        print(f'  sigma={self.sigma}  D={self.D}  alpha={self.alpha}')
        print(f'  clip_backbone={self.clip_backbone}  score_mode={self.score_mode}')
        print('=' * 50)

        net.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Load CLIP FIRST — must happen before the feature extraction loops
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
                              desc='Extracting train features (resnet+clip)'):
                data   = batch['data'].to(device).float()
                labels = batch['label'].to(device)
                feats  = self._extract_concat_features(net, data)
                train_features.append(feats.cpu())
                train_labels.append(labels.cpu())

        self.X_train_raw = torch.cat(train_features, dim=0).to(device)
        self.y_train     = torch.cat(train_labels,   dim=0).to(device)
        self.feature_dim = self.X_train_raw.shape[1]   # 1024
        self.num_classes = int(self.y_train.max().item()) + 1
        print(f'Train: {self.X_train_raw.shape[0]} samples, feature_dim={self.feature_dim}')

        # 3. Extract val features (also capture softmax for softmax/predictor_aware)
        val_features, val_labels = [], []
        _need_softmax = self.score_mode in ('softmax', 'predictor_aware')
        val_softmax   = [] if _need_softmax else None

        with torch.no_grad():
            for batch in tqdm(id_loader_dict['val'],
                              desc='Extracting val features (resnet+clip)'):
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

        # 4. Shared RFF fitting (whiten + compute μ̂ + threshold) — unchanged
        self._compute_whiten_matrix(device)
        self._compute_rff_embedding(device)

        print(f'Per-class embedding norms (mean): '
              f'{torch.linalg.norm(self.mu_hat, dim=1).mean():.4f}')
        print(f'Threshold at {self.alpha * 100:.1f}%: {self.threshold:.4f}')
        self.setup_flag = True
        print('RFFCLIPConcat setup complete.\n')

    # ── Postprocess ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data):
        # 1. Predictions (from ResNet logits)
        output, _ = net(data, return_feature=True)
        _, pred   = torch.max(output, dim=1)

        # 2. Extract concat features and compute RFF
        features = self._extract_concat_features(net, data)
        # Per-source L2 norm already applied in _extract_concat_features;
        # skip global normalize here. Apply normalize_phi if configured.
        phi = self._phi(features)
        if self.normalize_phi:
            phi = F.normalize(phi, p=2, dim=-1)

        # 3. Score against per-class mean embeddings
        mu_hat      = self.mu_hat.to(features.device)           # [C, D]
        class_scores = phi @ mu_hat.T                           # [B, C]
        if self.variance_weighted:
            class_scores = class_scores / torch.sqrt(
                self.var_hat.to(features.device))

        # 4. score_mode dispatch
        if self.score_mode == 'softmax':
            sm   = torch.softmax(output, dim=1)
            conf = (class_scores * sm).sum(dim=1)
        elif self.score_mode == 'predictor_aware':
            sm   = torch.softmax(output, dim=1)
            conf = (class_scores * sm).sum(dim=1) + class_scores.max(dim=1).values
        else:  # max (default)
            conf = class_scores.max(dim=1).values

        return pred, conf

    # ── APS: no hyperparams to sweep ─────────────────────────────────────────

    def set_hyperparam(self, hyperparam: list):
        pass  # nothing to sweep

    def get_hyperparam(self):
        return []

    # ── Helpers ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _extract_concat_features(self, net: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """Extract and return L2-normalized ResNet + CLIP features concatenated."""
        _, resnet_feats = net(data, return_feature=True)
        resnet_feats    = F.normalize(resnet_feats.float(), p=2, dim=1)  # [B, 512]

        clip_data = data.float() * self._renorm_scale + self._renorm_shift
        if clip_data.shape[-1] != 224:
            clip_data = F.interpolate(
                clip_data, size=224, mode='bicubic', align_corners=False)
        clip_feats = self.clip_model.encode_image(clip_data)             # [B, 512], L2 normed

        return torch.cat([resnet_feats, clip_feats], dim=1)              # [B, 1024]

    def _setup_renorm(self, device: torch.device):
        """Precompute per-channel renorm: source dataset stats → CLIP stats."""
        src_mean_v, src_std_v = normalization_dict[self.dataset_name]
        src_mean = torch.tensor(src_mean_v, dtype=torch.float32).view(1,3,1,1).to(device)
        src_std  = torch.tensor(src_std_v,  dtype=torch.float32).view(1,3,1,1).to(device)
        cl_mean  = torch.tensor(_CLIP_MEAN,  dtype=torch.float32).view(1,3,1,1).to(device)
        cl_std   = torch.tensor(_CLIP_STD,   dtype=torch.float32).view(1,3,1,1).to(device)
        self._renorm_scale = src_std / cl_std
        self._renorm_shift = (src_mean - cl_mean) / cl_std
