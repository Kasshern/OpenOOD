"""
RFFCLIPMlMinmaxPostprocessor: RFF on multilayer_minmax + CLIP feature concat.

Feature space:
  - ResNet layers 1-4, minmax spatial pooling, Yosinski weights → 1408-d  (L2 normalized)
  - CLIP ViT-B/16 image embedding                                 →  512-d  (L2 normalized)
  - Concatenated                                                  → 1920-d

Algorithm 1 (RFF kernel attention) runs on the full 1920-d space. Provides
the final step in the ablation:

  penultimate (512) → multilayer_minmax (1408) → penultimate+CLIP (1024)
                    → multilayer_minmax+CLIP (1920)  ← this class

Inherits setup(), postprocess(), CLIP loading, and __getstate__ from
RFFCLIPConcatPostprocessor. Only _extract_concat_features is overridden.
"""

import torch
import torch.nn.functional as F

from .rff_clip_concat_postprocessor import RFFCLIPConcatPostprocessor


class RFFCLIPMlMinmaxPostprocessor(RFFCLIPConcatPostprocessor):
    """RFF kernel OOD on multilayer_minmax (1408-d) + CLIP (512-d) = 1920-d."""

    @torch.no_grad()
    def _extract_concat_features(self, net, data: torch.Tensor) -> torch.Tensor:
        # 1. Multilayer minmax features — pool_mode, pca_layers, layer_weights
        #    (yosinski_weights) are all read from config in RFFPostprocessor.__init__
        _, feats = self._extract_multilayer_raw(net, data)
        ml_feats = self._apply_layer_weights(feats)             # [B, 1408]
        ml_feats = F.normalize(ml_feats.float(), p=2, dim=1)

        # 2. CLIP image embedding
        clip_data = data.float() * self._renorm_scale + self._renorm_shift
        if clip_data.shape[-1] != 224:
            clip_data = F.interpolate(
                clip_data, size=224, mode='bicubic', align_corners=False)
        clip_feats = self.clip_model.encode_image(clip_data)    # [B, 512], L2 normed

        return torch.cat([ml_feats, clip_feats], dim=1)         # [B, 1920]
