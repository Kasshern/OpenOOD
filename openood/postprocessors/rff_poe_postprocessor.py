"""
RFFPOEPostprocessor: RFF + CLIP Product of Experts combination.

Instead of late fusion (linear score averaging), this uses a principled
Product of Experts (PoE) combination in log-odds space:

    logit(posterior_id) = logit(p_rff) + λ * logit(p_clip_msp)

which is equivalent to:

    p_combined ∝ p_rff(x)^1 * p_clip(x)^λ

Key properties:
  - λ=0 → pure RFF (exact recovery of base method)
  - λ>0 → CLIP contributes multiplicatively; when CLIP is very confident OOD
    it can strongly pull the posterior down even if RFF disagrees
  - Theoretically framed as Product of Experts, not "Bayesian prior" — CLIP
    is also a posterior (it sees x), so PoE is the honest framing

RFF score calibration:
  p_rff = sigmoid((rff_score - τ) / poe_scale)
where τ = self.threshold (α-quantile of ID val scores, already computed by
the parent). rff_score > τ → p_rff > 0.5 (looks ID). poe_scale controls
sigmoid steepness; fixed at config value (default 1.0), not swept.

APS sweeps only λ (lambda_) over [0.0, 0.5, 1.0, 2.0, 4.0].
"""

from typing import Any

import torch
import torch.nn.functional as F

from .rff_postprocessor import RFFPostprocessor
from .rff_clip_postprocessor import RFFCLIPPostprocessor


class RFFPOEPostprocessor(RFFCLIPPostprocessor):
    """RFF + CLIP Product of Experts OOD detector."""

    def __init__(self, config):
        super().__init__(config)
        self.lambda_   = float(getattr(self.args, 'lambda_',   None) or 1.0)
        self.poe_scale = float(getattr(self.args, 'poe_scale', None) or 1.0)

    # ── Postprocess ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def postprocess(self, net, data: Any):
        # 1. Raw RFF score — call grandparent directly to skip CLIP fusion in parent
        rff_pred, rff_conf = RFFPostprocessor.postprocess(self, net, data)

        # 2. CLIP MSP score — reuse parent's renorm + clip_model
        clip_data = data.float() * self._renorm_scale + self._renorm_shift
        if clip_data.shape[-1] != 224:
            clip_data = F.interpolate(
                clip_data, size=224, mode='bicubic', align_corners=False)
        clip_logits  = self.clip_model(clip_data)
        logit_scale  = self.clip_model.model.logit_scale.exp().item()
        p_clip = torch.softmax(clip_logits * logit_scale, dim=1).max(dim=1).values

        # 3. Calibrate RFF score → probability via sigmoid centered at threshold
        eps   = 1e-6
        tau   = self.threshold.to(rff_conf.device) if hasattr(self.threshold, 'to') \
                else torch.tensor(self.threshold, device=rff_conf.device)
        p_rff = torch.sigmoid((rff_conf - tau) / self.poe_scale).clamp(eps, 1.0 - eps)
        p_clip = p_clip.clamp(eps, 1.0 - eps)

        # 4. Product of Experts in log-odds space
        logit_post = torch.logit(p_rff) + self.lambda_ * torch.logit(p_clip)
        posterior  = torch.sigmoid(logit_post)

        return rff_pred, posterior

    # ── APS interface — sweep only lambda_ ────────────────────────────────────

    def set_hyperparam(self, hyperparam: list):
        self.lambda_ = float(hyperparam[0])

    def get_hyperparam(self):
        return [self.lambda_]
