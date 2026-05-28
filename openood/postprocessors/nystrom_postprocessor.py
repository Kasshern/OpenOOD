"""
Algorithm 2: Kernel-Attention OOD via Anchor Keys (Nyström approximation).

Instead of the RFF mean embedding μ̂(x) ≈ φ(x)ᵀ μ̂_c, we approximate with
m << n anchor keys per class:

    m_c(x) = Σⱼ γ_{c,j} k(z_{c,j}, x)

where anchors Z_c and weights γ_c are fit by solving the representer equation:

    (K_{ZZ} + λI) γ_c = μ̂_Z,   μ̂_Z[j] = (1/n_c) Σᵢ k(z_j, xᵢ)

OOD score: m(x) = max_c m_c(x),   flag OOD if m(x) < τ_α

Theoretical backing: Theorem 11 (Nyström compression) + Theorem 14 (OOD consistency).
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .rff_postprocessor import RFFPostprocessor


class NystromOODPostprocessor(RFFPostprocessor):
    """Algorithm 2: Nyström Kernel-Attention OOD Detector.

    Inherits feature extraction, normalize, feature_space, alpha, sigma, and
    score_mode infrastructure from RFFPostprocessor. Replaces the RFF
    approximation with m exact kernel evaluations against learned anchor keys.
    """

    def __init__(self, config):
        super().__init__(config)

        # Nyström-specific hyperparameters
        self.n_anchors   = int(getattr(self.args, 'n_anchors',   None) or 200)
        self.anchor_init = str(getattr(self.args, 'anchor_init', None) or 'kmeans')
        self.reg_lambda  = float(getattr(self.args, 'reg_lambda', None) or 1e-4)
        self.kmeans_iter = int(getattr(self.args, 'kmeans_iter', None) or 50)
        # Entropy variant: use normalized kernel-attention entropy as OOD signal
        self.use_entropy = bool(getattr(self.args, 'use_entropy', False))

        # Learned parameters (set during setup)
        self.anchors = None   # [C, m, d] — per-class anchor keys
        self.gamma   = None   # [C, m]   — per-class Nyström weights

    # ── Gaussian kernel ────────────────────────────────────────────────────

    def _kernel(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """Gaussian kernel K[i,j] = exp(-‖xᵢ - zⱼ‖² / 2σ²).

        Args:
            X: [n, d]
            Z: [m, d]
        Returns:
            K: [n, m]
        """
        dist2 = torch.cdist(X.float(), Z.float(), p=2).pow(2)
        return torch.exp(-dist2 / (2.0 * self.sigma ** 2))

    # ── k-means and anchor init ────────────────────────────────────────────

    def _kmeans(self, X: torch.Tensor, k: int) -> torch.Tensor:
        """Lloyd's k-means on GPU. Returns [k, d] cluster centers."""
        n, d = X.shape
        k = min(k, n)
        # Random initialization
        idx = torch.randperm(n, device=X.device)[:k]
        centers = X[idx].clone().float()
        X_f = X.float()

        for _ in range(self.kmeans_iter):
            dists  = torch.cdist(X_f, centers)     # [n, k]
            assign = dists.argmin(dim=1)            # [n]
            new_c  = torch.zeros_like(centers)
            counts = torch.zeros(k, device=X.device)
            new_c.scatter_add_(0, assign.unsqueeze(1).expand(-1, d), X_f)
            counts.scatter_add_(0, assign, torch.ones(n, device=X.device))
            # Keep old center for empty clusters
            nonempty = counts > 0
            new_c[nonempty]  = new_c[nonempty] / counts[nonempty].unsqueeze(1)
            new_c[~nonempty] = centers[~nonempty]
            if torch.allclose(centers, new_c, atol=1e-6):
                break
            centers = new_c
        return centers

    def _init_anchors(self, X_c: torch.Tensor) -> torch.Tensor:
        """Return ≤ n_anchors anchor points for class features X_c [n_c, d]."""
        m = min(self.n_anchors, X_c.shape[0])
        if self.anchor_init == 'kmeans':
            return self._kmeans(X_c, m)
        else:  # 'random'
            idx = torch.randperm(X_c.shape[0], device=X_c.device)[:m]
            return X_c[idx].clone().float()

    # ── Nyström fit ────────────────────────────────────────────────────────

    def _fit_nystrom(self, device: torch.device):
        """Fit per-class anchors Z_c and weights γ_c from stored train features."""
        X = self.X_train_raw.to(device)
        if self.normalize:
            X = F.normalize(X.float(), p=2, dim=1)

        C   = self.num_classes
        m   = self.n_anchors
        d   = X.shape[1]

        self.anchors = torch.zeros(C, m, d,  device=device)
        self.gamma   = torch.zeros(C, m,     device=device)

        print(f'  Fitting Nyström anchors (m={m}, init={self.anchor_init}, '
              f'λ={self.reg_lambda})...')

        for c in tqdm(range(C), desc='Nyström per-class fit'):
            mask = (self.y_train.to(device) == c)
            X_c  = X[mask].float()
            n_c  = X_c.shape[0]
            if n_c == 0:
                continue

            Z_c  = self._init_anchors(X_c)           # [m_c, d], m_c ≤ m
            m_c  = Z_c.shape[0]

            K_ZZ = self._kernel(Z_c, Z_c)            # [m_c, m_c]
            K_ZX = self._kernel(Z_c, X_c)            # [m_c, n_c]
            mu_Z = K_ZX.mean(dim=1)                  # [m_c]

            # Solve (K_ZZ + λI) γ = μ̂_Z
            A = K_ZZ + self.reg_lambda * torch.eye(m_c, device=device)
            try:
                gamma_c = torch.linalg.solve(A, mu_Z)
            except torch.linalg.LinAlgError:
                gamma_c = torch.linalg.lstsq(
                    A, mu_Z.unsqueeze(1)).solution.squeeze(1)

            self.anchors[c, :m_c] = Z_c
            self.gamma[c,   :m_c] = gamma_c

        print(f'  Nyström fit complete. anchors: {self.anchors.shape}, '
              f'gamma: {self.gamma.shape}')

    # ── scoring ────────────────────────────────────────────────────────────

    def _nystrom_scores(self, X: torch.Tensor,
                        device: torch.device) -> torch.Tensor:
        """Compute per-class Nyström scores.

        Args:
            X: [n, d] — normalized feature matrix
        Returns:
            scores: [n, C]
        """
        C = self.num_classes
        n = X.shape[0]
        scores = torch.zeros(n, C, device=device)

        anchors = self.anchors.to(device)   # [C, m, d]
        gamma   = self.gamma.to(device)     # [C, m]

        for c in range(C):
            K = self._kernel(X, anchors[c])  # [n, m]
            scores[:, c] = K @ gamma[c]

        return scores

    # ── threshold calibration ──────────────────────────────────────────────

    def _compute_nystrom_threshold(self, device: torch.device):
        """Calibrate OOD threshold from validation Nyström scores."""
        X_val = self.X_val_raw.to(device)
        if self.normalize:
            X_val = F.normalize(X_val.float(), p=2, dim=1)

        val_scores_per_class = self._nystrom_scores(X_val, device)  # [n_val, C]

        if self.score_mode == 'softmax' and self.softmax_val is not None:
            sm     = self.softmax_val.to(device)
            scores = (val_scores_per_class * sm).sum(dim=1)
        elif self.score_mode == 'predictor_aware' and self.softmax_val is not None:
            sm      = self.softmax_val.to(device)
            pred    = sm.argmax(dim=1)
            n       = val_scores_per_class.size(0)
            pred_sim = val_scores_per_class[torch.arange(n, device=device), pred]
            entropy  = -(sm * torch.log(sm + 1e-8)).sum(dim=1)
            scores   = pred_sim - entropy
        else:
            scores = val_scores_per_class.max(dim=1).values

        self.threshold     = torch.quantile(scores, self.alpha)
        self.max_threshold = torch.quantile(
            val_scores_per_class.max(dim=1).values, self.alpha)
        print(f'  Nyström threshold ({self.score_mode}) at '
              f'{self.alpha * 100:.1f}%: {self.threshold:.4f}')

    # ── setup ──────────────────────────────────────────────────────────────

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        """Extract penultimate features, fit Nyström anchors, calibrate threshold."""
        if self.setup_flag:
            return

        print('\n' + '=' * 50)
        print('Setting up Nyström Kernel-Attention OOD detector...')
        print(f'  sigma:        {self.sigma}')
        print(f'  n_anchors:    {self.n_anchors}')
        print(f'  anchor_init:  {self.anchor_init}')
        print(f'  reg_lambda:   {self.reg_lambda}')
        print(f'  alpha:        {self.alpha}')
        print(f'  feature_space:{self.feature_space}')
        print(f'  normalize:    {self.normalize}')
        print(f'  score_mode:   {self.score_mode}')
        print('=' * 50)

        net.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ── Feature extraction ────────────────────────────────────────────
        if self.X_train_raw is None:
            _need_softmax = self.score_mode in ('softmax', 'predictor_aware')

            if self.feature_space == 'multilayer_pca':
                layer_feats_accum = [[] for _ in self.pca_layers]
                train_labels = []
                with torch.no_grad():
                    for batch in tqdm(id_loader_dict['train'],
                                      desc='Extracting train features (nystrom mlpca)'):
                        data = batch['data'].to(device).float()
                        _, feats = self._extract_multilayer_raw(net, data)
                        for j, f in enumerate(feats):
                            layer_feats_accum[j].append(f.cpu())
                        train_labels.append(batch['label'].to(device))

                self.y_train = torch.cat(train_labels, dim=0)
                layer_feats = [torch.cat(acc, dim=0)
                               for acc in layer_feats_accum]
                layer_feats = self._select_id_layers(layer_feats, self.y_train)
                reduced = []
                self.pca_W = []
                self.pca_mean = []
                for j in range(len(self.pca_layers)):
                    X_j = layer_feats[j].to(device)
                    mu_j = X_j.mean(dim=0)
                    X_c = X_j - mu_j
                    k = min(self.pca_components, X_c.shape[1])
                    Sigma = (X_c.T @ X_c) / (X_c.shape[0] - 1)
                    U, _, _ = torch.linalg.svd(Sigma, full_matrices=False)
                    W_j = U[:, :k]
                    self.pca_mean.append(mu_j)
                    self.pca_W.append(W_j)
                    reduced.append((X_c @ W_j).cpu())
                    del X_j, X_c, Sigma, U
                    torch.cuda.empty_cache()

                reduced_dev = [r.to(device) for r in reduced]
                self._compute_id_layer_weights(reduced_dev, self.y_train)
                self.X_train_raw = self._apply_layer_weights(reduced_dev)

                layer_val_accum = [[] for _ in self.pca_layers]
                val_labels = []
                val_softmax = [] if _need_softmax else None
                with torch.no_grad():
                    for batch in tqdm(id_loader_dict['val'],
                                      desc='Extracting val features (nystrom mlpca)'):
                        data = batch['data'].to(device).float()
                        output, feats = self._extract_multilayer_raw(net, data)
                        if _need_softmax:
                            val_softmax.append(torch.softmax(output, dim=1).cpu())
                        for j, f in enumerate(feats):
                            layer_val_accum[j].append(f.cpu())
                        val_labels.append(batch['label'].to(device))

                reduced_val = []
                for j in range(len(self.pca_layers)):
                    X_j = torch.cat(layer_val_accum[j], dim=0).to(device)
                    reduced_val.append(((X_j - self.pca_mean[j].to(device)) @
                                        self.pca_W[j].to(device)).cpu())
                    del X_j
                    torch.cuda.empty_cache()
                self.X_val_raw = self._apply_layer_weights(
                    [r.to(device) for r in reduced_val])
                self.y_val = torch.cat(val_labels, dim=0)
                if val_softmax is not None:
                    self.softmax_val = torch.cat(val_softmax, dim=0)

            elif self.feature_space == 'multilayer_minmax_concat':
                layer_feats_accum = [[] for _ in self.pca_layers]
                train_labels = []
                with torch.no_grad():
                    for batch in tqdm(id_loader_dict['train'],
                                      desc='Extracting train features (nystrom mlminmax)'):
                        data = batch['data'].to(device).float()
                        _, feats = self._extract_multilayer_raw(net, data)
                        for j, f in enumerate(feats):
                            layer_feats_accum[j].append(f.cpu())
                        train_labels.append(batch['label'].to(device))
                self.y_train = torch.cat(train_labels, dim=0)
                layer_feats = [torch.cat(acc, dim=0)
                               for acc in layer_feats_accum]
                layer_feats = self._select_id_layers(layer_feats, self.y_train)
                layer_feats = [features.to(device) for features in layer_feats]
                self._compute_id_layer_weights(layer_feats, self.y_train)
                self.X_train_raw = self._apply_layer_weights(layer_feats)

                layer_val_accum = [[] for _ in self.pca_layers]
                val_labels = []
                val_softmax = [] if _need_softmax else None
                with torch.no_grad():
                    for batch in tqdm(id_loader_dict['val'],
                                      desc='Extracting val features (nystrom mlminmax)'):
                        data = batch['data'].to(device).float()
                        output, feats = self._extract_multilayer_raw(net, data)
                        if _need_softmax:
                            val_softmax.append(torch.softmax(output, dim=1).cpu())
                        for j, f in enumerate(feats):
                            layer_val_accum[j].append(f.cpu())
                        val_labels.append(batch['label'].to(device))
                self.X_val_raw = self._apply_layer_weights([
                    torch.cat(acc, dim=0).to(device) for acc in layer_val_accum
                ])
                self.y_val = torch.cat(val_labels, dim=0)
                if val_softmax is not None:
                    self.softmax_val = torch.cat(val_softmax, dim=0)

            else:
                train_feats, train_labels = [], []
                with torch.no_grad():
                    for batch in tqdm(id_loader_dict['train'],
                                      desc='Extracting train features (nystrom)'):
                        data = batch['data'].to(device).float()
                        f = self._extract_features(net, data, apply_normalize=False)
                        train_feats.append(f.cpu())
                        train_labels.append(batch['label'].cpu())

                self.X_train_raw = torch.cat(train_feats, dim=0).to(device)
                self.y_train = torch.cat(train_labels, dim=0).to(device)

                val_feats, val_labels = [], []
                val_softmax = [] if _need_softmax else None
                with torch.no_grad():
                    for batch in tqdm(id_loader_dict['val'],
                                      desc='Extracting val features (nystrom)'):
                        data = batch['data'].to(device).float()
                        if _need_softmax:
                            output, f = self._extract_features_with_output(
                                net, data, apply_normalize=False)
                            val_softmax.append(torch.softmax(output, dim=1).cpu())
                        else:
                            f = self._extract_features(net, data, apply_normalize=False)
                        val_feats.append(f.cpu())
                        val_labels.append(batch['label'].cpu())

                self.X_val_raw = torch.cat(val_feats, dim=0).to(device)
                self.y_val = torch.cat(val_labels, dim=0).to(device)
                if val_softmax is not None:
                    self.softmax_val = torch.cat(val_softmax, dim=0)

            self.feature_dim = self.X_train_raw.shape[1]
            self.num_classes = int(self.y_train.max().item()) + 1
            print(f'Extracted {self.X_train_raw.shape[0]} train features '
                  f'of dim {self.feature_dim}')
            print(f'Extracted {self.X_val_raw.shape[0]} val features for threshold')

        # ── Nyström fit ───────────────────────────────────────────────────
        self._fit_nystrom(device)

        # ── Threshold calibration ─────────────────────────────────────────
        self._compute_nystrom_threshold(device)

        self.setup_flag = True
        print('Nyström setup complete.\n')

    # ── inference ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        """Compute Nyström OOD score for a batch.

        Returns:
            pred: [B] predicted class labels
            conf: [B] OOD scores (higher = more in-distribution)
        """
        device = data.device

        output, features = self._extract_features_with_output(
            net, data, apply_normalize=False)
        _, pred = torch.max(output, dim=1)

        if self.normalize:
            features = F.normalize(features.float(), p=2, dim=1)

        class_scores = self._nystrom_scores(features, device)  # [B, C]
        B = features.shape[0]

        if self.use_entropy:
            # Normalized kernel-attention entropy (lower = more concentrated = ID)
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
