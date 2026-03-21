import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False


class RFFPostprocessor(BasePostprocessor):
    """
    Kernel Attention OOD Detection using Random Fourier Features.

    This method approximates a Gaussian kernel mean embedding for dataset-free
    inference. The OOD score is the "attention mass" = max_c(μ̂_c^T φ(x)), where
    μ̂_c is the per-class mean embedding and φ(x) is the RFF map.

    For universal kernels (e.g., Gaussian), the attention mass vanishes outside
    the in-distribution support, providing principled OOD guarantees.

    Reference:
        Random Fourier Features approximate the Gaussian kernel:
        k(x, y) = exp(-||x-y||² / 2σ²) ≈ φ(x)^T φ(y)
        where φ(x)_j = √(2/D) · cos(Ω_j^T x + b_j)
    """

    def __init__(self, config):
        super(RFFPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep

        # Select sweep dict based on search_mode (fast=default, full=expanded grid)
        self.search_mode_aps = getattr(self.args, 'search_mode', 'fast')
        if self.search_mode_aps == 'full' and hasattr(
                self.config.postprocessor, 'postprocessor_sweep_full'):
            self.args_dict = self.config.postprocessor.postprocessor_sweep_full

        # Hyperparameters
        self.sigma = self.args.sigma          # Kernel bandwidth
        self.D = self.args.D                  # RFF dimension
        self.alpha = self.args.alpha          # Target FPR for threshold

        # Feature space: 'penultimate', 'all', or 'input'
        self.feature_space = getattr(self.args, 'feature_space', 'penultimate')

        # Whether to L2 normalize features (makes sigma transferable across datasets)
        self.normalize = getattr(self.args, 'normalize', True)

        # Scoring mode: 'max' = max class score, 'margin' = max - 2nd max
        self.score_mode = getattr(self.args, 'score_mode', 'max')

        # Whether to normalize class scores by per-class variance
        self.variance_weighted = getattr(self.args, 'variance_weighted', True)

        # Learned parameters (set during setup)
        self.omega = None        # [D, feature_dim] - RFF frequencies
        self.b = None            # [D] - RFF phases
        self.mu_hat = None       # [num_classes, D] - Per-class mean embeddings
        self.nu_hat = None       # [num_classes, D] - Two-hop weighted mean embeddings
        self.var_hat = None      # [num_classes] - Per-class score variance
        self.num_classes = None  # Number of classes
        self.threshold = None    # Scalar threshold (for current score_mode)
        self.max_threshold = None  # Max-based threshold (for diagnostics)
        self.feature_dim = None

        # Stored features and labels for hyperparameter search (avoid re-extraction)
        self.X_train_raw = None  # Raw (unnormalized) training features
        self.y_train = None      # Training labels
        self.X_val_raw = None    # Raw (unnormalized) validation features
        self.y_val = None        # Validation labels
        self.softmax_val = None  # Val softmax probs for softmax score_mode [n_val, C]

        # ZCA whitening
        self.whiten = getattr(self.args, 'whiten', False)
        self.W_whiten = None      # [feature_dim, feature_dim] ZCA whitening matrix
        self.mu_train_raw = None  # [feature_dim] mean of raw training features (for centering)

        # Per-class ZCA whitening
        self.whiten_per_class = getattr(self.args, 'whiten_per_class', False)
        self.W_whiten_per_class = None   # [num_classes, feature_dim, feature_dim]
        self.mu_raw_per_class = None     # [num_classes, feature_dim]

        # Multi-layer PCA
        self.pca_layers     = getattr(self.args, 'pca_layers', [2, 3, 4])
        self.pca_components = getattr(self.args, 'pca_components', 128)
        self.pca_W    = None   # list of [d_i, pca_components] projection matrices
        self.pca_mean = None   # list of [d_i] per-layer means

        # Debug mode: φ(x) consistency, score distributions, kernel plots
        self.debug = getattr(self.args, 'debug', False)
        self._current_dataset_tag = None       # set externally before each inference
        self._debug_phi_stats = {}             # {tag: {'mean': float, 'std': float}}
        self._debug_score_accum = {}           # {tag: [conf_tensors...]}
        self._debug_phi_batch_means = {}       # {tag: [batch_mean,...]}
        self._debug_phi_batch_stds = {}        # {tag: [batch_std,...]}
        self._phi_train_for_kernel = None      # stored during setup for kernel plot

        # Diagnostic mode: track class score distributions
        self.diagnose = getattr(self.args, 'diagnose', False)
        self._diag_accum = {
            'n_above_threshold': [],
            'max_score': [],
            'margin': [],
            'mean_score': [],
        }
        # Per-dataset diagnostics for multi-class analysis
        self._diag_by_dataset = {}  # dataset_tag -> {'n_above': [], 'class_pred': []}

        self.setup_flag = False

    def _extract_features(self, net: nn.Module, data: torch.Tensor,
                          apply_normalize=None) -> torch.Tensor:
        """
        Extract features based on configured feature space.

        Args:
            net: Neural network model
            data: Input batch [batch_size, C, H, W]
            apply_normalize: Override normalization; defaults to self.normalize if None.

        Returns:
            features: [batch_size, feature_dim]
        """
        if apply_normalize is None:
            apply_normalize = self.normalize

        if self.feature_space == 'input':
            features = torch.flatten(data, start_dim=1)
        elif self.feature_space == 'all':
            # Concatenate flattened features from all layers
            _, all_features = net(data, return_feature_list=True)
            features = torch.cat([f.flatten(start_dim=1) for f in all_features], dim=1)
        elif self.feature_space == 'multilayer_pca':
            _, feats = self._extract_multilayer_raw(net, data)
            if self.pca_W is not None:
                reduced = [(f - mu.to(f.device)) @ W.to(f.device)
                           for f, mu, W in zip(feats, self.pca_mean, self.pca_W)]
                features = torch.cat(reduced, dim=1)
            else:
                features = torch.cat(feats, dim=1)   # raw concat (pre-PCA fallback)
        else:
            # Penultimate layer features (default) - use return_feature=True like KNN/VIM
            _, features = net(data, return_feature=True)

        # L2 normalize features if enabled (makes sigma transferable across datasets)
        if apply_normalize:
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        return features

    def _sample_rff_params(self, feature_dim: int, device: torch.device):
        """Sample Random Fourier Feature parameters for Gaussian kernel."""
        self.feature_dim = feature_dim
        # Ω ~ N(0, σ^{-2} I) for Gaussian kernel
        self.omega = (torch.randn(self.D, feature_dim) / self.sigma).to(device)
        # b ~ Uniform[0, 2π]
        self.b = (torch.rand(self.D) * 2 * torch.pi).to(device)

    def _compute_whiten_matrix(self, device: torch.device):
        """Compute ZCA whitening matrix from raw training features."""
        X = self.X_train_raw                                    # [n_train, feature_dim]
        self.mu_train_raw = X.mean(dim=0)                       # [feature_dim]
        X_c = X - self.mu_train_raw                             # centered
        n = X_c.shape[0]
        Sigma = (X_c.T @ X_c) / (n - 1)                        # [feature_dim, feature_dim]
        U, S, _ = torch.linalg.svd(Sigma, full_matrices=False)  # U:[d,d], S:[d]
        S_inv_sqrt = 1.0 / torch.sqrt(S.clamp(min=1e-6))
        # ZCA: W = U diag(S^{-1/2}) U^T  (symmetric — produces identity covariance)
        self.W_whiten = (U * S_inv_sqrt.unsqueeze(0)) @ U.T     # [feature_dim, feature_dim]

    def _apply_whiten(self, X: torch.Tensor) -> torch.Tensor:
        """Apply ZCA whitening: center then project."""
        return (X - self.mu_train_raw) @ self.W_whiten

    def _compute_per_class_whiten_matrix(self, device: torch.device):
        """Compute per-class ZCA whitening matrices from raw training features."""
        C = self.num_classes
        d = self.feature_dim
        self.W_whiten_per_class = torch.zeros(C, d, d, device=device)
        self.mu_raw_per_class = torch.zeros(C, d, device=device)
        for c in range(C):
            mask = (self.y_train == c)
            X_c = self.X_train_raw[mask]               # [n_c, d]
            mu_c = X_c.mean(dim=0)
            self.mu_raw_per_class[c] = mu_c
            if X_c.shape[0] < 2:
                self.W_whiten_per_class[c] = torch.eye(d, device=device)
                continue
            X_cc = X_c - mu_c
            n_c = X_cc.shape[0]
            Sigma_c = (X_cc.T @ X_cc) / (n_c - 1)     # [d, d]
            U, S, _ = torch.linalg.svd(Sigma_c, full_matrices=False)
            # Regularize: clamp relative to max eigenvalue to avoid blowup on near-zero dims
            S_inv_sqrt = 1.0 / torch.sqrt(S.clamp(min=S.max() * 1e-4))
            self.W_whiten_per_class[c] = (U * S_inv_sqrt.unsqueeze(0)) @ U.T

    def _fit_pca(self, layer_features: list, device):
        """Fit PCA per layer on training features.

        Args:
            layer_features: list of [n, d_i] tensors, one per selected layer
        """
        self.pca_W    = []
        self.pca_mean = []
        for X in layer_features:
            mu = X.mean(dim=0)
            X_c = X - mu
            k = min(self.pca_components, X_c.shape[1])
            Sigma = (X_c.T @ X_c) / (X_c.shape[0] - 1)
            U, _, _ = torch.linalg.svd(Sigma, full_matrices=False)
            self.pca_W.append(U[:, :k])    # [d_i, k]
            self.pca_mean.append(mu)

    def _extract_multilayer_raw(self, net, data):
        """Extract & pool selected layers from feature_list.

        Args:
            net: neural network with return_feature_list support
            data: [B, C, H, W] input batch

        Returns:
            output: logits [B, num_classes]
            feats:  list of [B, d_i] tensors (global-avg-pooled if spatial)
        """
        output, feature_list = net(data, return_feature_list=True)
        feats = []
        for i in self.pca_layers:
            f = feature_list[i]
            if f.dim() > 2:                # spatial: [B, C, H, W]
                f = f.mean(dim=[2, 3])     # global average pool → [B, C]
            feats.append(f)
        return output, feats               # list of [B, d_i]

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute RFF feature map.
        φ(x)_j = √(2/D) · cos(Ω_j^T x + b_j)

        Args:
            x: [batch_size, feature_dim] torch tensor

        Returns:
            [batch_size, D] torch tensor
        """
        device = x.device
        omega = self.omega.to(device)
        b = self.b.to(device)

        proj = x @ omega.T + b  # [batch_size, D]
        return torch.sqrt(torch.tensor(2.0 / self.D, device=device)) * torch.cos(proj)

    def _compute_rff_embedding(self, device: torch.device):
        """
        Recompute RFF parameters and per-class mean embeddings using stored features.
        Called when hyperparameters change during APS.
        """
        if self.X_train_raw is None:
            return

        if self.whiten_per_class and self.W_whiten_per_class is not None:
            # Per-class whitening: mu_hat[c] computed from class-c-whitened features only
            self._sample_rff_params(self.feature_dim, device)
            self.mu_hat = torch.zeros(self.num_classes, self.D, device=device)
            for c in range(self.num_classes):
                mask = (self.y_train == c)
                if mask.sum() == 0:
                    continue
                X_c = (self.X_train_raw[mask] - self.mu_raw_per_class[c]) @ self.W_whiten_per_class[c]
                if self.normalize:
                    X_c = torch.nn.functional.normalize(X_c, p=2, dim=1)
                self.mu_hat[c] = self._phi(X_c).mean(dim=0)

            # Compute val class scores: each val point whitened by each class's W_c
            n_val = self.X_val_raw.shape[0]
            val_class_scores = torch.zeros(n_val, self.num_classes, device=device)
            for c in range(self.num_classes):
                X_val_c = (self.X_val_raw - self.mu_raw_per_class[c]) @ self.W_whiten_per_class[c]
                if self.normalize:
                    X_val_c = torch.nn.functional.normalize(X_val_c, p=2, dim=1)
                phi_val_c = self._phi(X_val_c)             # [n_val, D]
                val_class_scores[:, c] = phi_val_c @ self.mu_hat[c]

            # Variance weighting
            self.var_hat = torch.ones(self.num_classes, device=device)
            if self.variance_weighted:
                for c in range(self.num_classes):
                    mask = (self.y_train == c)
                    if mask.sum() > 1:
                        X_c = (self.X_train_raw[mask] - self.mu_raw_per_class[c]) @ self.W_whiten_per_class[c]
                        if self.normalize:
                            X_c = torch.nn.functional.normalize(X_c, p=2, dim=1)
                        scores_c = self._phi(X_c) @ self.mu_hat[c]
                        self.var_hat[c] = scores_c.var().clamp(min=1e-8)
                val_class_scores = val_class_scores / torch.sqrt(self.var_hat)

            # Threshold (max score only for per-class whitening — twohop/margin not supported)
            self.max_threshold = torch.quantile(val_class_scores.max(dim=1).values, self.alpha)
            if self.score_mode == 'softmax':
                val_softmax = self.softmax_val.to(device)
                val_scores = (val_class_scores * val_softmax).sum(dim=1)
            else:
                val_scores = val_class_scores.max(dim=1).values
            self.threshold = torch.quantile(val_scores, self.alpha)
            return   # early return — skip existing embedding code below

        # Step 1: apply whitening from raw features (if enabled)
        if self.whiten and self.W_whiten is not None:
            X_train = self._apply_whiten(self.X_train_raw)
            X_val   = self._apply_whiten(self.X_val_raw)
        else:
            X_train = self.X_train_raw
            X_val   = self.X_val_raw

        # Step 2: L2 normalize (if enabled) — applied after whitening
        if self.normalize:
            X_train = torch.nn.functional.normalize(X_train, p=2, dim=1)
            X_val   = torch.nn.functional.normalize(X_val,   p=2, dim=1)

        # Sample new RFF parameters with current sigma
        self._sample_rff_params(self.feature_dim, device)

        # Compute RFF features for all training samples
        phi_train = self._phi(X_train)  # [n_train, D]

        if self.debug:
            self._debug_phi_stats['train'] = {
                'mean': phi_train.mean().item(),
                'std':  phi_train.std().item(),
            }
            self._phi_train_for_kernel = phi_train.detach().cpu()

        # Compute per-class mean embeddings
        self.mu_hat = torch.zeros(self.num_classes, self.D, device=device)
        for c in range(self.num_classes):
            mask = (self.y_train == c)
            if mask.sum() > 0:
                self.mu_hat[c] = phi_train[mask].mean(dim=0)

        # Compute two-hop weighted mean embeddings (only when needed)
        if self.score_mode == 'twohop':
            self.nu_hat = torch.zeros(self.num_classes, self.D, device=device)
            for c in range(self.num_classes):
                mask = (self.y_train == c)
                if mask.sum() > 0:
                    phi_c = phi_train[mask]                              # [n_c, D]
                    w_c = phi_c @ self.mu_hat[c]                        # [n_c] - centrality weights
                    self.nu_hat[c] = (w_c.unsqueeze(1) * phi_c).mean(dim=0)  # [D]

        # Compute per-class score variance for normalization
        self.var_hat = torch.ones(self.num_classes, device=device)
        if self.variance_weighted:
            for c in range(self.num_classes):
                mask = (self.y_train == c)
                if mask.sum() > 1:
                    scores_c = phi_train[mask] @ self.mu_hat[c]  # [n_c]
                    self.var_hat[c] = scores_c.var().clamp(min=1e-8)

        # Compute validation class scores
        phi_val = self._phi(X_val)  # [n_val, D]

        if self.debug:
            self._debug_phi_stats['val'] = {
                'mean': phi_val.mean().item(),
                'std':  phi_val.std().item(),
            }
        val_class_scores = phi_val @ self.mu_hat.T  # [n_val, num_classes]
        if self.variance_weighted:
            val_class_scores = val_class_scores / torch.sqrt(self.var_hat)

        # Always compute max-based threshold for diagnostics
        self.max_threshold = torch.quantile(
            val_class_scores.max(dim=1).values, self.alpha)

        # Compute scoring threshold based on score_mode
        if self.score_mode == 'margin':
            sorted_val = val_class_scores.sort(dim=1, descending=True).values
            val_scores = sorted_val[:, 0] - sorted_val[:, 1]
        elif self.score_mode == 'exclusive':
            # Calibrate on single-match val samples only (clean ID signal)
            n_above_val = (val_class_scores > self.max_threshold).sum(dim=1)
            single_mask = (n_above_val == 1)
            max_vals = val_class_scores.max(dim=1).values
            val_scores = max_vals[single_mask] if single_mask.sum() > 0 else max_vals
        elif self.score_mode == 'softmax':
            val_softmax = self.softmax_val.to(device)          # [n_val, num_classes]
            val_fused = val_class_scores * val_softmax          # [n_val, num_classes]
            val_scores = val_fused.sum(dim=1)                  # [n_val]
        elif self.score_mode == 'entropy':
            # Normalize class scores like document: w_c = relu(score_c) / Σ relu(score_c)
            val_pos = torch.relu(val_class_scores)                       # [n_val, num_classes]
            val_denom = val_pos.sum(dim=1, keepdim=True).clamp(min=1e-10)
            val_w = val_pos / val_denom                                  # [n_val, num_classes]
            val_scores = -(val_w * torch.log(val_w + 1e-10)).sum(dim=1) # negative entropy [n_val]
        elif self.score_mode == 'twohop':
            nu_hat = self.nu_hat.to(device)
            val_twohop = phi_val @ nu_hat.T                  # [n_val, num_classes]
            if self.variance_weighted:
                val_twohop = val_twohop / torch.sqrt(self.var_hat)
            val_scores = val_twohop.max(dim=1).values
        else:
            val_scores = val_class_scores.max(dim=1).values

        self.threshold = torch.quantile(val_scores, self.alpha)

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        """
        Setup phase: compute RFF parameters, per-class mean embeddings, and threshold.

        Uses training data for mean embedding and validation data for threshold.
        """
        if self.setup_flag:
            return

        print('\n' + '=' * 50)
        print('Setting up RFF Kernel Attention OOD detector...')
        print(f'  sigma (bandwidth): {self.sigma}')
        print(f'  D (RFF dimension): {self.D}')
        print(f'  alpha (target FPR): {self.alpha}')
        print(f'  feature_space: {self.feature_space}')
        print(f'  normalize: {self.normalize}')
        print(f'  score_mode: {self.score_mode}')
        print(f'  variance_weighted: {self.variance_weighted}')
        print('=' * 50)

        net.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Extract features only if not already done (for APS reuse)
        if self.X_train_raw is None:
            if self.feature_space == 'multilayer_pca':
                # --- Multi-layer PCA extraction path ---
                # Pass 1: collect per-layer train features and fit PCA
                layer_feats_accum = [[] for _ in self.pca_layers]
                train_labels = []
                with torch.no_grad():
                    for batch in tqdm(id_loader_dict['train'],
                                      desc='Extracting train features (mlpca)'):
                        data = batch['data'].to(device).float()
                        _, feats = self._extract_multilayer_raw(net, data)
                        for j, f in enumerate(feats):
                            layer_feats_accum[j].append(f.cpu())
                        train_labels.append(batch['label'].to(device))
                layer_feats = [torch.cat(acc, dim=0).to(device)
                               for acc in layer_feats_accum]
                self._fit_pca(layer_feats, device)
                reduced = [(f - mu) @ W
                           for f, mu, W in zip(layer_feats, self.pca_mean, self.pca_W)]
                self.X_train_raw = torch.cat(reduced, dim=1)
                self.y_train = torch.cat(train_labels, dim=0)
                self.feature_dim = self.X_train_raw.shape[1]
                self.num_classes = int(self.y_train.max().item()) + 1
                print(f'Extracted {self.X_train_raw.shape[0]} train features of dim '
                      f'{self.feature_dim} (multilayer_pca, {len(self.pca_layers)} layers)')
                print(f'  Feature norm (mean): '
                      f'{torch.linalg.norm(self.X_train_raw, dim=1).mean():.4f}')
                print(f'  Number of classes: {self.num_classes}')

                # Pass 2: collect per-layer val features and compress with fitted PCA
                layer_val_accum = [[] for _ in self.pca_layers]
                val_labels = []
                val_softmax = [] if self.score_mode == 'softmax' else None
                with torch.no_grad():
                    for batch in tqdm(id_loader_dict['val'],
                                      desc='Extracting val features (mlpca)'):
                        data = batch['data'].to(device).float()
                        if self.score_mode == 'softmax':
                            output, feats = self._extract_multilayer_raw(net, data)
                            val_softmax.append(torch.softmax(output, dim=1).cpu())
                        else:
                            _, feats = self._extract_multilayer_raw(net, data)
                        for j, f in enumerate(feats):
                            layer_val_accum[j].append(f.cpu())
                        val_labels.append(batch['label'].to(device))
                layer_val_feats = [torch.cat(acc, dim=0).to(device)
                                   for acc in layer_val_accum]
                reduced_val = [(f - mu.to(f.device)) @ W.to(f.device)
                               for f, mu, W in zip(layer_val_feats,
                                                   self.pca_mean, self.pca_W)]
                self.X_val_raw = torch.cat(reduced_val, dim=1)
                self.y_val = torch.cat(val_labels, dim=0)
                if val_softmax is not None:
                    self.softmax_val = torch.cat(val_softmax, dim=0)
                print(f'Extracted {self.X_val_raw.shape[0]} val features for threshold')

            else:
                # --- Standard feature extraction path ---
                # Extract training features and labels for mean embedding
                train_features = []
                train_labels = []
                with torch.no_grad():
                    for batch in tqdm(id_loader_dict['train'],
                                      desc='Extracting train features'):
                        data = batch['data'].to(device).float()
                        labels = batch['label'].to(device)
                        features = self._extract_features(net, data, apply_normalize=False)
                        train_features.append(features)
                        train_labels.append(labels)

                self.X_train_raw = torch.cat(train_features, dim=0)
                self.y_train = torch.cat(train_labels, dim=0)
                self.feature_dim = self.X_train_raw.shape[1]
                self.num_classes = int(self.y_train.max().item()) + 1
                print(f'Extracted {self.X_train_raw.shape[0]} train features of dim {self.feature_dim}')
                print(f'  Feature norm (mean): {torch.linalg.norm(self.X_train_raw, dim=1).mean():.4f}')
                print(f'  Number of classes: {self.num_classes}')

                # Extract validation features and labels for threshold calibration
                val_features = []
                val_labels = []
                val_softmax = [] if self.score_mode == 'softmax' else None
                with torch.no_grad():
                    for batch in tqdm(id_loader_dict['val'],
                                      desc='Extracting val features'):
                        data = batch['data'].to(device).float()
                        labels = batch['label'].to(device)
                        if self.score_mode == 'softmax':
                            output, features = net(data, return_feature=True)
                            # Store raw features; _compute_rff_embedding normalizes on the fly
                            val_softmax.append(torch.softmax(output, dim=1).cpu())
                        else:
                            features = self._extract_features(net, data, apply_normalize=False)
                        val_features.append(features)
                        val_labels.append(labels)

                self.X_val_raw = torch.cat(val_features, dim=0)
                self.y_val = torch.cat(val_labels, dim=0)
                if val_softmax is not None:
                    self.softmax_val = torch.cat(val_softmax, dim=0)  # [n_val, num_classes]
                print(f'Extracted {self.X_val_raw.shape[0]} val features for threshold')

        # Compute ZCA whitening matrix from raw training features
        self._compute_whiten_matrix(device)
        if self.whiten_per_class:
            self._compute_per_class_whiten_matrix(device)

        # Compute RFF embedding and threshold
        self._compute_rff_embedding(device)

        print(f'Per-class embedding norms (mean): {torch.linalg.norm(self.mu_hat, dim=1).mean():.4f}')
        if self.variance_weighted:
            print(f'Per-class score variance (mean): {self.var_hat.mean():.6f}')
        print(f'Threshold ({self.score_mode}) at {self.alpha * 100:.1f}%: {self.threshold:.4f}')
        print(f'Max threshold (for diagnostics) at {self.alpha * 100:.1f}%: {self.max_threshold:.4f}')

        self.setup_flag = True
        print('RFF setup complete.\n')

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        """
        Compute OOD score for a batch.

        Returns:
            pred: predicted class labels [batch_size]
            conf: attention mass scores [batch_size] (higher = more in-distribution)
        """
        # Get predictions and features
        if self.feature_space == 'input':
            output = net(data)
            features = torch.flatten(data, start_dim=1)
        elif self.feature_space == 'all':
            output, all_features = net(data, return_feature_list=True)
            features = torch.cat([f.flatten(start_dim=1) for f in all_features], dim=1)
        elif self.feature_space == 'multilayer_pca':
            output, feats = self._extract_multilayer_raw(net, data)
            reduced = [(f - mu.to(f.device)) @ W.to(f.device)
                       for f, mu, W in zip(feats, self.pca_mean, self.pca_W)]
            features = torch.cat(reduced, dim=1)
        else:
            # Penultimate layer features (default) - use return_feature=True like KNN/VIM
            output, features = net(data, return_feature=True)

        _, pred = torch.max(output, dim=1)

        # Per-class whitening path (takes priority over global whiten/normalize)
        if self.whiten_per_class and self.W_whiten_per_class is not None:
            W = self.W_whiten_per_class.to(features.device)   # [C, d, d]
            mu_r = self.mu_raw_per_class.to(features.device)  # [C, d]
            # features: [batch, d]
            x_centered = features.unsqueeze(1) - mu_r.unsqueeze(0)  # [batch, C, d]
            x_whitened = torch.einsum('bcd,cde->bce', x_centered, W)  # [batch, C, d]
            if self.normalize:
                x_whitened = torch.nn.functional.normalize(x_whitened, p=2, dim=2)
            B, C, d = x_whitened.shape
            phi_all = self._phi(x_whitened.reshape(B * C, d)).view(B, C, self.D)  # [B, C, D]
            mu_hat = self.mu_hat.to(features.device)           # [C, D]
            class_scores = (phi_all * mu_hat.unsqueeze(0)).sum(dim=2)  # [B, C]
            if self.variance_weighted:
                var_hat = self.var_hat.to(features.device)
                class_scores = class_scores / torch.sqrt(var_hat)
            if self.score_mode == 'softmax':
                softmax_probs = torch.softmax(output, dim=1)
                conf = (class_scores * softmax_probs).sum(dim=1)
            else:
                conf = class_scores.max(dim=1).values
            return pred, conf

        # Apply whitening before L2 normalize (mirrors _compute_rff_embedding)
        if self.whiten and self.W_whiten is not None:
            features = self._apply_whiten(features)

        # L2 normalize features if enabled
        if self.normalize:
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        # Compute RFF features
        phi_x = self._phi(features)  # [batch_size, D]

        if self.debug and self._current_dataset_tag is not None:
            tag = self._current_dataset_tag
            if tag not in self._debug_score_accum:
                self.set_dataset_tag(tag)
            self._debug_phi_batch_means[tag].append(phi_x.mean().item())
            self._debug_phi_batch_stds[tag].append(phi_x.std().item())

        # Compute class scores
        mu_hat = self.mu_hat.to(features.device)  # [num_classes, D]
        class_scores = phi_x @ mu_hat.T  # [batch_size, num_classes]
        if self.variance_weighted:
            var_hat = self.var_hat.to(features.device)  # [num_classes]
            class_scores = class_scores / torch.sqrt(var_hat)

        # Compute confidence based on score mode
        if self.score_mode == 'margin':
            sorted_scores = class_scores.sort(dim=1, descending=True).values
            conf = sorted_scores[:, 0] - sorted_scores[:, 1]  # max - 2nd max
        elif self.score_mode == 'exclusive':
            n_above = (class_scores > self.max_threshold).sum(dim=1)
            max_score = class_scores.max(dim=1).values
            # Only single-match samples get a positive score; multi and zero are OOD
            conf = torch.where(n_above == 1, max_score,
                               torch.full_like(max_score, -1.0))
        elif self.score_mode == 'softmax':
            softmax_probs = torch.softmax(output, dim=1)        # [batch, num_classes]
            conf = (class_scores * softmax_probs).sum(dim=1)    # [batch]
        elif self.score_mode == 'entropy':
            pos = torch.relu(class_scores)                              # [batch, num_classes]
            denom = pos.sum(dim=1, keepdim=True).clamp(min=1e-10)
            w = pos / denom                                             # [batch, num_classes]
            conf = -(w * torch.log(w + 1e-10)).sum(dim=1)              # [batch]
        elif self.score_mode == 'twohop':
            nu_hat = self.nu_hat.to(features.device)
            twohop_scores = phi_x @ nu_hat.T                # [batch, num_classes]
            if self.variance_weighted:
                twohop_scores = twohop_scores / torch.sqrt(var_hat)
            conf = twohop_scores.max(dim=1).values
        else:
            conf = class_scores.max(dim=1).values

        if self.debug and self._current_dataset_tag is not None:
            self._debug_score_accum[self._current_dataset_tag].append(conf.cpu())

        # Accumulate diagnostics if enabled
        if self.diagnose and self.max_threshold is not None:
            sorted_scores = class_scores.sort(dim=1, descending=True).values
            # Use max-based threshold for class counting (meaningful regardless of score_mode)
            n_above_diag = (class_scores > self.max_threshold).sum(dim=1).cpu()
            self._diag_accum['n_above_threshold'].append(n_above_diag)
            self._diag_accum['max_score'].append(sorted_scores[:, 0].cpu())
            self._diag_accum['margin'].append(
                (sorted_scores[:, 0] - sorted_scores[:, 1]).cpu())
            self._diag_accum['mean_score'].append(
                class_scores.mean(dim=1).cpu())

            # Per-dataset accumulation for multi-class analysis
            if self._current_dataset_tag is not None:
                tag = self._current_dataset_tag
                if tag not in self._diag_by_dataset:
                    self._diag_by_dataset[tag] = {'n_above': [], 'class_pred': []}
                self._diag_by_dataset[tag]['n_above'].append(n_above_diag)
                self._diag_by_dataset[tag]['class_pred'].append(
                    class_scores.argmax(dim=1).cpu())

        return pred, conf

    def save_diagnostics(self, save_path: str):
        """Save accumulated diagnostic statistics to .npz file and print summary."""
        diag = {}
        for key, val_list in self._diag_accum.items():
            if val_list:
                diag[key] = torch.cat(val_list).numpy()

        if not diag:
            return

        np.savez(save_path, **diag)
        self._print_diag_summary(diag, save_path)

    def _print_diag_summary(self, diag: dict, label: str = ''):
        """Print aggregate diagnostic summary."""
        n_above = diag['n_above_threshold']
        margin = diag['margin']
        max_score = diag['max_score']
        n = len(n_above)

        print(f'\n--- RFF Diagnostics ({n} samples) {label} ---')
        print(f'  Classes above threshold: '
              f'mean={n_above.mean():.1f}, '
              f'zero={100 * (n_above == 0).mean():.1f}%, '
              f'one={100 * (n_above == 1).mean():.1f}%, '
              f'multi(>1)={100 * (n_above > 1).mean():.1f}%')
        print(f'  Max score: mean={max_score.mean():.4f}, std={max_score.std():.4f}')
        print(f'  Margin (max-2nd): mean={margin.mean():.4f}, std={margin.std():.4f}')

    def save_per_class_analysis(self, save_path: str):
        """
        Save per-class multi-rate analysis by dataset to CSV.

        For each class c and dataset tag, computes:
          - n_single: argmax=c AND n_above==1
          - n_multi:  argmax=c AND n_above>1
          - n_zero:   argmax=c AND n_above==0
          - multi_rate = n_multi / (n_single + n_multi + n_zero)

        A high OOD multi_rate vs low id_test multi_rate confirms the hypothesis
        that multi-class similarity indicates OOD.
        """
        import csv

        if not self._diag_by_dataset:
            return

        # Consolidate accumulated tensors per dataset
        datasets = {}
        for tag, accum in self._diag_by_dataset.items():
            if accum['n_above'] and accum['class_pred']:
                datasets[tag] = {
                    'n_above': torch.cat(accum['n_above']).numpy(),
                    'class_pred': torch.cat(accum['class_pred']).numpy(),
                }

        if not datasets:
            return

        tags = list(datasets.keys())
        K = self.num_classes

        rows = []
        for c in range(K):
            row = {'class': c}
            for tag in tags:
                n_above = datasets[tag]['n_above']
                class_pred = datasets[tag]['class_pred']
                mask_c = (class_pred == c)
                n_single = int(((n_above == 1) & mask_c).sum())
                n_multi = int(((n_above > 1) & mask_c).sum())
                n_zero = int(((n_above == 0) & mask_c).sum())
                total = n_single + n_multi + n_zero
                multi_rate = round(n_multi / total, 4) if total > 0 else 0.0
                if tag == 'id_test':
                    row['id_test_single'] = n_single
                    row['id_test_multi'] = n_multi
                    row['id_test_multi_rate'] = multi_rate
                else:
                    row[f'{tag}_multi_rate'] = multi_rate
            rows.append(row)

        # Build column names: id_test gets 3 cols, others get 1
        fieldnames = ['class']
        if 'id_test' in tags:
            fieldnames += ['id_test_single', 'id_test_multi', 'id_test_multi_rate']
        other_tags = [t for t in tags if t != 'id_test']
        for tag in other_tags:
            fieldnames.append(f'{tag}_multi_rate')

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)

        print(f'\n--- RFF Per-Class Analysis → {save_path} ---')
        if 'id_test' in datasets:
            id_rates = np.array([row.get('id_test_multi_rate', 0.0) for row in rows])
            print(f'  id_test  multi_rate: mean={id_rates.mean():.4f}, '
                  f'max={id_rates.max():.4f}')
        for tag in other_tags:
            ood_rates = np.array([row.get(f'{tag}_multi_rate', 0.0) for row in rows])
            print(f'  {tag:<15s} multi_rate: mean={ood_rates.mean():.4f}, '
                  f'max={ood_rates.max():.4f}')

    def reset_diagnostics(self):
        """Reset diagnostic accumulators for next dataset."""
        for key in self._diag_accum:
            self._diag_accum[key] = []
        self._diag_by_dataset = {}

    def set_dataset_tag(self, tag: str):
        """Called from evaluator before inference on each dataset."""
        self._current_dataset_tag = tag
        if tag not in self._debug_score_accum:
            self._debug_score_accum[tag] = []
            self._debug_phi_batch_means[tag] = []
            self._debug_phi_batch_stds[tag] = []

    def _plot_kernel_distribution(self, save_dir: str):
        """Plot histogram of k(xᵢ,xⱼ) = φ(xᵢ)ᵀφ(xⱼ) for random train pairs."""
        phi = self._phi_train_for_kernel  # [n_train, D]
        n = phi.shape[0]
        n_pairs = min(2000, n * (n - 1) // 2)
        idx_i = torch.randint(0, n, (n_pairs,))
        idx_j = torch.randint(0, n, (n_pairs,))
        same = (idx_i == idx_j)
        idx_j[same] = (idx_j[same] + 1) % n  # ensure i ≠ j

        kernel_vals = (phi[idx_i] * phi[idx_j]).sum(dim=1).numpy()

        x_clip = np.percentile(kernel_vals, 99)
        n_clipped = int((kernel_vals > x_clip).sum())

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(kernel_vals, bins=60, color='steelblue', edgecolor='white', alpha=0.85)
        ax.axvline(kernel_vals.mean(), color='red', linestyle='--',
                   label=f'mean={kernel_vals.mean():.3f}')
        ax.set_xlim(left=kernel_vals.min() - 0.01, right=x_clip)
        if n_clipped > 0:
            ax.annotate(f'{n_clipped} pairs clipped (>{x_clip:.2f})',
                        xy=(0.97, 0.95), xycoords='axes fraction',
                        ha='right', va='top', fontsize=8, color='gray')
        ax.set_xlabel('k(xᵢ, xⱼ) = φ(xᵢ)ᵀ φ(xⱼ)')
        ax.set_ylabel('Count')
        ax.set_title('RFF Approximate Kernel Value Distribution (Random Train Pairs)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'rff_kernel_distribution.png'), dpi=150)
        plt.close()
        print(f'  Kernel dist: mean={kernel_vals.mean():.4f}, '
              f'std={kernel_vals.std():.4f}, '
              f'min={kernel_vals.min():.4f}, max={kernel_vals.max():.4f}')
        return {
            'mean': float(kernel_vals.mean()),
            'std': float(kernel_vals.std()),
            'min': float(kernel_vals.min()),
            'max': float(kernel_vals.max()),
            'n_pairs': int(n_pairs),
        }

    def _plot_phi_consistency(self, save_dir: str):
        """Bar chart of φ(x) mean ± std across train/val/test-ID/OOD splits."""
        for tag, means in self._debug_phi_batch_means.items():
            if means:
                self._debug_phi_stats[tag] = {
                    'mean': float(np.mean(means)),
                    'std':  float(np.mean(self._debug_phi_batch_stds[tag])),
                }

        labels = list(self._debug_phi_stats.keys())
        means  = [self._debug_phi_stats[l]['mean'] for l in labels]
        stds   = [self._debug_phi_stats[l]['std']  for l in labels]
        x = list(range(len(labels)))
        expected_std = float(np.sqrt(1.0 / self.D))

        fig, (ax_mean, ax_std) = plt.subplots(
            2, 1, figsize=(max(6, len(labels) * 0.9), 6), sharex=True)

        ax_mean.bar(x, means, color='cornflowerblue', edgecolor='white', alpha=0.85)
        ax_mean.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax_mean.set_ylim(-0.002, 0.002)
        ax_mean.set_ylabel('φ(x) mean')
        ax_mean.set_title('RFF Feature φ(x) Consistency Across Splits')

        ax_std.bar(x, stds, color='salmon', edgecolor='white', alpha=0.85)
        ax_std.axhline(expected_std, color='black', linewidth=0.8, linestyle='--',
                       label=f'expected √(1/D)={expected_std:.4f}')
        ax_std.set_ylim(0.025, 0.035)
        ax_std.set_ylabel('φ(x) std')
        ax_std.legend(fontsize=8)
        ax_std.set_xticks(x)
        ax_std.set_xticklabels(labels, rotation=30, ha='right')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'rff_phi_consistency.png'), dpi=150)
        plt.close()
        print('  φ(x) stats per split:')
        for label in labels:
            s = self._debug_phi_stats[label]
            print(f'    {label:15s}: mean={s["mean"]:.5f}, std={s["std"]:.5f}')
        return self._debug_phi_stats

    def _plot_score_distributions(self, save_dir: str):
        """Overlapping KDE of OOD scores: ID test vs each OOD dataset."""
        if not self._debug_score_accum:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        id_tag = 'id_test'
        palette = sns.color_palette('tab10', n_colors=len(self._debug_score_accum))

        for i, (tag, tensors) in enumerate(self._debug_score_accum.items()):
            if not tensors:
                continue
            scores = torch.cat(tensors).numpy()
            color = 'royalblue' if tag == id_tag else palette[i]
            lw = 2.5 if tag == id_tag else 1.5
            label = f'{tag} (ID)' if tag == id_tag else tag
            if tag == id_tag:
                sns.kdeplot(scores, ax=ax, label=label, color=color,
                            linewidth=lw, fill=True, alpha=0.2)
            else:
                sns.kdeplot(scores, ax=ax, label=label, color=color,
                            linewidth=lw, fill=False, alpha=0.85)

        ax.set_xlabel('OOD Score (higher = more in-distribution)')
        ax.set_ylabel('Density')
        ax.set_title('ID vs OOD Score Distributions')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'rff_score_distributions.png'), dpi=150)
        plt.close()

        score_stats = {}
        print('  Score dist:')
        for tag, tensors in self._debug_score_accum.items():
            if not tensors:
                continue
            scores = torch.cat(tensors).numpy()
            score_stats[tag] = {
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'n': int(len(scores)),
            }
            s = score_stats[tag]
            print(f'    {tag:15s}: mean={s["mean"]:.4f}, std={s["std"]:.4f}, n={s["n"]}')
        return score_stats

    def save_debug_plots(self, save_dir: str):
        """Generate and save all three debug plots to save_dir."""
        if not _PLOTTING_AVAILABLE:
            print('matplotlib/seaborn not available — skipping debug plots.')
            return
        os.makedirs(save_dir, exist_ok=True)
        print(f'\n--- RFF Debug Plots → {save_dir} ---')

        kernel_stats = None
        if self._phi_train_for_kernel is not None:
            kernel_stats = self._plot_kernel_distribution(save_dir)
        phi_stats = self._plot_phi_consistency(save_dir)
        score_stats = self._plot_score_distributions(save_dir)

        # Write all numerical stats to a single CSV
        import csv
        rows = []
        if kernel_stats is not None:
            for stat, value in kernel_stats.items():
                rows.append(('kernel_dist', 'train_pairs', stat, value))
        if phi_stats:
            for tag, stats in phi_stats.items():
                for stat, value in stats.items():
                    rows.append(('phi_consistency', tag, stat, value))
        if score_stats:
            for tag, stats in score_stats.items():
                for stat, value in stats.items():
                    rows.append(('score_dist', tag, stat, value))

        csv_path = os.path.join(save_dir, 'rff_debug_stats.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['check', 'tag', 'stat', 'value'])
            writer.writerows(rows)
        print(f'  Debug stats → {csv_path}')

    def set_hyperparam(self, hyperparam: list):
        """
        Update hyperparameters for APS (Automatic Parameter Search) mode.
        Recomputes RFF parameters and mean embedding using stored features.
        """
        self.sigma = hyperparam[0]
        if len(hyperparam) > 1:
            self.D = int(hyperparam[1])
        if len(hyperparam) > 2:
            self.alpha = float(hyperparam[2])
        if len(hyperparam) > 3:
            self.normalize = bool(hyperparam[3])  # only present in full grid

        # Recompute RFF embedding with new hyperparameters
        if self.X_train_raw is not None:
            device = self.X_train_raw.device
            self._compute_rff_embedding(device)

    def get_hyperparam(self):
        """Return current hyperparameters (always 4 values for JSON saving)."""
        return [self.sigma, self.D, self.alpha, self.normalize]
