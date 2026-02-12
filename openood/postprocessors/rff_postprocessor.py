from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


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

        # Hyperparameters
        self.sigma = self.args.sigma          # Kernel bandwidth
        self.D = self.args.D                  # RFF dimension
        self.alpha = self.args.alpha          # Target FPR for threshold

        # Feature space: 'penultimate', 'all', or 'input'
        self.feature_space = getattr(self.args, 'feature_space', 'penultimate')

        # Whether to L2 normalize features (makes sigma transferable across datasets)
        self.normalize = getattr(self.args, 'normalize', True)

        # Learned parameters (set during setup)
        self.omega = None        # [D, feature_dim] - RFF frequencies
        self.b = None            # [D] - RFF phases
        self.mu_hat = None       # [num_classes, D] - Per-class mean embeddings
        self.num_classes = None  # Number of classes
        self.threshold = None    # Scalar threshold
        self.feature_dim = None

        # Stored features and labels for hyperparameter search (avoid re-extraction)
        self.X_train = None      # Training features for mean embedding
        self.y_train = None      # Training labels
        self.X_val = None        # Validation features for threshold
        self.y_val = None        # Validation labels

        self.setup_flag = False

    def _extract_features(self, net: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """
        Extract features based on configured feature space.

        Args:
            net: Neural network model
            data: Input batch [batch_size, C, H, W]

        Returns:
            features: [batch_size, feature_dim]
        """
        if self.feature_space == 'input':
            features = torch.flatten(data, start_dim=1)
        elif self.feature_space == 'all':
            # Concatenate flattened features from all layers
            _, all_features = net(data, return_feature_list=True)
            features = torch.cat([f.flatten(start_dim=1) for f in all_features], dim=1)
        else:
            # Penultimate layer features (default) - use return_feature=True like KNN/VIM
            _, features = net(data, return_feature=True)

        # L2 normalize features if enabled (makes sigma transferable across datasets)
        if self.normalize:
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        return features

    def _sample_rff_params(self, feature_dim: int, device: torch.device):
        """Sample Random Fourier Feature parameters for Gaussian kernel."""
        self.feature_dim = feature_dim
        # Ω ~ N(0, σ^{-2} I) for Gaussian kernel
        self.omega = (torch.randn(self.D, feature_dim) / self.sigma).to(device)
        # b ~ Uniform[0, 2π]
        self.b = (torch.rand(self.D) * 2 * torch.pi).to(device)

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
        if self.X_train is None:
            return

        # Sample new RFF parameters with current sigma
        self._sample_rff_params(self.feature_dim, device)

        # Compute RFF features for all training samples
        phi_train = self._phi(self.X_train)  # [n_train, D]

        # Compute per-class mean embeddings
        self.mu_hat = torch.zeros(self.num_classes, self.D, device=device)
        for c in range(self.num_classes):
            mask = (self.y_train == c)
            if mask.sum() > 0:
                self.mu_hat[c] = phi_train[mask].mean(dim=0)

        # Compute validation scores: max similarity to any class centroid
        phi_val = self._phi(self.X_val)  # [n_val, D]
        # [n_val, num_classes] -> max over classes -> [n_val]
        val_scores = (phi_val @ self.mu_hat.T).max(dim=1).values

        # Threshold at alpha quantile (low scores are OOD)
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
        print('=' * 50)

        net.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Extract features only if not already done (for APS reuse)
        if self.X_train is None:
            # Extract training features and labels for mean embedding
            train_features = []
            train_labels = []
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Extracting train features'):
                    data = batch['data'].to(device).float()
                    labels = batch['label'].to(device)
                    features = self._extract_features(net, data)
                    train_features.append(features)
                    train_labels.append(labels)

            self.X_train = torch.cat(train_features, dim=0)
            self.y_train = torch.cat(train_labels, dim=0)
            self.feature_dim = self.X_train.shape[1]
            self.num_classes = int(self.y_train.max().item()) + 1
            print(f'Extracted {self.X_train.shape[0]} train features of dim {self.feature_dim}')
            print(f'  Feature norm (mean): {torch.linalg.norm(self.X_train, dim=1).mean():.4f}')
            print(f'  Number of classes: {self.num_classes}')

            # Extract validation features and labels for threshold calibration
            val_features = []
            val_labels = []
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['val'],
                                  desc='Extracting val features'):
                    data = batch['data'].to(device).float()
                    labels = batch['label'].to(device)
                    features = self._extract_features(net, data)
                    val_features.append(features)
                    val_labels.append(labels)

            self.X_val = torch.cat(val_features, dim=0)
            self.y_val = torch.cat(val_labels, dim=0)
            print(f'Extracted {self.X_val.shape[0]} val features for threshold')

        # Compute RFF embedding and threshold
        self._compute_rff_embedding(device)

        print(f'Per-class embedding norms (mean): {torch.linalg.norm(self.mu_hat, dim=1).mean():.4f}')
        print(f'Threshold at {self.alpha * 100:.1f}%: {self.threshold:.4f}')

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
        else:
            # Penultimate layer features (default) - use return_feature=True like KNN/VIM
            output, features = net(data, return_feature=True)

        # L2 normalize features if enabled
        if self.normalize:
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        _, pred = torch.max(output, dim=1)

        # Compute RFF features
        phi_x = self._phi(features)  # [batch_size, D]

        # Compute attention mass: max_c(μ̂_c^T φ(x))
        mu_hat = self.mu_hat.to(features.device)  # [num_classes, D]
        class_scores = phi_x @ mu_hat.T  # [batch_size, num_classes]
        conf = class_scores.max(dim=1).values  # [batch_size]

        return pred, conf

    def set_hyperparam(self, hyperparam: list):
        """
        Update hyperparameters for APS (Automatic Parameter Search) mode.
        Recomputes RFF parameters and mean embedding using stored features.
        """
        self.sigma = hyperparam[0]
        if len(hyperparam) > 1:
            self.D = int(hyperparam[1])

        # Recompute RFF embedding with new hyperparameters
        if self.X_train is not None:
            device = self.X_train.device
            self._compute_rff_embedding(device)

    def get_hyperparam(self):
        """Return current hyperparameters."""
        return [self.sigma, self.D]
