from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class RFFPostprocessor(BasePostprocessor):
    """
    Kernel Attention OOD Detection using Random Fourier Features.

    This method approximates a Gaussian kernel mean embedding for dataset-free
    inference. The OOD score is the "attention mass" = μ̂^T φ(x), where μ̂ is
    the mean embedding of the training distribution and φ(x) is the RFF map.

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

        # Learned parameters (set during setup)
        self.omega = None       # [D, feature_dim] - RFF frequencies
        self.b = None           # [D] - RFF phases
        self.mu_hat = None      # [D] - Mean embedding
        self.threshold = None   # Scalar threshold
        self.feature_dim = None

        # Stored features for hyperparameter search (avoid re-extraction)
        self.X_train = None     # Training features for mean embedding
        self.X_val = None       # Validation features for threshold

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
            return torch.flatten(data, start_dim=1)

        logits, all_features = net(data, return_feature_list=True)

        if self.feature_space == 'all':
            # Concatenate flattened features from all layers
            return torch.cat([f.flatten(start_dim=1) for f in all_features], dim=1)
        else:
            # Penultimate layer features (default)
            return all_features[-1].view(all_features[-1].size(0), -1)

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
        Recompute RFF parameters and mean embedding using stored features.
        Called when hyperparameters change during APS.
        """
        if self.X_train is None:
            return

        # Sample new RFF parameters with current sigma
        self._sample_rff_params(self.feature_dim, device)

        # Compute mean embedding from training set
        phi_train = self._phi(self.X_train)  # [n_train, D]
        self.mu_hat = phi_train.mean(dim=0)  # [D]

        # Compute threshold from validation set
        phi_val = self._phi(self.X_val)  # [n_val, D]
        val_scores = phi_val @ self.mu_hat  # [n_val]

        # Threshold at alpha quantile (low scores are OOD)
        self.threshold = torch.quantile(val_scores, self.alpha)

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        """
        Setup phase: compute RFF parameters, mean embedding, and threshold.

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
        print('=' * 50)

        net.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Extract features only if not already done (for APS reuse)
        if self.X_train is None:
            # Extract training features for mean embedding
            train_features = []
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Extracting train features'):
                    data = batch['data'].to(device).float()
                    features = self._extract_features(net, data)
                    train_features.append(features)

            self.X_train = torch.cat(train_features, dim=0)
            self.feature_dim = self.X_train.shape[1]
            print(f'Extracted {self.X_train.shape[0]} train features of dim {self.feature_dim}')

            # Extract validation features for threshold calibration
            val_features = []
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['val'],
                                  desc='Extracting val features'):
                    data = batch['data'].to(device).float()
                    features = self._extract_features(net, data)
                    val_features.append(features)

            self.X_val = torch.cat(val_features, dim=0)
            print(f'Extracted {self.X_val.shape[0]} val features for threshold')

        # Compute RFF embedding and threshold
        self._compute_rff_embedding(device)

        print(f'Mean embedding norm: {torch.linalg.norm(self.mu_hat):.4f}')
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
        # Get predictions
        if self.feature_space == 'input':
            output = net(data)
            features = torch.flatten(data, start_dim=1)
        else:
            output, all_features = net(data, return_feature_list=True)
            if self.feature_space == 'all':
                features = torch.cat([f.flatten(start_dim=1) for f in all_features], dim=1)
            else:
                features = all_features[-1].view(all_features[-1].size(0), -1)

        _, pred = torch.max(output, dim=1)

        # Compute RFF features and attention mass
        phi_x = self._phi(features)  # [batch_size, D]
        mu_hat = self.mu_hat.to(features.device)
        conf = phi_x @ mu_hat  # [batch_size]

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
