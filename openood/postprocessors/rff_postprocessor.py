from typing import Any

import numpy as np
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
        self.cal_ratio = self.args.cal_ratio  # Calibration split ratio

        # Learned parameters (set during setup)
        self.omega = None      # [D, feature_dim] - RFF frequencies
        self.b = None          # [D] - RFF phases
        self.mu_hat = None     # [D] - Mean embedding
        self.threshold = None  # Scalar threshold
        self.feature_dim = None

        self.setup_flag = False

    def _sample_rff_params(self, feature_dim):
        """Sample Random Fourier Feature parameters for Gaussian kernel."""
        self.feature_dim = feature_dim
        # Ω ~ N(0, σ^{-2} I) for Gaussian kernel
        self.omega = np.random.randn(self.D, feature_dim) / self.sigma
        # b ~ Uniform[0, 2π]
        self.b = np.random.uniform(0, 2 * np.pi, self.D)

    def _phi_numpy(self, x):
        """
        Compute RFF feature map using numpy.
        φ(x)_j = √(2/D) · cos(Ω_j^T x + b_j)

        Args:
            x: [batch_size, feature_dim] numpy array
        Returns:
            [batch_size, D] numpy array
        """
        proj = x @ self.omega.T + self.b  # [batch_size, D]
        return np.sqrt(2.0 / self.D) * np.cos(proj)

    def _phi_torch(self, x):
        """
        Compute RFF feature map using torch tensors.

        Args:
            x: [batch_size, feature_dim] torch tensor
        Returns:
            [batch_size, D] torch tensor
        """
        device = x.device
        omega = self.omega_torch.to(device)
        b = self.b_torch.to(device)

        proj = x @ omega.T + b  # [batch_size, D]
        return np.sqrt(2.0 / self.D) * torch.cos(proj)

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        """
        Setup phase: compute RFF parameters, mean embedding, and threshold.

        Following the paper's Algorithm 1:
        1. Split training data into fitting and calibration sets
        2. Sample RFF parameters (Ω, b)
        3. Compute mean embedding μ̂ from fitting set
        4. Compute calibration scores and set threshold τ
        """
        if self.setup_flag:
            return

        print('\n' + '=' * 50)
        print('Setting up RFF Kernel Attention OOD detector...')
        print(f'  sigma (bandwidth): {self.sigma}')
        print(f'  D (RFF dimension): {self.D}')
        print(f'  alpha (target FPR): {self.alpha}')
        print(f'  cal_ratio: {self.cal_ratio}')
        print('=' * 50)

        net.eval()

        # Step 1: Extract all training features
        all_features = []
        with torch.no_grad():
            for batch in tqdm(id_loader_dict['train'],
                              desc='Extracting features',
                              position=0,
                              leave=True):
                data = batch['data'].cuda()
                data = data.float()

                _, feature = net(data, return_feature=True)
                all_features.append(feature.cpu().numpy())

        all_features = np.concatenate(all_features, axis=0)
        n_samples, feature_dim = all_features.shape
        print(f'Extracted {n_samples} features of dimension {feature_dim}')

        # Step 2: Split into fitting and calibration sets
        n_cal = int(n_samples * self.cal_ratio)
        indices = np.random.permutation(n_samples)
        cal_indices = indices[:n_cal]
        fit_indices = indices[n_cal:]

        X_fit = all_features[fit_indices]
        X_cal = all_features[cal_indices]
        print(f'Split: {len(fit_indices)} fitting, {len(cal_indices)} calibration')

        # Step 3: Sample RFF parameters
        self._sample_rff_params(feature_dim)

        # Step 4: Compute mean embedding from fitting set
        phi_fit = self._phi_numpy(X_fit)  # [n_fit, D]
        self.mu_hat = phi_fit.mean(axis=0)  # [D]
        print(f'Mean embedding computed, norm: {np.linalg.norm(self.mu_hat):.4f}')

        # Step 5: Compute calibration scores and set threshold
        phi_cal = self._phi_numpy(X_cal)  # [n_cal, D]
        cal_scores = phi_cal @ self.mu_hat  # [n_cal]

        # Threshold at alpha quantile (low scores are OOD)
        self.threshold = np.percentile(cal_scores, self.alpha * 100)
        print(f'Calibration scores - min: {cal_scores.min():.4f}, '
              f'max: {cal_scores.max():.4f}, mean: {cal_scores.mean():.4f}')
        print(f'Threshold set at {self.alpha * 100:.1f}% percentile: {self.threshold:.4f}')

        # Convert to torch tensors for inference
        self.omega_torch = torch.from_numpy(self.omega).float()
        self.b_torch = torch.from_numpy(self.b).float()
        self.mu_hat_torch = torch.from_numpy(self.mu_hat).float()

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
        output, feature = net(data, return_feature=True)
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)

        # Compute RFF features
        phi_x = self._phi_torch(feature)  # [batch_size, D]

        # Compute attention mass: m(x) = μ̂^T φ(x)
        mu_hat = self.mu_hat_torch.to(feature.device)
        conf = phi_x @ mu_hat  # [batch_size]

        return pred, conf

    def set_hyperparam(self, hyperparam: list):
        """
        Update hyperparameters for APS (Automatic Parameter Search) mode.

        Note: Changing sigma or D requires re-running setup to recompute
        the RFF parameters and mean embedding.
        """
        self.sigma = hyperparam[0]
        if len(hyperparam) > 1:
            self.D = int(hyperparam[1])
        # Reset setup flag to force recomputation
        self.setup_flag = False

    def get_hyperparam(self):
        """Return current hyperparameters."""
        return [self.sigma, self.D]
