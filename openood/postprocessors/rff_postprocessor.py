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
        self.var_hat = None      # [num_classes] - Per-class score variance
        self.num_classes = None  # Number of classes
        self.threshold = None    # Scalar threshold (for current score_mode)
        self.max_threshold = None  # Max-based threshold (for diagnostics)
        self.feature_dim = None

        # Stored features and labels for hyperparameter search (avoid re-extraction)
        self.X_train = None      # Training features for mean embedding
        self.y_train = None      # Training labels
        self.X_val = None        # Validation features for threshold
        self.y_val = None        # Validation labels

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

        # Compute per-class score variance for normalization
        self.var_hat = torch.ones(self.num_classes, device=device)
        if self.variance_weighted:
            for c in range(self.num_classes):
                mask = (self.y_train == c)
                if mask.sum() > 1:
                    scores_c = phi_train[mask] @ self.mu_hat[c]  # [n_c]
                    self.var_hat[c] = scores_c.var().clamp(min=1e-8)

        # Compute validation class scores
        phi_val = self._phi(self.X_val)  # [n_val, D]

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
        else:
            # Penultimate layer features (default) - use return_feature=True like KNN/VIM
            output, features = net(data, return_feature=True)

        # L2 normalize features if enabled
        if self.normalize:
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        _, pred = torch.max(output, dim=1)

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
        else:
            conf = class_scores.max(dim=1).values

        if self.debug and self._current_dataset_tag is not None:
            self._debug_score_accum[self._current_dataset_tag].append(conf.cpu())

        # Accumulate diagnostics if enabled
        if self.diagnose and self.max_threshold is not None:
            sorted_scores = class_scores.sort(dim=1, descending=True).values
            # Use max-based threshold for class counting (meaningful regardless of score_mode)
            self._diag_accum['n_above_threshold'].append(
                (class_scores > self.max_threshold).sum(dim=1).cpu())
            self._diag_accum['max_score'].append(sorted_scores[:, 0].cpu())
            self._diag_accum['margin'].append(
                (sorted_scores[:, 0] - sorted_scores[:, 1]).cpu())
            self._diag_accum['mean_score'].append(
                class_scores.mean(dim=1).cpu())

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

    def reset_diagnostics(self):
        """Reset diagnostic accumulators for next dataset."""
        for key in self._diag_accum:
            self._diag_accum[key] = []

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

        # Recompute RFF embedding with new hyperparameters
        if self.X_train is not None:
            device = self.X_train.device
            self._compute_rff_embedding(device)

    def get_hyperparam(self):
        """Return current hyperparameters."""
        return [self.sigma, self.D, self.alpha]
