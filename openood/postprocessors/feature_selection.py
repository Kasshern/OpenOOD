import torch


def compute_id_layer_weights(layer_features,
                             labels,
                             metric='id_fisher',
                             normalize='sum',
                             eps=1e-8):
    if metric != 'id_fisher':
        raise ValueError(f'Unsupported layer weighting metric: {metric}')
    if normalize != 'sum':
        raise ValueError(f'Unsupported layer weight normalization: {normalize}')

    labels = labels.to(layer_features[0].device)
    classes = labels.unique()
    scores = []
    for features in layer_features:
        features = features.float()
        global_mean = features.mean(dim=0)
        between_class = torch.tensor(0.0, device=features.device)
        within_class = torch.tensor(0.0, device=features.device)
        for cls in classes:
            mask = labels == cls
            if mask.sum() == 0:
                continue
            class_features = features[mask]
            class_mean = class_features.mean(dim=0)
            between_class += mask.sum().float() * (
                (class_mean - global_mean) ** 2).sum()
            within_class += ((class_features - class_mean) ** 2).sum()
        scores.append(between_class / within_class.clamp(min=eps))

    weights = torch.stack(scores).float()
    weights = torch.clamp(weights, min=0.0)
    total = weights.sum()
    if not torch.isfinite(total) or total <= eps:
        weights = torch.ones_like(weights) / max(len(weights), 1)
    else:
        weights = weights / total
    return weights


def format_layer_weights(layer_ids, weights):
    return ', '.join(
        f'layer{layer_id}={weight:.3f}'
        for layer_id, weight in zip(layer_ids, weights.tolist()))
