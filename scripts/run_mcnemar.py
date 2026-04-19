"""
McNemar's test for paired OOD detection / ID classification comparisons.

Usage — OOD task (compare two methods on the same test set):
    python scripts/run_mcnemar.py \\
        --root  ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \\
        --method-a  rff_max_vw \\
        --method-b  msp \\
        --task  ood \\
        --out   results/mcnemar_rff_vs_msp.json

Usage — ID classification task:
    python scripts/run_mcnemar.py \\
        --root  ./results/cifar10_... \\
        --method-a  rff_max_vw \\
        --method-b  msp \\
        --task  id \\
        --out   results/mcnemar_id.json

Usage — batch mode (compare method-a against every other method found):
    python scripts/run_mcnemar.py \\
        --root  ./results/cifar10_... \\
        --method-a  rff_max_vw \\
        --batch \\
        --task  ood \\
        --out   results/mcnemar_batch.json

Notes:
  - Requires score pkls saved with --save-score during eval_ood.py runs.
  - Sample order must be identical across methods (guaranteed when shuffle=False).
  - McNemar's test applies to paired binary decisions, NOT raw AUROC scores.
  - Per-method threshold at target TPR (--fpr-level); each method thresholded independently.
"""

import argparse
import glob
import json
import os
import pickle
import sys
import warnings

import numpy as np

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)


# ---------------------------------------------------------------------------
# Score loading
# ---------------------------------------------------------------------------

def load_scores(root, method, seed):
    """
    Load score pkl for a given method and seed.
    Matches files of the form {method}_{8-char-hex}_scores.pkl exactly,
    so 'rff' does not accidentally match 'rff_max_vw' etc.
    """
    import re
    scores_dir = os.path.join(root, f's{seed}', 'scores')
    all_pkls = glob.glob(os.path.join(scores_dir, '*.pkl'))
    # Require: filename == method + '_' + exactly 8 hex chars + '_scores.pkl'
    pattern = re.compile(
        r'^' + re.escape(method) + r'_[0-9a-f]{8}_scores\.pkl$')
    matches = sorted(f for f in all_pkls
                     if pattern.match(os.path.basename(f)))
    if not matches:
        raise FileNotFoundError(
            f'No score file found for method "{method}" at {scores_dir}\n'
            f'Ensure eval_ood.py was run with --save-score.')
    if len(matches) > 1:
        print(f'  [warn] Multiple score files for {method}, using most recent: {matches[-1]}')
    with open(matches[-1], 'rb') as f:
        return pickle.load(f)


def discover_methods(root, seed):
    """Return all method names found in the scores directory."""
    import re
    scores_dir = os.path.join(root, f's{seed}', 'scores')
    files = glob.glob(os.path.join(scores_dir, '*_scores.pkl'))
    # Only match files of the form: {method}_{8-hex}_scores.pkl
    pat = re.compile(r'^(.+)_[0-9a-f]{8}_scores\.pkl$')
    methods = []
    for f in files:
        m = pat.match(os.path.basename(f))
        if m:
            methods.append(m.group(1))
    return sorted(set(methods))


# ---------------------------------------------------------------------------
# McNemar's test core
# ---------------------------------------------------------------------------

def run_mcnemar(correct_a, correct_b):
    """
    Run McNemar's test on two paired boolean arrays.

    Args:
        correct_a: boolean array, shape (N,) — was method A correct per sample?
        correct_b: boolean array, shape (N,) — was method B correct per sample?

    Returns:
        dict with table, p_value, discordant counts, and significance flags.
    """
    from scipy import stats as _scipy_stats

    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    assert len(correct_a) == len(correct_b), (
        f'Sample count mismatch: A has {len(correct_a)}, B has {len(correct_b)}. '
        'Ensure both methods were evaluated on the same split with shuffle=False.')

    n11 = int(np.sum( correct_a &  correct_b))   # both right
    b   = int(np.sum( correct_a & ~correct_b))   # A right, B wrong
    c   = int(np.sum(~correct_a &  correct_b))   # A wrong, B right
    n00 = int(np.sum(~correct_a & ~correct_b))   # both wrong

    # Use exact binomial test when discordant pairs are few (standard threshold: 25)
    use_exact = (min(b, c) < 25)

    if use_exact:
        # Exact two-sided binomial: H0: P(b) = 0.5
        p_value = float(_scipy_stats.binomtest(b, b + c, 0.5).pvalue)
        statistic = float(b)
        test_type = 'exact_binomial'
    else:
        # Chi-square with Yates continuity correction
        statistic = float((abs(b - c) - 1) ** 2 / (b + c))
        p_value = float(_scipy_stats.chi2.sf(statistic, df=1))
        test_type = 'chi2_yates'

    return {
        'n_samples': len(correct_a),
        'table': [[n11, b], [c, n00]],   # [[A&B, A only], [B only, neither]]
        'b_A_right_B_wrong': b,
        'c_A_wrong_B_right': c,
        'discordant_pairs': b + c,
        'p_value': p_value,
        'statistic': statistic,
        'test_type': test_type,
        'method_a_better': bool(b > c),
        'effect': 'A>B' if b > c else ('A<B' if c > b else 'tie'),
        'significant_005': bool(p_value < 0.05),
        'significant_001': bool(p_value < 0.01),
        'significant_0001': bool(p_value < 0.001),
    }


# ---------------------------------------------------------------------------
# OOD task helpers
# ---------------------------------------------------------------------------

def _threshold_at_tpr(id_conf, tpr_level):
    """
    Compute the confidence threshold that achieves `tpr_level` TPR on ID data.
    i.e., the (1 - tpr_level) quantile of ID confidences (lower tail = OOD side).
    """
    return float(np.quantile(id_conf, 1.0 - tpr_level))


def ood_decisions_for_dataset(scores_a, scores_b, split, dataset_name, tpr_level):
    """
    Build binary correct/incorrect arrays for one OOD dataset.

    Correct = sample correctly classified as ID (if it's ID) or OOD (if it's OOD).
    ID samples: from scores['id']['test'], label = 1
    OOD samples: from scores['ood'][split][dataset_name], label = 0
    """
    # --- method A ---
    id_conf_a = np.asarray(scores_a['id']['test'][1])
    ood_conf_a = np.asarray(scores_a['ood'][split][dataset_name][1])
    thr_a = _threshold_at_tpr(id_conf_a, tpr_level)

    conf_a = np.concatenate([id_conf_a, ood_conf_a])
    # True label: 1=ID, 0=OOD
    true_label = np.concatenate([
        np.ones(len(id_conf_a), dtype=bool),
        np.zeros(len(ood_conf_a), dtype=bool)
    ])
    # Decision: predict ID if conf >= threshold
    decision_a = conf_a >= thr_a
    correct_a = (decision_a == true_label)

    # --- method B ---
    id_conf_b = np.asarray(scores_b['id']['test'][1])
    ood_conf_b = np.asarray(scores_b['ood'][split][dataset_name][1])
    thr_b = _threshold_at_tpr(id_conf_b, tpr_level)

    conf_b = np.concatenate([id_conf_b, ood_conf_b])
    decision_b = conf_b >= thr_b
    correct_b = (decision_b == true_label)

    return correct_a, correct_b


def run_ood_task(scores_a, scores_b, tpr_level, target_dataset='all'):
    """
    Run McNemar's test for OOD detection across all (or one) OOD dataset(s).
    Returns dict keyed by '{split}/{dataset}' plus aggregated 'near' and 'far' entries.
    """
    results = {}
    near_ca_all, near_cb_all = [], []
    far_ca_all,  far_cb_all  = [], []

    ood_scores_a = scores_a.get('ood', {})
    ood_scores_b = scores_b.get('ood', {})

    for split in ('near', 'far'):
        datasets_a = ood_scores_a.get(split, {}) or {}
        datasets_b = ood_scores_b.get(split, {}) or {}
        common = set(datasets_a.keys()) & set(datasets_b.keys())

        for dset in sorted(common):
            if datasets_a[dset] is None or datasets_b[dset] is None:
                continue
            if target_dataset != 'all' and dset != target_dataset:
                continue

            try:
                ca, cb = ood_decisions_for_dataset(
                    scores_a, scores_b, split, dset, tpr_level)
            except Exception as e:
                warnings.warn(f'Skipping {split}/{dset}: {e}')
                continue

            key = f'{split}/{dset}'
            results[key] = run_mcnemar(ca, cb)

            if split == 'near':
                near_ca_all.append(ca); near_cb_all.append(cb)
            else:
                far_ca_all.append(ca);  far_cb_all.append(cb)

    # Aggregate over near / far
    if near_ca_all:
        results['aggregate_near'] = run_mcnemar(
            np.concatenate(near_ca_all), np.concatenate(near_cb_all))
    if far_ca_all:
        results['aggregate_far'] = run_mcnemar(
            np.concatenate(far_ca_all), np.concatenate(far_cb_all))
    if near_ca_all or far_ca_all:
        all_ca = near_ca_all + far_ca_all
        all_cb = near_cb_all + far_cb_all
        results['aggregate_all'] = run_mcnemar(
            np.concatenate(all_ca), np.concatenate(all_cb))

    return results


# ---------------------------------------------------------------------------
# ID classification task
# ---------------------------------------------------------------------------

def run_id_task(scores_a, scores_b):
    """Compare per-example ID classification correctness."""
    pred_a = np.asarray(scores_a['id_preds'] if scores_a.get('id_preds') is not None
                        else scores_a['id']['test'][0])
    pred_b = np.asarray(scores_b['id_preds'] if scores_b.get('id_preds') is not None
                        else scores_b['id']['test'][0])
    gt_a   = np.asarray(scores_a['id_labels'] if scores_a.get('id_labels') is not None
                        else scores_a['id']['test'][2])
    gt_b   = np.asarray(scores_b['id_labels'] if scores_b.get('id_labels') is not None
                        else scores_b['id']['test'][2])

    if not np.array_equal(gt_a, gt_b):
        warnings.warn('Ground truth labels differ between methods — check alignment.')

    correct_a = (pred_a == gt_a)
    correct_b = (pred_b == gt_b)
    return {'id_classification': run_mcnemar(correct_a, correct_b)}


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def build_summary(method_a, method_b, results):
    sig_better = sum(1 for k, v in results.items()
                     if not k.startswith('aggregate') and v.get('significant_001') and v.get('method_a_better'))
    sig_worse  = sum(1 for k, v in results.items()
                     if not k.startswith('aggregate') and v.get('significant_001') and not v.get('method_a_better'))
    total      = sum(1 for k in results if not k.startswith('aggregate'))
    return (f'{method_a} vs {method_b}: '
            f'{method_a} significantly better on {sig_better}/{total} datasets (p<0.01), '
            f'significantly worse on {sig_worse}/{total}.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="McNemar's test for paired OOD/ID comparisons.")
    parser.add_argument('--root',      required=True,
                        help='Results root directory (e.g. results/cifar10_resnet18_...)')
    parser.add_argument('--method-a',  required=True,
                        help='Primary method name (postprocessor name used in eval_ood.py)')
    parser.add_argument('--method-b',  default=None,
                        help='Comparison method name. Omit when using --batch.')
    parser.add_argument('--task',      default='ood', choices=['ood', 'id'],
                        help='Task type: ood (OOD detection) or id (ID classification)')
    parser.add_argument('--seed',      type=int, default=0,
                        help='Which seed folder (s0, s1, ...) to use (default: 0)')
    parser.add_argument('--fpr-level', type=float, default=0.95,
                        help='TPR level for OOD threshold (default: 0.95 = 95%% TPR)')
    parser.add_argument('--dataset',   default='all',
                        help='Specific OOD dataset name or "all" (default: all)')
    parser.add_argument('--batch',     action='store_true',
                        help='Compare method-a against all discovered methods')
    parser.add_argument('--out',       default=None,
                        help='Output JSON path (default: print to stdout)')
    args = parser.parse_args()

    if not args.batch and args.method_b is None:
        parser.error('--method-b is required unless --batch is used.')

    # Discover comparison methods in batch mode
    if args.batch:
        all_methods = discover_methods(args.root, args.seed)
        methods_b = [m for m in all_methods if m != args.method_a]
        if not methods_b:
            print('No other methods found to compare against.', file=sys.stderr)
            sys.exit(1)
        print(f'Batch mode: comparing {args.method_a} against {methods_b}')
    else:
        methods_b = [args.method_b]

    # Load method A once
    print(f'Loading scores for {args.method_a}...')
    scores_a = load_scores(args.root, args.method_a, args.seed)

    batch_output = {}

    for method_b in methods_b:
        print(f'Loading scores for {method_b}...')
        try:
            scores_b = load_scores(args.root, method_b, args.seed)
        except FileNotFoundError as e:
            warnings.warn(str(e))
            continue

        if args.task == 'ood':
            results = run_ood_task(scores_a, scores_b, args.fpr_level, args.dataset)
        else:
            results = run_id_task(scores_a, scores_b)

        summary = build_summary(args.method_a, method_b, results)
        print(summary)

        output = {
            'method_a':  args.method_a,
            'method_b':  method_b,
            'root':      args.root,
            'task':      args.task,
            'seed':      args.seed,
            'fpr_level': args.fpr_level,
            'results':   results,
            'summary':   summary,
        }

        if args.batch:
            batch_output[method_b] = output
        else:
            batch_output = output

    # Output
    out_str = json.dumps(batch_output, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
        with open(args.out, 'w') as f:
            f.write(out_str)
        print(f'Saved results to {args.out}')
    else:
        print(out_str)


if __name__ == '__main__':
    main()
