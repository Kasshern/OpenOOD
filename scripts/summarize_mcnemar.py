"""
Pretty-print McNemar's test results from run_mcnemar.py output JSON.

Usage:
    python scripts/summarize_mcnemar.py results/mcnemar_rff_vs_msp.json
    python scripts/summarize_mcnemar.py results/mcnemar_batch.json --format latex
"""

import argparse
import json
import sys


def fmt_pval(p):
    if p < 0.001:
        return 'p<0.001***'
    if p < 0.01:
        return f'p={p:.3f}**'
    if p < 0.05:
        return f'p={p:.3f}*'
    return f'p={p:.3f}'


def fmt_effect(r):
    direction = '↑A' if r.get('method_a_better') else '↓A'
    return f"{fmt_pval(r['p_value'])} {direction}"


def print_comparison(entry, latex=False):
    method_a = entry['method_a']
    method_b = entry['method_b']
    results  = entry['results']
    print(f'\n{"="*60}')
    print(f'  {method_a}  vs  {method_b}')
    print(f'  task={entry["task"]}  seed={entry["seed"]}  TPR={entry["fpr_level"]*100:.0f}%')
    print(f'{"="*60}')
    print(entry['summary'])
    print()

    # Dataset rows (skip aggregates for table, print separately)
    rows = [(k, v) for k, v in results.items() if not k.startswith('aggregate')]
    aggs = [(k, v) for k, v in results.items() if k.startswith('aggregate')]

    if latex:
        print('% LaTeX table snippet')
        print(r'\begin{tabular}{lrrrl}')
        print(r'\toprule')
        print(r'Dataset & N & b (A$\succ$B) & c (B$\succ$A) & p-value \\')
        print(r'\midrule')
        for k, v in rows:
            pv = fmt_pval(v['p_value'])
            print(f"{k.replace('_', '-')} & {v['n_samples']} & "
                  f"{v['b_A_right_B_wrong']} & {v['c_A_wrong_B_right']} & {pv} \\\\")
        if aggs:
            print(r'\midrule')
            for k, v in aggs:
                pv = fmt_pval(v['p_value'])
                label = k.replace('aggregate_', 'Aggregate ').title()
                print(f"\\textbf{{{label}}} & {v['n_samples']} & "
                      f"{v['b_A_right_B_wrong']} & {v['c_A_wrong_B_right']} & {pv} \\\\")
        print(r'\bottomrule')
        print(r'\end{tabular}')
    else:
        col_w = max(len(k) for k, _ in rows + aggs) + 2 if rows or aggs else 20
        header = f"{'Dataset':<{col_w}} {'N':>8} {'b(A>B)':>8} {'c(B>A)':>8}  Result"
        print(header)
        print('-' * len(header))
        for k, v in rows:
            print(f"{k:<{col_w}} {v['n_samples']:>8} "
                  f"{v['b_A_right_B_wrong']:>8} {v['c_A_wrong_B_right']:>8}  "
                  f"{fmt_effect(v)}")
        if aggs:
            print('-' * len(header))
            for k, v in aggs:
                label = k.replace('aggregate_', 'AGGREGATE ').upper()
                print(f"{label:<{col_w}} {v['n_samples']:>8} "
                      f"{v['b_A_right_B_wrong']:>8} {v['c_A_wrong_B_right']:>8}  "
                      f"{fmt_effect(v)}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', help='Output JSON from run_mcnemar.py')
    parser.add_argument('--format', choices=['text', 'latex'], default='text')
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    # Single comparison vs batch
    if 'method_a' in data and 'method_b' in data:
        print_comparison(data, latex=(args.format == 'latex'))
    else:
        # Batch: dict of {method_b: comparison_entry}
        for method_b, entry in data.items():
            print_comparison(entry, latex=(args.format == 'latex'))


if __name__ == '__main__':
    main()
