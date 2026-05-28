#!/usr/bin/env python3
"""
Parse OOD benchmark .out log files from /Users/kasshern/projects/OpenOOD/logs/
Average all OOD metrics across all seeds per (variant, dataset).
Prints a ranked table per dataset.
"""

import os
import re
from collections import defaultdict

LOG_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(LOG_DIR, "average_results.txt")
DATASETS = ["cifar10", "cifar100", "imagenet200", "imagenet"]

# ── helpers ───────────────────────────────────────────────────────────────────

def split_blocks(text):
    block_start = re.compile(
        r'^=== (?!Done\b)(?!=+\s*$)(.+?) \| (cifar10|cifar100|imagenet200|imagenet) \|'
    )
    lines = text.splitlines()
    blocks = []
    current_header = None
    current_lines = []

    for line in lines:
        m = block_start.match(line)
        if m:
            if current_header is not None:
                blocks.append((current_header, "\n".join(current_lines)))
            current_header = line
            current_lines = [line]
        else:
            if current_header is not None:
                current_lines.append(line)

    if current_header is not None:
        blocks.append((current_header, "\n".join(current_lines)))

    return blocks


def parse_header(header_line):
    m = re.match(r'^=== (.+?) \| (cifar10|cifar100|imagenet200|imagenet) \|', header_line)
    if not m:
        return None, None
    return m.group(1).strip(), m.group(2).strip()


def parse_summary_table(block_text):
    """Return averaged per-row results dict for all rows found in a block.

    eval_ood.py prints one table per seed but writes the final mean±std to CSV
    (not stdout), so we collect all plain-number tables and average them here.
    Rows include individual datasets (cifar100, tin, mnist, ssb_hard, etc.)
    as well as summary rows (nearood, farood).
    """
    table_header_re = re.compile(
        r'^\s+FPR@95\s+AUROC\s+AUPR_IN\s+AUPR_OUT\s+ACC'
    )
    row_re = re.compile(
        r'^(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
    )

    lines = block_text.splitlines()

    # Collect all table start indices
    header_indices = [i for i, l in enumerate(lines) if table_header_re.match(l)]

    if not header_indices:
        return None

    all_tables = []
    for hdr_idx in header_indices:
        tbl = {}
        for line in lines[hdr_idx + 1:]:
            m = row_re.match(line)
            if m:
                tbl[m.group(1)] = {
                    "FPR@95":   float(m.group(2)),
                    "AUROC":    float(m.group(3)),
                    "AUPR_IN":  float(m.group(4)),
                    "AUPR_OUT": float(m.group(5)),
                    "ACC":      float(m.group(6)),
                }
            elif line.strip() == "" or line.startswith("---"):
                break
        if "nearood" in tbl and "farood" in tbl:
            all_tables.append(tbl)

    if not all_tables:
        return None

    # Union of all row names across seeds, preserving first-seen order
    seen_keys = {}
    for tbl in all_tables:
        for k in tbl:
            if k not in seen_keys:
                seen_keys[k] = True
    row_names = list(seen_keys.keys())

    # Average each metric across seeds for every row
    averaged = {}
    for k in row_names:
        vals_per_metric = {m: [t[k][m] for t in all_tables if k in t] for m in METRICS}
        averaged[k] = {m: sum(v) / len(v) for m, v in vals_per_metric.items() if v}
    averaged["_n_seeds_in_block"] = len(all_tables)
    averaged["_row_order"] = row_names
    return averaged


# ── helpers: walltime ─────────────────────────────────────────────────────────

WALLTIME_RE = re.compile(r'walltime=(\d+):(\d+):(\d+)')

def parse_walltime(text):
    """Return walltime in seconds from 'Rsrc Used: ... walltime=HH:MM:SS ...' line."""
    m = WALLTIME_RE.search(text)
    if not m:
        return None
    h, mn, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return h * 3600 + mn * 60 + s

def fmt_duration(seconds):
    """Format seconds as H:MM:SS."""
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h}:{m:02d}:{s:02d}"


# ── main parsing ──────────────────────────────────────────────────────────────

# raw[variant][dataset] = list of {"near": {metric: val}, "far": {metric: val},
#                                   "walltime": float|None, "n_seeds": int, "job_id": str}
METRICS = ["FPR@95", "AUROC", "AUPR_IN", "AUPR_OUT", "ACC"]
raw = defaultdict(lambda: defaultdict(list))

out_files = sorted(f for f in os.listdir(LOG_DIR) if f.endswith(".out"))

# Extract job ID from filename: rff_variant_JOBID_ARRAYID.out → JOBID
JOB_ID_RE = re.compile(r'_(\d{5,})_\d+\.out$')

parse_errors = []
total_blocks = 0
parsed_blocks = 0

for fname in out_files:
    fpath = os.path.join(LOG_DIR, fname)
    with open(fpath, "r", errors="replace") as fh:
        text = fh.read()

    file_walltime = parse_walltime(text)
    job_id_m = JOB_ID_RE.search(fname)
    job_id = job_id_m.group(1) if job_id_m else "?"

    blocks = split_blocks(text)
    total_blocks += len(blocks)

    # Parse valid blocks first; each block may contain multiple per-seed tables
    valid_blocks = []
    for header, block_text in blocks:
        variant, dataset = parse_header(header)
        if variant is None:
            continue
        table = parse_summary_table(block_text)
        if table is None:
            parse_errors.append(f"  No table in {fname}: {header[:80]}")
            continue
        valid_blocks.append((variant, dataset, table))

    # Total seeds across valid blocks (for walltime-per-seed calculation)
    total_seeds = sum(t.get("_n_seeds_in_block", 1) for _, _, t in valid_blocks)
    per_seed_walltime = (file_walltime / total_seeds) if (file_walltime and total_seeds > 0) else None

    for variant, dataset, table in valid_blocks:
        row_order = table.get("_row_order", [k for k in table if not k.startswith("_")])
        rows = {k: table[k] for k in row_order}
        raw[variant][dataset].append({
            "rows":     rows,
            "row_order": row_order,
            "walltime": per_seed_walltime,
            "n_seeds":  table.get("_n_seeds_in_block", 1),
            "job_id":   job_id,
        })
        parsed_blocks += 1


# ── helpers ───────────────────────────────────────────────────────────────────

def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


# ── display ───────────────────────────────────────────────────────────────────

import datetime

COL_W = 48
JOB_W = 10
ROW_W = 12
M_W   = 7
HDR = (f"{'Variant':<{COL_W}}  {'JobID':<{JOB_W}}  {'Seeds':>5}  {'Row':<{ROW_W}}"
       f"  {'FPR@95':>{M_W}}  {'AUROC':>{M_W}}  {'AUPR_IN':>{M_W}}  {'AUPR_OUT':>{M_W}}  {'ACC':>{M_W}}"
       f"  {'Time/seed':>9}")
SEP = "-" * len(HDR)

lines_out = []
lines_out.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def emit(s=""):
    print(s)
    lines_out.append(s)

for dataset in DATASETS:
    # Collect all (variant, record) pairs for this dataset, sorted by nearood AUROC desc
    entries = []
    for variant, ds_dict in sorted(raw.items()):
        if dataset not in ds_dict:
            continue
        for rec in ds_dict[dataset]:
            entries.append((variant, rec))

    if not entries:
        continue

    # Sort by nearood AUROC descending within each variant, then group by variant
    # First build best-nearood per variant for sorting
    best = {}
    for variant, rec in entries:
        auroc = rec["rows"].get("nearood", {}).get("AUROC", 0)
        if auroc > best.get(variant, 0):
            best[variant] = auroc

    # Sort entries: primary = variant rank (by best nearood), secondary = job_id
    variant_rank = {v: -s for v, s in sorted(best.items(), key=lambda x: x[1], reverse=True)}
    entries.sort(key=lambda e: (variant_rank[e[0]], e[1]["job_id"]))

    emit(f"\n{'='*len(HDR)}")
    emit(f"  DATASET: {dataset.upper()}  (one block per job file, averaged over seeds within file)")
    emit(f"{'='*len(HDR)}")
    emit(HDR)
    emit(SEP)

    prev_variant = None
    for variant, rec in entries:
        wt = fmt_duration(rec["walltime"]) if rec["walltime"] else "       --"
        job_id = rec["job_id"]
        n = rec["n_seeds"]
        first_row = True
        for row_name in rec["row_order"]:
            r = rec["rows"].get(row_name)
            if r is None:
                continue
            if first_row:
                var_col  = f"{variant:<{COL_W}}" if variant != prev_variant else f"{'':>{COL_W}}"
                job_col  = f"{job_id:<{JOB_W}}"
                seed_col = f"{n:>5}"
                wt_col   = f"  {wt:>9}"
                first_row = False
            else:
                var_col  = f"{'':>{COL_W}}"
                job_col  = f"{'':>{JOB_W}}"
                seed_col = f"{'':>5}"
                wt_col   = ""
            emit(f"{var_col}  {job_col}  {seed_col}  {row_name:<{ROW_W}}"
                 f"  {r['FPR@95']:>{M_W}.2f}  {r['AUROC']:>{M_W}.2f}"
                 f"  {r['AUPR_IN']:>{M_W}.2f}  {r['AUPR_OUT']:>{M_W}.2f}  {r['ACC']:>{M_W}.2f}"
                 + wt_col)
        emit()
        prev_variant = variant

    # TOP-10 by Near and Far AUROC (per file entry)
    near_sorted = sorted(entries, key=lambda e: e[1]["rows"].get("nearood", {}).get("AUROC", 0), reverse=True)
    far_sorted  = sorted(entries, key=lambda e: e[1]["rows"].get("farood",  {}).get("AUROC", 0), reverse=True)

    emit(f"  TOP-10 by Near AUROC:")
    for i, (variant, rec) in enumerate(near_sorted[:10], 1):
        near = rec["rows"].get("nearood", {}).get("AUROC", float("nan"))
        far  = rec["rows"].get("farood",  {}).get("AUROC", float("nan"))
        emit(f"    {i}. [{rec['job_id']}] {variant:<46} Near={near:.2f}  Far={far:.2f}")

    emit(f"\n  TOP-10 by Far AUROC:")
    for i, (variant, rec) in enumerate(far_sorted[:10], 1):
        near = rec["rows"].get("nearood", {}).get("AUROC", float("nan"))
        far  = rec["rows"].get("farood",  {}).get("AUROC", float("nan"))
        emit(f"    {i}. [{rec['job_id']}] {variant:<46} Near={near:.2f}  Far={far:.2f}")
    emit()


# ── diagnostics ───────────────────────────────────────────────────────────────
emit(f"\n{'='*70}")
emit(f"  DIAGNOSTICS")
emit(f"{'='*70}")
emit(f"  .out files read     : {len(out_files)}")
emit(f"  Total blocks found  : {total_blocks}")
emit(f"  Blocks parsed OK    : {parsed_blocks}")
emit(f"  Blocks w/ no table  : {len(parse_errors)}")
if parse_errors:
    emit("  -- Failed blocks --")
    for e in parse_errors:
        emit(e)

all_variants = sorted(raw.keys())
emit(f"\n  Unique variants: {len(all_variants)}")
for v in all_variants:
    ds_info = ", ".join(
        f"{ds}({sum(r['n_seeds'] for r in raw[v][ds])}s/{len(raw[v][ds])}runs)"
        for ds in DATASETS if ds in raw[v]
    )
    emit(f"    {v}: {ds_info}")

# ── write file ────────────────────────────────────────────────────────────────
with open(OUT_FILE, "w") as fh:
    fh.write("\n".join(lines_out) + "\n")
print(f"\nSaved to {OUT_FILE}")
