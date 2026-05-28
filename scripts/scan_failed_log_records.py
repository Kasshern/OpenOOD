#!/usr/bin/env python3
"""Scan OpenOOD .out/.err logs and list failed run records.

The scanner is intended to run on either a local checkout or the remote cluster.
It reads the log files that are actually present in --logs-dir and writes one
failed record base name per line when --write-list is provided.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


DIAGNOSTIC_PREFIXES = (
    "layer_probe_diagnostic_",
    "id_layer_weight_diagnostic_",
)

FATAL_PATTERNS = (
    ("disk_quota", re.compile(r"Disk quota exceeded", re.IGNORECASE)),
    (
        "oom_or_killed",
        re.compile(
            r"CUDA out of memory|OutOfMemoryError|oom_kill|OOM Killed|\bKilled\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cancelled_or_timed_out",
        re.compile(r"CANCELLED|DUE TO TIME LIMIT|TIMEOUT", re.IGNORECASE),
    ),
    (
        "missing_clip_dependency",
        re.compile(r"No module named ['\"]clip['\"]", re.IGNORECASE),
    ),
    (
        "no_gpu_or_cpu_only",
        re.compile(
            r"Found no NVIDIA driver|torch\.cuda\.is_available\(\) is False",
            re.IGNORECASE,
        ),
    ),
    (
        "shape_mismatch",
        re.compile(r"mat1 and mat2 shapes cannot be multiplied", re.IGNORECASE),
    ),
    (
        "none_config_type_error",
        re.compile(
            r"NoneType|float\(\) argument must be a string or a real number",
            re.IGNORECASE,
        ),
    ),
    (
        "missing_or_inaccessible_data",
        re.compile(r"No subfolders found|Permission denied", re.IGNORECASE),
    ),
    (
        "view_vs_reshape_bug",
        re.compile(r"view size is not compatible", re.IGNORECASE),
    ),
    (
        "missing_diag_attribute",
        re.compile(r"_diag_by_dataset", re.IGNORECASE),
    ),
    (
        "dual_gated_type_error",
        re.compile(
            r"unsupported operand type\(s\) for /: 'Tensor' and 'str'",
            re.IGNORECASE,
        ),
    ),
    (
        "traceback_or_exception",
        re.compile(
            r"Traceback|RuntimeError:|TypeError:|ValueError:|AttributeError:"
            r"|ModuleNotFoundError|ImportError|OSError:|PermissionError:"
            r"|\] error:",
            re.IGNORECASE,
        ),
    ),
)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def strip_nonfatal_noise(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if "CondaError: Run" not in line
    )


def is_diagnostic_record(record: str) -> bool:
    return record.startswith(DIAGNOSTIC_PREFIXES)


def classify_record(logs_dir: Path, record: str, include_no_metrics: bool) -> list[str]:
    out_path = logs_dir / f"{record}.out"
    err_path = logs_dir / f"{record}.err"
    out_text = read_text(out_path)
    err_text = read_text(err_path)
    combined_text = strip_nonfatal_noise(f"{err_text}\n{out_text}")
    reasons = []

    if not out_path.exists():
        reasons.append("missing_out")
    if not err_path.exists():
        reasons.append("missing_err")
    if out_path.exists() and out_path.stat().st_size == 0:
        reasons.append("empty_out")
    if err_path.exists() and err_path.stat().st_size == 0:
        reasons.append("empty_err")
    if out_path.exists() and "=== Done" not in out_text:
        reasons.append("missing_done_marker")

    if include_no_metrics and out_path.exists() and not is_diagnostic_record(record):
        out_lower = out_text.lower()
        if "nearood" not in out_lower or "farood" not in out_lower:
            reasons.append("missing_ood_metrics")

    for reason, pattern in FATAL_PATTERNS:
        if pattern.search(combined_text):
            reasons.append(reason)

    return list(dict.fromkeys(reasons))


def discover_records(logs_dir: Path) -> list[str]:
    records = {
        path.stem
        for path in logs_dir.iterdir()
        if path.is_file() and path.suffix in {".out", ".err"}
    }
    return sorted(records)


def write_record_list(path: Path, records: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(records) + ("\n" if records else ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan OpenOOD logs and list records that failed or lack OOD metrics."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing .out/.err log files. Default: logs",
    )
    parser.add_argument(
        "--write-list",
        type=Path,
        help="Optional path to write failed record base names, one per line.",
    )
    parser.add_argument(
        "--keep-no-metrics",
        action="store_true",
        help="Do not flag non-diagnostic records solely for missing nearood/farood rows.",
    )
    args = parser.parse_args()

    logs_dir = args.logs_dir.expanduser().resolve()
    include_no_metrics = not args.keep_no_metrics
    if not logs_dir.exists():
        raise SystemExit(f"logs directory does not exist: {logs_dir}")

    records_by_reason = defaultdict(list)
    failed_records = []

    records = discover_records(logs_dir)
    for record in records:
        reasons = classify_record(logs_dir, record, include_no_metrics)
        if not reasons:
            continue
        failed_records.append(record)
        for reason in reasons:
            records_by_reason[reason].append(record)

    print(f"Scanned records: {len(records)}")
    print(f"Failed records: {len(failed_records)}")

    for reason in sorted(records_by_reason):
        records = records_by_reason[reason]
        print(f"\n[{reason}] count={len(records)}")
        for record in records:
            print(f"  {record}")

    if args.write_list:
        output_path = args.write_list.expanduser().resolve()
        write_record_list(output_path, failed_records)
        print(f"\nWrote failed record list: {output_path}")


if __name__ == "__main__":
    main()
