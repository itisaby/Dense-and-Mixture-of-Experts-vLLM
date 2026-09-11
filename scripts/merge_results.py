#!/usr/bin/env python3
"""Merge per-model raw CSVs while enforcing experiment comparability."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FIXED_COLUMNS = [
    "gpu_name",
    "gpu_memory_mib",
    "gpu_driver_version",
    "gpu_compute_capability",
    "python_version",
    "torch_version",
    "cuda_version",
    "vllm_version",
    "weight_quantization",
    "kv_cache_dtype",
    "tensor_parallel_size",
    "max_model_len",
    "gpu_memory_utilization",
    "max_num_seqs",
    "max_num_batched_tokens",
    "seed",
]
KEY_COLUMNS = [
    "model_key",
    "workload",
    "target_prompt_tokens_per_request",
    "target_output_tokens_per_request",
    "concurrency",
    "repeat",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-environment-mismatch",
        action="store_true",
        help="merge despite hardware/software differences (must be disclosed)",
    )
    args = parser.parse_args()

    frames = [pd.read_csv(path) for path in args.inputs]
    columns = [list(frame.columns) for frame in frames]
    if any(item != columns[0] for item in columns[1:]):
        raise SystemExit("input CSV schemas differ")
    merged = pd.concat(frames, ignore_index=True)
    if merged.duplicated(KEY_COLUMNS).any():
        duplicates = merged[merged.duplicated(KEY_COLUMNS, keep=False)][KEY_COLUMNS]
        raise SystemExit("duplicate measurement keys:\n" + duplicates.to_string(index=False))
    changed = [column for column in FIXED_COLUMNS if merged[column].nunique(dropna=False) != 1]
    if changed and not args.allow_environment_mismatch:
        raise SystemExit("environment/settings mismatch: " + ", ".join(changed))
    merged = merged.sort_values(KEY_COLUMNS)
    changed_revisions = [
        key
        for key, count in merged.groupby("model_key")["model_revision"].nunique(dropna=False).items()
        if count != 1
    ]
    if changed_revisions:
        raise SystemExit(
            "multiple checkpoint revisions used for: " + ", ".join(changed_revisions)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Merged {len(merged)} rows into {args.output}")
    if changed:
        print("WARNING: mismatched columns retained: " + ", ".join(changed))


if __name__ == "__main__":
    main()
