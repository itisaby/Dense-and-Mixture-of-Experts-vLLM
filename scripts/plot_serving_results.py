#!/usr/bin/env python3
"""Validate, summarize, and plot raw dense-vs-MoE serving measurements."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_ORDER = ["small_dense", "moe", "large_dense"]
MODEL_LABELS = {
    "small_dense": "Dense 1B",
    "moe": "MoE 1B active / 7B total",
    "large_dense": "Dense 7B",
}
COLORS = {
    "small_dense": "#0072B2",
    "moe": "#D55E00",
    "large_dense": "#009E73",
}
MARKERS = {"small_dense": "o", "moe": "s", "large_dense": "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/serving_raw.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/serving_summary.csv"))
    parser.add_argument("--latex-table", type=Path, default=Path("results/summary_table.tex"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def validate(df: pd.DataFrame) -> None:
    if df.empty:
        raise SystemExit("raw CSV has no measurements")
    required = {
        "model_key",
        "workload",
        "repeat",
        "concurrency",
        "target_prompt_tokens_per_request",
        "prompt_tokens_per_second",
        "output_tokens_per_second",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"raw CSV is missing columns: {', '.join(missing)}")
    models = set(df["model_key"])
    if models != set(MODEL_ORDER):
        raise SystemExit(f"expected exactly {MODEL_ORDER}; found {sorted(models)}")
    workloads = set(df["workload"])
    if workloads != {"prefill", "decode"}:
        raise SystemExit(f"expected prefill and decode rows; found {sorted(workloads)}")

    fixed_columns = [
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
    changed = [column for column in fixed_columns if column in df and df[column].nunique(dropna=False) != 1]
    if changed:
        raise SystemExit("incomparable fixed settings in raw CSV: " + ", ".join(changed))
    if set(df["weight_quantization"].astype(str)) != {"fp8"}:
        raise SystemExit("raw rows do not all use FP8 weights")
    if set(df["kv_cache_dtype"].astype(str)) != {"fp8"}:
        raise SystemExit("raw rows do not all use an FP8 KV cache")

    key_columns = [
        "model_key",
        "workload",
        "target_prompt_tokens_per_request",
        "target_output_tokens_per_request",
        "concurrency",
        "repeat",
    ]
    if df.duplicated(key_columns).any():
        raise SystemExit("raw CSV contains duplicate measurement keys")
    if "model_revision" in df:
        changed_revisions = [
            key
            for key, count in df.groupby("model_key")["model_revision"].nunique(dropna=False).items()
            if count != 1
        ]
        if changed_revisions:
            raise SystemExit(
                "multiple checkpoint revisions used for: " + ", ".join(changed_revisions)
            )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "model_key",
        "model_id",
        "architecture",
        "role",
        "workload",
        "concurrency",
        "target_prompt_tokens_per_request",
        "target_output_tokens_per_request",
    ]
    metric_columns = [
        "elapsed_seconds",
        "prompt_tokens_per_second",
        "output_tokens_per_second",
        "total_tokens_per_second",
        "mean_request_latency_seconds",
        "p95_request_latency_seconds",
    ]
    grouped = df.groupby(group_columns, as_index=False)[metric_columns].agg(["mean", "std", "count"])
    grouped.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in grouped.columns
    ]
    return grouped.sort_values(["workload", "model_key", "target_prompt_tokens_per_request", "concurrency"])


def plot_workload(summary: pd.DataFrame, workload: str, output_base: Path) -> None:
    subset = summary[summary["workload"] == workload].copy()
    if workload == "prefill":
        x_column = "target_prompt_tokens_per_request"
        y_column = "prompt_tokens_per_second_mean"
        err_column = "prompt_tokens_per_second_std"
        x_label = "Input context length (tokens/request)"
        y_label = "Aggregate prefill throughput (prompt tokens/s)"
        title = "Prefill-dominated serving"
    else:
        x_column = "concurrency"
        y_column = "output_tokens_per_second_mean"
        err_column = "output_tokens_per_second_std"
        x_label = "Parallel generations"
        y_label = "Aggregate decode throughput (output tokens/s)"
        title = "Decode-dominated serving"

    fig, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    for key in MODEL_ORDER:
        model_rows = subset[subset["model_key"] == key].sort_values(x_column)
        axis.errorbar(
            model_rows[x_column],
            model_rows[y_column],
            yerr=model_rows[err_column].fillna(0),
            label=MODEL_LABELS[key],
            color=COLORS[key],
            marker=MARKERS[key],
            markersize=6,
            linewidth=2,
            capsize=3,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(sorted(subset[x_column].unique()))
    axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(True, which="major", alpha=0.25)
    axis.legend(frameon=False)
    for suffix in ("pdf", "png"):
        fig.savefig(output_base.with_suffix(f".{suffix}"), dpi=200)
    plt.close(fig)


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    rows: list[str] = []
    for _, row in summary.iterrows():
        if row["workload"] == "prefill":
            setting = f"$L={int(row['target_prompt_tokens_per_request'])}$"
            throughput = float(row["prompt_tokens_per_second_mean"])
            deviation = float(row["prompt_tokens_per_second_std"] or 0)
        else:
            setting = f"$C={int(row['concurrency'])}$"
            throughput = float(row["output_tokens_per_second_mean"])
            deviation = float(row["output_tokens_per_second_std"] or 0)
        rows.append(
            f"{latex_escape(str(row['workload']).capitalize())} & "
            f"{latex_escape(MODEL_LABELS[str(row['model_key'])])} & {setting} & "
            f"{throughput:.1f} $\\pm$ {deviation:.1f} \\\\"
        )
    content = "\n".join(
        [
            r"\begin{tabular}{llrr}",
            r"\toprule",
            r"Workload & Model & Setting & Mean tokens/s \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    validate(df)
    summary = summarize(df)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    plot_workload(summary, "prefill", args.figure_dir / "prefill_throughput")
    plot_workload(summary, "decode", args.figure_dir / "decode_throughput")
    write_latex_table(summary, args.latex_table)
    print(f"Wrote {args.summary}")
    print(f"Wrote {args.latex_table}")
    print(f"Wrote figures to {args.figure_dir}")


if __name__ == "__main__":
    main()
