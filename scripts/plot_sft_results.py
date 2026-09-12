#!/usr/bin/env python3
"""Create report figures and a LaTeX table from the SFT outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ORDER = ["lora_r1", "lora_r4", "lora_r16", "full"]
LABELS = {
    "lora_r1": "LoRA r=1", "lora_r4": "LoRA r=4",
    "lora_r16": "LoRA r=16", "full": "Full tuning",
}
COLORS = {
    "lora_r1": "#0072B2", "lora_r4": "#E69F00",
    "lora_r16": "#D55E00", "full": "#009E73",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/sft"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def validate(metrics: pd.DataFrame, history: pd.DataFrame) -> None:
    if set(metrics["variant"]) != set(ORDER):
        raise SystemExit(f"expected {ORDER}; found {sorted(set(metrics['variant']))}")
    if metrics["variant"].duplicated().any():
        raise SystemExit("metrics.csv contains duplicate variants")
    if history.empty or not {"training", "validation"}.issubset(set(history["phase"])):
        raise SystemExit("loss_history.csv is incomplete")
    missing = set(ORDER) - set(history["variant"])
    if missing:
        raise SystemExit("missing loss histories for: " + ", ".join(sorted(missing)))


def plot_losses(history: pd.DataFrame, output_base: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    for variant in ORDER:
        rows = history[history["variant"] == variant]
        training = rows[rows["phase"] == "training"].sort_values("step")
        validation = rows[rows["phase"] == "validation"].sort_values("step")
        axes[0].plot(training["step"], training["loss"], label=LABELS[variant],
                     color=COLORS[variant], linewidth=1.8)
        axes[1].plot(validation["step"], validation["loss"], label=LABELS[variant],
                     color=COLORS[variant], marker="o", markersize=3.5, linewidth=1.8)
    axes[0].set_title("Training loss")
    axes[1].set_title("Held-out validation loss")
    for axis in axes:
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("Assistant-token cross-entropy loss")
        axis.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    for suffix in ("pdf", "png"):
        fig.savefig(output_base.with_suffix(f".{suffix}"), dpi=200)
    plt.close(fig)


def plot_resources(metrics: pd.DataFrame, output_base: Path) -> None:
    indexed = metrics.set_index("variant").loc[ORDER]
    x = list(range(len(ORDER)))
    colors = [COLORS[key] for key in ORDER]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), constrained_layout=True)
    axes[0].bar(x, indexed["trainable_parameters"], color=colors)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Trainable parameters (log scale)")
    axes[1].bar(x, indexed["peak_gpu_memory_mib"] / 1024, color=colors)
    axes[1].set_ylabel("Peak GPU memory (GiB)")
    axes[2].bar(x, indexed["training_time_seconds"] / 60, color=colors)
    axes[2].set_ylabel("Training time (minutes)")
    for axis in axes:
        axis.set_xticks(x, [LABELS[key] for key in ORDER], rotation=25, ha="right")
        axis.grid(True, axis="y", alpha=0.25)
    for suffix in ("pdf", "png"):
        fig.savefig(output_base.with_suffix(f".{suffix}"), dpi=200)
    plt.close(fig)


def write_latex_table(metrics: pd.DataFrame, path: Path) -> None:
    indexed = metrics.set_index("variant").loc[ORDER]
    lines = [
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Method & Trainable params & Peak GiB & Time (min) & Mean train loss & Final val. loss \\",
        r"\midrule",
    ]
    for variant, row in indexed.iterrows():
        lines.append(
            f"{LABELS[variant]} & {int(row['trainable_parameters']):,} & "
            f"{row['peak_gpu_memory_mib'] / 1024:.2f} & "
            f"{row['training_time_seconds'] / 60:.2f} & "
            f"{row['mean_training_loss']:.4f} & "
            f"{row['final_validation_loss']:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    metrics = pd.read_csv(args.input_dir / "metrics.csv")
    history = pd.read_csv(args.input_dir / "loss_history.csv")
    validate(metrics, history)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    plot_losses(history, args.figure_dir / "sft_loss_curves")
    plot_resources(metrics, args.figure_dir / "sft_resource_comparison")
    write_latex_table(metrics, args.input_dir / "metrics_table.tex")
    print(f"Wrote SFT figures to {args.figure_dir}")
    print(f"Wrote {args.input_dir / 'metrics_table.tex'}")


if __name__ == "__main__":
    main()
