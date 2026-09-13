# Empirical GPU experiments

This repository contains two controlled experiments: dense-versus-MoE serving
with vLLM and supervised fine-tuning with LoRA.

## Dense vs. Mixture-of-Experts serving

This repository contains a single-GPU experiment comparing:

| Key | Checkpoint | Architecture |
|---|---|---|
| `small_dense` | `allenai/OLMo-2-0425-1B-Instruct` | dense, about 1B parameters |
| `moe` | `allenai/OLMoE-1B-7B-0924-Instruct` | MoE, about 1B active / 7B total parameters |
| `large_dense` | `allenai/OLMo-2-1124-7B` | dense, about 7B parameters |

Every server is launched with FP8 weight quantization, an FP8 KV cache,
tensor parallelism 1, the same 4096-token model-length cap, the same scheduler
limits, and the same GPU-memory utilization target. The controller starts one
OpenAI-compatible vLLM server at a time and records every measured repetition
to CSV immediately.

No benchmark values are checked into this initial scaffold. The files under
`results/` and `figures/` are created from an actual NVIDIA-GPU run; this avoids
mistaking example or synthetic numbers for empirical measurements.

## Recommended Colab workflow

Open [`benchmark_colab.ipynb`](benchmark_colab.ipynb) in Colab and choose an L4
or H100 runtime. An L4 (24 GB) is the practical minimum recommended here. The
runner checks for an NVIDIA GPU with native FP8 tensor-core support and exits
on older devices unless explicitly overridden.

The full experiment is:

```bash
python scripts/benchmark_serving.py --models all \
  --output results/serving_raw.csv --overwrite
python scripts/plot_serving_results.py \
  --input results/serving_raw.csv
```

The default design is:

- Prefill-dominated: input lengths `128,512,1024,2048,3072`, eight parallel
  requests, and one generated token.
- Decode-dominated: 32 input tokens, 256 generated tokens, and concurrency
  `1,2,4,8,16,32,64`.
- One warm-up and three recorded repetitions per condition.
- Exact pre-tokenized inputs sent to the vLLM Completions API. The text source
  is deterministic natural prose and each request uses a different rotation,
  avoiding accidental prefix-cache reuse while preserving input length.

OLMoE's native context limit is 4096 tokens, so the largest prefill input is
3072 rather than 4096. This leaves room for generation and keeps the same
`--max-model-len 4096` setting for all three checkpoints.

## Running models in separate Colab sessions

Running all checkpoints in one runtime gives the strongest control over
hardware and software. If a Colab session is interrupted, run one model at a
time and download each CSV:

```bash
python scripts/benchmark_serving.py --models small_dense \
  --output results/raw_small_dense.csv \
  --metadata results/metadata_small_dense.json \
  --log-dir results/logs/small_dense_session --overwrite
python scripts/benchmark_serving.py --models moe \
  --output results/raw_moe.csv \
  --metadata results/metadata_moe.json \
  --log-dir results/logs/moe_session --overwrite
python scripts/benchmark_serving.py --models large_dense \
  --output results/raw_large_dense.csv \
  --metadata results/metadata_large_dense.json \
  --log-dir results/logs/large_dense_session --overwrite
```

Put the three CSVs in the same checkout and merge them:

```bash
python scripts/merge_results.py results/raw_small_dense.csv \
  results/raw_moe.csv results/raw_large_dense.csv \
  --output results/serving_raw.csv
```

The merge step rejects duplicate measurements and, by default, rejects files
whose GPU name, vLLM version, CUDA version, or fixed benchmark settings differ.
Use different physical Colab runtimes only when they expose the same GPU model
and software stack, and disclose that limitation in the report.

## Outputs

After a complete run, the repository contains:

```text
results/serving_raw.csv          raw per-repetition measurements
results/serving_summary.csv      mean and standard deviation by condition
results/serving_metadata.json    hardware, software, revisions, and settings
results/summary_table.tex        compact LaTeX table of measured means
results/logs/*.log               vLLM startup/runtime logs
figures/prefill_throughput.pdf   report-ready prefill figure
figures/decode_throughput.pdf    report-ready decode figure
figures/*.png                    notebook previews
```

The experiment implementation requested by the assignment is
[`scripts/benchmark_serving.py`](scripts/benchmark_serving.py). The report
skeleton is [`latex/solution.tex`](latex/solution.tex); replace its clearly
marked interpretation placeholders only after inspecting the generated CSV
and figures.

## Measurement definitions

For a batch that finishes in wall-clock time \(t\):

```text
prefill throughput = sum(prompt tokens) / t
decode throughput  = sum(output tokens) / t
```

The plotted point is the mean across recorded repetitions; error bars are one
sample standard deviation. Aggregate throughput is used because the question
is about serving capacity, not single-request latency. The raw file also keeps
total-token throughput and request latency statistics for auditing.

## Supervised fine-tuning with LoRA

The second experiment compares LoRA ranks 1, 4, and 16 with full-weight
fine-tuning of `Qwen/Qwen2.5-0.5B`. Every variant uses the same deterministic
1,000-conversation `train_sft` subset from
`HuggingFaceH4/ultrachat_200k`, the same 100 held-out `test_sft`
conversations, and the same training hyperparameters and example order.
Selection takes the first usable examples after a seeded shuffle; conversations
whose assistant target begins beyond the 512-token limit are skipped and their
identifiers are recorded in the selection manifest.

Open [`sft_colab.ipynb`](sft_colab.ipynb) on one NVIDIA L4 runtime and run all
cells. The experiment is implemented in
[`scripts/sft_experiment.py`](scripts/sft_experiment.py):

```bash
python scripts/sft_experiment.py --variants all \
  --output-dir results/sft --overwrite
python scripts/plot_sft_results.py --input-dir results/sft
```

`requirements_sft.txt` intentionally does not reinstall PyTorch: Colab ships a
CUDA-matched build, and replacing that core package in a live kernel can leave
an inconsistent installation. Use a fresh runtime when switching from the
serving experiment to this SFT experiment.

The default matched setup uses two epochs, maximum length 512, BF16, effective
batch size 16, learning rate `5e-5`, AdamW, cosine decay, and validation every
10 optimizer steps. User and system tokens are masked from the loss. LoRA uses
PEFT's `target_modules="all-linear"`; `lora_alpha` is twice the rank so the
scaling ratio remains constant while rank changes.

Generated artifacts include `metrics.csv`, raw `loss_history.csv`, immutable
model/dataset revisions, selected conversation identifiers, unedited held-out
generations, report-ready LaTeX tables, and PDF/PNG figures. The report skeleton
is [`latex/sft_solution.tex`](latex/sft_solution.tex); its interpretation is
intentionally left open until a real GPU run is available.
