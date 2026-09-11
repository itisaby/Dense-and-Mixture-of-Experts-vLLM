#!/usr/bin/env python3
"""Benchmark dense and MoE checkpoints through a vLLM HTTP server.

The script launches one model at a time with FP8 weights and an FP8 KV cache,
sends synchronized concurrent requests to the OpenAI-compatible Completions
endpoint, and appends every measured repetition to a raw CSV immediately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    architecture: str
    role: str


MODELS = {
    "small_dense": ModelSpec(
        "small_dense",
        "allenai/OLMo-2-0425-1B-Instruct",
        "dense",
        "small dense baseline",
    ),
    "moe": ModelSpec(
        "moe",
        "allenai/OLMoE-1B-7B-0924-Instruct",
        "moe",
        "~1B active / 7B total",
    ),
    "large_dense": ModelSpec(
        "large_dense",
        "allenai/OLMo-2-1124-7B",
        "dense",
        "large dense baseline",
    ),
}

RAW_FIELDS = [
    "timestamp_utc",
    "model_key",
    "model_id",
    "model_revision",
    "architecture",
    "role",
    "workload",
    "repeat",
    "num_requests",
    "concurrency",
    "target_prompt_tokens_per_request",
    "target_output_tokens_per_request",
    "actual_prompt_tokens",
    "actual_output_tokens",
    "elapsed_seconds",
    "prompt_tokens_per_second",
    "output_tokens_per_second",
    "total_tokens_per_second",
    "mean_request_latency_seconds",
    "p95_request_latency_seconds",
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

SOURCE_TEXT = """
Serving a language model has two distinct phases. During prefill, the model
processes the input context in parallel and constructs attention keys and
values. During decode, it generates one new token per active sequence at each
step. Dense networks use every feed-forward parameter for every token. A
mixture-of-experts network instead routes each token to a small subset of its
experts. This reduces arithmetic per token, although all expert weights still
occupy accelerator memory and a diverse batch can access many experts. A fair
throughput experiment controls the accelerator, numerical formats, scheduler,
prompt lengths, output lengths, concurrency, and warm-up procedure. Repeated
wall-clock measurements expose both central tendency and run-to-run variation.
""".strip()


def parse_int_list(value: str) -> list[int]:
    values = [int(part) for part in value.split(",") if part.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all", *MODELS],
        help="models to run; 'all' runs the three checkpoints sequentially",
    )
    parser.add_argument("--output", type=Path, default=Path("results/serving_raw.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("results/serving_metadata.json"))
    parser.add_argument("--log-dir", type=Path, default=Path("results/logs"))
    parser.add_argument("--prefill-lengths", type=parse_int_list, default=parse_int_list("128,512,1024,2048,3072"))
    parser.add_argument("--prefill-concurrency", type=int, default=8)
    parser.add_argument("--prefill-output-tokens", type=int, default=1)
    parser.add_argument("--decode-concurrencies", type=parse_int_list, default=parse_int_list("1,2,4,8,16,32,64"))
    parser.add_argument("--decode-input-tokens", type=int, default=32)
    parser.add_argument("--decode-output-tokens", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20250911)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing raw CSV")
    parser.add_argument("--resume", action="store_true", help="skip measurement keys already present in the raw CSV")
    parser.add_argument("--allow-nonnative-fp8", action="store_true", help="allow GPUs below compute capability 8.9")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and server commands without using a GPU")
    return parser.parse_args()


def selected_models(values: list[str]) -> list[ModelSpec]:
    if "all" in values:
        if len(values) != 1:
            raise SystemExit("--models all cannot be combined with individual model keys")
        return list(MODELS.values())
    return [MODELS[value] for value in values]


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "prefill concurrency": args.prefill_concurrency,
        "prefill output tokens": args.prefill_output_tokens,
        "decode input tokens": args.decode_input_tokens,
        "decode output tokens": args.decode_output_tokens,
        "repetitions": args.repetitions,
        "max model length": args.max_model_len,
    }
    for label, value in positive.items():
        if value <= 0:
            raise SystemExit(f"{label} must be positive")
    if args.warmups < 0:
        raise SystemExit("warmups cannot be negative")
    if max(args.prefill_lengths) + args.prefill_output_tokens > args.max_model_len:
        raise SystemExit("largest prefill request exceeds --max-model-len")
    if args.decode_input_tokens + args.decode_output_tokens > args.max_model_len:
        raise SystemExit("decode request exceeds --max-model-len")
    if max(args.decode_concurrencies) > args.max_num_seqs:
        raise SystemExit("decode concurrency exceeds --max-num-seqs")
    if args.prefill_concurrency > args.max_num_seqs:
        raise SystemExit("prefill concurrency exceeds --max-num-seqs")
    if args.prefill_concurrency * max(args.prefill_lengths) > args.max_num_batched_tokens:
        raise SystemExit("prefill batch exceeds --max-num-batched-tokens")
    if not 0 < args.gpu_memory_utilization < 1:
        raise SystemExit("--gpu-memory-utilization must be between 0 and 1")
    if args.overwrite and args.resume:
        raise SystemExit("choose either --overwrite or --resume")


def package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def gpu_and_software_info(allow_nonnative: bool) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch/vLLM is not installed; run pip install -r requirements.txt") from exc

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Select a Colab NVIDIA GPU runtime.")
    capability = torch.cuda.get_device_capability(0)
    if capability < (8, 9) and not allow_nonnative:
        raise SystemExit(
            f"GPU compute capability {capability[0]}.{capability[1]} lacks native FP8 tensor cores. "
            "Use an L4/H100-class runtime, or pass --allow-nonnative-fp8 only if the limitation is disclosed."
        )

    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    gpu_name, memory_mib, driver = [part.strip() for part in query.split(",", 2)]
    return {
        "gpu_name": gpu_name,
        "gpu_memory_mib": int(memory_mib),
        "gpu_driver_version": driver,
        "gpu_compute_capability": f"{capability[0]}.{capability[1]}",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": str(torch.version.cuda),
        "vllm_version": package_version("vllm"),
    }


def resolve_revision(model_id: str) -> str:
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id).sha
    except Exception as exc:
        print(f"Warning: could not resolve revision for {model_id}: {exc}", file=sys.stderr)
        return "unresolved"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def server_command(spec: ModelSpec, revision: str, port: int, args: argparse.Namespace) -> list[str]:
    executable = shutil.which("vllm") or "vllm"
    command = [
        executable,
        "serve",
        spec.model_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        spec.model_id,
        "--quantization",
        "fp8",
        "--kv-cache-dtype",
        "fp8",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--disable-log-requests",
    ]
    if revision != "unresolved":
        command.extend(["--revision", revision, "--tokenizer-revision", revision])
    return command


class VLLMServer:
    def __init__(self, command: list[str], base_url: str, log_path: Path, startup_timeout: int):
        self.command = command
        self.base_url = base_url
        self.log_path = log_path
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen[str] | None = None
        self.log_handle: Any = None

    def __enter__(self) -> "VLLMServer":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        print("Starting:", " ".join(self.command), flush=True)
        self.process = subprocess.Popen(
            self.command,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env={**os.environ, "VLLM_LOGGING_LEVEL": "INFO"},
        )
        deadline = time.monotonic() + self.startup_timeout
        last_error = "server did not answer"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited with code {self.process.returncode}; inspect {self.log_path}"
                )
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as response:
                    if response.status == 200:
                        print("Server is ready.", flush=True)
                        return self
            except Exception as exc:
                last_error = str(exc)
            time.sleep(2)
        raise TimeoutError(f"vLLM did not become healthy: {last_error}; inspect {self.log_path}")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        if self.log_handle is not None:
            self.log_handle.close()
        time.sleep(5)


def load_tokenizer(model_id: str, revision: str):
    from transformers import AutoTokenizer

    kwargs = {"revision": revision} if revision != "unresolved" else {}
    return AutoTokenizer.from_pretrained(model_id, **kwargs)


def source_token_ids(tokenizer: Any) -> list[int]:
    token_ids = tokenizer.encode(SOURCE_TEXT, add_special_tokens=False)
    special_ids = set(tokenizer.all_special_ids)
    token_ids = [token_id for token_id in token_ids if token_id not in special_ids]
    if not token_ids:
        raise RuntimeError("tokenizer produced no usable source tokens")
    return token_ids


def exact_prompt(source_ids: list[int], length: int, salt: str) -> list[int]:
    digest = hashlib.sha256(salt.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big") % len(source_ids)
    rotated = source_ids[offset:] + source_ids[:offset]
    repeats = (length + len(rotated) - 1) // len(rotated)
    result = (rotated * repeats)[:length]
    if len(result) != length:
        raise AssertionError("failed to create exact-length prompt")
    return result


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((percentile * len(ordered) + 0.999999)) - 1))
    return ordered[index]


def post_completion(
    base_url: str,
    model_id: str,
    prompt_ids: list[int],
    output_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model_id,
            "prompt": prompt_ids,
            "max_tokens": output_tokens,
            "min_tokens": output_tokens,
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM returned HTTP {exc.code}: {detail}") from exc
    latency = time.perf_counter() - started
    usage = body.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "output_tokens": int(usage.get("completion_tokens", 0)),
        "latency": latency,
    }


def synchronized_batch(
    base_url: str,
    model_id: str,
    prompts: list[list[int]],
    output_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    concurrency = len(prompts)
    barrier = threading.Barrier(concurrency + 1)

    def worker(prompt: list[int]) -> dict[str, Any]:
        barrier.wait()
        return post_completion(base_url, model_id, prompt, output_tokens, timeout)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, prompt) for prompt in prompts]
        started = time.perf_counter()
        barrier.wait()
        responses = [future.result() for future in futures]
        elapsed = time.perf_counter() - started

    prompt_count = sum(item["prompt_tokens"] for item in responses)
    output_count = sum(item["output_tokens"] for item in responses)
    expected_prompt = sum(len(prompt) for prompt in prompts)
    expected_output = concurrency * output_tokens
    if prompt_count != expected_prompt:
        raise RuntimeError(f"server counted {prompt_count} prompt tokens; expected {expected_prompt}")
    if output_count != expected_output:
        raise RuntimeError(f"server returned {output_count} output tokens; expected {expected_output}")
    latencies = [float(item["latency"]) for item in responses]
    return {
        "actual_prompt_tokens": prompt_count,
        "actual_output_tokens": output_count,
        "elapsed_seconds": elapsed,
        "prompt_tokens_per_second": prompt_count / elapsed,
        "output_tokens_per_second": output_count / elapsed,
        "total_tokens_per_second": (prompt_count + output_count) / elapsed,
        "mean_request_latency_seconds": mean(latencies),
        "p95_request_latency_seconds": percentile_nearest_rank(latencies, 0.95),
    }


def measurement_key(row: dict[str, Any]) -> tuple[str, str, int, int, int]:
    return (
        str(row["model_key"]),
        str(row["workload"]),
        int(row["target_prompt_tokens_per_request"]),
        int(row["concurrency"]),
        int(row["repeat"]),
    )


def existing_keys(path: Path) -> set[tuple[str, str, int, int, int]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RAW_FIELDS:
            raise SystemExit(f"existing CSV schema does not match this runner: {path}")
        return {measurement_key(row) for row in reader}


def prepare_output(path: Path, overwrite: bool, resume: bool) -> set[tuple[str, str, int, int, int]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and overwrite:
        path.unlink()
    elif path.exists() and not resume:
        raise SystemExit(f"{path} already exists; use --resume or --overwrite")
    keys = existing_keys(path) if resume else set()
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=RAW_FIELDS).writeheader()
    return keys


def append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writerow({field: row[field] for field in RAW_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def base_row(
    spec: ModelSpec,
    revision: str,
    workload: str,
    repeat: int,
    concurrency: int,
    input_tokens: int,
    output_tokens: int,
    info: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_key": spec.key,
        "model_id": spec.model_id,
        "model_revision": revision,
        "architecture": spec.architecture,
        "role": spec.role,
        "workload": workload,
        "repeat": repeat,
        "num_requests": concurrency,
        "concurrency": concurrency,
        "target_prompt_tokens_per_request": input_tokens,
        "target_output_tokens_per_request": output_tokens,
        **info,
        "weight_quantization": "fp8",
        "kv_cache_dtype": "fp8",
        "tensor_parallel_size": 1,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "seed": args.seed,
    }


def run_condition(
    spec: ModelSpec,
    revision: str,
    workload: str,
    concurrency: int,
    input_tokens: int,
    output_tokens: int,
    source_ids: list[int],
    base_url: str,
    info: dict[str, Any],
    args: argparse.Namespace,
    completed: set[tuple[str, str, int, int, int]],
) -> None:
    total_rounds = args.warmups + args.repetitions
    for round_index in range(total_rounds):
        measured_repeat = round_index - args.warmups
        is_warmup = round_index < args.warmups
        prompts = [
            exact_prompt(
                source_ids,
                input_tokens,
                f"{args.seed}:{spec.key}:{workload}:{input_tokens}:{concurrency}:{round_index}:{request_index}",
            )
            for request_index in range(concurrency)
        ]
        if not is_warmup:
            candidate = base_row(
                spec,
                revision,
                workload,
                measured_repeat,
                concurrency,
                input_tokens,
                output_tokens,
                info,
                args,
            )
            key = measurement_key(candidate)
            if key in completed:
                print(f"Skipping completed measurement {key}", flush=True)
                continue
        label = "warmup" if is_warmup else f"repeat {measured_repeat + 1}/{args.repetitions}"
        print(
            f"{spec.key}: {workload}, input={input_tokens}, output={output_tokens}, "
            f"concurrency={concurrency}, {label}",
            flush=True,
        )
        metrics = synchronized_batch(
            base_url,
            spec.model_id,
            prompts,
            output_tokens,
            args.request_timeout,
        )
        if not is_warmup:
            row = {**candidate, **metrics}
            append_row(args.output, row)
            completed.add(measurement_key(row))


def write_metadata(
    path: Path,
    specs: list[ModelSpec],
    revisions: dict[str, str],
    info: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": [{**asdict(spec), "revision": revisions[spec.key]} for spec in specs],
        "hardware_and_software": info,
        "fixed_server_settings": {
            "weight_quantization": "fp8",
            "kv_cache_dtype": "fp8",
            "tensor_parallel_size": 1,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
        },
        "prefill": {
            "input_lengths": args.prefill_lengths,
            "concurrency": args.prefill_concurrency,
            "output_tokens_per_request": args.prefill_output_tokens,
        },
        "decode": {
            "input_tokens_per_request": args.decode_input_tokens,
            "output_tokens_per_request": args.decode_output_tokens,
            "concurrencies": args.decode_concurrencies,
        },
        "warmups": args.warmups,
        "recorded_repetitions": args.repetitions,
        "seed": args.seed,
        "throughput_definition": {
            "prefill": "sum of prompt tokens divided by batch wall-clock time",
            "decode": "sum of output tokens divided by batch wall-clock time",
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def print_plan(specs: Iterable[ModelSpec], args: argparse.Namespace) -> None:
    print("Models:", ", ".join(spec.key for spec in specs))
    print("Prefill lengths:", args.prefill_lengths)
    print("Decode concurrencies:", args.decode_concurrencies)
    for index, spec in enumerate(specs):
        print(" ".join(server_command(spec, "REVISION_SHA", 8000 + index, args)))


def main() -> None:
    args = parse_args()
    specs = selected_models(args.models)
    validate_args(args)
    if args.dry_run:
        print_plan(specs, args)
        return

    completed = prepare_output(args.output, args.overwrite, args.resume)
    info = gpu_and_software_info(args.allow_nonnative_fp8)
    revisions = {spec.key: resolve_revision(spec.model_id) for spec in specs}
    write_metadata(args.metadata, specs, revisions, info, args)

    for spec in specs:
        revision = revisions[spec.key]
        tokenizer = load_tokenizer(spec.model_id, revision)
        source_ids = source_token_ids(tokenizer)
        port = free_port()
        base_url = f"http://127.0.0.1:{port}"
        command = server_command(spec, revision, port, args)
        log_path = args.log_dir / f"{spec.key}.log"
        with VLLMServer(command, base_url, log_path, args.startup_timeout):
            for length in args.prefill_lengths:
                run_condition(
                    spec,
                    revision,
                    "prefill",
                    args.prefill_concurrency,
                    length,
                    args.prefill_output_tokens,
                    source_ids,
                    base_url,
                    info,
                    args,
                    completed,
                )
            for concurrency in args.decode_concurrencies:
                run_condition(
                    spec,
                    revision,
                    "decode",
                    concurrency,
                    args.decode_input_tokens,
                    args.decode_output_tokens,
                    source_ids,
                    base_url,
                    info,
                    args,
                    completed,
                )
        del tokenizer

    print(f"Raw measurements written to {args.output}")
    print(f"Run: {sys.executable} scripts/plot_serving_results.py --input {args.output}")


if __name__ == "__main__":
    main()

