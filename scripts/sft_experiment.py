#!/usr/bin/env python3
"""Run matched LoRA-rank and full-weight SFT experiments on UltraChat."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen2.5-0.5B"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
SYSTEM_MESSAGE = "You are a helpful, respectful, and concise assistant."


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    rank: int | None


VARIANTS = {
    "lora_r1": Variant("lora_r1", "LoRA rank 1", 1),
    "lora_r4": Variant("lora_r4", "LoRA rank 4", 4),
    "lora_r16": Variant("lora_r16", "LoRA rank 16", 16),
    "full": Variant("full", "Full-weight tuning", None),
}

METRIC_FIELDS = [
    "variant", "label", "lora_rank", "trainable_parameters",
    "total_parameters", "trainable_percent", "peak_gpu_memory_mib",
    "peak_gpu_memory_percent", "peak_torch_allocated_mib",
    "peak_torch_reserved_mib", "training_time_seconds",
    "trainer_reported_runtime_seconds", "optimizer_steps",
    "examples_per_second", "mean_training_loss",
    "initial_validation_loss", "final_validation_loss",
]
HISTORY_FIELDS = ["variant", "phase", "step", "epoch", "loss"]


def package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=["all", *VARIANTS], default=["all"])
    parser.add_argument("--output-dir", type=Path, default=Path("results/sft"))
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--eval-size", type=int, default=100)
    parser.add_argument("--qualitative-size", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--generation-max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20250912)
    parser.add_argument("--memory-poll-seconds", type=float, default=0.1)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_variants(values: list[str]) -> list[Variant]:
    if "all" in values:
        if len(values) != 1:
            raise SystemExit("--variants all cannot be combined with individual variants")
        return list(VARIANTS.values())
    return [VARIANTS[value] for value in values]


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "train size": args.train_size, "eval size": args.eval_size,
        "qualitative size": args.qualitative_size, "max length": args.max_length,
        "epochs": args.epochs, "train batch size": args.train_batch_size,
        "eval batch size": args.eval_batch_size,
        "gradient accumulation steps": args.gradient_accumulation_steps,
        "learning rate": args.learning_rate, "logging steps": args.logging_steps,
        "evaluation steps": args.eval_steps,
        "generation length": args.generation_max_new_tokens,
        "memory polling interval": args.memory_poll_seconds,
    }
    for label, value in positive.items():
        if value <= 0:
            raise SystemExit(f"{label} must be positive")
    if args.qualitative_size > args.eval_size:
        raise SystemExit("qualitative size cannot exceed evaluation size")
    if args.generation_max_new_tokens >= args.max_length:
        raise SystemExit("generation length must be smaller than max length")
    if not 0 <= args.warmup_ratio < 1:
        raise SystemExit("warmup ratio must be in [0, 1)")


def resolve_revisions() -> tuple[str, str]:
    from huggingface_hub import HfApi

    api = HfApi()
    model_revision = api.model_info(MODEL_ID).sha
    dataset_revision = api.dataset_info(DATASET_ID).sha
    if not model_revision or not dataset_revision:
        raise RuntimeError("could not resolve immutable model/dataset revisions")
    return model_revision, dataset_revision


def normalized_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role in {"system", "user", "assistant"} and content:
            result.append({"role": role, "content": content})
    if not result or result[0]["role"] != "system":
        result.insert(0, {"role": "system", "content": SYSTEM_MESSAGE})
    return result


def encode_conversation(tokenizer: Any, messages: list[dict[str, Any]], max_length: int) -> dict[str, Any]:
    input_ids: list[int] = []
    labels: list[int] = []
    for message in normalized_messages(messages):
        role = message["role"]
        prefix = tokenizer.encode(f"<|im_start|>{role}\n", add_special_tokens=False)
        content = tokenizer.encode(message["content"], add_special_tokens=False)
        suffix = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
        segment = prefix + content + suffix
        input_ids.extend(segment)
        if role == "assistant":
            labels.extend([-100] * len(prefix) + content + suffix)
        else:
            labels.extend([-100] * len(segment))
    original_length = len(input_ids)
    input_ids, labels = input_ids[:max_length], labels[:max_length]
    if not input_ids or not any(label != -100 for label in labels):
        raise ValueError("conversation has no assistant target within max_length")
    return {
        "input_ids": input_ids, "attention_mask": [1] * len(input_ids),
        "labels": labels, "length": len(input_ids),
        "truncated": original_length > max_length,
    }


def conversation_identifier(row: dict[str, Any]) -> str:
    for key in ("prompt_id", "conversation_id", "id"):
        if row.get(key) is not None:
            return str(row[key])
    payload = json.dumps(row.get("messages", []), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_usable_subset(source: Any, tokenizer: Any, size: int,
                         max_length: int, seed: int):
    """Take the first `size` encodable rows after a deterministic shuffle."""
    selected_rows: list[dict[str, Any]] = []
    selected_encoded: list[dict[str, Any]] = []
    selected_positions: list[int] = []
    skipped_identifiers: list[str] = []
    for shuffled_position, source_row in enumerate(source.shuffle(seed=seed)):
        row = dict(source_row)
        try:
            encoded = encode_conversation(tokenizer, row["messages"], max_length)
        except ValueError:
            skipped_identifiers.append(conversation_identifier(row))
            continue
        selected_rows.append(row)
        selected_encoded.append(encoded)
        selected_positions.append(shuffled_position)
        if len(selected_rows) == size:
            break
    if len(selected_rows) != size:
        raise RuntimeError(
            f"found only {len(selected_rows)} usable conversations; requested {size}"
        )
    return selected_rows, selected_encoded, selected_positions, skipped_identifiers


def prepare_datasets(tokenizer: Any, dataset_revision: str, args: argparse.Namespace):
    from datasets import Dataset, load_dataset

    train_source = load_dataset(DATASET_ID, split="train_sft", revision=dataset_revision)
    eval_source = load_dataset(DATASET_ID, split="test_sft", revision=dataset_revision)
    train_rows, train_encoded, train_positions, train_skipped = select_usable_subset(
        train_source, tokenizer, args.train_size, args.max_length, args.seed
    )
    eval_rows, eval_encoded, eval_positions, eval_skipped = select_usable_subset(
        eval_source, tokenizer, args.eval_size, args.max_length, args.seed
    )
    manifest = {
        "selection_rule": (
            "first requested number of conversations with an assistant target "
            "within max_length after seeded shuffle"
        ),
        "train_identifiers": [conversation_identifier(row) for row in train_rows],
        "eval_identifiers": [conversation_identifier(row) for row in eval_rows],
        "train_shuffled_positions": train_positions,
        "eval_shuffled_positions": eval_positions,
        "train_skipped_identifiers": train_skipped,
        "eval_skipped_identifiers": eval_skipped,
        "train_mean_tokens": sum(row["length"] for row in train_encoded) / len(train_encoded),
        "eval_mean_tokens": sum(row["length"] for row in eval_encoded) / len(eval_encoded),
        "train_truncated": sum(bool(row["truncated"]) for row in train_encoded),
        "eval_truncated": sum(bool(row["truncated"]) for row in eval_encoded),
    }
    keep = {"input_ids", "attention_mask", "labels"}
    train_dataset = Dataset.from_list([{k: v for k, v in row.items() if k in keep} for row in train_encoded])
    eval_dataset = Dataset.from_list([{k: v for k, v in row.items() if k in keep} for row in eval_encoded])
    return train_dataset, eval_dataset, eval_rows, manifest


class CausalCollator:
    def __init__(self, pad_token_id: int, multiple: int = 8):
        self.pad_token_id, self.multiple = pad_token_id, multiple

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        longest = max(len(item["input_ids"]) for item in features)
        target = int(math.ceil(longest / self.multiple) * self.multiple)
        input_ids, attention_masks, labels = [], [], []
        for item in features:
            padding = target - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            attention_masks.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def gpu_memory_mib() -> tuple[int, int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()[0]
    used, total = (int(part.strip()) for part in output.split(",", 1))
    return used, total


class GPUMemoryMonitor:
    def __init__(self, interval: float):
        self.interval, self.peak_mib, self.total_mib = interval, 0, 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                used, total = gpu_memory_mib()
                self.peak_mib, self.total_mib = max(self.peak_mib, used), total
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self) -> None:
        self.peak_mib, self.total_mib = gpu_memory_mib()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval * 4))
        try:
            used, total = gpu_memory_mib()
            self.peak_mib, self.total_mib = max(self.peak_mib, used), total
        except Exception:
            pass


def qualitative_cases(eval_rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in eval_rows:
        messages = normalized_messages(row["messages"])
        positions = [i for i, m in enumerate(messages) if m["role"] == "assistant" and i and messages[i - 1]["role"] == "user"]
        if positions:
            target = positions[-1]
            cases.append({
                "id": conversation_identifier(row), "history": messages[:target],
                "held_out_reference": messages[target]["content"], "generations": {},
            })
        if len(cases) == count:
            break
    if len(cases) != count:
        raise RuntimeError(f"found only {len(cases)} usable qualitative cases")
    return cases


def render_generation_prompt(tokenizer: Any, history: list[dict[str, str]], max_tokens: int) -> list[int]:
    messages = normalized_messages(history)

    def render(items: list[dict[str, str]]) -> str:
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in items)
        return body + "<|im_start|>assistant\n"

    token_ids = tokenizer.encode(render(messages), add_special_tokens=False)
    while len(token_ids) > max_tokens and len(messages) > 2:
        # Remove the oldest complete turn so that trimming never leaves an
        # assistant message without the user message that preceded it.
        messages.pop(1)
        if len(messages) > 2 and messages[1]["role"] == "assistant":
            messages.pop(1)
        token_ids = tokenizer.encode(render(messages), add_special_tokens=False)
    return token_ids[-max_tokens:]


def generate_responses(model: Any, tokenizer: Any, cases: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    import torch

    model.eval()
    model.config.use_cache = True
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids = [end_id]
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id != end_id:
        stop_ids.append(tokenizer.eos_token_id)
    max_prompt_tokens = args.max_length - args.generation_max_new_tokens
    responses: list[str] = []
    for case in cases:
        token_ids = render_generation_prompt(tokenizer, case["history"], max_prompt_tokens)
        inputs = torch.tensor([token_ids], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            output = model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                max_new_tokens=args.generation_max_new_tokens, do_sample=False,
                eos_token_id=stop_ids, pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=False)
        responses.append(text.split("<|im_end|>", 1)[0].replace("<|endoftext|>", "").strip())
    model.config.use_cache = False
    return responses


def load_base_model(model_revision: str):
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=model_revision, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )


def configure_variant(model: Any, variant: Variant) -> Any:
    if variant.rank is None:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return model
    from peft import LoraConfig, TaskType, get_peft_model

    return get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, target_modules="all-linear",
        r=variant.rank, lora_alpha=2 * variant.rank,
        lora_dropout=0.0, bias="none",
    ))


def training_arguments(variant: Variant, args: argparse.Namespace):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(args.output_dir / "trainer" / variant.key),
        overwrite_output_dir=args.overwrite, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio, lr_scheduler_type="cosine",
        max_grad_norm=1.0, optim="adamw_torch_fused", bf16=True,
        bf16_full_eval=True, tf32=True, gradient_checkpointing=False,
        logging_strategy="steps", logging_steps=args.logging_steps,
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="no", report_to="none", seed=args.seed,
        data_seed=args.seed, dataloader_num_workers=0,
        remove_unused_columns=False, prediction_loss_only=True,
    )


def train_one(variant: Variant, tokenizer: Any, train_dataset: Any,
              eval_dataset: Any, cases: list[dict[str, Any]],
              model_revision: str, args: argparse.Namespace):
    import torch
    from transformers import Trainer, set_seed

    set_seed(args.seed)
    model = configure_variant(load_base_model(model_revision), variant)
    model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    trainer = Trainer(
        model=model, args=training_arguments(variant, args),
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=CausalCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    initial_eval = trainer.evaluate(metric_key_prefix="eval_initial")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    monitor = GPUMemoryMonitor(args.memory_poll_seconds)
    torch.cuda.synchronize()
    monitor.start()
    started = time.perf_counter()
    try:
        train_output = trainer.train()
        torch.cuda.synchronize()
    finally:
        training_time = time.perf_counter() - started
        monitor.stop()
    peak_allocated_mib = torch.cuda.max_memory_allocated() / 2**20
    peak_reserved_mib = torch.cuda.max_memory_reserved() / 2**20
    final_eval = trainer.evaluate(metric_key_prefix="eval_final")
    history = [{"variant": variant.key, "phase": "validation", "step": 0, "epoch": 0.0,
                "loss": initial_eval["eval_initial_loss"]}]
    for entry in trainer.state.log_history:
        if "loss" in entry:
            history.append({"variant": variant.key, "phase": "training",
                            "step": int(entry.get("step", 0)),
                            "epoch": float(entry.get("epoch", 0.0) or 0.0),
                            "loss": float(entry["loss"])})
        if "eval_loss" in entry:
            history.append({"variant": variant.key, "phase": "validation",
                            "step": int(entry.get("step", 0)),
                            "epoch": float(entry.get("epoch", 0.0) or 0.0),
                            "loss": float(entry["eval_loss"])})
    history.append({"variant": variant.key, "phase": "validation",
                    "step": int(trainer.state.global_step), "epoch": float(args.epochs),
                    "loss": float(final_eval["eval_final_loss"])})
    generated = generate_responses(model, tokenizer, cases, args)
    metrics = train_output.metrics
    result = {
        "variant": variant.key, "label": variant.label,
        "lora_rank": "" if variant.rank is None else variant.rank,
        "trainable_parameters": trainable, "total_parameters": total,
        "trainable_percent": 100.0 * trainable / total,
        "peak_gpu_memory_mib": monitor.peak_mib,
        "peak_gpu_memory_percent": 100.0 * monitor.peak_mib / monitor.total_mib,
        "peak_torch_allocated_mib": peak_allocated_mib,
        "peak_torch_reserved_mib": peak_reserved_mib,
        "training_time_seconds": training_time,
        "trainer_reported_runtime_seconds": float(metrics.get("train_runtime", float("nan"))),
        "optimizer_steps": int(trainer.state.global_step),
        "examples_per_second": float(metrics.get("train_samples_per_second", float("nan"))),
        "mean_training_loss": float(metrics["train_loss"]),
        "initial_validation_loss": float(initial_eval["eval_initial_loss"]),
        "final_validation_loss": float(final_eval["eval_final_loss"]),
    }
    if args.save_models:
        save_path = args.output_dir / "models" / variant.key
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    return result, history, generated


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def write_qualitative(path: Path, cases: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Held-out qualitative generations", ""]
    for index, case in enumerate(cases, 1):
        lines.extend([f"## Example {index}", "", "### Conversation presented to the model", ""])
        for message in case["history"]:
            lines.extend([f"**{message['role'].capitalize()}:** {message['content']}", ""])
        lines.extend(["### Held-out reference", "", case["held_out_reference"], ""])
        for key in ["base", *VARIANTS]:
            if key in case["generations"]:
                label = "Base model" if key == "base" else VARIANTS[key].label
                lines.extend([f"### {label}", "", case["generations"][key], ""])
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def hardware_info() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; select a Colab GPU runtime")
    _, total = gpu_memory_mib()
    return {
        "gpu_name": torch.cuda.get_device_name(0), "gpu_total_memory_mib": total,
        "compute_capability": ".".join(str(v) for v in torch.cuda.get_device_capability(0)),
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "cuda_version": str(torch.version.cuda),
        "transformers_version": package_version("transformers"),
        "datasets_version": package_version("datasets"),
        "peft_version": package_version("peft"),
        "accelerate_version": package_version("accelerate"),
    }


def main() -> None:
    args = parse_args()
    variants = selected_variants(args.variants)
    validate_args(args)
    plan = {
        "model": MODEL_ID, "dataset": DATASET_ID,
        "variants": [asdict(v) for v in variants], "train_size": args.train_size,
        "eval_size": args.eval_size, "max_length": args.max_length,
        "epochs": args.epochs,
        "effective_batch_size": args.train_batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{args.output_dir} is not empty; pass --overwrite")
        resolved = args.output_dir.resolve()
        if resolved == Path("/") or len(resolved.parts) < 3:
            raise SystemExit(f"refusing to remove unsafe output path: {resolved}")
        shutil.rmtree(resolved)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoTokenizer, set_seed

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    model_revision, dataset_revision = resolve_revisions()
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID, "model_revision": model_revision,
        "dataset_id": DATASET_ID, "dataset_revision": dataset_revision,
        "dataset_splits": {"training": "train_sft", "evaluation": "test_sft"},
        "hardware_and_software": hardware_info(),
        "training_configuration": {
            **plan, "per_device_train_batch_size": args.train_batch_size,
            "per_device_eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "weight_decay": args.weight_decay, "warmup_ratio": args.warmup_ratio,
            "lr_scheduler": "cosine", "optimizer": "adamw_torch_fused",
            "precision": "bfloat16", "lora_target_modules": "all-linear",
            "lora_alpha": "2 * rank (constant alpha/r = 2)",
            "lora_dropout": 0.0, "assistant_only_loss": True, "seed": args.seed,
            "training_time_scope": (
                "Trainer.train wall time, including scheduled validation; "
                "excludes model loading, separate initial/final validation, and generation"
            ),
            "peak_gpu_memory_scope": (
                "maximum total device memory polled by nvidia-smi during Trainer.train"
            ),
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_dataset, eval_dataset, eval_rows, manifest = prepare_datasets(tokenizer, dataset_revision, args)
    (args.output_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    cases = qualitative_cases(eval_rows, args.qualitative_size)
    base_model = load_base_model(model_revision).to("cuda")
    for case, response in zip(cases, generate_responses(base_model, tokenizer, cases, args)):
        case["generations"]["base"] = response
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    write_qualitative(args.output_dir / "qualitative_generations.json", cases)

    metric_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for variant in variants:
        print(f"\n===== {variant.label} =====", flush=True)
        metrics, history, generations = train_one(
            variant, tokenizer, train_dataset, eval_dataset, cases,
            model_revision, args,
        )
        metric_rows.append(metrics)
        history_rows.extend(history)
        for case, response in zip(cases, generations):
            case["generations"][variant.key] = response
        write_csv(args.output_dir / "metrics.csv", METRIC_FIELDS, metric_rows)
        write_csv(args.output_dir / "loss_history.csv", HISTORY_FIELDS, history_rows)
        write_qualitative(args.output_dir / "qualitative_generations.json", cases)

    print(f"\nRaw SFT artifacts written to {args.output_dir}")
    print("Run: python scripts/plot_sft_results.py --input-dir results/sft")


if __name__ == "__main__":
    main()
