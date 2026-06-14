"""
Compare multiple post-trained chat models side by side.

This script is standalone and does not modify checkpoints or training files.
It writes evaluation results to the nanochat report directory.

You can compare models from:
- built-in sources: base, sft, rl
- custom checkpoint directories such as chatdistill_checkpoints, chatdpo_checkpoints, chatppo_checkpoints
- any other compatible checkpoint directory loadable via load_model_from_dir

Quantized export artifacts are evaluated separately with chat_quant_eval.py.

Model spec format:
- label=source[:model_tag[:step]]
- label=@/absolute/or/relative/checkpoints_dir[:model_tag[:step]]

Examples:
python3 -m scripts.chat_post_eval \
  --models sft=sft:d12 \
  --models rl=rl:d12_reinforce \
  --models dpo=@~/.cache/nanochat/chatdpo_checkpoints:d12

python3 -m scripts.chat_post_eval \
  --models distill=@~/.cache/nanochat/chatdistill_checkpoints:d12 \
  --models dpo=@~/.cache/nanochat/chatdpo_checkpoints:d12 \
  --task-name GSM8K

python3 -m scripts.chat_post_eval \
  --models ppo=@~/.cache/nanochat/chatppo_checkpoints:d12 \
  --task-name 'GSM8K|MMLU'
"""

import argparse
import gc
import math
import os

import torch

from nanochat.checkpoint_manager import find_largest_model, find_last_step, load_model_from_dir
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0
from nanochat.engine import Engine
from nanochat.report import get_report
from scripts.chat_eval import run_chat_eval


SOURCE_DIRS = {
    "base": "base_checkpoints",
    "sft": "chatsft_checkpoints",
    "rl": "chatrl_checkpoints",
    "ppo": "chatppo_checkpoints",
    "ppo_standard": "chatppo_standard_checkpoints",
    "grpo": "chatgrpo_checkpoints",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare post-trained chat models")
    parser.add_argument(
        "--models",
        action="append",
        required=True,
        help="Model spec: label=source[:tag[:step]] or label=@checkpoints_dir[:tag[:step]]",
    )
    parser.add_argument("-a", "--task-name", type=str, default=None, help="Task name or | separated list. Default = standard suite")
    parser.add_argument("-t", "--temperature", type=float, default=0.0)
    parser.add_argument("-m", "--max-new-tokens", type=int, default=512)
    parser.add_argument("-n", "--num-samples", type=int, default=1)
    parser.add_argument("-k", "--top-k", type=int, default=50)
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size for categorical evaluation")
    parser.add_argument("-x", "--max-problems", type=int, default=None, help="Max problems to evaluate")
    parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"], help="cuda|cpu|mps, empty => autodetect")
    args = parser.parse_args()
    if not math.isfinite(args.temperature) or args.temperature < 0.0:
        parser.error("--temperature must be a finite non-negative value")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.top_k < 0:
        parser.error("--top-k must be non-negative")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_problems is not None and args.max_problems <= 0:
        parser.error("--max-problems must be positive when provided")
    return args


def parse_model_spec(spec):
    if "=" not in spec:
        raise ValueError(f"Invalid model spec: {spec}. Expected label=source[:tag[:step]]")
    label, rhs = spec.split("=", 1)
    if label == "":
        raise ValueError(f"Invalid model spec: {spec}. Label must be non-empty")
    if rhs == "":
        raise ValueError(f"Invalid model spec: {spec}. Source must be non-empty")
    parts = rhs.split(":")
    if len(parts) > 3:
        raise ValueError(f"Invalid model spec: {spec}. Expected label=source[:tag[:step]]")
    source_or_dir = parts[0]
    if source_or_dir == "@":
        raise ValueError(f"Invalid model spec: {spec}. Custom checkpoint directory must be non-empty")
    model_tag = parts[1] if len(parts) >= 2 and parts[1] != "" else None
    try:
        step = int(parts[2]) if len(parts) >= 3 and parts[2] != "" else None
    except ValueError:
        raise ValueError(f"Invalid model spec: {spec}. Step must be an integer") from None
    if step is not None and step < 0:
        raise ValueError(f"Invalid model spec: {spec}. Step must be non-negative")
    return {
        "label": label,
        "source_or_dir": source_or_dir,
        "model_tag": model_tag,
        "step": step,
    }


def validate_unique_labels(model_specs):
    seen = set()
    duplicates = set()
    for spec in model_specs:
        label = spec["label"]
        if label in seen:
            duplicates.add(label)
        seen.add(label)
    if duplicates:
        labels = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate model label(s): {labels}")


def validate_task_names(task_names, all_tasks):
    unknown = [task for task in task_names if task not in all_tasks]
    if unknown:
        valid = ", ".join(all_tasks)
        raise ValueError(f"Unknown task name(s): {'|'.join(unknown)}. Expected one or more of: {valid}")


def validate_model_sources(model_specs):
    unknown = sorted({
        spec["source_or_dir"]
        for spec in model_specs
        if not spec["source_or_dir"].startswith("@") and spec["source_or_dir"] not in SOURCE_DIRS
    })
    if unknown:
        valid = ", ".join(sorted(SOURCE_DIRS))
        raise ValueError(f"Unknown model source(s): {', '.join(unknown)}. Expected one of: {valid}")


def resolve_checkpoint(spec):
    source_or_dir = spec["source_or_dir"]
    model_tag = spec["model_tag"]
    step = spec["step"]
    if source_or_dir.startswith("@"):
        checkpoints_dir = os.path.expanduser(source_or_dir[1:])
        if not os.path.isabs(checkpoints_dir):
            checkpoints_dir = os.path.abspath(checkpoints_dir)
        origin = checkpoints_dir
    else:
        if source_or_dir not in SOURCE_DIRS:
            valid = ", ".join(sorted(SOURCE_DIRS))
            raise ValueError(f"Unknown model source: {source_or_dir}. Expected one of: {valid}")
        checkpoints_dir = os.path.join(get_base_dir(), SOURCE_DIRS[source_or_dir])
        origin = source_or_dir

    resolved_model_tag = model_tag if model_tag is not None else find_largest_model(checkpoints_dir)
    checkpoint_dir = os.path.join(checkpoints_dir, resolved_model_tag)
    resolved_step = step if step is not None else find_last_step(checkpoint_dir)
    return origin, checkpoints_dir, checkpoint_dir, resolved_model_tag, resolved_step


def load_spec_model(spec, device):
    origin, checkpoints_dir, checkpoint_dir, model_tag, step = resolve_checkpoint(spec)
    model, tokenizer, meta = load_model_from_dir(checkpoints_dir, device, phase="eval", model_tag=model_tag, step=step)
    return model, tokenizer, meta, {
        "origin": origin,
        "checkpoint_dir": checkpoint_dir,
        "model_tag": model_tag,
        "step": step,
    }


def clear_device_cache(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def format_pct(x):
    return f"{100.0 * x:.2f}%"


def render_table(task_names, results_by_model):
    headers = ["Model"] + task_names + ["Mean"]
    rows = []
    for label, task_scores in results_by_model.items():
        scores = [task_scores[task] for task in task_names]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        rows.append([label] + [format_pct(score) for score in scores] + [format_pct(mean_score)])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    divider = "-+-".join("-" * width for width in widths)
    lines = [fmt_row(headers), divider]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def main():
    args = parse_args()
    all_tasks = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "SpellingBee"]
    task_names = all_tasks if args.task_name is None else args.task_name.split("|")
    try:
        validate_task_names(task_names, all_tasks)
        model_specs = [parse_model_spec(spec) for spec in args.models]
        validate_unique_labels(model_specs)
        validate_model_sources(model_specs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    if device_type != "cuda" and all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        raise SystemExit("torchrun/DDP evaluation is only supported with --device-type cuda")

    try:
        _, ddp_rank, _, _, device = compute_init(device_type)
        results_by_model = {}
        metadata_rows = []
        for spec in model_specs:
            label = spec["label"]
            model = None
            tokenizer = None
            engine = None
            try:
                model, tokenizer, meta, resolved = load_spec_model(spec, device)
                engine = Engine(model, tokenizer)
                print0("=" * 80)
                print0(f"Evaluating {label} from {resolved['origin']} | tag={resolved['model_tag']} | step={resolved['step']}")
                task_scores = {}
                for task_name in task_names:
                    acc = run_chat_eval(
                        task_name,
                        model,
                        tokenizer,
                        engine,
                        batch_size=args.batch_size,
                        num_samples=args.num_samples,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        max_problems=args.max_problems,
                    )
                    task_scores[task_name] = acc
                    print0(f"{label} | {task_name}: {format_pct(acc)}")
                results_by_model[label] = task_scores
                metadata_rows.append(
                    {
                        "label": label,
                        "origin": resolved["origin"],
                        "checkpoint_dir": resolved["checkpoint_dir"],
                        "model_tag": resolved["model_tag"],
                        "step": resolved["step"],
                        "resolved_meta_keys": sorted(meta.keys()),
                    }
                )
            finally:
                del engine, tokenizer, model
                clear_device_cache(device)

        if ddp_rank == 0:
            print()
            print(render_table(task_names, results_by_model))
            ranking = []
            for label, task_scores in results_by_model.items():
                mean_score = sum(task_scores[task] for task in task_names) / len(task_names)
                ranking.append((mean_score, label))
            ranking.sort(reverse=True)
            print()
            print("Ranking by mean score:")
            for rank, (mean_score, label) in enumerate(ranking, start=1):
                print(f"{rank}. {label}: {format_pct(mean_score)}")

        report_rows = []
        for label, task_scores in results_by_model.items():
            report_rows.append({"label": label, **task_scores})
        get_report().log(
            section="Chat Post Eval",
            data=[
                {
                    "tasks": task_names,
                    "num_samples": args.num_samples,
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                    "batch_size": args.batch_size,
                    "max_problems": args.max_problems,
                },
                {"models": metadata_rows},
                {"results": report_rows},
            ],
        )
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
