"""
Compare multiple post-trained chat models side by side.

This script is standalone and does not modify any existing file.

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
  --task-name GSM8K|MMLU
"""

import argparse
import os

from nanochat.checkpoint_manager import load_model, load_model_from_dir
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0
from nanochat.engine import Engine
from nanochat.report import get_report
from scripts.chat_eval import run_chat_eval


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
    return parser.parse_args()


def parse_model_spec(spec):
    if "=" not in spec:
        raise ValueError(f"Invalid model spec: {spec}. Expected label=source[:tag[:step]]")
    label, rhs = spec.split("=", 1)
    parts = rhs.split(":")
    source_or_dir = parts[0]
    model_tag = parts[1] if len(parts) >= 2 and parts[1] != "" else None
    step = int(parts[2]) if len(parts) >= 3 and parts[2] != "" else None
    return {
        "label": label,
        "source_or_dir": source_or_dir,
        "model_tag": model_tag,
        "step": step,
    }


def load_spec_model(spec, device):
    source_or_dir = spec["source_or_dir"]
    model_tag = spec["model_tag"]
    step = spec["step"]
    if source_or_dir.startswith("@"):
        checkpoints_dir = os.path.expanduser(source_or_dir[1:])
        if not os.path.isabs(checkpoints_dir):
            checkpoints_dir = os.path.abspath(checkpoints_dir)
        model, tokenizer, meta = load_model_from_dir(checkpoints_dir, device, phase="eval", model_tag=model_tag, step=step)
        origin = checkpoints_dir
    else:
        model, tokenizer, meta = load_model(source_or_dir, device, phase="eval", model_tag=model_tag, step=step)
        origin = source_or_dir
    return model, tokenizer, meta, origin


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
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    all_tasks = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "SpellingBee"]
    task_names = all_tasks if args.task_name is None else args.task_name.split("|")
    model_specs = [parse_model_spec(spec) for spec in args.models]

    results_by_model = {}
    metadata_rows = []
    for spec in model_specs:
        label = spec["label"]
        model, tokenizer, meta, origin = load_spec_model(spec, device)
        engine = Engine(model, tokenizer)
        print0("=" * 80)
        print0(f"Evaluating {label} from {origin} | tag={spec['model_tag']} | step={spec['step']}")
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
                "origin": origin,
                "model_tag": spec["model_tag"],
                "step": spec["step"],
                "resolved_meta_keys": sorted(meta.keys()),
            }
        )

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
    compute_cleanup()


if __name__ == "__main__":
    main()
