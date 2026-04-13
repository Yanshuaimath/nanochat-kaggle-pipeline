"""
Evaluate a quantized nanochat export artifact by reconstructing it into a float model.

This script is standalone and does not modify any existing file.
It can evaluate:
- a regular checkpoint loaded from source/checkpoint-dir
- a quantized export produced by chat_quantize.py or chat_quant_awq.py
- or both side by side

Examples:
python3 -m scripts.chat_quant_eval --quant-artifact ~/.cache/nanochat/chatquant_exports/d12_int8_linear/quant_000100.pt
python3 -m scripts.chat_quant_eval --source sft --model-tag d12 --quant-artifact ~/.cache/nanochat/chatquant_exports/d12_awq_int4/quant_000100.pt --task-name GSM8K|MMLU
"""

import argparse
import os

import torch

from nanochat.checkpoint_manager import (
    _patch_missing_config_keys,
    _patch_missing_keys,
    load_model,
    load_model_from_dir,
)
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, print0
from nanochat.engine import Engine
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from scripts.chat_eval import run_chat_eval


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate quantized nanochat exports")
    parser.add_argument("--quant-artifact", type=str, default=None, help="Path to quantized artifact .pt")
    parser.add_argument("--source", type=str, default=None, choices=["base", "sft", "rl"], help="Optional baseline source")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Optional custom checkpoint root for baseline")
    parser.add_argument("--model-tag", type=str, default=None, help="Model tag for baseline")
    parser.add_argument("--step", type=int, default=None, help="Model step for baseline")
    parser.add_argument("-a", "--task-name", type=str, default=None, help="Task name or | separated list. Default = standard suite")
    parser.add_argument("-t", "--temperature", type=float, default=0.0)
    parser.add_argument("-m", "--max-new-tokens", type=int, default=512)
    parser.add_argument("-n", "--num-samples", type=int, default=1)
    parser.add_argument("-k", "--top-k", type=int, default=50)
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("-x", "--max-problems", type=int, default=None)
    parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"])
    return parser.parse_args()


def unpack_int4_packed(torch_mod, packed, original_shape):
    flat = packed.view(-1)
    lo = (flat & 0x0F).to(torch_mod.int8)
    hi = ((flat >> 4) & 0x0F).to(torch_mod.int8)
    vals = torch_mod.empty(flat.numel() * 2, dtype=torch_mod.int8)
    vals[0::2] = lo
    vals[1::2] = hi
    vals = torch_mod.where(vals >= 8, vals - 16, vals)
    total = 1
    for dim in original_shape:
        total *= dim
    vals = vals[:total]
    return vals.view(*original_shape)


def dequantize_state(artifact, device):
    quantized_state = artifact["quantized_state"]
    tensor_meta = artifact["tensor_meta"]
    state_dict = {}
    for name, stored in quantized_state.items():
        meta = tensor_meta[name]
        scheme = meta["scheme"]
        if scheme == "identity":
            tensor = stored
        elif scheme == "cast_fp16":
            tensor = stored.float()
        elif scheme == "symmetric_per_tensor":
            tensor = stored.float() * meta["scale"].float()
        elif scheme == "awq_int4_per_out_channel":
            q = unpack_int4_packed(torch, stored, meta["original_shape"]).float()
            scale = meta["scale"].float().view(-1, *([1] * (q.ndim - 1)))
            pre_scale = meta["pre_scale"].float().view(1, -1)
            tensor = (q * scale) / pre_scale
        else:
            raise ValueError(f"Unsupported quantization scheme: {scheme}")
        state_dict[name] = tensor.to(device)
    return state_dict


def load_quantized_model(artifact_path, device):
    artifact = torch.load(os.path.expanduser(artifact_path), map_location="cpu")
    model_config_kwargs = dict(artifact["model_config"])
    _patch_missing_config_keys(model_config_kwargs)
    model_config = GPTConfig(**model_config_kwargs)
    state_dict = dequantize_state(artifact, device)
    _patch_missing_keys(state_dict, model_config)
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.eval()
    tokenizer = get_tokenizer()
    return model, tokenizer, artifact


def maybe_load_baseline(args, device):
    if args.source is None and args.checkpoint_dir is None:
        return None
    if (args.source is None) == (args.checkpoint_dir is None):
        raise SystemExit("Specify at most one of --source or --checkpoint-dir for baseline loading")
    if args.source is not None:
        model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
        label = f"baseline:{args.source}"
    else:
        checkpoints_dir = os.path.abspath(os.path.expanduser(args.checkpoint_dir))
        model, tokenizer, meta = load_model_from_dir(checkpoints_dir, device, phase="eval", model_tag=args.model_tag, step=args.step)
        label = f"baseline:{checkpoints_dir}"
    return label, model, tokenizer, meta


def run_suite(label, model, tokenizer, args, task_names):
    engine = Engine(model, tokenizer)
    results = {}
    print0("=" * 80)
    print0(f"Evaluating {label}")
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
        results[task_name] = acc
        print0(f"{label} | {task_name}: {100 * acc:.2f}%")
    return results


def print_comparison_table(task_names, rows):
    headers = ["Model"] + task_names + ["Mean"]
    widths = [len(h) for h in headers]
    rendered_rows = []
    for label, result in rows:
        scores = [result[t] for t in task_names]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        row = [label] + [f"{100 * s:.2f}%" for s in scores] + [f"{100 * mean_score:.2f}%"]
        rendered_rows.append(row)
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rendered_rows:
        print(fmt(row))


def main():
    args = parse_args()
    if args.quant_artifact is None and args.source is None and args.checkpoint_dir is None:
        raise SystemExit("Provide at least one model: --quant-artifact and/or a baseline source/checkpoint-dir")

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    all_tasks = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "SpellingBee"]
    task_names = all_tasks if args.task_name is None else args.task_name.split("|")

    rows = []

    baseline = maybe_load_baseline(args, device)
    if baseline is not None:
        label, model, tokenizer, meta = baseline
        rows.append((label, run_suite(label, model, tokenizer, args, task_names)))

    if args.quant_artifact is not None:
        qpath = os.path.abspath(os.path.expanduser(args.quant_artifact))
        model, tokenizer, artifact = load_quantized_model(qpath, device)
        label = f"quant:{artifact.get('quant_method', 'unknown')}"
        rows.append((label, run_suite(label, model, tokenizer, args, task_names)))

    if ddp_rank == 0 and rows:
        print()
        print_comparison_table(task_names, rows)

    compute_cleanup()


if __name__ == "__main__":
    main()
