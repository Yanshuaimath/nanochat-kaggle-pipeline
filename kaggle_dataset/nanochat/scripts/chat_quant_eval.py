"""
Evaluate a quantized nanochat export artifact by reconstructing it into a float model.

This script is standalone and does not modify any existing file.
It can evaluate:
- a regular checkpoint loaded from source/checkpoint-dir
- a quantized export produced by chat_quantize.py or chat_quant_awq.py
- or both side by side

Examples:
python3 -m scripts.chat_quant_eval --quant-artifact ~/.cache/nanochat/chatquant_exports/d12_int8_linear/quant_000100.pt
python3 -m scripts.chat_quant_eval --source sft --model-tag d12 --quant-artifact ~/.cache/nanochat/chatquant_exports/d12_awq_int4/quant_000100.pt --task-name 'GSM8K|MMLU'
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F

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
    parser.add_argument("--runtime-quantized", action="store_true", help="Keep eligible Linear weights as quantized runtime buffers instead of fully dequantizing them")
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


def as_tensor(value, device=None):
    tensor = value if torch.is_tensor(value) else torch.tensor(value)
    return tensor.to(device) if device is not None else tensor


def normalize_state_name(name):
    return name.removeprefix("_orig_mod.")


def tensor_for_device(tensor, device):
    if device.type in {"cpu", "mps"} and tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.to(device)


def unpack_int4_packed(torch_mod, packed, original_shape):
    flat = packed.view(-1)
    lo = (flat & 0x0F).to(torch_mod.int8)
    hi = ((flat >> 4) & 0x0F).to(torch_mod.int8)
    vals = torch_mod.empty(flat.numel() * 2, dtype=torch_mod.int8, device=packed.device)
    vals[0::2] = lo
    vals[1::2] = hi
    vals = torch_mod.where(vals >= 8, vals - 16, vals)
    total = 1
    for dim in original_shape:
        total *= dim
    vals = vals[:total]
    return vals.view(*original_shape)


def dequantize_tensor(name, stored, meta):
    scheme = meta["scheme"]
    if scheme == "identity":
        tensor = stored
    elif scheme == "cast_fp16":
        tensor = stored.float()
    elif scheme == "symmetric_per_tensor":
        tensor = stored.float() * as_tensor(meta["scale"]).float()
    elif scheme == "awq_int4_per_out_channel":
        q = unpack_int4_packed(torch, stored, meta["original_shape"]).float()
        scale = as_tensor(meta["scale"]).float().view(-1, *([1] * (q.ndim - 1)))
        pre_scale = as_tensor(meta["pre_scale"]).float().view(1, -1)
        tensor = (q * scale) / pre_scale
    else:
        raise ValueError(f"Unsupported quantization scheme for {name}: {scheme}")
    return tensor


def dequantize_state(artifact, device):
    quantized_state = artifact["quantized_state"]
    tensor_meta = artifact["tensor_meta"]
    state_dict = {}
    for name, stored in quantized_state.items():
        clean_name = normalize_state_name(name)
        if clean_name in state_dict:
            raise ValueError(f"Duplicate tensor name after removing _orig_mod. prefix: {clean_name}")
        meta = tensor_meta.get(name) or tensor_meta.get(clean_name)
        if meta is None:
            raise KeyError(f"Missing tensor metadata for {name}")
        tensor = dequantize_tensor(clean_name, stored, meta)
        state_dict[clean_name] = tensor_for_device(tensor, device)
    return state_dict


def move_state_dict_to_device(state_dict, device):
    for name, tensor in list(state_dict.items()):
        state_dict[name] = tensor_for_device(tensor, device)
    return state_dict


class QuantizedLinear(torch.nn.Module):
    def __init__(self, stored, meta, device):
        super().__init__()
        self.scheme = meta["scheme"]
        self.original_shape = tuple(meta["original_shape"])
        if len(self.original_shape) != 2:
            raise ValueError(f"QuantizedLinear expects a 2D weight, got {self.original_shape}")
        self.out_features, self.in_features = self.original_shape
        self.bias = None

        if self.scheme == "cast_fp16":
            self.register_buffer("weight_fp16", stored.to(device=device, dtype=torch.float16), persistent=False)
        elif self.scheme == "symmetric_per_tensor":
            self.register_buffer("qweight", stored.to(device=device, dtype=torch.int8), persistent=False)
            self.register_buffer("scale", as_tensor(meta["scale"], device=device).float(), persistent=False)
        elif self.scheme == "awq_int4_per_out_channel":
            self.register_buffer("packed_weight", stored.to(device=device, dtype=torch.uint8), persistent=False)
            self.register_buffer("scale", as_tensor(meta["scale"], device=device).float(), persistent=False)
            self.register_buffer("pre_scale", as_tensor(meta["pre_scale"], device=device).float(), persistent=False)
        else:
            raise ValueError(f"Unsupported runtime quantized linear scheme: {self.scheme}")

    def dequantized_weight(self):
        if self.scheme == "cast_fp16":
            return self.weight_fp16.float()
        if self.scheme == "symmetric_per_tensor":
            return self.qweight.float() * self.scale.float()
        if self.scheme == "awq_int4_per_out_channel":
            q = unpack_int4_packed(torch, self.packed_weight, self.original_shape).float()
            scale = self.scale.float().view(-1, 1)
            pre_scale = self.pre_scale.float().view(1, -1)
            return (q * scale) / pre_scale
        raise AssertionError(f"Unhandled scheme: {self.scheme}")

    def forward(self, x):
        return F.linear(x, self.dequantized_weight().to(dtype=x.dtype))


def set_submodule(root, module_name, module):
    parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
    parent = root.get_submodule(parent_name) if parent_name else root
    if isinstance(parent, (torch.nn.ModuleList, torch.nn.Sequential)) and child_name.isdigit():
        parent[int(child_name)] = module
    elif isinstance(parent, torch.nn.ModuleDict):
        parent[child_name] = module
    else:
        setattr(parent, child_name, module)


def build_runtime_quantized_state(artifact, model, device):
    quantized_state = artifact["quantized_state"]
    tensor_meta = artifact["tensor_meta"]
    state_dict = {}
    quantized_linears = {}
    seen_names = set()

    for name, stored in quantized_state.items():
        clean_name = normalize_state_name(name)
        if clean_name in seen_names:
            raise ValueError(f"Duplicate tensor name after removing _orig_mod. prefix: {clean_name}")
        seen_names.add(clean_name)
        meta = tensor_meta.get(name) or tensor_meta.get(clean_name)
        if meta is None:
            raise KeyError(f"Missing tensor metadata for {name}")
        scheme = meta["scheme"]

        if scheme != "identity" and clean_name.endswith(".weight"):
            module_name = clean_name.rsplit(".", 1)[0]
            try:
                module = model.get_submodule(module_name)
            except AttributeError:
                module = None
            if isinstance(module, torch.nn.Linear):
                quantized_linears[module_name] = QuantizedLinear(stored, meta, device)
                continue

        tensor = dequantize_tensor(clean_name, stored, meta)
        state_dict[clean_name] = tensor_for_device(tensor, device)

    for module_name, module in quantized_linears.items():
        set_submodule(model, module_name, module)
    return state_dict, len(quantized_linears)


def check_load_result(load_result):
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(f"Model load failed: missing keys={missing}, unexpected keys={unexpected}")


def load_quantized_model(artifact_path, device, runtime_quantized=False):
    artifact = torch.load(os.path.expanduser(artifact_path), map_location="cpu")
    if runtime_quantized and device.type == "mps":
        print0("Runtime-quantized Linear modules are disabled on MPS; loading a dequantized model.")
        runtime_quantized = False
    model_config_kwargs = dict(artifact["model_config"])
    _patch_missing_config_keys(model_config_kwargs)
    model_config = GPTConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()

    if runtime_quantized:
        state_dict, quantized_linear_count = build_runtime_quantized_state(artifact, model, device)
        _patch_missing_keys(state_dict, model_config)
        move_state_dict_to_device(state_dict, device)
        check_load_result(model.load_state_dict(state_dict, strict=False, assign=True))
    else:
        state_dict = dequantize_state(artifact, device)
        _patch_missing_keys(state_dict, model_config)
        move_state_dict_to_device(state_dict, device)
        model.load_state_dict(state_dict, strict=True, assign=True)
        quantized_linear_count = 0

    model.eval()
    tokenizer = get_tokenizer()
    artifact["runtime_quantized"] = runtime_quantized and quantized_linear_count > 0
    artifact["runtime_quantized_linears"] = quantized_linear_count
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
    from scripts.chat_eval import run_chat_eval

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
        model, tokenizer, artifact = load_quantized_model(qpath, device, runtime_quantized=args.runtime_quantized)
        label = f"quant:{artifact.get('quant_method', 'unknown')}"
        rows.append((label, run_suite(label, model, tokenizer, args, task_names)))

    if ddp_rank == 0 and rows:
        print()
        print_comparison_table(task_names, rows)

    compute_cleanup()


if __name__ == "__main__":
    main()
