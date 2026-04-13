"""
Approximate AWQ-style INT4 exporter for nanochat checkpoints.

This script is standalone and does not modify any existing repo file.
It performs:
- calibration on a prompt set
- activation-aware per-input-channel scaling for linear weights
- per-output-channel signed INT4 quantization
- packed export artifact for later evaluation with chat_quant_eval.py

This is an AWQ-style implementation intended for project work in nanochat,
not a claim of exact parity with the original AWQ codebase.

Examples:
python3 -m scripts.chat_quant_awq --source sft --model-tag d12
python3 -m scripts.chat_quant_awq --source rl --model-tag d12_reinforce --calibration-source gsm8k
"""

import argparse
import json
import math
import os

import torch

from nanochat.checkpoint_manager import find_largest_model, find_last_step, load_checkpoint
from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from tasks.gsm8k import GSM8K


def parse_args():
    parser = argparse.ArgumentParser(description="AWQ-style INT4 exporter for nanochat")
    parser.add_argument("--source", type=str, default=None, choices=["base", "sft", "rl"])
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--calibration-source", type=str, default="gsm8k", choices=["gsm8k"])
    parser.add_argument("--calibration-examples", type=int, default=128)
    parser.add_argument("--max-calibration-tokens", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.5, help="Blend exponent for activation-aware scaling")
    parser.add_argument("--quantize-embeddings", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--suffix", type=str, default="")
    return parser.parse_args()


def resolve_checkpoint_root(args):
    if (args.source is None) == (args.checkpoint_dir is None):
        raise SystemExit("Specify exactly one of --source or --checkpoint-dir")
    if args.source is not None:
        source_map = {
            "base": "base_checkpoints",
            "sft": "chatsft_checkpoints",
            "rl": "chatrl_checkpoints",
        }
        return os.path.join(get_base_dir(), source_map[args.source]), args.source
    return os.path.abspath(os.path.expanduser(args.checkpoint_dir)), "custom"


def load_state(root_dir, model_tag, step):
    if model_tag is None:
        model_tag = find_largest_model(root_dir)
    checkpoint_dir = os.path.join(root_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, torch.device("cpu"), load_optimizer=False)
    return checkpoint_dir, model_tag, step, model_data, meta_data


def build_model_for_calibration(model_data, meta_data):
    from nanochat.checkpoint_manager import _patch_missing_config_keys, _patch_missing_keys

    model_config_kwargs = dict(meta_data["model_config"])
    _patch_missing_config_keys(model_config_kwargs)
    model_config = GPTConfig(**model_config_kwargs)
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=torch.device("cpu"))
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()
    return model


def calibration_sequences(tokenizer, args):
    if args.calibration_source != "gsm8k":
        raise ValueError(f"Unsupported calibration source: {args.calibration_source}")
    task = GSM8K(subset="main", split="train")
    rows = []
    for i in range(min(args.calibration_examples, len(task))):
        conv = task[i]
        ids = tokenizer.render_for_completion(conv)
        rows.append(torch.tensor(ids[:args.max_calibration_tokens], dtype=torch.long))
    return rows


def gather_input_channel_stats(model, sequences):
    stats = {}
    handles = []

    def make_hook(name):
        def hook(module, inputs):
            x = inputs[0].detach().float()
            flat = x.reshape(-1, x.size(-1))
            current = flat.abs().mean(dim=0).cpu()
            if name not in stats:
                stats[name] = {"sum": current, "count": 1}
            else:
                stats[name]["sum"] += current
                stats[name]["count"] += 1
        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_pre_hook(make_hook(name)))

    with torch.no_grad():
        for ids in sequences:
            _ = model(ids.unsqueeze(0))

    for handle in handles:
        handle.remove()

    final = {}
    for name, value in stats.items():
        final[name] = value["sum"] / max(value["count"], 1)
    return final


def should_quantize(name, tensor, quantize_embeddings):
    if not tensor.is_floating_point() or tensor.ndim != 2:
        return False
    if not quantize_embeddings and ("transformer.wte" in name or "value_embeds" in name):
        return False
    return True


def pack_int4_signed(q):
    q = q.to(torch.int16)
    q = torch.where(q < 0, q + 16, q).to(torch.uint8)
    flat = q.reshape(-1)
    if flat.numel() % 2 == 1:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8)], dim=0)
    lo = flat[0::2]
    hi = flat[1::2] << 4
    return (lo | hi).contiguous()


def awq_quantize_weight(weight, act_mean, alpha):
    weight = weight.float()
    act_mean = act_mean.float().clamp(min=1e-6)
    pre_scale = act_mean.pow(alpha)
    pre_scale = pre_scale / pre_scale.mean().clamp(min=1e-6)
    scaled = weight * pre_scale.view(1, -1)
    row_max = scaled.abs().amax(dim=1).clamp(min=1e-6)
    q_scale = row_max / 7.0
    q = torch.round(scaled / q_scale.view(-1, 1)).clamp(-8, 7).to(torch.int8)
    return q, q_scale.to(torch.float32), pre_scale.to(torch.float32)


def export_awq(model_data, meta_data, channel_stats, alpha, quantize_embeddings):
    export_tensors = {}
    tensor_meta = {}
    original_bytes = 0
    exported_bytes = 0
    quantized_count = 0

    for name, tensor in model_data.items():
        original_bytes += tensor.numel() * tensor.element_size()
        if should_quantize(name, tensor, quantize_embeddings):
            module_name = name.rsplit(".", 1)[0]
            if module_name not in channel_stats:
                export_tensors[name] = tensor.cpu()
                exported_bytes += tensor.numel() * tensor.element_size()
                tensor_meta[name] = {
                    "scheme": "identity",
                    "original_dtype": str(tensor.dtype).replace("torch.", ""),
                    "original_shape": list(tensor.shape),
                }
                continue
            q, q_scale, pre_scale = awq_quantize_weight(tensor, channel_stats[module_name], alpha)
            packed = pack_int4_signed(q.cpu())
            export_tensors[name] = packed
            tensor_meta[name] = {
                "scheme": "awq_int4_per_out_channel",
                "original_dtype": str(tensor.dtype).replace("torch.", ""),
                "original_shape": list(tensor.shape),
                "scale": q_scale.cpu(),
                "pre_scale": pre_scale.cpu(),
            }
            exported_bytes += packed.numel() * packed.element_size()
            exported_bytes += q_scale.numel() * q_scale.element_size()
            exported_bytes += pre_scale.numel() * pre_scale.element_size()
            quantized_count += 1
        else:
            export_tensors[name] = tensor.cpu()
            exported_bytes += tensor.numel() * tensor.element_size()
            tensor_meta[name] = {
                "scheme": "identity",
                "original_dtype": str(tensor.dtype).replace("torch.", ""),
                "original_shape": list(tensor.shape),
            }

    stats = {
        "original_bytes": int(original_bytes),
        "exported_bytes": int(exported_bytes),
        "compression_ratio": float(original_bytes / max(exported_bytes, 1)),
        "quantized_tensors": quantized_count,
        "total_tensors": len(model_data),
        "alpha": alpha,
    }
    return export_tensors, tensor_meta, stats


def main():
    args = parse_args()
    root_dir, source_name = resolve_checkpoint_root(args)
    checkpoint_dir, model_tag, step, model_data, meta_data = load_state(root_dir, args.model_tag, args.step)

    print(f"Loaded checkpoint: {checkpoint_dir} step {step}")
    model = build_model_for_calibration(model_data, meta_data)
    tokenizer = get_tokenizer()
    sequences = calibration_sequences(tokenizer, args)
    print(f"Collected {len(sequences)} calibration prompts")
    channel_stats = gather_input_channel_stats(model, sequences)
    print(f"Gathered activation stats for {len(channel_stats)} linear modules")

    export_tensors, tensor_meta, stats = export_awq(
        model_data={k.removeprefix("_orig_mod."): v for k, v in model_data.items()},
        meta_data=meta_data,
        channel_stats=channel_stats,
        alpha=args.alpha,
        quantize_embeddings=bool(args.quantize_embeddings),
    )

    if args.output_dir is None:
        suffix = f"_{args.suffix}" if args.suffix else ""
        export_dir = os.path.join(get_base_dir(), "chatquant_exports", f"{model_tag}_awq_int4{suffix}")
    else:
        export_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(export_dir, exist_ok=True)

    artifact_path = os.path.join(export_dir, f"quant_{step:06d}.pt")
    meta_path = os.path.join(export_dir, f"meta_{step:06d}.json")

    torch.save(
        {
            "quantized_state": export_tensors,
            "tensor_meta": tensor_meta,
            "source_checkpoint_dir": checkpoint_dir,
            "source_model_tag": model_tag,
            "source_step": step,
            "source_family": source_name,
            "quant_method": "awq_int4",
            "quantize_embeddings": bool(args.quantize_embeddings),
            "model_config": meta_data.get("model_config"),
        },
        artifact_path,
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_checkpoint_dir": checkpoint_dir,
                "source_model_tag": model_tag,
                "source_step": step,
                "source_family": source_name,
                "quant_method": "awq_int4",
                "quantize_embeddings": bool(args.quantize_embeddings),
                "stats": stats,
                "model_config": meta_data.get("model_config"),
            },
            f,
            indent=2,
        )

    print(f"Exported AWQ-style INT4 artifact to {artifact_path}")
    print(f"Metadata written to {meta_path}")
    print(
        "Compression stats: "
        f"original={stats['original_bytes']:,} bytes | "
        f"exported={stats['exported_bytes']:,} bytes | "
        f"ratio={stats['compression_ratio']:.2f}x | "
        f"quantized_tensors={stats['quantized_tensors']}/{stats['total_tensors']}"
    )


if __name__ == "__main__":
    main()
