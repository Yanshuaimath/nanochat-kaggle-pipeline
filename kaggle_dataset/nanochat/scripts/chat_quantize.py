"""
Standalone post-training quantization exporter for nanochat checkpoints.

This script does not modify existing model files or checkpoint loaders.
Instead, it exports a separate quantized artifact containing:
- quantized tensors
- per-tensor scales / zero-points when needed
- metadata for reconstruction

Supported methods in this first version:
- int8_sym      : symmetric per-tensor int8 quantization for floating tensors
- int8_linear   : same, but only for matrix-like tensors (ndim >= 2)
- fp16_copy     : cast floating tensors to fp16 for a lightweight compressed export

Examples:
python3 -m scripts.chat_quantize --source=sft --model-tag=d12 --method=int8_linear
python3 -m scripts.chat_quantize --checkpoint-dir ~/.cache/nanochat/chatdpo_checkpoints --model-tag=d12 --method=int8_sym
"""

import argparse
import json
import os
from typing import Dict, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize a nanochat checkpoint into a separate export artifact")
    parser.add_argument("--source", type=str, default=None, choices=["base", "sft", "rl"], help="Built-in checkpoint family")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Custom checkpoints root directory")
    parser.add_argument("--model-tag", type=str, default=None, help="Model tag inside the checkpoint root")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step")
    parser.add_argument("--method", type=str, default="int8_linear", choices=["int8_sym", "int8_linear", "fp16_copy"])
    parser.add_argument("--quantize-embeddings", type=int, default=0, help="Include 2D embedding tables when using int8_linear")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for exported quantized artifacts")
    parser.add_argument("--suffix", type=str, default="", help="Optional suffix for export directory naming")
    return parser.parse_args()


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("This script requires PyTorch. Please install it first.") from exc
    return torch


def resolve_checkpoint_root(args):
    if (args.source is None) == (args.checkpoint_dir is None):
        raise SystemExit("Specify exactly one of --source or --checkpoint-dir")

    if args.source is not None:
        from nanochat.common import get_base_dir
        source_map = {
            "base": "base_checkpoints",
            "sft": "chatsft_checkpoints",
            "rl": "chatrl_checkpoints",
        }
        base_dir = get_base_dir()
        root = os.path.join(base_dir, source_map[args.source])
        return root, args.source

    root = os.path.abspath(os.path.expanduser(args.checkpoint_dir))
    return root, "custom"


def load_checkpoint_state(root_dir, model_tag, step):
    from nanochat.checkpoint_manager import find_largest_model, find_last_step, load_checkpoint
    torch = require_torch()

    if model_tag is None:
        model_tag = find_largest_model(root_dir)
    checkpoint_dir = os.path.join(root_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, torch.device("cpu"), load_optimizer=False)
    return checkpoint_dir, model_tag, step, model_data, meta_data


def should_quantize_tensor(name, tensor, method, quantize_embeddings):
    if not tensor.is_floating_point():
        return False
    if method == "fp16_copy":
        return True
    if method == "int8_sym":
        return True
    if method == "int8_linear":
        if tensor.ndim < 2:
            return False
        if not quantize_embeddings and ("transformer.wte" in name or "value_embeds" in name):
            return False
        return True
    return False


def quantize_int8_symmetric(torch, tensor):
    if tensor.numel() == 0:
        q = torch.zeros_like(tensor, dtype=torch.int8)
        scale = torch.tensor(1.0, dtype=torch.float32)
        return q, {"scheme": "symmetric_per_tensor", "scale": scale}

    max_abs = tensor.abs().max().float()
    scale = torch.clamp(max_abs / 127.0, min=1e-12)
    q = torch.clamp(torch.round(tensor.float() / scale), min=-127, max=127).to(torch.int8)
    return q, {"scheme": "symmetric_per_tensor", "scale": scale}


def quantize_fp16(torch, tensor):
    return tensor.to(torch.float16), {"scheme": "cast_fp16"}


def tensor_nbytes(tensor):
    return tensor.numel() * tensor.element_size()


def export_quantized_state(torch, state_dict, method, quantize_embeddings):
    export_tensors: Dict[str, object] = {}
    tensor_meta: Dict[str, dict] = {}
    original_bytes = 0
    exported_bytes = 0
    quantized_count = 0

    for name, tensor in state_dict.items():
        original_bytes += tensor_nbytes(tensor)

        if should_quantize_tensor(name, tensor, method, quantize_embeddings):
            if method == "fp16_copy":
                qtensor, qmeta = quantize_fp16(torch, tensor)
                export_tensors[name] = qtensor.cpu()
                exported_bytes += tensor_nbytes(qtensor)
            else:
                qtensor, qmeta = quantize_int8_symmetric(torch, tensor)
                export_tensors[name] = qtensor.cpu()
                exported_bytes += tensor_nbytes(qtensor)
                exported_bytes += tensor_nbytes(qmeta["scale"])
                qmeta["scale"] = qmeta["scale"].cpu()
            qmeta["original_dtype"] = str(tensor.dtype).replace("torch.", "")
            qmeta["original_shape"] = list(tensor.shape)
            tensor_meta[name] = qmeta
            quantized_count += 1
        else:
            export_tensors[name] = tensor.cpu()
            exported_bytes += tensor_nbytes(tensor)
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
        "total_tensors": len(state_dict),
    }
    return export_tensors, tensor_meta, stats


def main():
    args = parse_args()
    torch = require_torch()

    root_dir, source_name = resolve_checkpoint_root(args)
    checkpoint_dir, model_tag, step, model_data, meta_data = load_checkpoint_state(root_dir, args.model_tag, args.step)

    export_tensors, tensor_meta, stats = export_quantized_state(
        torch=torch,
        state_dict=model_data,
        method=args.method,
        quantize_embeddings=bool(args.quantize_embeddings),
    )

    if args.output_dir is None:
        from nanochat.common import get_base_dir
        base_dir = get_base_dir()
        suffix = f"_{args.suffix}" if args.suffix else ""
        export_dir = os.path.join(base_dir, "chatquant_exports", f"{model_tag}_{args.method}{suffix}")
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
            "quant_method": args.method,
            "quantize_embeddings": bool(args.quantize_embeddings),
            "model_config": meta_data.get("model_config"),
        },
        artifact_path,
    )

    export_meta = {
        "source_checkpoint_dir": checkpoint_dir,
        "source_model_tag": model_tag,
        "source_step": step,
        "source_family": source_name,
        "quant_method": args.method,
        "quantize_embeddings": bool(args.quantize_embeddings),
        "stats": stats,
        "model_config": meta_data.get("model_config"),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(export_meta, f, indent=2)

    print(f"Exported quantized artifact to {artifact_path}")
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
