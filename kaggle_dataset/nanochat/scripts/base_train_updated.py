"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
import traceback
import sys
from dataclasses import asdict
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, Linear, set_fp16_safe_mlp
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0 as _print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core


def print0(s="", **kwargs):
    """Rank-0 print routed to stderr so Kaggle keeps torchrun child logs."""
    kwargs.setdefault("file", sys.stderr)
    _print0(s, **kwargs)


print_banner()

# -----------------------------------------------------------------------------
# DDP / NCCL debugging helpers
# Enable with environment variables:
#   NANOCHAT_DEBUG_DIST=1       print per-rank debug logs
#   NANOCHAT_DEBUG_STDOUT=1     also mirror debug logs to stdout (file-only by default)
#   NANOCHAT_DEBUG_STACK=1      include stack traces for logged collectives (verbose)
#   NANOCHAT_DEBUG_STEPS=2      how many optimizer steps to debug-print
# Logs are written to /kaggle/working/nanochat_debug_rank{rank}.log
_DEBUG_DIST = os.environ.get("NANOCHAT_DEBUG_DIST", "0") == "1"
_DEBUG_STDOUT = os.environ.get("NANOCHAT_DEBUG_STDOUT", "0") == "1"
_DEBUG_STACK = os.environ.get("NANOCHAT_DEBUG_STACK", "0") == "1"
_DEBUG_STEPS = int(os.environ.get("NANOCHAT_DEBUG_STEPS", "2"))
_DEBUG_COLL_SEQ = 0
_TRAIN_PRINT_EVERY = max(1, int(os.environ.get("NANOCHAT_TRAIN_PRINT_EVERY", "1")))

def dprint(msg):
    """Write a per-rank debug log, optionally mirroring to stdout."""
    if not _DEBUG_DIST:
        return

    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world = int(os.environ.get("WORLD_SIZE", "1"))

    try:
        cuda_dev = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    except Exception:
        cuda_dev = "unknown"

    line = (
        f"[DBG {time.strftime('%H:%M:%S')} "
        f"rank={rank}/{world} local={local_rank} cuda={cuda_dev}] {msg}"
    )

    try:
        with open(f"/kaggle/working/nanochat_debug_rank{rank}.log", "a") as f:
            f.write(line + "\n")
            f.flush()
    except Exception:
        pass

    if _DEBUG_STDOUT:
        print(line, flush=True)


def should_debug_step(step):
    return _DEBUG_DIST and step < _DEBUG_STEPS


def install_dist_debug_hooks():
    """Wrap dist.all_reduce so we can see which rank enters which collective."""
    global _DEBUG_COLL_SEQ

    if not _DEBUG_DIST:
        return

    if getattr(dist, "_nanochat_debug_wrapped", False):
        return

    dist._nanochat_debug_wrapped = True
    orig_all_reduce = dist.all_reduce

    def all_reduce_debug(tensor, *args, **kwargs):
        global _DEBUG_COLL_SEQ
        _DEBUG_COLL_SEQ += 1

        try:
            numel = tensor.numel()
            shape = tuple(tensor.shape)
            dtype = tensor.dtype
            dev = tensor.device
        except Exception:
            numel, shape, dtype, dev = -1, "?", "?", "?"

        async_op = kwargs.get("async_op", False)

        # The observed crash was a scalar ALLREDUCE NumelIn=1.
        # Always log tiny collectives, first few collectives, and milestones.
        should_log = (numel <= 4) or (_DEBUG_COLL_SEQ <= 20) or (_DEBUG_COLL_SEQ % 1000 == 0)

        if should_log:
            dprint(
                f"ENTER all_reduce seq={_DEBUG_COLL_SEQ} "
                f"numel={numel} shape={shape} dtype={dtype} device={dev} async={async_op}"
            )
            if _DEBUG_STACK:
                stack = ''.join(traceback.format_stack(limit=12))
                dprint("STACK all_reduce:\n" + stack)

        work = orig_all_reduce(tensor, *args, **kwargs)

        if should_log:
            dprint(f"RETURN all_reduce seq={_DEBUG_COLL_SEQ}")

        return work

    dist.all_reduce = all_reduce_debug
    dprint("Installed dist.all_reduce debug hook")
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--stop-at-step", type=int, default=-1, help="absolute step at which this run stops/saves; useful for split runs while --num-iterations keeps the full LR schedule (-1 = same as --num-iterations)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--lr-scale", type=float, default=1.0, help="multiply all learning-rate groups by this factor; useful for fp16 stability tests on T4")
parser.add_argument("--fp16-safe-mlp", action="store_true", help="enable fp16-only clamps around squared-ReLU MLPs and residual streams")
parser.add_argument("--fp16-safe-mlp-clamp", type=float, default=128.0, help="activation clamp used by --fp16-safe-mlp before squared ReLU")
parser.add_argument("--fp16-safe-proj-clamp", type=float, default=1024.0, help="fp16 clamp for attention/MLP projection outputs before residual adds")
parser.add_argument("--fp16-safe-resid-clamp", type=float, default=128.0, help="fp16 clamp for residual stream values")
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
parser.add_argument("--max-consecutive-skipped-steps", type=int, default=20, help="fail after this many consecutive fp16 GradScaler skipped optimizer steps (<=0 disables)")
parser.add_argument("--max-total-skipped-steps", type=int, default=100, help="fail after this many total fp16 GradScaler skipped optimizer steps (<=0 disables)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging
set_fp16_safe_mlp(
    args.fp16_safe_mlp,
    args.fp16_safe_mlp_clamp,
    args.fp16_safe_proj_clamp,
    args.fp16_safe_resid_clamp,
)
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
install_dist_debug_hooks()
dprint(f"After compute_init: ddp={ddp}, rank={ddp_rank}, local_rank={ddp_local_rank}, world_size={ddp_world_size}, device={device}")
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
print0(
    "fp16_safe_mlp: "
    f"{args.fp16_safe_mlp} "
    f"(mlp_clamp={args.fp16_safe_mlp_clamp}, "
    f"proj_clamp={args.fp16_safe_proj_clamp}, "
    f"resid_clamp={args.fp16_safe_resid_clamp})"
)

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Flash Attention status
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    # Use device="meta" to avoid VRAM spike - the weight tensor will be swapped in afterwards
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # Use meta device to avoid unnecessary VRAM allocation
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
if os.environ.get("NANOCHAT_DISABLE_COMPILE", "0") == "1":
    print0("torch.compile disabled by NANOCHAT_DISABLE_COMPILE=1")
    dprint("torch.compile disabled")
else:
    model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe
    dprint("torch.compile enabled")

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # optimal tokens for the model we are about to train

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12) # creates the model on meta device
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
embedding_lr = args.embedding_lr * args.lr_scale
unembedding_lr = args.unembedding_lr * args.lr_scale
matrix_lr = args.matrix_lr * args.lr_scale
scalar_lr = args.scalar_lr * args.lr_scale
if args.lr_scale != 1.0:
    print0(
        f"Applying lr_scale={args.lr_scale}: "
        f"embedding_lr={embedding_lr}, unembedding_lr={unembedding_lr}, "
        f"matrix_lr={matrix_lr}, scalar_lr={scalar_lr}"
    )
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=unembedding_lr * batch_lr_scale,
    embedding_lr=embedding_lr * batch_lr_scale,
    scalar_lr=scalar_lr * batch_lr_scale,
    # Muon hyperparameters
    matrix_lr=matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler("cuda", init_scale=1024) if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")
    if resuming:
        scaler_state_dict = meta_data.get("scaler_state_dict")
        if scaler_state_dict:
            scaler.load_state_dict(scaler_state_dict)
            print0(f"Loaded GradScaler state from checkpoint (scale={scaler.get_scale()})")
        else:
            print0("WARNING: checkpoint has no GradScaler state; starting with fresh GradScaler")
    dprint(f"GradScaler enabled: initial scale={scaler.get_scale()}")

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

stop_at_step = args.stop_at_step if args.stop_at_step > 0 else num_iterations
if stop_at_step > num_iterations:
    raise ValueError(f"--stop-at-step ({stop_at_step}) cannot be greater than --num-iterations/schedule horizon ({num_iterations})")
if args.resume_from_step != -1 and stop_at_step <= args.resume_from_step:
    raise ValueError(f"--stop-at-step ({stop_at_step}) must be greater than --resume-from-step ({args.resume_from_step})")
if stop_at_step != num_iterations:
    print0(f"Split run: using full schedule horizon {num_iterations:,} but stopping/saving at step {stop_at_step:,}")

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer (warms up to 0.97, warms down to 0.90 during LR warmdown)
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
    optimizer_step_count = 0
    skipped_step_count = 0
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]
    if "optimizer_step_count" not in loop_state or "skipped_step_count" not in loop_state:
        print0("WARNING: checkpoint has no optimizer step/skip counters; assuming old checkpoint had one applied update per saved step")
    optimizer_step_count = loop_state.get("optimizer_step_count", step)
    skipped_step_count = loop_state.get("skipped_step_count", 0)
    if not math.isfinite(float(smooth_train_loss)):
        print0("WARNING: loaded nonfinite smooth_train_loss from checkpoint; resetting EMA to 0 for logging")
        smooth_train_loss = 0

consecutive_skipped_steps = 0

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
if scaler is not None:
    print0(
        "fp16 health checks: "
        f"initial GradScaler scale={scaler.get_scale()}, "
        f"max_consecutive_skipped_steps={args.max_consecutive_skipped_steps}, "
        f"max_total_skipped_steps={args.max_total_skipped_steps}"
    )

# Go!
while True:
    if step > stop_at_step:
        raise RuntimeError(f"Internal error: step {step} advanced past stop_at_step {stop_at_step}")
    last_step = step == stop_at_step # loop runs through stop_at_step so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict(), # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "schedule_num_iterations": num_iterations,
                "stop_at_step": stop_at_step,
                "effective_lrs": {
                    "embedding_lr": embedding_lr * batch_lr_scale,
                    "unembedding_lr": unembedding_lr * batch_lr_scale,
                    "matrix_lr": matrix_lr * batch_lr_scale,
                    "scalar_lr": scalar_lr * batch_lr_scale,
                    "lr_scale": args.lr_scale,
                    "batch_lr_scale": batch_lr_scale,
                },
                "fp16_safe_mlp": args.fp16_safe_mlp,
                "fp16_safe_mlp_clamp": args.fp16_safe_mlp_clamp,
                "fp16_safe_proj_clamp": args.fp16_safe_proj_clamp,
                "fp16_safe_resid_clamp": args.fp16_safe_resid_clamp,
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "scaler_scale": scaler.get_scale() if scaler is not None else None,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                    "optimizer_step_count": optimizer_step_count,
                    "skipped_step_count": skipped_step_count,
                    "consecutive_skipped_steps": consecutive_skipped_steps,
                    "max_consecutive_skipped_steps": args.max_consecutive_skipped_steps,
                    "max_total_skipped_steps": args.max_total_skipped_steps,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    debug_this_step = should_debug_step(step)
    if debug_this_step:
        dprint(f"step={step}: START training step, grad_accum_steps={grad_accum_steps}")

    for micro_step in range(grad_accum_steps):
        if debug_this_step:
            dprint(f"step={step} micro={micro_step}: BEFORE forward")

        loss = model(x, y)
        local_loss_finite = torch.isfinite(loss.detach()).all()
        loss_finite_flag = local_loss_finite.to(dtype=torch.int32, device=device)
        if is_ddp_initialized():
            dist.all_reduce(loss_finite_flag, op=dist.ReduceOp.MIN)
        if loss_finite_flag.item() == 0:
            local_loss_value = loss.detach().float().item()
            if not bool(local_loss_finite.item()):
                print(
                    f"[rank {ddp_rank}] ERROR: nonfinite training loss at step={step} "
                    f"micro_step={micro_step}: loss={local_loss_value}, "
                    f"scaler_scale={scaler.get_scale() if scaler is not None else None}",
                    file=sys.stderr,
                    flush=True,
                )
            raise FloatingPointError(
                f"Nonfinite training loss detected at step={step}, micro_step={micro_step}. "
                "Aborting instead of saving a checkpoint with invalid training state."
            )

        if debug_this_step:
            try:
                dprint(f"step={step} micro={micro_step}: AFTER forward raw_loss={loss.detach().float().item()}")
            except Exception as e:
                dprint(f"step={step} micro={micro_step}: AFTER forward, but loss.item() failed: {repr(e)}")

        if debug_this_step:
            dprint(f"step={step} micro={micro_step}: BEFORE train_loss detach")

        train_loss = loss.detach() # for logging

        if debug_this_step:
            dprint(f"step={step} micro={micro_step}: AFTER train_loss detach")
            dprint(f"step={step} micro={micro_step}: BEFORE loss normalize")

        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here

        if debug_this_step:
            dprint(f"step={step} micro={micro_step}: AFTER loss normalize")

        if debug_this_step:
            dprint(f"step={step} micro={micro_step}: BEFORE backward")

        if scaler is not None:
            if debug_this_step:
                dprint(f"step={step} micro={micro_step}: BEFORE scaler.scale(loss)")
            scaled_loss = scaler.scale(loss)
            if debug_this_step:
                dprint(f"step={step} micro={micro_step}: AFTER scaler.scale(loss), BEFORE scaled backward")
            scaled_loss.backward()
            if debug_this_step:
                dprint(f"step={step} micro={micro_step}: AFTER scaled backward call")
        else:
            if debug_this_step:
                dprint(f"step={step} micro={micro_step}: BEFORE raw backward call")
            loss.backward()
            if debug_this_step:
                dprint(f"step={step} micro={micro_step}: AFTER raw backward call")

        if debug_this_step:
            dprint(f"step={step} micro={micro_step}: AFTER backward")

        if debug_this_step:
            dprint(f"step={step} micro={micro_step}: BEFORE next(train_loader)")

        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward

        if debug_this_step:
            dprint(f"step={step} micro={micro_step}: AFTER next(train_loader)")
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if scaler is not None:
        if debug_this_step:
            dprint(f"step={step}: BEFORE scaler.unscale_ scale={scaler.get_scale()}")

        scaler.unscale_(optimizer)

        if debug_this_step:
            dprint(f"step={step}: AFTER scaler.unscale_")

        # Build one explicit overflow flag on every rank.
        # This is safer for this custom distributed optimizer than scaler.step(optimizer),
        # because every rank makes the same Python branch decision.
        optimizer_state = scaler._per_optimizer_states[id(optimizer)]
        found_inf_per_device = optimizer_state["found_inf_per_device"]

        found_inf = torch.zeros((), dtype=torch.float32, device=device)

        if debug_this_step:
            dprint(f"step={step}: local found_inf_per_device keys={list(found_inf_per_device.keys())}")

        for dev_key, v in found_inf_per_device.items():
            if debug_this_step:
                dprint(f"step={step}: local found_inf[{dev_key}]={v.detach().float().item()}")
            found_inf = torch.maximum(found_inf, v.to(device))

        if debug_this_step:
            dprint(f"step={step}: BEFORE found_inf all_reduce local_value={found_inf.item()}")

        if is_ddp_initialized():
            # Every rank must call this exactly once.
            dist.all_reduce(found_inf, op=dist.ReduceOp.MAX)

        if debug_this_step:
            dprint(f"step={step}: AFTER found_inf all_reduce synced_value={found_inf.item()}")

        # Write synchronized value back for scaler.update().
        if len(found_inf_per_device) == 0:
            found_inf_per_device[torch.device(device)] = found_inf
        else:
            for v in found_inf_per_device.values():
                v.copy_(found_inf.to(v.device))

        # Explicit synchronized step/skip. This is the main correction.
        found_inf_value = found_inf.item()
        if found_inf_value == 0.0:
            if debug_this_step:
                dprint(f"step={step}: BEFORE optimizer.step()")
            optimizer.step()
            optimizer_step_count += 1
            consecutive_skipped_steps = 0
            if debug_this_step:
                dprint(f"step={step}: AFTER optimizer.step()")
        else:
            skipped_step_count += 1
            consecutive_skipped_steps += 1
            dprint(f"step={step}: SKIP optimizer.step() due to fp16 overflow, scale={scaler.get_scale()}")
            print0(
                f"WARNING: skipped optimizer.step() at step {step} due to fp16 overflow "
                f"(found_inf={found_inf_value}, scale={scaler.get_scale()}, lrm={lrm:.6f}, "
                f"consecutive_skips={consecutive_skipped_steps}, total_skips={skipped_step_count}, "
                f"optimizer_steps={optimizer_step_count})"
            )
            if args.max_consecutive_skipped_steps > 0 and consecutive_skipped_steps >= args.max_consecutive_skipped_steps:
                raise FloatingPointError(
                    f"Aborting after {consecutive_skipped_steps} consecutive skipped optimizer steps. "
                    "This usually means fp16 training is numerically unstable for this configuration."
                )
            if args.max_total_skipped_steps > 0 and skipped_step_count >= args.max_total_skipped_steps:
                raise FloatingPointError(
                    f"Aborting after {skipped_step_count} total skipped optimizer steps. "
                    "This usually means fp16 training is numerically unstable for this configuration."
                )

        if debug_this_step:
            dprint(f"step={step}: BEFORE scaler.update()")

        scaler.update()

        if debug_this_step:
            dprint(f"step={step}: AFTER scaler.update() new_scale={scaler.get_scale()}")

    else:
        if debug_this_step:
            dprint(f"step={step}: BEFORE optimizer.step() fp32/bf16")
        optimizer.step()
        optimizer_step_count += 1
        if debug_this_step:
            dprint(f"step={step}: AFTER optimizer.step() fp32/bf16")

    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / stop_at_step
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = stop_at_step - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    if step < 5 or step % _TRAIN_PRINT_EVERY == 0 or step == stop_at_step - 1:
        scaler_scale = scaler.get_scale() if scaler is not None else None
        print0(f"step {step:05d}/{stop_at_step:05d} ({pct_done:.2f}%, schedule={num_iterations:05d}) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | scaler: {scaler_scale} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | optimizer_steps: {optimizer_step_count} | skipped_steps: {skipped_step_count} | consecutive_skips: {consecutive_skipped_steps} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
            "train/optimizer_step_count": optimizer_step_count,
            "train/skipped_step_count": skipped_step_count,
            "train/consecutive_skipped_steps": consecutive_skipped_steps,
            "train/scaler_scale": scaler.get_scale() if scaler is not None else None,
        }
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Optimizer steps applied: {optimizer_step_count}")
print0(f"Optimizer steps skipped: {skipped_step_count}")
if scaler is not None:
    print0(f"Final GradScaler scale: {scaler.get_scale()}")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Stop at step": stop_at_step,
        "Number of training tokens": total_tokens,
        "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
        "DDP world size": ddp_world_size,
        "warmup_steps": args.warmup_steps,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
        "lr_scale": args.lr_scale,
        "effective_embedding_lr": embedding_lr * batch_lr_scale,
        "effective_unembedding_lr": unembedding_lr * batch_lr_scale,
        "effective_matrix_lr": matrix_lr * batch_lr_scale,
        "effective_scalar_lr": scalar_lr * batch_lr_scale,
        "fp16_safe_mlp": args.fp16_safe_mlp,
        "fp16_safe_mlp_clamp": args.fp16_safe_mlp_clamp,
        "fp16_safe_proj_clamp": args.fp16_safe_proj_clamp,
        "fp16_safe_resid_clamp": args.fp16_safe_resid_clamp,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
        "Optimizer steps applied": optimizer_step_count,
        "Optimizer steps skipped": skipped_step_count,
        "Final GradScaler scale": scaler.get_scale() if scaler is not None else None,
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
