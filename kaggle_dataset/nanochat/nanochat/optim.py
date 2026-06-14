"""
A nice and efficient mixed AdamW/Muon Combined Optimizer.
Usually the embeddings and scalars go into AdamW, and the matrix parameters go into Muon.
Two versions are provided (MuonAdamW, DistMuonAdamW), for single GPU and distributed.

Addapted from: https://github.com/KellerJordan/modded-nanogpt
Further contributions from @karpathy and @chrisjmccormick.
"""

import os
import time

import torch
import torch.distributed as dist
from torch import Tensor

# -----------------------------------------------------------------------------
# Distributed optimizer debugging
# Enable with NANOCHAT_DEBUG_DIST=1. Logs go to per-rank files under /kaggle/working.
_DEBUG_DIST = os.environ.get("NANOCHAT_DEBUG_DIST", "0") == "1"
_DEBUG_STDOUT = os.environ.get("NANOCHAT_DEBUG_STDOUT", "0") == "1"
_ADAMW_ALLREDUCE = os.environ.get("NANOCHAT_ADAMW_ALLREDUCE", "0") == "1"
_SERIAL_OPTIM_COMM = os.environ.get("NANOCHAT_SERIAL_OPTIM_COMM", "0") == "1"

def _debug_rank_info():
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world

def _debug_log(msg: str) -> None:
    if not _DEBUG_DIST:
        return
    rank, local_rank, world = _debug_rank_info()
    line = f"[OPTDBG {time.strftime('%H:%M:%S')} rank={rank}/{world} local={local_rank}] {msg}"
    try:
        with open(f"/kaggle/working/nanochat_optim_debug_rank{rank}.log", "a") as f:
            f.write(line + "\n")
            f.flush()
    except Exception:
        pass
    if _DEBUG_STDOUT:
        print(line, flush=True)

def _tensor_desc(t: Tensor | None) -> str:
    if t is None:
        return "None"
    return f"shape={tuple(t.shape)} numel={t.numel()} dtype={t.dtype} device={t.device}"

# -----------------------------------------------------------------------------
"""
Good old AdamW optimizer, fused kernel.
https://arxiv.org/abs/1711.05101
"""

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,              # (32768, 768) - parameter tensor
    grad: Tensor,           # (32768, 768) - gradient, same shape as p
    exp_avg: Tensor,        # (32768, 768) - first moment, same shape as p
    exp_avg_sq: Tensor,     # (32768, 768) - second moment, same shape as p
    step_t: Tensor,         # () - 0-D CPU tensor, step count
    lr_t: Tensor,           # () - 0-D CPU tensor, learning rate
    beta1_t: Tensor,        # () - 0-D CPU tensor, beta1
    beta2_t: Tensor,        # () - 0-D CPU tensor, beta2
    eps_t: Tensor,          # () - 0-D CPU tensor, epsilon
    wd_t: Tensor,           # () - 0-D CPU tensor, weight decay
) -> None:
    """
    Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update
    All in one compiled graph to eliminate Python overhead between ops.
    The 0-D CPU tensors avoid recompilation when hyperparameter values change.
    """
    # Weight decay (decoupled, applied before the update)
    p.mul_(1 - lr_t * wd_t)
    # Update running averages (lerp_ is cleaner and fuses well)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    # Bias corrections
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    # Compute update and apply
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

# -----------------------------------------------------------------------------
"""
Muon optimizer adapted and simplified from modded-nanogpt.
https://github.com/KellerJordan/modded-nanogpt

Background:
Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
zero even beyond the point where the iteration no longer converges all the way to one everywhere
on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
performance at all relative to UV^T, where USV^T = G is the SVD.

Here, an alternative to Newton-Schulz iteration with potentially better convergence properties:
Polar Express Sign Method for orthogonalization.
https://arxiv.org/pdf/2505.16932
by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.

NorMuon variance reduction: per-neuron/column adaptive learning rate that normalizes
update scales after orthogonalization (Muon's output has non-uniform scales across neurons).
https://arxiv.org/pdf/2510.05491

Some of the changes in nanochat implementation:
- Uses a simpler, more general approach to parameter grouping and stacking
- Uses a single fused kernel for the momentum -> polar_express -> variance_reduction -> update step
- Makes no assumptions about model architecture (e.g. that attention weights are fused into QKVO format)
"""

# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# From https://arxiv.org/pdf/2505.16932
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,          # (12, 768, 3072) - stacked gradients
    stacked_params: Tensor,         # (12, 768, 3072) - stacked parameters
    momentum_buffer: Tensor,        # (12, 768, 3072) - first moment buffer
    second_momentum_buffer: Tensor, # (12, 768, 1) or (12, 1, 3072) - factored second moment
    momentum_t: Tensor,             # () - 0-D CPU tensor, momentum coefficient
    lr_t: Tensor,                   # () - 0-D CPU tensor, learning rate
    wd_t: Tensor,                   # () - 0-D CPU tensor, weight decay
    beta2_t: Tensor,                # () - 0-D CPU tensor, beta2 for second moment
    ns_steps: int,                  # 5 - number of Newton-Schulz/Polar Express iterations
    red_dim: int,                   # -1 or -2 - reduction dimension for variance
) -> None:
    """
    Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update
    All in one compiled graph to eliminate Python overhead between ops.
    Some of the constants are 0-D CPU tensors to avoid recompilation when values change.
    """

    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar express
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1): # Tall matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else: # Wide matrix (original math)
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

# -----------------------------------------------------------------------------
# Single GPU version of the MuonAdamW optimizer.
# Used mostly for reference, debugging and testing.

class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Muon for 2D matrix params, AdamW for others, single GPU version.

    AdamW - Fused AdamW optimizer step.

    Muon - MomentUm Orthogonalized by Newton-schulz
    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - The Muon optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        # AdamW tensors
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # Muon tensors
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: dict) -> None:
        """
        AdamW update for each param in the group individually.
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
        """
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]

            # State init
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            state['step'] += 1

            # Fill 0-D tensors with current values
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])

            # Fused update: weight_decay -> momentum -> bias_correction -> param_update
            adamw_step_fused(
                p, grad, exp_avg, exp_avg_sq,
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

    def _step_muon(self, group: dict) -> None:
        """
        Muon update for all params in the group (stacked for efficiency).
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
        """
        params: list[Tensor] = group['params']
        if not params:
            return

        # Get or create group-level buffers (stored in first param's state for convenience)
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        # Momentum for every individual parameter
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        momentum_buffer = state["momentum_buffer"]

        # Second momentum buffer is factored, either per-row or per-column
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Stack grads and params (NOTE: this assumes all params have the same shape)
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        # Fill all the 0-D tensors with current values
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        # Single fused kernel: momentum -> polar_express -> variance_reduction -> update
        muon_step_fused(
            stacked_grads,
            stacked_params,
            momentum_buffer,
            second_momentum_buffer,
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )

        # Copy back to original params
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

# -----------------------------------------------------------------------------
# Distributed version of the MuonAdamW optimizer.
# Used for training on multiple GPUs.

class DistMuonAdamW(torch.optim.Optimizer):
    """
    Combined distributed optimizer: Muon for 2D matrix params, AdamW for others.

    See MuonAdamW for the algorithmic details of each optimizer. This class adds
    distributed communication to enable multi-GPU training without PyTorch DDP.

    Design Goals:
    - Overlap communication with computation (async ops)
    - Minimize memory by sharding optimizer states across ranks (ZeRO-2 style)
    - Batch small tensors into single comm ops where possible

    Communication Pattern (3-phase async):
    We use a 3-phase structure to maximize overlap between communication and compute:

        Phase 1: Launch all async reduce ops
            - Kick off all reduce_scatter/all_reduce operations
            - Don't wait - let them run in background while we continue

        Phase 2: Wait for reduces, compute updates, launch gathers
            - For each group: wait for its reduce, compute the update, launch gather
            - By processing groups in order, earlier gathers run while later computes happen

        Phase 3: Wait for gathers, copy back
            - Wait for all gathers to complete
            - Copy updated params back to original tensors (Muon only)

    AdamW Communication (ZeRO-2 style):
    - Small params (<1024 elements): all_reduce gradients, update full param on each rank.
      Optimizer state is replicated but these params are tiny (scalars, biases).
    - Large params: reduce_scatter gradients so each rank gets 1/N of the grad, update
      only that slice, then all_gather the updated slices. Optimizer state (exp_avg,
      exp_avg_sq) is sharded - each rank only stores state for its slice.
      Requires param.shape[0] divisible by world_size.

    Muon Communication (stacked + chunked):
    - All params in a Muon group must have the same shape (caller's responsibility).
    - Stack all K params into a single (K, *shape) tensor for efficient comm.
    - Divide K params across N ranks: each rank "owns" ceil(K/N) params.
    - reduce_scatter the stacked grads so each rank gets its chunk.
    - Each rank computes Muon update only for params it owns.
    - all_gather the updated params back to all ranks.
    - Optimizer state (momentum_buffer, second_momentum_buffer) is sharded by chunk.
    - Padding: if K doesn't divide evenly, we zero-pad to (ceil(K/N) * N) for comm,
      then ignore the padding when copying back.

    Buffer Reuse:
    - For Muon, we allocate stacked_grads for reduce_scatter input, then reuse the
      same buffer as the output for all_gather (stacked_params). This saves memory
      since we don't need both buffers simultaneously.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._debug_step = 0

    def _reduce_adamw(self, group: dict, world_size: int, group_idx: int) -> dict:
        """Launch async reduce ops for AdamW group. Returns info dict with per-param infos."""
        param_infos = {}
        params = group['params']
        grads = []
        for param_idx, p in enumerate(params):
            grad = p.grad
            if grad is None:
                _debug_log(f"step={self._debug_step} adamw group={group_idx} param={param_idx} grad=None")
                raise RuntimeError(f"AdamW parameter has no gradient: group={group_idx} param={param_idx}")
            grads.append(grad)

        total_numel = sum(g.numel() for g in grads)
        can_all_reduce_group = all(p.numel() < 1024 or _ADAMW_ALLREDUCE for p in params)
        if _SERIAL_OPTIM_COMM and can_all_reduce_group and len(params) > 1 and total_numel < 4096:
            flat_grad = torch.cat([g.reshape(-1) for g in grads])
            _debug_log(
                f"step={self._debug_step} ENTER adamw_packed_all_reduce "
                f"group={group_idx} params={len(params)} flat_{_tensor_desc(flat_grad)}"
            )
            dist.all_reduce(flat_grad, op=dist.ReduceOp.AVG)
            _debug_log(f"step={self._debug_step} DONE adamw_packed_all_reduce group={group_idx}")

            offset = 0
            for param_idx, (p, grad) in enumerate(zip(params, grads)):
                next_offset = offset + grad.numel()
                grad_slice = flat_grad[offset:next_offset].view_as(grad)
                param_infos[p] = dict(
                    future=None,
                    grad_slice=grad_slice,
                    is_small=True,
                    param_idx=param_idx,
                    group_idx=group_idx,
                )
                offset = next_offset
            return dict(param_infos=param_infos, flat_grad=flat_grad)

        for param_idx, p in enumerate(group['params']):
            grad = p.grad
            if p.numel() < 1024 or _ADAMW_ALLREDUCE:
                # Small params, or Kaggle-safe fallback: all_reduce gradients and
                # update the full AdamW parameter on every rank. This avoids the
                # large-param all_gather path, which can hang on Kaggle 2xT4.
                _debug_log(
                    f"step={self._debug_step} ENTER adamw_all_reduce "
                    f"group={group_idx} param={param_idx} param_{_tensor_desc(p)} grad_{_tensor_desc(grad)}"
                )
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                _debug_log(
                    f"step={self._debug_step} RETURN adamw_all_reduce "
                    f"group={group_idx} param={param_idx} replicated={_ADAMW_ALLREDUCE and p.numel() >= 1024}"
                )
                if _SERIAL_OPTIM_COMM:
                    _debug_log(f"step={self._debug_step} WAIT immediate adamw_all_reduce group={group_idx} param={param_idx}")
                    future.wait()
                    _debug_log(f"step={self._debug_step} DONE immediate adamw_all_reduce group={group_idx} param={param_idx}")
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True, param_idx=param_idx, group_idx=group_idx)
            else:
                # Large params: reduce_scatter
                assert grad.shape[0] % world_size == 0, f"AdamW reduce_scatter requires shape[0] ({grad.shape[0]}) divisible by world_size ({world_size})"
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                _debug_log(
                    f"step={self._debug_step} ENTER adamw_reduce_scatter "
                    f"group={group_idx} param={param_idx} param_{_tensor_desc(p)} grad_{_tensor_desc(grad)}"
                )
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                _debug_log(
                    f"step={self._debug_step} RETURN adamw_reduce_scatter "
                    f"group={group_idx} param={param_idx} grad_slice_{_tensor_desc(grad_slice)}"
                )
                if _SERIAL_OPTIM_COMM:
                    _debug_log(f"step={self._debug_step} WAIT immediate adamw_reduce_scatter group={group_idx} param={param_idx}")
                    future.wait()
                    _debug_log(f"step={self._debug_step} DONE immediate adamw_reduce_scatter group={group_idx} param={param_idx}")
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False, param_idx=param_idx, group_idx=group_idx)
        return dict(param_infos=param_infos)

    def _reduce_muon(self, group: dict, world_size: int, group_idx: int) -> dict:
        """Launch async reduce op for Muon group. Returns info dict."""
        _debug_log(f"step={self._debug_step} ENTER reduce_muon group={group_idx}")
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # Stack grads and zero-pad to padded_num_params
        _debug_log(f"step={self._debug_step} PREP muon_stack group={group_idx} params={len(params)} shape={shape}")
        grad_stack = torch.stack([p.grad for p in params])
        _debug_log(f"step={self._debug_step} DONE muon_stack group={group_idx} grad_stack_{_tensor_desc(grad_stack)}")
        _debug_log(f"step={self._debug_step} PREP muon_stacked_grads_alloc group={group_idx} padded_num_params={padded_num_params}")
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        _debug_log(f"step={self._debug_step} DONE muon_stacked_grads_alloc group={group_idx} stacked_grads_{_tensor_desc(stacked_grads)}")
        _debug_log(f"step={self._debug_step} PREP muon_stacked_grads_copy group={group_idx}")
        stacked_grads[:len(params)].copy_(grad_stack)
        _debug_log(f"step={self._debug_step} DONE muon_stacked_grads_copy group={group_idx}")
        if len(params) < padded_num_params:
            _debug_log(f"step={self._debug_step} PREP muon_stacked_grads_zero_pad group={group_idx}")
            stacked_grads[len(params):].zero_()
            _debug_log(f"step={self._debug_step} DONE muon_stacked_grads_zero_pad group={group_idx}")

        # Reduce_scatter to get this rank's chunk
        _debug_log(f"step={self._debug_step} PREP muon_grad_chunk_alloc group={group_idx} chunk_size={chunk_size}")
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        _debug_log(f"step={self._debug_step} DONE muon_grad_chunk_alloc group={group_idx} grad_chunk_{_tensor_desc(grad_chunk)}")
        _debug_log(
            f"step={self._debug_step} ENTER muon_reduce_scatter "
            f"group={group_idx} params={len(params)} chunk_size={chunk_size} "
            f"stacked_grads_{_tensor_desc(stacked_grads)}"
        )
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()
        _debug_log(
            f"step={self._debug_step} RETURN muon_reduce_scatter "
            f"group={group_idx} grad_chunk_{_tensor_desc(grad_chunk)}"
        )
        if _SERIAL_OPTIM_COMM:
            _debug_log(f"step={self._debug_step} WAIT immediate muon_reduce_scatter group={group_idx}")
            future.wait()
            _debug_log(f"step={self._debug_step} DONE immediate muon_reduce_scatter group={group_idx}")

        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size, group_idx=group_idx)

    def _compute_adamw(self, group: dict, info: dict, gather_list: list, rank: int, world_size: int) -> None:
        """Wait for reduce, compute AdamW updates, launch gathers for large params."""
        param_infos = info['param_infos']
        for p in group['params']:
            pinfo = param_infos[p]
            group_idx = pinfo['group_idx']
            param_idx = pinfo['param_idx']
            _debug_log(f"step={self._debug_step} WAIT adamw_reduce group={group_idx} param={param_idx} small={pinfo['is_small']}")
            if pinfo['future'] is not None:
                pinfo['future'].wait()
            _debug_log(f"step={self._debug_step} DONE adamw_reduce group={group_idx} param={param_idx} small={pinfo['is_small']}")
            grad_slice = pinfo['grad_slice']
            state = self.state[p]

            # For small params, operate on full param; for large, operate on slice
            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]

            # State init
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            state['step'] += 1

            # Fill 0-D tensors and run fused kernel
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(
                p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

            # Large params need all_gather
            if not pinfo['is_small']:
                gathered_param = torch.empty_like(p)
                _debug_log(
                    f"step={self._debug_step} ENTER adamw_all_gather "
                    f"group={group_idx} param={param_idx} output_{_tensor_desc(gathered_param)} input_{_tensor_desc(p_slice)}"
                )
                future = dist.all_gather_into_tensor(gathered_param, p_slice, async_op=True).get_future()
                _debug_log(f"step={self._debug_step} RETURN adamw_all_gather group={group_idx} param={param_idx}")
                gather_list.append(dict(
                    future=future,
                    param=p,
                    gathered_param=gathered_param,
                    params=None,
                    kind="adamw",
                    group_idx=group_idx,
                    param_idx=param_idx,
                ))

    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        """Wait for reduce, compute Muon updates, launch gather."""
        group_idx = info["group_idx"]
        _debug_log(f"step={self._debug_step} WAIT muon_reduce group={group_idx}")
        info['future'].wait()
        _debug_log(f"step={self._debug_step} DONE muon_reduce group={group_idx}")
        params = group['params']
        chunk_size = info['chunk_size']
        grad_chunk = info['grad_chunk']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # How many params does this rank own?
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        # Get or create group-level state
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Build output buffer for all_gather
        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            # Fill 0-D tensors and run fused kernel
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
            self._muon_wd_t.fill_(group["weight_decay"])
            muon_step_fused(
                grad_chunk[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                group["ns_steps"], red_dim,
            )
            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        # Reuse stacked_grads buffer for all_gather output
        stacked_params = info["stacked_grads"]
        _debug_log(f"step={self._debug_step} ENTER muon_all_gather group={group_idx}")
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        _debug_log(f"step={self._debug_step} RETURN muon_all_gather group={group_idx}")
        if _SERIAL_OPTIM_COMM:
            _debug_log(f"step={self._debug_step} WAIT immediate muon_all_gather group={group_idx}")
            future.wait()
            _debug_log(f"step={self._debug_step} DONE immediate muon_all_gather group={group_idx}")
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params, kind="muon", group_idx=group_idx, param_idx=None))

    def _finish_gathers(self, gather_list: list) -> None:
        """Wait for all gathers and copy Muon params back."""
        for info in gather_list:
            _debug_log(
                f"step={self._debug_step} WAIT {info['kind']}_gather "
                f"group={info['group_idx']} param={info['param_idx']}"
            )
            info["future"].wait()
            _debug_log(
                f"step={self._debug_step} DONE {info['kind']}_gather "
                f"group={info['group_idx']} param={info['param_idx']}"
            )
            if info["kind"] == "adamw":
                info["param"].copy_(info["gathered_param"])
            elif info["params"] is not None:
                # Muon: copy from stacked buffer back to individual params
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        self._debug_step += 1
        _debug_log(f"BEGIN optimizer.step internal_step={self._debug_step} groups={len(self.param_groups)}")

        # Phase 1: launch all async reduce ops
        reduce_infos: list[dict] = []
        for group_idx, group in enumerate(self.param_groups):
            _debug_log(
                f"step={self._debug_step} REDUCE_PHASE group={group_idx} "
                f"kind={group['kind']} params={len(group['params'])}"
            )
            if group['kind'] == 'adamw':
                reduce_infos.append(self._reduce_adamw(group, world_size, group_idx))
            elif group['kind'] == 'muon':
                reduce_infos.append(self._reduce_muon(group, world_size, group_idx))
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # Phase 2: wait for reduces, compute updates, launch gathers
        gather_list: list[dict] = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group['kind'] == 'adamw':
                self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group['kind'] == 'muon':
                self._compute_muon(group, info, gather_list, rank)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # Phase 3: wait for gathers, copy back
        self._finish_gathers(gather_list)
        _debug_log(f"END optimizer.step internal_step={self._debug_step}")
