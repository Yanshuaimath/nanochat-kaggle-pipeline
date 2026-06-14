"""
Standard token-level PPO with reward-model training for nanochat.

Compared with scripts.chat_ppo, this version follows the common RLHF PPO recipe:
- train/load a scalar reward model from preferences
- sample completions from the current policy
- shape per-token rewards with KL to a frozen reference policy
- add the reward-model score to the final sampled response token
- compute token-level GAE advantages with a learned value head
- optimize clipped policy loss + clipped value loss - entropy bonus

JSONL preference format:
{"prompt": "Question text", "chosen": "Preferred assistant response", "rejected": "Dispreferred assistant response"}

Examples:
python3 -m scripts.chat_ppo_standard --preference-source=gsm8k
python3 -m scripts.chat_ppo_standard --preference-source=jsonl --preference-path=/path/prefs.jsonl
python3 -m scripts.chat_ppo_standard --rm-load-path=/path/reward_000200.pt
"""

import argparse
import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb

from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.common import DummyWandb, autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0
from nanochat.engine import Engine
from nanochat.report import get_report
from scripts.chat_ppo import (
    RewardModel,
    all_reduce_reward_model_grads,
    build_conversation,
    collate_conversations,
    decode_generated_text,
    evaluate_reward_model,
    forward_hidden,
    get_ppo_lr_multiplier,
    load_preferences,
    maybe_load_reward_model_checkpoint,
    pad_rollout_batch,
    policy_tensors_from_sequence,
    reward_model_optimizer,
    reward_sequence_from_rollout,
    sample_rollout_sequence,
    stable_rollout_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Standard token-level PPO with reward model training for nanochat")
    parser.add_argument("--run", type=str, default="dummy")
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--policy-source", type=str, default="sft", choices=["sft", "rl", "ppo", "ppo_standard"])
    parser.add_argument("--policy-tag", type=str, default=None)
    parser.add_argument("--policy-step", type=int, default=None)
    parser.add_argument("--reference-source", type=str, default="sft", choices=["sft", "rl", "ppo", "ppo_standard"])
    parser.add_argument("--reference-tag", type=str, default=None)
    parser.add_argument("--reference-step", type=int, default=None)
    parser.add_argument("--preference-source", type=str, default="gsm8k", choices=["gsm8k", "jsonl"])
    parser.add_argument("--preference-path", type=str, default=None)
    parser.add_argument("--max-train-examples", type=int, default=4096)
    parser.add_argument("--max-val-examples", type=int, default=256)
    parser.add_argument("--rm-batch-size", type=int, default=8)
    parser.add_argument("--rm-epochs", type=int, default=1)
    parser.add_argument("--rm-lr", type=float, default=1e-4)
    parser.add_argument("--rm-train-backbone", type=int, default=0, help="0=head only, 1=fine-tune reward backbone")
    parser.add_argument("--rm-load-path", type=str, default=None)
    parser.add_argument("--value-load-path", type=str, default=None)
    parser.add_argument("--ppo-steps", type=int, default=200)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    parser.add_argument("--device-batch-size", type=int, default=8)
    parser.add_argument("--ppo-minibatch-size", type=int, default=0, help="0=use --device-batch-size")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0, help="0 disables top-k; PPO ratios require full-support sampling")
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--advantage-whiten", type=int, default=1)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--value-lr", type=float, default=1e-4)
    parser.add_argument("--value-train-backbone", type=int, default=1, help="1=value loss updates policy backbone, 0=value head only")
    parser.add_argument("--embedding-lr", type=float, default=0.05)
    parser.add_argument("--unembedding-lr", type=float, default=0.002)
    parser.add_argument("--matrix-lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--init-lr-frac", type=float, default=0.2)
    parser.add_argument("--save-every", type=int, default=100)
    return parser.parse_args()


class ValueHead(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.head = nn.Linear(n_embd, 1, bias=False)

    def forward(self, hidden):
        return self.head(hidden).squeeze(-1)


def maybe_load_value_head_checkpoint(value_head, load_path, device):
    if load_path is None:
        return
    checkpoint = torch.load(load_path, map_location=device)
    state_dict = checkpoint["value_head"] if "value_head" in checkpoint else checkpoint
    missing_keys, unexpected_keys = value_head.load_state_dict(state_dict, strict=False)
    assert not missing_keys, f"Missing value-head keys in checkpoint {load_path}: {missing_keys}"
    assert not unexpected_keys, f"Unexpected value-head keys in checkpoint {load_path}: {unexpected_keys}"
    print0(f"Loaded value head checkpoint from {load_path}")


def all_reduce_trainable_grads(module, ddp_world_size):
    if ddp_world_size <= 1:
        return
    for param in module.parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(ddp_world_size)


def apply_sampling_warp(logits, temperature, top_k):
    assert temperature > 0.0, "PPO logprob ratios require a positive sampling temperature"
    warped = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, warped.size(-1))
        topk_values, topk_indices = torch.topk(warped, k, dim=-1)
        masked = torch.full_like(warped, torch.finfo(warped.dtype).min)
        warped = masked.scatter(-1, topk_indices, topk_values)
    return warped


def compute_policy_outputs(model, inputs, targets, temperature=1.0, top_k=0):
    valid = targets >= 0
    safe_targets = targets.clamp(min=0)
    logits = model(inputs)
    logits = apply_sampling_warp(logits, temperature, top_k)
    logprobs = F.log_softmax(logits, dim=-1)
    probs = logprobs.exp()
    token_logps = logprobs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    entropy = -(probs * logprobs).sum(dim=-1)
    token_logps = token_logps.masked_fill(~valid, 0.0)
    entropy = entropy.masked_fill(~valid, 0.0)
    return token_logps, entropy, valid


def compute_values(policy_model, value_head, inputs, train_backbone=True):
    if train_backbone:
        hidden = forward_hidden(policy_model, inputs)
    else:
        with torch.no_grad():
            hidden = forward_hidden(policy_model, inputs)
    return value_head(hidden)


@torch.no_grad()
def collect_policy_outputs(model, inputs_all, targets_all, batch_size, temperature=1.0, top_k=0):
    logp_chunks = []
    entropy_chunks = []
    valid_chunks = []
    for b0 in range(0, inputs_all.size(0), batch_size):
        b1 = min(b0 + batch_size, inputs_all.size(0))
        token_logps, entropy, valid = compute_policy_outputs(
            model,
            inputs_all[b0:b1],
            targets_all[b0:b1],
            temperature,
            top_k,
        )
        logp_chunks.append(token_logps)
        entropy_chunks.append(entropy)
        valid_chunks.append(valid)
    return torch.cat(logp_chunks, dim=0), torch.cat(entropy_chunks, dim=0), torch.cat(valid_chunks, dim=0)


@torch.no_grad()
def collect_values(policy_model, value_head, inputs_all, batch_size):
    chunks = []
    for b0 in range(0, inputs_all.size(0), batch_size):
        b1 = min(b0 + batch_size, inputs_all.size(0))
        chunks.append(compute_values(policy_model, value_head, inputs_all[b0:b1]))
    return torch.cat(chunks, dim=0)


def policy_sequence_from_rollout(tokenizer, sequence, mask):
    bos = tokenizer.get_bos_token_id()
    while sequence and sequence[-1] == bos:
        sequence = sequence[:-1]
        mask = mask[:-1]
    return sequence, mask


def masked_global_whiten(values, mask, ddp_world_size):
    mask_f = mask.to(values.dtype)
    count = mask_f.sum()
    total = (values * mask_f).sum()
    sumsq = (values.square() * mask_f).sum()
    if ddp_world_size > 1:
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(sumsq, op=dist.ReduceOp.SUM)
    if count.item() <= 1:
        return torch.zeros_like(values)
    mean = total / count
    var = (sumsq / count - mean.square()).clamp(min=1e-8)
    whitened = (values - mean) * torch.rsqrt(var)
    return whitened.masked_fill(~mask, 0.0)


def global_token_normalizer(valid_f, ddp_world_size):
    count = valid_f.sum()
    if ddp_world_size > 1:
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        count = count / ddp_world_size
    return count.clamp(min=1.0)


def add_terminal_rewards(token_rewards, sequence_rewards, valid):
    token_rewards = token_rewards.clone()
    for row, reward in enumerate(sequence_rewards):
        valid_positions = torch.nonzero(valid[row], as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            continue
        token_rewards[row, valid_positions[-1]] += reward
    return token_rewards


def compute_gae_returns(rewards, values, valid, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    B, _ = rewards.shape
    for b in range(B):
        valid_positions = torch.nonzero(valid[b], as_tuple=False).flatten()
        last_gae = rewards.new_zeros(())
        for i in reversed(range(valid_positions.numel())):
            t = valid_positions[i]
            has_next = i + 1 < valid_positions.numel()
            next_t = valid_positions[i + 1] if has_next else None
            next_value = values[b, next_t] if has_next else values.new_zeros(())
            next_nonterminal = 1.0 if has_next else 0.0
            delta = rewards[b, t] + gamma * next_value * next_nonterminal - values[b, t]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            advantages[b, t] = last_gae
            returns[b, t] = advantages[b, t] + values[b, t]
    return advantages, returns


def build_token_rewards(old_token_logps, ref_token_logps, valid, sequence_rewards, kl_beta):
    valid_f = valid.to(old_token_logps.dtype)
    token_kl = (old_token_logps - ref_token_logps) * valid_f
    token_rewards = -kl_beta * token_kl
    token_rewards = add_terminal_rewards(token_rewards, sequence_rewards, valid)
    return token_rewards, token_kl


def save_ppo_state(args, policy_model, reward_model, value_head, step):
    base_dir = get_base_dir()
    depth = policy_model.config.n_layer
    model_tag = args.policy_tag if args.policy_tag else f"d{depth}"
    policy_dir = os.path.join(base_dir, "chatppo_standard_checkpoints", model_tag)
    reward_dir = os.path.join(base_dir, "chatppo_standard_reward_checkpoints", model_tag)
    value_dir = os.path.join(base_dir, "chatppo_standard_value_checkpoints", model_tag)
    save_checkpoint(policy_dir, step, policy_model.state_dict(), None, {"model_config": policy_model.config.__dict__})
    os.makedirs(reward_dir, exist_ok=True)
    os.makedirs(value_dir, exist_ok=True)
    torch.save({"reward_model": reward_model.state_dict()}, os.path.join(reward_dir, f"reward_{step:06d}.pt"))
    torch.save({"value_head": value_head.state_dict()}, os.path.join(value_dir, f"value_{step:06d}.pt"))
    print0(f"Saved standard PPO checkpoints to {policy_dir}, {reward_dir}, and {value_dir}")


def resolve_reference_checkpoint(args):
    same_source = args.reference_source == args.policy_source
    ref_tag = args.reference_tag if args.reference_tag is not None else (args.policy_tag if same_source else None)
    ref_step = args.reference_step if args.reference_step is not None else (args.policy_step if same_source else None)
    return ref_tag, ref_step


def main():
    args = parse_args()
    user_config = vars(args).copy()
    assert args.prompt_batch_size > 0, "prompt_batch_size must be positive"
    assert args.device_batch_size > 0, "device_batch_size must be positive"
    assert args.ppo_steps > 0, "ppo_steps must be positive"
    assert args.ppo_epochs > 0, "ppo_epochs must be positive"
    assert args.max_new_tokens > 0, "max_new_tokens must be positive"
    assert 0.0 <= args.gae_lambda <= 1.0, "gae_lambda must be in [0, 1]"
    assert 0.0 <= args.gamma <= 1.0, "gamma must be in [0, 1]"
    assert args.temperature > 0.0, "PPO training requires stochastic sampling; use --temperature > 0"
    assert args.top_k == 0, "PPO ratios in this script require full-support sampling; use --top-k=0"
    assert args.clip_epsilon >= 0.0, "clip_epsilon must be non-negative"
    assert args.value_clip_epsilon >= 0.0, "value_clip_epsilon must be non-negative"
    assert args.kl_beta >= 0.0, "kl_beta must be non-negative"
    assert args.init_lr_frac >= 0.0, "init_lr_frac must be non-negative"
    assert args.value_train_backbone in (0, 1), "value_train_backbone must be 0 or 1"
    assert args.save_every > 0, "save_every must be positive"

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0

    use_dummy_wandb = args.run == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-ppo-standard", name=args.run, config=user_config)

    policy_model, tokenizer, meta = load_model(args.policy_source, device, phase="train", model_tag=args.policy_tag, step=args.policy_step)
    ref_tag, ref_step = resolve_reference_checkpoint(args)
    ref_model, _, _ = load_model(args.reference_source, device, phase="eval", model_tag=ref_tag, step=ref_step)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)
    engine = Engine(policy_model, tokenizer)

    reward_backbone, _, _ = load_model(args.policy_source, device, phase="train", model_tag=args.policy_tag, step=args.policy_step)
    reward_model = RewardModel(reward_backbone).to(device)
    maybe_load_reward_model_checkpoint(reward_model, args.rm_load_path, device)
    rm_optimizer = reward_model_optimizer(reward_model, args.rm_lr, bool(args.rm_train_backbone))

    value_head = ValueHead(policy_model.config.n_embd).to(device)
    maybe_load_value_head_checkpoint(value_head, args.value_load_path, device)
    value_optimizer = torch.optim.AdamW(value_head.parameters(), lr=args.value_lr)

    prefs = load_preferences(args)
    random.Random(42).shuffle(prefs)
    split = min(args.max_train_examples, len(prefs))
    train_prefs = prefs[:split]
    val_prefs = prefs[split:split + args.max_val_examples]
    assert train_prefs, "No training preferences available; increase --max-train-examples or provide a non-empty preference dataset"
    prompts = [row["prompt"] for row in train_prefs]
    print0(f"Standard PPO prefs train: {len(train_prefs)} | val: {len(val_prefs)}")

    for epoch in range(args.rm_epochs):
        random.Random(100 + epoch).shuffle(train_prefs)
        for start in range(0, len(train_prefs), ddp_world_size * args.rm_batch_size):
            local_start = start + ddp_rank * args.rm_batch_size
            batch = train_prefs[local_start:local_start + args.rm_batch_size]
            rm_optimizer.zero_grad(set_to_none=True)
            if batch:
                chosen = [build_conversation(row["prompt"], row["chosen"]) for row in batch]
                rejected = [build_conversation(row["prompt"], row["rejected"]) for row in batch]
                chosen_ids, _, _, chosen_last = collate_conversations(tokenizer, chosen, device)
                rejected_ids, _, _, rejected_last = collate_conversations(tokenizer, rejected, device)
                chosen_scores = reward_model(chosen_ids, chosen_last)
                rejected_scores = reward_model(rejected_ids, rejected_last)
                rm_loss = -F.logsigmoid(chosen_scores - rejected_scores).mean()
                rm_loss.backward()
            all_reduce_reward_model_grads(reward_model, ddp_world_size)
            rm_optimizer.step()

        if val_prefs:
            reward_model.eval()
            rm_acc = evaluate_reward_model(reward_model, tokenizer, val_prefs, args.rm_batch_size, device, ddp_rank, ddp_world_size)
            print0(f"Reward epoch {epoch} | val_pref_acc: {rm_acc:.4f}")
            wandb_run.log({"reward_epoch": epoch, "reward_val_pref_acc": rm_acc})
            reward_model.train()

    optimizer = policy_model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        weight_decay=args.weight_decay,
    )
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]

    reward_model.eval()
    prompt_cursor = ddp_rank
    minibatch_size = args.ppo_minibatch_size if args.ppo_minibatch_size > 0 else args.device_batch_size
    for step in range(args.ppo_steps):
        lrm = get_ppo_lr_multiplier(step, args.ppo_steps)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm

        batch_prompts = []
        for _ in range(args.prompt_batch_size):
            batch_prompts.append(prompts[prompt_cursor % len(prompts)])
            prompt_cursor += ddp_world_size

        rollout_records = []
        rewards_for_logging = []
        seq_lens_for_logging = []
        policy_model.eval()
        for prompt_index, prompt in enumerate(batch_prompts):
            prompt_conv = {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}]}
            prompt_tokens = tokenizer.render_for_completion(prompt_conv)
            sequence, mask = sample_rollout_sequence(engine, prompt_tokens, args, stable_rollout_seed(step, ddp_rank, prompt_index))
            generated_text = decode_generated_text(tokenizer, sequence, len(prompt_tokens))
            policy_sequence, policy_mask = policy_sequence_from_rollout(tokenizer, sequence, mask)
            inputs, targets = policy_tensors_from_sequence(policy_sequence, policy_mask, device)
            reward_sequence = reward_sequence_from_rollout(tokenizer, sequence)
            reward_ids = torch.tensor([reward_sequence], dtype=torch.long, device=device)
            last_positions = torch.tensor([len(reward_sequence) - 1], dtype=torch.long, device=device)
            with torch.no_grad():
                rm_score = reward_model(reward_ids, last_positions)
            rollout_records.append(
                {
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "inputs": inputs,
                    "targets": targets,
                    "reward": rm_score.squeeze(0),
                    "sequence_length": len(sequence),
                }
            )
            rewards_for_logging.append(rm_score.item())
            seq_lens_for_logging.append(len(sequence))

        inputs_all, targets_all = pad_rollout_batch(tokenizer, rollout_records, device)
        sequence_rewards = torch.stack([r["reward"] for r in rollout_records]).detach()
        old_token_logps_all, _, valid_all = collect_policy_outputs(
            policy_model,
            inputs_all,
            targets_all,
            args.device_batch_size,
            args.temperature,
            args.top_k,
        )
        ref_token_logps_all, _, _ = collect_policy_outputs(
            ref_model,
            inputs_all,
            targets_all,
            args.device_batch_size,
            args.temperature,
            args.top_k,
        )
        old_values_all = collect_values(policy_model, value_head, inputs_all, args.device_batch_size).detach()

        token_rewards_all, token_kl_all = build_token_rewards(
            old_token_logps_all,
            ref_token_logps_all,
            valid_all,
            sequence_rewards,
            args.kl_beta,
        )
        advantages_all, returns_all = compute_gae_returns(
            token_rewards_all,
            old_values_all,
            valid_all,
            args.gamma,
            args.gae_lambda,
        )
        if args.advantage_whiten:
            advantages_all = masked_global_whiten(advantages_all, valid_all, ddp_world_size)

        pg_loss_logging = []
        value_loss_logging = []
        entropy_logging = []
        ratio_logging = []
        clipfrac_logging = []
        approx_kl_logging = []
        policy_model.train()
        value_head.train()
        for epoch in range(args.ppo_epochs):
            order = torch.randperm(inputs_all.size(0), device=device)
            for mb0 in range(0, inputs_all.size(0), minibatch_size):
                mb_idx = order[mb0:mb0 + minibatch_size]
                inputs = inputs_all[mb_idx]
                targets = targets_all[mb_idx]
                old_token_logps = old_token_logps_all[mb_idx]
                old_values = old_values_all[mb_idx]
                advantages = advantages_all[mb_idx]
                returns = returns_all[mb_idx]

                optimizer.zero_grad(set_to_none=True)
                value_optimizer.zero_grad(set_to_none=True)

                token_logps, entropy, valid = compute_policy_outputs(policy_model, inputs, targets, args.temperature, args.top_k)
                values = compute_values(policy_model, value_head, inputs, bool(args.value_train_backbone))
                valid_f = valid.to(token_logps.dtype)
                local_valid_count = valid_f.sum().clamp(min=1.0)
                loss_normalizer = global_token_normalizer(valid_f, ddp_world_size)

                log_ratio = (token_logps - old_token_logps).clamp(min=-20.0, max=20.0)
                ratio = torch.exp(log_ratio)
                pg_unclipped = ratio * advantages
                pg_clipped = ratio.clamp(1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
                policy_loss = -(torch.minimum(pg_unclipped, pg_clipped) * valid_f).sum() / loss_normalizer

                values_clipped = old_values + (values - old_values).clamp(
                    min=-args.value_clip_epsilon,
                    max=args.value_clip_epsilon,
                )
                value_loss_unclipped = (values - returns).square()
                value_loss_clipped = (values_clipped - returns).square()
                value_loss = 0.5 * (torch.maximum(value_loss_unclipped, value_loss_clipped) * valid_f).sum() / loss_normalizer

                entropy_bonus = (entropy * valid_f).sum() / loss_normalizer
                entropy_metric = (entropy * valid_f).sum() / local_valid_count
                loss = policy_loss + args.value_loss_coef * value_loss - args.entropy_coef * entropy_bonus
                loss.backward()
                all_reduce_trainable_grads(value_head, ddp_world_size)
                optimizer.step()
                value_optimizer.step()

                with torch.no_grad():
                    approx_kl = (((ratio - 1.0) - log_ratio) * valid_f).sum() / local_valid_count
                    clipfrac = (((ratio - 1.0).abs() > args.clip_epsilon).to(token_logps.dtype) * valid_f).sum() / local_valid_count
                    pg_loss_logging.append(policy_loss.detach().item())
                    value_loss_logging.append(value_loss.detach().item())
                    entropy_logging.append(entropy_metric.detach().item())
                    ratio_logging.append(((ratio * valid_f).sum() / local_valid_count).item())
                    clipfrac_logging.append(clipfrac.item())
                    approx_kl_logging.append(approx_kl.item())

        valid_f_all = valid_all.to(token_rewards_all.dtype)
        sequence_return = (token_rewards_all * valid_f_all).sum(dim=-1)
        mean_reward = torch.tensor(sum(rewards_for_logging) / len(rewards_for_logging), device=device)
        mean_seq_len = torch.tensor(sum(seq_lens_for_logging) / len(seq_lens_for_logging), device=device)
        mean_return = sequence_return.mean().detach()
        mean_ref_kl = token_kl_all.sum(dim=-1).mean().detach()
        mean_pg_loss = torch.tensor(sum(pg_loss_logging) / len(pg_loss_logging) if pg_loss_logging else 0.0, device=device)
        mean_value_loss = torch.tensor(sum(value_loss_logging) / len(value_loss_logging) if value_loss_logging else 0.0, device=device)
        mean_entropy = torch.tensor(sum(entropy_logging) / len(entropy_logging) if entropy_logging else 0.0, device=device)
        mean_ratio = torch.tensor(sum(ratio_logging) / len(ratio_logging) if ratio_logging else 1.0, device=device)
        mean_clipfrac = torch.tensor(sum(clipfrac_logging) / len(clipfrac_logging) if clipfrac_logging else 0.0, device=device)
        mean_approx_kl = torch.tensor(sum(approx_kl_logging) / len(approx_kl_logging) if approx_kl_logging else 0.0, device=device)

        stats = [
            mean_reward,
            mean_seq_len,
            mean_return,
            mean_ref_kl,
            mean_pg_loss,
            mean_value_loss,
            mean_entropy,
            mean_ratio,
            mean_clipfrac,
            mean_approx_kl,
        ]
        if ddp:
            for tensor in stats:
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        (
            mean_reward,
            mean_seq_len,
            mean_return,
            mean_ref_kl,
            mean_pg_loss,
            mean_value_loss,
            mean_entropy,
            mean_ratio,
            mean_clipfrac,
            mean_approx_kl,
        ) = [tensor.item() for tensor in stats]

        print0(
            f"Step {step} | pg_loss: {mean_pg_loss:.6f} | value_loss: {mean_value_loss:.6f} | "
            f"reward: {mean_reward:.4f} | return: {mean_return:.4f} | ref_kl: {mean_ref_kl:.4f} | "
            f"ratio: {mean_ratio:.4f} | clipfrac: {mean_clipfrac:.4f}"
        )
        wandb_run.log(
            {
                "step": step,
                "policy_loss": mean_pg_loss,
                "value_loss": mean_value_loss,
                "entropy": mean_entropy,
                "reward_model_score": mean_reward,
                "return": mean_return,
                "ref_kl": mean_ref_kl,
                "ratio_mean": mean_ratio,
                "clipfrac": mean_clipfrac,
                "approx_kl_to_old": mean_approx_kl,
                "sequence_length": mean_seq_len,
                "lrm": lrm,
            }
        )

        if master_process and step > 0 and step % args.save_every == 0:
            save_ppo_state(args, policy_model, reward_model, value_head, step)

    if master_process:
        save_ppo_state(args, policy_model, reward_model, value_head, args.ppo_steps)

    get_report().log(section="Chat PPO Standard", data=[user_config])
    if hasattr(wandb_run, "finish"):
        wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
