"""
Critic-free GRPO-style post-training for nanochat.

This script keeps the PPO files unchanged and provides a separate path for
group-relative policy optimization:
- train/load a scalar reward model from preferences
- sample multiple completions per prompt from the current policy
- compute group-relative advantages from rewards within each prompt group
- optimize a token-level clipped policy objective without a value head

JSONL preference format:
{"prompt": "Question text", "chosen": "Preferred assistant response", "rejected": "Dispreferred assistant response"}

Examples:
python3 -m scripts.chat_GRPO --preference-source=gsm8k
python3 -m scripts.chat_GRPO --preference-source=jsonl --preference-path=/path/prefs.jsonl
python3 -m scripts.chat_GRPO --rm-load-path=/path/reward_000200.pt --rm-epochs=0
"""

import argparse
import os
import random

import torch
import torch.distributed as dist
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
    compute_token_logps,
    decode_generated_text,
    evaluate_reward_model,
    load_preferences,
    maybe_load_reward_model_checkpoint,
    pad_rollout_batch,
    policy_sequence_from_rollout,
    policy_tensors_from_sequence,
    reward_model_optimizer,
    reward_sequence_from_rollout,
    sample_rollout_sequence,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Critic-free GRPO-style post-training for nanochat")
    parser.add_argument("--run", type=str, default="dummy")
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--policy-source", type=str, default="sft", choices=["sft", "rl", "ppo", "ppo_standard", "grpo"])
    parser.add_argument("--policy-tag", type=str, default=None)
    parser.add_argument("--policy-step", type=int, default=None)
    parser.add_argument("--reference-source", type=str, default="sft", choices=["sft", "rl", "ppo", "ppo_standard", "grpo"])
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
    parser.add_argument("--grpo-steps", type=int, default=200)
    parser.add_argument("--grpo-epochs", type=int, default=4)
    parser.add_argument("--prompt-batch-size", type=int, default=4, help="number of prompts per rank")
    parser.add_argument("--group-size", type=int, default=4, help="number of completions sampled per prompt")
    parser.add_argument("--device-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0, help="0 disables top-k; GRPO ratios require full-support sampling")
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.0, help="weight for differentiable current-policy KL to the reference")
    parser.add_argument("--scale-rewards", type=int, default=1, help="1=z-score rewards within each prompt group, 0=center only")
    parser.add_argument("--embedding-lr", type=float, default=0.05)
    parser.add_argument("--unembedding-lr", type=float, default=0.002)
    parser.add_argument("--matrix-lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--init-lr-frac", type=float, default=0.2)
    parser.add_argument("--save-every", type=int, default=100)
    return parser.parse_args()


def stable_group_seed(step, ddp_rank, prompt_index, sample_index):
    seed = (step + 1) * 1_000_003 + (ddp_rank + 1) * 9_176 + prompt_index * 997 + sample_index
    return seed & 0x7FFFFFFF


def get_grpo_lr_multiplier(step, total_steps):
    if total_steps <= 0:
        return 1.0
    return 1.0 - (step / total_steps)


def global_token_normalizer(valid_f, ddp_world_size):
    count = valid_f.sum()
    if ddp_world_size > 1:
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        count = count / ddp_world_size
    return count.clamp(min=1.0)


def compute_group_advantages(sequence_rewards, num_prompts, group_size, scale_rewards):
    grouped = sequence_rewards.view(num_prompts, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    group_std = centered.square().mean(dim=1, keepdim=True).clamp(min=1e-8).sqrt()
    if scale_rewards:
        centered = centered / group_std
    return centered.reshape(-1), group_std.squeeze(-1)


def sampled_current_ref_kl(token_logps, ref_token_logps, valid_f):
    log_ref_over_policy = (ref_token_logps - token_logps).clamp(min=-20.0, max=20.0)
    kl_tokens = torch.exp(log_ref_over_policy) - 1.0 - log_ref_over_policy
    return kl_tokens * valid_f


@torch.no_grad()
def score_rollout_reward(reward_model, tokenizer, sequence, device):
    reward_sequence = reward_sequence_from_rollout(tokenizer, sequence)
    reward_ids = torch.tensor([reward_sequence], dtype=torch.long, device=device)
    last_positions = torch.tensor([len(reward_sequence) - 1], dtype=torch.long, device=device)
    return reward_model(reward_ids, last_positions).squeeze(0)


def save_grpo_state(args, policy_model, reward_model, step):
    base_dir = get_base_dir()
    depth = policy_model.config.n_layer
    model_tag = args.policy_tag if args.policy_tag else f"d{depth}"
    policy_dir = os.path.join(base_dir, "chatgrpo_checkpoints", model_tag)
    reward_dir = os.path.join(base_dir, "chatgrpo_reward_checkpoints", model_tag)
    save_checkpoint(policy_dir, step, policy_model.state_dict(), None, {"model_config": policy_model.config.__dict__})
    os.makedirs(reward_dir, exist_ok=True)
    torch.save({"reward_model": reward_model.state_dict()}, os.path.join(reward_dir, f"reward_{step:06d}.pt"))
    print0(f"Saved GRPO checkpoints to {policy_dir} and {reward_dir}")


def resolve_reference_checkpoint(args):
    same_source = args.reference_source == args.policy_source
    ref_tag = args.reference_tag if args.reference_tag is not None else (args.policy_tag if same_source else None)
    ref_step = args.reference_step if args.reference_step is not None else (args.policy_step if same_source else None)
    return ref_tag, ref_step


def main():
    args = parse_args()
    user_config = vars(args).copy()
    assert args.prompt_batch_size > 0, "prompt_batch_size must be positive"
    assert args.group_size >= 2, "group_size must be at least 2 for group-relative advantages"
    assert args.device_batch_size > 0, "device_batch_size must be positive"
    assert args.rm_batch_size > 0, "rm_batch_size must be positive"
    assert args.rm_epochs > 0 or args.rm_load_path is not None, "Use --rm-epochs>0 to train a reward model or provide --rm-load-path"
    assert args.grpo_steps > 0, "grpo_steps must be positive"
    assert args.grpo_epochs > 0, "grpo_epochs must be positive"
    assert args.max_new_tokens > 0, "max_new_tokens must be positive"
    assert args.temperature > 0.0, "GRPO training requires stochastic sampling; use --temperature > 0"
    assert args.top_k == 0, "GRPO ratios in this script require full-support sampling; use --top-k=0"
    assert args.clip_epsilon >= 0.0, "clip_epsilon must be non-negative"
    assert args.kl_beta >= 0.0, "kl_beta must be non-negative"
    assert args.scale_rewards in (0, 1), "scale_rewards must be 0 or 1"
    assert args.init_lr_frac >= 0.0, "init_lr_frac must be non-negative"
    assert args.save_every > 0, "save_every must be positive"
    use_reference_model = args.kl_beta > 0.0

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0

    use_dummy_wandb = args.run == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-grpo", name=args.run, config=user_config)

    policy_model, tokenizer, meta = load_model(args.policy_source, device, phase="train", model_tag=args.policy_tag, step=args.policy_step)
    ref_model = None
    if use_reference_model:
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

    prefs = load_preferences(args)
    random.Random(42).shuffle(prefs)
    split = min(args.max_train_examples, len(prefs))
    train_prefs = prefs[:split]
    val_prefs = prefs[split:split + args.max_val_examples]
    assert train_prefs, "No training preferences available; increase --max-train-examples or provide a non-empty preference dataset"
    prompts = [row["prompt"] for row in train_prefs]
    print0(f"GRPO prefs train: {len(train_prefs)} | val: {len(val_prefs)}")

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
    for step in range(args.grpo_steps):
        lrm = get_grpo_lr_multiplier(step, args.grpo_steps)
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
            for sample_index in range(args.group_size):
                sequence, mask = sample_rollout_sequence(
                    engine,
                    prompt_tokens,
                    args,
                    stable_group_seed(step, ddp_rank, prompt_index, sample_index),
                )
                generated_text = decode_generated_text(tokenizer, sequence, len(prompt_tokens))
                policy_sequence, policy_mask = policy_sequence_from_rollout(tokenizer, sequence, mask)
                inputs, targets = policy_tensors_from_sequence(policy_sequence, policy_mask, device)
                with torch.no_grad():
                    rm_score = score_rollout_reward(reward_model, tokenizer, sequence, device)
                rollout_records.append(
                    {
                        "prompt_index": prompt_index,
                        "sample_index": sample_index,
                        "generated_text": generated_text,
                        "inputs": inputs,
                        "targets": targets,
                        "reward": rm_score,
                        "sequence_length": len(sequence),
                    }
                )
                rewards_for_logging.append(rm_score.item())
                seq_lens_for_logging.append(len(sequence))

        inputs_all, targets_all = pad_rollout_batch(tokenizer, rollout_records, device)
        sequence_rewards = torch.stack([record["reward"] for record in rollout_records]).detach()
        advantages_all, group_reward_std = compute_group_advantages(
            sequence_rewards,
            len(batch_prompts),
            args.group_size,
            bool(args.scale_rewards),
        )
        old_token_logps_all = compute_batched_logps(policy_model, inputs_all, targets_all, args.device_batch_size, args.temperature)
        ref_token_logps_all = None
        if use_reference_model:
            ref_token_logps_all = compute_batched_logps(ref_model, inputs_all, targets_all, args.device_batch_size, args.temperature)
        valid_all = targets_all >= 0
        valid_f_all = valid_all.to(old_token_logps_all.dtype)
        loss_normalizer = global_token_normalizer(valid_f_all, ddp_world_size)

        pg_loss_logging = []
        kl_loss_logging = []
        ratio_sum_logging = 0.0
        ratio_count_logging = 0.0
        clipfrac_sum_logging = 0.0
        approx_kl_sum_logging = 0.0
        ref_kl_sum_logging = 0.0
        policy_model.train()
        for epoch in range(args.grpo_epochs):
            optimizer.zero_grad(set_to_none=True)
            epoch_pg_loss = 0.0
            epoch_kl_loss = 0.0
            for b0 in range(0, inputs_all.size(0), args.device_batch_size):
                b1 = min(b0 + args.device_batch_size, inputs_all.size(0))
                inputs = inputs_all[b0:b1]
                targets = targets_all[b0:b1]
                old_token_logps = old_token_logps_all[b0:b1]
                advantages = advantages_all[b0:b1].unsqueeze(-1)

                token_logps, valid = compute_token_logps(policy_model, inputs, targets, args.temperature)
                valid_f = valid.to(token_logps.dtype)
                log_ratio = (token_logps - old_token_logps).clamp(min=-20.0, max=20.0)
                ratio = torch.exp(log_ratio)
                pg_unclipped = ratio * advantages
                pg_clipped = ratio.clamp(1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
                policy_loss = -(torch.minimum(pg_unclipped, pg_clipped) * valid_f).sum() / loss_normalizer
                kl_loss = token_logps.new_zeros(())
                loss = policy_loss
                if use_reference_model:
                    ref_token_logps = ref_token_logps_all[b0:b1]
                    ref_kl_tokens = sampled_current_ref_kl(token_logps, ref_token_logps, valid_f)
                    kl_loss = ref_kl_tokens.sum() / loss_normalizer
                    loss = policy_loss + args.kl_beta * kl_loss
                loss.backward()

                epoch_pg_loss += policy_loss.detach().item()
                epoch_kl_loss += kl_loss.detach().item()
                ratio_sum_logging += (ratio * valid_f).sum().item()
                ratio_count_logging += valid_f.sum().item()
                clipfrac_sum_logging += (((ratio - 1.0).abs() > args.clip_epsilon).to(token_logps.dtype) * valid_f).sum().item()
                approx_kl_sum_logging += (((ratio - 1.0) - log_ratio) * valid_f).sum().item()
                if use_reference_model:
                    ref_kl_sum_logging += ref_kl_tokens.sum().item()
            optimizer.step()
            pg_loss_logging.append(epoch_pg_loss)
            kl_loss_logging.append(epoch_kl_loss)

        mean_reward = torch.tensor(sum(rewards_for_logging) / len(rewards_for_logging), device=device)
        mean_seq_len = torch.tensor(sum(seq_lens_for_logging) / len(seq_lens_for_logging), device=device)
        mean_pg_loss = torch.tensor(sum(pg_loss_logging) / len(pg_loss_logging) if pg_loss_logging else 0.0, device=device)
        mean_kl_loss = torch.tensor(sum(kl_loss_logging) / len(kl_loss_logging) if kl_loss_logging else 0.0, device=device)
        mean_group_reward_std = group_reward_std.mean().detach()
        ratio_sum = torch.tensor(ratio_sum_logging, device=device)
        ratio_count = torch.tensor(ratio_count_logging, device=device)
        clipfrac_sum = torch.tensor(clipfrac_sum_logging, device=device)
        approx_kl_sum = torch.tensor(approx_kl_sum_logging, device=device)
        ref_kl_sum = torch.tensor(ref_kl_sum_logging, device=device)
        if ddp:
            for tensor in [ratio_sum, ratio_count, clipfrac_sum, approx_kl_sum, ref_kl_sum]:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        ratio_count = ratio_count.clamp(min=1.0)
        mean_ratio = torch.tensor((ratio_sum / ratio_count).item(), device=device)
        mean_clipfrac = torch.tensor((clipfrac_sum / ratio_count).item(), device=device)
        mean_approx_kl = torch.tensor((approx_kl_sum / ratio_count).item(), device=device)
        mean_ref_kl = torch.tensor((ref_kl_sum / ratio_count).item(), device=device)

        stats = [
            mean_reward,
            mean_seq_len,
            mean_pg_loss,
            mean_kl_loss,
            mean_group_reward_std,
            mean_ratio,
            mean_clipfrac,
            mean_approx_kl,
            mean_ref_kl,
        ]
        if ddp:
            for tensor in stats:
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        (
            mean_reward,
            mean_seq_len,
            mean_pg_loss,
            mean_kl_loss,
            mean_group_reward_std,
            mean_ratio,
            mean_clipfrac,
            mean_approx_kl,
            mean_ref_kl,
        ) = [tensor.item() for tensor in stats]

        print0(
            f"Step {step} | pg_loss: {mean_pg_loss:.6f} | kl_loss: {mean_kl_loss:.6f} | "
            f"reward: {mean_reward:.4f} | reward_group_std: {mean_group_reward_std:.4f} | "
            f"ref_kl: {mean_ref_kl:.4f} | ratio: {mean_ratio:.4f} | clipfrac: {mean_clipfrac:.4f}"
        )
        wandb_run.log(
            {
                "step": step,
                "policy_loss": mean_pg_loss,
                "kl_loss": mean_kl_loss,
                "reward_model_score": mean_reward,
                "reward_group_std": mean_group_reward_std,
                "ref_kl": mean_ref_kl,
                "ratio_mean": mean_ratio,
                "clipfrac": mean_clipfrac,
                "approx_kl_to_old": mean_approx_kl,
                "sequence_length": mean_seq_len,
                "lrm": lrm,
            }
        )

        if master_process and step > 0 and step % args.save_every == 0:
            save_grpo_state(args, policy_model, reward_model, step)

    if master_process:
        save_grpo_state(args, policy_model, reward_model, args.grpo_steps)

    get_report().log(section="Chat GRPO", data=[user_config])
    if hasattr(wandb_run, "finish"):
        wandb_run.finish()
    compute_cleanup()


@torch.no_grad()
def compute_batched_logps(model, inputs_all, targets_all, batch_size, temperature):
    chunks = []
    for b0 in range(0, inputs_all.size(0), batch_size):
        b1 = min(b0 + batch_size, inputs_all.size(0))
        token_logps, _ = compute_token_logps(model, inputs_all[b0:b1], targets_all[b0:b1], temperature)
        chunks.append(token_logps)
    return torch.cat(chunks, dim=0)


if __name__ == "__main__":
    main()
