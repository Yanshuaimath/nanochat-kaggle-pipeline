"""
PPO with reward-model training for nanochat.

This script is standalone and keeps existing repo files unchanged.
It supports:
- reward-model training from preference JSONL or synthetic GSM8K preferences
- optional reward-model checkpoint loading for PPO warm starts
- PPO policy optimization on prompt rollouts scored by the reward model

JSONL preference format:
{"prompt": "Question text", "chosen": "Preferred assistant response", "rejected": "Dispreferred assistant response"}

Examples:
python3 -m scripts.chat_ppo --preference-source=gsm8k
python3 -m scripts.chat_ppo --preference-source=jsonl --preference-path=/path/prefs.jsonl
python3 -m scripts.chat_ppo --preference-source=jsonl --preference-path=/path/prefs.jsonl --rm-load-path=/path/reward_000200.pt
"""

import argparse
import json
import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb

from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.common import COMPUTE_DTYPE, DummyWandb, autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0
from nanochat.engine import Engine
from nanochat.gpt import norm
from nanochat.report import get_report
from tasks.gsm8k import GSM8K, extract_answer


def parse_args():
    parser = argparse.ArgumentParser(description="PPO with reward model training for nanochat")
    parser.add_argument("--run", type=str, default="dummy")
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--policy-source", type=str, default="sft", choices=["sft", "rl"])
    parser.add_argument("--policy-tag", type=str, default=None)
    parser.add_argument("--policy-step", type=int, default=None)
    parser.add_argument("--reference-source", type=str, default="sft", choices=["sft", "rl"])
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
    parser.add_argument("--rm-load-path", type=str, default=None, help="optional path to a saved reward checkpoint (.pt) with key 'reward_model'")
    parser.add_argument("--ppo-steps", type=int, default=200)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    parser.add_argument("--device-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument("--embedding-lr", type=float, default=0.05)
    parser.add_argument("--unembedding-lr", type=float, default=0.002)
    parser.add_argument("--matrix-lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--init-lr-frac", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)
    return parser.parse_args()


def flatten_assistant_content(content):
    if isinstance(content, str):
        return content
    return "".join(part["text"] for part in content)


def synthesize_wrong_answer(answer_text):
    gold = extract_answer(answer_text)
    if gold is None:
        return "I am not sure.\n#### 0"
    try:
        if "." in gold:
            wrong = str(float(gold) + 1.0)
        else:
            wrong = str(int(gold) + 1)
    except ValueError:
        wrong = "0"
    return f"I think the answer is {wrong}.\n#### {wrong}"


def load_preferences(args):
    if args.preference_source == "jsonl":
        assert args.preference_path is not None, "--preference-path is required for JSONL preferences"
        prefs = []
        with open(args.preference_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prefs.append({"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]})
        return prefs

    task = GSM8K(subset="main", split="train")
    prefs = []
    max_examples = min(args.max_train_examples + args.max_val_examples, len(task))
    for i in range(max_examples):
        conversation = task[i]
        prompt = conversation["messages"][0]["content"]
        chosen = flatten_assistant_content(conversation["messages"][1]["content"])
        rejected = synthesize_wrong_answer(chosen)
        prefs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return prefs


def build_conversation(prompt, assistant_text):
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": assistant_text}]}


def collate_conversations(tokenizer, conversations, device):
    bos = tokenizer.get_bos_token_id()
    rows = []
    masks = []
    lengths = []
    for conversation in conversations:
        ids, mask = tokenizer.render_conversation(conversation)
        rows.append(ids)
        masks.append(mask)
        lengths.append(len(ids))
    max_len = max(lengths)
    padded_ids = [row + [bos] * (max_len - len(row)) for row in rows]
    padded_masks = [mask + [0] * (max_len - len(mask)) for mask in masks]
    ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
    mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
    inputs = ids[:, :-1]
    targets = ids[:, 1:].clone()
    targets[mask_ids[:, 1:] == 0] = -1
    last_positions = torch.tensor([length - 1 for length in lengths], dtype=torch.long, device=device)
    return ids, inputs, targets, last_positions


def pad_rollout_batch(tokenizer, rollout_records, device):
    bos = tokenizer.get_bos_token_id()
    max_len = max(record["inputs"].size(1) for record in rollout_records)
    padded_inputs = []
    padded_targets = []
    for record in rollout_records:
        inputs = record["inputs"]
        targets = record["targets"]
        pad = max_len - inputs.size(1)
        if pad > 0:
            inputs = F.pad(inputs, (0, pad), value=bos)
            targets = F.pad(targets, (0, pad), value=-1)
        padded_inputs.append(inputs)
        padded_targets.append(targets)
    return torch.cat(padded_inputs, dim=0).to(device), torch.cat(padded_targets, dim=0).to(device)


def compute_token_logps(model, inputs, targets):
    valid = targets >= 0
    safe_targets = targets.clamp(min=0)
    logits = model(inputs)
    logprobs = F.log_softmax(logits, dim=-1)
    token_logps = logprobs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps.masked_fill(~valid, 0.0)
    return token_logps, valid


@torch.no_grad()
def collect_token_logps(model, inputs_all, targets_all, batch_size):
    chunks = []
    for b0 in range(0, inputs_all.size(0), batch_size):
        b1 = min(b0 + batch_size, inputs_all.size(0))
        token_logps, _ = compute_token_logps(model, inputs_all[b0:b1], targets_all[b0:b1])
        chunks.append(token_logps)
    return torch.cat(chunks, dim=0)


def forward_hidden(model, idx):
    B, T = idx.size()
    assert T <= model.cos.size(1)
    cos_sin = model.cos[:, :T], model.sin[:, :T]
    x = model.transformer.wte(idx)
    x = x.to(COMPUTE_DTYPE)
    x = norm(x)
    gate = model.smear_lambda.to(x.dtype) * torch.sigmoid(model.smear_gate(x[:, 1:, :24]))
    x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
    x0 = x
    n_layer = model.config.n_layer
    backout_layer = n_layer // 2
    x_backout = None
    for i, block in enumerate(model.transformer.h):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        ve = model.value_embeds[str(i)](idx).to(x.dtype) if str(i) in model.value_embeds else None
        x = block(x, ve, cos_sin, model.window_sizes[i], kv_cache=None)
        if i == backout_layer:
            x_backout = x
    if x_backout is not None:
        x = x - model.backout_lambda.to(x.dtype) * x_backout
    x = norm(x)
    return x.float()


class RewardModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.reward_head = nn.Linear(backbone.config.n_embd, 1, bias=False)

    def forward(self, ids, last_positions):
        hidden = forward_hidden(self.backbone, ids)
        pooled = hidden[torch.arange(hidden.size(0), device=ids.device), last_positions]
        return self.reward_head(pooled).squeeze(-1)


def reward_model_optimizer(reward_model, lr, train_backbone):
    if train_backbone:
        return torch.optim.AdamW(reward_model.parameters(), lr=lr)
    for param in reward_model.backbone.parameters():
        param.requires_grad_(False)
    return torch.optim.AdamW(reward_model.reward_head.parameters(), lr=lr)


def all_reduce_reward_model_grads(reward_model, ddp_world_size):
    if ddp_world_size <= 1:
        return
    for param in reward_model.parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(ddp_world_size)


def maybe_load_reward_model_checkpoint(reward_model, load_path, device):
    if load_path is None:
        return
    checkpoint = torch.load(load_path, map_location=device)
    state_dict = checkpoint["reward_model"] if "reward_model" in checkpoint else checkpoint
    missing_keys, unexpected_keys = reward_model.load_state_dict(state_dict, strict=False)
    assert not missing_keys, f"Missing reward-model keys in checkpoint {load_path}: {missing_keys}"
    assert not unexpected_keys, f"Unexpected reward-model keys in checkpoint {load_path}: {unexpected_keys}"
    print0(f"Loaded reward model checkpoint from {load_path}")


def sampled_kl(token_logps, ref_token_logps, valid):
    valid_f = valid.to(token_logps.dtype)
    lengths = valid.sum(dim=-1).clamp(min=1)
    return ((token_logps - ref_token_logps) * valid_f).sum(dim=-1) / lengths


def compute_advantages(rewards):
    if rewards.numel() == 1:
        return torch.zeros_like(rewards)
    return rewards - rewards.mean()


def get_ppo_lr_multiplier(step, total_steps):
    if total_steps <= 0:
        return 1.0
    return 1.0 - (step / total_steps)


@torch.no_grad()
def evaluate_reward_model(reward_model, tokenizer, dataset, batch_size, device, ddp_rank, ddp_world_size):
    num_correct = torch.zeros(1, dtype=torch.float32, device=device)
    num_examples = torch.zeros(1, dtype=torch.float32, device=device)
    for start in range(0, len(dataset), ddp_world_size * batch_size):
        local_start = start + ddp_rank * batch_size
        batch = dataset[local_start:local_start + batch_size]
        if not batch:
            continue
        chosen = [build_conversation(row["prompt"], row["chosen"]) for row in batch]
        rejected = [build_conversation(row["prompt"], row["rejected"]) for row in batch]
        chosen_ids, _, _, chosen_last = collate_conversations(tokenizer, chosen, device)
        rejected_ids, _, _, rejected_last = collate_conversations(tokenizer, rejected, device)
        chosen_scores = reward_model(chosen_ids, chosen_last)
        rejected_scores = reward_model(rejected_ids, rejected_last)
        num_correct += (chosen_scores > rejected_scores).float().sum()
        num_examples += len(batch)
    if ddp_world_size > 1:
        dist.all_reduce(num_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
    if num_examples.item() == 0:
        return 0.0
    return (num_correct / num_examples).item()


def main():
    args = parse_args()
    user_config = vars(args).copy()
    assert args.prompt_batch_size > 0, "prompt_batch_size must be positive"
    assert args.device_batch_size > 0, "device_batch_size must be positive"
    assert args.ppo_steps > 0, "ppo_steps must be positive"

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0

    use_dummy_wandb = args.run == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-ppo", name=args.run, config=user_config)

    policy_model, tokenizer, meta = load_model(args.policy_source, device, phase="train", model_tag=args.policy_tag, step=args.policy_step)
    ref_tag = args.reference_tag if args.reference_tag is not None else args.policy_tag
    ref_step = args.reference_step if args.reference_step is not None else args.policy_step
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
    print0(f"PPO prefs train: {len(train_prefs)} | val: {len(val_prefs)}")

    # Stage 1: reward model training
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

    # Stage 2: PPO on prompts scored by reward model
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
        for prompt in batch_prompts:
            prompt_conv = {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}]}
            prompt_tokens = tokenizer.render_for_completion(prompt_conv)
            result_tokens, masks = engine.generate_batch(
                prompt_tokens,
                num_samples=1,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=hash((step, prompt)) & 0x7FFFFFFF,
            )
            sequence = result_tokens[0]
            generated_text = tokenizer.decode(sequence[len(prompt_tokens):])
            sampled_conv = build_conversation(prompt, generated_text)
            ids, inputs, targets, last_positions = collate_conversations(tokenizer, [sampled_conv], device)
            with torch.no_grad():
                rm_score = reward_model(ids, last_positions)
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
        rewards_all = torch.stack([r["reward"] for r in rollout_records])
        old_token_logps_all = collect_token_logps(policy_model, inputs_all, targets_all, args.device_batch_size)
        ref_token_logps_all = collect_token_logps(ref_model, inputs_all, targets_all, args.device_batch_size)
        old_valid = targets_all >= 0
        kl_penalty = sampled_kl(old_token_logps_all, ref_token_logps_all, old_valid)
        returns = rewards_all - args.kl_beta * kl_penalty
        advantages_all = compute_advantages(returns)

        ratio_logging = []
        loss_logging = []
        policy_model.train()
        for epoch in range(args.ppo_epochs):
            optimizer.zero_grad(set_to_none=True)
            for b0 in range(0, inputs_all.size(0), args.device_batch_size):
                b1 = min(b0 + args.device_batch_size, inputs_all.size(0))
                inputs = inputs_all[b0:b1]
                targets = targets_all[b0:b1]
                old_token_logps = old_token_logps_all[b0:b1]
                advantages = advantages_all[b0:b1]

                token_logps, valid = compute_token_logps(policy_model, inputs, targets)
                valid_f = valid.to(token_logps.dtype)
                ratio = torch.exp((token_logps - old_token_logps).clamp(min=-20.0, max=20.0))
                unclipped = ratio * advantages.unsqueeze(-1)
                clipped = ratio.clamp(1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages.unsqueeze(-1)
                surrogate = torch.minimum(unclipped, clipped)
                loss = -(surrogate * valid_f).sum() / valid.sum().clamp(min=1)
                loss.backward()
                loss_logging.append(loss.detach().item())
                ratio_logging.append((ratio * valid_f).sum().item() / valid.sum().clamp(min=1).item())
            optimizer.step()

        mean_reward = sum(rewards_for_logging) / len(rewards_for_logging)
        mean_seq_len = sum(seq_lens_for_logging) / len(seq_lens_for_logging)
        mean_loss = sum(loss_logging) / len(loss_logging) if loss_logging else 0.0
        mean_ratio = sum(ratio_logging) / len(ratio_logging) if ratio_logging else 1.0
        mean_return = returns.mean().detach()
        mean_kl = kl_penalty.mean().detach()

        if ddp:
            stats = [torch.tensor(mean_reward, device=device), torch.tensor(mean_seq_len, device=device), torch.tensor(mean_loss, device=device), torch.tensor(mean_ratio, device=device), mean_return, mean_kl]
            for tensor in stats:
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            mean_reward, mean_seq_len, mean_loss, mean_ratio, mean_return, mean_kl = [t.item() for t in stats]
        else:
            mean_return = mean_return.item()
            mean_kl = mean_kl.item()

        print0(
            f"Step {step} | ppo_loss: {mean_loss:.6f} | reward: {mean_reward:.4f} | "
            f"return: {mean_return:.4f} | sampled_kl: {mean_kl:.4f} | ratio: {mean_ratio:.4f}"
        )
        wandb_run.log(
            {
                "step": step,
                "ppo_loss": mean_loss,
                "reward_model_score": mean_reward,
                "return": mean_return,
                "sampled_kl": mean_kl,
                "ratio_mean": mean_ratio,
                "sequence_length": mean_seq_len,
                "lrm": lrm,
            }
        )

        if master_process and step > 0 and step % args.save_every == 0:
            base_dir = get_base_dir()
            depth = policy_model.config.n_layer
            model_tag = args.policy_tag if args.policy_tag else f"d{depth}"
            policy_dir = os.path.join(base_dir, "chatppo_checkpoints", model_tag)
            reward_dir = os.path.join(base_dir, "chatppo_reward_checkpoints", model_tag)
            save_checkpoint(policy_dir, step, policy_model.state_dict(), None, {"model_config": policy_model.config.__dict__})
            os.makedirs(reward_dir, exist_ok=True)
            torch.save({"reward_model": reward_model.state_dict()}, os.path.join(reward_dir, f"reward_{step:06d}.pt"))
            print0(f"Saved PPO checkpoints to {policy_dir} and {reward_dir}")

    if master_process:
        base_dir = get_base_dir()
        depth = policy_model.config.n_layer
        model_tag = args.policy_tag if args.policy_tag else f"d{depth}"
        policy_dir = os.path.join(base_dir, "chatppo_checkpoints", model_tag)
        reward_dir = os.path.join(base_dir, "chatppo_reward_checkpoints", model_tag)
        save_checkpoint(policy_dir, args.ppo_steps, policy_model.state_dict(), None, {"model_config": policy_model.config.__dict__})
        os.makedirs(reward_dir, exist_ok=True)
        torch.save({"reward_model": reward_model.state_dict()}, os.path.join(reward_dir, f"reward_{args.ppo_steps:06d}.pt"))
        print0(f"Saved final PPO checkpoints to {policy_dir} and {reward_dir}")

    get_report().log(section="Chat PPO", data=[user_config])
    if hasattr(wandb_run, "finish"):
        wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
