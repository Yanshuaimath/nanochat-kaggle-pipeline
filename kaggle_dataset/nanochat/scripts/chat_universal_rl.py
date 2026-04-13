"""
Universal chat RL trainer for GSM8K.

This is a standalone sibling of scripts/chat_rl.py that keeps the existing
rollout/evaluation flow, but makes the optimization objective selectable.

Currently supported objectives:
- reinforce : current nanochat-style grouped REINFORCE baseline
- rloo_kl   : leave-one-out REINFORCE with optional sampled KL-to-reference
- gspo      : sequence-level clipped policy optimization with RLOO advantages

Examples:

python -m scripts.chat_universal_rl --objective=reinforce
python -m scripts.chat_universal_rl --objective=rloo_kl --kl-beta=0.02
python -m scripts.chat_universal_rl --objective=gspo --update-epochs=4 --clip-epsilon=0.2
"""

import argparse
import itertools
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb

from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    print0,
)
from nanochat.engine import Engine
from nanochat.report import get_report
from tasks.gsm8k import GSM8K


def parse_args():
    parser = argparse.ArgumentParser(description="Universal reinforcement learning on GSM8K")
    # Logging
    parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
    # Runtime
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    # Model loading
    parser.add_argument("--model-source", type=str, default="sft", choices=["sft", "rl"], help="checkpoint family to train from")
    parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
    parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
    parser.add_argument("--reference-source", type=str, default="sft", choices=["sft", "rl"], help="checkpoint family used for frozen KL reference")
    parser.add_argument("--reference-tag", type=str, default=None, help="reference model tag (defaults to model tag)")
    parser.add_argument("--reference-step", type=int, default=None, help="reference model step (defaults to model step)")
    # Objective
    parser.add_argument("--objective", type=str, default="reinforce", choices=["reinforce", "rloo_kl", "gspo"], help="policy optimization objective")
    parser.add_argument("--kl-beta", type=float, default=0.0, help="sampled KL penalty coefficient")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="clipping epsilon for GSPO")
    parser.add_argument("--update-epochs", type=int, default=1, help="number of optimization epochs per sampled rollout batch")
    parser.add_argument("--log-ratio-cap", type=float, default=20.0, help="clamp sequence log-ratio before exp for numerical stability")
    # Training horizon
    parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over GSM8K")
    parser.add_argument("--max-steps", type=int, default=None, help="optional cap on total optimizer steps")#added by Y.Q
    # Batch sizes / sampling
    parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
    parser.add_argument("--examples-per-step", type=int, default=16, help="total examples per optimization step across all ranks")
    parser.add_argument("--num-samples", type=int, default=16, help="number of samples per example/question")
    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens to generate per sample")
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="top-k sampling (0 = disabled)")
    # Optimization
    parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
    parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
    parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
    parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
    # Evaluation / checkpointing
    parser.add_argument("--eval-every", type=int, default=60, help="evaluate pass@k every N steps")
    parser.add_argument("--eval-examples", type=int, default=400, help="number of examples for pass@k evaluation")
    parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
    return parser.parse_args()


def freeze_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def build_advantages(rewards, objective):
    if rewards.numel() == 1:
        return torch.zeros_like(rewards)
    if objective in {"rloo_kl", "gspo"}:
        # Leave-one-out baseline: b_i = (sum_j r_j - r_i) / (K - 1)
        total = rewards.sum()
        baseline = (total - rewards) / (rewards.numel() - 1)
        return rewards - baseline
    # Group mean baseline, matching the existing chat_rl.py behavior.
    return rewards - rewards.mean()


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
    token_chunks = []
    for b0 in range(0, inputs_all.size(0), batch_size):
        b1 = min(b0 + batch_size, inputs_all.size(0))
        token_logps, _ = compute_token_logps(model, inputs_all[b0:b1], targets_all[b0:b1])
        token_chunks.append(token_logps)
    return torch.cat(token_chunks, dim=0)


def sampled_kl_per_sequence(token_logps, ref_token_logps, valid):
    lengths = valid.sum(dim=-1).clamp(min=1)
    return ((token_logps - ref_token_logps) * valid).sum(dim=-1) / lengths


def compute_objective_loss(
    objective,
    token_logps,
    valid,
    advantages,
    old_token_logps=None,
    ref_token_logps=None,
    kl_beta=0.0,
    clip_epsilon=0.2,
    log_ratio_cap=20.0,
):
    metrics = {}
    valid_f = valid.to(token_logps.dtype)
    lengths = valid.sum(dim=-1).clamp(min=1)

    if objective in {"reinforce", "rloo_kl"}:
        per_token_adv = advantages.unsqueeze(-1)
        pg_obj = (token_logps * per_token_adv * valid_f).sum()
        pg_obj = pg_obj / lengths.sum().clamp(min=1)
        metrics["pg_obj"] = pg_obj.detach()

        if ref_token_logps is not None and kl_beta > 0.0:
            kl_seq = sampled_kl_per_sequence(token_logps, ref_token_logps, valid_f)
            kl_term = kl_seq.mean()
            metrics["sampled_kl"] = kl_term.detach()
            loss = -pg_obj + kl_beta * kl_term
        else:
            metrics["sampled_kl"] = torch.zeros((), device=token_logps.device)
            loss = -pg_obj

        return loss, metrics

    if objective == "gspo":
        assert old_token_logps is not None, "GSPO requires frozen rollout logprobs"
        seq_logp = (token_logps * valid_f).sum(dim=-1)
        seq_logp_old = (old_token_logps * valid_f).sum(dim=-1)
        log_ratio = torch.clamp(seq_logp - seq_logp_old, min=-log_ratio_cap, max=log_ratio_cap)
        ratio = torch.exp(log_ratio)
        clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
        unclipped = ratio * advantages
        clipped = clipped_ratio * advantages
        seq_obj = torch.minimum(unclipped, clipped)
        loss = -seq_obj.mean()

        metrics["ratio_mean"] = ratio.mean().detach()
        metrics["ratio_min"] = ratio.min().detach()
        metrics["ratio_max"] = ratio.max().detach()
        metrics["pg_obj"] = seq_obj.mean().detach()

        if ref_token_logps is not None and kl_beta > 0.0:
            kl_seq = sampled_kl_per_sequence(token_logps, ref_token_logps, valid_f)
            kl_term = kl_seq.mean()
            metrics["sampled_kl"] = kl_term.detach()
            loss = loss + kl_beta * kl_term
        else:
            metrics["sampled_kl"] = torch.zeros((), device=token_logps.device)

        return loss, metrics

    raise ValueError(f"Unknown objective: {objective}")


def maybe_reduce_mean_scalar(value, ddp, device):
    tensor = torch.tensor(float(value), dtype=torch.float, device=device)
    if ddp:
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor.item()


def main():
    args = parse_args()
    user_config = vars(args).copy()

    # YQ workaround: force float32 compute in this script to avoid CUDA generation dtype mismatch
    #os.environ["NANOCHAT_DTYPE"] = "float32"

    assert args.num_samples % args.device_batch_size == 0, "num_samples must be divisible by device_batch_size"
    if args.objective == "gspo":
        assert args.update_epochs >= 1, "GSPO requires at least one update epoch"

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0

    use_dummy_wandb = args.run == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rl", name=args.run, config=user_config)

    model, tokenizer, meta = load_model(args.model_source, device, phase="eval", model_tag=args.model_tag, step=args.model_step)
    engine = Engine(model, tokenizer)

    need_reference = args.kl_beta > 0.0
    ref_model = None
    if need_reference:
        ref_tag = args.reference_tag if args.reference_tag is not None else args.model_tag
        ref_step = args.reference_step if args.reference_step is not None else args.model_step
        ref_model, _, _ = load_model(args.reference_source, device, phase="eval", model_tag=ref_tag, step=ref_step)
        freeze_model(ref_model)

    train_task = GSM8K(subset="main", split="train")
    val_task = GSM8K(subset="main", split="test")
    num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
    if args.max_steps is not None:#227-229 added by YQ
        assert args.max_steps > 0, "--max-steps must be positive"
        num_steps = min(num_steps, args.max_steps)
    print0(f"Calculated number of steps: {num_steps}")

    @torch.no_grad()
    def get_batch(step):
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        rank_indices = range(ddp_rank, len(train_task), ddp_world_size)
        rank_iter = itertools.cycle(rank_indices)
        for example_idx in rank_iter:
            conversation = train_task[example_idx]
            tokens = tokenizer.render_for_completion(conversation)
            prefix_length = len(tokens)

            model.eval()
            generated_token_sequences = []
            masks = []
            num_sampling_steps = args.num_samples // args.device_batch_size
            for sampling_step in range(num_sampling_steps):
                seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
                generated_batch, masks_batch = engine.generate_batch(
                    tokens,
                    num_samples=args.device_batch_size,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    seed=seed,
                )
                generated_token_sequences.extend(generated_batch)
                masks.extend(masks_batch)

            rewards = []
            for sample_tokens in generated_token_sequences:
                generated_tokens = sample_tokens[prefix_length:]
                generated_text = tokenizer.decode(generated_tokens)
                rewards.append(train_task.reward(conversation, generated_text))

            max_length = max(len(seq) for seq in generated_token_sequences)
            padded_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
            padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]

            ids = torch.tensor(padded_sequences, dtype=torch.long, device=device)
            mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
            inputs = ids[:, :-1]
            targets = ids[:, 1:].clone()
            targets[mask_ids[:, 1:] == 0] = -1

            rewards = torch.tensor(rewards, dtype=torch.float, device=device)
            advantages = build_advantages(rewards, args.objective)
            yield generated_token_sequences, inputs, targets, rewards, advantages

    def run_gsm8k_eval(task, tokenizer, engine, max_examples=None, num_samples=1, max_completion_tokens=256, temperature=0.0, top_k=50):
        max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
        for idx in range(ddp_rank, max_examples, ddp_world_size):
            conversation = task[idx]
            tokens = tokenizer.render_for_completion(conversation)
            prefix_length = len(tokens)
            assert num_samples <= args.device_batch_size
            generated_token_sequences, masks = engine.generate_batch(
                tokens,
                num_samples=num_samples,
                max_tokens=max_completion_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            outcomes = []
            for sample_tokens in generated_token_sequences:
                generated_tokens = sample_tokens[prefix_length:]
                generated_text = tokenizer.decode(generated_tokens)
                outcomes.append({"is_correct": task.evaluate(conversation, generated_text)})
            yield {"idx": idx, "outcomes": outcomes}

    optimizer = model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        weight_decay=args.weight_decay,
    )
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]

    def get_lr_multiplier(it):
        return 1.0 - it / num_steps

    print0(f"Total sequences per step: {args.examples_per_step * args.num_samples}")
    assert args.examples_per_step % ddp_world_size == 0, "examples_per_step must be divisible by number of ranks"
    examples_per_rank = args.examples_per_step // ddp_world_size
    print0(f"Calculated examples per rank: {examples_per_rank}")

    for step in range(num_steps):
        batch_iterator = get_batch(step)

        if step % args.eval_every == 0:
            model.eval()
            passk = torch.zeros(args.device_batch_size, device=device)
            records = list(
                run_gsm8k_eval(
                    val_task,
                    tokenizer,
                    engine,
                    num_samples=args.device_batch_size,
                    max_examples=args.eval_examples,
                    temperature=1.0,
                )
            )
            for k in range(1, args.device_batch_size + 1):
                passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
            num_records = torch.tensor(len(records), dtype=torch.long, device=device)
            if ddp:
                dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
                dist.all_reduce(passk, op=dist.ReduceOp.SUM)
            passk = passk / num_records.item()
            print0(f"Step {step} | " + ", ".join(f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, args.device_batch_size + 1)))
            wandb_run.log({"step": step, **{f"pass@{k}": passk[k - 1].item() for k in range(1, args.device_batch_size + 1)}})

        rewards_list = []
        sequence_lengths = []
        loss_values = []
        kl_values = []
        ratio_values = []
        rollout_batches = []

        for example_step in range(examples_per_rank):
            sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
            old_token_logps_all = collect_token_logps(model, inputs_all, targets_all, args.device_batch_size)
            ref_token_logps_all = None
            if ref_model is not None:
                ref_token_logps_all = collect_token_logps(ref_model, inputs_all, targets_all, args.device_batch_size)

            rollout_batches.append(
                {
                    "example_step": example_step,
                    "sequences_all": sequences_all,
                    "inputs_all": inputs_all,
                    "targets_all": targets_all,
                    "rewards_all": rewards_all,
                    "advantages_all": advantages_all,
                    "old_token_logps_all": old_token_logps_all,
                    "ref_token_logps_all": ref_token_logps_all,
                }
            )
            rewards_list.append(rewards_all.mean().item())
            sequence_lengths.extend(len(seq) for seq in sequences_all)

        lrm = get_lr_multiplier(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm

        model.train()
        for update_epoch in range(args.update_epochs):
            model.zero_grad(set_to_none=True)
            for batch in rollout_batches:
                inputs_all = batch["inputs_all"]
                targets_all = batch["targets_all"]
                advantages_all = batch["advantages_all"]
                old_token_logps_all = batch["old_token_logps_all"]
                ref_token_logps_all = batch["ref_token_logps_all"]
                rewards_all = batch["rewards_all"]

                for b0 in range(0, inputs_all.size(0), args.device_batch_size):
                    b1 = min(b0 + args.device_batch_size, inputs_all.size(0))
                    inputs = inputs_all[b0:b1]
                    targets = targets_all[b0:b1]
                    advantages = advantages_all[b0:b1]
                    old_token_logps = old_token_logps_all[b0:b1]
                    ref_token_logps = None if ref_token_logps_all is None else ref_token_logps_all[b0:b1]

                    token_logps, valid = compute_token_logps(model, inputs, targets)
                    loss, metrics = compute_objective_loss(
                        objective=args.objective,
                        token_logps=token_logps,
                        valid=valid,
                        advantages=advantages,
                        old_token_logps=old_token_logps,
                        ref_token_logps=ref_token_logps,
                        kl_beta=args.kl_beta,
                        clip_epsilon=args.clip_epsilon,
                        log_ratio_cap=args.log_ratio_cap,
                    )
                    loss = loss / examples_per_rank
                    loss.backward()

                    loss_values.append(loss.detach().item())
                    kl_values.append(metrics["sampled_kl"].item())
                    if "ratio_mean" in metrics:
                        ratio_values.append(metrics["ratio_mean"].item())

                    print0(
                        f"Step {step}/{num_steps} | Example {batch['example_step']} | Epoch {update_epoch} | "
                        f"loss: {loss.item():.6f} | reward: {rewards_all.mean().item():.4f}"
                    )

            optimizer.step()

        mean_reward = sum(rewards_list) / len(rewards_list)
        mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
        mean_loss = sum(loss_values) / len(loss_values) if loss_values else 0.0
        mean_kl = sum(kl_values) / len(kl_values) if kl_values else 0.0
        mean_ratio = sum(ratio_values) / len(ratio_values) if ratio_values else 1.0

        mean_reward = maybe_reduce_mean_scalar(mean_reward, ddp, device)
        mean_sequence_length = maybe_reduce_mean_scalar(mean_sequence_length, ddp, device)
        mean_loss = maybe_reduce_mean_scalar(mean_loss, ddp, device)
        mean_kl = maybe_reduce_mean_scalar(mean_kl, ddp, device)
        mean_ratio = maybe_reduce_mean_scalar(mean_ratio, ddp, device)

        print0(
            f"Step {step}/{num_steps} | objective={args.objective} | "
            f"reward: {mean_reward:.4f} | seq_len: {mean_sequence_length:.2f} | "
            f"loss: {mean_loss:.6f} | sampled_kl: {mean_kl:.6f} | ratio: {mean_ratio:.4f}"
        )
        wandb_run.log(
            {
                "step": step,
                "reward": mean_reward,
                "sequence_length": mean_sequence_length,
                "loss": mean_loss,
                "sampled_kl": mean_kl,
                "ratio_mean": mean_ratio,
            }
        )

        model.zero_grad(set_to_none=True)
        wandb_run.log({"step": step, "lrm": lrm})

        if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
            base_dir = get_base_dir()
            depth = model.config.n_layer
            output_dirname = args.model_tag if args.model_tag else f"d{depth}"
            output_dirname = f"{output_dirname}_{args.objective}"
            checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", output_dirname)
            save_checkpoint(
                checkpoint_dir,
                step,
                model.state_dict(),
                None,
                {"model_config": model.config.__dict__},
            )
            print0(f"Saved model checkpoint to {checkpoint_dir}")

    get_report().log(section="Chat Universal RL", data=[user_config])
    if hasattr(wandb_run, "finish"):
        wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
