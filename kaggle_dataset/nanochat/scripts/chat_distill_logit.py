"""
Soft-logit distillation trainer for nanochat students.

This script distills a nanochat student from a nanochat teacher by matching the
teacher's next-token distribution on assistant tokens. Teacher and student must
share the same nanochat tokenizer/vocabulary; external HF teachers with different
tokenizers are not supported for token-level logit distillation.

Supported input formats:
- sft JSONL:
    [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]
- preference JSONL:
    {"prompt":"...","chosen":"...","rejected":"..."}
  If the row also has "messages", the prompt context is preserved and `chosen`
  is appended as the assistant response.

Examples:
python3 -m scripts.chat_distill_logit --teacher-source=sft --student-source=base --data-path=teacher_sft.jsonl
python3 -m scripts.chat_distill_logit --teacher-checkpoint-dir ~/.cache/nanochat/chatsft_checkpoints --teacher-tag=d12 --data-path=teacher_sft.jsonl
"""

import argparse
import copy
import json
import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb

from nanochat.checkpoint_manager import load_model, load_model_from_dir, save_checkpoint
from nanochat.common import (
    COMPUTE_DTYPE,
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    is_ddp_initialized,
    print0,
)
from nanochat.report import get_report
from tasks.common import Task


BUILTIN_SOURCES = ["base", "sft", "rl", "ppo", "ppo_standard", "grpo"]


class DistillJSON(Task):
    def __init__(self, filepath, data_format="sft", **kwargs):
        super().__init__(**kwargs)
        self.filepath = os.path.expanduser(filepath)
        self.data_format = data_format
        self.rows = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if data_format == "sft":
                    assert isinstance(row, list), f"Expected list-of-messages JSONL for sft, got {type(row)}"
                    self.rows.append(row)
                elif data_format == "preference":
                    assert isinstance(row, dict), f"Expected object JSONL for preference, got {type(row)}"
                    if "messages" in row:
                        messages = copy.deepcopy(row["messages"])
                    else:
                        messages = [{"role": "user", "content": row["prompt"]}]
                    while messages and messages[-1]["role"] == "assistant":
                        messages.pop()
                    messages.append({"role": "assistant", "content": row["chosen"]})
                    self.rows.append(messages)
                else:
                    raise ValueError(f"Unsupported data_format: {data_format}")

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.rows)

    def get_example(self, index):
        return {"messages": self.rows[index]}


def parse_args():
    parser = argparse.ArgumentParser(description="Soft-logit distill a nanochat student from a nanochat teacher")
    parser.add_argument("--run", type=str, default="dummy")
    parser.add_argument("--device-type", type=str, default="")

    parser.add_argument("--student-source", type=str, default="base", choices=BUILTIN_SOURCES + ["custom"])
    parser.add_argument("--student-checkpoint-dir", type=str, default=None)
    parser.add_argument("--student-tag", type=str, default=None)
    parser.add_argument("--student-step", type=int, default=None)

    parser.add_argument("--teacher-source", type=str, default="sft", choices=BUILTIN_SOURCES + ["custom"])
    parser.add_argument("--teacher-checkpoint-dir", type=str, default=None)
    parser.add_argument("--teacher-tag", type=str, default=None)
    parser.add_argument("--teacher-step", type=int, default=None)

    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-format", type=str, default="sft", choices=["sft", "preference"])
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=1024)

    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=0, help="0 = exact full-vocab KL; >0 = sparse teacher top-k KD against the student's full-softmax mass")
    parser.add_argument("--soft-loss-weight", type=float, default=1.0)
    parser.add_argument("--hard-loss-weight", type=float, default=0.0, help="Optional CE on the hard target tokens")

    parser.add_argument("--embedding-lr", type=float, default=0.05)
    parser.add_argument("--unembedding-lr", type=float, default=0.001)
    parser.add_argument("--matrix-lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--init-lr-frac", type=float, default=0.2)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--final-lr-frac", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=200)
    return parser.parse_args()


def load_nanochat_model(source, checkpoint_dir, device, phase, model_tag=None, step=None):
    if checkpoint_dir is not None:
        checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir))
        return load_model_from_dir(checkpoint_dir, device, phase=phase, model_tag=model_tag, step=step)
    assert source != "custom", "--*-checkpoint-dir is required when source=custom"
    return load_model(source, device, phase=phase, model_tag=model_tag, step=step)


def split_dataset(dataset, val_ratio, seed):
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    val_size = int(len(indices) * val_ratio)
    if val_ratio > 0.0 and len(indices) > 1:
        val_size = max(1, val_size)
        val_size = min(len(indices) - 1, val_size)
    train_rows = [dataset.rows[i] for i in indices[val_size:]]
    val_rows = [dataset.rows[i] for i in indices[:val_size]]

    train_ds = DistillJSON.__new__(DistillJSON)
    Task.__init__(train_ds)
    train_ds.filepath = dataset.filepath
    train_ds.data_format = dataset.data_format
    train_ds.rows = train_rows

    val_ds = DistillJSON.__new__(DistillJSON)
    Task.__init__(val_ds)
    val_ds.filepath = dataset.filepath
    val_ds.data_format = dataset.data_format
    val_ds.rows = val_rows
    return train_ds, val_ds


def num_global_batches(num_examples, batch_size, ddp_world_size):
    if num_examples == 0:
        return 0
    global_batch_size = batch_size * ddp_world_size
    return (num_examples + global_batch_size - 1) // global_batch_size


def iter_rank_indices(num_examples, batch_size, ddp_rank, ddp_world_size, shuffle, seed, pad_to_global):
    indices = list(range(num_examples))
    if shuffle:
        random.Random(seed).shuffle(indices)
    if not indices:
        return

    global_batch_size = batch_size * ddp_world_size
    if pad_to_global:
        steps = num_global_batches(num_examples, batch_size, ddp_world_size)
        total_size = steps * global_batch_size
        if total_size > len(indices):
            repeat = (total_size + len(indices) - 1) // len(indices)
            indices = (indices * repeat)[:total_size]
    else:
        total_size = len(indices)

    for global_start in range(0, total_size, global_batch_size):
        local_start = global_start + ddp_rank * batch_size
        chunk = indices[local_start:local_start + batch_size]
        if chunk:
            yield chunk


def render_for_distill(tokenizer, conversation, max_seq_len):
    max_tokens = max_seq_len + 1
    ids, mask = tokenizer.render_conversation(conversation, max_tokens=1_000_000_000)
    if len(ids) <= max_tokens:
        return ids, mask

    messages = conversation["messages"]
    has_system = bool(messages) and messages[0]["role"] == "system"
    user_starts = [i for i, message in enumerate(messages) if message["role"] == "user"]
    keep_system_options = [True, False] if has_system else [False]
    for keep_system in keep_system_options:
        for start in user_starts:
            candidate_messages = messages[start:]
            if keep_system:
                candidate_messages = [messages[0]] + candidate_messages
            candidate = {"messages": candidate_messages}
            ids, mask = tokenizer.render_conversation(candidate, max_tokens=1_000_000_000)
            if len(ids) <= max_tokens:
                return ids, mask

    bos = tokenizer.get_bos_token_id()
    ids = [bos] * max_tokens
    mask = [0] * max_tokens
    return ids, mask


def make_batches(dataset, tokenizer, batch_size, max_seq_len, device, shuffle, seed, ddp_rank=0, ddp_world_size=1, pad_to_global=False):
    bos = tokenizer.get_bos_token_id()
    for chunk in iter_rank_indices(len(dataset), batch_size, ddp_rank, ddp_world_size, shuffle, seed, pad_to_global):
        conversations = [dataset[idx] for idx in chunk]
        ids_rows = []
        mask_rows = []
        for conversation in conversations:
            ids, mask = render_for_distill(tokenizer, conversation, max_seq_len)
            ids_rows.append(ids)
            mask_rows.append(mask)
        max_len = max(len(row) for row in ids_rows)
        padded_ids = [row + [bos] * (max_len - len(row)) for row in ids_rows]
        padded_masks = [row + [0] * (max_len - len(row)) for row in mask_rows]
        ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        mask = torch.tensor(padded_masks, dtype=torch.long, device=device)
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()
        targets[mask[:, 1:] == 0] = -1
        yield inputs, targets


def soft_kl_per_token(student_logits, teacher_logits, temperature, top_k):
    if top_k > 0:
        top_k = min(top_k, teacher_logits.size(-1))
        teacher_top_logits, teacher_top_idx = torch.topk(teacher_logits, top_k, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_top_logits / temperature, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        student_top_log_probs = student_log_probs.gather(-1, teacher_top_idx)
        kl = (teacher_probs * (teacher_log_probs - student_top_log_probs)).sum(dim=-1)
    else:
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    return kl * (temperature ** 2)


def compute_distill_sums(student, teacher, inputs, targets, args):
    valid = targets >= 0
    valid_tokens = valid.sum()
    if valid_tokens.item() == 0:
        zero = student(inputs).sum() * 0.0
        return zero, zero.detach(), zero.detach(), valid_tokens

    student_logits = student(inputs)

    if args.soft_loss_weight > 0.0:
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        soft_per_token = soft_kl_per_token(student_logits, teacher_logits, args.temperature, args.top_k)
        soft_sum = soft_per_token.masked_select(valid).sum()
    else:
        soft_sum = student_logits.sum() * 0.0

    if args.hard_loss_weight > 0.0:
        hard_sum = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction="sum",
        )
    else:
        hard_sum = soft_sum.detach() * 0.0

    loss_sum = args.soft_loss_weight * soft_sum + args.hard_loss_weight * hard_sum
    return loss_sum, soft_sum.detach(), hard_sum.detach(), valid_tokens


@torch.no_grad()
def evaluate_dataset(student, teacher, dataset, tokenizer, args, device, ddp_rank, ddp_world_size):
    if len(dataset) == 0:
        return None

    was_training = student.training
    student.eval()
    teacher.eval()

    total_sum = torch.tensor(0.0, dtype=torch.float, device=device)
    soft_sum = torch.tensor(0.0, dtype=torch.float, device=device)
    hard_sum = torch.tensor(0.0, dtype=torch.float, device=device)
    token_count = torch.tensor(0, dtype=torch.long, device=device)
    for inputs, targets in make_batches(
        dataset,
        tokenizer,
        args.batch_size,
        args.max_seq_len,
        device,
        shuffle=False,
        seed=0,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        pad_to_global=False,
    ):
        batch_total, batch_soft, batch_hard, valid_tokens = compute_distill_sums(student, teacher, inputs, targets, args)
        if valid_tokens.item() == 0:
            continue
        total_sum += batch_total.detach()
        soft_sum += batch_soft
        hard_sum += batch_hard
        token_count += valid_tokens

    if ddp_world_size > 1:
        for tensor in (total_sum, soft_sum, hard_sum, token_count):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    if was_training:
        student.train()
    if token_count.item() == 0:
        return None

    denom = token_count.clamp_min(1)
    return {
        "loss": (total_sum / denom).item(),
        "soft_loss": (soft_sum / denom).item(),
        "hard_loss": (hard_sum / denom).item(),
    }


def get_lr_multiplier(progress, init_lr_frac, warmup_ratio, final_lr_frac):
    progress = min(max(progress, 0.0), 1.0)
    if warmup_ratio > 0.0 and progress < warmup_ratio:
        warmup_progress = progress / warmup_ratio
        return init_lr_frac + (1.0 - init_lr_frac) * warmup_progress
    decay_progress = (progress - warmup_ratio) / max(1.0 - warmup_ratio, 1e-8)
    decay_progress = min(max(decay_progress, 0.0), 1.0)
    return (1.0 - decay_progress) * 1.0 + decay_progress * final_lr_frac


def assert_tokenizers_compatible(tokenizer, teacher_tokenizer):
    assert teacher_tokenizer.get_vocab_size() == tokenizer.get_vocab_size(), (
        f"Teacher/student tokenizer mismatch: {teacher_tokenizer.get_vocab_size()} vs {tokenizer.get_vocab_size()}"
    )
    special_tokens = sorted(set(tokenizer.get_special_tokens()) | set(teacher_tokenizer.get_special_tokens()))
    for token in special_tokens:
        student_id = tokenizer.encode_special(token)
        teacher_id = teacher_tokenizer.encode_special(token)
        assert student_id == teacher_id, (
            f"Teacher/student special-token mismatch for {token}: {teacher_id} vs {student_id}"
        )
    for token_id in range(tokenizer.get_vocab_size()):
        assert teacher_tokenizer.id_to_token(token_id) == tokenizer.id_to_token(token_id), (
            f"Teacher/student tokenizer id {token_id} maps to different tokens"
        )


def make_grad_scaler(device):
    if device.type != "cuda" or COMPUTE_DTYPE != torch.float16:
        return None
    try:
        return torch.amp.GradScaler("cuda", init_scale=1024)
    except TypeError:
        return torch.amp.GradScaler(init_scale=1024)


def main():
    args = parse_args()
    user_config = vars(args).copy()
    assert 0.0 <= args.val_ratio < 1.0, "--val-ratio must be in [0, 1)"
    assert args.num_epochs >= 0, "--num-epochs must be non-negative"
    assert args.batch_size > 0, "--batch-size must be positive"
    assert args.max_seq_len >= 2, "--max-seq-len must be at least 2"
    assert args.temperature > 0.0, "--temperature must be positive"
    assert args.top_k >= 0, "--top-k must be non-negative"
    assert args.soft_loss_weight >= 0.0, "--soft-loss-weight must be non-negative"
    assert args.hard_loss_weight >= 0.0, "--hard-loss-weight must be non-negative"
    assert args.soft_loss_weight + args.hard_loss_weight > 0.0, "At least one loss weight must be positive"
    assert 0.0 <= args.init_lr_frac <= 1.0, "--init-lr-frac must be between 0 and 1"
    assert 0.0 <= args.warmup_ratio <= 1.0, "--warmup-ratio must be between 0 and 1"
    assert 0.0 <= args.final_lr_frac <= 1.0, "--final-lr-frac must be between 0 and 1"
    assert args.save_every > 0, "--save-every must be positive"
    assert args.eval_every >= 0, "--eval-every must be non-negative"

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0

    use_dummy_wandb = args.run == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-logit-distill", name=args.run, config=user_config)

    student, tokenizer, meta = load_nanochat_model(
        args.student_source,
        args.student_checkpoint_dir,
        device,
        phase="train",
        model_tag=args.student_tag,
        step=args.student_step,
    )
    teacher, teacher_tokenizer, _ = load_nanochat_model(
        args.teacher_source,
        args.teacher_checkpoint_dir,
        device,
        phase="eval",
        model_tag=args.teacher_tag,
        step=args.teacher_step,
    )
    assert student.config.vocab_size == teacher.config.vocab_size, (
        f"Teacher/student vocab mismatch: {teacher.config.vocab_size} vs {student.config.vocab_size}"
    )
    assert_tokenizers_compatible(tokenizer, teacher_tokenizer)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    dataset = DistillJSON(args.data_path, data_format=args.data_format)
    train_dataset, val_dataset = split_dataset(dataset, args.val_ratio, args.shuffle_seed)
    print0(f"Logit distill dataset: train={len(train_dataset)} | val={len(val_dataset)}")
    assert len(train_dataset) > 0, "No logit-distillation training examples loaded"

    optimizer = student.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        weight_decay=args.weight_decay,
    )
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    scaler = make_grad_scaler(device)
    if scaler is not None:
        print0("GradScaler enabled for fp16 logit distillation")

    train_steps_per_epoch = num_global_batches(len(train_dataset), args.batch_size, ddp_world_size)
    total_steps = max(1, train_steps_per_epoch * args.num_epochs)
    print0(f"Planned logit distillation steps: {total_steps}")

    step = 0
    best_val_loss = None
    smooth_loss = 0.0
    ema_beta = 0.9

    for epoch in range(args.num_epochs):
        batch_iter = make_batches(
            train_dataset,
            tokenizer,
            args.batch_size,
            args.max_seq_len,
            device,
            shuffle=True,
            seed=args.shuffle_seed + epoch,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            pad_to_global=ddp_world_size > 1,
        )
        for inputs, targets in batch_iter:
            valid_tokens = (targets >= 0).sum()
            global_valid_tokens = valid_tokens.detach().clone()
            if ddp:
                dist.all_reduce(global_valid_tokens, op=dist.ReduceOp.SUM)
            if global_valid_tokens.item() == 0:
                print0(f"step {step:05d} | epoch {epoch} | skipped batch with no supervised tokens")
                continue

            progress = step / max(total_steps - 1, 1)
            lrm = get_lr_multiplier(progress, args.init_lr_frac, args.warmup_ratio, args.final_lr_frac)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lrm

            if args.eval_every > 0 and len(val_dataset) > 0 and (step == 0 or step % args.eval_every == 0):
                val_metrics = evaluate_dataset(student, teacher, val_dataset, tokenizer, args, device, ddp_rank, ddp_world_size)
                if val_metrics is None:
                    print0(f"Step {step} | val skipped: no supervised validation tokens")
                else:
                    best_val_loss = val_metrics["loss"] if best_val_loss is None else min(best_val_loss, val_metrics["loss"])
                    print0(
                        f"Step {step} | val_loss: {val_metrics['loss']:.6f} | "
                        f"val_soft: {val_metrics['soft_loss']:.6f} | val_hard: {val_metrics['hard_loss']:.6f}"
                    )
                    wandb_run.log({
                        "step": step,
                        "val/loss": val_metrics["loss"],
                        "val/soft_loss": val_metrics["soft_loss"],
                        "val/hard_loss": val_metrics["hard_loss"],
                    })
                student.train()

            t0 = time.time()
            loss_sum, soft_sum, hard_sum, _ = compute_distill_sums(student, teacher, inputs, targets, args)
            loss = loss_sum * ddp_world_size / global_valid_tokens.clamp_min(1).to(loss_sum.dtype)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if is_ddp_initialized():
                    for found_inf in scaler._found_inf_per_device(optimizer).values():
                        dist.all_reduce(found_inf, op=dist.ReduceOp.MAX)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            dt = time.time() - t0

            total_item = loss_sum.detach()
            soft_item = soft_sum.detach()
            hard_item = hard_sum.detach()
            if ddp:
                for tensor in (total_item, soft_item, hard_item):
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            total_item = total_item / global_valid_tokens.clamp_min(1)
            soft_item = soft_item / global_valid_tokens.clamp_min(1)
            hard_item = hard_item / global_valid_tokens.clamp_min(1)
            smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * total_item.item()
            debiased = smooth_loss / (1 - ema_beta ** (step + 1))
            print0(
                f"step {step:05d} | epoch {epoch} | loss: {debiased:.6f} | "
                f"soft: {soft_item.item():.6f} | hard: {hard_item.item():.6f} | "
                f"lrm: {lrm:.4f} | dt: {1000 * dt:.2f}ms"
            )
            wandb_run.log({
                "step": step,
                "train/loss": debiased,
                "train/soft_loss": soft_item.item(),
                "train/hard_loss": hard_item.item(),
                "train/lrm": lrm,
                "train/dt": dt,
            })

            if master_process and step > 0 and step % args.save_every == 0:
                base_dir = get_base_dir()
                depth = student.config.n_layer
                model_tag = args.student_tag if args.student_tag else f"d{depth}"
                checkpoint_dir = os.path.join(base_dir, "chatdistill_logit_checkpoints", model_tag)
                save_checkpoint(
                    checkpoint_dir,
                    step,
                    student.state_dict(),
                    None,
                    {
                        "step": step,
                        "best_val_loss": best_val_loss,
                        "model_config": student.config.__dict__,
                        "user_config": user_config,
                    },
                )
                print0(f"Saved logit distillation checkpoint to {checkpoint_dir}")

            step += 1

    if master_process:
        base_dir = get_base_dir()
        depth = student.config.n_layer
        model_tag = args.student_tag if args.student_tag else f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "chatdistill_logit_checkpoints", model_tag)
        save_checkpoint(
            checkpoint_dir,
            step,
            student.state_dict(),
            None,
            {
                "step": step,
                "best_val_loss": best_val_loss,
                "model_config": student.config.__dict__,
                "user_config": user_config,
            },
        )
        print0(f"Saved final logit distillation checkpoint to {checkpoint_dir}")

    get_report().log(
        section="Chat Logit Distill",
        data=[
            user_config,
            {
                "Train examples": len(train_dataset),
                "Val examples": len(val_dataset),
                "Best val loss": best_val_loss,
                "Total steps": step,
            },
        ],
    )
    if hasattr(wandb_run, "finish"):
        wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
