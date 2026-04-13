"""
Standalone distillation trainer for nanochat students.

This script does not modify any existing file. It trains a nanochat student on
teacher-generated JSONL data, typically produced by scripts/chat_distill_data.py.

Supported input formats:
- sft JSONL:
    [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]
- preference JSONL:
    {"prompt":"...","chosen":"...","rejected":"..."}
  In this case the student is trained on the `chosen` response.

Examples:
python3 -m scripts.chat_distill --data-path teacher_sft.jsonl --data-format sft
python3 -m scripts.chat_distill --data-path teacher_prefs.jsonl --data-format preference
"""

import argparse
import json
import math
import os
import random
import time

import torch
import torch.distributed as dist
import wandb

from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.common import DummyWandb, autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0
from nanochat.report import get_report
from tasks.common import Task


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
                    self.rows.append(
                        [
                            {"role": "user", "content": row["prompt"]},
                            {"role": "assistant", "content": row["chosen"]},
                        ]
                    )
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
    parser = argparse.ArgumentParser(description="Distill a nanochat student on teacher JSONL data")
    parser.add_argument("--run", type=str, default="dummy")
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--student-source", type=str, default="base", choices=["base", "sft", "rl"])
    parser.add_argument("--student-tag", type=str, default=None)
    parser.add_argument("--student-step", type=int, default=None)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-format", type=str, default="sft", choices=["sft", "preference"])
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--embedding-lr", type=float, default=0.1)
    parser.add_argument("--unembedding-lr", type=float, default=0.002)
    parser.add_argument("--matrix-lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--init-lr-frac", type=float, default=0.2)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--final-lr-frac", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=200)
    return parser.parse_args()


def split_dataset(dataset, val_ratio, seed):
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    val_size = int(len(indices) * val_ratio)
    val_idx = set(indices[:val_size])
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


def make_batches(dataset, tokenizer, batch_size, max_seq_len, device, shuffle, seed):
    bos = tokenizer.get_bos_token_id()
    indices = list(range(len(dataset)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    for i in range(0, len(indices), batch_size):
        chunk = indices[i:i + batch_size]
        conversations = [dataset[idx] for idx in chunk]
        ids_rows = []
        mask_rows = []
        for conversation in conversations:
            ids, mask = tokenizer.render_conversation(conversation, max_tokens=max_seq_len + 1)
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


@torch.no_grad()
def evaluate_dataset(model, dataset, tokenizer, batch_size, max_seq_len, device, ddp_rank, ddp_world_size):
    if len(dataset) == 0:
        return 0.0
    losses = []
    for step, (inputs, targets) in enumerate(make_batches(dataset, tokenizer, batch_size, max_seq_len, device, shuffle=False, seed=0)):
        if step % ddp_world_size != ddp_rank:
            continue
        loss = model(inputs, targets)
        losses.append(loss.detach())
    if not losses:
        loss_tensor = torch.tensor(0.0, device=device)
    else:
        loss_tensor = torch.stack(losses).mean()
    if ddp_world_size > 1:
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    return loss_tensor.item()


def get_lr_multiplier(progress, warmup_ratio, final_lr_frac):
    if progress < warmup_ratio:
        return (progress + 1e-8) / max(warmup_ratio, 1e-8)
    decay_progress = (progress - warmup_ratio) / max(1.0 - warmup_ratio, 1e-8)
    decay_progress = min(max(decay_progress, 0.0), 1.0)
    return (1.0 - decay_progress) * 1.0 + decay_progress * final_lr_frac


def main():
    args = parse_args()
    user_config = vars(args).copy()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0

    use_dummy_wandb = args.run == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-distill", name=args.run, config=user_config)

    model, tokenizer, meta = load_model(args.student_source, device, phase="train", model_tag=args.student_tag, step=args.student_step)
    dataset = DistillJSON(args.data_path, data_format=args.data_format)
    train_dataset, val_dataset = split_dataset(dataset, args.val_ratio, args.shuffle_seed)
    print0(f"Distill dataset: train={len(train_dataset)} | val={len(val_dataset)}")

    optimizer = model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        weight_decay=args.weight_decay,
    )
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]

    train_steps_per_epoch = max(1, math.ceil(len(train_dataset) / max(args.batch_size * ddp_world_size, 1)))
    total_steps = max(1, train_steps_per_epoch * args.num_epochs)
    print0(f"Planned distillation steps: {total_steps}")

    step = 0
    best_val_loss = float("inf")
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
        )
        for local_step, (inputs, targets) in enumerate(batch_iter):
            if local_step % ddp_world_size != ddp_rank:
                continue

            progress = step / max(total_steps - 1, 1)
            lrm = get_lr_multiplier(progress, args.warmup_ratio, args.final_lr_frac)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lrm

            if args.eval_every > 0 and (step == 0 or step % args.eval_every == 0):
                model.eval()
                val_loss = evaluate_dataset(model, val_dataset, tokenizer, args.batch_size, args.max_seq_len, device, ddp_rank, ddp_world_size)
                best_val_loss = min(best_val_loss, val_loss)
                print0(f"Step {step} | val_loss: {val_loss:.6f}")
                wandb_run.log({"step": step, "val/loss": val_loss})
                model.train()

            t0 = time.time()
            loss = model(inputs, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            dt = time.time() - t0

            loss_item = loss.detach()
            if ddp:
                dist.all_reduce(loss_item, op=dist.ReduceOp.AVG)
            smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_item.item()
            debiased = smooth_loss / (1 - ema_beta ** (step + 1))
            print0(
                f"step {step:05d} | epoch {epoch} | loss: {debiased:.6f} | "
                f"lrm: {lrm:.4f} | dt: {1000 * dt:.2f}ms"
            )
            wandb_run.log({"step": step, "train/loss": debiased, "train/lrm": lrm, "train/dt": dt})

            if master_process and step > 0 and step % args.save_every == 0:
                base_dir = get_base_dir()
                depth = model.config.n_layer
                model_tag = args.student_tag if args.student_tag else f"d{depth}"
                checkpoint_dir = os.path.join(base_dir, "chatdistill_checkpoints", model_tag)
                save_checkpoint(
                    checkpoint_dir,
                    step,
                    model.state_dict(),
                    None,
                    {
                        "step": step,
                        "best_val_loss": best_val_loss,
                        "model_config": model.config.__dict__,
                        "user_config": user_config,
                    },
                )
                print0(f"Saved distillation checkpoint to {checkpoint_dir}")

            step += 1

    if master_process:
        base_dir = get_base_dir()
        depth = model.config.n_layer
        model_tag = args.student_tag if args.student_tag else f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "chatdistill_checkpoints", model_tag)
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None,
            {
                "step": step,
                "best_val_loss": best_val_loss,
                "model_config": model.config.__dict__,
                "user_config": user_config,
            },
        )
        print0(f"Saved final distillation checkpoint to {checkpoint_dir}")

    get_report().log(
        section="Chat Distill",
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
