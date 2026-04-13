"""
Direct Preference Optimization (DPO) for nanochat.

This script is standalone and does not modify the existing SFT/RL scripts.
It supports either:
- preference JSONL data
- synthetic GSM8K preferences for quick experimentation

JSONL format (one object per line):
{"prompt": "Question text", "chosen": "Preferred assistant response", "rejected": "Dispreferred assistant response"}

Examples:
python3 -m scripts.chat_dpo --preference-source=gsm8k
python3 -m scripts.chat_dpo --preference-source=jsonl --preference-path=/path/prefs.jsonl
"""

import argparse
import json
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
from tasks.gsm8k import GSM8K, extract_answer


def parse_args():
    parser = argparse.ArgumentParser(description="DPO fine-tuning for nanochat")
    parser.add_argument("--run", type=str, default="dummy")
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--model-source", type=str, default="sft", choices=["sft", "rl"])
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument("--model-step", type=int, default=None)
    parser.add_argument("--reference-source", type=str, default="sft", choices=["sft", "rl"])
    parser.add_argument("--reference-tag", type=str, default=None)
    parser.add_argument("--reference-step", type=int, default=None)
    parser.add_argument("--preference-source", type=str, default="gsm8k", choices=["gsm8k", "jsonl"])
    parser.add_argument("--preference-path", type=str, default=None, help="JSONL file with prompt/chosen/rejected triples")
    parser.add_argument("--max-train-examples", type=int, default=4096)
    parser.add_argument("--max-val-examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="optional cap on total optimizer steps")#add by YQ
    parser.add_argument("--beta", type=float, default=0.1, help="DPO inverse temperature")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="conservative DPO smoothing")
    parser.add_argument("--embedding-lr", type=float, default=0.05)
    parser.add_argument("--unembedding-lr", type=float, default=0.002)
    parser.add_argument("--matrix-lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--init-lr-frac", type=float, default=0.2)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=100)
    return parser.parse_args()


def flatten_assistant_content(content):
    if isinstance(content, str):
        return content
    text_parts = []
    for part in content:
        text_parts.append(part["text"])
    return "".join(text_parts)


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
                prefs.append(
                    {
                        "prompt": row["prompt"],
                        "chosen": row["chosen"],
                        "rejected": row["rejected"],
                    }
                )
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
    for conversation in conversations:
        ids, mask = tokenizer.render_conversation(conversation)
        rows.append(ids)
        masks.append(mask)
    max_len = max(len(x) for x in rows)
    padded_ids = [row + [bos] * (max_len - len(row)) for row in rows]
    padded_masks = [mask + [0] * (max_len - len(mask)) for mask in masks]
    ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
    mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
    inputs = ids[:, :-1]
    targets = ids[:, 1:].clone()
    targets[mask_ids[:, 1:] == 0] = -1
    return inputs, targets


def sequence_logps(model, inputs, targets):
    valid = targets >= 0
    safe_targets = targets.clamp(min=0)
    logits = model(inputs)
    logprobs = F.log_softmax(logits, dim=-1)
    token_logps = logprobs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps.masked_fill(~valid, 0.0)
    return token_logps.sum(dim=-1)


@torch.no_grad()
def evaluate_dpo_loss(model, ref_model, tokenizer, dataset, batch_size, beta, label_smoothing, device, ddp_rank, ddp_world_size):
    losses = []
    for i in range(ddp_rank, len(dataset), ddp_world_size * batch_size):
        batch = dataset[i:i + batch_size]
        if not batch:
            continue
        chosen_conv = [build_conversation(row["prompt"], row["chosen"]) for row in batch]
        rejected_conv = [build_conversation(row["prompt"], row["rejected"]) for row in batch]
        chosen_inputs, chosen_targets = collate_conversations(tokenizer, chosen_conv, device)
        rejected_inputs, rejected_targets = collate_conversations(tokenizer, rejected_conv, device)
        pi_chosen = sequence_logps(model, chosen_inputs, chosen_targets)
        pi_rejected = sequence_logps(model, rejected_inputs, rejected_targets)
        ref_chosen = sequence_logps(ref_model, chosen_inputs, chosen_targets)
        ref_rejected = sequence_logps(ref_model, rejected_inputs, rejected_targets)
        logits = beta * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected))
        loss = -(1.0 - label_smoothing) * F.logsigmoid(logits) - label_smoothing * F.logsigmoid(-logits)
        losses.append(loss.mean())
    if not losses:
        return 0.0
    loss_tensor = torch.stack(losses).mean()
    if ddp_world_size > 1:
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    return loss_tensor.item()


def main():
    args = parse_args()
    user_config = vars(args).copy()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0

    use_dummy_wandb = args.run == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-dpo", name=args.run, config=user_config)

    model, tokenizer, meta = load_model(args.model_source, device, phase="train", model_tag=args.model_tag, step=args.model_step)
    ref_tag = args.reference_tag if args.reference_tag is not None else args.model_tag
    ref_step = args.reference_step if args.reference_step is not None else args.model_step
    ref_model, _, _ = load_model(args.reference_source, device, phase="eval", model_tag=ref_tag, step=ref_step)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    prefs = load_preferences(args)
    random.Random(42).shuffle(prefs)
    split = min(args.max_train_examples, len(prefs))
    train_dataset = prefs[:split]
    val_dataset = prefs[split:split + args.max_val_examples]
    print0(f"DPO train examples: {len(train_dataset)} | val examples: {len(val_dataset)}")

    optimizer = model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        weight_decay=args.weight_decay,
    )
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]

    num_steps_per_epoch = max(1, (len(train_dataset) + args.batch_size - 1) // args.batch_size)
    total_steps = num_steps_per_epoch * args.num_epochs #213-217 added by YQ
    if args.max_steps is not None:
        assert args.max_steps > 0, "--max-steps must be positive"
        total_steps = min(total_steps, args.max_steps)
    print0(f"Calculated number of steps: {total_steps}")
    global_step = 0

    for epoch in range(args.num_epochs):
        if global_step >= total_steps: #221-222 added by YQ to break out of epoch loop if max_steps is reached
            break
        random.Random(42 + epoch).shuffle(train_dataset)
        for start in range(ddp_rank * args.batch_size, len(train_dataset), ddp_world_size * args.batch_size):
            if global_step >= total_steps: #225-226 added by YQ to break out of batch loop if max_steps is reached
                break
            batch = train_dataset[start:start + args.batch_size]
            if not batch:
                continue

            chosen_conv = [build_conversation(row["prompt"], row["chosen"]) for row in batch]
            rejected_conv = [build_conversation(row["prompt"], row["rejected"]) for row in batch]
            chosen_inputs, chosen_targets = collate_conversations(tokenizer, chosen_conv, device)
            rejected_inputs, rejected_targets = collate_conversations(tokenizer, rejected_conv, device)

            model.train()
            pi_chosen = sequence_logps(model, chosen_inputs, chosen_targets)
            pi_rejected = sequence_logps(model, rejected_inputs, rejected_targets)
            with torch.no_grad():
                ref_chosen = sequence_logps(ref_model, chosen_inputs, chosen_targets)
                ref_rejected = sequence_logps(ref_model, rejected_inputs, rejected_targets)

            logits = args.beta * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected))
            loss = -(1.0 - args.label_smoothing) * F.logsigmoid(logits) - args.label_smoothing * F.logsigmoid(-logits)
            loss = loss.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            preference_acc = (logits > 0).float().mean()
            loss_item = loss.detach()
            acc_item = preference_acc.detach()
            if ddp:
                dist.all_reduce(loss_item, op=dist.ReduceOp.AVG)
                dist.all_reduce(acc_item, op=dist.ReduceOp.AVG)

            print0(
                f"Step {global_step} | epoch {epoch} | dpo_loss: {loss_item.item():.6f} | "
                f"pref_acc: {acc_item.item():.4f}"
            )
            wandb_run.log({"step": global_step, "dpo_loss": loss_item.item(), "preference_accuracy": acc_item.item()})

            if val_dataset and global_step % args.eval_every == 0:
                val_loss = evaluate_dpo_loss(
                    model, ref_model, tokenizer, val_dataset, args.batch_size,
                    args.beta, args.label_smoothing, device, ddp_rank, ddp_world_size
                )
                print0(f"Step {global_step} | val_dpo_loss: {val_loss:.6f}")
                wandb_run.log({"step": global_step, "val_dpo_loss": val_loss})

            if master_process and global_step > 0 and global_step % args.save_every == 0:
                base_dir = get_base_dir()
                depth = model.config.n_layer
                model_tag = args.model_tag if args.model_tag else f"d{depth}"
                checkpoint_dir = os.path.join(base_dir, "chatdpo_checkpoints", model_tag)
                save_checkpoint(checkpoint_dir, global_step, model.state_dict(), None, {"model_config": model.config.__dict__})
                print0(f"Saved DPO checkpoint to {checkpoint_dir}")

            global_step += 1

    if master_process:
        base_dir = get_base_dir()
        depth = model.config.n_layer
        model_tag = args.model_tag if args.model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "chatdpo_checkpoints", model_tag)
        save_checkpoint(checkpoint_dir, global_step, model.state_dict(), None, {"model_config": model.config.__dict__})
        print0(f"Saved final DPO checkpoint to {checkpoint_dir}")

    get_report().log(section="Chat DPO", data=[user_config])
    if hasattr(wandb_run, "finish"):
        wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
