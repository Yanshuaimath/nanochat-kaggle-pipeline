"""
Generate distillation data for nanochat using an external teacher model.

Default teacher:
    meta-llama/Llama-3.1-8B-Instruct

Supported input sources:
- gsm8k : prompts from tasks.gsm8k
- jsonl : prompts/messages from a JSONL file

Supported output formats:
- sft        : JSONL lines compatible with tasks.customjson.CustomJSON
- preference : JSONL with prompt/chosen/rejected triples, suitable for chat_dpo.py/chat_ppo.py

Examples:
python3 -m scripts.chat_distill_data --input-source=gsm8k --output-path=teacher_sft.jsonl
python3 -m scripts.chat_distill_data --input-source=gsm8k --output-format=preference --output-path=teacher_prefs.jsonl
python3 -m scripts.chat_distill_data --input-source=jsonl --input-path=prompts.jsonl --output-path=teacher_sft.jsonl
"""

import argparse
import json
import os
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Generate teacher distillation data for nanochat")
    parser.add_argument("--teacher-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--input-source", type=str, default="gsm8k", choices=["gsm8k", "jsonl"])
    parser.add_argument("--input-path", type=str, default=None, help="Required for --input-source=jsonl")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--output-format", type=str, default="sft", choices=["sft", "preference"])
    parser.add_argument("--max-examples", type=int, default=512)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--load-in-8bit", type=int, default=0, help="Requires bitsandbytes")
    parser.add_argument("--chat-style", type=str, default="direct", choices=["direct", "solution"], help="How to prompt the teacher")
    parser.add_argument("--system-prompt", type=str, default="You are a precise and helpful assistant.")
    parser.add_argument("--rejected-style", type=str, default="perturb", choices=["perturb", "resample"], help="How to create rejected samples for preference output")
    return parser.parse_args()


def require_transformers():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency for teacher generation. Please install at least:\n"
            "  pip install torch transformers accelerate\n"
            "Optional for --load-in-8bit:\n"
            "  pip install bitsandbytes\n"
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def flatten_assistant_content(content):
    if isinstance(content, str):
        return content
    return "".join(part["text"] for part in content)


def load_prompts_from_gsm8k(args):
    try:
        from tasks.gsm8k import GSM8K
    except ModuleNotFoundError as exc:
        raise SystemExit("tasks.gsm8k requires the datasets package. Please install: pip install datasets") from exc
    task = GSM8K(subset="main", split=args.split)
    examples = []
    stop = min(len(task), args.start + args.max_examples)
    for i in range(args.start, stop):
        conversation = task[i]
        prompt = conversation["messages"][0]["content"]
        examples.append({"prompt": prompt, "conversation": conversation, "source_index": i})
    return examples


def load_prompts_from_jsonl(args):
    assert args.input_path is not None, "--input-path is required for --input-source=jsonl"
    input_path = os.path.expanduser(args.input_path)
    examples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < args.start:
                continue
            if len(examples) >= args.max_examples:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, list):
                messages = row
                prompt = next((m["content"] for m in messages if m["role"] == "user"), None)
            else:
                messages = row.get("messages")
                prompt = row.get("prompt")
                if prompt is None and messages is not None:
                    prompt = next((m["content"] for m in messages if m["role"] == "user"), None)
            if prompt is None:
                raise ValueError(f"Could not find prompt in row {idx}")
            examples.append({"prompt": prompt, "messages": messages, "source_index": idx})
    return examples


def load_examples(args):
    if args.input_source == "gsm8k":
        return load_prompts_from_gsm8k(args)
    return load_prompts_from_jsonl(args)


def build_teacher_messages(prompt, args):
    if args.chat_style == "solution":
        user_prompt = (
            "Solve the following problem carefully. Show a concise reasoning process and finish "
            "with a final answer on a new line in the form '#### answer'.\n\n"
            f"{prompt}"
        )
    else:
        user_prompt = prompt
    return [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def load_teacher(args):
    torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = require_transformers()
    torch_dtype = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]
    model_kwargs = {
        "torch_dtype": torch_dtype,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    if args.load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    model = AutoModelForCausalLM.from_pretrained(args.teacher_model, **model_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return torch, tokenizer, model


def generate_text(torch, tokenizer, model, messages, args, seed_offset=0):
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    if args.temperature > 0:
        # Some transformers versions reject the `generator` kwarg in generate(),
        # so set the device-local seed directly before sampling instead.
        if model.device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + seed_offset)
        else:
            torch.manual_seed(args.seed + seed_offset)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0, inputs["input_ids"].size(1):]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text


def synthesize_rejected(answer_text):
    try:
        from tasks.gsm8k import extract_answer
    except ModuleNotFoundError:
        extract_answer = None
    gold = extract_answer(answer_text) if extract_answer is not None else None
    if gold is None:
        truncated = answer_text.splitlines()
        if len(truncated) > 1:
            return truncated[0].strip()
        return "I am not sure."
    try:
        if "." in gold:
            wrong = str(float(gold) + 1.0)
        else:
            wrong = str(int(gold) + 1)
    except ValueError:
        wrong = "0"
    return f"I think the answer is {wrong}.\n#### {wrong}"


def write_sft_line(f, prompt, answer_text):
    row = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer_text},
    ]
    f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_preference_line(f, prompt, chosen, rejected):
    row = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(os.path.expanduser(args.output_path))), exist_ok=True)
    random.seed(args.seed)

    examples = load_examples(args)
    print(f"Loaded {len(examples)} prompt examples from {args.input_source}")
    torch, tokenizer, model = load_teacher(args)
    print(f"Loaded teacher model: {args.teacher_model}")

    output_path = os.path.expanduser(args.output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, example in enumerate(examples):
            prompt = example["prompt"]
            messages = build_teacher_messages(prompt, args)
            chosen = generate_text(torch, tokenizer, model, messages, args, seed_offset=2 * idx)

            if args.output_format == "sft":
                write_sft_line(f, prompt, chosen)
            else:
                if args.rejected_style == "resample":
                    rejected = generate_text(torch, tokenizer, model, messages, args, seed_offset=2 * idx + 1)
                    if rejected.strip() == chosen.strip():
                        rejected = synthesize_rejected(chosen)
                else:
                    rejected = synthesize_rejected(chosen)
                write_preference_line(f, prompt, chosen, rejected)

            if (idx + 1) % 10 == 0 or idx == len(examples) - 1:
                print(f"Wrote {idx + 1}/{len(examples)} examples to {output_path}")

    print(f"Finished writing teacher data to {output_path}")


if __name__ == "__main__":
    main()
