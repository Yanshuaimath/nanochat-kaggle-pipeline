"""
Generate a reproducible nanochat post-training experiment command plan.

This script prints shell commands for a full pipeline, without executing them.
It is intended as a lightweight orchestration helper for the standalone scripts.

Examples:
python3 -m scripts.chat_make_experiment
python3 -m scripts.chat_make_experiment --teacher-model meta-llama/Llama-3.1-8B-Instruct --student-source base --student-tag d12
python3 -m scripts.chat_make_experiment --ppo=1 --ppo-reuse-rm=1
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Print a reproducible post-training experiment plan")
    parser.add_argument("--teacher-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--student-source", type=str, default="base", choices=["base", "sft", "rl"])
    parser.add_argument("--student-tag", type=str, default="d12")
    parser.add_argument("--teacher-max-examples", type=int, default=512)
    parser.add_argument("--distill-epochs", type=int, default=1)
    parser.add_argument("--dpo", type=int, default=1, help="Include DPO stage")
    parser.add_argument("--ppo", type=int, default=0, help="Include PPO stage")
    parser.add_argument("--ppo-reuse-rm", type=int, default=0, help="For PPO, load a previously saved reward-model checkpoint instead of starting RM from scratch")
    parser.add_argument("--ppo-rm-load-path", type=str, default=None, help="Explicit reward-model checkpoint path for PPO; overrides the default generated path")
    parser.add_argument("--universal-rl", type=int, default=0, help="Include universal RL stage")
    parser.add_argument("--rl-objective", type=str, default="rloo_kl", choices=["reinforce", "rloo_kl", "gspo"])
    parser.add_argument("--quantize", type=int, default=1, help="Include int8 quantization stage")
    parser.add_argument("--awq", type=int, default=1, help="Include AWQ int4 stage")
    parser.add_argument("--task-name", type=str, default="GSM8K|MMLU")
    parser.add_argument("--artifact-prefix", type=str, default="exp1")
    return parser.parse_args()


def quote(value):
    if any(ch in value for ch in " |:"):
        return f'"{value}"'
    return value


def main():
    args = parse_args()

    teacher_sft = f"{args.artifact_prefix}_teacher_sft.jsonl"
    teacher_prefs = f"{args.artifact_prefix}_teacher_prefs.jsonl"
    quant_dir = f"{args.student_tag}_int8_linear_{args.artifact_prefix}"
    awq_dir = f"{args.student_tag}_awq_int4_{args.artifact_prefix}"
    default_rm_checkpoint = f"~/.cache/nanochat/chatppo_reward_checkpoints/{args.student_tag}/reward_000200.pt"

    commands = []

    commands.append("# 1. Generate teacher SFT data")
    commands.append(
        "python3 -m scripts.chat_distill_data "
        f"--teacher-model {quote(args.teacher_model)} "
        "--input-source=gsm8k "
        "--output-format=sft "
        f"--max-examples={args.teacher_max_examples} "
        f"--output-path={teacher_sft}"
    )

    commands.append("")
    commands.append("# 2. Generate teacher preference data")
    commands.append(
        "python3 -m scripts.chat_distill_data "
        f"--teacher-model {quote(args.teacher_model)} "
        "--input-source=gsm8k "
        "--output-format=preference "
        f"--max-examples={args.teacher_max_examples} "
        f"--output-path={teacher_prefs}"
    )

    commands.append("")
    commands.append("# 3. Distill the student on teacher SFT data")
    commands.append(
        "python3 -m scripts.chat_distill "
        f"--student-source={args.student_source} "
        f"--student-tag={args.student_tag} "
        f"--data-path={teacher_sft} "
        "--data-format=sft "
        f"--num-epochs={args.distill_epochs}"
    )

    if args.dpo:
        commands.append("")
        commands.append("# 4. Run DPO on teacher preference data")
        commands.append(
            "python3 -m scripts.chat_dpo "
            "--preference-source=jsonl "
            f"--preference-path={teacher_prefs} "
            f"--model-tag={args.student_tag}"
        )

    if args.ppo:
        commands.append("")
        commands.append("# 5. Run PPO + reward model on teacher preference data")
        ppo_cmd = (
            "python3 -m scripts.chat_ppo "
            "--preference-source=jsonl "
            f"--preference-path={teacher_prefs} "
            f"--policy-tag={args.student_tag}"
        )
        if args.ppo_rm_load_path:
            ppo_cmd += f" --rm-load-path={quote(args.ppo_rm_load_path)}"
        elif args.ppo_reuse_rm:
            ppo_cmd += f" --rm-load-path={default_rm_checkpoint}"
        commands.append(ppo_cmd)

    if args.universal_rl:
        commands.append("")
        commands.append("# 6. Run universal RL objective")
        commands.append(
            "python3 -m scripts.chat_universal_rl "
            f"--objective={args.rl_objective} "
            f"--model-tag={args.student_tag}"
        )

    commands.append("")
    commands.append("# 7. Compare post-trained checkpoints")
    compare_cmd = (
        "python3 -m scripts.chat_post_eval "
        f"--models base={args.student_source}:{args.student_tag} "
        f"--models distill=@~/.cache/nanochat/chatdistill_checkpoints:{args.student_tag} "
        f"--task-name={quote(args.task_name)}"
    )
    if args.dpo:
        compare_cmd += f" --models dpo=@~/.cache/nanochat/chatdpo_checkpoints:{args.student_tag}"
    if args.ppo:
        compare_cmd += f" --models ppo=@~/.cache/nanochat/chatppo_checkpoints:{args.student_tag}"
    commands.append(compare_cmd)

    if args.quantize:
        commands.append("")
        commands.append("# 8. Export int8 quantized artifact")
        commands.append(
            "python3 -m scripts.chat_quantize "
            "--checkpoint-dir ~/.cache/nanochat/chatdistill_checkpoints "
            f"--model-tag={args.student_tag} "
            "--method=int8_linear "
            f"--suffix={args.artifact_prefix}"
        )
        commands.append("")
        commands.append("# 9. Evaluate int8 quantized artifact")
        commands.append(
            "python3 -m scripts.chat_quant_eval "
            "--checkpoint-dir ~/.cache/nanochat/chatdistill_checkpoints "
            f"--model-tag={args.student_tag} "
            f"--quant-artifact ~/.cache/nanochat/chatquant_exports/{quant_dir}/quant_000001.pt "
            f"--task-name={quote(args.task_name)}"
        )

    if args.awq:
        commands.append("")
        commands.append("# 10. Export AWQ-style int4 artifact")
        commands.append(
            "python3 -m scripts.chat_quant_awq "
            "--checkpoint-dir ~/.cache/nanochat/chatdistill_checkpoints "
            f"--model-tag={args.student_tag} "
            f"--suffix={args.artifact_prefix}"
        )
        commands.append("")
        commands.append("# 11. Evaluate AWQ-style int4 artifact")
        commands.append(
            "python3 -m scripts.chat_quant_eval "
            "--checkpoint-dir ~/.cache/nanochat/chatdistill_checkpoints "
            f"--model-tag={args.student_tag} "
            f"--quant-artifact ~/.cache/nanochat/chatquant_exports/{awq_dir}/quant_000001.pt "
            f"--task-name={quote(args.task_name)}"
        )

    print("\n".join(commands))


if __name__ == "__main__":
    main()
