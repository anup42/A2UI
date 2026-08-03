import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="Anthropic/hh-rlhf")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--eval_samples", type=int, default=200)

    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)

    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=768)
    parser.add_argument("--max_new_tokens", type=int, default=192)

    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    return args