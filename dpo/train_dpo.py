import os
import re
import json
import argparse
import warnings
from typing import List, Dict, Any, Tuple

import torch
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from rouge_score import rouge_scorer

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

from transformers import AutoModelForCausalLM, Qwen3VLConfig, Qwen3VLForConditionalGeneration

from peft import LoraConfig, PeftModel
from trl import DPOTrainer, DPOConfig
from tqdm import tqdm

# Tensorboard imports
try:
    from transformers import TensorBoardCallback
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    warnings.warn("TensorBoard is not available. Please install it with 'pip install tensorboard' to enable logging.")

# -----------------------------
# Parsing Anthropic HH dataset
# -----------------------------
ROLE_PATTERN = re.compile(r"(?:^|\n\n)(Human|Assistant):\s*")


def parse_hh_text(text: str) -> List[Tuple[str, str]]:
    """
    Parse raw Anthropic HH text:
      Human: ...
      Assistant: ...
      Human: ...
      Assistant: ...
    into a list of (role, content) tuples.
    """
    text = text.strip()
    matches = list(ROLE_PATTERN.finditer(text))
    if not matches:
        raise ValueError(f"Could not parse HH text:\n{text[:300]}")

    turns = []
    for i, match in enumerate(matches):
        role = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        turns.append((role, content))
    return turns


def hh_to_explicit_dpo(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert implicit HH row:
      chosen = shared history + preferred assistant reply
      rejected = shared history + dispreferred assistant reply

    into explicit DPO format:
      prompt, chosen, rejected
    """
    try:
        chosen_turns = parse_hh_text(example["chosen"])
        rejected_turns = parse_hh_text(example["rejected"])

        if len(chosen_turns) < 2 or len(rejected_turns) < 2:
            return {"valid": False, "prompt": "", "chosen": "", "rejected": ""}

        if chosen_turns[-1][0] != "Assistant" or rejected_turns[-1][0] != "Assistant":
            return {"valid": False, "prompt": "", "chosen": "", "rejected": ""}

        chosen_prefix = chosen_turns[:-1]
        rejected_prefix = rejected_turns[:-1]

        # Keep only rows where the shared history matches
        if chosen_prefix != rejected_prefix:
            return {"valid": False, "prompt": "", "chosen": "", "rejected": ""}

        prompt = "\n\n".join([f"{role}: {content}" for role, content in chosen_prefix]).strip()
        chosen_resp = chosen_turns[-1][1].strip()
        rejected_resp = rejected_turns[-1][1].strip()

        if not prompt or not chosen_resp or not rejected_resp:
            return {"valid": False, "prompt": "", "chosen": "", "rejected": ""}

        if chosen_resp == rejected_resp:
            return {"valid": False, "prompt": "", "chosen": "", "rejected": ""}

        return {
            "valid": True,
            "prompt": prompt,
            "chosen": chosen_resp,
            "rejected": rejected_resp,
        }
    except Exception:
        return {"valid": False, "prompt": "", "chosen": "", "rejected": ""}


# -----------------------------
# Model and tokenizer
# -----------------------------
def build_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_quant_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )


def build_model(model_name: str):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=build_quant_config(),
        device_map="cuda",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.use_cache = False
    return model


# -----------------------------
# Prompt formatting for generation
# -----------------------------
def make_chat_prompt(prompt_text: str, tokenizer) -> str:
    """
    Convert explicit HH prompt text into Qwen chat template format.
    Since HH prompt text contains several Human/Assistant turns,
    convert them into chat messages first.
    """
    turns = parse_hh_text(prompt_text)

    messages = []
    for role, content in turns:
        if role == "Human":
            messages.append({"role": "user", "content": content})
        elif role == "Assistant":
            messages.append({"role": "assistant", "content": content})

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return rendered


@torch.inference_mode()
def generate_outputs(model, tokenizer, prompts: List[str], max_new_tokens: int = 192) -> List[str]:
    outputs = []
    for prompt in tqdm(prompts):
        rendered = make_chat_prompt(prompt, tokenizer)
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)

        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        new_tokens = gen[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        outputs.append(text)
    return outputs


# -----------------------------
# Metrics and reports
# -----------------------------
def compute_rouge_l(preds: List[str], refs: List[str]) -> List[float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(preds, refs):
        score = scorer.score(ref, pred)["rougeL"].fmeasure
        scores.append(float(score))
    return scores


def save_comparison_report(
    output_dir: str,
    prompts: List[str],
    refs: List[str],
    rejected_refs: List[str],
    before_outputs: List[str],
    after_outputs: List[str],
):
    rouge_before = compute_rouge_l(before_outputs, refs)
    rouge_after = compute_rouge_l(after_outputs, refs)

    rows = []
    for i in range(len(prompts)):
        rows.append({
            "idx": i,
            "prompt": prompts[i],
            "chosen_reference": refs[i],
            "rejected_reference": rejected_refs[i],
            "before_output": before_outputs[i],
            "after_output": after_outputs[i],
            "rougeL_before": rouge_before[i],
            "rougeL_after": rouge_after[i],
            "rougeL_delta": rouge_after[i] - rouge_before[i],
            "before_len_words": len(before_outputs[i].split()),
            "after_len_words": len(after_outputs[i].split()),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "before_after_comparison.csv"), index=False)

    with open(os.path.join(output_dir, "before_after_comparison.jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(os.path.join(output_dir, "before_after_samples.md"), "w", encoding="utf-8") as f:
        for row in rows[:20]:
            f.write(f"# Sample {row['idx']}\n\n")
            f.write("## Prompt\n")
            f.write(row["prompt"] + "\n\n")
            f.write("## Chosen reference\n")
            f.write(row["chosen_reference"] + "\n\n")
            f.write("## Before output\n")
            f.write(row["before_output"] + "\n\n")
            f.write("## After output\n")
            f.write(row["after_output"] + "\n\n")
            f.write(f"- ROUGE-L before: {row['rougeL_before']:.4f}\n")
            f.write(f"- ROUGE-L after: {row['rougeL_after']:.4f}\n")
            f.write(f"- Delta: {row['rougeL_delta']:.4f}\n\n")
            f.write("---\n\n")

    return df


def plot_simple_metrics(df: pd.DataFrame, output_dir: str):
    plt.figure(figsize=(6, 4))
    plt.bar(["before", "after"], [df["rougeL_before"].mean(), df["rougeL_after"].mean()])
    plt.ylabel("Average ROUGE-L")
    plt.title("Before vs After")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_avg_rougel_before_after.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df["rougeL_delta"], bins=20)
    plt.xlabel("ROUGE-L delta")
    plt.ylabel("Count")
    plt.title("Distribution of improvement")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_rougel_delta_hist.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(
        ["before_len", "after_len"],
        [df["before_len_words"].mean(), df["after_len_words"].mean()],
    )
    plt.ylabel("Avg words")
    plt.title("Output length")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_output_length.png"), dpi=160)
    plt.close()


def plot_trainer_logs(output_dir: str):
    trainer_state_path = os.path.join(output_dir, "trainer_state.json")
    if not os.path.exists(trainer_state_path):
        return

    with open(trainer_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    history = state.get("log_history", [])
    if not history:
        return

    df = pd.DataFrame(history)
    df.to_csv(os.path.join(output_dir, "trainer_log_history.csv"), index=False)

    for col, fname, title in [
        ("loss", "plot_train_loss.png", "Train loss"),
        ("eval_loss", "plot_eval_loss.png", "Eval loss"),
        ("rewards/margins", "plot_reward_margin.png", "Reward margin"),
        ("rewards/accuracies", "plot_reward_accuracy.png", "Reward accuracy"),
        ("logps/chosen", "plot_logps_chosen.png", "Chosen log-prob"),
        ("logps/rejected", "plot_logps_rejected.png", "Rejected log-prob"),
    ]:
        if col in df.columns and "step" in df.columns:
            sub = df[df[col].notna() & df["step"].notna()]
            if len(sub) > 0:
                plt.figure(figsize=(8, 5))
                plt.plot(sub["step"], sub[col])
                plt.xlabel("Step")
                plt.ylabel(col)
                plt.title(title)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, fname), dpi=160)
                plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
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
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # Create tensorboard logging directory
    tensorboard_log_dir = os.path.join(args.output_dir, "tensorboard_logs")
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    set_seed(args.seed)

    print("CUDA device count: ", torch.cuda.device_count())

    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("Loading tokenizer...")
    tokenizer = build_tokenizer(args.model_name)

    print("Loading raw dataset...")
    raw_train = load_dataset(args.dataset_name, split="train")
    raw_test = load_dataset(args.dataset_name, split="test")

    print("Converting dataset to explicit prompt/chosen/rejected...")
    train_ds = raw_train.map(hh_to_explicit_dpo)
    eval_ds = raw_test.map(hh_to_explicit_dpo)

    train_ds = train_ds.filter(lambda x: x["valid"])
    eval_ds = eval_ds.filter(lambda x: x["valid"])

    keep_cols = ["prompt", "chosen", "rejected"]
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
    eval_ds = eval_ds.remove_columns([c for c in eval_ds.column_names if c not in keep_cols])

    train_ds = train_ds.shuffle(seed=args.seed).select(range(min(args.train_samples, len(train_ds))))
    eval_ds = eval_ds.shuffle(seed=args.seed).select(range(min(args.eval_samples, len(eval_ds))))

    print("Dataset columns:", train_ds.column_names)
    print("Example prompt:\n", train_ds[0]["prompt"][:500])
    print("Example chosen:\n", train_ds[0]["chosen"][:200])
    print("Example rejected:\n", train_ds[0]["rejected"][:200])

    assert "prompt" in train_ds.column_names
    assert "chosen" in train_ds.column_names
    assert "rejected" in train_ds.column_names

    eval_prompts = [eval_ds[i]["prompt"] for i in range(len(eval_ds))]
    eval_refs = [eval_ds[i]["chosen"] for i in range(len(eval_ds))]
    eval_rejected = [eval_ds[i]["rejected"] for i in range(len(eval_ds))]

    print("Loading base model for BEFORE generation...")
    base_model = build_model(args.model_name)

    print("Generating BEFORE outputs...")
    before_outputs = generate_outputs(
        model=base_model,
        tokenizer=tokenizer,
        prompts=eval_prompts,
        max_new_tokens=args.max_new_tokens,
    )

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading train model...")
    model = build_model(args.model_name)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        report_to="tensorboard",
        remove_unused_columns=False,
        load_best_model_at_end=False,
        logging_dir=os.path.join(args.output_dir, "tensorboard_logs"),
        # Additional tensorboard logging parameters
        logging_first_step=True,
        logging_nan_inf_filter=True,
    )

    # Add tensorboard callback if available
    if TENSORBOARD_AVAILABLE:
        tensorboard_callback = TensorBoardCallback()
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=[tensorboard_callback],
        )
    else:
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

    print("Training...")
    trainer.train()

    adapter_dir = os.path.join(args.output_dir, "final_adapter")
    os.makedirs(adapter_dir, exist_ok=True)

    print("Saving LoRA adapter...")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print("Saving trainer model snapshot...")
    trainer.save_model(os.path.join(args.output_dir, "trainer_saved_model"))

    plot_trainer_logs(args.output_dir)

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading base model + saved adapter for AFTER generation...")
    base_eval_model = build_model(args.model_name)
    tuned_model = PeftModel.from_pretrained(base_eval_model, adapter_dir)
    tuned_model.eval()

    print("Generating AFTER outputs...")
    after_outputs = generate_outputs(
        model=tuned_model,
        tokenizer=tokenizer,
        prompts=eval_prompts,
        max_new_tokens=args.max_new_tokens,
    )

    print("Saving comparison reports...")
    df = save_comparison_report(
        output_dir=args.output_dir,
        prompts=eval_prompts,
        refs=eval_refs,
        rejected_refs=eval_rejected,
        before_outputs=before_outputs,
        after_outputs=after_outputs,
    )

    plot_simple_metrics(df, args.output_dir)

    summary = {
        "avg_rougeL_before": float(df["rougeL_before"].mean()),
        "avg_rougeL_after": float(df["rougeL_after"].mean()),
        "avg_rougeL_delta": float(df["rougeL_delta"].mean()),
        "avg_before_len_words": float(df["before_len_words"].mean()),
        "avg_after_len_words": float(df["after_len_words"].mean()),
        "adapter_dir": adapter_dir,
    }

    with open(os.path.join(args.output_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
