import json
import os
from typing import List

import torch
import pandas as pd
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer

from data_utils import parse_hh_text
from tqdm import tqdm

def make_chat_prompt(prompt_text: str, tokenizer) -> str:
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