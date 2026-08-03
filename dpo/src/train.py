import json
import os

import torch
from transformers import set_seed, PrinterCallback
from trl import DPOTrainer, DPOConfig

from config import parse_args
from data_utils import load_and_prepare_datasets
from model_utils import (
    build_tokenizer,
    build_model,
    build_lora_config,
    load_tuned_model,
)
from eval_utils import (
    generate_outputs,
    save_comparison_report,
    plot_simple_metrics,
)
from logging_utils import (
    setup_logger,
    log_dataset_summary,
    log_model_summary,
    plot_trainer_logs,
    LiveMetricsCallback,
)


def main():
    args = parse_args()
    logger = setup_logger(args.output_dir)

    logger.info("Starting DPO run")
    logger.info(json.dumps(vars(args), indent=2))

    set_seed(args.seed)

    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    logger.info("Loading tokenizer...")
    tokenizer = build_tokenizer(args.model_name)

    logger.info("Loading and preparing datasets...")
    train_ds, eval_ds = load_and_prepare_datasets(
        dataset_name=args.dataset_name,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        seed=args.seed,
    )
    log_dataset_summary(logger, train_ds, eval_ds)

    eval_prompts = [eval_ds[i]["prompt"] for i in range(len(eval_ds))]
    eval_refs = [eval_ds[i]["chosen"] for i in range(len(eval_ds))]
    eval_rejected = [eval_ds[i]["rejected"] for i in range(len(eval_ds))]

    logger.info("Loading base model for BEFORE generation...")
    base_model = build_model(args.model_name)
    log_model_summary(logger, base_model, args.model_name)

    logger.info("Generating BEFORE outputs...")
    before_outputs = generate_outputs(
        model=base_model,
        tokenizer=tokenizer,
        prompts=eval_prompts,
        max_new_tokens=args.max_new_tokens,
    )

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Loading train model...")
    model = build_model(args.model_name)
    log_model_summary(logger, model, args.model_name)

    peft_config = build_lora_config(args)

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        logging_strategy="steps",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        report_to="tensorboard",
        disable_tqdm=True,
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.add_callback(LiveMetricsCallback(args.output_dir, logger))

    logger.info("Training...")
    train_result = trainer.train()
    logger.info(f"Train result: {train_result}")
    trainer.save_state()
    logger.info("Trainer state saved")

    adapter_dir = os.path.join(args.output_dir, "final_adapter")
    os.makedirs(adapter_dir, exist_ok=True)

    logger.info("Saving LoRA adapter...")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info(f"Adapter dir: {adapter_dir}")
    logger.info(f"Adapter files: {os.listdir(adapter_dir)}")

    logger.info("Saving trainer model snapshot...")
    trainer.save_model(os.path.join(args.output_dir, "trainer_saved_model"))

    plot_trainer_logs(args.output_dir, logger=logger)

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Loading base model + saved adapter for AFTER generation...")
    tuned_model = load_tuned_model(args.model_name, adapter_dir)
    logger.info(f"Tuned model type: {type(tuned_model)}")

    logger.info("Generating AFTER outputs...")
    after_outputs = generate_outputs(
        model=tuned_model,
        tokenizer=tokenizer,
        prompts=eval_prompts,
        max_new_tokens=args.max_new_tokens,
    )

    logger.info("Saving comparison reports...")
    df = save_comparison_report(
        output_dir=args.output_dir,
        prompts=eval_prompts,
        refs=eval_refs,
        rejected_refs=eval_rejected,
        before_outputs=before_outputs,
        after_outputs=after_outputs,
    )

    plot_simple_metrics(df, args.output_dir)

    logger.info(f"Average ROUGE-L before: {df['rougeL_before'].mean():.6f}")
    logger.info(f"Average ROUGE-L after: {df['rougeL_after'].mean():.6f}")
    logger.info(f"Average ROUGE-L delta: {df['rougeL_delta'].mean():.6f}")
    logger.info(f"Average before length: {df['before_len_words'].mean():.2f}")
    logger.info(f"Average after length: {df['after_len_words'].mean():.2f}")

    if len(eval_prompts) > 0:
        logger.info("==== SAMPLE COMPARISON ====")
        logger.info(f"PROMPT:\n{eval_prompts[0][:1000]}")
        logger.info(f"REFERENCE CHOSEN:\n{eval_refs[0][:500]}")
        logger.info(f"BEFORE OUTPUT:\n{before_outputs[0][:500]}")
        logger.info(f"AFTER OUTPUT:\n{after_outputs[0][:500]}")
        logger.info("===========================")

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

    logger.info("Done")
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
