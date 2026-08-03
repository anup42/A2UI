import json
import logging
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

import csv
import time
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter


class LiveMetricsCallback(TrainerCallback):
    def __init__(self, output_dir: str, logger):
        self.output_dir = output_dir
        self.logger = logger
        self.csv_path = os.path.join(output_dir, "live_metrics.csv")
        self.start_time = None
        self.fieldnames_written = False
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.logger.info("LiveMetricsCallback started")
        self.logger.info(f"Logging every {args.logging_steps} steps")
        self.logger.info(f"Eval every {args.eval_steps} steps")
        self.logger.info(f"Saving every {args.save_steps} steps")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        row = {"step": state.global_step}
        row.update(logs)

        # console output
        printable = []
        for k, v in row.items():
            if isinstance(v, float):
                printable.append(f"{k}={v:.6f}")
            else:
                printable.append(f"{k}={v}")
        self.logger.info(" | ".join(printable))

        # Log to TensorBoard
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, state.global_step)

        # append CSV live
        file_exists = os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        # Create intermediate plots every 10 steps
        if state.global_step % 10 == 0:
            self._create_intermediate_plots()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.logger.info(f"[EVAL] step={state.global_step} | metrics={metrics}")

    def on_save(self, args, state, control, **kwargs):
        self.logger.info(f"[CHECKPOINT] step={state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time if self.start_time else 0
        self.logger.info(f"Training finished in {total_time:.2f} seconds")
        self.writer.close()

    def _create_intermediate_plots(self):
        # Create intermediate plots based on the metrics collected so far
        if not os.path.exists(self.csv_path):
            return

        try:
            df = pd.read_csv(self.csv_path)
            if len(df) == 0:
                return

            # Create directory for intermediate plots
            plots_dir = os.path.join(self.output_dir, "intermediate_plots")
            os.makedirs(plots_dir, exist_ok=True)

            # Plot key metrics
            key_metrics = ['loss', 'rewards/accuracies', 'rewards/margins', 'learning_rate']
            available_metrics = [m for m in key_metrics if m in df.columns]

            for metric in available_metrics:
                if 'step' in df.columns and metric in df.columns:
                    # Remove NaN values
                    sub_df = df[df[metric].notna() & df['step'].notna()]
                    if len(sub_df) > 0:
                        plt.figure(figsize=(8, 5))
                        plt.plot(sub_df['step'], sub_df[metric])
                        plt.xlabel('Step')
                        plt.ylabel(metric)
                        plt.title(f'{metric} (Step {sub_df["step"].iloc[-1]})')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()

                        # Save plot
                        filename = metric.replace("/", "_")
                        plt.savefig(os.path.join(plots_dir, f'{filename}_step_{sub_df["step"].iloc[-1]}.png'), dpi=150)
                        plt.close()
        except Exception as e:
            self.logger.warning(f"Failed to create intermediate plots: {e}")

def setup_logger(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger("dpo_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(output_dir, "training.log"), encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log_dataset_summary(logger, train_ds, eval_ds):
    logger.info(f"Train dataset size: {len(train_ds)}")
    logger.info(f"Eval dataset size: {len(eval_ds)}")
    logger.info(f"Train columns: {train_ds.column_names}")
    logger.info(f"Eval columns: {eval_ds.column_names}")

    if len(train_ds) > 0:
        logger.info(f"Prompt preview: {train_ds[0]['prompt'][:300]}")
        logger.info(f"Chosen preview: {train_ds[0]['chosen'][:200]}")
        logger.info(f"Rejected preview: {train_ds[0]['rejected'][:200]}")


def log_model_summary(logger, model, model_name: str):
    logger.info(f"Model source: {model_name}")
    logger.info(f"Model class: {type(model)}")
    logger.info(f"Device map: {getattr(model, 'hf_device_map', None)}")
    try:
        p = next(model.parameters())
        logger.info(f"First param device: {p.device}")
        logger.info(f"First param dtype: {p.dtype}")
    except Exception as e:
        logger.warning(f"Could not inspect model parameters: {e}")


def plot_trainer_logs(output_dir: str, logger=None):
    trainer_state_path = os.path.join(output_dir, "trainer_state.json")

    if not os.path.exists(trainer_state_path):
        checkpoints = [
            os.path.join(output_dir, d)
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]

        if not checkpoints:
            if logger:
                logger.warning("No trainer_state.json found at top level or in checkpoints.")
            return

        checkpoints = sorted(
            checkpoints,
            key=lambda x: int(os.path.basename(x).split("-")[-1])
        )
        trainer_state_path = os.path.join(checkpoints[-1], "trainer_state.json")

        if not os.path.exists(trainer_state_path):
            if logger:
                logger.warning("No trainer_state.json found in latest checkpoint either.")
            return

    if logger:
        logger.info(f"Reading trainer state from: {trainer_state_path}")

    with open(trainer_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    history = state.get("log_history", [])
    if not history:
        if logger:
            logger.warning("trainer_state.json found, but log_history is empty.")
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
                if logger:
                    logger.info(f"Saved {fname}")
            else:
                if logger:
                    logger.warning(f"Column present but empty: {col}")
        else:
            if logger:
                logger.warning(f"Column missing: {col}")
