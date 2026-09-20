from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.cuda_env import normalize_cuda_visible_devices

from ir_training.common.config import load_yaml
from ir_training.common.logging import configure_logging
from ir_training.train.sft import train_sft


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a response-to-IR SFT adapter or full model.")
    parser.add_argument("--config", required=True, help="Path to model training YAML config.")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Load the real model/dataset, run QAT-off versus zero-adapter QAT-on "
            "numeric gates and exit. The separate all-parameter lane additionally performs "
            "one disposable optimizer probe; no checkpoint is saved and its model is discarded."
        ),
    )
    args = parser.parse_args()
    configure_logging()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    _apply_config_cuda_visibility(config)
    cuda_visible_devices = normalize_cuda_visible_devices()
    print(f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}", flush=True)
    _print_launch_rank()
    result = train_sft(
        config,
        config_path=config_path,
        preflight_only=bool(args.preflight_only),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _apply_config_cuda_visibility(config: dict) -> None:
    import os

    if "LOCAL_RANK" in os.environ or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return
    if "CUDA_VISIBLE_DEVICES" in os.environ or os.environ.get("A2UI_CUDA_VISIBLE_DEVICES"):
        return
    runtime_cfg = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    training_cfg = config.get("training") if isinstance(config.get("training"), dict) else {}
    configured = (
        runtime_cfg.get("cuda_visible_devices")
        or model_cfg.get("cuda_visible_devices")
        or training_cfg.get("cuda_visible_devices")
    )
    if configured is not None and str(configured).strip():
        os.environ["A2UI_CUDA_VISIBLE_DEVICES"] = str(configured).strip()


def _print_launch_rank() -> None:
    import os

    print(
        "Launch rank: "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE', '1')}, "
        f"RANK={os.environ.get('RANK', '0')}, "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK', '0')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
