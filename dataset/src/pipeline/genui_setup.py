"""Validate dashboard preparation options before creating a fresh pilot."""
from pathlib import Path

from pipeline.chatgpt_stage3 import worker_count
from pipeline.genui_codex_client import validate_auth_location
from pipeline.v11_regeneration_io import safe_path

FIELDS = {"input_dir", "output_dir", "model", "reasoning_effort", "reasoning_summary",
          "samples", "seed", "selection_profile", "timeout"}
EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}


def settings(body, current, auth_home=None):
    if set(body) - {"workers"} != FIELDS:
        raise ValueError("Provide the displayed generation settings; unsupported overrides are not allowed")
    for key in FIELDS - {"samples", "seed", "timeout"}:
        if not isinstance(body[key], str) or not body[key].strip():
            raise ValueError(f"{key} must be nonempty text")
    for key, lower, upper in (("samples", 1, 1000), ("seed", 0, 2**32 - 1), ("timeout", 30, 3600)):
        if type(body[key]) is not int or not lower <= body[key] <= upper:
            raise ValueError(f"{key} must be an integer from {lower} to {upper}")
    if body["selection_profile"] not in {"random", "failure_mix_v1"}:
        raise ValueError("Unknown sample selection profile")
    if body["selection_profile"] == "failure_mix_v1" and body["samples"] not in {100, 1000}:
        raise ValueError("failure_mix_v1 requires exactly 100 or 1000 samples")
    if body["reasoning_effort"] not in EFFORTS or body["reasoning_summary"] not in {"none", "auto"}:
        raise ValueError("Unsupported reasoning or summary setting")
    result = dict(body)
    result["workers"] = worker_count(body.get("workers", 2))
    for key in ("input_dir", "output_dir"):
        result[key] = safe_path(Path(body[key]))
    output, source, current = result["output_dir"], result["input_dir"], safe_path(current)
    validate_auth_location(output, auth_home)
    validate_auth_location(source, auth_home)
    if output.exists():
        raise ValueError("Choose a fresh output folder; an existing run cannot be changed")
    for protected in (source, current):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("New output must be separate from the source dataset and current run")
    if not source.is_dir():
        raise ValueError("Training data folder does not exist")
    return result
