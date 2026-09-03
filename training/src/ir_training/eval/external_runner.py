from __future__ import annotations

import json
import math
import os
import string
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ir_training.common.jsonl import read_jsonl, write_jsonl
from ir_training.eval.generate import build_prediction_record
from ir_training.eval.golden_set import load_fixed_golden_rows


PROTOCOL_VERSION = "a2ui_external_generation_v1"
RUNNER_PLACEHOLDERS = {
    "model_path",
    "requests_path",
    "outputs_path",
    "output_dir",
    "max_input_tokens",
    "max_new_tokens",
    "mtp_enabled",
}
REQUIRED_RUNNER_PLACEHOLDERS = {"requests_path", "outputs_path"}


def validate_external_runner_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the versioned runner template without starting a process."""

    if not isinstance(config, Mapping):
        raise ValueError("Runner config must be a YAML object.")
    if config.get("protocol") != PROTOCOL_VERSION:
        raise ValueError(f"runner.protocol must be {PROTOCOL_VERSION!r}.")
    command = config.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) and item for item in command)
    ):
        raise ValueError("runner.command must be a non-empty YAML list of strings.")
    templates = list(command)
    working_dir = config.get("working_dir")
    if working_dir is not None:
        if not isinstance(working_dir, str):
            raise ValueError("runner.working_dir must be a string when set.")
        templates.append(working_dir)
    environment = config.get("environment")
    if environment is not None:
        if not isinstance(environment, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in environment.items()
        ):
            raise ValueError("runner.environment must map strings to strings.")
        templates.extend(environment.values())

    placeholders: set[str] = set()
    formatter = string.Formatter()
    for template in templates:
        try:
            fields = [field for _, field, _, _ in formatter.parse(template) if field]
        except ValueError as exc:
            raise ValueError(f"Invalid runner template {template!r}: {exc}") from exc
        unknown = set(fields) - RUNNER_PLACEHOLDERS
        if unknown:
            raise ValueError(
                "Unknown runner placeholder(s): " + ", ".join(sorted(unknown))
            )
        placeholders.update(fields)
    missing = REQUIRED_RUNNER_PLACEHOLDERS - placeholders
    if missing:
        raise ValueError(
            "Runner config must route request and output paths; missing placeholder(s): "
            + ", ".join(sorted(missing))
        )
    placeholder_markers = ("/absolute/path/to/", "<runner", "<absolute-path")
    is_placeholder = any(
        marker in template.lower()
        for template in templates
        for marker in placeholder_markers
    )
    return {
        "protocol": PROTOCOL_VERSION,
        "command_length": len(command),
        "placeholders": sorted(placeholders),
        "placeholder": is_placeholder,
    }


def run_external_generation(
    *,
    command_template: Sequence[str],
    model_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    max_input_tokens: int,
    max_new_tokens: int,
    max_rows: int | None = None,
    required_rows: int | None = None,
    mtp_enabled: bool = False,
    timeout_seconds: int | None = None,
    working_dir: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run a LiteRT-LM-capable process using the JSONL v1 protocol.

    The external command receives paths through placeholders and must write
    exactly one ``{"id": ..., "generated_text": ...}`` row per request.
    The external process never receives the expected Golden completion.
    """

    if not command_template or not all(isinstance(item, str) for item in command_template):
        raise ValueError("runner.command must be a non-empty list of strings.")
    if int(max_input_tokens) < 1 or int(max_new_tokens) < 1:
        raise ValueError("max_input_tokens and max_new_tokens must be positive.")
    model = Path(model_path).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(f"LiteRT-LM package is missing: {model}")
    split = Path(split_path).expanduser().resolve()
    rows = load_fixed_golden_rows(
        split,
        max_rows=max_rows,
        required_rows=required_rows,
        require_exact_rows=required_rows is not None,
        require_unique_rows=True,
    )

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    requests_path = out_dir / "runner_requests.jsonl"
    runner_outputs_path = out_dir / "runner_outputs.jsonl"
    predictions_path = out_dir / "predictions.jsonl"
    request_rows = [
        _request_from_golden_row(
            row,
            model=model,
            max_input_tokens=int(max_input_tokens),
            max_new_tokens=int(max_new_tokens),
            mtp_enabled=bool(mtp_enabled),
        )
        for row in rows
    ]
    write_jsonl(requests_path, request_rows)
    placeholders = {
        "model_path": str(model),
        "requests_path": str(requests_path),
        "outputs_path": str(runner_outputs_path),
        "output_dir": str(out_dir),
        "max_input_tokens": str(int(max_input_tokens)),
        "max_new_tokens": str(int(max_new_tokens)),
        "mtp_enabled": "true" if mtp_enabled else "false",
    }
    command = [_format_token(token, placeholders) for token in command_template]
    run_env = os.environ.copy()
    run_env.update(
        {
            str(key): _format_token(str(value), placeholders)
            for key, value in (environment or {}).items()
        }
    )
    completed = subprocess.run(
        command,
        cwd=str(Path(working_dir).expanduser().resolve()) if working_dir else None,
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    runner_log_path = out_dir / "runner.log"
    runner_log_path.write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"External LiteRT-LM runner exited {completed.returncode}; "
            f"see {runner_log_path}."
        )
    if not runner_outputs_path.is_file():
        raise RuntimeError(
            f"External LiteRT-LM runner did not create {runner_outputs_path}."
        )
    output_rows = list(read_jsonl(runner_outputs_path))
    predictions = merge_external_outputs(rows, output_rows)
    write_jsonl(predictions_path, predictions)
    runtime_metrics = aggregate_external_runtime_metrics(output_rows)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "model_path": str(model),
        "split_path": str(split),
        "row_count": len(predictions),
        "mtp_enabled": bool(mtp_enabled),
        "max_input_tokens": int(max_input_tokens),
        "max_new_tokens": int(max_new_tokens),
        "command": command,
        "requests_path": str(requests_path),
        "runner_outputs_path": str(runner_outputs_path),
        "predictions_path": str(predictions_path),
        "runner_log_path": str(runner_log_path),
        "returncode": completed.returncode,
        "runtime_metrics": runtime_metrics,
    }
    manifest_path = out_dir / "external_runner_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def merge_external_outputs(
    golden_rows: Sequence[dict[str, Any]],
    output_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    _require_unique_ids(golden_rows, source="Golden split")
    _require_unique_ids(output_rows, source="runner output")
    by_id = {str(row.get("id")): row for row in output_rows}
    expected_ids = {str(row.get("id")) for row in golden_rows}
    output_ids = set(by_id)
    missing = sorted(expected_ids - output_ids)
    extra = sorted(output_ids - expected_ids)
    if missing or extra:
        raise ValueError(
            "External runner output IDs do not match requests: "
            f"missing={missing}, extra={extra}."
        )
    predictions: list[dict[str, Any]] = []
    for golden in golden_rows:
        result = by_id[str(golden.get("id"))]
        generated = result.get("generated_text")
        if not isinstance(generated, str):
            raise ValueError(
                f"Runner output {golden.get('id')!r} lacks string generated_text."
            )
        runtime = {
            str(key): value
            for key, value in result.items()
            if key not in {"id", "generated_text"}
        }
        predictions.append(
            build_prediction_record(golden, generated, runtime=runtime or None)
        )
    return predictions


def aggregate_external_runtime_metrics(
    output_rows: Sequence[dict[str, Any]],
) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in output_rows:
        for key, value in row.items():
            if key in {"id", "generated_text"}:
                continue
            if isinstance(value, bool):
                values.setdefault(str(key), []).append(1.0 if value else 0.0)
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.setdefault(str(key), []).append(float(value))
    return {
        f"runtime_{key}_avg": sum(items) / len(items)
        for key, items in sorted(values.items())
        if items
    }


def _request_from_golden_row(
    row: dict[str, Any],
    *,
    model: Path,
    max_input_tokens: int,
    max_new_tokens: int,
    mtp_enabled: bool,
) -> dict[str, Any]:
    messages = [
        dict(message)
        for message in list(row.get("messages") or [])
        if isinstance(message, dict)
    ]
    # A prepared conversation can contain few-shot assistant examples. Keep
    # those demonstrations, but remove the final target assistant completion.
    if messages and messages[-1].get("role") == "assistant":
        messages.pop()
    request = {
        "protocol_version": PROTOCOL_VERSION,
        "id": row.get("id"),
        "response_id": row.get("response_id"),
        "prompt": row.get("prompt"),
        "messages": messages,
        "model_path": str(model),
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "mtp_enabled": mtp_enabled,
    }
    _assert_target_not_exposed(row, request)
    return request


def _assert_target_not_exposed(
    row: Mapping[str, Any], request: Mapping[str, Any]
) -> None:
    target = row.get("completion")
    if not isinstance(target, str) or not target.strip():
        targets = row.get("completion_targets")
        if isinstance(targets, Mapping):
            target = targets.get("a2ui_express_v1")
    if not isinstance(target, str) or not target.strip():
        return
    for field in ("prompt", "messages"):
        # Inspect raw string values instead of JSON-serialized text. JSON
        # escaping changes quotes and backslashes, so a normal A2UI completion
        # embedded in a request could otherwise evade this guard.
        if any(target in value for value in _iter_string_values(request.get(field))):
            raise ValueError(
                "Golden expected completion is present in the external runner "
                f"request field {field!r}; refusing to leak the target."
            )


def _iter_string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        strings: list[str] = []
        for item in value.values():
            strings.extend(_iter_string_values(item))
        return strings
    if isinstance(value, (list, tuple)):
        strings = []
        for item in value:
            strings.extend(_iter_string_values(item))
        return strings
    return []


def _require_unique_ids(rows: Sequence[dict[str, Any]], *, source: str) -> None:
    ids = [str(row.get("id") or "").strip() for row in rows]
    if any(not value for value in ids):
        raise ValueError(f"{source} contains a missing row id.")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{source} contains duplicate row ids.")


def _format_token(token: str, values: Mapping[str, str]) -> str:
    try:
        return token.format_map(values)
    except KeyError as exc:
        raise ValueError(f"Unknown runner command placeholder: {exc.args[0]}") from exc
