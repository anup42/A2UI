"""Versioned production Express scaffold shared by training and inference.

This is prompt preparation, not target regeneration. Source responses and
completed Stage 3 targets remain unchanged by ``normalize_row``.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from ir_training.common.config import repo_root
from ir_training.data.chat_templates import build_messages
from ir_training.data.express_preparation import TASK_PREFIX, prepare_row, serialize_checked


SHARED_PROMPT_VERSION = "a2ui_express_shared_prompt_v1"
PRODUCTION_PROMPT = "dataset/prompts/genui_gen_mobile_a2ui_express_v1.md"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def create_shared_prompt_contract(*, ordering: str = "root-first") -> dict[str, Any]:
    """Freeze the current full production contract and builder's few-shot turns.

    Missing repository prompts fail closed; the packaged short-prompt fallback
    must never accidentally become this explicitly production-bound contract.
    Text hashing normalizes filesystem line endings for Windows/Linux clones.
    """
    if ordering not in {"root-first", "bottom-up"}:
        raise ValueError("ordering must be root-first or bottom-up")
    source = (repo_root() / PRODUCTION_PROMPT).read_text(encoding="utf-8")
    system_prompt = source.replace(
        "{response_text}", "[RESPONSE_TEXT_IS_PROVIDED_IN_THE_USER_MESSAGE]"
    ).strip()
    messages = build_messages(system_prompt, "placeholder", target_format="a2ui_express_v1")[:-1]
    for message in messages:
        if message["role"] == "assistant":
            message["content"] = serialize_checked(message["content"], ordering).text
    scaffold = {"messages": messages, "task_prefix": TASK_PREFIX, "serialization_order": ordering}
    contract = {
        "schema_version": 1, "version": SHARED_PROMPT_VERSION,
        "target_format": "a2ui_express_v1", "serialization_order": ordering,
        "source_path": PRODUCTION_PROMPT, "source_text_sha256": _sha(source),
        "system_prompt_sha256": _sha(system_prompt),
        "scaffold": scaffold, "scaffold_sha256": _sha(_json(scaffold)),
    }
    contract["contract_sha256"] = _sha(_json(contract))
    return contract


def validate_shared_prompt_snapshot(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate saved prompt integrity without rebuilding it from a live checkout.

    Training and evaluation consume the exact prepared messages. Their saved
    contract must not silently change when a repository prompt/builder changes.
    New preparation separately requires parity with current production below.
    """
    if not isinstance(contract, Mapping):
        raise ValueError("shared_prompt must be a versioned contract object")
    def invalid(reason: str) -> None:
        raise ValueError(f"Shared prompt contract is stale or changed: {reason}; restore the original prepared artifacts, do not edit their hashes")

    fields = {"schema_version", "version", "target_format", "serialization_order", "source_path",
              "source_text_sha256", "system_prompt_sha256", "scaffold", "scaffold_sha256", "contract_sha256"}
    if set(contract) != fields or type(contract.get("schema_version")) is not int or contract["schema_version"] != 1:
        invalid("unsupported snapshot schema")
    if contract["version"] != SHARED_PROMPT_VERSION or contract["target_format"] != "a2ui_express_v1":
        invalid("unsupported snapshot version/format")
    if (not isinstance(contract["serialization_order"], str) or contract["serialization_order"] not in {"root-first", "bottom-up"}
            or contract["source_path"] != PRODUCTION_PROMPT):
        invalid("invalid snapshot ordering/source")
    for name in ("source_text_sha256", "system_prompt_sha256", "scaffold_sha256", "contract_sha256"):
        if not isinstance(contract[name], str) or re.fullmatch(r"[0-9a-f]{64}", contract[name]) is None:
            invalid(f"invalid {name}")
    scaffold = contract["scaffold"]
    if not isinstance(scaffold, dict) or set(scaffold) != {"messages", "task_prefix", "serialization_order"}:
        invalid("invalid scaffold structure")
    if scaffold["serialization_order"] != contract["serialization_order"] or not isinstance(scaffold["task_prefix"], str) or not scaffold["task_prefix"].strip():
        invalid("invalid scaffold ordering/task prefix")
    messages = scaffold["messages"]
    if not isinstance(messages, list) or len(messages) < 3 or len(messages) % 2 != 1:
        invalid("invalid few-shot message structure")
    for index, message in enumerate(messages):
        role = "system" if index == 0 else ("user" if index % 2 else "assistant")
        if (not isinstance(message, dict) or set(message) != {"role", "content"} or message["role"] != role
                or not isinstance(message["content"], str) or not message["content"].strip()):
            invalid(f"invalid scaffold message {index}")
    if _sha(messages[0]["content"]) != contract["system_prompt_sha256"]:
        invalid("system prompt checksum mismatch")
    if _sha(_json(scaffold)) != contract["scaffold_sha256"]:
        invalid("scaffold checksum mismatch")
    if _sha(_json({key: value for key, value in contract.items() if key != "contract_sha256"})) != contract["contract_sha256"]:
        invalid("contract checksum mismatch")
    return deepcopy(dict(contract))


def validate_shared_prompt_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """New preparation must match current production as well as its own hashes."""
    saved = validate_shared_prompt_snapshot(contract)
    expected = create_shared_prompt_contract(ordering=saved["serialization_order"])
    if saved != expected:
        changed = ", ".join(key for key in expected if saved.get(key) != expected[key])
        raise ValueError(
            f"Shared prompt contract is stale or changed versus current production ({changed}); "
            f"saved={saved['contract_sha256']} current={expected['contract_sha256']}. "
            "Use a fresh preparation/run directory for the new prompt; do not modify an active job's checkout."
        )
    return expected


def load_shared_prompt_contract(path: str | Path, *, require_current: bool = False) -> dict[str, Any]:
    """Load saved inference evidence; fresh preparation can require current parity."""
    validate = validate_shared_prompt_contract if require_current else validate_shared_prompt_snapshot
    return validate(json.loads(Path(path).read_text(encoding="utf-8")))


def source_prompt_scaffold(row: Mapping[str, Any]) -> dict[str, Any]:
    """Retain old scaffold evidence separately from large per-row prompt text."""
    scaffold = {"messages": deepcopy(row["messages"][:-2]), "task_prefix": TASK_PREFIX}
    return {"sha256": _sha(_json(scaffold)), **scaffold}


def build_inference_messages(response_text: str, contract: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build the exact prepared generation prefix, excluding any target answer."""
    validated = validate_shared_prompt_snapshot(contract)
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("Inference requires a nonempty source response")
    return [*deepcopy(validated["scaffold"]["messages"]),
            {"role": "user", "content": validated["scaffold"]["task_prefix"] + response_text.strip()}]


def normalize_row(row: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    """Standardize the scaffold and task whitespace after validating bindings.

    The later canonical preparation pass may reorder Express assignments, but
    this function does not rewrite the stored response, reference, IDs or
    benchmarks. The task uses the production builder's outer-whitespace trim;
    original prompt, task and scaffold hashes record that input-only change.
    """
    validated = validate_shared_prompt_contract(contract)
    return _normalize_row_validated(row, validated)


def _normalize_row_validated(row: Mapping[str, Any], validated: Mapping[str, Any]) -> dict[str, Any]:
    """Preparation workers validate their immutable scaffold once at startup."""
    prepare_row(row, validated["serialization_order"])
    result = deepcopy(dict(row))
    original = source_prompt_scaffold(row)
    source_task = row["messages"][-2]["content"]
    normalized_task = TASK_PREFIX + source_task[len(TASK_PREFIX):].strip()
    result["messages"] = [*deepcopy(validated["scaffold"]["messages"]),
                          {"role": "user", "content": normalized_task}, deepcopy(row["messages"][-1])]
    result["prompt"] = "\n\n".join(
        f"{message['role'].title()}:\n{message['content']}" for message in result["messages"][:-1]
    ) + "\n\nAssistant:\n"
    evidence = {
        "version": validated["version"], "contract_sha256": validated["contract_sha256"],
        "scaffold_sha256": validated["scaffold_sha256"],
        "source_prompt_sha256": _sha(str(row.get("prompt", ""))),
        "source_task_sha256": _sha(source_task),
        "source_scaffold_sha256": original["sha256"],
    }
    result.setdefault("metadata", {})["shared_prompt"] = evidence
    return result
