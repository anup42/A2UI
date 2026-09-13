"""Versioned production Express scaffold shared by training and inference.

This is prompt preparation, not target regeneration. Source responses and
completed Stage 3 targets remain unchanged by ``normalize_row``.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
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


def validate_shared_prompt_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Reject tampered evidence and contracts stale against current production."""
    if not isinstance(contract, Mapping):
        raise ValueError("shared_prompt must be a versioned contract object")
    ordering = contract.get("serialization_order")
    expected = create_shared_prompt_contract(ordering=ordering)
    if dict(contract) != expected:
        raise ValueError("Shared prompt contract is stale or changed; recreate preparation with the current production prompt")
    return expected


def load_shared_prompt_contract(path: str | Path) -> dict[str, Any]:
    return validate_shared_prompt_contract(json.loads(Path(path).read_text(encoding="utf-8")))


def source_prompt_scaffold(row: Mapping[str, Any]) -> dict[str, Any]:
    """Retain old scaffold evidence separately from large per-row prompt text."""
    scaffold = {"messages": deepcopy(row["messages"][:-2]), "task_prefix": TASK_PREFIX}
    return {"sha256": _sha(_json(scaffold)), **scaffold}


def build_inference_messages(response_text: str, contract: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build the exact prepared generation prefix, excluding any target answer."""
    validated = validate_shared_prompt_contract(contract)
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("Inference requires a nonempty source response")
    return [*deepcopy(validated["scaffold"]["messages"]),
            {"role": "user", "content": TASK_PREFIX + response_text.strip()}]


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
