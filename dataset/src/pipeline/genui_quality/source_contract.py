"""Candidate-independent source contract extraction, validation, and caching."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping
import unicodedata

from ._core import (
    ActionRef,
    MediaRef,
    SourceContract,
    SourceTable,
    _clean_source_for_content,
    parse_source_contract,
    source_contract_from_mapping,
)


CONTRACT_VERSION = "1.0.0"
EXTRACTOR_VERSION = "1.0.0"
DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3] / "schema" / "expected_ui_contract.schema.json"
)


@dataclass(frozen=True)
class ContractResolution:
    contract: dict[str, Any]
    source: str
    cache_hit: bool
    errors: tuple[str, ...] = ()


def normalize_source_text(text: Any) -> str:
    """Normalize source text only for stable cache identity, never for judging it."""
    value = unicodedata.normalize("NFKC", str(text or "")).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[\t ]+", " ", line).rstrip() for line in value.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def source_contract_cache_key(response_text: Any) -> str:
    normalized = normalize_source_text(response_text)
    payload = f"{EXTRACTOR_VERSION}\0{normalized}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expanded(counter: Counter[str]) -> list[str]:
    return [value for value in sorted(counter) for _ in range(max(0, counter[value]))]


def _semantic_content_units(response_text: str) -> list[str]:
    cleaned = _clean_source_for_content(response_text)
    return [line.strip() for line in cleaned.splitlines() if line.strip()]


def source_contract_to_mapping(
    contract: SourceContract,
    *,
    source_hash: str,
) -> dict[str, Any]:
    """Serialize deterministic extraction to the normative persisted schema."""
    explicit_urls = {action.url for action in contract.explicit_actions}
    return {
        "contract_version": CONTRACT_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "source_hash": source_hash,
        "intent": contract.intent,
        "content_units": _semantic_content_units(contract.raw_text),
        "headings": list(contract.headings),
        "exact_values": _expanded(contract.exact_values),
        "tables": [
            {
                "headers": list(table.headers),
                "rows": [list(row) for row in table.rows],
                "required": True,
            }
            for table in contract.tables
        ],
        "actions": [
            {
                "label": action.label,
                "target": action.url,
                "action_type": action.action_type,
                "required": True,
            }
            for action in contract.explicit_actions
        ],
        "source_links": [
            {"url": url}
            for url in sorted(contract.actionable_urls - explicit_urls)
        ],
        "media": [
            {
                "kind": media.kind,
                "url": media.url,
                **({"alt": media.alt} if media.alt else {}),
                "required": True,
            }
            for media in contract.media
        ],
        "code_blocks": [
            {"language": language, "code": code}
            for language, code in contract.code_blocks
        ],
        "inline_code": list(contract.inline_code),
        "required_roles": dict(contract.required_roles),
        "expected_role_counts": dict(contract.expected_role_counts),
    }


def extract_expected_ui_contract(
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
) -> dict[str, Any]:
    key = source_contract_cache_key(response_text)
    extracted = parse_source_contract(response_text, intent=intent, assets=assets)
    return source_contract_to_mapping(extracted, source_hash=key)


def load_expected_ui_contract_schema(path: Path | None = None) -> dict[str, Any]:
    schema_path = (path or DEFAULT_SCHEMA_PATH).resolve()
    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate_expected_ui_contract(
    contract: Mapping[str, Any],
    *,
    schema_path: Path | None = None,
) -> tuple[bool, list[str]]:
    try:
        import jsonschema  # type: ignore
    except ImportError:
        required_shape = isinstance(contract, Mapping) and any(
            key in contract
            for key in ("content_units", "reference_content", "headings", "tables", "actions", "media", "required_roles")
        )
        return required_shape, ([] if required_shape else ["jsonschema unavailable and contract has no requirement fields"])

    schema = load_expected_ui_contract_schema(schema_path)
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(dict(contract)), key=lambda error: list(error.path))
    messages = [
        f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
        for error in errors
    ]
    return not messages, messages


class SourceContractCache:
    """Small content-addressed cache keyed by normalized source + extractor version."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory.resolve() if directory is not None else None
        self._memory: dict[str, dict[str, Any]] = {}

    def get_or_extract(
        self,
        response_text: str,
        *,
        intent: str | None = None,
        assets: Any = None,
    ) -> tuple[dict[str, Any], bool]:
        key = source_contract_cache_key(response_text)
        cached = self._memory.get(key)
        if cached is not None:
            return deepcopy(cached), True

        cache_path = self.directory / f"{key}.json" if self.directory is not None else None
        if cache_path is not None and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                valid, _ = validate_expected_ui_contract(payload)
                if valid and payload.get("source_hash") == key:
                    self._memory[key] = payload
                    return deepcopy(payload), True
            except Exception:
                pass

        payload = extract_expected_ui_contract(response_text, intent=intent, assets=assets)
        valid, errors = validate_expected_ui_contract(payload)
        if not valid:
            raise ValueError("deterministic expected_ui_contract is invalid: " + "; ".join(errors[:5]))
        self._memory[key] = payload
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(f".tmp.{hashlib.sha256(key.encode()).hexdigest()[:8]}")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            temporary.replace(cache_path)
        return deepcopy(payload), False


def resolve_expected_ui_contract(
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
    persisted: Mapping[str, Any] | None = None,
    persisted_source: str | None = None,
    cache: SourceContractCache | None = None,
) -> ContractResolution:
    """Prefer a valid persisted/human contract; otherwise use deterministic extraction."""
    errors: list[str] = []
    if isinstance(persisted, Mapping):
        candidate = deepcopy(dict(persisted))
        candidate.setdefault("contract_version", CONTRACT_VERSION)
        candidate.setdefault("source_hash", source_contract_cache_key(response_text))
        candidate.setdefault("intent", intent)
        valid, validation_errors = validate_expected_ui_contract(candidate)
        if valid:
            source = "human benchmark" if str(persisted_source or "").replace("_", " ").casefold() == "human benchmark" else "persisted"
            return ContractResolution(candidate, source, False)
        errors.extend(f"persisted contract: {message}" for message in validation_errors)

    resolver = cache or SourceContractCache()
    contract, cache_hit = resolver.get_or_extract(response_text, intent=intent, assets=assets)
    return ContractResolution(contract, "deterministic fallback", cache_hit, tuple(errors))


__all__ = [
    "CONTRACT_VERSION",
    "ContractResolution",
    "DEFAULT_SCHEMA_PATH",
    "EXTRACTOR_VERSION",
    "SourceContract",
    "SourceContractCache",
    "SourceTable",
    "extract_expected_ui_contract",
    "load_expected_ui_contract_schema",
    "normalize_source_text",
    "parse_source_contract",
    "resolve_expected_ui_contract",
    "source_contract_cache_key",
    "source_contract_from_mapping",
    "source_contract_to_mapping",
    "validate_expected_ui_contract",
]
