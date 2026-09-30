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


CONTRACT_VERSION = "2.0.0"
EXTRACTOR_VERSION = "2.0.1"
EXTRACTION_POLICY_VERSION = "1.0.0"
V5_1_CONTRACT_VERSION = "2.1.0"
V5_1_EXTRACTOR_VERSION = "2.1.0"
V5_1_EXTRACTION_POLICY_VERSION = "1.1.0"
V5_2_CONTRACT_VERSION = "2.2.0"
V5_2_EXTRACTOR_VERSION = "2.2.0"
V5_2_EXTRACTION_POLICY_VERSION = "1.2.0"
V5_2_MIGRATION_POLICY_VERSION = "1.0.0"
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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def source_text_hash(response_text: Any) -> str:
    return hashlib.sha256(normalize_source_text(response_text).encode("utf-8")).hexdigest()


def normalize_intent(intent: Any) -> str:
    return unicodedata.normalize("NFKC", str(intent or "")).strip().casefold()


def relevant_asset_fingerprint(assets: Any) -> str:
    normalized: list[dict[str, str]] = []
    if isinstance(assets, (list, tuple)):
        for item in assets:
            if not isinstance(item, Mapping):
                continue
            normalized.append(
                {
                    "url": str(item.get("url") or "").strip(),
                    "path": str(item.get("path") or "").replace("\\", "/").strip(),
                    "kind": str(item.get("kind") or item.get("type") or "").strip().casefold(),
                }
            )
    return hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()


def source_contract_cache_key(
    response_text: Any,
    *,
    intent: Any = None,
    assets: Any = None,
) -> str:
    payload = {
        "extractor_version": EXTRACTOR_VERSION,
        "extraction_policy_version": EXTRACTION_POLICY_VERSION,
        "source_text": normalize_source_text(response_text),
        "intent": normalize_intent(intent),
        "asset_fingerprint": relevant_asset_fingerprint(assets),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def source_contract_cache_key_v5_1(
    response_text: Any,
    *,
    intent: Any = None,
    assets: Any = None,
) -> str:
    payload = {
        "extractor_version": V5_1_EXTRACTOR_VERSION,
        "extraction_policy_version": V5_1_EXTRACTION_POLICY_VERSION,
        "source_text": normalize_source_text(response_text),
        "intent": normalize_intent(intent),
        "asset_fingerprint": relevant_asset_fingerprint(assets),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def source_contract_cache_key_v5_2(
    response_text: Any,
    *,
    intent: Any = None,
    assets: Any = None,
) -> str:
    payload = {
        "extractor_version": V5_2_EXTRACTOR_VERSION,
        "extraction_policy_version": V5_2_EXTRACTION_POLICY_VERSION,
        "source_text": normalize_source_text(response_text),
        "intent": normalize_intent(intent),
        "asset_fingerprint": relevant_asset_fingerprint(assets),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _expanded(counter: Counter[str]) -> list[str]:
    return [value for value in sorted(counter) for _ in range(max(0, counter[value]))]


def _semantic_content_units(response_text: str) -> list[str]:
    cleaned = _clean_source_for_content(response_text)
    return [line.strip() for line in cleaned.splitlines() if line.strip()]


def source_contract_to_mapping(
    contract: SourceContract,
    *,
    source_hash: str,
    cache_key: str,
) -> dict[str, Any]:
    """Serialize deterministic extraction to the normative persisted schema."""
    explicit_urls = {action.url for action in contract.explicit_actions}
    required_roles = dict(contract.required_roles)
    expected_role_counts = dict(contract.expected_role_counts)
    for role, kind in (
        ("image", "Image"),
        ("video", "Video"),
        ("audio", "AudioPlayer"),
    ):
        relevant = [
            item
            for item in contract.media
            if item.required and item.minimum_count > 0 and item.kind == kind
        ]
        count = (
            1
            if relevant
            and any(item.media_policy.casefold() == "representative" for item in relevant)
            else sum(item.minimum_count for item in relevant)
        )
        required_roles[role] = count > 0
        expected_role_counts[role] = count
    return {
        "contract_version": CONTRACT_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "extraction_policy_version": EXTRACTION_POLICY_VERSION,
        "source_hash": source_hash,
        "cache_key": cache_key,
        "intent": contract.intent,
        "content_units": list(contract.content_units or _semantic_content_units(contract.raw_text)),
        "headings": list(contract.headings),
        "exact_values": _expanded(contract.exact_values),
        "tables": [
            {
                "headers": list(table.headers),
                "rows": [list(row) for row in table.rows],
                "required": table.required,
                "minimum_count": table.minimum_count,
                "order_sensitive": table.order_sensitive,
                "representation_policy": table.representation_policy,
                **({"id": table.id} if table.id else {}),
                **({"row_key": list(table.row_key)} if table.row_key else {}),
            }
            for table in contract.tables
        ],
        "actions": [
            {
                "label": action.label,
                "target": action.url,
                "action_type": action.action_type,
                "required": action.required,
                "minimum_count": action.minimum_count,
                **({"id": action.id} if action.id else {}),
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
                "required": media.required,
                "minimum_count": media.minimum_count,
                "media_policy": media.media_policy,
                **({"id": media.id} if media.id else {}),
            }
            for media in contract.media
        ],
        "code_blocks": [
            {"language": language, "code": code}
            for language, code in contract.code_blocks
        ],
        "inline_code": list(contract.inline_code),
        "required_roles": required_roles,
        "expected_role_counts": expected_role_counts,
        **(
            {"role_requirements": deepcopy(contract.role_requirements)}
            if contract.role_requirements
            else {}
        ),
    }


def extract_expected_ui_contract(
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
) -> dict[str, Any]:
    key = source_contract_cache_key(response_text, intent=intent, assets=assets)
    extracted = parse_source_contract(response_text, intent=intent, assets=assets)
    return source_contract_to_mapping(
        extracted,
        source_hash=source_text_hash(response_text),
        cache_key=key,
    )


def _role_requirements_v5_1(contract: SourceContract) -> dict[str, list[dict[str, Any]]]:
    text = contract.raw_text
    chart_lines = [
        normalize_source_text(value)
        for value in re.findall(
            r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?(?:chart(?:\s+\d+)?(?:\s+title)?)\s*:\s*(.+?)\s*$",
            text,
        )
        if normalize_source_text(value)
    ]
    charts: list[dict[str, Any]] = []
    for index, title in enumerate(chart_lines):
        chart_type_match = re.search(
            r"\b(bar|line|column|scatter|pie|stacked\s+bar|area)\b",
            title,
            re.IGNORECASE,
        )
        charts.append(
            {
                "id": f"chart_{index + 1}",
                "required": True,
                "minimum_count": 1,
                "interchangeable": False,
                "title": title,
                **(
                    {"chart_type": chart_type_match.group(1).casefold()}
                    if chart_type_match
                    else {}
                ),
            }
        )

    formulas = [
        {
            "id": f"formula_{index + 1}",
            "required": True,
            "minimum_count": 1,
            "interchangeable": False,
            "content": normalize_source_text(value),
        }
        for index, value in enumerate(
            re.findall(r"(?im)^\s*(?:formula|equation)\s*:\s*(.+?)\s*$", text)
        )
        if normalize_source_text(value)
    ]
    code = [
        {
            "id": f"code_{index + 1}",
            "required": True,
            "minimum_count": 1,
            "interchangeable": False,
            "language": language,
            "content": body,
        }
        for index, (language, body) in enumerate(contract.code_blocks)
    ]
    emails: list[dict[str, Any]] = []
    subject = re.search(r"(?im)^\s*(?:\*\*)?Subject(?:\*\*)?\s*:\s*(.+?)\s*$", text)
    if subject:
        emails.append(
            {
                "id": "email_1",
                "required": True,
                "minimum_count": 1,
                "interchangeable": False,
                "subject": normalize_source_text(subject.group(1)),
            }
        )
    result = {
        "charts": charts,
        "formulas": formulas,
        "code_blocks": code,
        "emails": emails,
    }
    return {key: value for key, value in result.items() if value}


def extract_expected_ui_contract_v5_1(
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
) -> dict[str, Any]:
    contract = parse_source_contract(response_text, intent=intent, assets=assets)
    key = source_contract_cache_key_v5_1(
        response_text, intent=intent, assets=assets
    )
    mapping = source_contract_to_mapping(
        contract,
        source_hash=source_text_hash(response_text),
        cache_key=key,
    )
    mapping["contract_version"] = V5_1_CONTRACT_VERSION
    mapping["extractor_version"] = V5_1_EXTRACTOR_VERSION
    mapping["extraction_policy_version"] = V5_1_EXTRACTION_POLICY_VERSION
    action_instances = [
        {
            "id": f"action_{index + 1}",
            "label": normalize_source_text(match.group(1)),
            "target": match.group(2).strip(),
            "action_type": "openUrl",
            "required": True,
            "minimum_count": 1,
        }
        for index, match in enumerate(
            re.finditer(
                r"(?im)^\s*Action:\s*\[\s*(?:Button\s*:\s*)?([^\]]+?)\s*\]\s*(https://\S+|\{\{u\d+\}\})\s*$",
                response_text,
            )
        )
    ]
    if action_instances:
        mapping["actions"] = action_instances
        explicit_targets = {
            str(item["target"]).strip() for item in action_instances
        }
        mapping["source_links"] = [
            item
            for item in mapping.get("source_links") or []
            if (
                str(item.get("url") or "").strip()
                if isinstance(item, Mapping)
                else str(item).strip()
            )
            not in explicit_targets
        ]
        mapping["required_roles"]["action"] = True
        mapping["expected_role_counts"]["action"] = len(action_instances)
    role_requirements = _role_requirements_v5_1(contract)
    if role_requirements:
        mapping["role_requirements"] = role_requirements
    return mapping


_ROLE_ALIASES_V5_2 = {
    "chart": "chart",
    "charts": "chart",
    "formula": "formula",
    "formulas": "formula",
    "equation": "formula",
    "equations": "formula",
    "code": "code",
    "codes": "code",
    "code example": "code",
    "code examples": "code",
    "code block": "code",
    "code blocks": "code",
    "codeblock": "code",
    "codeblocks": "code",
    "console log": "console",
    "console logs": "console",
    "consolelog": "console",
    "consolelogs": "console",
    "console output": "console",
    "console outputs": "console",
    "email": "email",
    "emails": "email",
    "email preview": "email",
    "email previews": "email",
    "emailpreview": "email",
    "emailpreviews": "email",
    "form": "form",
    "forms": "form",
    "form control": "form",
    "form controls": "form",
    "control": "form",
    "controls": "form",
}
_CARDINALITY_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_ROLE_TERM_PATTERN = (
    r"charts?|formulas?|equations?|code(?:\s*(?:examples?|blocks?))?|"
    r"console(?:\s*(?:logs?|outputs?))?|email(?:\s*previews?)?|"
    r"forms?|form\s+controls?|controls?"
)


def _canonical_role_name_v5_2(value: Any) -> str:
    normalized = re.sub(
        r"[_-]+", " ", normalize_source_text(value).casefold()
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return _ROLE_ALIASES_V5_2.get(normalized, normalized.rstrip("s"))


def _declared_count_v5_2(value: Any) -> int | None:
    token = normalize_source_text(value).strip().casefold()
    if token.isdigit():
        return max(0, int(token))
    return _CARDINALITY_WORDS.get(token)


def _split_named_role_items_v5_2(
    value: str,
    declared_count: int | None,
) -> list[str]:
    normalized = normalize_source_text(value).strip().strip(".")
    if not normalized:
        return []
    parts = [
        part.strip(" -*\t")
        for part in re.split(r"\s*;\s*|\s+\band\b\s+", normalized)
        if part.strip(" -*\t")
    ]
    if declared_count and len(parts) < declared_count and "," in normalized:
        comma_parts = [
            part.strip(" -*\t")
            for part in normalized.split(",")
            if part.strip(" -*\t")
        ]
        if len(comma_parts) >= len(parts):
            parts = comma_parts
    return parts[:declared_count] if declared_count else parts


def _role_requirement_v5_2(
    role: str,
    text: str,
    index: int,
    *,
    language: str = "",
) -> dict[str, Any]:
    normalized = normalize_source_text(text).strip()
    slug = re.sub(r"[^a-z0-9]+", "_", normalized.casefold()).strip("_")
    payload: dict[str, Any] = {
        "id": f"{role}_{slug[:48] or index + 1}",
        "required": True,
        "minimum_count": 1,
        "interchangeable": False,
    }
    if role in {"chart", "email", "form"}:
        payload["title"] = normalized
    elif role in {"formula", "console"}:
        payload["content"] = normalized
    elif role == "code":
        payload["content"] = normalized
        if language:
            payload["language"] = language.casefold()
    return payload


def _extract_role_requirements_v5_2(
    contract: SourceContract,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, int],
    list[str],
]:
    lines = contract.raw_text.splitlines()
    requirements: dict[str, list[dict[str, Any]]] = {}
    declared_counts: dict[str, int] = {}
    diagnostics: list[str] = []

    def add(role: str, text: str, source: str, *, language: str = "") -> None:
        canonical = _canonical_role_name_v5_2(role)
        normalized = normalize_source_text(text).strip()
        if canonical not in {"chart", "formula", "code", "console", "email", "form"}:
            return
        if not normalized:
            return
        candidate = _role_requirement_v5_2(
            canonical,
            normalized,
            len(requirements.get(canonical, ())),
            language=language,
        )
        semantic = {
            key: value
            for key, value in candidate.items()
            if key not in {"id", "required", "minimum_count", "interchangeable"}
        }
        if any(
            {
                key: value
                for key, value in existing.items()
                if key not in {"id", "required", "minimum_count", "interchangeable"}
            }
            == semantic
            for existing in requirements.get(canonical, ())
        ):
            return
        requirements.setdefault(canonical, []).append(candidate)
        diagnostics.append(f"semantic_role:{canonical}:{source}")

    explicit = re.compile(
        rf"(?i)^\s*(?:[-*]\s*)?(?P<role>{_ROLE_TERM_PATTERN})"
        r"(?:\s+\d+)?(?:\s+title)?\s*:\s*(?P<text>.+?)\s*$"
    )
    for line in lines:
        match = explicit.match(line)
        if match:
            add(
                match.group("role"),
                match.group("text"),
                "explicit_label",
            )

    section = re.compile(
        rf"(?i)^\s*(?:#{{1,6}}\s*)?"
        rf"(?:(?P<count>\d+|{'|'.join(_CARDINALITY_WORDS)})\s+)?"
        rf"(?P<role>{_ROLE_TERM_PATTERN})"
        r"(?:\s+to\s+include)?\s*:?\s*$"
    )
    bullet = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.+?)\s*$")
    for index, line in enumerate(lines):
        match = section.match(line)
        if not match:
            continue
        role = _canonical_role_name_v5_2(match.group("role"))
        count = _declared_count_v5_2(match.group("count"))
        values: list[str] = []
        for following in lines[index + 1 :]:
            if not following.strip():
                if values:
                    break
                continue
            item = bullet.match(following)
            if not item:
                break
            values.append(item.group(1))
            if count is not None and len(values) >= count:
                break
        for value in values:
            add(role, value, "section_list")
        declared_counts[role] = max(
            declared_counts.get(role, 0),
            count if count is not None else len(values),
        )
        if count and len(values) < count:
            diagnostics.append(
                f"count_only_role:{role}:{count - len(values)}"
            )

    inline = re.compile(
        rf"(?i)\b(?:create|include|show|render)\s+"
        rf"(?P<count>\d+|{'|'.join(_CARDINALITY_WORDS)})\s+"
        rf"(?P<role>{_ROLE_TERM_PATTERN})\s*:\s*(?P<items>.+?)(?:\n|$)"
    )
    for match in inline.finditer(contract.raw_text):
        role = _canonical_role_name_v5_2(match.group("role"))
        count = _declared_count_v5_2(match.group("count"))
        values = _split_named_role_items_v5_2(
            match.group("items"), count
        )
        for value in values:
            add(role, value, "inline_cardinality")
        declared_counts[role] = max(
            declared_counts.get(role, 0),
            count or len(values),
        )
        if count and len(values) < count:
            diagnostics.append(
                f"count_only_role:{role}:{count - len(values)}"
            )

    for language, body in contract.code_blocks:
        add("code", body, "fenced_code", language=language)
    for role, values in requirements.items():
        declared_counts[role] = max(
            declared_counts.get(role, 0), len(values)
        )
    return requirements, declared_counts, diagnostics


def _normalize_role_requirements_v5_2(
    value: Any,
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, list[dict[str, Any]]] = {}
    for raw_role, raw_items in value.items():
        role = _canonical_role_name_v5_2(raw_role)
        if not isinstance(raw_items, (list, tuple)):
            continue
        result[role] = [
            {str(key): deepcopy(nested) for key, nested in item.items()}
            for item in raw_items
            if isinstance(item, Mapping)
        ]
    return result


def extract_expected_ui_contract_v5_2(
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
) -> dict[str, Any]:
    contract = parse_source_contract(
        response_text, intent=intent, assets=assets
    )
    mapping = source_contract_to_mapping(
        contract,
        source_hash=source_text_hash(response_text),
        cache_key=source_contract_cache_key_v5_2(
            response_text, intent=intent, assets=assets
        ),
    )
    mapping["contract_version"] = V5_2_CONTRACT_VERSION
    mapping["extractor_version"] = V5_2_EXTRACTOR_VERSION
    mapping["extraction_policy_version"] = V5_2_EXTRACTION_POLICY_VERSION
    roles, declared_counts, diagnostics = _extract_role_requirements_v5_2(
        contract
    )
    mapping["role_requirements"] = roles
    required_roles = dict(mapping.get("required_roles") or {})
    expected_counts = dict(mapping.get("expected_role_counts") or {})
    for role, count in declared_counts.items():
        required_roles[role] = count > 0
        expected_counts[role] = count
    mapping["required_roles"] = required_roles
    mapping["expected_role_counts"] = expected_counts
    total_required = sum(declared_counts.values())
    total_semantic = sum(len(values) for values in roles.values())
    mapping["contract_semantic_specificity"] = (
        total_semantic / total_required if total_required else 1.0
    )
    mapping["extraction_diagnostics"] = diagnostics
    return mapping


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
        key = source_contract_cache_key(response_text, intent=intent, assets=assets)
        cached = self._memory.get(key)
        if cached is not None:
            return deepcopy(cached), True

        cache_path = self.directory / f"{key}.json" if self.directory is not None else None
        if cache_path is not None and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                valid, _ = validate_expected_ui_contract(payload)
                if (
                    valid
                    and payload.get("source_hash") == source_text_hash(response_text)
                    and payload.get("cache_key") == key
                    and payload.get("contract_version") == CONTRACT_VERSION
                    and payload.get("extractor_version") == EXTRACTOR_VERSION
                ):
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


_DEFAULT_SOURCE_CONTRACT_CACHE = SourceContractCache()


class SourceContractCacheV5_1:
    """Content-addressed v5.1 cache with policy-safe identity."""

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
        key = source_contract_cache_key_v5_1(
            response_text, intent=intent, assets=assets
        )
        cached = self._memory.get(key)
        if cached is not None:
            return deepcopy(cached), True
        cache_path = self.directory / f"{key}.json" if self.directory else None
        if cache_path is not None and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                valid, _ = validate_expected_ui_contract(payload)
                if (
                    valid
                    and payload.get("source_hash") == source_text_hash(response_text)
                    and payload.get("cache_key") == key
                    and payload.get("contract_version") == V5_1_CONTRACT_VERSION
                    and payload.get("extractor_version") == V5_1_EXTRACTOR_VERSION
                ):
                    self._memory[key] = payload
                    return deepcopy(payload), True
            except Exception:
                pass
        payload = extract_expected_ui_contract_v5_1(
            response_text, intent=intent, assets=assets
        )
        valid, errors = validate_expected_ui_contract(payload)
        if not valid:
            raise ValueError(
                "deterministic v5.1 expected_ui_contract is invalid: "
                + "; ".join(errors[:5])
            )
        self._memory[key] = payload
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(
                f".tmp.{hashlib.sha256(key.encode()).hexdigest()[:8]}"
            )
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(cache_path)
        return deepcopy(payload), False


_DEFAULT_SOURCE_CONTRACT_CACHE_V5_1 = SourceContractCacheV5_1()


class SourceContractCacheV5_2:
    """Content-addressed v5.2 cache with semantic extraction identity."""

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
        key = source_contract_cache_key_v5_2(
            response_text, intent=intent, assets=assets
        )
        cached = self._memory.get(key)
        if cached is not None:
            return deepcopy(cached), True
        cache_path = self.directory / f"{key}.json" if self.directory else None
        if cache_path is not None and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                valid, _ = validate_expected_ui_contract(payload)
                if (
                    valid
                    and payload.get("source_hash")
                    == source_text_hash(response_text)
                    and payload.get("cache_key") == key
                    and payload.get("contract_version")
                    == V5_2_CONTRACT_VERSION
                    and payload.get("extractor_version")
                    == V5_2_EXTRACTOR_VERSION
                    and payload.get("extraction_policy_version")
                    == V5_2_EXTRACTION_POLICY_VERSION
                ):
                    self._memory[key] = payload
                    return deepcopy(payload), True
            except Exception:
                pass
        payload = extract_expected_ui_contract_v5_2(
            response_text, intent=intent, assets=assets
        )
        valid, errors = validate_expected_ui_contract(payload)
        if not valid:
            raise ValueError(
                "deterministic v5.2 expected_ui_contract is invalid: "
                + "; ".join(errors[:5])
            )
        self._memory[key] = payload
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(
                f".tmp.{hashlib.sha256(key.encode()).hexdigest()[:8]}"
            )
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(cache_path)
        return deepcopy(payload), False


_DEFAULT_SOURCE_CONTRACT_CACHE_V5_2 = SourceContractCacheV5_2()


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
    expected_source_hash = source_text_hash(response_text)
    expected_cache_key = source_contract_cache_key(
        response_text,
        intent=intent,
        assets=assets,
    )
    if isinstance(persisted, Mapping):
        candidate = deepcopy(dict(persisted))
        valid, validation_errors = validate_expected_ui_contract(candidate)
        if candidate.get("source_hash") != expected_source_hash:
            validation_errors.append("source_hash: does not match normalized response text")
        if candidate.get("cache_key") != expected_cache_key:
            validation_errors.append("cache_key: does not match source, intent, assets, and extraction policy")
        if candidate.get("contract_version") != CONTRACT_VERSION:
            validation_errors.append(
                f"contract_version: unsupported {candidate.get('contract_version')!r}; expected {CONTRACT_VERSION}"
            )
        if candidate.get("extractor_version") != EXTRACTOR_VERSION:
            validation_errors.append(
                f"extractor_version: unsupported {candidate.get('extractor_version')!r}; expected {EXTRACTOR_VERSION}"
            )
        if normalize_intent(candidate.get("intent")) != normalize_intent(intent):
            validation_errors.append("intent: does not match the requested extraction intent")
        if valid and not validation_errors:
            source = "human benchmark" if str(persisted_source or "").replace("_", " ").casefold() == "human benchmark" else "persisted"
            return ContractResolution(candidate, source, False)
        errors.extend(f"persisted contract: {message}" for message in validation_errors)

    resolver = cache or _DEFAULT_SOURCE_CONTRACT_CACHE
    contract, cache_hit = resolver.get_or_extract(response_text, intent=intent, assets=assets)
    return ContractResolution(contract, "deterministic fallback", cache_hit, tuple(errors))


def resolve_expected_ui_contract_v5_1(
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
    persisted: Mapping[str, Any] | None = None,
    persisted_source: str | None = None,
    cache: SourceContractCacheV5_1 | None = None,
) -> ContractResolution:
    """Resolve a v5.1 contract, migrating valid v5.0 contracts immutably."""

    errors: list[str] = []
    expected_source_hash = source_text_hash(response_text)
    expected_cache_key = source_contract_cache_key_v5_1(
        response_text, intent=intent, assets=assets
    )
    if isinstance(persisted, Mapping):
        candidate = deepcopy(dict(persisted))
        if (
            candidate.get("source_hash") == expected_source_hash
            and normalize_intent(candidate.get("intent")) == normalize_intent(intent)
            and candidate.get("contract_version") == CONTRACT_VERSION
        ):
            migrated = extract_expected_ui_contract_v5_1(
                response_text, intent=intent, assets=assets
            )
            for key in (
                "content_units",
                "reference_content",
                "headings",
                "exact_values",
                "tables",
                "actions",
                "source_links",
                "media",
                "code_blocks",
                "inline_code",
                "required_roles",
                "expected_role_counts",
                "role_requirements",
            ):
                if key in candidate:
                    migrated[key] = deepcopy(candidate[key])
            migrated["contract_version"] = V5_1_CONTRACT_VERSION
            migrated["extractor_version"] = V5_1_EXTRACTOR_VERSION
            migrated["extraction_policy_version"] = V5_1_EXTRACTION_POLICY_VERSION
            migrated["cache_key"] = expected_cache_key
            valid, migration_errors = validate_expected_ui_contract(migrated)
            if valid:
                return ContractResolution(
                    migrated,
                    "migrated persisted v5.0",
                    False,
                    ("persisted contract migrated from v5.0 identity",),
                )
            errors.extend(
                f"persisted migration: {message}" for message in migration_errors
            )
        else:
            valid, validation_errors = validate_expected_ui_contract(candidate)
            if candidate.get("source_hash") != expected_source_hash:
                validation_errors.append(
                    "source_hash: does not match normalized response text"
                )
            if candidate.get("cache_key") != expected_cache_key:
                validation_errors.append(
                    "cache_key: does not match v5.1 source identity"
                )
            if candidate.get("contract_version") != V5_1_CONTRACT_VERSION:
                validation_errors.append(
                    "contract_version: unsupported for v5.1"
                )
            if candidate.get("extractor_version") != V5_1_EXTRACTOR_VERSION:
                validation_errors.append(
                    "extractor_version: unsupported for v5.1"
                )
            if normalize_intent(candidate.get("intent")) != normalize_intent(intent):
                validation_errors.append(
                    "intent: does not match the requested extraction intent"
                )
            if valid and not validation_errors:
                source = (
                    "human benchmark"
                    if str(persisted_source or "")
                    .replace("_", " ")
                    .casefold()
                    == "human benchmark"
                    else "persisted"
                )
                return ContractResolution(candidate, source, False)
            errors.extend(
                f"persisted contract: {message}" for message in validation_errors
            )

    resolver = cache or _DEFAULT_SOURCE_CONTRACT_CACHE_V5_1
    contract, cache_hit = resolver.get_or_extract(
        response_text, intent=intent, assets=assets
    )
    return ContractResolution(
        contract, "deterministic fallback", cache_hit, tuple(errors)
    )


def resolve_expected_ui_contract_v5_2(
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
    persisted: Mapping[str, Any] | None = None,
    persisted_source: str | None = None,
    cache: SourceContractCacheV5_2 | None = None,
) -> ContractResolution:
    """Resolve v5.2 identity or migrate a source-valid older contract."""

    errors: list[str] = []
    expected_source_hash = source_text_hash(response_text)
    expected_cache_key = source_contract_cache_key_v5_2(
        response_text, intent=intent, assets=assets
    )
    if isinstance(persisted, Mapping):
        candidate = deepcopy(dict(persisted))
        version = str(candidate.get("contract_version") or "")
        source_matches = candidate.get("source_hash") == expected_source_hash
        intent_matches = (
            normalize_intent(candidate.get("intent"))
            == normalize_intent(intent)
        )
        if version == V5_2_CONTRACT_VERSION:
            valid, validation_errors = validate_expected_ui_contract(
                candidate
            )
            if not source_matches:
                validation_errors.append(
                    "source_hash: does not match normalized response text"
                )
            if candidate.get("cache_key") != expected_cache_key:
                validation_errors.append(
                    "cache_key: does not match v5.2 source identity"
                )
            if candidate.get("extractor_version") != V5_2_EXTRACTOR_VERSION:
                validation_errors.append(
                    "extractor_version: unsupported for v5.2"
                )
            if (
                candidate.get("extraction_policy_version")
                != V5_2_EXTRACTION_POLICY_VERSION
            ):
                validation_errors.append(
                    "extraction_policy_version: unsupported for v5.2"
                )
            if not intent_matches:
                validation_errors.append(
                    "intent: does not match the requested extraction intent"
                )
            if valid and not validation_errors:
                source = (
                    "human benchmark"
                    if str(persisted_source or "")
                    .replace("_", " ")
                    .casefold()
                    == "human benchmark"
                    else "persisted"
                )
                return ContractResolution(candidate, source, False)
            errors.extend(
                f"persisted contract: {message}"
                for message in validation_errors
            )
        elif (
            version in {CONTRACT_VERSION, V5_1_CONTRACT_VERSION}
            and source_matches
            and intent_matches
        ):
            migrated = extract_expected_ui_contract_v5_2(
                response_text, intent=intent, assets=assets
            )
            for key in (
                "content_units",
                "reference_content",
                "headings",
                "exact_values",
                "tables",
                "actions",
                "source_links",
                "media",
                "code_blocks",
                "inline_code",
                "required_roles",
                "expected_role_counts",
            ):
                if key in candidate:
                    migrated[key] = deepcopy(candidate[key])
            if "role_requirements" in candidate:
                # An explicit empty human field is authoritative.
                migrated["role_requirements"] = (
                    _normalize_role_requirements_v5_2(
                        candidate.get("role_requirements")
                    )
                )
            migrated["contract_version"] = V5_2_CONTRACT_VERSION
            migrated["extractor_version"] = V5_2_EXTRACTOR_VERSION
            migrated[
                "extraction_policy_version"
            ] = V5_2_EXTRACTION_POLICY_VERSION
            migrated["cache_key"] = expected_cache_key
            migrated["contract_migration"] = {
                "policy_version": V5_2_MIGRATION_POLICY_VERSION,
                "source_contract_version": version,
                "source": (
                    "human"
                    if str(persisted_source or "")
                    .replace("_", " ")
                    .casefold()
                    == "human benchmark"
                    else "persisted"
                ),
                "confidence": 1.0,
                "heuristic_semantics_claimed_as_human": False,
            }
            role_requirements = migrated.get("role_requirements")
            semantic_count = (
                sum(
                    len(values)
                    for values in role_requirements.values()
                    if isinstance(values, list)
                )
                if isinstance(role_requirements, Mapping)
                else 0
            )
            required_count = sum(
                max(0, int(value))
                for value in (
                    migrated.get("expected_role_counts") or {}
                ).values()
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
            )
            migrated["contract_semantic_specificity"] = (
                min(1.0, semantic_count / required_count)
                if required_count
                else 1.0
            )
            valid, migration_errors = validate_expected_ui_contract(
                migrated
            )
            if valid:
                return ContractResolution(
                    migrated,
                    f"migrated persisted {version}",
                    False,
                    (
                        f"persisted contract migrated from {version}",
                    ),
                )
            errors.extend(
                f"persisted migration: {message}"
                for message in migration_errors
            )
        else:
            if not source_matches:
                errors.append(
                    "persisted contract: source_hash does not match "
                    "normalized response text"
                )
            if not intent_matches:
                errors.append(
                    "persisted contract: intent does not match requested "
                    "extraction intent"
                )
            if version not in {
                CONTRACT_VERSION,
                V5_1_CONTRACT_VERSION,
                V5_2_CONTRACT_VERSION,
            }:
                errors.append(
                    f"persisted contract: unsupported version {version!r}"
                )

    resolver = cache or _DEFAULT_SOURCE_CONTRACT_CACHE_V5_2
    contract, cache_hit = resolver.get_or_extract(
        response_text, intent=intent, assets=assets
    )
    return ContractResolution(
        contract, "deterministic fallback", cache_hit, tuple(errors)
    )


__all__ = [
    "CONTRACT_VERSION",
    "ContractResolution",
    "DEFAULT_SCHEMA_PATH",
    "EXTRACTOR_VERSION",
    "EXTRACTION_POLICY_VERSION",
    "V5_1_CONTRACT_VERSION",
    "V5_1_EXTRACTOR_VERSION",
    "V5_1_EXTRACTION_POLICY_VERSION",
    "V5_2_CONTRACT_VERSION",
    "V5_2_EXTRACTOR_VERSION",
    "V5_2_EXTRACTION_POLICY_VERSION",
    "V5_2_MIGRATION_POLICY_VERSION",
    "SourceContract",
    "SourceContractCache",
    "SourceContractCacheV5_1",
    "SourceContractCacheV5_2",
    "SourceTable",
    "extract_expected_ui_contract",
    "extract_expected_ui_contract_v5_1",
    "extract_expected_ui_contract_v5_2",
    "load_expected_ui_contract_schema",
    "normalize_source_text",
    "normalize_intent",
    "parse_source_contract",
    "relevant_asset_fingerprint",
    "resolve_expected_ui_contract",
    "resolve_expected_ui_contract_v5_1",
    "resolve_expected_ui_contract_v5_2",
    "source_contract_cache_key",
    "source_contract_cache_key_v5_1",
    "source_contract_cache_key_v5_2",
    "source_text_hash",
    "source_contract_from_mapping",
    "source_contract_to_mapping",
    "validate_expected_ui_contract",
]
