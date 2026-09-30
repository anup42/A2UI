"""Candidate-independent expected-contract extraction and resolution for v5.3."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .source_contract import (
    CONTRACT_VERSION,
    ContractResolution,
    V5_1_CONTRACT_VERSION,
    V5_2_CONTRACT_VERSION,
    extract_expected_ui_contract_v5_2,
    normalize_intent,
    normalize_source_text,
    relevant_asset_fingerprint,
    source_text_hash,
    validate_expected_ui_contract,
)


V5_3_CONTRACT_VERSION = "2.3.0"
V5_3_EXTRACTOR_VERSION = "2.3.0"
V5_3_EXTRACTION_POLICY_VERSION = "1.3.0"
V5_3_MIGRATION_POLICY_VERSION = "1.1.0"

_COUNT_WORDS = {
    "a": 1,
    "an": 1,
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
_ROLE_ALIASES = {
    "chart": "chart",
    "graph": "chart",
    "plot": "chart",
    "visualization": "chart",
    "formula": "formula",
    "equation": "formula",
    "code": "code",
    "code example": "code",
    "code block": "code",
    "console output": "console",
    "console log": "console",
    "email": "email",
    "email preview": "email",
    "form": "form",
    "controls": "form",
    "input form": "form",
}
_ROLE_TERM = (
    r"charts?|graphs?|plots?|visualizations?|"
    r"formulas?|equations?|"
    r"code(?:\s+(?:examples?|blocks?))?|"
    r"console(?:\s+(?:outputs?|logs?))?|"
    r"email(?:\s+previews?)?|"
    r"input\s+forms?|forms?|controls?"
)
_VERB = r"create|show|add|include|provide|display|use|visualize|render|draft"
_CHART_TYPE = r"bar|line|column|scatter|pie|stacked\s+bar|area"
_META_FIELDS = {
    "id",
    "required",
    "minimum_count",
    "interchangeable",
    "expected_component_id",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def source_contract_cache_key_v5_3(
    response_text: Any,
    *,
    intent: Any = None,
    assets: Any = None,
) -> str:
    payload = {
        "extractor_version": V5_3_EXTRACTOR_VERSION,
        "extraction_policy_version": V5_3_EXTRACTION_POLICY_VERSION,
        "source_text": normalize_source_text(response_text),
        "intent": normalize_intent(intent),
        "asset_fingerprint": relevant_asset_fingerprint(assets),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_role(value: Any) -> str:
    token = normalize_source_text(value).casefold()
    token = re.sub(r"[_-]+", " ", token)
    token = re.sub(r"\s+", " ", token).strip()
    if token.endswith("s") and token not in {"console outputs"}:
        token = token[:-1]
    return _ROLE_ALIASES.get(token, token)


def _count(value: Any) -> int | None:
    token = str(value or "").strip().casefold()
    if token.isdigit():
        return max(0, int(token))
    return _COUNT_WORDS.get(token)


def _split_payload(value: str, count: int | None) -> list[str]:
    text = normalize_source_text(value).strip().rstrip(".")
    if not text:
        return []
    pieces = [
        item.strip(" -*\t")
        for item in re.split(r"\s*;\s*|\s*,\s*|\s+\band\b\s+", text)
        if item.strip(" -*\t")
    ]
    if count is None or count <= 1:
        return [text]
    return pieces[:count]


def _chart_payload(text: str, chart_type: str = "") -> dict[str, Any]:
    title = normalize_source_text(text).strip().rstrip(".")
    payload: dict[str, Any] = {"title": title}
    detected = chart_type or (
        re.search(rf"(?i)\b({_CHART_TYPE})\s+(?:chart|graph|plot)\b", title)
        or [None, ""]
    )[1]
    if detected:
        payload["chart_type"] = re.sub(r"\s+", "_", str(detected).casefold())
        title = re.sub(
            rf"(?i)\b(?:as\s+)?(?:a\s+|an\s+)?{_CHART_TYPE}\s+"
            r"(?:chart|graph|plot)\b",
            "",
            title,
        ).strip(" :-")
        if title:
            payload["title"] = title
    relation = re.match(r"(?i)^(.+?)\s+by\s+(.+)$", payload["title"])
    if relation:
        payload["y_fields"] = [relation.group(1).strip()]
        payload["x_fields"] = [relation.group(2).strip()]
    payload["data_identity"] = normalize_source_text(payload["title"]).casefold()
    return payload


def _semantic_payload(role: str, text: str, chart_type: str = "") -> dict[str, Any]:
    normalized = normalize_source_text(text).strip().rstrip(".")
    if role == "chart":
        return _chart_payload(normalized, chart_type)
    if role in {"formula", "code", "console"}:
        return {"content": normalized}
    if role == "email":
        payload: dict[str, Any] = {}
        recipient = re.search(r"(?i)\bto\s+([^,.;]+)", normalized)
        if recipient:
            payload["to"] = recipient.group(1).strip()
        elif normalized:
            payload["to"] = normalized
        return payload
    if role == "form":
        return {"title": normalized}
    return {}


def _requirement(
    role: str,
    payload: Mapping[str, Any],
    ordinal: int,
    *,
    source_section: str = "",
) -> dict[str, Any]:
    semantic = {
        str(key): value
        for key, value in payload.items()
        if value not in (None, "", [], {})
    }
    identity = hashlib.sha256(
        _canonical_json({"role": role, "payload": semantic}).encode("utf-8")
    ).hexdigest()[:16]
    result: dict[str, Any] = {
        "id": f"{role}_{identity}_{ordinal + 1}",
        "required": True,
        "minimum_count": 1,
        "interchangeable": False,
        **semantic,
    }
    if source_section:
        result["source_section"] = source_section
    return result


def _extract_roles_v5_3(
    response_text: str,
    code_blocks: Sequence[Mapping[str, Any] | str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int], list[str]]:
    text = normalize_source_text(response_text)
    lines = text.splitlines()
    requirements: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    diagnostics: list[str] = []

    def add(
        role: str,
        payload_text: str,
        source: str,
        *,
        chart_type: str = "",
        source_section: str = "",
    ) -> None:
        canonical = _canonical_role(role)
        payload = _semantic_payload(canonical, payload_text, chart_type)
        if canonical not in {
            "chart",
            "formula",
            "code",
            "console",
            "email",
            "form",
        } or not payload:
            return
        semantic_key = _canonical_json(payload)
        for existing in requirements.get(canonical, ()):
            existing_payload = {
                key: value
                for key, value in existing.items()
                if key not in _META_FIELDS and key != "source_section"
            }
            if _canonical_json(existing_payload) == semantic_key:
                return
        item = _requirement(
            canonical,
            payload,
            len(requirements.get(canonical, ())),
            source_section=source_section,
        )
        requirements.setdefault(canonical, []).append(item)
        diagnostics.append(f"semantic_role:{canonical}:{source}")

    current_heading = ""
    heading_for_line: list[str] = []
    for line in lines:
        heading = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
        if heading:
            current_heading = heading.group(1).strip()
        heading_for_line.append(current_heading)

    # Explicit labels such as "Chart: Revenue by quarter".
    explicit = re.compile(
        rf"(?i)^\s*(?:[-*]\s*)?(?P<role>{_ROLE_TERM})"
        r"(?:\s+\d+)?(?:\s+title)?\s*:\s*(?P<payload>.+?)\s*$"
    )
    for index, line in enumerate(lines):
        match = explicit.match(line)
        if match:
            add(
                match.group("role"),
                match.group("payload"),
                "explicit_label",
                source_section=heading_for_line[index],
            )

    # "## Visualizations" followed by bullets.
    section = re.compile(
        rf"(?i)^\s*(?:#{{1,6}}\s*)?"
        rf"(?:(?P<count>\d+|{'|'.join(_COUNT_WORDS)})\s+)?"
        rf"(?P<role>{_ROLE_TERM})\s*:?\s*$"
    )
    bullet = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.+?)\s*$")
    for index, line in enumerate(lines):
        match = section.match(line)
        if not match:
            continue
        role = _canonical_role(match.group("role"))
        declared = _count(match.group("count"))
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
            if declared is not None and len(values) >= declared:
                break
        for value in values:
            add(role, value, "section_list", source_section=lines[index].strip())
        inferred = declared if declared is not None else len(values)
        counts[role] = max(counts.get(role, 0), inferred)
        if inferred > len(values):
            diagnostics.append(f"count_only_role:{role}:{inferred-len(values)}")

    ordinary = re.compile(
        rf"(?is)\b(?P<verb>{_VERB})\s+"
        r"(?:(?:the)\s+)?"
        rf"(?:(?P<count>\d+|{'|'.join(_COUNT_WORDS)})\s+)?"
        rf"(?:(?P<chart_type>{_CHART_TYPE})\s+)?"
        rf"(?P<role>{_ROLE_TERM})"
        r"(?:\s+(?:showing|of|for|to\s+show|to)\s+"
        r"(?P<payload>[^.!?\n]+))?"
    )
    for match in ordinary.finditer(text):
        role = _canonical_role(match.group("role"))
        declared = _count(match.group("count")) or 1
        payload = normalize_source_text(match.group("payload") or "").strip()
        values = _split_payload(payload, declared) if payload else []
        for value in values:
            add(
                role,
                value,
                "ordinary_instruction",
                chart_type=str(match.group("chart_type") or ""),
            )
        counts[role] = max(counts.get(role, 0), declared, len(values))
        if not values or len(values) < declared:
            diagnostics.append(
                f"count_only_role:{role}:{declared-len(values)}"
            )

    # "Profit by region as a pie chart".
    as_chart = re.compile(
        rf"(?i)(?P<payload>[^.!?\n]+?)\s+as\s+(?:a|an)\s+"
        rf"(?P<chart_type>{_CHART_TYPE})\s+(?:chart|graph|plot)"
    )
    for match in as_chart.finditer(text):
        add(
            "chart",
            match.group("payload"),
            "as_chart_instruction",
            chart_type=match.group("chart_type"),
        )
        counts["chart"] = max(counts.get("chart", 0), 1)

    for raw in code_blocks:
        if isinstance(raw, Mapping):
            body = str(raw.get("code") or "")
            language = str(raw.get("language") or "")
        else:
            body = str(raw)
            language = ""
        if body.strip():
            payload = {"content": body}
            if language:
                payload["language"] = language.casefold()
            item = _requirement(
                "code", payload, len(requirements.get("code", ()))
            )
            requirements.setdefault("code", []).append(item)
            diagnostics.append("semantic_role:code:fenced_code")

    for role, values in requirements.items():
        counts[role] = max(counts.get(role, 0), len(values))
    return requirements, counts, list(dict.fromkeys(diagnostics))


def _semantic_specificity(
    role_requirements: Mapping[str, Sequence[Mapping[str, Any]]],
    expected_counts: Mapping[str, Any],
) -> tuple[float, int]:
    total = 0
    semantic = 0
    roles = {"chart", "formula", "code", "console", "email", "form"}
    for role in roles:
        raw_count = expected_counts.get(role, 0)
        try:
            declared = max(0, int(raw_count))
        except (TypeError, ValueError):
            declared = 0
        required_items = [
            item
            for item in role_requirements.get(role, ())
            if bool(item.get("required", True))
            and int(item.get("minimum_count", 1) or 0) > 0
        ]
        required_from_items = sum(
            max(1, int(item.get("minimum_count", 1) or 1))
            for item in required_items
        )
        role_total = max(declared, required_from_items)
        role_semantic = sum(
            max(1, int(item.get("minimum_count", 1) or 1))
            for item in required_items
            if any(
                value not in (None, "", [], {})
                for key, value in item.items()
                if key not in _META_FIELDS
            )
        )
        total += role_total
        semantic += min(role_total, role_semantic)
    count_only = max(0, total - semantic)
    return (
        (semantic / total if total else 1.0),
        count_only,
    )


def extract_expected_ui_contract_v5_3(
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
) -> dict[str, Any]:
    mapping = extract_expected_ui_contract_v5_2(
        response_text, intent=intent, assets=assets
    )
    mapping["contract_version"] = V5_3_CONTRACT_VERSION
    mapping["extractor_version"] = V5_3_EXTRACTOR_VERSION
    mapping["extraction_policy_version"] = V5_3_EXTRACTION_POLICY_VERSION
    mapping["cache_key"] = source_contract_cache_key_v5_3(
        response_text, intent=intent, assets=assets
    )
    roles, declared_counts, diagnostics = _extract_roles_v5_3(
        response_text, mapping.get("code_blocks") or []
    )
    mapping["role_requirements"] = roles
    required_roles = dict(mapping.get("required_roles") or {})
    expected_counts = dict(mapping.get("expected_role_counts") or {})
    for role, count in declared_counts.items():
        required_roles[role] = count > 0
        expected_counts[role] = max(
            count, int(expected_counts.get(role, 0) or 0)
        )
    mapping["required_roles"] = required_roles
    mapping["expected_role_counts"] = expected_counts

    chart_directive = bool(
        re.search(
            r"(?im)^\s*(?:#{1,6}\s*)?(?:chart|graph|plot)\s*:",
            response_text,
        )
    ) or any(
        item.startswith("semantic_role:chart:")
        for item in diagnostics
    )
    if chart_directive:
        tables = mapping.get("tables")
        if isinstance(tables, list):
            for table in tables:
                if isinstance(table, dict):
                    table["representation_policy"] = "structured_equivalent"

    specificity, count_only = _semantic_specificity(roles, expected_counts)
    mapping["contract_semantic_specificity"] = specificity
    mapping["count_only_role_count"] = count_only
    mapping["extraction_diagnostics"] = list(
        dict.fromkeys(
            [*(mapping.get("extraction_diagnostics") or []), *diagnostics]
        )
    )
    return mapping


@dataclass
class SourceContractCacheV5_3:
    directory: Path | None = None

    def __post_init__(self) -> None:
        if self.directory is not None:
            self.directory = Path(self.directory).resolve()
        self._memory: dict[str, dict[str, Any]] = {}

    def get_or_extract(
        self,
        response_text: str,
        *,
        intent: str | None = None,
        assets: Any = None,
    ) -> tuple[dict[str, Any], bool]:
        key = source_contract_cache_key_v5_3(
            response_text, intent=intent, assets=assets
        )
        if key in self._memory:
            return deepcopy(self._memory[key]), True
        cache_path = (
            self.directory / f"{key}.json"
            if self.directory is not None
            else None
        )
        if cache_path is not None and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                valid, _ = validate_expected_ui_contract(cached)
                if (
                    valid
                    and cached.get("source_hash")
                    == source_text_hash(response_text)
                    and cached.get("cache_key") == key
                    and cached.get("contract_version")
                    == V5_3_CONTRACT_VERSION
                    and cached.get("extractor_version")
                    == V5_3_EXTRACTOR_VERSION
                    and cached.get("extraction_policy_version")
                    == V5_3_EXTRACTION_POLICY_VERSION
                    and normalize_intent(cached.get("intent"))
                    == normalize_intent(intent)
                ):
                    self._memory[key] = deepcopy(cached)
                    return deepcopy(cached), True
            except Exception:
                pass
        payload = extract_expected_ui_contract_v5_3(
            response_text, intent=intent, assets=assets
        )
        valid, errors = validate_expected_ui_contract(payload)
        if not valid:
            raise ValueError(
                "deterministic v5.3 expected_ui_contract is invalid: "
                + "; ".join(errors[:5])
            )
        self._memory[key] = deepcopy(payload)
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


_DEFAULT_CACHE = SourceContractCacheV5_3()


def resolve_expected_ui_contract_v5_3(
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
    persisted: Mapping[str, Any] | None = None,
    persisted_source: str | None = None,
    cache: SourceContractCacheV5_3 | None = None,
) -> ContractResolution:
    expected_hash = source_text_hash(response_text)
    expected_key = source_contract_cache_key_v5_3(
        response_text, intent=intent, assets=assets
    )
    errors: list[str] = []
    if isinstance(persisted, Mapping):
        candidate = deepcopy(dict(persisted))
        version = str(candidate.get("contract_version") or "")
        source_matches = candidate.get("source_hash") == expected_hash
        intent_matches = (
            normalize_intent(candidate.get("intent"))
            == normalize_intent(intent)
        )
        if version == V5_3_CONTRACT_VERSION:
            valid, validation_errors = validate_expected_ui_contract(candidate)
            if not source_matches:
                validation_errors.append("source_hash mismatch")
            if not intent_matches:
                validation_errors.append("intent mismatch")
            if candidate.get("cache_key") != expected_key:
                validation_errors.append("cache_key mismatch")
            if candidate.get("extractor_version") != V5_3_EXTRACTOR_VERSION:
                validation_errors.append("extractor_version mismatch")
            if (
                candidate.get("extraction_policy_version")
                != V5_3_EXTRACTION_POLICY_VERSION
            ):
                validation_errors.append("extraction_policy_version mismatch")
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
            version
            in {
                CONTRACT_VERSION,
                V5_1_CONTRACT_VERSION,
                V5_2_CONTRACT_VERSION,
            }
            and source_matches
            and intent_matches
        ):
            migrated = extract_expected_ui_contract_v5_3(
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
            specificity, count_only = _semantic_specificity(
                migrated.get("role_requirements") or {},
                migrated.get("expected_role_counts") or {},
            )
            migrated["contract_semantic_specificity"] = specificity
            migrated["count_only_role_count"] = count_only
            migrated["contract_migration"] = {
                "policy_version": V5_3_MIGRATION_POLICY_VERSION,
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
            valid, migration_errors = validate_expected_ui_contract(migrated)
            if valid:
                return ContractResolution(
                    migrated,
                    f"migrated persisted {version}",
                    False,
                    (f"persisted contract migrated from {version}",),
                )
            errors.extend(
                f"persisted migration: {message}"
                for message in migration_errors
            )
        else:
            if not source_matches:
                errors.append("persisted contract: source_hash mismatch")
            if not intent_matches:
                errors.append("persisted contract: intent mismatch")
            if version not in {
                CONTRACT_VERSION,
                V5_1_CONTRACT_VERSION,
                V5_2_CONTRACT_VERSION,
                V5_3_CONTRACT_VERSION,
            }:
                errors.append(
                    f"persisted contract: unsupported version {version!r}"
                )

    resolver = cache or _DEFAULT_CACHE
    contract, cache_hit = resolver.get_or_extract(
        response_text, intent=intent, assets=assets
    )
    return ContractResolution(
        contract, "deterministic fallback", cache_hit, tuple(errors)
    )


__all__ = [
    "SourceContractCacheV5_3",
    "V5_3_CONTRACT_VERSION",
    "V5_3_EXTRACTOR_VERSION",
    "V5_3_EXTRACTION_POLICY_VERSION",
    "V5_3_MIGRATION_POLICY_VERSION",
    "extract_expected_ui_contract_v5_3",
    "resolve_expected_ui_contract_v5_3",
    "source_contract_cache_key_v5_3",
]
