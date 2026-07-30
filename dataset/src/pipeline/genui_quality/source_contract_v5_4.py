"""Candidate-independent expected-contract extraction for metric v5.4."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .ownership_v5_4 import generic_source_texts
from .source_contract import (
    CONTRACT_VERSION,
    ContractResolution,
    V5_1_CONTRACT_VERSION,
    V5_2_CONTRACT_VERSION,
    normalize_intent,
    normalize_source_text,
    relevant_asset_fingerprint,
    source_text_hash,
    validate_expected_ui_contract,
)
from .source_contract_v5_3 import (
    V5_3_CONTRACT_VERSION,
    extract_expected_ui_contract_v5_3,
)


V5_4_CONTRACT_VERSION = "2.4.0"
V5_4_EXTRACTOR_VERSION = "2.4.0"
V5_4_EXTRACTION_POLICY_VERSION = "1.4.0"
V5_4_MIGRATION_POLICY_VERSION = "1.2.0"
SOURCE_EXTRACTION_BENCHMARK_VERSION = "1.0.0"

_CHART_TYPE = (
    r"stacked\s+bar|horizontal\s+bar|vertical\s+bar|"
    r"bar|line|column|scatter|pie|area|donut|doughnut"
)
_COUNT = {
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
_META = {
    "id",
    "required",
    "minimum_count",
    "interchangeable",
    "expected_component_id",
    "source_section",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def source_contract_cache_key_v5_4(
    response_text: Any,
    *,
    intent: Any = None,
    assets: Any = None,
) -> str:
    payload = {
        "extractor_version": V5_4_EXTRACTOR_VERSION,
        "extraction_policy_version": V5_4_EXTRACTION_POLICY_VERSION,
        "source_text": normalize_source_text(response_text),
        "intent": normalize_intent(intent),
        "asset_fingerprint": relevant_asset_fingerprint(assets),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _chart_payload(title: str, chart_type: str = "") -> dict[str, Any]:
    cleaned = normalize_source_text(title).strip().strip(" :-")
    payload: dict[str, Any] = {"title": cleaned}
    if chart_type:
        payload["chart_type"] = re.sub(
            r"\s+", "_", chart_type.strip().casefold()
        )
    relation = re.match(r"(?i)^(.+?)\s+by\s+(.+)$", cleaned)
    if relation:
        payload["y_fields"] = [relation.group(1).strip()]
        payload["x_fields"] = [relation.group(2).strip()]
    return payload


def _requirement(
    role: str,
    payload: Mapping[str, Any],
    *,
    start: int,
    ordinal: int,
) -> dict[str, Any]:
    semantic = {
        str(key): value
        for key, value in payload.items()
        if value not in (None, "", [], {})
    }
    digest = hashlib.sha256(
        _canonical_json(
            {
                "role": role,
                "payload": semantic,
                "source_start": start,
                "ordinal": ordinal,
            }
        ).encode("utf-8")
    ).hexdigest()[:18]
    return {
        "id": f"{role}_{digest}",
        "required": True,
        "minimum_count": 1,
        "interchangeable": False,
        **semantic,
    }


def _payload_signature(value: Mapping[str, Any]) -> str:
    return _canonical_json(
        {
            key: nested
            for key, nested in value.items()
            if key not in _META and key != "data_identity"
            and nested not in (None, "", [], {})
        }
    )


def _add_requirement(
    roles: dict[str, list[dict[str, Any]]],
    role: str,
    payload: Mapping[str, Any],
    *,
    start: int,
) -> bool:
    clean = {
        key: value
        for key, value in payload.items()
        if key != "data_identity" and value not in (None, "", [], {})
    }
    if not clean:
        return False
    signature = _payload_signature(clean)
    if any(_payload_signature(item) == signature for item in roles.get(role, ())):
        return False
    roles.setdefault(role, []).append(
        _requirement(
            role,
            clean,
            start=start,
            ordinal=len(roles.get(role, ())),
        )
    )
    return True


def _split_semantic_list(value: str) -> list[str]:
    return [
        part.strip(" -*\t.")
        for part in re.split(r"\s*;\s*|\s*,\s*|\s+\band\b\s+", value)
        if part.strip(" -*\t.")
    ]


def _strengthen_roles(
    response_text: str,
    mapping: dict[str, Any],
) -> None:
    text = normalize_source_text(response_text)
    roles = {
        str(role): [
            {
                key: value
                for key, value in dict(item).items()
                if key != "data_identity"
            }
            for item in values
            if isinstance(item, Mapping)
        ]
        for role, values in (mapping.get("role_requirements") or {}).items()
        if isinstance(values, list)
    }
    counts = {
        str(role): max(0, int(value))
        for role, value in (mapping.get("expected_role_counts") or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    diagnostics = list(mapping.get("extraction_diagnostics") or [])

    parenthetical = re.compile(
        rf"(?im)^\s*(?:#{{1,6}}\s*)?"
        rf"(?P<title>.+?)\s*\("
        rf"(?:as\s+(?:(?:a|an)\s+)?)?"
        rf"(?P<type>{_CHART_TYPE})?\s*chart\)\s*$"
    )
    for match in parenthetical.finditer(text):
        if _add_requirement(
            roles,
            "chart",
            _chart_payload(match.group("title"), match.group("type")),
            start=match.start(),
        ):
            diagnostics.append("semantic_role:chart:parenthetical_heading")

    suffix = re.compile(
        rf"(?im)^\s*(?:#{{1,6}}\s*)?"
        rf"(?P<title>.+?)\s*(?:[-:]\s*|\s+as\s+(?:(?:a|an)\s+)?)"
        rf"(?P<type>{_CHART_TYPE})?\s*chart\s*$"
    )
    for match in suffix.finditer(text):
        if _add_requirement(
            roles,
            "chart",
            _chart_payload(match.group("title"), match.group("type")),
            start=match.start(),
        ):
            diagnostics.append("semantic_role:chart:suffix_heading")

    colon_instruction = re.compile(
        rf"(?i)\b(?:create|show|include|provide|display|use)\s+"
        rf"(?P<count>\d+|{'|'.join(_COUNT)})\s+charts?\s*:\s*"
        r"(?P<payload>[^.!?\n]+)"
    )
    for match in colon_instruction.finditer(text):
        token = match.group("count").casefold()
        declared = int(token) if token.isdigit() else _COUNT.get(token, 1)
        values = _split_semantic_list(match.group("payload"))
        for value in values[:declared]:
            if _add_requirement(
                roles,
                "chart",
                _chart_payload(value),
                start=match.start(),
            ):
                diagnostics.append("semantic_role:chart:colon_instruction")
        counts["chart"] = max(counts.get("chart", 0), declared)

    section = re.compile(
        r"(?im)^\s*(?:#{1,6}\s*)?"
        r"(?:charts?|graphs?|plots?|visualizations?)\s+to\s+include\s*:?\s*$"
    )
    bullet = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.+?)\s*$")
    lines = text.splitlines()
    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line) + 1
    for index, line in enumerate(lines):
        if not section.match(line):
            continue
        found = 0
        for following_index in range(index + 1, len(lines)):
            raw = lines[following_index]
            if not raw.strip():
                if found:
                    break
                continue
            item = bullet.match(raw)
            if item is None:
                break
            value = item.group(1).strip()
            type_match = re.search(
                rf"(?i)\((?P<type>{_CHART_TYPE})\s+chart\)\s*$",
                value,
            )
            chart_type = type_match.group("type") if type_match else ""
            title = value[: type_match.start()].strip() if type_match else value
            if _add_requirement(
                roles,
                "chart",
                _chart_payload(title, chart_type),
                start=offsets[following_index],
            ):
                diagnostics.append("semantic_role:chart:include_section")
            found += 1
        counts["chart"] = max(counts.get("chart", 0), found)

    equivalents = {
        "formula": re.compile(
            r"(?im)^\s*(?:#{1,6}\s*)?(?:formula|equation)\s*:\s*(.+?)\s*$"
        ),
        "console": re.compile(
            r"(?im)^\s*(?:#{1,6}\s*)?(?:console\s*(?:log|output))\s*:\s*(.+?)\s*$"
        ),
        "email": re.compile(
            r"(?im)^\s*(?:#{1,6}\s*)?(?:email\s*preview|email)\s*:\s*(.+?)\s*$"
        ),
        "form": re.compile(
            r"(?im)^\s*(?:#{1,6}\s*)?(?:input\s*form|form|controls?)\s*:\s*(.+?)\s*$"
        ),
        "code": re.compile(
            r"(?im)^\s*(?:#{1,6}\s*)?(?:code\s*block|codeblock)\s*:\s*(.+?)\s*$"
        ),
    }
    for role, pattern in equivalents.items():
        for match in pattern.finditer(text):
            payload = (
                {"title": match.group(1).strip()}
                if role in {"form"}
                else {"to": match.group(1).strip()}
                if role == "email"
                else {"content": match.group(1).strip()}
            )
            if _add_requirement(
                roles, role, payload, start=match.start()
            ):
                diagnostics.append(f"semantic_role:{role}:equivalent")
        counts[role] = max(counts.get(role, 0), len(roles.get(role, ())))

    mapping["role_requirements"] = roles
    mapping["expected_role_counts"] = counts
    required = dict(mapping.get("required_roles") or {})
    for role, count in counts.items():
        required[role] = count > 0
    mapping["required_roles"] = required
    semantic_total = sum(
        max(1, int(item.get("minimum_count", 1) or 1))
        for values in roles.values()
        for item in values
        if bool(item.get("required", True))
        and int(item.get("minimum_count", 1) or 0) > 0
        and _payload_signature(item) != "{}"
    )
    applicable_total = sum(
        max(
            counts.get(role, 0),
            sum(
                max(1, int(item.get("minimum_count", 1) or 1))
                for item in roles.get(role, ())
                if bool(item.get("required", True))
                and int(item.get("minimum_count", 1) or 0) > 0
            ),
        )
        for role in {"chart", "formula", "code", "console", "email", "form"}
    )
    mapping["contract_semantic_specificity"] = (
        min(1.0, semantic_total / applicable_total)
        if applicable_total
        else 1.0
    )
    mapping["count_only_role_count"] = max(
        0, applicable_total - semantic_total
    )
    mapping["extraction_diagnostics"] = list(dict.fromkeys(diagnostics))


def extract_expected_ui_contract_v5_4(
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
) -> dict[str, Any]:
    mapping = extract_expected_ui_contract_v5_3(
        response_text, intent=intent, assets=assets
    )
    mapping["contract_version"] = V5_4_CONTRACT_VERSION
    mapping["extractor_version"] = V5_4_EXTRACTOR_VERSION
    mapping["extraction_policy_version"] = V5_4_EXTRACTION_POLICY_VERSION
    mapping["cache_key"] = source_contract_cache_key_v5_4(
        response_text, intent=intent, assets=assets
    )
    _strengthen_roles(response_text, mapping)
    mapping["content_units"] = list(
        generic_source_texts(response_text, mapping)
    )
    return mapping


@dataclass
class SourceContractCacheV5_4:
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
        key = source_contract_cache_key_v5_4(
            response_text, intent=intent, assets=assets
        )
        if key in self._memory:
            return deepcopy(self._memory[key]), True
        path = self.directory / f"{key}.json" if self.directory else None
        if path is not None and path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                valid, errors = validate_expected_ui_contract(cached)
                if (
                    valid
                    and not errors
                    and cached.get("source_hash")
                    == source_text_hash(response_text)
                    and cached.get("cache_key") == key
                    and cached.get("contract_version")
                    == V5_4_CONTRACT_VERSION
                    and cached.get("extractor_version")
                    == V5_4_EXTRACTOR_VERSION
                    and cached.get("extraction_policy_version")
                    == V5_4_EXTRACTION_POLICY_VERSION
                    and normalize_intent(cached.get("intent"))
                    == normalize_intent(intent)
                ):
                    self._memory[key] = deepcopy(cached)
                    return deepcopy(cached), True
            except Exception:
                pass
        payload = extract_expected_ui_contract_v5_4(
            response_text, intent=intent, assets=assets
        )
        valid, errors = validate_expected_ui_contract(payload)
        if not valid:
            raise ValueError(
                "deterministic v5.4 expected_ui_contract is invalid: "
                + "; ".join(errors[:5])
            )
        self._memory[key] = deepcopy(payload)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(path)
        return deepcopy(payload), False


_DEFAULT_CACHE = SourceContractCacheV5_4()


def resolve_expected_ui_contract_v5_4(
    response_text: str,
    *,
    intent: str | None = None,
    assets: Any = None,
    persisted: Mapping[str, Any] | None = None,
    persisted_source: str | None = None,
    cache: SourceContractCacheV5_4 | None = None,
) -> ContractResolution:
    expected_hash = source_text_hash(response_text)
    expected_key = source_contract_cache_key_v5_4(
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
        if version == V5_4_CONTRACT_VERSION:
            valid, validation_errors = validate_expected_ui_contract(candidate)
            if not source_matches:
                validation_errors.append("source_hash mismatch")
            if not intent_matches:
                validation_errors.append("intent mismatch")
            if candidate.get("cache_key") != expected_key:
                validation_errors.append("cache_key mismatch")
            if candidate.get("extractor_version") != V5_4_EXTRACTOR_VERSION:
                validation_errors.append("extractor_version mismatch")
            if (
                candidate.get("extraction_policy_version")
                != V5_4_EXTRACTION_POLICY_VERSION
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
                V5_3_CONTRACT_VERSION,
            }
            and source_matches
            and intent_matches
        ):
            migrated = extract_expected_ui_contract_v5_4(
                response_text, intent=intent, assets=assets
            )
            for key in (
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
            migrated["contract_migration"] = {
                "policy_version": V5_4_MIGRATION_POLICY_VERSION,
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
            errors.append(
                f"persisted contract: unsupported version {version!r}"
            )

    contract, hit = (cache or _DEFAULT_CACHE).get_or_extract(
        response_text, intent=intent, assets=assets
    )
    return ContractResolution(
        contract, "deterministic fallback", hit, tuple(errors)
    )


__all__ = [
    "SOURCE_EXTRACTION_BENCHMARK_VERSION",
    "SourceContractCacheV5_4",
    "V5_4_CONTRACT_VERSION",
    "V5_4_EXTRACTOR_VERSION",
    "V5_4_EXTRACTION_POLICY_VERSION",
    "V5_4_MIGRATION_POLICY_VERSION",
    "extract_expected_ui_contract_v5_4",
    "resolve_expected_ui_contract_v5_4",
    "source_contract_cache_key_v5_4",
]
