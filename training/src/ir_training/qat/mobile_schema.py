from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ir_training.common.config import load_yaml, resolve_path, training_root

PUBLIC_GEMMA4_E2B_MOBILE_SCHEMA = (
    training_root() / "configs" / "quantization" / "gemma4_e2b_mobile_public_schema.yaml"
)


@dataclass(frozen=True)
class MobileQuantRule:
    pattern: str
    num_bits: int
    group_size: int | None = None


@dataclass(frozen=True)
class MobileQuantSchema:
    profile: str
    quant_method: str
    num_bits: int
    quantize_embeddings: bool
    module_quant_configs: tuple[MobileQuantRule, ...]
    modules_to_not_convert: tuple[str, ...]

    def bits_for_module(self, module_name: str) -> int | None:
        """Return the first matching public bit assignment or the default."""

        if any(re.search(pattern, module_name) for pattern in self.modules_to_not_convert):
            return None
        for rule in self.module_quant_configs:
            if re.search(rule.pattern, module_name):
                return rule.num_bits
        return self.num_bits

    def group_size_for_module(self, module_name: str) -> int | None:
        for rule in self.module_quant_configs:
            if re.search(rule.pattern, module_name):
                return rule.group_size
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "quant_method": self.quant_method,
            "num_bits": self.num_bits,
            "quantize_embeddings": self.quantize_embeddings,
            "module_quant_configs": {
                rule.pattern: {
                    "num_bits": rule.num_bits,
                    **(
                        {"group_size": rule.group_size}
                        if rule.group_size is not None
                        else {}
                    ),
                }
                for rule in self.module_quant_configs
            },
            "modules_to_not_convert": list(self.modules_to_not_convert),
        }


@dataclass(frozen=True)
class SchemaIssue:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


def load_mobile_quant_schema(path: str | Path = PUBLIC_GEMMA4_E2B_MOBILE_SCHEMA) -> MobileQuantSchema:
    config = load_yaml(path)
    return schema_from_config(config)


def schema_from_config(config: dict[str, Any]) -> MobileQuantSchema:
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        quantization = config
    module_configs = quantization.get("module_quant_configs") or {}
    rules: list[MobileQuantRule] = []
    if isinstance(module_configs, dict):
        for pattern, rule in module_configs.items():
            bits = rule.get("num_bits") if isinstance(rule, dict) else rule
            group_size = rule.get("group_size") if isinstance(rule, dict) else None
            rules.append(
                MobileQuantRule(
                    str(pattern),
                    int(bits),
                    int(group_size) if group_size not in (None, "", 0) else None,
                )
            )
    elif isinstance(module_configs, list):
        for item in module_configs:
            if isinstance(item, dict) and "pattern" in item:
                group_size = item.get("group_size")
                rules.append(
                    MobileQuantRule(
                        str(item["pattern"]),
                        int(item["num_bits"]),
                        int(group_size)
                        if group_size not in (None, "", 0)
                        else None,
                    )
                )
    excluded = quantization.get("modules_to_not_convert") or []
    if isinstance(excluded, str):
        excluded = [excluded]
    return MobileQuantSchema(
        profile=str(config.get("profile") or "gemma4_e2b_mobile_public_schema"),
        quant_method=str(quantization.get("quant_method") or ""),
        num_bits=int(quantization.get("num_bits", 0)),
        quantize_embeddings=bool(quantization.get("quantize_embeddings", False)),
        module_quant_configs=tuple(rules),
        modules_to_not_convert=tuple(str(item) for item in excluded),
    )


def compare_to_public_schema(
    observed_config: dict[str, Any],
    expected: MobileQuantSchema | None = None,
) -> list[SchemaIssue]:
    expected = expected or load_mobile_quant_schema()
    observed = schema_from_config(observed_config)
    issues: list[SchemaIssue] = []
    if observed.quant_method != expected.quant_method:
        issues.append(SchemaIssue("quant_method_mismatch", f"Expected {expected.quant_method!r}, got {observed.quant_method!r}."))
    if observed.num_bits != expected.num_bits:
        issues.append(SchemaIssue("default_bits_mismatch", f"Expected {expected.num_bits}, got {observed.num_bits}."))
    if observed.quantize_embeddings != expected.quantize_embeddings:
        issues.append(
            SchemaIssue(
                "embedding_flag_mismatch",
                f"Expected quantize_embeddings={expected.quantize_embeddings}, got {observed.quantize_embeddings}.",
            )
        )
    expected_rules = [
        (rule.pattern, rule.num_bits, rule.group_size)
        for rule in expected.module_quant_configs
    ]
    observed_rules = [
        (rule.pattern, rule.num_bits, rule.group_size)
        for rule in observed.module_quant_configs
    ]
    if observed_rules != expected_rules:
        issues.append(
            SchemaIssue(
                "module_rules_mismatch",
                "Module bit/group rules or their order differ from the public schema.",
            )
        )
    if observed.modules_to_not_convert != expected.modules_to_not_convert:
        issues.append(SchemaIssue("excluded_modules_mismatch", "modules_to_not_convert differs from the public schema."))
    return issues


def audit_module_names(
    module_names: Iterable[str],
    schema: MobileQuantSchema | None = None,
) -> dict[str, Any]:
    schema = schema or load_mobile_quant_schema()
    assignments: dict[str, int | None] = {}
    for name in module_names:
        assignments[str(name)] = schema.bits_for_module(str(name))
    counts: dict[str, int] = {}
    for bits in assignments.values():
        label = "excluded" if bits is None else str(bits)
        counts[label] = counts.get(label, 0) + 1
    return {
        "module_count": len(assignments),
        "bit_counts": dict(sorted(counts.items())),
        "assignments": assignments,
    }


def resolve_schema_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    training_relative = resolve_path(path, training_root())
    if training_relative.exists():
        return training_relative
    return resolve_path(path, training_root().parent)
