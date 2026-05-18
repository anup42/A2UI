from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ir_training.common.config import repo_root


@dataclass(frozen=True)
class ValidationOutcome:
    valid: bool
    reason: str | None = None


class FlatSpecValidator:
    def __init__(self, require_strict: bool = True) -> None:
        self.require_strict = require_strict
        self._dataset_contract = self._load_dataset_contract()

    def validate(self, value: Any) -> ValidationOutcome:
        if value is None:
            return ValidationOutcome(False, "missing_genui_json")
        if self._dataset_contract is not None:
            result = self._dataset_contract.coerce_and_validate(value)
            if not result.is_valid:
                return ValidationOutcome(False, result.error or "flat_spec_invalid")
            return ValidationOutcome(True)
        if not isinstance(value, dict):
            return ValidationOutcome(False, "not_object")
        root = value.get("root")
        elements = value.get("elements")
        if not isinstance(root, str) or not root:
            return ValidationOutcome(False, "missing_root")
        if not isinstance(elements, dict) or not elements:
            return ValidationOutcome(False, "missing_elements")
        if root not in elements:
            return ValidationOutcome(False, "root_not_in_elements")
        for element_id, element in elements.items():
            if not isinstance(element, dict):
                return ValidationOutcome(False, f"element_not_object:{element_id}")
            if not isinstance(element.get("type"), str):
                return ValidationOutcome(False, f"missing_type:{element_id}")
            if not isinstance(element.get("props"), dict):
                return ValidationOutcome(False, f"missing_props:{element_id}")
            if not isinstance(element.get("children"), list):
                return ValidationOutcome(False, f"missing_children:{element_id}")
        return ValidationOutcome(True)

    @staticmethod
    def output_size(value: Any) -> int:
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))

    def _load_dataset_contract(self):
        root = repo_root()
        dataset_src = root / "dataset" / "src"
        if dataset_src.exists() and str(dataset_src) not in sys.path:
            sys.path.insert(0, str(dataset_src))
        try:
            from pipeline import flat_spec_contract  # type: ignore
            return flat_spec_contract
        except Exception:
            return None


def row_passes_basic_filters(
    response_text: str,
    genui_json: Any,
    validator: FlatSpecValidator,
    max_input_chars: int,
    max_output_chars: int,
) -> ValidationOutcome:
    if not response_text.strip():
        return ValidationOutcome(False, "missing_response_text")
    if len(response_text) > max_input_chars:
        return ValidationOutcome(False, "input_too_large")
    output_size = FlatSpecValidator.output_size(genui_json)
    if output_size > max_output_chars:
        return ValidationOutcome(False, "output_too_large")
    return validator.validate(genui_json)
