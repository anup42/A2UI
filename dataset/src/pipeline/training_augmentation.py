"""Dataset-owned, bounded synthetic source augmentation and Stage 3 labeling.

Training supplies already split training donors. This module never reads a
heldout dataset, donor targets, or edits a generated label. Model source review
is an assessment of synthetic coherence, never independent fact verification.
"""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

VERSION = "training-augmentation-v1"
DEFAULT_TEACHER = "muse_glimmer_30b_sglang_reasoning_dflash"
DATASET_ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = (
    "multisection_completeness", "table_count_variations", "coherent_numeric_entities",
    "action_no_action_contrasts", "wording_formats", "missing_uncertain_facts",
    "length_position", "forms_rare_controls", "literal_media_references",
)
RECIPES = {
    "multisection_completeness": "Produce at least three meaningful sections, with facts in every section and an essential final section. Preserve all required facts across sections.",
    "table_count_variations": "Produce a coherent comparison or schedule table; vary row and column counts, explicit totals and singular/plural wording. All cells and counts must agree.",
    "coherent_numeric_entities": "Create synthetic substituted entities and numeric values. Recompute every dependent total, percentage, date/time constraint and summary; never perform isolated numeric replacement.",
    "action_no_action_contrasts": "Alternate by variant parity: even variants have explicit grounded Action: [Button: label] destination lines; odd variants are informational with no action requests, buttons or invented action destinations.",
    "wording_formats": "Rewrite the same information using a different natural wording and presentation (prose, bullets, headings or table), preserving semantic roles and exact required literals.",
    "missing_uncertain_facts": "Make selected facts explicitly unavailable, unknown, tentative or conditional. Keep qualifications and uncertainty visible; never replace missing facts with plausible invented answers.",
    "length_position": "Vary source length and ordering. Place required facts at both beginning and end and preserve middle sections. Odd variants are compact; even variants are longer with useful concrete details, not repetition.",
    "forms_rare_controls": "Describe a synthetic usable form with explicit labels, initial values, allowed options, required/optional status and constraints. Rotate among supported input, date/time, slider, checkbox and choice controls. State any submit operation as mock; never imply a real external capability.",
    "literal_media_references": "Stress exact punctuation, quotes, identifiers, Unicode and declared references. Use only supplied reference tokens in their declared roles, with useful media only; do not add decorative icons or invent placeholder tokens, destinations or assets. Never claim an asset exists or was downloaded.",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def text_hash(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Malformed JSONL at {path.name}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path.name}:{line_number}")  # noqa: TRY004 - external JSON schema violation
            rows.append(row)
    return rows


def load_donors(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("No training donors supplied")
    ids: set[str] = set()
    sources: set[str] = set()
    for row in rows:
        for key in ("donor_id", "source_group_id", "response_text"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Donor requires nonempty {key}")
        if row.get("split") != "train":
            raise ValueError("Only explicit split='train' donors are permitted")
        source = row["response_text"].strip()
        if row["donor_id"] in ids or source in sources:
            raise ValueError("Duplicate donor ID or exact source text")
        if len(source) > 60000:
            raise ValueError("Donor source exceeds the 60000-character budget")
        ids.add(row["donor_id"])
        sources.add(source)
    # Deliberately discard donor labels, contracts, review claims and other fields.
    return [{key: row[key] for key in ("donor_id", "source_group_id", "response_text", "split")} for row in rows]


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _json_result(result: Any) -> dict[str, Any]:
    if result.error or result.completion_complete is not True:
        raise ValueError("teacher_completion_failed_or_incomplete")
    value = json.loads(result.text)
    if not isinstance(value, dict):
        raise ValueError("teacher_result_not_object")  # noqa: TRY004 - external JSON schema violation
    return value


def validate_review(value: dict[str, Any], category: str) -> None:
    if value.get("category") != category:
        raise ValueError("source_review_category_mismatch")
    for key in ("approved", "coherent", "category_satisfied", "synthetic_provenance_clear"):
        if value.get(key) is not True:
            raise ValueError(f"source_review_failed:{key}")
    if value.get("issues") != []:
        raise ValueError("source_review_has_issues_or_malformed_issues")


def admission_errors(row: dict[str, Any], response_text: str) -> list[str]:
    reasons = []
    if row.get("record_status") != "accepted":
        reasons.append("stage3_not_accepted")
    acceptance = row.get("training_acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("eligible") is not True:
        reasons.append("training_not_eligible")
    if not isinstance(acceptance, dict) or acceptance.get("blocking_reasons") != [] or acceptance.get("review_reasons") != []:
        reasons.append("training_blocking_or_review_reasons")
    validation = row.get("validation") or {}
    for field in ("schema_valid_strict", "standard_a2ui_valid"):
        if validation.get(field) is not True:
            reasons.append(field)
    gen = row.get("gen") or {}
    if gen.get("error") or gen.get("completion_complete") is not True:
        reasons.append("generation_failed_or_incomplete")
    if row.get("response_text") != response_text:
        reasons.append("source_binding_mismatch")
    if row.get("source_format") != "a2ui_express_v1" or not isinstance(row.get("a2ui_express"), str) or not row["a2ui_express"].strip():
        reasons.append("missing_express_completion")
    return reasons


@contextmanager
def bounded_teacher_environment():
    # Isolated dataset subprocess; restore settings for embedded/test callers.
    overrides = {
        "LOCAL_ALLOW_HTTP_ENDPOINT": "1", "LOCAL_STRICT_OFFLINE": "0",
        "LOCAL_VLLM_ENABLE_THINKING": "1", "LOCAL_MUSE_REQUIRE_REASONING": "1",
        "LOCAL_VLLM_SEND_CHAT_TEMPLATE_KWARGS": "1",
        "LOCAL_VLLM_TIMEOUT_SECONDS": "180", "LOCAL_VLLM_RETRY_CONNECTION_ERRORS": "0",
        "LOCAL_VLLM_RETRY_RESULT_ERRORS": "0", "LOCAL_VLLM_RETRY_MAX_SECONDS": "180",
        "LOCAL_VLLM_BATCH_PARALLELISM": "1", "LOCAL_VLLM_PARALLEL_REQUESTS": "1",
        "STAGE3_FINAL_REGEN_ATTEMPTS": "1", "STAGE3_RESPECT_CONFIG_PROMPT_MAX": "1",
        "LOCAL_VLLM_MAX_OUTPUT_TOKENS": "12288",
        "LOCAL_VLLM_REASONING_STRENGTH": os.environ.get("LOCAL_VLLM_REASONING_STRENGTH", "high"),
        "LOCAL_VLLM_SERVED_MODEL": os.environ.get("LOCAL_VLLM_SERVED_MODEL", "muse-glimmer"),
    }
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield overrides
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def generate_training_augmentations(
    donors_path: Path, output_dir: Path, *, teacher_model: str = DEFAULT_TEACHER,
    max_new_samples: int = 90, seed: int = 123, adapter: Any = None,
    stage3_runner: Any = None, resume: bool = False,
    reference_bindings_path: Path | None = None,
) -> dict[str, Any]:
    """Generate or strictly resume a bounded set of logical augmentation slots."""
    from pipeline.training_augmentation_resume import generate
    return generate(donors_path, output_dir, teacher_model=teacher_model,
                    max_new_samples=max_new_samples, seed=seed, adapter=adapter,
                    stage3_runner=stage3_runner, resume=resume,
                    reference_bindings_path=reference_bindings_path)


def validate_generation_resume(
    donors_path: Path, output_dir: Path, *, teacher_model: str = DEFAULT_TEACHER,
    max_new_samples: int = 90, seed: int = 123,
    reference_bindings_path: Path | None = None,
) -> dict[str, Any]:
    """Validate resume compatibility without calls or writes, including legacy v1."""
    from pipeline.training_augmentation_resume import (
        validate_generation_resume as validate,
    )

    return validate(donors_path, output_dir, teacher_model=teacher_model,
                    max_new_samples=max_new_samples, seed=seed,
                    reference_bindings_path=reference_bindings_path)
