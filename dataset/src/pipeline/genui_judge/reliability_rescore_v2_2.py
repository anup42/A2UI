"""Second immutable reliability refresh for the Single-Codex judge.

The v2 and v2.1 artifacts remain immutable.  This module applies the shared
rescore engine with a v2.2 protocol profile, a more deterministic severity
ledger, and a separate sidecar/output namespace.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Mapping, Sequence

from . import reliability_rescore as _engine


RESCORE_SCHEMA_VERSION = "genui_single_codex_reliability_rescore.v2.2"
RESCORE_PROTOCOL_VERSION = "genui_single_codex_visual_rubric.v2.2.0"
EXPECTED_MODEL_IDENTIFIER = "gpt-5.6-sol"
DEFAULT_TARGET_MILESTONES = (0, 3, 7)

_PROFILE_LOCK = RLock()
_PROFILE: dict[str, Any] = {
    "RESCORE_SCHEMA_VERSION": RESCORE_SCHEMA_VERSION,
    "RESCORE_PROTOCOL_VERSION": RESCORE_PROTOCOL_VERSION,
    "RESCORE_SELECTION_POLICY_VERSION": (
        "milestone-plus-repeat-counterparts.v2"
    ),
    "RESCORE_PACKET_SCHEMA_VERSION": (
        "genui_single_codex_blinded_packet.v2.2"
    ),
    "RESCORE_JUDGMENT_SCHEMA_VERSION": (
        "genui_single_codex_judgment.v2.2"
    ),
    "EXPECTED_MODEL_IDENTIFIER": EXPECTED_MODEL_IDENTIFIER,
    "RESCORE_DIR_NAME": "reliability_rescore_v2_2",
    "RUBRIC_REFRESH_FILENAME": (
        "genui_single_codex_judge_refresh_v2_2.md"
    ),
    "RUBRIC_REFRESH_VERSION": "v2.2",
    "REPEAT_RELIABILITY_NAME": "repeat_reliability_v2_2.json",
    "FINALIZED_SCHEMA_VERSION": (
        "genui_single_codex_groundtruth.v2.2"
    ),
    "LABEL_ORIGIN": "rubric_refresh_v2_2",
    "RAW_NAME": "raw_judgments_v2_2.jsonl",
    "COMBINED_NAME": "judgments_by_packet_v2_2.jsonl",
    "PENDING_NAME": "pending_adjudications_v2_2.jsonl",
    "REPEAT_NAME": "repeat_analysis_v2_2.jsonl",
    "OVERALL_REPEAT_NAME": "overall_repeat_analysis_v2_2.jsonl",
    "FINALIZED_NAME": "finalized_groundtruth_v2_2.jsonl",
}


@contextmanager
def _v2_2_profile() -> Iterator[None]:
    """Temporarily apply the v2.2 constants to the shared rescore engine."""

    with _PROFILE_LOCK:
        previous = {
            name: getattr(_engine, name)
            for name in _PROFILE
        }
        try:
            for name, value in _PROFILE.items():
                setattr(_engine, name, value)
            yield
        finally:
            for name, value in previous.items():
                setattr(_engine, name, value)


def rescore_protocol_mapping(
    target_milestones: Sequence[int] = DEFAULT_TARGET_MILESTONES,
) -> dict[str, Any]:
    with _v2_2_profile():
        return _engine.rescore_protocol_mapping(target_milestones)


def rescore_protocol_fingerprint(
    target_milestones: Sequence[int] = DEFAULT_TARGET_MILESTONES,
) -> str:
    with _v2_2_profile():
        return _engine.rescore_protocol_fingerprint(target_milestones)


def initialize_reliability_rescore(
    base_benchmark_dir: str | Path,
    workspace_dir: str | Path,
    *,
    target_milestones: Sequence[int] = DEFAULT_TARGET_MILESTONES,
) -> dict[str, Any]:
    with _v2_2_profile():
        return _engine.initialize_reliability_rescore(
            base_benchmark_dir,
            workspace_dir,
            target_milestones=target_milestones,
        )


def select_rescore_positions(
    schedule_positions: Sequence[int],
    repeat_rows: Sequence[Mapping[str, Any]],
    target_milestones: Sequence[int],
) -> tuple[list[list[int]], list[int]]:
    with _v2_2_profile():
        return _engine.select_rescore_positions(
            schedule_positions,
            repeat_rows,
            target_milestones,
        )


def prepare_rescore_batch(*args: Any, **kwargs: Any) -> dict[str, Any]:
    with _v2_2_profile():
        return _engine.prepare_rescore_batch(*args, **kwargs)


def validate_rescore_pass(*args: Any, **kwargs: Any) -> dict[str, Any]:
    with _v2_2_profile():
        return _engine.validate_rescore_pass(*args, **kwargs)


def import_rescore_batch(*args: Any, **kwargs: Any) -> dict[str, Any]:
    with _v2_2_profile():
        return _engine.import_rescore_batch(*args, **kwargs)


def rescore_status(*args: Any, **kwargs: Any) -> dict[str, Any]:
    with _v2_2_profile():
        return _engine.rescore_status(*args, **kwargs)


def analyze_rescore_repeats(*args: Any, **kwargs: Any) -> dict[str, Any]:
    with _v2_2_profile():
        return _engine.analyze_rescore_repeats(*args, **kwargs)


def finalize_rescore_overlay(*args: Any, **kwargs: Any) -> dict[str, Any]:
    with _v2_2_profile():
        return _engine.finalize_rescore_overlay(*args, **kwargs)


__all__ = [
    "DEFAULT_TARGET_MILESTONES",
    "EXPECTED_MODEL_IDENTIFIER",
    "RESCORE_PROTOCOL_VERSION",
    "analyze_rescore_repeats",
    "finalize_rescore_overlay",
    "import_rescore_batch",
    "initialize_reliability_rescore",
    "prepare_rescore_batch",
    "rescore_protocol_fingerprint",
    "rescore_protocol_mapping",
    "rescore_status",
    "select_rescore_positions",
    "validate_rescore_pass",
]
