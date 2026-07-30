"""Fail-fast release invariants for metric v5.4."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from .config_v5_4 import REWARD_VERSION_V54, load_v5_4_reward_config
from .evidence_v5_4 import DYNAMIC_PARITY_VECTOR_VERSION


VALIDATION_POLICY_VERSION = "5.4.0"
_LOCK = Lock()
_READY = False


class MetricV54InitializationError(RuntimeError):
    pass


def ensure_v5_4_validation_ready() -> None:
    global _READY
    if _READY:
        return
    with _LOCK:
        if _READY:
            return
        repo = Path(__file__).resolve().parents[4]
        dataset = repo / "dataset"
        config = load_v5_4_reward_config()
        if not config.atomic_weights:
            raise MetricV54InitializationError("v5.4 atomic weights are empty")
        fixture = (
            dataset
            / "tests"
            / "fixtures"
            / "flat_expr_parity_vectors_v5_4.json"
        )
        android = (
            repo
            / "android"
            / "app"
            / "src"
            / "test"
            / "resources"
            / "flat_expr_parity_vectors_v5_4.json"
        )
        required = [
            fixture,
            android,
            dataset / "schema" / "genui_flatspec.schema.json",
            dataset / "schema" / "expected_ui_contract.schema.json",
            dataset / "prompts" / "genui_gen_mobile_flatspec_v11.md",
            dataset / "scripts" / "capture_android_run_screenshots.py",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise MetricV54InitializationError(
                "v5.4 release dependency missing: " + ", ".join(missing)
            )
        if fixture.read_bytes() != android.read_bytes():
            raise MetricV54InitializationError(
                "Python and Android v5.4 parity corpora differ"
            )
        left = json.loads(fixture.read_text(encoding="utf-8"))
        if left.get("version") != DYNAMIC_PARITY_VECTOR_VERSION:
            raise MetricV54InitializationError(
                "v5.4 dynamic parity version mismatch"
            )
        config_version = REWARD_VERSION_V54
        if config_version != "5.4.0":
            raise MetricV54InitializationError("unexpected v5.4 version")
        _READY = True


__all__ = [
    "MetricV54InitializationError",
    "VALIDATION_POLICY_VERSION",
    "ensure_v5_4_validation_ready",
]
