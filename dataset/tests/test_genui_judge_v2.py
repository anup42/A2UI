from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator
from PIL import Image


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(DATASET_ROOT / "scripts"))

from pipeline.genui_judge.judgments import (  # noqa: E402
    append_judgment_pass,
    finalize_groundtruth,
)
from pipeline.genui_judge.blind_workspace import (  # noqa: E402
    import_blind_batch,
    initialize_blind_workspace,
    prepare_blind_adjudication_batch,
    prepare_blind_batch,
)
from pipeline.genui_judge.packets import build_blinded_packets  # noqa: E402
from pipeline.genui_judge.analysis import (  # noqa: E402
    _fit_calibrators,
    _icc_absolute_agreement,
    _judge_task_provenance,
    _metric_vector_stats,
)
from pipeline.genui_judge.protocol import (  # noqa: E402
    DIMENSIONS,
    PASS_DIMENSIONS,
    compute_judged_scores,
    protocol_fingerprint,
)
from pipeline.genui_judge.provenance import (  # noqa: E402
    seal_implementation_provenance,
)
from pipeline.genui_judge.selection import (  # noqa: E402
    BLIND_REPEAT_COUNT,
    build_benchmark_selection,
)
import pipeline.genui_judge.selection as selection_module  # noqa: E402
from capture_single_codex_judge_native import (  # noqa: E402
    _set_state_action_variants,
    _tabs_state_variants,
)
from watch_single_codex_judge_block import _wait_for_import  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _pass(packet_id: str, pass_type: str, *, model: str = "codex-gpt-5") -> dict:
    return {
        "packet_id": packet_id,
        "pass_type": pass_type,
        "protocol_fingerprint": protocol_fingerprint(),
        "dimensions": {
            name: 75 for name in PASS_DIMENSIONS[pass_type]
        },
        "confidence_0_1": 0.9,
        "evidence_observations": ["Visible evidence."],
        "visible_defects": [],
        "cannot_assess": [],
        "rationale": "The visible evidence is good with noticeable defects.",
        "judge_task_id": "task-test",
        "judge_model_identifier": model,
        "judged_at": "2026-07-29T00:00:00+00:00",
    }


def test_exact_host_formulas() -> None:
    dimensions = {
        name: float(index * 10)
        for index, name in enumerate(DIMENSIONS, start=1)
    }
    result = compute_judged_scores(dimensions)
    assert result.source_representation_0_100 == pytest.approx(
        (0.18 * 10 + 0.14 * 20) / 0.32
    )
    assert result.rendered_ux_0_100 == pytest.approx(
        (
            0.14 * 30
            + 0.12 * 40
            + 0.10 * 50
            + 0.10 * 60
            + 0.08 * 70
            + 0.04 * 80
            + 0.05 * 90
            + 0.05 * 100
        )
        / 0.68
    )
    assert result.composite_0_100 == pytest.approx(
        0.18 * 10
        + 0.14 * 20
        + 0.14 * 30
        + 0.12 * 40
        + 0.10 * 50
        + 0.10 * 60
        + 0.08 * 70
        + 0.04 * 80
        + 0.05 * 90
        + 0.05 * 100
    )


def test_host_watcher_imports_only_after_result_exists(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "result.jsonl"
    ready_path = tmp_path / "READY.json"
    result_path.write_text("{}\n", encoding="utf-8")
    ready_path.write_text('{"ready":true}\n', encoding="utf-8")
    observed = []

    def importer(**kwargs: object) -> dict:
        observed.append(kwargs)
        return {"ok": True}

    receipt = _wait_for_import(
        result_path,
        ready_path,
        importer,
        {"pass_type": "screenshot_only"},
        timeout_seconds=1.0,
        poll_seconds=0.01,
    )
    assert receipt == {"ok": True}
    assert observed == [{"pass_type": "screenshot_only"}]


def test_judge_task_provenance_requires_fresh_80_packet_tasks() -> None:
    rows = []
    schedule = {}
    for position, task_id in ((0, "task-a"), (80, "task-b")):
        packet_id = f"p-{position}"
        schedule[packet_id] = {"schedule_position": position}
        for pass_type in ("screenshot_only", "source_conditioned"):
            rows.append(
                {
                    "packet_id": packet_id,
                    "pass_type": pass_type,
                    "judge_task_id": task_id,
                    "judge_model_identifier": "gpt-5.6-sol",
                    "judged_at": (
                        "2026-07-29T00:00:00+00:00"
                        if pass_type == "screenshot_only"
                        else "2026-07-29T00:01:00+00:00"
                    ),
                }
            )
    result = _judge_task_provenance(rows, schedule)
    assert result["stable_single_model"]
    assert result["fresh_task_after_each_80"]
    rows[-1]["judge_task_id"] = "task-a"
    drifted = _judge_task_provenance(rows, schedule)
    assert not drifted["fresh_task_after_each_80"]
    assert drifted["packet_pass_task_mismatches"] == ["p-80"]


def test_scores_require_five_point_increments() -> None:
    dimensions = {name: 75 for name in DIMENSIONS}
    dimensions[DIMENSIONS[0]] = 76
    with pytest.raises(ValueError, match="increments of 5"):
        compute_judged_scores(dimensions)


def test_pass_order_and_model_identity_are_enforced(tmp_path: Path) -> None:
    root = tmp_path / "benchmark"
    (root / "sealed").mkdir(parents=True)
    _write_jsonl(
        root / "sealed" / "packet_identity_map.jsonl",
        [
            {
                "packet_id": "p_one",
                "ui_id": "ui-one",
                "query_id": "q-one",
                "response_id": "r-one",
                "intent_bucket": "intent-one",
                "schedule_position": 0,
                "occurrence": 0,
                "repeat_of_packet_id": None,
            }
        ],
    )
    with pytest.raises(ValueError, match="screenshot_only"):
        append_judgment_pass(
            root, _pass("p_one", "source_conditioned")
        )
    first = append_judgment_pass(
        root, _pass("p_one", "screenshot_only")
    )
    assert not first["packet_complete"]
    with pytest.raises(ValueError, match="model identifier changed"):
        append_judgment_pass(
            root,
            _pass(
                "p_one",
                "source_conditioned",
                model="different-model",
            ),
        )
    second = append_judgment_pass(
        root, _pass("p_one", "source_conditioned")
    )
    assert second["packet_complete"]
    assert second["packet_judgment"]["composite_0_100"] == 75
    judgment_schema = json.loads(
        (
            DATASET_ROOT
            / "schema"
            / "genui_codex_judgment_v2.schema.json"
        ).read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(judgment_schema)
    for row in _read_jsonl(root / "raw_judgments.jsonl"):
        validator.validate(row)


def test_verified_candidate_render_failure_requires_zero(
    tmp_path: Path,
) -> None:
    root = tmp_path / "benchmark"
    (root / "sealed").mkdir(parents=True)
    _write_jsonl(
        root / "sealed" / "packet_identity_map.jsonl",
        [
            {
                "packet_id": "p_failed",
                "ui_id": "ui-failed",
                "query_id": "q-failed",
                "response_id": "r-failed",
                "intent_bucket": "intent-failed",
                "schedule_position": 0,
                "occurrence": 0,
                "repeat_of_packet_id": None,
            }
        ],
    )
    _write_jsonl(
        root / "native_capture_manifest.jsonl",
        [
            {
                "ui_id": "ui-failed",
                "required": True,
                "failure_class": "candidate",
                "image_captured": True,
            }
        ],
    )
    with pytest.raises(ValueError, match="must receive zero"):
        append_judgment_pass(
            root, _pass("p_failed", "screenshot_only")
        )
    value = _pass("p_failed", "screenshot_only")
    value["dimensions"] = {
        name: 0 for name in PASS_DIMENSIONS["screenshot_only"]
    }
    saved = append_judgment_pass(root, value)
    assert not saved["packet_complete"]


def test_finalized_row_carries_split_and_contract_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "benchmark"
    (root / "sealed").mkdir(parents=True)
    identity = {
        "packet_id": "p_final",
        "ui_id": "ui-final",
        "query_id": "q-final",
        "response_id": "r-final",
        "intent_bucket": "intent-final",
        "schedule_position": 0,
        "occurrence": 0,
        "repeat_of_packet_id": None,
    }
    _write_jsonl(
        root / "sealed" / "packet_identity_map.jsonl",
        [identity],
    )
    _write_jsonl(
        root / "selection_manifest.jsonl",
        [
            {
                "ui_id": "ui-final",
                "selection_stratum": "stress",
                "split": "validation",
                "migration_anchor": True,
            }
        ],
    )
    _write_jsonl(
        root / "expected_contracts.jsonl",
        [
            {
                "ui_id": "ui-final",
                "response_text_sha256": "source-hash",
                "expected_ui_contract_hash": "contract-hash",
            }
        ],
    )
    append_judgment_pass(
        root, _pass("p_final", "screenshot_only")
    )
    append_judgment_pass(
        root, _pass("p_final", "source_conditioned")
    )
    rows = finalize_groundtruth(root, require_complete=False)
    assert rows[0]["selection_stratum"] == "stress"
    assert rows[0]["split"] == "validation"
    assert rows[0]["migration_anchor"] is True
    assert rows[0]["source_text_sha256"] == "source-hash"
    assert rows[0]["expected_ui_contract_hash"] == "contract-hash"


def test_blind_workspace_withholds_source_until_screenshot_import(
    tmp_path: Path,
) -> None:
    root = tmp_path / "host_with_generator_name"
    workspace = tmp_path / "neutral_judge_workspace"
    (root / "packets" / "assets").mkdir(parents=True)
    (root / "packets" / "screenshot_only").mkdir()
    (root / "packets" / "source_conditioned").mkdir()
    (root / "sealed").mkdir()
    image_path = root / "packets" / "assets" / "p_one_00.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    overview_path = root / "packets" / "assets" / "p_one_overview.jpg"
    Image.new("RGB", (10, 10), "white").save(overview_path)
    common = {
        "packet_id": "p_one",
        "protocol_fingerprint": protocol_fingerprint(),
        "screenshots": [
            {
                "asset": "../assets/p_one_00.png",
                "sha256": hashlib.sha256(
                    image_path.read_bytes()
                ).hexdigest(),
            }
        ],
        "overview_asset": "../assets/p_one_overview.jpg",
    }
    (root / "packets" / "screenshot_only" / "p_one.json").write_text(
        json.dumps(
            {
                **common,
                "pass_type": "screenshot_only",
            }
        ),
        encoding="utf-8",
    )
    (root / "packets" / "source_conditioned" / "p_one.json").write_text(
        json.dumps(
            {
                **common,
                "pass_type": "source_conditioned",
                "source_text": "Secret source released second.",
                "expected_ui_contract": {},
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(root / "packets" / "packet_manifest.jsonl", [{}])
    _write_jsonl(
        root / "judge_schedule.jsonl",
        [
            {
                "schedule_position": 0,
                "packet_id": "p_one",
                "pilot": True,
                "block_index": 0,
                "milestone_index": 0,
            }
        ],
    )
    _write_jsonl(
        root / "sealed" / "packet_identity_map.jsonl",
        [
            {
                "packet_id": "p_one",
                "ui_id": "ui-one",
                "query_id": "q-one",
                "response_id": "r-one",
                "intent_bucket": "intent-one",
                "schedule_position": 0,
                "occurrence": 0,
                "repeat_of_packet_id": None,
            }
        ],
    )
    (root / "judge_instructions.md").write_text(
        "Frozen rubric.\n", encoding="utf-8"
    )
    (root / "judge_protocol.json").write_text(
        json.dumps(
            {"protocol_fingerprint": protocol_fingerprint()}
        ),
        encoding="utf-8",
    )
    seal_implementation_provenance(root)
    initialize_blind_workspace(root, workspace)
    screen = prepare_blind_batch(
        root,
        start=0,
        count=1,
        pass_type="screenshot_only",
    )
    batch_dir = Path(screen["batch_manifest"]).parent
    for path in workspace.rglob("*"):
        if path.suffix in {".json", ".jsonl", ".md"}:
            assert str(root) not in path.read_text(encoding="utf-8")
    assert not (batch_dir / "source_conditioned").exists()
    with pytest.raises(ValueError, match="remain sealed"):
        prepare_blind_batch(
            root,
            start=0,
            count=1,
            pass_type="source_conditioned",
        )
    _write_jsonl(
        Path(screen["result_path"]),
        [_pass("p_one", "screenshot_only")],
    )
    import_blind_batch(
        root,
        start=0,
        count=1,
        pass_type="screenshot_only",
    )
    assert not Path(screen["result_path"]).exists()
    assert (
        root
        / "sealed"
        / "judge_block_results"
        / "batch_0000_0000"
        / "screenshot_only.jsonl"
    ).exists()
    source = prepare_blind_batch(
        root,
        start=0,
        count=1,
        pass_type="source_conditioned",
    )
    released = (
        Path(source["batch_manifest"]).parent
        / "source_conditioned"
        / "p_one.json"
    )
    assert "Secret source" in released.read_text(encoding="utf-8")
    for path in workspace.rglob("*"):
        if path.suffix in {".json", ".jsonl", ".md"}:
            assert str(root) not in path.read_text(encoding="utf-8")
    _write_jsonl(
        Path(source["result_path"]),
        [_pass("p_one", "source_conditioned")],
    )
    import_blind_batch(
        root,
        start=0,
        count=1,
        pass_type="source_conditioned",
    )
    assert not (batch_dir / "assets").exists()
    assert not (batch_dir / "screenshot_only").exists()
    assert not (batch_dir / "source_conditioned").exists()
    for pass_type in ("screenshot_only", "source_conditioned"):
        original = root / "packets" / pass_type / "p_one.json"
        value = json.loads(original.read_text(encoding="utf-8"))
        value["packet_id"] = "p_extra"
        (
            root / "packets" / pass_type / "p_extra.json"
        ).write_text(json.dumps(value), encoding="utf-8")
    _write_jsonl(
        root / "pending_adjudications.jsonl",
        [{"packet_id": "p_extra"}],
    )
    extra = prepare_blind_adjudication_batch(
        root,
        start=0,
        count=1,
        pass_type="screenshot_only",
    )
    extra_manifest = json.loads(
        Path(extra["batch_manifest"]).read_text(encoding="utf-8")
    )
    assert extra_manifest["packets"][0]["packet_id"] == "p_extra"
    assert "repeat" not in json.dumps(extra_manifest).casefold()


def test_tabs_state_variants_are_bounded_and_do_not_mutate_source() -> None:
    record = {
        "ui_id": "ui-tabs",
        "genui_json": {
            "root": "root",
            "state": {},
            "elements": {
                "root": {
                    "type": "Stack",
                    "children": ["tabs"],
                    "props": {},
                },
                "tabs": {
                    "type": "Tabs",
                    "children": [],
                    "props": {
                        "tabs": [
                            {"label": f"T{index}", "child": f"child-{index}"}
                            for index in range(7)
                        ]
                    },
                },
            },
        },
    }
    source = json.loads(json.dumps(record))
    variants = _tabs_state_variants(record)
    assert len(variants) == 4
    assert record == source
    assert [
        variant["genui_json"]["elements"]["tabs"]["props"][
            "activeTabId"
        ]
        for _, _, variant in variants
    ] == ["child-1", "child-2", "child-3", "child-4"]


def test_set_state_capture_variant_is_bounded_and_non_mutating() -> None:
    record = {
        "ui_id": "ui-state",
        "genui_json": {
            "root": "root",
            "state": {"showQR": False},
            "elements": {
                "root": {
                    "type": "Button",
                    "props": {"label": "Generate"},
                    "children": [],
                    "on": {
                        "press": {
                            "action": "setState",
                            "params": {
                                "path": "/showQR",
                                "value": True,
                            },
                        }
                    },
                }
            },
        },
    }
    original = json.loads(json.dumps(record))
    variants = _set_state_action_variants(record)
    assert len(variants) == 1
    assert variants[0][2]["genui_json"]["state"]["showQR"] is True
    assert record == original


def test_packet_surfaces_are_blinded_and_ordered(tmp_path: Path) -> None:
    root = tmp_path / "benchmark"
    (root / "sealed").mkdir(parents=True)
    screenshot = root / "capture" / "generator_model_ui-1.png"
    screenshot.parent.mkdir()
    Image.new("RGB", (20, 30), "white").save(screenshot)
    _write_jsonl(
        root / "judge_schedule.jsonl",
        [
            {
                "schedule_position": 0,
                "packet_id": "p_opaque",
                "pilot": True,
                "block_index": 0,
                "milestone_index": 0,
            }
        ],
    )
    _write_jsonl(
        root / "sealed" / "packet_identity_map.jsonl",
        [
            {
                "schedule_position": 0,
                "packet_id": "p_opaque",
                "ui_id": "ui-1",
                "query_id": "q-1",
                "response_id": "r-1",
                "intent_bucket": "booking",
                "occurrence": 0,
                "repeat_of_packet_id": None,
            }
        ],
    )
    _write_jsonl(
        root / "selected_genui.jsonl",
        [
            {
                "ui_id": "ui-1",
                "response_text": "Book the 10:30 train.",
                "genui_json": {"secret": "must not leak"},
                "metrics": {"overall_score": 99},
            }
        ],
    )
    _write_jsonl(
        root / "expected_contracts.jsonl",
        [
            {
                "ui_id": "ui-1",
                "expected_ui_contract": {
                    "content_units": ["Book the 10:30 train."]
                },
                "expected_ui_contract_hash": "contract-hash",
            }
        ],
    )
    _write_jsonl(
        root / "native_capture_manifest.jsonl",
        [
            {
                "ui_id": "ui-1",
                "ok": True,
                "image_captured": True,
                "local_path": str(screenshot),
                "viewport_profile": "compact",
                "viewport_width_dp": 411,
                "viewport_height_dp": 960,
                "capture_kind": "initial_viewport",
                "state_id": "initial",
                "state_description": "Initial state",
            }
        ],
    )
    result = build_blinded_packets(root)
    assert result["packet_count"] == 1
    screenshot_packet = json.loads(
        (
            root / "packets" / "screenshot_only" / "p_opaque.json"
        ).read_text(encoding="utf-8")
    )
    conditioned_packet = json.loads(
        (
            root / "packets" / "source_conditioned" / "p_opaque.json"
        ).read_text(encoding="utf-8")
    )
    serialized = json.dumps(screenshot_packet)
    assert "Book the 10:30 train" not in serialized
    assert "genui_json" not in serialized
    assert "overall_score" not in serialized
    assert "generator_model" not in serialized
    assert screenshot_packet["pass_type"] == "screenshot_only"
    assert conditioned_packet["pass_type"] == "source_conditioned"
    assert conditioned_packet["source_text"] == "Book the 10:30 train."


def test_reliability_and_calibration_helpers_are_exact() -> None:
    assert _icc_absolute_agreement([10, 30, 70], [10, 30, 70]) == pytest.approx(
        1.0
    )
    stats = _metric_vector_stats([10, 20, 30], [10, 20, 30])
    assert stats["pearson_r"] == pytest.approx(1.0)
    assert stats["spearman_rho"] == pytest.approx(1.0)
    assert stats["mae"] == 0.0
    rows = []
    for split_index, split in enumerate(
        ("calibration", "validation", "holdout")
    ):
        for index in range(10):
            metric = float(split_index * 10 + index)
            rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "target": 2.0 * metric + 5.0,
                }
            )
    calibration = _fit_calibrators(
        rows,
        metric_key="metric",
        target_key="target",
    )
    assert calibration["selected_model"] == "affine"
    assert calibration["validation"]["affine"]["mae"] == pytest.approx(0.0)
    assert calibration["holdout_selected_model_once"]["mae"] == pytest.approx(
        0.0
    )


def _synthetic_source_rows() -> list[dict]:
    rows: list[dict] = []
    role_types = (
        "Text",
        "Table",
        "Chart",
        "Formula",
        "Tabs",
        "Image",
        "Modal",
    )
    for intent_index in range(32):
        intent = f"intent_{intent_index:02d}"
        for sample_index in range(40):
            ui_id = f"ui_{intent_index:02d}_{sample_index:02d}"
            element_type = role_types[sample_index % len(role_types)]
            props: dict = {"text": f"Value {sample_index}"}
            if element_type == "Table":
                props = {
                    "columns": [
                        {"key": f"c{index}", "label": f"C{index}"}
                        for index in range(1 + sample_index % 7)
                    ],
                    "rows": [{"c0": "value"}],
                }
            elements = {
                "root": {
                    "type": "Stack",
                    "children": ["content"],
                    "props": {},
                },
                "content": {
                    "type": element_type,
                    "children": [],
                    "props": props,
                },
            }
            rows.append(
                {
                    "ui_id": ui_id,
                    "query_id": f"query_{intent_index:02d}_{sample_index:02d}",
                    "response_id": (
                        f"response_{intent_index:02d}_{sample_index:02d}"
                    ),
                    "intent_bucket": intent,
                    "response_text": (
                        f"Source {intent} {sample_index} "
                        + ("long " * sample_index)
                    ),
                    "assets": [],
                    "genui_json": {
                        "root": "root",
                        "state": {},
                        "elements": elements,
                    },
                    "metrics": {
                        "overall_score": float(50 + sample_index)
                    },
                }
            )
    return rows


def test_selection_is_exact_deterministic_grouped_and_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = source_dir / "genui.jsonl"
    rows = _synthetic_source_rows()
    _write_jsonl(source_path, rows)
    original_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    _write_jsonl(
        prior_dir / "selection_manifest.jsonl",
        [
            {
                "ui_id": f"ui_{intent_index:02d}_00",
                "intent_bucket": f"intent_{intent_index:02d}",
            }
            for intent_index in range(32)
        ],
    )

    def resolve(text: str, **_: object) -> SimpleNamespace:
        source_hash = hashlib.sha256(text.encode()).hexdigest()
        return SimpleNamespace(
            contract={
                "contract_version": "test",
                "extractor_version": "test",
                "source_hash": source_hash,
                "content_units": [text],
            },
            source="deterministic fallback",
            errors=(),
        )

    monkeypatch.setattr(
        selection_module,
        "resolve_expected_ui_contract_v5_4",
        resolve,
    )
    monkeypatch.setattr(
        selection_module,
        "expected_contract_hash_v5_4",
        lambda value: hashlib.sha256(
            json.dumps(value, sort_keys=True).encode()
        ).hexdigest(),
    )

    first = tmp_path / "benchmark_one"
    second = tmp_path / "benchmark_two"
    manifest = build_benchmark_selection(
        source_dir,
        first,
        prior_v1_path=prior_dir,
        seed="unit-test-seed",
    )
    build_benchmark_selection(
        source_dir,
        second,
        prior_v1_path=prior_dir,
        seed="unit-test-seed",
    )
    selected = _read_jsonl(first / "selection_manifest.jsonl")
    assert len(selected) == 960
    assert set(Counter(row["intent_bucket"] for row in selected).values()) == {
        30
    }
    assert Counter(row["selection_stratum"] for row in selected) == {
        "population": 640,
        "stress": 320,
    }
    assert Counter(row["split"] for row in selected) == {
        "calibration": 640,
        "validation": 160,
        "holdout": 160,
    }
    assert sum(bool(row["migration_anchor"]) for row in selected) == 32
    assert len({row["query_id"] for row in selected}) == 960
    schedule = _read_jsonl(first / "judge_schedule.jsonl")
    packet_map = {
        row["packet_id"]: row
        for row in _read_jsonl(
            first / "sealed" / "packet_identity_map.jsonl"
        )
    }
    assert len(schedule) == 960 + BLIND_REPEAT_COUNT
    repeats = [
        packet_map[row["packet_id"]]
        for row in schedule
        if packet_map[row["packet_id"]]["occurrence"] == 1
    ]
    assert len(repeats) == BLIND_REPEAT_COUNT
    positions = {
        row["packet_id"]: row["schedule_position"]
        for row in packet_map.values()
    }
    assert all(
        int(row["schedule_position"])
        - int(positions[row["repeat_of_packet_id"]])
        >= 200
        for row in repeats
    )
    anchor_ids = {
        row["ui_id"] for row in selected if row["migration_anchor"]
    }
    assert {
        row["ui_id"] for row in repeats[-32:]
    } == anchor_ids
    assert all(row["ui_id"] in anchor_ids for row in repeats[-32:])
    assert all(
        row["schedule_position"] >= len(schedule) - 32
        for row in repeats
        if row["ui_id"] in anchor_ids
    )
    assert all(row["pilot"] for row in schedule[:32])
    assert not any(row["pilot"] for row in schedule[32:])
    assert {row["block_index"] for row in schedule[:20]} == {0}
    assert {row["block_index"] for row in schedule[20:32]} == {1}
    for start in range(32, len(schedule), 20):
        assert len(
            {row["block_index"] for row in schedule[start : start + 20]}
        ) == 1
    for name in (
        "selection_manifest.jsonl",
        "selected_genui.jsonl",
        "expected_contracts.jsonl",
        "judge_schedule.jsonl",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert manifest["source_genui_sha256"] == original_hash
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == original_hash
    with pytest.raises(FileExistsError, match="immutable"):
        build_benchmark_selection(
            source_dir,
            first,
            prior_v1_path=prior_dir,
        )
