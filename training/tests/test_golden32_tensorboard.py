from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.data.build_pairs import prepare_dataset
from ir_training.eval.external_runner import (
    _assert_target_not_exposed,
    aggregate_external_runtime_metrics,
    merge_external_outputs,
    run_external_generation,
    validate_external_runner_config,
)
from ir_training.eval.golden_set import (
    load_fixed_golden_rows,
    validate_prepared_golden_contract,
)
from ir_training.eval.generate import build_prediction_record
from ir_training.eval.tensorboard_logging import (
    artifact_identity,
    log_evaluation_result,
    resolve_tensorboard_root,
)
from ir_training.train.sft import _resolve_training_tensorboard_dir


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_dataset_preparation_binds_source_genui_sha256(tmp_path):
    run_dir = tmp_path / "source"
    genui_path = run_dir / "genui.jsonl"
    _write_jsonl(
        genui_path,
        [
            {
                "response_id": "r1",
                "source_id": "source-1",
                "response_text": "Show a status card.",
                "genui_json": {
                    "root": "root",
                    "state": {},
                    "elements": {
                        "root": {
                            "type": "Text",
                            "props": {"text": "Ready"},
                            "children": [],
                        }
                    },
                },
            }
        ],
    )
    digest = hashlib.sha256(genui_path.read_bytes()).hexdigest()
    responses_path = run_dir / "responses.jsonl"
    _write_jsonl(
        responses_path,
        [{"response_id": "r1", "response_text": "Show a status card."}],
    )
    responses_digest = hashlib.sha256(responses_path.read_bytes()).hexdigest()
    config = {
        "run": {
            "source_run_dir": str(run_dir),
            "source_genui_sha256": digest,
            "source_responses_sha256": responses_digest,
            "output_dir": str(tmp_path / "prepared"),
        },
        "filters": {
            "required_accepted_rows": 1,
            "require_exact_accepted_rows": True,
            "require_unique_source_ids": True,
        },
        "split": {"train": 1.0, "val": 0.0, "test": 0.0},
    }
    config_path = tmp_path / "dataset_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    manifest = prepare_dataset(config, config_path=config_path)

    assert manifest["source_genui_sha256s"] == {str(genui_path): digest}
    assert manifest["source_responses_sha256s"] == {
        str(responses_path): responses_digest
    }
    assert manifest["config_sha256"] == hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    assert manifest["output_sha256s"]["all.jsonl"] == hashlib.sha256(
        (tmp_path / "prepared" / "all.jsonl").read_bytes()
    ).hexdigest()
    persisted_manifest = json.loads(
        (tmp_path / "prepared" / "manifest.json").read_text(encoding="utf-8")
    )
    assert persisted_manifest["config_sha256"] == manifest["config_sha256"]
    contract = validate_prepared_golden_contract(
        tmp_path / "prepared" / "all.jsonl",
        dataset_config_path=config_path,
        source_genui_path=genui_path,
        source_responses_path=responses_path,
        expected_genui_sha256=digest,
        expected_responses_sha256=responses_digest,
        required_rows=1,
    )
    assert contract["rows"] == 1
    assert contract["split_sha256"] == manifest["output_sha256s"]["all.jsonl"]
    pinned_config_sha = manifest["config_sha256"]
    pinned_split_sha = manifest["output_sha256s"]["all.jsonl"]
    validate_prepared_golden_contract(
        tmp_path / "prepared" / "all.jsonl",
        dataset_config_path=config_path,
        source_genui_path=genui_path,
        source_responses_path=responses_path,
        expected_genui_sha256=digest,
        expected_responses_sha256=responses_digest,
        expected_dataset_config_sha256=pinned_config_sha,
        expected_split_sha256=pinned_split_sha,
        required_rows=1,
    )

    # A coordinated config edit plus regeneration is self-consistent with the
    # new manifest but must still fail the independent immutable config pin.
    config["filters"]["max_input_chars"] = 59999
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    prepare_dataset(config, config_path=config_path)
    with pytest.raises(ValueError, match="independently pinned digest"):
        validate_prepared_golden_contract(
            tmp_path / "prepared" / "all.jsonl",
            dataset_config_path=config_path,
            source_genui_path=genui_path,
            source_responses_path=responses_path,
            expected_genui_sha256=digest,
            expected_responses_sha256=responses_digest,
            expected_dataset_config_sha256=pinned_config_sha,
            expected_split_sha256=pinned_split_sha,
            required_rows=1,
        )

    # Restore the pinned config, then prove a coordinated split+manifest edit
    # cannot replace the fixed examples while retaining a valid score lane.
    config["filters"].pop("max_input_chars")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    prepare_dataset(config, config_path=config_path)
    prepared_split = tmp_path / "prepared" / "all.jsonl"
    prepared_rows = list(
        map(json.loads, prepared_split.read_text(encoding="utf-8").splitlines())
    )
    prepared_rows[0]["coordinated_drift"] = True
    _write_jsonl(prepared_split, prepared_rows)
    prepared_manifest_path = tmp_path / "prepared" / "manifest.json"
    prepared_manifest = json.loads(
        prepared_manifest_path.read_text(encoding="utf-8")
    )
    prepared_manifest["output_sha256s"]["all.jsonl"] = hashlib.sha256(
        prepared_split.read_bytes()
    ).hexdigest()
    prepared_manifest_path.write_text(
        json.dumps(prepared_manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Prepared Golden split differs"):
        validate_prepared_golden_contract(
            prepared_split,
            dataset_config_path=config_path,
            source_genui_path=genui_path,
            source_responses_path=responses_path,
            expected_genui_sha256=digest,
            expected_responses_sha256=responses_digest,
            expected_dataset_config_sha256=pinned_config_sha,
            expected_split_sha256=pinned_split_sha,
            required_rows=1,
        )
    prepare_dataset(config, config_path=config_path)

    config["run"]["source_genui_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare_dataset(config)
    config["run"]["source_genui_sha256"] = digest
    config["run"]["source_responses_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="responses.jsonl SHA-256 mismatch"):
        prepare_dataset(config)
    config["run"]["source_responses_sha256"] = responses_digest
    responses_path.unlink()
    with pytest.raises(FileNotFoundError, match="responses.jsonl is missing"):
        prepare_dataset(config)


def test_prepared_golden_contract_rejects_stale_split(tmp_path):
    run_dir = tmp_path / "source"
    genui_path = run_dir / "genui.jsonl"
    responses_path = run_dir / "responses.jsonl"
    _write_jsonl(
        genui_path,
        [
            {
                "response_id": "r1",
                "source_id": "source-1",
                "response_text": "Show a status card.",
                "genui_json": {
                    "root": "root",
                    "state": {},
                    "elements": {
                        "root": {
                            "type": "Text",
                            "props": {"text": "Ready"},
                            "children": [],
                        }
                    },
                },
            }
        ],
    )
    _write_jsonl(
        responses_path,
        [{"response_id": "r1", "response_text": "Show a status card."}],
    )
    genui_sha = hashlib.sha256(genui_path.read_bytes()).hexdigest()
    responses_sha = hashlib.sha256(responses_path.read_bytes()).hexdigest()
    output_dir = tmp_path / "prepared"
    config = {
        "run": {
            "source_run_dir": str(run_dir),
            "source_genui_sha256": genui_sha,
            "source_responses_sha256": responses_sha,
            "output_dir": str(output_dir),
        },
        "filters": {
            "required_accepted_rows": 1,
            "require_exact_accepted_rows": True,
            "require_unique_source_ids": True,
        },
        "split": {"train": 1.0, "val": 0.0, "test": 0.0},
    }
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    prepare_dataset(config, config_path=config_path)
    split = output_dir / "all.jsonl"
    split.write_text(split.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="split hash"):
        validate_prepared_golden_contract(
            split,
            dataset_config_path=config_path,
            source_genui_path=genui_path,
            source_responses_path=responses_path,
            expected_genui_sha256=genui_sha,
            expected_responses_sha256=responses_sha,
            required_rows=1,
        )


def test_fixed_golden_loader_requires_exact_unique_rows(tmp_path):
    split = tmp_path / "all.jsonl"
    _write_jsonl(split, [{"id": "a"}, {"id": "b"}])
    assert len(load_fixed_golden_rows(split, max_rows=2, required_rows=2)) == 2

    _write_jsonl(split, [{"id": "a"}, {"id": "a"}])
    with pytest.raises(ValueError, match="duplicate identities"):
        load_fixed_golden_rows(split, max_rows=2, required_rows=2)


def test_tensorboard_result_logs_scalars_and_full_json_record(tmp_path):
    calls: dict[str, object] = {"scalars": [], "texts": []}

    class Writer:
        def __init__(self, *, log_dir: str):
            calls["log_dir"] = log_dir

        def add_scalar(self, tag, value, step):
            calls["scalars"].append((tag, value, step))

        def add_text(self, tag, value, step):
            calls["texts"].append((tag, value, step))

        def flush(self):
            calls["flushed"] = True

        def close(self):
            calls["closed"] = True

    artifact = tmp_path / "aggregate.json"
    artifact.write_text('{"score": 9.5}', encoding="utf-8")
    record = log_evaluation_result(
        tmp_path / "tensorboard",
        run_id="e2b/run",
        evaluation_name="litertlm w8",
        metrics={"score": 9.5, "valid": True, "nested": {"rate": 0.75}},
        step=500,
        artifacts={"aggregate": artifact},
        metadata={"bits": 8},
        source_aggregate_path=artifact,
        writer_factory=Writer,
    )

    record_path = Path(record["record_path"])
    assert record_path.is_file()
    assert record_path.parent.name == "litertlm_w8"
    assert Path(str(calls["log_dir"])).name == "e2b_run"
    assert (
        "evaluation/litertlm_w8/nested/rate",
        0.75,
        500,
    ) in calls["scalars"]
    assert record["artifacts"]["aggregate"]["sha256"] == hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    assert calls["flushed"] is True
    assert calls["closed"] is True


def test_directory_artifact_identity_is_deterministic_and_content_bound(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "nested").mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "nested" / "weights.bin").write_bytes(b"weights-v1")

    first = artifact_identity(checkpoint)
    second = artifact_identity(checkpoint)

    assert first == second
    assert first["kind"] == "directory"
    assert first["file_count"] == 2
    assert [item["path"] for item in first["files"]] == [
        "adapter_config.json",
        "nested/weights.bin",
    ]
    assert first["hash_semantics"] == (
        "sha256_of_sorted_path_size_sha256_manifest"
    )

    (checkpoint / "nested" / "weights.bin").write_bytes(b"weights-v2")
    changed = artifact_identity(checkpoint)

    assert changed["sha256"] != first["sha256"]


def test_tensorboard_root_env_overrides_config(monkeypatch, tmp_path):
    mounted = tmp_path / "mounted-tensorboard"
    monkeypatch.setenv("A2UI_TENSORBOARD_ROOT", str(mounted))

    assert resolve_tensorboard_root("ignored") == mounted.resolve()
    assert _resolve_training_tensorboard_dir(
        {"tensorboard_root": "also-ignored"}, run_id="qat run"
    ) == mounted.resolve() / "qat_run" / "training"
    assert _resolve_training_tensorboard_dir(
        {"tensorboard_root": "also-ignored", "tensorboard_subdir": "training"},
        run_id="qat run",
    ) == mounted.resolve() / "qat_run" / "training"


def test_relative_tensorboard_root_is_repository_relative(tmp_path):
    assert resolve_tensorboard_root(
        "tensorboard", repository_root=tmp_path
    ) == tmp_path.resolve() / "tensorboard"


def test_prediction_scoring_uses_response_text_or_final_user_turn():
    row = {
        "id": "one",
        "response_text": "Authoritative response",
        "messages": [
            {"role": "user", "content": "Demo response"},
            {"role": "assistant", "content": "Demo target"},
            {
                "role": "user",
                "content": (
                    "Create A2UI Express v1 GenUI IR for this response:\n\n"
                    "Held-out response"
                ),
            },
            {"role": "assistant", "content": "Golden target"},
        ],
        "completion": "Golden target",
    }

    with pytest.raises(ValueError, match="Scoring source mismatch"):
        build_prediction_record(row, "Generated")
    row["response_text"] = "Held-out response"
    assert build_prediction_record(row, "Generated")["response_text"] == "Held-out response"
    del row["response_text"]
    assert build_prediction_record(row, "Generated")["response_text"] == (
        "Held-out response"
    )


def test_external_results_must_match_every_golden_row():
    golden = [
        {"id": "a", "response_id": "r1", "messages": [], "completion": "A"},
        {"id": "b", "response_id": "r2", "messages": [], "completion": "B"},
    ]
    predictions = merge_external_outputs(
        golden,
        [
            {"id": "b", "generated_text": "generated B", "latency_ms": 4},
            {"id": "a", "generated_text": "generated A", "latency_ms": 3},
        ],
    )
    assert [row["id"] for row in predictions] == ["a", "b"]
    assert predictions[0]["runtime"] == {"latency_ms": 3}
    assert aggregate_external_runtime_metrics(
        [
            {"id": "a", "generated_text": "A", "latency_ms": 3, "tokens": 2},
            {"id": "b", "generated_text": "B", "latency_ms": 5, "tokens": 4},
        ]
    ) == {"runtime_latency_ms_avg": 4.0, "runtime_tokens_avg": 3.0}

    with pytest.raises(ValueError, match="do not match"):
        merge_external_outputs(golden, [{"id": "a", "generated_text": "only"}])


def test_external_runner_config_requires_versioned_request_output_routing():
    valid = validate_external_runner_config(
        {
            "protocol": "a2ui_external_generation_v1",
            "command": [
                "runner",
                "--requests={requests_path}",
                "--outputs={outputs_path}",
            ],
        }
    )
    assert valid["placeholder"] is False
    assert valid["placeholders"] == ["outputs_path", "requests_path"]

    with pytest.raises(ValueError, match="missing placeholder"):
        validate_external_runner_config(
            {
                "protocol": "a2ui_external_generation_v1",
                "command": ["runner", "--requests={requests_path}"],
            }
        )
    with pytest.raises(ValueError, match="Unknown runner placeholder"):
        validate_external_runner_config(
            {
                "protocol": "a2ui_external_generation_v1",
                "command": [
                    "runner",
                    "{requests_path}",
                    "{outputs_path}",
                    "{golden_completion}",
                ],
            }
        )


def test_external_runner_protocol_does_not_expose_expected_completion(tmp_path):
    split = tmp_path / "golden.jsonl"
    _write_jsonl(
        split,
        [
            {
                "id": "one",
                "response_id": "r1",
                "prompt": "User prompt only",
                "messages": [
                    {"role": "user", "content": "User prompt only"},
                    {"role": "assistant", "content": "SAFE_FEW_SHOT_EXAMPLE"},
                    {"role": "user", "content": "Actual held-out prompt"},
                    {"role": "assistant", "content": "SECRET_GOLDEN_COMPLETION"},
                ],
                "completion": "SECRET_GOLDEN_COMPLETION",
            }
        ],
    )
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"fixture")
    runner_code = (
        "import json,sys;src,dst=sys.argv[1:];"
        "rows=[json.loads(x) for x in open(src,encoding='utf-8') if x.strip()];"
        "out=open(dst,'w',encoding='utf-8');"
        "[out.write(json.dumps(dict(id=r['id'],generated_text='<a2ui>\\n"
        "root=Text(\\\"ok\\\")\\n</a2ui>',latency_ms=2))+'\\n') for r in rows];"
        "out.close()"
    )

    manifest = run_external_generation(
        command_template=[
            sys.executable,
            "-c",
            runner_code,
            "{requests_path}",
            "{outputs_path}",
        ],
        model_path=model,
        split_path=split,
        output_dir=tmp_path / "eval",
        max_input_tokens=128,
        max_new_tokens=64,
        max_rows=1,
        required_rows=1,
    )

    request_text = Path(manifest["requests_path"]).read_text(encoding="utf-8")
    assert "SECRET_GOLDEN_COMPLETION" not in request_text
    assert "SAFE_FEW_SHOT_EXAMPLE" in request_text
    assert manifest["runtime_metrics"] == {"runtime_latency_ms_avg": 2.0}
    predictions = list(
        json.loads(line)
        for line in Path(manifest["predictions_path"])
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert predictions[0]["expected"] == "SECRET_GOLDEN_COMPLETION"


def test_external_runner_leak_guard_handles_quoted_a2ui_targets():
    target = '<a2ui>{"root":"x","state":{}}</a2ui>'
    row = {"completion": target}

    with pytest.raises(ValueError, match="refusing to leak"):
        _assert_target_not_exposed(
            row,
            {
                "prompt": "safe",
                "messages": [
                    {"role": "user", "content": f"accidental target: {target}"}
                ],
            },
        )
