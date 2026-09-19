from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ir_training.eval.android_native_quality import (
    COHORTS,
    AdbNativeBatchRunner,
    build_bound_requests,
    comparison_rows,
    device_requests,
    parse_delegation_evidence,
    relocate_run_path,
    require_nonoverlapping_paths,
    scope_logcat_to_probe_pid,
    validate_native_outputs,
)


def _binding(tmp_path: Path) -> dict:
    best = tmp_path / "best"
    best.mkdir()
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"model")
    config = {"model": {"tokenizer_source": "old"}}
    cohorts = {}
    for name, count in COHORTS.items():
        rows = [{"id": f"row-{index}"} for index in range(count)]
        predictions = []
        for row in rows:
            prompt = f"<bos>formatted {name} {row['id']}"
            ids = [1, len(prompt)]
            predictions.append(
                {
                    "id": row["id"],
                    "runtime": {
                        "prompt_sha256": __import__("hashlib")
                        .sha256(prompt.encode())
                        .hexdigest(),
                        "input_token_ids_sha256": __import__("hashlib")
                        .sha256(json.dumps(ids).encode())
                        .hexdigest(),
                        "generation_policy": {
                            "version": "a2ui-envelope-v1",
                            "eos_token_ids": [1, 2],
                            "stop_strings": ["</a2ui>"],
                            "stop_scope": "generated_tokens_outside_quoted_literals",
                            "max_new_tokens": 8,
                            "add_special_tokens": False,
                        },
                    },
                }
            )
        cohorts[name] = {
            "rows": rows,
            "count": count,
            "hf_predictions": predictions,
            "prepared_contract": {
                "tokenizer": {
                    "chat_template_sha256": "template",
                    "vocabulary_sha256": "vocab",
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                "prompt_scaffolds_sha256": "scaffold",
            },
            "hf_result": {"row_count": count, "aggregate": {}},
        }
        for prediction in predictions:
            policy = prediction["runtime"]["generation_policy"]
            prediction["runtime"]["generation_policy_sha256"] = (
                __import__("hashlib")
                .sha256(json.dumps(policy, sort_keys=True).encode())
                .hexdigest()
            )
    return {
        "config": config,
        "paths": {"best_checkpoint": best, "litertlm": model},
        "max_input_tokens": 16,
        "max_new_tokens": 8,
        "cohorts": cohorts,
    }


def test_run_and_output_must_be_disjoint_and_fresh(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ValueError, match="disjoint"):
        require_nonoverlapping_paths(run, run / "reports")
    with pytest.raises(ValueError, match="disjoint"):
        require_nonoverlapping_paths(run, tmp_path)
    output = tmp_path / "fresh"
    assert require_nonoverlapping_paths(run, output) == (
        run.resolve(),
        output.resolve(),
    )
    output.mkdir()
    with pytest.raises(FileExistsError, match="fresh"):
        require_nonoverlapping_paths(run, output)


def test_receipt_path_relocation_accepts_posix_run_on_windows_host(tmp_path):
    current = tmp_path / "copied_run"
    current.mkdir()
    assert (
        relocate_run_path(
            "/mnt/training/run/prepared/golden32.jsonl",
            original_root="/mnt/training/run",
            current_root=current,
        )
        == (current / "prepared" / "golden32.jsonl").resolve()
    )
    with pytest.raises(ValueError, match="outside"):
        relocate_run_path(
            "/mnt/other/golden32.jsonl",
            original_root="/mnt/training/run",
            current_root=current,
        )


def test_absence_requires_positive_sentinel_not_failed_adb_return_code():
    seen = []

    def denied(command, **_kwargs):
        seen.append(command)
        return subprocess.CompletedProcess([], 1, stdout="", stderr="denied")

    runner = AdbNativeBatchRunner(adb="adb", serial="SERIAL", command_runner=denied)
    with pytest.raises(RuntimeError, match="positively prove"):
        runner._require_absent("/data/local/tmp/collision")
    assert seen == [
        [
            "adb",
            "-s",
            "SERIAL",
            "shell",
            "sh -c 'test ! -e /data/local/tmp/collision && printf A2UI_ABSENT'",
        ]
    ]


def test_native_runner_never_clears_or_dumps_global_logcat():
    source = inspect.getsource(AdbNativeBatchRunner.run)
    assert '"logcat", "-c"' not in source
    assert '"logcat", "-d"' not in source
    assert "_start_scoped_logcat" in source


def test_bound_requests_match_hf_prompt_and_token_hashes_and_namespace_ids(
    tmp_path, monkeypatch
):
    binding = _binding(tmp_path)

    def fake_build(rows, **kwargs):
        cohort = next(
            name
            for name, details in binding["cohorts"].items()
            if details["rows"] is rows
        )
        result = []
        for row in rows:
            prompt = f"<bos>formatted {cohort} {row['id']}"
            result.append(
                {
                    "id": row["id"],
                    "formatted_prompt": prompt,
                    "messages": [{"role": "user", "content": prompt}],
                    "prompt_sha256": __import__("hashlib")
                    .sha256(prompt.encode())
                    .hexdigest(),
                    "expected_input_ids": [1, len(prompt)],
                    "max_input_tokens": 16,
                    "max_new_tokens": 8,
                }
            )
        return result

    from ir_training.eval import litert_gpu

    monkeypatch.setattr(litert_gpu, "build_gpu_requests", fake_build)
    requests = build_bound_requests(binding)

    assert len(requests) == 117
    assert len({row["id"] for row in requests}) == 117
    assert requests[0]["id"].startswith("golden32::")
    assert requests[32]["id"].startswith("golden35::")
    assert requests[67]["id"].startswith("bixby50::")
    assert requests[0]["host_input_token_ids"] == [
        1,
        len(requests[0]["formatted_prompt"]),
    ]
    assert requests[0]["native_input_token_ids"] is None
    assert requests[0]["native_token_ids_available"] is False


def test_device_requests_exclude_hf_tokens_and_any_reference_target():
    request = {
        "id": "golden32::row-1",
        "formatted_prompt": "prompt",
        "prompt_sha256": "a" * 64,
        "messages": [{"role": "user", "content": "prompt"}],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_input_tokens": 10,
        "max_new_tokens": 4,
        "host_input_token_ids": [1, 2],
        "completion": "must not leave host",
    }
    staged = device_requests([request])
    assert staged == [
        {
            "id": "golden32::row-1",
            "formatted_prompt": "prompt",
            "prompt_sha256": "a" * 64,
            "messages": [{"role": "user", "content": "prompt"}],
            "chat_template_kwargs": {"enable_thinking": False},
            "max_input_tokens": 10,
            "max_new_tokens": 4,
        }
    ]


def test_native_output_requires_exact_order_gpu_delegation_and_records_stopped_text():
    requests = [
        {
            "id": "golden32::one",
            "cohort": "golden32",
            "formatted_prompt": "prompt",
            "prompt_sha256": "a" * 64,
            "max_new_tokens": 8,
            "host_input_token_ids": [1],
            "hf_generation_policy": {},
            "hf_generation_policy_sha256": "d" * 64,
            "host_input_token_ids_sha256": "b" * 64,
            "native_token_ids_unavailable_reason": "unsupported",
            "chat_template_sha256": "template",
            "tokenizer_vocabulary_sha256": "vocab",
            "chat_template_kwargs": {},
            "prompt_scaffolds_sha256": "scaffold",
        }
    ]
    outputs = [
        {
            "id": "golden32::one",
            "prompt_sha256": "a" * 64,
            "prompt_transport_verified": True,
            "requested_backend": "GPU",
            "mtp_enabled": False,
            "runtime_rendered_prompt_sha256": "a" * 64,
            "runtime_template_prompt_verified": True,
            "raw_generated_text": '<a2ui>\nroot=Text("ok")\n</a2ui> trailing',
            "native_output_token_count": 4,
            "stop_reason": "closing_sentinel",
            "sampler": {
                "do_sample": False,
                "top_k": 1,
                "top_p": 1.0,
                "temperature": 0.0,
                "seed": 42,
            },
            "stop_policy_version": "a2ui-envelope-v1",
            "raw_output_scope": "chunks_actually_returned_by_runtime; continuation_after_stop_unobserved",
        }
    ]
    manifest = {
        "success": True,
        "completed_cases": 1,
        "requested_backend": "GPU",
        "mtp_enabled": False,
        "raw_session": False,
        "conversation_template_applied": True,
        "runtime_rendered_prompt_required": True,
        "engine_instance_count": 1,
        "fresh_conversation_per_case": True,
        "stop_policy_version": "a2ui-envelope-v1",
        "sampler": {
            "do_sample": False,
            "top_k": 1,
            "top_p": 1.0,
            "temperature": 0.0,
            "seed": 42,
        },
    }
    delegation = {"all_gpu_subgraphs_fully_delegated": True}
    rows = validate_native_outputs(requests, outputs, manifest, delegation)
    assert rows[0]["serving_stopped_text"].endswith("</a2ui>")
    assert rows[0]["native_output_token_ids"] is None

    with pytest.raises(ValueError, match="full GPU delegation"):
        validate_native_outputs(
            requests, outputs, manifest, {"all_gpu_subgraphs_fully_delegated": False}
        )
    with pytest.raises(ValueError, match="exactly cover"):
        validate_native_outputs(requests, [], manifest, delegation)


def test_delegation_parser_reuses_parity_log_grammar():
    log = (
        "Replacing 42 out of 42 node(s) with delegate (LITERT_CL) node, "
        "yielding 1 partitions for subgraph 0 (decode)"
    )
    evidence = parse_delegation_evidence(log)
    assert evidence["gpu_delegation_count"] == 1
    assert evidence["all_gpu_subgraphs_fully_delegated"] is True


def test_logcat_delegation_is_scoped_to_instrumentation_pid():
    log = (
        "09-19 12:00:00.000  111  112 I Other: Replacing 1 out of 1 node(s) with delegate (LITERT_CL) node, yielding 1 partitions for subgraph 0 (wrong)\n"
        '09-19 12:00:00.010  222  223 I OfficialMobileNative: a2ui_native_event={"phase":"engine_ready"}\n'
        "09-19 12:00:00.020  222  224 I LiteRt: Replacing 42 out of 42 node(s) with delegate (LITERT_CL) node, yielding 1 partitions for subgraph 0 (decode)"
    )
    scoped, pids = scope_logcat_to_probe_pid(log)
    assert pids == [222]
    assert "wrong" not in scoped
    assert "decode" in scoped


def test_comparison_keeps_checkpoint_and_native_columns_separate(tmp_path):
    binding = _binding(tmp_path)
    native = {}
    for index, (name, details) in enumerate(binding["cohorts"].items()):
        details["hf_result"]["aggregate"] = {
            "generation_reward_v5_4_avg": 80 + index,
            "schema_valid_strict_rate": 0.9,
            "unique_source_generation_reward_v5_4_avg": 79,
        }
        native[name] = {
            "generation_reward_v5_4_avg": 70 + index,
            "schema_valid_strict_rate": 0.8,
            "unique_source_generation_reward_v5_4_avg": 69,
        }
    rows = comparison_rows(binding, native)
    assert rows[0]["checkpoint_generation_reward_v5_4_avg"] == 80
    assert rows[0]["native_generation_reward_v5_4_avg"] == 70
    assert rows[0]["checkpoint_selector"] == 79
    assert rows[1]["checkpoint_selector"] is None
    assert rows[2]["reference_available"] is False
