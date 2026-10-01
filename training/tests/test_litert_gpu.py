from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "dataset" / "src"))

from ir_training.common.jsonl import read_jsonl, write_jsonl
from ir_training.eval import litert_gpu as runner
from ir_training.generation_policy import stop_express_completion

_read_graphics_processes = runner._nvidia_graphics_processes


@pytest.fixture(autouse=True)
def isolate_native_process_queries(monkeypatch):
    monkeypatch.setattr(runner, "_nvidia_graphics_processes", lambda: [])


class FakeGPU:
    def __init__(self, *, gpu_decode_steps_per_sync=8):
        self.gpu_decode_steps_per_sync = gpu_decode_steps_per_sync

    def get_name(self):
        return "gpu"


class FakeSession:
    def __init__(self, engine):
        self.engine, self.cancelled = engine, False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.engine.closed_sessions += 1

    def run_prefill(self, contents):
        self.engine.prompts.append(contents)

    def run_decode_async(self):
        for text in self.engine.chunks:
            yield SimpleNamespace(texts=[text])

    def cancel_process(self):
        self.cancelled = True

    def get_benchmark_info(self):
        return SimpleNamespace(last_prefill_token_count=3, last_prefill_tokens_per_second=1500.0,
                               last_decode_token_count=8, last_decode_tokens_per_second=80.0)


class FakeEngine:
    instances: ClassVar[list] = []
    chunks: ClassVar[list[str]] = ['<a2ui><Text text="</a2ui> is quoted" />', '</a2ui> trailing']
    bos_token_id = None

    def __init__(self, model_path, backend, max_num_tokens, cache_dir, **kwargs):
        self.model_path, self.backend = model_path, backend
        self.context, self.cache_dir, self.kwargs = max_num_tokens, cache_dir, kwargs
        self.prompts, self.sessions = [], []
        self.closed, self.closed_sessions = False, 0
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def tokenize(self, text):
        return [1, 2, 3]

    def detokenize(self, tokens):
        return "<bos>"

    def create_session(self, *, apply_prompt_template, sampler_config, max_output_tokens):
        session = FakeSession(self)
        session.config = (apply_prompt_template, sampler_config, max_output_tokens)
        self.sessions.append(session)
        return session


@pytest.fixture
def runtime(monkeypatch):
    FakeEngine.instances.clear()
    monkeypatch.delenv("A2UI_LITERT_ALLOWED_GPU_UUIDS", raising=False)
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _: runner.RUNTIME_VERSION)
    monkeypatch.setattr(runner, "nvidia_inventory", lambda: [{"uuid": "GPU-test", "name": "H100"}])
    monkeypatch.setattr(runner, "probe_vulkan_gpu", lambda: {"status": "passed", "usable_device_count": 1,
                                                          "webgpu_adapter_tested": False})
    return SimpleNamespace(Engine=FakeEngine, Session=FakeSession,
                           Backend=SimpleNamespace(GPU=FakeGPU),
                           ActivationDataType=SimpleNamespace(FLOAT32="float32", FLOAT16="float16"),
                           _ffi=SimpleNamespace(_get_lib=lambda: SimpleNamespace(litert_lm_engine_settings_create=lambda: None)),
                           SamplerConfig=lambda **kwargs: kwargs)


def request(model, *, id="golden_1"):
    return {
        "protocol_version": runner.PROTOCOL_VERSION, "id": id,
        "formatted_prompt": "exact formatted prompt", "prompt_sha256": runner.sha256_text("exact formatted prompt"),
        "model_path": str(model), "do_sample": False, "mtp_enabled": False,
        "max_input_tokens": 4096, "max_new_tokens": 2048, "expected_input_ids": [1, 2, 3],
    }


def evidence():
    return {"pid": 1, "devices": [{"gpu_uuid": "GPU-test", "allocated_mib": 50.0}]}


def run_worker(tmp_path, runtime, rows=None, evidence_reader=evidence):
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"fixture")
    requests, outputs = tmp_path / "requests.jsonl", tmp_path / "outputs.jsonl"
    write_jsonl(requests, rows or [request(model)])
    result = runner.run_gpu_worker(model_path=model, requests_path=requests, outputs_path=outputs,
                                  runtime=runtime, evidence_reader=evidence_reader)
    return result, list(read_jsonl(outputs)), runtime.Engine.instances[-1]


def test_preflight_reports_only_prerequisites(runtime):
    result = runner.runtime_preflight(runtime=runtime)
    assert result["runtime_version"] == "0.17.0"
    assert result["model_kernel_tested"] is False
    assert result["multi_gpu_supported"] is False
    assert result["gpu_workers"] == 1
    assert result["vulkan_compute_device_verified"] is True
    assert result["vulkan"]["webgpu_adapter_tested"] is False


def test_preflight_rejects_missing_vulkan_even_when_cuda_and_native_import_work(runtime, monkeypatch):
    def broken():
        raise RuntimeError("Vulkan loader libvulkan.so.1 could not load")

    monkeypatch.setattr(runner, "probe_vulkan_gpu", broken)
    with pytest.raises(RuntimeError, match="libvulkan.so.1"):
        runner.runtime_preflight(runtime=runtime)
    assert not FakeEngine.instances


def test_engine_creation_failure_explains_prerequisite_limit(tmp_path, runtime, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("Failed to initialize WebGPU environment: No adapters found")

    monkeypatch.setattr(runner, "runtime_preflight", lambda **_: {"status": "prerequisites_passed"})
    monkeypatch.setattr(runtime, "Engine", broken)
    with pytest.raises(RuntimeError, match="does not certify WebGPU adapter features") as error:
        run_worker(tmp_path, runtime)
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics" in str(error.value)
    assert "No adapters found" in str(error.value.__cause__)


@pytest.mark.parametrize("workers", [0, 2, 4, 8])
def test_preflight_rejects_unproven_device_sharding(runtime, workers):
    with pytest.raises(ValueError, match="no device-index selector"):
        runner.runtime_preflight(runtime=runtime, gpu_workers=workers)


def test_preflight_rejects_runtime_drift(runtime, monkeypatch):
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _: "0.16.0")
    with pytest.raises(RuntimeError, match="Install litert-lm-api==0.17.0"):
        runner.runtime_preflight(runtime=runtime)


def test_preflight_rejects_missing_token_limit_api(runtime, monkeypatch):
    monkeypatch.setattr(FakeEngine, "create_session", lambda self: None)
    with pytest.raises(RuntimeError, match="max_output_tokens"):
        runner.runtime_preflight(runtime=runtime)


def test_preflight_native_library_load_failure(runtime, monkeypatch):
    def broken():
        raise OSError("missing dependent shared library")

    monkeypatch.setattr(runtime._ffi, "_get_lib", broken)
    with pytest.raises(RuntimeError, match="native library could not load"):
        runner.runtime_preflight(runtime=runtime)


def test_raw_prefill_does_not_double_insert_native_bos(tmp_path, runtime, monkeypatch):
    monkeypatch.setattr(FakeEngine, "bos_token_id", 1)
    monkeypatch.setattr(FakeEngine, "tokenize", lambda self, text: [2, 3] if text == "source" else [99])
    row = request(tmp_path / "model.litertlm")
    row.update(formatted_prompt="<bos>source", prompt_sha256=runner.sha256_text("<bos>source"))
    _, rows, engine = run_worker(tmp_path, runtime, [row])
    assert engine.prompts == [["source"]]
    assert rows[0]["input_tokens"] == 3
    assert rows[0]["native_bos_inserted"] is True
    assert rows[0]["native_bos_token_id"] == 1


def test_raw_prefill_rejects_added_bos_not_present_in_training(tmp_path, runtime, monkeypatch):
    monkeypatch.setattr(FakeEngine, "bos_token_id", 1)
    with pytest.raises(RuntimeError, match="Native BOS parity failed"):
        run_worker(tmp_path, runtime)
    assert not FakeEngine.instances[-1].sessions


def test_raw_prompt_token_parity_gpu_selection_limits_and_incremental_output(tmp_path, runtime):
    result, rows, engine = run_worker(tmp_path, runtime)
    assert result["model_kernel_tested"] is True
    assert engine.context == 6144
    assert engine.kwargs["enable_speculative_decoding"] is False
    assert engine.prompts == [["exact formatted prompt"]]
    assert engine.sessions[0].config == (False, {"top_k": 1, "top_p": 1.0, "temperature": 0.0, "seed": 42}, 2048)
    assert engine.sessions[0].cancelled
    assert engine.closed and engine.closed_sessions == 1
    assert rows[0]["stop_reason"] == "closing_sentinel"
    assert rows[0]["raw_completion"].endswith("</a2ui> trailing")
    assert rows[0]["tokenizer_parity_passed"]
    assert rows[0]["gpu_evidence"]["devices"][0]["gpu_uuid"] == "GPU-test"
    assert rows[0]["exact_output_token_ids_available"] is False
    assert engine.kwargs["enable_benchmark"] is True
    assert engine.kwargs["activation_data_type"] == "float32"
    assert engine.backend.gpu_decode_steps_per_sync == 8
    assert rows[0]["native_last_decode_token_count"] == 8
    assert rows[0]["native_last_decode_tokens_per_second"] == 80.0
    assert rows[0]["time_to_first_token_seconds"] >= 0
    assert rows[0]["generation_seconds"] >= rows[0]["inference_seconds"] >= 0
    assert rows[0]["prefill_seconds"] >= 0 and rows[0]["decode_seconds"] >= 0
    assert rows[0]["engine_load_seconds"] >= 0


def test_stop_candidate_split_across_chunks_and_quoted_tag(tmp_path, runtime, monkeypatch):
    monkeypatch.setattr(FakeEngine, "chunks", ['<a2ui><Text text="</a2', 'ui>"/>', '</a', '2ui> extra'])
    _, rows, engine = run_worker(tmp_path, runtime)
    assert rows[0]["raw_completion"] == '<a2ui><Text text="</a2ui>"/></a2ui> extra'
    assert engine.sessions[0].cancelled
    assert rows[0]["stop_reason"] == "closing_sentinel"


def test_one_loaded_engine_is_reused_between_cases(tmp_path, runtime):
    model = tmp_path / "model.litertlm"
    _, rows, _ = run_worker(tmp_path, runtime, [request(model), request(model, id="golden_2")])
    assert len(FakeEngine.instances) == 1
    assert [row["engine_reused"] for row in rows] == [False, True]


def test_native_tokenizer_mismatch_fails_before_generation(tmp_path, runtime, monkeypatch):
    monkeypatch.setattr(FakeEngine, "tokenize", lambda self, text: [1, 99])
    with pytest.raises(RuntimeError, match="Native tokenizer parity failed"):
        run_worker(tmp_path, runtime)
    assert not FakeEngine.instances[-1].sessions
    assert FakeEngine.instances[-1].closed


def test_gpu_failure_is_not_retried_on_cpu(tmp_path, runtime):
    def no_gpu():
        raise RuntimeError("missing GPU allocation")

    with pytest.raises(RuntimeError, match="missing GPU allocation"):
        run_worker(tmp_path, runtime, evidence_reader=no_gpu)
    assert len(FakeEngine.instances) == 1
    assert FakeEngine.instances[-1].closed
    assert not list(read_jsonl(tmp_path / "outputs.jsonl"))


def test_later_case_failure_preserves_completed_rows(tmp_path, runtime):
    model = tmp_path / "model.litertlm"
    calls = []

    def second_fails():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("native GPU failure")
        return evidence()

    with pytest.raises(RuntimeError, match="native GPU failure"):
        run_worker(tmp_path, runtime, [request(model), request(model, id="golden_2")], second_fails)
    assert [row["id"] for row in read_jsonl(tmp_path / "outputs.jsonl")] == ["golden_1"]


@pytest.mark.parametrize("change", [
    {"completion": "secret target"}, {"do_sample": True}, {"mtp_enabled": "false"},
    {"max_new_tokens": 0}, {"max_input_tokens": 2}, {"expected_input_ids": [True]},
    {"formatted_prompt": "changed"}, {"id": ""}, {"protocol_version": "v2"},
])
def test_malformed_requests_fail_before_engine(tmp_path, runtime, change):
    row = request(tmp_path / "model.litertlm")
    row.update(change)
    with pytest.raises(ValueError):
        run_worker(tmp_path, runtime, [row])
    assert not FakeEngine.instances


def test_mtp_is_forwarded_to_engine(tmp_path, runtime):
    row = request(tmp_path / "model.litertlm")
    row["mtp_enabled"] = True
    _, rows, engine = run_worker(tmp_path, runtime, [row])
    assert engine.kwargs["enable_speculative_decoding"] is True
    assert rows[0]["mtp_enabled"] is True


def test_existing_output_never_overwritten(tmp_path, runtime):
    (tmp_path / "outputs.jsonl").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_worker(tmp_path, runtime)
    assert (tmp_path / "outputs.jsonl").read_text() == "preserve"
    assert not FakeEngine.instances


def test_process_gpu_evidence_uses_exact_pid_not_other_job(monkeypatch):
    monkeypatch.setattr(runner.os, "getpid", lambda: 456)
    monkeypatch.setattr(runner, "_nvidia_query", lambda *args, **kwargs: [
        ["123", "GPU-other", "12000"], ["456", "GPU-own", "2048"],
    ])
    result = runner.gpu_process_evidence()
    assert result["devices"] == [{"gpu_uuid": "GPU-own", "allocated_mib": 2048.0}]
    assert result["all_operations_gpu_verified"] is False
    monkeypatch.setattr(runner.os, "getpid", lambda: 999)
    with pytest.raises(RuntimeError, match="no GPU allocation"):
        runner.gpu_process_evidence()


@pytest.mark.parametrize("memory", ["N/A", "0", "-1", "nan", "inf"])
def test_missing_allocated_memory_rejected(monkeypatch, memory):
    monkeypatch.setattr(runner.os, "getpid", lambda: 123)
    monkeypatch.setattr(runner, "_nvidia_query", lambda *args, **kwargs: [["123", "GPU", memory]])
    with pytest.raises(RuntimeError, match="no GPU allocation"):
        runner.gpu_process_evidence()


def test_exact_graphics_context_handles_unavailable_container_memory(monkeypatch):
    monkeypatch.setattr(runner.os, "getpid", lambda: 456)
    monkeypatch.setattr(runner, "_nvidia_query", lambda *args, **kwargs: [["456", "GPU-own", "0"]])
    monkeypatch.setattr(runner, "_nvidia_graphics_processes", lambda: [
        {"pid": "123", "gpu_uuid": "GPU-other", "allocated_mib": 8000, "process_type": "G"},
        {"pid": "456", "gpu_uuid": "GPU-own", "allocated_mib": 0, "process_type": "C+G"},
    ])
    result = runner.gpu_process_evidence()
    assert result["devices"][0]["gpu_uuid"] == "GPU-own"
    assert result["devices"][0]["graphics_process_registered"] is True
    assert result["allocation_observed"] is False
    assert result["all_operations_gpu_verified"] is False


def test_graphics_xml_does_not_accept_compute_only_zero_memory(monkeypatch):
    xml = '''<nvidia_smi_log><gpu><uuid>GPU-own</uuid><processes>
      <process_info><pid>123</pid><type>C</type><used_memory>0 MiB</used_memory></process_info>
      <process_info><pid>456</pid><type>C+G</type><used_memory>0 MiB</used_memory></process_info>
    </processes></gpu></nvidia_smi_log>'''
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=xml))
    rows = _read_graphics_processes()
    assert [row["pid"] for row in rows] == ["456"]


def test_kernel_self_namespace_host_pid_can_match_nvidia(monkeypatch):
    monkeypatch.setattr(runner.os, "getpid", lambda: 7)
    monkeypatch.setattr(runner, "_read_self_status", lambda: "Name:\tpython\nNSpid:\t4712\t205\t7\n")
    monkeypatch.setattr(runner, "_nvidia_query", lambda *args, **kwargs: [
        ["8888", "GPU-other", "12000"], ["4712", "GPU-self", "2048"],
    ])
    result = runner.gpu_process_evidence()
    assert result["pid"] == 7
    assert result["nvidia_smi_pid"] == 4712
    assert result["pid_namespace_identity"]["candidate_pids"] == [4712, 205, 7]
    assert result["devices"] == [{"gpu_uuid": "GPU-self", "allocated_mib": 2048.0}]


def test_multiple_namespace_pid_matches_fail_closed(monkeypatch):
    monkeypatch.setattr(runner.os, "getpid", lambda: 7)
    monkeypatch.setattr(runner, "_read_self_status", lambda: "NSpid:\t4712\t7\n")
    monkeypatch.setattr(runner, "_nvidia_query", lambda *args, **kwargs: [
        ["7", "GPU-collision", "12000"], ["4712", "GPU-self", "2048"],
    ])
    with pytest.raises(RuntimeError, match="Ambiguous NVIDIA PID-namespace evidence"):
        runner.gpu_process_evidence()


@pytest.mark.parametrize("status", [
    "Name:\tpython\n", "NSpid:\n", "NSpid:\tx7 7\n", "NSpid:\t123 9\n",
    "NSpid:\t0 7\n", "NSpid:\t-1 7\n", "NSpid:\t123 7\nNSpid:\t456 7\n",
])
def test_unverified_namespace_values_never_expand_self_candidates(monkeypatch, status):
    monkeypatch.setattr(runner.os, "getpid", lambda: 7)
    monkeypatch.setattr(runner, "_read_self_status", lambda: status)
    assert runner.self_pid_identity()["candidate_pids"] == [7]


def test_inaccessible_proc_keeps_exact_pid_and_actionable_diagnostic(monkeypatch):
    monkeypatch.setattr(runner.os, "getpid", lambda: 7)

    def denied():
        raise PermissionError("procfs unavailable")

    monkeypatch.setattr(runner, "_read_self_status", denied)
    assert runner.self_pid_identity()["candidate_pids"] == [7]
    monkeypatch.setattr(runner, "_nvidia_query", lambda *args, **kwargs: [["4712", "GPU-other", "2048"]])
    with pytest.raises(RuntimeError, match="administrator-approved host-PID-visible runtime container"):
        runner.gpu_process_evidence()


def test_observed_native_device_must_be_allocated(monkeypatch):
    monkeypatch.setenv("A2UI_LITERT_ALLOWED_GPU_UUIDS", '["GPU-allocated"]')
    with pytest.raises(RuntimeError, match="unallocated GPU"):
        runner.validate_gpu_allocation(evidence())
    runner.validate_gpu_allocation({"devices": [{"gpu_uuid": "GPU-allocated"}]})


def test_compute_context_does_not_hide_unallocated_graphics_context(monkeypatch):
    monkeypatch.setenv("A2UI_LITERT_ALLOWED_GPU_UUIDS", '["GPU-allocated"]')
    monkeypatch.setattr(runner, "self_pid_identity", lambda: {"local_pid": 7, "candidate_pids": [7]})
    monkeypatch.setattr(runner, "_nvidia_query", lambda *args, **kwargs: [["7", "GPU-allocated", "100"]])
    monkeypatch.setattr(runner, "_nvidia_graphics_processes", lambda: [
        {"pid": "7", "gpu_uuid": "GPU-other", "allocated_mib": 0, "process_type": "G"}])
    with pytest.raises(RuntimeError, match="unallocated GPU"):
        runner.gpu_process_evidence()


def test_compute_and_graphics_reports_merge_one_device(monkeypatch):
    monkeypatch.delenv("A2UI_LITERT_ALLOWED_GPU_UUIDS", raising=False)
    monkeypatch.setattr(runner, "self_pid_identity", lambda: {"local_pid": 7, "candidate_pids": [7]})
    monkeypatch.setattr(runner, "_nvidia_query", lambda *args, **kwargs: [["7", "GPU-test", "100"]])
    monkeypatch.setattr(runner, "_nvidia_graphics_processes", lambda: [
        {"pid": "7", "gpu_uuid": "GPU-test", "allocated_mib": 0, "process_type": "C+G"}])
    result = runner.gpu_process_evidence()
    assert len(result["devices"]) == 1
    assert result["devices"][0]["graphics_process_registered"] is True
    assert result["devices"][0]["allocated_mib"] == 100
    assert result["devices"][0]["allocation_size_verified"] is True


@pytest.mark.parametrize("value", ["", "0,1", "[]", '["GPU-a", "GPU-a"]', '["0"]'])
def test_invalid_device_allocation_environment_rejected(monkeypatch, value):
    monkeypatch.setenv("A2UI_LITERT_ALLOWED_GPU_UUIDS", value)
    with pytest.raises(ValueError, match="A2UI_LITERT_ALLOWED_GPU_UUIDS"):
        runner.allowed_gpu_uuids()


def test_preflight_checks_allowed_inventory(runtime, monkeypatch):
    monkeypatch.setenv("A2UI_LITERT_ALLOWED_GPU_UUIDS", '["GPU-missing"]')
    with pytest.raises(RuntimeError, match="absent from the runtime inventory"):
        runner.runtime_preflight(runtime=runtime)


def test_worker_checks_allocation_before_decode(tmp_path, runtime, monkeypatch):
    monkeypatch.setenv("A2UI_LITERT_ALLOWED_GPU_UUIDS", '["GPU-test"]')
    with pytest.raises(RuntimeError, match="unallocated GPU"):
        run_worker(tmp_path, runtime, evidence_reader=lambda: {"devices": [{"gpu_uuid": "GPU-other"}]})
    assert not list(read_jsonl(tmp_path / "outputs.jsonl"))


def test_bounded_worker_tees_native_logs_and_finishes(tmp_path):
    command = [sys.executable, "-u", "-c", "print('native load log'); print('A2UI_LITERT_EVENT {\"phase\":\"finished\"}')"]
    runner.run_bounded_worker(command, log_path=tmp_path / "runner.log", timeout_seconds=10)
    assert "native load log" in (tmp_path / "runner.log").read_text()


def test_bounded_worker_hard_case_timeout(tmp_path):
    code = "import time; print('A2UI_LITERT_EVENT {\"phase\":\"case_start\",\"id\":\"golden_1\"}',flush=True); time.sleep(30)"
    with pytest.raises(TimeoutError, match="case=golden_1"):
        runner.run_bounded_worker([sys.executable, "-u", "-c", code],
                                  log_path=tmp_path / "runner.log", case_timeout_seconds=0.2, timeout_seconds=10)
    assert "golden_1" in (tmp_path / "runner.log").read_text()


def test_bounded_worker_refuses_zero_exit_without_finish(tmp_path):
    with pytest.raises(RuntimeError, match="completion event"):
        runner.run_bounded_worker([sys.executable, "-c", "print('not finished')"], log_path=tmp_path / "runner.log")


def test_bounded_worker_cleans_child_when_log_reader_cannot_start(tmp_path, monkeypatch):
    processes = []
    original = runner.subprocess.Popen

    def launch(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    def broken_thread(self):
        raise RuntimeError("cannot create reader thread")

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    monkeypatch.setattr(runner.threading.Thread, "start", broken_thread)
    with pytest.raises(RuntimeError, match="cannot create reader thread"):
        runner.run_bounded_worker([sys.executable, "-c", "import time; time.sleep(30)"],
                                  log_path=tmp_path / "runner.log")
    assert len(processes) == 1
    assert processes[0].poll() is not None


def test_build_requests_preserves_training_template_and_never_targets(tmp_path, monkeypatch):
    import ir_training.models.registry

    fake_adapter = SimpleNamespace(
        load_tokenizer=lambda: lambda *args, **kwargs: {"input_ids": [1, 2, 3]},
        format_example=lambda row, **kwargs: "exact formatted prompt",
    )
    monkeypatch.setattr(ir_training.models.registry, "create_adapter", lambda cfg: fake_adapter)
    rows = [{"id": "golden", "prompt": "source", "completion": "secret target",
             "messages": [{"role": "user", "content": "source"}, {"role": "assistant", "content": "secret target"}]}]
    requests = runner.build_gpu_requests(rows, model=tmp_path / "model.litertlm", model_config={},
                                         max_input_tokens=4096, max_new_tokens=2048, mtp_enabled=False)
    assert "secret target" not in json.dumps(requests)
    assert requests[0]["expected_input_ids"] == [1, 2, 3]
    assert requests[0]["formatted_prompt"] == "exact formatted prompt"


def test_builtin_manifest_merge_keeps_golden_occurrence_order(tmp_path, monkeypatch):
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"fixture")
    split = tmp_path / "golden.jsonl"
    rows = [{"id": name, "prompt": name} for name in ("golden_2", "golden_1")]
    write_jsonl(split, rows)
    monkeypatch.setattr(runner, "build_gpu_requests", lambda *args, **kwargs: [])

    def finish(command, **kwargs):
        outputs = Path(command[command.index("--outputs") + 1])
        write_jsonl(outputs, [
            {"id": row["id"], "generated_text": "<a2ui></a2ui>", "engine_backend": "gpu",
             "gpu_evidence": evidence(), "tokenizer_parity_passed": True} for row in rows[::-1]
        ])

    monkeypatch.setattr(runner, "run_bounded_worker", finish)
    manifest = runner.run_litert_gpu_generation(model_path=model, split_path=split,
                                                output_dir=tmp_path / "out", model_config={}, required_rows=2)
    assert [row["id"] for row in read_jsonl(manifest["predictions_path"])] == [row["id"] for row in rows]
    assert manifest["gpu_execution"]["gpu_uuids"] == ["GPU-test"]
    assert manifest["gpu_execution"]["allocation_observed"] is True


def test_native_cache_binds_model_bytes_and_runtime_not_package_basename(tmp_path, monkeypatch):
    split = tmp_path / "golden.jsonl"
    write_jsonl(split, [{"id": "golden_1", "prompt": "source"}])
    monkeypatch.setattr(runner, "build_gpu_requests", lambda *args, **kwargs: [])
    cache_paths = []

    def finish(command, **kwargs):
        cache_paths.append(Path(command[command.index("--cache-dir") + 1]))
        outputs = Path(command[command.index("--outputs") + 1])
        write_jsonl(outputs, [{"id": "golden_1", "generated_text": "<a2ui></a2ui>",
                               "engine_backend": "gpu", "gpu_evidence": evidence(),
                               "tokenizer_parity_passed": True}])

    monkeypatch.setattr(runner, "run_bounded_worker", finish)
    base = tmp_path / "compiled_cache"
    variants = [(b"float32 package", "float32"), (b"int4 package", "float32"),
                (b"float32 package", "float32"), (b"float32 package", "float16")]
    for index, (content, activation_dtype) in enumerate(variants):
        model = tmp_path / f"variant_{index}" / "model.litertlm"
        model.parent.mkdir()
        model.write_bytes(content)
        result = runner.run_litert_gpu_generation(model_path=model, split_path=split,
                                                  output_dir=tmp_path / f"evaluation_{index}",
                                                  model_config={}, required_rows=1, cache_dir=base,
                                                  activation_dtype=activation_dtype)
        assert Path(result["runtime_cache_dir"]) == base / runner.RUNTIME_VERSION / result["model_sha256"] / activation_dtype
        assert Path(result["runtime_cache_dir"]).is_dir()
    assert cache_paths[0] != cache_paths[1]
    assert cache_paths[0] == cache_paths[2]
    assert cache_paths[0] != cache_paths[3]


def test_cancelled_decode_is_drained_before_session_close(tmp_path, runtime, monkeypatch):
    completed = []

    def asynchronous_decode(self):
        yield SimpleNamespace(texts=['<a2ui><Text text="ok" /></a2ui>'])
        # A native callback queued before cancellation can still arrive.
        yield SimpleNamespace(texts=[' queued after cancellation'])
        completed.append(self)

    def close_after_final_callback(self, *_):
        assert self in completed, "Session freed while native decode callbacks are still active"
        self.engine.closed_sessions += 1

    monkeypatch.setattr(FakeSession, "run_decode_async", asynchronous_decode)
    monkeypatch.setattr(FakeSession, "__exit__", close_after_final_callback)
    _, rows, engine = run_worker(tmp_path, runtime)
    assert engine.sessions[0].cancelled
    assert engine.closed_sessions == 1
    assert rows[0]["raw_completion"].endswith("queued after cancellation")
    assert stop_express_completion(rows[0]["raw_completion"]).endswith("</a2ui>")
