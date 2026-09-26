"""CPU-only checks for the Muse Stage 3 profile and HTTP contract."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
from llm.base import ModelSpec  # noqa: E402
from llm.local_adapter import LocalAdapter  # noqa: E402
from utils.config import load_yaml  # noqa: E402
from llm.factory import load_model_specs  # noqa: E402
from main import _stage3_input_run_paths  # noqa: E402

SCRIPT = DATASET_ROOT / "scripts" / "run_muse_glimmer_stage3.py"
module_spec = importlib.util.spec_from_file_location("muse_stage3_launcher", SCRIPT)
muse = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(muse)


@pytest.mark.parametrize("gpus,tp,replicas", [(4, 1, 4), (8, 1, 8), (4, 2, 2), (8, 2, 4)])
def test_plan_groups_gpus_and_enables_muse_dflash(gpus, tp, replicas):
    args = muse.parse_args(["plan", "--gpus", str(gpus), "--tp", str(tp)])
    assert len(muse.gpu_groups(args)) == replicas
    assert len(muse.endpoints(args)) == replicas
    command = muse.server_command(args, 0)
    assert command[command.index("--tp-size") + 1] == str(tp)
    assert command[command.index("--reasoning-parser") + 1] == "muse"
    assert command[command.index("--speculative-algorithm") + 1] == "DFLASH"
    assert command[command.index("--speculative-draft-model-path") + 1] == muse.DRAFT_ID
    assert command[command.index("--speculative-dflash-block-size") + 1] == "16"
    assert "--language-model-only" in command


def test_invalid_layout_or_context_fails_early():
    with pytest.raises(SystemExit):
        muse.parse_args(["plan", "--gpus", "4", "--tp", "3"])
    with pytest.raises(SystemExit):
        muse.parse_args(["plan", "--gpus", "4", "--context-length", "8192",
                         "--output-tokens", "8192"])


def test_gpu_ids_reject_duplicates():
    args = muse.parse_args(["plan", "--gpus", "4", "--gpu-ids", "0,0,1,2"])
    with pytest.raises(ValueError, match="distinct"):
        muse.gpu_groups(args)


def test_context_and_client_reasoning_controls_ignore_stale_gemma_settings(monkeypatch):
    monkeypatch.setenv("LOCAL_VLLM_ENABLE_THINKING", "0")
    monkeypatch.setenv("LOCAL_VLLM_SEND_CHAT_TEMPLATE_KWARGS", "0")
    monkeypatch.setenv("LOCAL_VLLM_ENDPOINTS", "http://stale:8000/v1/chat/completions")
    args = muse.parse_args(["plan", "--gpus", "8", "--tp", "2", "--context-length", "32768",
                            "--output-tokens", "12288", "--prompt-tokens", "30000"])
    env = muse.client_env(args)
    assert env["LOCAL_VLLM_ENABLE_THINKING"] == "1"
    assert env["LOCAL_VLLM_SEND_CHAT_TEMPLATE_KWARGS"] == "1"
    assert env["LOCAL_VLLM_REASONING_STRENGTH"] == "high"
    assert env["LOCAL_STAGE3_PROMPT_MAX_TOKENS"] == str(32768 - 12288 - 512)
    assert env["LOCAL_VLLM_MIN_RETRY_OUTPUT_TOKENS"] == "12288"
    assert env["LOCAL_VLLM_BATCH_PARALLELISM"] == "32"
    assert "stale" not in env["LOCAL_VLLM_ENDPOINTS"]


def test_registered_muse_model_is_stage3_local_http():
    specs = load_model_specs(load_yaml(DATASET_ROOT / "configs" / "models.yaml"))
    spec = next(item for item in specs if item.name == muse.MODEL_CONFIG)
    assert spec.model == muse.MODEL_ID
    assert spec.provider == "local" and not spec.supports_json_mode
    assert spec.endpoint.endswith(":30000/v1/chat/completions")


def test_stage3_can_read_source_run_and_write_to_distinct_output(tmp_path):
    source = tmp_path / "dataset_v1"
    source.mkdir()
    (source / "queries.jsonl").write_text("{}\n", encoding="utf-8")
    (source / "responses.jsonl").write_text("{}\n", encoding="utf-8")
    paths = _stage3_input_run_paths(tmp_path, "muse_output", "dataset_v1", "artifacts")
    assert paths.queries_path.parent == source
    assert paths.responses_path.parent == source
    assert not (tmp_path / "muse_output").exists()
    with pytest.raises(SystemExit, match="missing queries.jsonl"):
        _stage3_input_run_paths(tmp_path, "muse_output", "missing", "artifacts")


def test_launcher_passes_distinct_source_run_to_stage3():
    args = muse.parse_args(["generate", "--gpus", "4", "--run-id", "muse_output",
                            "--source-run-id", "dataset_v1"])
    command = muse.stage3_command(args, 100)
    assert command[command.index("--run_id") + 1] == "muse_output"
    assert command[command.index("--stage3_source_run_id") + 1] == "dataset_v1"
    assert command[command.index("--max_genui_total") + 1] == "100"


def test_cycle_requires_supplied_size_and_total():
    with pytest.raises(SystemExit):
        muse.parse_args(["cycle", "--gpus", "4", "--run-id", "muse", "--total", "1000"])
    with pytest.raises(SystemExit):
        muse.parse_args(["cycle", "--gpus", "4", "--run-id", "muse", "--cycle-size", "1000"])
    with pytest.raises(SystemExit):
        muse.parse_args(["cycle", "--gpus", "4", "--run-id", "muse", "--total", "1000",
                         "--cycle-size", "1000", "--source-run-id", "old"])


def test_cycle_stage_batches_scale_with_eight_gpu_replicas():
    args = muse.parse_args(["cycle", "--gpus", "8", "--run-id", "muse",
                            "--cycle-size", "1000", "--total", "10000"])
    stage2 = muse.stage_command(args, 2, 1000)
    stage3 = muse.stage_command(args, 3, 1000)
    assert muse.client_env(args)["A2UI_STAGE1_INTENT_BATCH_SIZE"] == "64"
    assert stage2[stage2.index("--stage2_batch_size") + 1] == "64"
    assert stage3[stage3.index("--genui_batch_size") + 1] == "64"


def test_cycle_keeps_each_stage_output_budget_without_global_clamping():
    args = muse.parse_args(["cycle", "--gpus", "4", "--run-id", "muse",
                            "--cycle-size", "1000", "--total", "1000",
                            "--query-output-tokens", "16000", "--response-output-tokens", "14000"])
    for stage, budget in ((1, "16000"), (2, "14000"), (3, "12288")):
        env = muse.client_env(args, stage)
        assert env["LOCAL_VLLM_MAX_OUTPUT_TOKENS"] == budget
        assert env["LOCAL_VLLM_MIN_RETRY_OUTPUT_TOKENS"] == budget
    # Stage 3-only runs do not need room for the unused Stage 1/2 defaults.
    muse.parse_args(["generate", "--gpus", "4", "--run-id", "muse",
                     "--context-length", "8192", "--output-tokens", "4096"])


def cyclic_repo(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    (dataset / "configs").mkdir(parents=True)
    (dataset / "configs" / "run.yaml").write_text(
        "run:\n  output_dir: data/runs\n  intents_file: intents.info\n", encoding="utf-8")
    (dataset / "intents.info").write_text("weather\ntravel\n", encoding="utf-8")
    run = dataset / "data" / "runs" / "muse"
    run.mkdir(parents=True)
    monkeypatch.setattr(muse, "REPO_ROOT", tmp_path)
    return run


def mock_cycle_runner(run, calls):
    def fake_call(command, cwd, env):
        stage = int(command[command.index("--stage") + 1])
        quota_flag = {1: "--max_queries_total", 2: "--max_responses_total", 3: "--max_genui_total"}[stage]
        amount = int(command[command.index(quota_flag) + 1])
        assert env["LOCAL_VLLM_REASONING_STRENGTH"] == "high"
        assert env["A2UI_QUERY_MAX_TOKENS"] == "8192"
        assert env["A2UI_RESPONSE_MAX_TOKENS"] == "8192"
        assert env["LOCAL_VLLM_MAX_OUTPUT_TOKENS"] == ("12288" if stage == 3 else "8192")
        path = run / {1: "queries.jsonl", 2: "responses.jsonl", 3: "genui.jsonl"}[stage]
        with path.open("a", encoding="utf-8") as handle:
            for _ in range(amount):
                handle.write(json.dumps({"gen": {"model": muse.MODEL_ID}}) + "\n")
        calls.append((stage, amount, command))
        return 0
    return fake_call


def test_cycle_generates_exact_configured_chunks_in_stage_order(tmp_path, monkeypatch):
    run = cyclic_repo(tmp_path, monkeypatch)
    calls = []
    probes = []
    monkeypatch.setattr(muse, "probe", lambda args: probes.append(args.run_id))
    monkeypatch.setattr(muse.subprocess, "call", mock_cycle_runner(run, calls))
    args = muse.parse_args(["cycle", "--gpus", "4", "--run-id", "muse",
                            "--cycle-size", "2", "--total", "5"])
    muse.cycle(args)
    assert [(stage, amount) for stage, amount, _ in calls] == [
        (1, 2), (2, 2), (3, 2), (1, 2), (2, 2), (3, 2), (1, 1), (2, 1), (3, 1)]
    assert calls[0][2][calls[0][2].index("--k_queries_per_intent") + 1] == "3"
    assert probes == ["muse"]
    assert [muse.count_records(run / name, require_muse=True) for name in
            ("queries.jsonl", "responses.jsonl", "genui.jsonl")] == [5, 5, 5]


def test_cycle_resumes_incomplete_chunk_before_advancing(tmp_path, monkeypatch):
    run = cyclic_repo(tmp_path, monkeypatch)
    (run / "queries.jsonl").write_text(
        (json.dumps({"gen": {"model": muse.MODEL_ID}}) + "\n") * 2, encoding="utf-8")
    (run / "responses.jsonl").write_text(
        json.dumps({"gen": {"model": muse.MODEL_ID}}) + "\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(muse, "probe", lambda args: None)
    monkeypatch.setattr(muse.subprocess, "call", mock_cycle_runner(run, calls))
    args = muse.parse_args(["cycle", "--gpus", "4", "--run-id", "muse",
                            "--cycle-size", "2", "--total", "3"])
    muse.cycle(args)
    assert [(stage, amount) for stage, amount, _ in calls] == [
        (2, 1), (3, 2), (1, 1), (2, 1), (3, 1)]


def test_cycle_fails_if_stage_returns_without_new_records(tmp_path, monkeypatch):
    cyclic_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(muse, "probe", lambda args: None)
    monkeypatch.setattr(muse.subprocess, "call", lambda *args, **kwargs: 0)
    args = muse.parse_args(["cycle", "--gpus", "4", "--run-id", "muse",
                            "--cycle-size", "2", "--total", "4"])
    with pytest.raises(RuntimeError, match="made no progress"):
        muse.cycle(args)


def test_cycle_rejects_mixed_model_before_probe_or_write(tmp_path, monkeypatch):
    run = cyclic_repo(tmp_path, monkeypatch)
    (run / "responses.jsonl").write_text(
        '{"gen":{"model":"google/gemma-4-31b-it"}}\n', encoding="utf-8")
    monkeypatch.setattr(muse, "probe", lambda args: pytest.fail("mixed run should not be probed"))
    monkeypatch.setattr(muse.subprocess, "call", lambda *args, **kwargs: pytest.fail("should not generate"))
    args = muse.parse_args(["cycle", "--gpus", "4", "--run-id", "muse",
                            "--cycle-size", "2", "--total", "4"])
    with pytest.raises(RuntimeError, match="separate --run-id"):
        muse.cycle(args)


def test_cycle_requires_one_response_and_ui_per_query(tmp_path, monkeypatch):
    cyclic_repo(tmp_path, monkeypatch)
    (tmp_path / "dataset" / "configs" / "run.yaml").write_text(
        "run:\n  output_dir: data/runs\n  intents_file: intents.info\n  n_responses_per_query: 2\n",
        encoding="utf-8")
    monkeypatch.setattr(muse, "probe", lambda args: pytest.fail("invalid config should not be probed"))
    args = muse.parse_args(["cycle", "--gpus", "4", "--run-id", "muse",
                            "--cycle-size", "2", "--total", "4"])
    with pytest.raises(RuntimeError, match="one-to-one counts"):
        muse.cycle(args)


def test_custom_endpoints_normalize_once():
    args = muse.parse_args(["plan", "--gpus", "4", "--tp", "2", "--endpoints",
                            "http://node:30000/v1,http://node:30001/v1/chat/completions"])
    assert muse.endpoints(args) == [
        "http://node:30000/v1/chat/completions",
        "http://node:30001/v1/chat/completions",
    ]
    duplicate = muse.parse_args(["plan", "--gpus", "4", "--tp", "2", "--endpoints",
                                 "http://node:30000/v1,http://node:30000/v1/chat/completions"])
    with pytest.raises(ValueError, match="distinct after"):
        muse.endpoints(duplicate)


def test_http_request_uses_muse_reasoning_strength_and_keeps_only_final(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({
                "choices": [{"message": {"reasoning_content": "Build a single text node.",
                                         "content": '<a2ui>\nroot=Text("Hi")\n</a2ui>'},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 20},
            }).encode()

    def capture(request, **kwargs):
        requests.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr("llm.local_adapter.urlopen", capture)
    monkeypatch.delenv("LOCAL_VLLM_ENDPOINTS", raising=False)
    monkeypatch.setenv("LOCAL_VLLM_ENABLE_THINKING", "1")
    monkeypatch.setenv("LOCAL_VLLM_SEND_CHAT_TEMPLATE_KWARGS", "1")
    monkeypatch.setenv("LOCAL_VLLM_REASONING_STRENGTH", "high")
    result = LocalAdapter(ModelSpec("muse", "local", muse.MODEL_ID,
                                    endpoint="http://unused.test/v1"))._http_generate(
        "Create A2UI Express", None, 1.0, 1024, 42, False)
    assert requests[0]["chat_template_kwargs"] == {"reasoning_strength": "high"}
    assert "enable_thinking" not in requests[0]["chat_template_kwargs"]
    assert result.reasoning_source == "message.reasoning_content"
    assert result.text.startswith("<a2ui>") and not result.error


def test_probe_rejects_endpoint_without_separate_reasoning(monkeypatch):
    args = muse.parse_args(["probe", "--gpus", "4"])

    def fake_read(url, payload=None, timeout=10):
        if url.endswith("/models"):
            return {"data": [{"id": args.served_model}]}
        if url.endswith("/server_info"):
            return {"speculative_algorithm": "DFLASH", "speculative_draft_model_path": muse.DRAFT_ID,
                    "reasoning_parser": "muse", "tool_call_parser": "muse"}
        return {"choices": [{"message": {"content": "437"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(muse, "read_json", fake_read)
    with pytest.raises(RuntimeError, match="separate, complete"):
        muse.probe(args)


def test_probe_rejects_server_without_dflash_even_if_chat_works(monkeypatch):
    args = muse.parse_args(["probe", "--gpus", "4"])

    def fake_read(url, payload=None, timeout=10):
        if url.endswith("/models"):
            return {"data": [{"id": args.served_model}]}
        if url.endswith("/server_info"):
            return {"speculative_algorithm": None, "reasoning_parser": "muse",
                    "tool_call_parser": "muse"}
        pytest.fail("Chat probe should not run without DFlash")

    monkeypatch.setattr(muse, "read_json", fake_read)
    with pytest.raises(RuntimeError, match="must have DFLASH"):
        muse.probe(args)


def test_probe_accepts_dflash_and_separate_reasoning(monkeypatch):
    args = muse.parse_args(["probe", "--gpus", "4"])
    seen = []

    def fake_read(url, payload=None, timeout=10):
        seen.append(url)
        if url.endswith("/models"):
            return {"data": [{"id": args.served_model}]}
        if url.endswith("/server_info"):
            return {"speculative_algorithm": "DFLASH", "speculative_draft_model_path": muse.DRAFT_ID,
                    "reasoning_parser": "muse", "tool_call_parser": "muse"}
        assert payload["chat_template_kwargs"] == {"reasoning_strength": "high"}
        return {"choices": [{"message": {"content": "437", "reasoning_content": "19 * 23 = 437"},
                             "finish_reason": "stop"}]}

    monkeypatch.setattr(muse, "read_json", fake_read)
    muse.probe(args)
    assert sum(url.endswith("/server_info") for url in seen) == 4


def test_generate_does_not_write_when_endpoint_preflight_fails(tmp_path, monkeypatch):
    run_dir = tmp_path / "dataset" / "data" / "runs" / "source"
    run_dir.mkdir(parents=True)
    (run_dir / "queries.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "responses.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(muse, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(muse, "probe", lambda args: (_ for _ in ()).throw(RuntimeError("wrong model")))
    monkeypatch.setattr(muse.subprocess, "call", lambda *args, **kwargs: pytest.fail("should not launch Stage 3"))
    args = muse.parse_args(["generate", "--gpus", "4", "--run-id", "source"])
    with pytest.raises(RuntimeError, match="wrong model"):
        muse.generate(args)


def test_generate_rejects_existing_output_from_other_model(tmp_path, monkeypatch):
    source = tmp_path / "dataset" / "data" / "runs" / "source"
    output = source.parent / "muse_output"
    source.mkdir(parents=True)
    output.mkdir()
    (source / "queries.jsonl").write_text("{}\n", encoding="utf-8")
    (source / "responses.jsonl").write_text("{}\n", encoding="utf-8")
    (output / "genui.jsonl").write_text('{"gen":{"model":"google/gemma-4-31b-it"}}\n', encoding="utf-8")
    monkeypatch.setattr(muse, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(muse, "probe", lambda args: pytest.fail("should not probe a mixed output run"))
    args = muse.parse_args(["generate", "--gpus", "4", "--source-run-id", "source",
                            "--run-id", "muse_output"])
    with pytest.raises(RuntimeError, match="choose a separate --run-id"):
        muse.generate(args)
