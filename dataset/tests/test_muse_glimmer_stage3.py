"""CPU-only checks for the Muse Stage 3 profile and HTTP contract."""

import importlib.util
import json
import logging
import sys
from pathlib import Path
from threading import Lock

import pytest


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
from llm.base import ModelSpec  # noqa: E402
from llm.local_adapter import LocalAdapter  # noqa: E402
from pipeline.cache import PromptCache  # noqa: E402
from pipeline.ir_formats import A2UI_EXPRESS_V1  # noqa: E402
from pipeline.stage3_genui import run_stage3  # noqa: E402
from utils.rate_limit import RateLimiter  # noqa: E402
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
    assert env["LOCAL_MUSE_REQUIRE_REASONING"] == "1"
    assert muse.client_env(args, 1)["LOCAL_MUSE_REQUIRE_REASONING"] == "0"
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
                row = {"gen": {"model": muse.MODEL_ID}}
                if stage == 3:
                    row["reasoning_text"] = "Plan the UI."
                    row["gen"]["reasoning_source"] = "message.reasoning_content"
                handle.write(json.dumps(row) + "\n")
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


def test_cycle_rejects_stage3_rows_without_saved_thinking(tmp_path, monkeypatch):
    run = cyclic_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(muse, "probe", lambda args: None)

    def without_thinking(command, cwd, env):
        stage = int(command[command.index("--stage") + 1])
        path = run / {1: "queries.jsonl", 2: "responses.jsonl", 3: "genui.jsonl"}[stage]
        path.write_text(json.dumps({"gen": {"model": muse.MODEL_ID}}) + "\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(muse.subprocess, "call", without_thinking)
    args = muse.parse_args(["cycle", "--gpus", "4", "--run-id", "muse",
                            "--cycle-size", "1", "--total", "1"])
    with pytest.raises(RuntimeError, match="no saved Muse to=self reasoning"):
        muse.cycle(args)


@pytest.mark.parametrize("generation_attempted", [None, True, False])
def test_reasoning_count_only_exempts_rejections_before_generation(tmp_path, generation_attempted):
    row = {"gen": {"model": muse.MODEL_ID}, "record_status": "quality_rejected"}
    if generation_attempted is not None:
        row["validation"] = {"generation_attempted": generation_attempted}
    output = tmp_path / "genui.jsonl"
    output.write_text(json.dumps(row) + "\n", encoding="utf-8")
    if generation_attempted is False:
        assert muse.count_records(output, require_muse=True, require_reasoning=True) == 1
    else:
        with pytest.raises(RuntimeError, match="no saved Muse to=self reasoning"):
            muse.count_records(output, require_muse=True, require_reasoning=True)
        row["reasoning_text"] = "\n The generated UI failed semantic checks. \n"
        row["gen"]["reasoning_source"] = "message.reasoning_content"
        output.write_text(json.dumps(row) + "\n", encoding="utf-8")
        assert muse.count_records(output, require_muse=True, require_reasoning=True) == 1


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
                "choices": [{"message": {"reasoning_content": "\n Build a single text node. \n",
                                         "content": '<a2ui>\nroot=Text("Hi")\n</a2ui>'},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 20,
                          "completion_tokens_details": {"reasoning_tokens": 11}},
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
    assert result.reasoning_text == "\n Build a single text node. \n"
    assert result.reasoning_tokens == 11
    assert result.text.startswith("<a2ui>") and not result.error


@pytest.mark.parametrize("message_fields,other_choice", [
    pytest.param({}, False, id="absent"),
    pytest.param({"reasoning_content": None}, False, id="null"),
    pytest.param({"reasoning_content": " \n\t"}, False, id="blank"),
    pytest.param({"reasoning_content": ["Structured reasoning"]}, False, id="list"),
    pytest.param({"reasoning_content": {"text": "Structured reasoning"}}, False, id="object"),
    pytest.param({"reasoning_text": "Wrong server field"}, False, id="other-field"),
    pytest.param({"content": '<think>Inline reasoning</think>\n<a2ui>\nroot=Text("Hi")\n</a2ui>'},
                 False, id="inline"),
    pytest.param({}, True, id="other-choice"),
])
def test_stage3_requires_separate_muse_thinking(monkeypatch, message_fields, other_choice):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            message = {"content": '<a2ui>\nroot=Text("Hi")\n</a2ui>', **message_fields}
            choices = [{"message": message, "finish_reason": "stop"}]
            if other_choice:
                choices.append({"message": {"reasoning_content": "Belongs to another completion"},
                                "finish_reason": "stop"})
            return json.dumps({
                "choices": choices,
                "usage": {"prompt_tokens": 12, "completion_tokens": 20},
            }).encode()

    monkeypatch.setattr("llm.local_adapter.urlopen", lambda *args, **kwargs: Response())
    monkeypatch.setenv("LOCAL_MUSE_REQUIRE_REASONING", "1")
    result = LocalAdapter(ModelSpec("muse", "local", muse.MODEL_ID,
                                    endpoint="http://unused.test/v1"))._http_generate(
        "Create A2UI Express", None, 1.0, 1024, 42, False)
    assert result.error == "incomplete_completion: missing_muse_reasoning_content"


def muse_http_response(reasoning, completion='<a2ui>\nroot=Text("Hi")\n</a2ui>'):
    message = {"content": completion}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    payload = {
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 20,
                  "completion_tokens_details": {"reasoning_tokens": 11}},
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(payload).encode()

    return Response()


@pytest.fixture
def muse_stage3(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_MUSE_REQUIRE_REASONING", "1")
    monkeypatch.setenv("LOCAL_ALLOW_HTTP_ENDPOINT", "1")
    monkeypatch.setenv("LOCAL_STRICT_OFFLINE", "0")
    monkeypatch.setenv("LOCAL_VLLM_USE_HF_GENERATION_CONFIG", "0")
    monkeypatch.setenv("LOCAL_VLLM_RETRY_RESULT_ERRORS", "1")
    monkeypatch.setenv("LOCAL_VLLM_RETRY_MAX_SECONDS", "1")
    monkeypatch.setenv("STAGE3_FINAL_REGEN_ATTEMPTS", "0")
    monkeypatch.setenv("DATASET_OFFLINE_MODE", "1")
    monkeypatch.delenv("LOCAL_VLLM_ENDPOINTS", raising=False)

    def run(*, count=1, cache=None, output_name="genui.jsonl", max_repair_attempts=0):
        queries = tmp_path / "queries.jsonl"
        responses = tmp_path / "responses.jsonl"
        output = tmp_path / output_name
        artifacts = tmp_path / (output.stem + "_artifacts")
        queries.write_text("".join(json.dumps({"query_id": f"q{i}", "intent": "generic"}) + "\n"
                                   for i in range(1, count + 1)), encoding="utf-8")
        responses.write_text("".join(json.dumps({"response_id": f"r{i}", "query_id": f"q{i}",
                                                "n_idx": 1, "response_text": "Hi"}) + "\n"
                                     for i in range(1, count + 1)), encoding="utf-8")
        adapter = LocalAdapter(ModelSpec("muse", "local", muse.MODEL_ID,
                                         endpoint="http://unused.test/v1"))
        run_stage3(
            queries_path=queries, responses_path=responses,
            prompt_path=DATASET_ROOT / "prompts/genui_gen_mobile_a2ui_express_v1.md",
            adapter=adapter, genui_path=output,
            schema_path=DATASET_ROOT / "schema/canonical_ui_graph_v1.schema.json",
            artifacts_dir=artifacts, candidates_per_response=1,
            max_repair_attempts=max_repair_attempts, max_tokens=2048, prompt_max_tokens=None,
            seed=4, rate_limiter=RateLimiter(0),
            cache=cache or PromptCache(tmp_path / "cache", enabled=False),
            logger=logging.getLogger("muse-reasoning-test"), batch_size=count,
            max_attempts=1, metric_version="legacy", ir_formats=[A2UI_EXPRESS_V1],
            phase_invocation_id="test-phase",
        )
        return [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()], artifacts

    return run


def test_stage3_saves_exact_muse_thinking_and_attempt_artifact(tmp_path, monkeypatch, muse_stage3):
    reasoning = "\n First, make one text node. \n"
    monkeypatch.setattr("llm.local_adapter.urlopen", lambda *args, **kwargs: muse_http_response(reasoning))
    rows, artifacts = muse_stage3()
    row = rows[0]
    assert row["record_status"] == "accepted"
    assert row["reasoning_text"] == reasoning
    assert row["gen"]["reasoning_source"] == "message.reasoning_content"
    assert row["gen"]["reasoning_tokens"] == 11
    assert row["gen"]["reasoning_format"] == "muse_atem_to_self"
    assert row["generation_attempts"][0]["reasoning_available"] is True
    attempt_path = artifacts / row["generation_attempts"][0]["artifact"]
    assert json.loads(attempt_path.read_text(encoding="utf-8"))["reasoning_text"] == reasoning
    assert muse.count_records(tmp_path / "genui.jsonl", require_muse=True, require_reasoning=True) == 1


def test_stage3_restores_exact_muse_reasoning_from_legacy_cache(tmp_path, monkeypatch, muse_stage3):
    reasoning = "\n\t Plan the text node.  \n"
    monkeypatch.setattr("llm.local_adapter.urlopen", lambda *args, **kwargs: muse_http_response(reasoning))
    cache_path = tmp_path / "cache.jsonl"
    muse_stage3(cache=PromptCache(cache_path, enabled=True), output_name="warm.jsonl")
    legacy = json.loads(cache_path.read_text(encoding="utf-8").splitlines()[0])
    legacy["reasoning_text"] = reasoning.strip()
    cache_path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    monkeypatch.setattr("llm.local_adapter.urlopen", lambda *args, **kwargs: pytest.fail("cache should avoid HTTP"))
    rows, artifacts = muse_stage3(cache=PromptCache(cache_path, enabled=True))
    row = rows[0]
    assert row["reasoning_text"] == reasoning
    assert row["generation_totals"]["provider_call_count"] == 0
    attempt = row["generation_attempts"][0]
    assert attempt["phase"] == "cache"
    assert json.loads((artifacts / attempt["artifact"]).read_text(encoding="utf-8"))["reasoning_text"] == reasoning


def test_stage3_regenerates_cache_without_exact_raw_reasoning(tmp_path, monkeypatch, muse_stage3):
    monkeypatch.setattr("llm.local_adapter.urlopen", lambda *args, **kwargs: muse_http_response("Old trace"))
    cache_path = tmp_path / "cache.jsonl"
    muse_stage3(cache=PromptCache(cache_path, enabled=True), output_name="warm.jsonl")
    legacy = json.loads(cache_path.read_text(encoding="utf-8").splitlines()[0])
    del legacy["raw"]["choices"][0]["message"]["reasoning_content"]
    cache_path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    fresh_reasoning = "\n Exact fresh reasoning. \n"
    requests = []

    def generate(request, **kwargs):
        requests.append(json.loads(request.data))
        return muse_http_response(fresh_reasoning)

    monkeypatch.setattr("llm.local_adapter.urlopen", generate)
    rows, _ = muse_stage3(cache=PromptCache(cache_path, enabled=True))
    assert len(requests) == 1
    assert rows[0]["reasoning_text"] == fresh_reasoning
    assert rows[0]["generation_totals"]["provider_call_count"] == 1
    assert rows[0]["generation_attempts"][0]["phase"] == "initial"


def test_stage3_retries_only_batch_item_missing_muse_reasoning(monkeypatch, muse_stage3):
    results = [muse_http_response(None), muse_http_response("\n Batch success. \n"),
               muse_http_response("\n Retry success. \n")]
    requests = []
    lock = Lock()

    def generate(request, **kwargs):
        with lock:
            requests.append(json.loads(request.data))
            return results.pop(0)

    monkeypatch.setattr("llm.local_adapter.urlopen", generate)
    rows, artifacts = muse_stage3(count=2)
    assert len(requests) == 3
    assert len(rows) == 2
    assert {row["reasoning_text"] for row in rows} == {"\n Batch success. \n", "\n Retry success. \n"}
    retried = next(row for row in rows if len(row["generation_attempts"]) == 2)
    initial, retry = retried["generation_attempts"]
    assert initial["phase"] == "initial_batch"
    assert initial["error"] == "incomplete_completion: missing_muse_reasoning_content"
    assert retry["error"] is None
    assert json.loads((artifacts / retry["artifact"]).read_text(encoding="utf-8"))["reasoning_text"] == "\n Retry success. \n"


@pytest.mark.parametrize("retry_enabled", [False, True])
def test_stage3_batch_missing_reasoning_respects_retry_limits(monkeypatch, muse_stage3, retry_enabled):
    monkeypatch.setenv("LOCAL_VLLM_RETRY_RESULT_ERRORS", "1" if retry_enabled else "0")
    monkeypatch.setenv("LOCAL_VLLM_RETRY_MAX_SECONDS", "1")
    requests = []
    sleeps = []
    lock = Lock()
    clock_ticks = iter(range(1000))
    monkeypatch.setattr("pipeline.stage3_genui.time.time", lambda: float(next(clock_ticks)))
    monkeypatch.setattr("pipeline.stage3_genui.time.sleep", sleeps.append)

    def generate(request, **kwargs):
        with lock:
            requests.append(json.loads(request.data))
            return muse_http_response("\n Batch success. \n" if len(requests) == 2 else None)

    monkeypatch.setattr("llm.local_adapter.urlopen", generate)
    rows, artifacts = muse_stage3(count=2)
    assert len(requests) == (4 if retry_enabled else 2)
    assert len(sleeps) == (1 if retry_enabled else 0)
    assert len(rows) == 1
    assert rows[0]["reasoning_text"] == "\n Batch success. \n"
    errors = [json.loads(path.read_text(encoding="utf-8")) for path in artifacts.glob("error_*.json")]
    error = next(item for item in errors if item.get("error") == "incomplete_completion: missing_muse_reasoning_content")
    assert len(error["generation_attempts"]) == (3 if retry_enabled else 1)
    assert all(attempt["error"] == error["error"] for attempt in error["generation_attempts"])


@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_stage3_repair_and_format_rejection_keep_matching_reasoning(monkeypatch, muse_stage3, repair_succeeds):
    initial_reasoning = "\n Initial invalid draft. \n"
    repair_reasoning = "\n Fix the invalid draft. \n"
    valid = '<a2ui>\nroot=Text("Hi")\n</a2ui>'
    results = [muse_http_response(initial_reasoning, "invalid initial"),
               muse_http_response(repair_reasoning, valid if repair_succeeds else "invalid repair")]
    monkeypatch.setattr("llm.local_adapter.urlopen", lambda *args, **kwargs: results.pop(0))
    rows, artifacts = muse_stage3(max_repair_attempts=1)
    row = rows[0]
    assert row["record_status"] == ("accepted" if repair_succeeds else "format_rejected")
    assert row["reasoning_text"] == repair_reasoning
    assert row["gen"]["reasoning_attempt"] == "repair_1"
    assert len(row["generation_attempts"]) == 2
    saved = [json.loads((artifacts / attempt["artifact"]).read_text(encoding="utf-8"))
             for attempt in row["generation_attempts"]]
    assert [attempt["reasoning_text"] for attempt in saved] == [initial_reasoning, repair_reasoning]
    assert saved[0]["completion"] == "invalid initial"
    assert saved[1]["completion"] == (valid if repair_succeeds else "invalid repair")


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
