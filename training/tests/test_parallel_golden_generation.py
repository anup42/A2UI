"""CPU verification of isolated Golden workers, exact merging and diagnostics."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ir_training.common.jsonl import write_jsonl
from ir_training.eval import generate
from ir_training.eval import parallel_generate as parallel


def cohort(count=35):
    return [{"id": f"case-{index}", "response_text": f"response {index}", "completion": "<a2ui></a2ui>",
             "messages": [{"role": "user", "content": f"response {index}"}]} for index in range(count)]


@pytest.mark.parametrize("count", [1, 32, 35])
@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_shards_and_merge_are_exact_original_order(tmp_path, count, workers):
    workers = min(workers, count)
    rows = cohort(count)
    shards = parallel.shard_indices(count, workers)
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
    paths = []
    for rank, indices in enumerate(shards):
        path = tmp_path / f"{rank}.jsonl"
        write_jsonl(path, (generate.build_prediction_record(rows[index], f"generated {index}") for index in indices))
        paths.append(path)
    output = tmp_path / "predictions.jsonl"
    assert parallel.merge_shards(rows, shards, paths, output) == count
    assert [row["generated_text"] for row in parallel.read_rows_strict(output)] == [f"generated {index}" for index in range(count)]


@pytest.mark.parametrize("corruption", ["missing", "extra", "id", "source", "response", "expected", "query", "order", "bad_json", "duplicate_index", "missing_shard"])
def test_merge_rejects_partial_or_substituted_cases_without_publishing(tmp_path, corruption):
    rows, shards = cohort(4), [[0, 2], [1, 3]]
    predictions = [[generate.build_prediction_record(rows[index], "generated") for index in indices] for indices in shards]
    if corruption == "missing":
        predictions[1].pop()
    elif corruption == "extra":
        predictions[0].append(predictions[0][0])
    elif corruption == "id":
        predictions[1][0]["id"] = "wrong"
    elif corruption == "source":
        predictions[1][0]["source_context_sha256"] = "wrong"
    elif corruption == "response":
        predictions[1][0]["response_text"] = "wrong"
    elif corruption == "expected":
        predictions[1][0]["expected"] = "wrong"
    elif corruption == "query":
        predictions[1][0]["query_id"] = "wrong"
    elif corruption == "order":
        predictions[1].reverse()
    elif corruption == "duplicate_index":
        shards[1] = [0, 3]
    paths = [tmp_path / f"{rank}.jsonl" for rank in range(2)]
    for path, records in zip(paths, predictions, strict=True):
        write_jsonl(path, records)
    if corruption == "bad_json":
        paths[1].write_text("{bad}\n")
    if corruption == "missing_shard":
        paths.pop()
    output = tmp_path / "predictions.jsonl"
    with pytest.raises(ValueError):
        parallel.merge_shards(rows, shards, paths, output)
    assert not output.exists()


class FakeProcess:
    def __init__(self, code=None, stubborn=False):
        self.code, self.stubborn = code, stubborn
        self.terminated = self.killed = False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True
        if not self.stubborn:
            self.code = -15

    def wait(self, timeout):
        if self.code is None:
            raise subprocess.TimeoutExpired("fixture", timeout)
        return self.code

    def kill(self):
        self.killed, self.code = True, -9


def test_late_rank_failure_stops_other_workers_without_waiting_for_rank_zero():
    workers = [FakeProcess(), FakeProcess(0), FakeProcess(7)]
    with pytest.raises(RuntimeError, match="worker failed"):
        parallel.wait_for_workers(workers, timeout_seconds=10, started=time.monotonic())
    assert workers[0].terminated and not workers[1].terminated


def test_timeout_terminates_then_kills_stuck_child():
    worker = FakeProcess(stubborn=True)
    with pytest.raises(TimeoutError, match="exceeded"):
        parallel.wait_for_workers([worker], timeout_seconds=0.01, started=time.monotonic() - 1)
    assert worker.terminated and worker.killed


def test_actual_subprocess_timeout_has_no_surviving_child():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    with pytest.raises(TimeoutError):
        parallel.wait_for_workers([child], timeout_seconds=0.1, started=time.monotonic())
    assert child.poll() is not None


@pytest.mark.parametrize("selection,required", [("cpu", True), ("auto", True), ("0", False)])
def test_gpu_required_never_falls_back_to_cpu(monkeypatch, selection, required):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises((ValueError, RuntimeError)):
        parallel.resolve_generation_devices(selection, require_gpu=required)


def test_worker_selection_preserves_scheduler_masks_and_uuids(monkeypatch):
    import torch
    from ir_training.train import gpu_profile

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gpu_profile, "detect_cuda_devices", lambda: {"devices": [
        {"visible_index": 0, "launch_identifier": "GPU-aaaa", "uuid": "GPU-aaaa"},
        {"visible_index": 1, "launch_identifier": "GPU-bbbb", "uuid": "GPU-bbbb"},
    ]})
    assert parallel.resolve_generation_devices("1,0") == ["GPU-bbbb", "GPU-aaaa"]


@pytest.mark.parametrize("workers", [2, 4, 8])
def test_spawn_environments_are_isolated_and_metrics_are_wall_measured(tmp_path, monkeypatch, workers):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setattr(parallel, "resolve_generation_devices", lambda *a, **k: [f"GPU-{rank}" for rank in range(workers)])
    launched = []

    def popen(command, env):
        launched.append(env)
        job = json.loads(Path(command[-1]).read_text())
        rows = parallel.read_rows_strict(job["split_path"])
        write_jsonl(job["output_path"], [generate.build_prediction_record(row, "generated", runtime={
            "output_tokens": 2, "inference_device": "cuda:0"}) for row in rows])
        return FakeProcess(0)

    monkeypatch.setattr(parallel.subprocess, "Popen", popen)
    split, output = tmp_path / "split.jsonl", tmp_path / "predictions.jsonl"
    write_jsonl(split, cohort())
    metrics = {}
    assert parallel.generate_predictions_parallel({}, split, output, performance_metrics=metrics) == 35
    assert [env["CUDA_VISIBLE_DEVICES"] for env in launched] == [f"GPU-{rank}" for rank in range(workers)]
    assert all("RANK" not in env and "WORLD_SIZE" not in env for env in launched)
    assert all(int(env["OMP_NUM_THREADS"]) >= 1 for env in launched)
    assert metrics["generation_runtime_gpu_count"] == workers
    assert metrics["generation_runtime_case_count"] == 35
    assert metrics["generation_runtime_wall_seconds"] > 0
    with pytest.raises(FileExistsError):
        parallel.generate_predictions_parallel({}, split, output)


def test_fresh_attempt_required_after_worker_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(parallel, "resolve_generation_devices", lambda *a, **k: ["GPU-one", "GPU-two"])
    workers = [FakeProcess(), FakeProcess(1)]
    monkeypatch.setattr(parallel.subprocess, "Popen", lambda *a, **k: workers.pop(0))
    split, output = tmp_path / "split.jsonl", tmp_path / "predictions.jsonl"
    write_jsonl(split, cohort(2))
    with pytest.raises(RuntimeError, match="worker failed"):
        parallel.generate_predictions_parallel({}, split, output)
    assert not output.exists()
    with pytest.raises(FileExistsError):
        parallel.generate_predictions_parallel({}, split, output)


def test_nested_torchrun_launch_is_rejected_before_spawning(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    with pytest.raises(ValueError, match="not with torchrun"):
        parallel.generate_predictions_parallel({}, tmp_path / "none", tmp_path / "output")


def test_worker_only_uses_one_assigned_cuda_gpu_and_keeps_parent_config(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    devices, configs = [], []
    monkeypatch.setattr(torch.cuda, "set_device", devices.append)
    config = {"model": {"device_map": "auto", "dtype": "bfloat16"}}
    path = tmp_path / "job.json"
    output = tmp_path / "result.jsonl"
    path.write_text(json.dumps({"config": config, "gpu": True, "max_rows": 1, "output_path": str(output)}))

    def generator(config, **kwargs):
        configs.append(config)
        write_jsonl(output, [{"runtime": {"inference_device": "cuda:0"}}])
        return 1

    monkeypatch.setattr(parallel, "generate_predictions", generator)
    parallel.run_worker(path)
    assert configs[0]["model"]["device_map"] == {"": 0}
    assert configs[0]["model"]["inference_device"] == "auto"
    assert devices == [0] and config["model"]["device_map"] == "auto"


@pytest.mark.parametrize("fail_second", [False, True])
def test_sequential_generation_flushes_progress_and_keeps_failure_partial(tmp_path, monkeypatch, capsys, fail_second):
    import torch

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id, eos_token_id, vocab_size = 0, 2, 8

        def __call__(self, *args, **kwargs):
            return Batch(input_ids=torch.tensor([[1, 3]]), attention_mask=torch.ones((1, 2), dtype=torch.long))

        def decode(self, *args, **kwargs):
            return "<a2ui></a2ui>"

    calls = []
    model = SimpleNamespace(device=torch.device("cpu"), config=SimpleNamespace(use_cache=False),
                            generation_config=SimpleNamespace(eos_token_id=2), eval=lambda: None)

    def model_generate(**kwargs):
        calls.append(kwargs)
        assert not torch.backends.cuda.cudnn_sdp_enabled()
        if fail_second and len(calls) == 2:
            raise RuntimeError("inference fixture failure")
        assert model.config.use_cache is True
        return torch.tensor([[1, 3, 4, 2]])

    model.generate = model_generate
    adapter = SimpleNamespace(load_tokenizer=Tokenizer, load_model=lambda: model, format_example=lambda *a, **k: "prompt")
    monkeypatch.setattr(generate, "create_adapter", lambda _: adapter)
    monkeypatch.setattr(generate, "build_stopping_criteria", lambda *a: [])
    monkeypatch.setattr(generate, "generation_diagnostics", lambda *a, **k: {"output_tokens": 2})
    split, output = tmp_path / "split.jsonl", tmp_path / "predictions.jsonl"
    write_jsonl(split, cohort(2))
    prior = torch.backends.cuda.cudnn_sdp_enabled()
    if fail_second:
        with pytest.raises(RuntimeError, match="inference fixture failure"):
            generate.generate_predictions({"model": {"inference_device": "cpu"}}, split, output)
        assert not output.exists()
        assert len(parallel.read_rows_strict(output.with_name(output.name + ".partial"))) == 1
    else:
        assert generate.generate_predictions({"model": {"inference_device": "cpu"}}, split, output) == 2
        assert len(parallel.read_rows_strict(output)) == 2
        assert not output.with_name(output.name + ".partial").exists()
    assert model.config.use_cache is False
    assert torch.backends.cuda.cudnn_sdp_enabled() == prior
    console = capsys.readouterr().out
    assert "case 1/2" in console and "output_tokens=2" in console


@pytest.mark.parametrize("checkpoint_kind", ["merged", "adapter"])
def test_real_cpu_gemma_worker_subprocess_smoke(tmp_path, monkeypatch, checkpoint_kind):
    """Actual model/tokenizer load, generation, exit, merge; no GPU/download claim."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers", minversion="5.10.1")
    tokenizers = pytest.importorskip("tokenizers")
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    directory = tmp_path / "model"
    directory.mkdir()
    vocabulary = {word: index for index, word in enumerate(["[PAD]", "[UNK]", "[EOS]", "[BOS]", "user", "assistant", ":", "response"])}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
        eos_token="[EOS]", bos_token="[BOS]", model_max_length=64)
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' : ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant : {% endif %}"
    tokenizer.save_pretrained(directory)
    text = transformers.Gemma4TextConfig(vocab_size=8, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=16, max_position_embeddings=64,
        vocab_size_per_layer_input=8, hidden_size_per_layer_input=4, layer_types=["full_attention", "full_attention"],
        num_kv_shared_layers=0, pad_token_id=0, bos_token_id=3, eos_token_id=2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = transformers.Gemma4ForCausalLM(text)
        model.save_pretrained(directory)
    adapter_checkpoint = None
    if checkpoint_kind == "adapter":
        peft = pytest.importorskip("peft")
        adapter_checkpoint = tmp_path / "adapter"
        adapter = peft.get_peft_model(model, peft.LoraConfig(r=2, lora_alpha=2,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        adapter.save_pretrained(adapter_checkpoint)
    config = {"model": {"family": "gemma", "model_id": "tiny-local-gemma-4", "model_source": str(directory),
        "tokenizer_source": str(directory), "tokenizer_loader": "pretrained_tokenizer_fast", "dtype": "float32",
        "device_map": "none", "attn_implementation": "sdpa"}}
    original = copy.deepcopy(config)
    split, output = tmp_path / "split.jsonl", tmp_path / "predictions.jsonl"
    write_jsonl(split, cohort(2))
    metrics = {}
    assert parallel.generate_predictions_parallel(config, split, output, max_new_tokens=2, max_input_tokens=32,
        devices="cpu", timeout_seconds=120, performance_metrics=metrics, adapter_checkpoint=adapter_checkpoint) == 2
    predictions = parallel.read_rows_strict(output)
    assert [row["id"] for row in predictions] == ["case-0", "case-1"]
    assert all(row["runtime"]["inference_device"] == "cpu" for row in predictions)
    assert metrics["generation_runtime_gpu_count"] == 0 and config == original
