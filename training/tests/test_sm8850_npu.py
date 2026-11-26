import contextlib
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ir_training.common.jsonl import write_jsonl
from ir_training.data.bixby50 import APPROVED_SOURCE_IDS
from ir_training.eval import qualcomm_npu as device
from ir_training.export import qualcomm_npu as npu


def graph_bytes(signatures, cpu=()):
    flatbuffers = pytest.importorskip("flatbuffers")
    s = pytest.importorskip("ai_edge_litert.schema_py_generated")
    model = s.ModelT()
    model.version = 3
    dispatch, add = s.OperatorCodeT(), s.OperatorCodeT()
    dispatch.builtinCode = dispatch.deprecatedBuiltinCode = s.BuiltinOperator.CUSTOM
    dispatch.customCode = b"DISPATCH_OP"
    add.builtinCode = s.BuiltinOperator.ADD
    model.operatorCodes = [dispatch, add]
    model.buffers = [s.BufferT()]
    model.signatureDefs, model.subgraphs = [], []
    for i, name in enumerate(signatures):
        op, graph, sig = s.OperatorT(), s.SubGraphT(), s.SignatureDefT()
        op.opcodeIndex = int(name in cpu)
        graph.operators = [op]
        sig.signatureKey, sig.subgraphIndex = name.encode(), i
        model.subgraphs.append(graph)
        model.signatureDefs.append(sig)
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def package(tmp_path, profile="e2b", *, cpu=(), omit=None, drafter=False, capacity=128):
    lm = pytest.importorskip("litert_lm_builder")
    from litert_lm_builder.runtime.proto import executor_metadata_pb2

    builder = lm.LitertLmFileBuilder()
    metadata, tokenizer = tmp_path / "llm.pb", tmp_path / "sp.model"
    metadata.write_bytes(b"\xf8\x07\x01")  # Valid unknown protobuf field.
    tokenizer.write_bytes(b"fixture-tokenizer")
    builder.add_llm_metadata(str(metadata)).add_sentencepiece_tokenizer(str(tokenizer))
    executor = executor_metadata_pb2.ExecutorMetadata()
    executor.llm_executor_metadata.max_history_size = capacity
    for i in range(15 if profile == "e2b" else 18):
        for kind in ("k", "v"):
            executor.llm_executor_metadata.state_buffers.add(
                decode_input_name=f"kv_cache_{kind}_{i}"
            )
    executor_path = tmp_path / "executor.pb"
    executor_path.write_bytes(executor.SerializeToString())
    builder.add_executor_metadata(str(executor_path))
    by_type = {
        npu.TARGET: ("",),
        npu.EMBEDDER: ("embedder",),
        "tf_lite_aux": ("cache_update", "mask", "rope"),
    }
    if profile == "e2b":
        by_type[npu.PER_LAYER] = ("per_layer_embedder",)
    if drafter:
        by_type["tf_lite_mtp_drafter"] = ("drafter",)
    for name, roles in by_type.items():
        signatures = (
            ["prefill_128", "decode"]
            if name == npu.TARGET
            else [
                sig
                for role in roles
                for sig in (f"prefill_{role}_128", f"decode_{role}")
            ]
        )
        graph = tmp_path / f"{name}.tflite"
        graph.write_bytes(
            graph_bytes([sig for sig in signatures if sig != omit], cpu=cpu)
        )
        builder.add_tflite_model(str(graph), lm.TfLiteModelType(name))
    path = tmp_path / npu.PACKAGE
    with path.open("wb") as stream:
        builder.build(stream)
    return path


@pytest.mark.parametrize("profile", ["270m", "e2b"])
def test_packaged_dispatch_both_profiles(tmp_path, profile):
    audit = npu.inspect_npu_package(package(tmp_path, profile), profile)
    assert audit["dispatch_ops_per_signature"] == {"decode": 1, "prefill_128": 1}
    assert (npu.PER_LAYER in audit["sections"]) == (profile == "e2b")


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"cpu": ("decode",)}, "uncompiled"),
        ({"omit": "prefill_128"}, "Missing compiled"),
        ({"omit": "decode_cache_update"}, "lacks runtime signatures"),
        ({"omit": "prefill_per_layer_embedder_128"}, "lacks runtime signatures"),
        ({"drafter": True}, "target-only"),
    ],
)
def test_package_rejects_unusable_or_speculative_graph(tmp_path, kwargs, error):
    with pytest.raises(ValueError, match=error):
        npu.inspect_npu_package(package(tmp_path, **kwargs), "e2b")


def test_export_receipt_and_request_tampering(tmp_path):
    path = package(tmp_path, "270m")
    requests = [
        {"id": i, "input_ids": [2, 3], "prompt": "hello"} for i in APPROVED_SOURCE_IDS
    ]
    request_file = tmp_path / npu.REQUESTS
    write_jsonl(request_file, requests)
    (tmp_path / npu.SPLIT).write_bytes((npu.BIXBY / npu.SPLIT).read_bytes())
    (tmp_path / "manifest.json").write_text("{}")
    receipt = dict(
        npu.inspect_npu_package(path, "270m"),
        format="a2ui-sm8850-npu-v1",
        profile="270m",
        soc="SM8850",
        mtp=False,
        model_sha256=npu.file_sha256(path),
        bixby50_split_sha256=npu.file_sha256(tmp_path / npu.SPLIT),
        bixby50_manifest_sha256=npu.file_sha256(tmp_path / "manifest.json"),
        requests_sha256=npu.file_sha256(request_file),
        max_output_tokens=4,
        max_input_tokens=64,
        cache_length=128,
    )
    (tmp_path / "export.json").write_text(json.dumps(receipt))
    assert device.validate_export(tmp_path) == (receipt, requests)
    write_jsonl(request_file, requests[:-1])
    with pytest.raises(ValueError, match="binding changed"):
        device.validate_export(tmp_path)
    receipt["requests_sha256"] = npu.file_sha256(request_file)
    (tmp_path / "export.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="50 ordered"):
        device.validate_export(tmp_path)


@pytest.mark.parametrize("profile", ["270m", "e2b"])
def test_converter_routes_float_main_and_w8_lookups_then_restores_hooks(
    profile, monkeypatch, tmp_path
):
    from ir_training.export import gemma4_text_compat

    @dataclasses.dataclass
    class Config:
        quantization_recipe: str

    def lookup(source, config, artifacts):
        return config.quantization_recipe

    hooks = SimpleNamespace(calib_func=lambda **kwargs: kwargs, quant_func=object())
    lib = SimpleNamespace(export_embedder_model=lookup, export_additional_models=lookup)
    calls = []

    def export(**kwargs):
        calls.append(kwargs)
        config = Config(kwargs["quantization_recipe"])
        assert config.quantization_recipe == ""
        assert lib.export_embedder_model(None, config, None) == "dynamic_wi8_afp32"
        assert lib.export_additional_models(None, config, None) == "dynamic_wi8_afp32"
        assert config.quantization_recipe == ""
        assert hooks.calib_func() == {"enable_formatting": False}
        assert hooks.quant_func is npu._quantize_sections
        raise RuntimeError("compiler fixture failure")

    scoped = []

    @contextlib.contextmanager
    def compat(*, qualcomm_npu):
        scoped.append(qualcomm_npu)
        yield

    monkeypatch.setattr(gemma4_text_compat, "gemma4_text_export_context", compat)
    mod = "litert_torch.generative.export_hf"
    monkeypatch.setitem(
        sys.modules,
        mod,
        SimpleNamespace(export=SimpleNamespace(export=export), export_utils=hooks),
    )
    monkeypatch.setitem(sys.modules, mod + ".core", SimpleNamespace(export_lib=lib))
    with pytest.raises(RuntimeError, match="compiler fixture"):
        npu._convert(
            profile, tmp_path, tmp_path, 1024, [{"text": "rendered training prompt"}]
        )
    assert (
        lib.export_embedder_model is lookup and lib.export_additional_models is lookup
    )
    assert hooks.calib_func() == {} and hooks.quant_func is not npu._quantize_sections
    assert scoped == ([True] if profile == "e2b" else [])
    (opts,) = calls
    assert (opts["aot_backend"], opts["aot_soc_model"]) == ("qualcomm", "SM8850")
    assert (
        opts["use_litert_lm_compiler"]
        and opts["split_cache"]
        and opts["externalize_embedder"]
    )
    assert opts["max_calibration_decode_steps"] == 32
    assert (
        not opts["enable_gpu_dynamic_cache"] and not opts["enable_gpu_dynamic_prefill"]
    )
    assert ("jinja_chat_template_override" in opts) == (profile == "e2b")


@pytest.mark.parametrize(
    ("weight_bits", "activation_bits", "accepted"),
    [(8, 16, True), (8, 32, False), (4, 16, False)],
)
def test_precompile_check_requires_calibrated_physical_tensors(
    weight_bits, activation_bits, accepted
):
    s = pytest.importorskip("ai_edge_litert.schema_py_generated")
    blob = graph_bytes(["prefill_128", "decode"])
    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(blob, 0))
    weight_buf = s.BufferT()
    weight_buf.data = [3]
    model.buffers.append(weight_buf)
    for graph in model.subgraphs:
        weight, activation = s.TensorT(), s.TensorT()
        weight.type = s.TensorType.INT8 if weight_bits == 8 else s.TensorType.INT4
        weight.buffer = 1
        activation.type = (
            s.TensorType.INT16 if activation_bits == 16 else s.TensorType.FLOAT32
        )
        for tensor in (weight, activation):
            tensor.quantization = s.QuantizationParametersT()
            tensor.quantization.scale = [0.1]
        graph.tensors = [weight, activation]
    import flatbuffers

    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    graph = s.Model.GetRootAsModel(builder.Output(), 0)
    if accepted:
        npu._check_static_quantization(graph, s)
    else:
        with pytest.raises(ValueError, match="Static W8/A16"):
            npu._check_static_quantization(graph, s)


def test_selected_e2b_merge_weights_and_config_must_match(tmp_path):
    ckpt = tmp_path / "merged_best_hf"
    ckpt.mkdir()
    config = tmp_path / "training.yaml"
    config.write_text("model: {}\n")
    (ckpt / "config.json").write_text('{"model_type":"gemma4_text"}')
    weight = ckpt / "model.safetensors"
    weight.write_bytes(b"selected weights")
    evidence = {
        "manifest_version": 4,
        "continued_qat_performed": True,
        "qat_profile": "gemma4_e2b_mobile_retained_scale_wna8o8",
        "training_config_sha256": npu.file_sha256(config),
        "merged_model_files": [
            {
                "path": weight.name,
                "size": weight.stat().st_size,
                "sha256": npu.file_sha256(weight),
            }
        ],
    }
    evidence.update(
        {
            key: {"verified": True}
            for key in (
                "mobile_training_seed",
                "mobile_qparams",
                "training_run_metadata",
            )
        }
    )
    (ckpt / "qat_mtp_merge_metadata.json").write_text(json.dumps(evidence))
    assert npu.checkpoint_binding("e2b", ckpt, config)[0] == {"model": {}}
    config.write_text("model: {edited: true}\n")
    with pytest.raises(ValueError, match="Training config"):
        npu.checkpoint_binding("e2b", ckpt, config)
    weight.write_bytes(b"replaced weights")
    with pytest.raises(ValueError, match="inventory differs"):
        npu.checkpoint_binding("e2b", ckpt, config)


def test_calibration_uses_distinct_train_sources_and_refuses_holdout(tmp_path):
    def row(key):
        return {
            "id": key,
            "source_id": key,
            "response_text": f"Source {key}",
            "messages": [
                {"role": "user", "content": f"Source {key}"},
                {"role": "assistant", "content": "target must stay out"},
            ],
        }

    for name in ("golden32", "golden35", "bixby50"):
        write_jsonl(tmp_path / f"{name}.jsonl", [row(name)])
    train = tmp_path / "train.jsonl"
    inputs = [row(str(i)) for i in range(50)]
    write_jsonl(train, inputs + inputs)
    manifest = {"splits": {"train": {"output_sha256": npu.file_sha256(train)}}}

    def format_example(record, *, tokenizer, include_assistant):
        assert include_assistant is False
        return record["messages"][0]["content"]

    adapter = SimpleNamespace(format_example=format_example)

    def tokenize(text, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [2, 3]}

    def calibrate():
        return npu.training_calibration(
            tmp_path,
            manifest,
            adapter,
            tokenize,
            128,
            [
                tmp_path / f"{name}.jsonl"
                for name in ("golden32", "golden35", "bixby50")
            ],
        )

    prompts, proof = calibrate()
    assert (prompts, proof) == calibrate()
    assert len(set(proof["sample_ids"])) == 32
    assert prompts == [{"text": f"Source {item}"} for item in proof["sample_ids"]]
    write_jsonl(train, [row("golden35")] + inputs)
    with pytest.raises(ValueError, match="preparation receipt"):
        calibrate()
    manifest["splits"]["train"]["output_sha256"] = npu.file_sha256(train)
    with pytest.raises(ValueError, match="Held-out"):
        calibrate()


def test_wrong_soc_stops_before_transfer(monkeypatch, tmp_path):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="Exynos\n")

    monkeypatch.setattr(device.subprocess, "run", run)
    with pytest.raises(ValueError, match="requires SM8850"):
        device.test_on_device(tmp_path, tmp_path, "phone", tmp_path / "report")
    assert calls == [["adb", "-s", "phone", "shell", "getprop", "ro.soc.model"]]
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("changed_manifest", [False, True])
def test_export_publishes_bound_scoring_manifest_atomically(
    tmp_path, monkeypatch, changed_manifest
):
    checkpoint = tmp_path / "trained"
    checkpoint.mkdir()
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    split = prepared / npu.SPLIT
    split.write_bytes((npu.BIXBY / npu.SPLIT).read_bytes())
    digest = npu.file_sha256(split)
    manifest = prepared / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "splits": {
                    "bixby50": {
                        "output_sha256": digest,
                        "benchmark": npu.read_json(
                            npu.BIXBY / "benchmark_manifest.json"
                        ),
                    }
                }
            }
        )
    )
    config = {"final_evaluation_datasets": {"bixby50": {"split_path": str(split)}}}
    requests = [
        {"id": k, "input_ids": [2], "prompt": "input"} for k in APPROVED_SOURCE_IDS
    ]
    limits = {
        "bixby50_split_sha256": digest,
        "bixby50_manifest_sha256": npu.file_sha256(manifest),
        "cache_length": 256,
        "max_output_tokens": 4,
        "max_input_tokens": 16,
    }
    monkeypatch.setattr(npu.platform, "system", lambda: "Linux")
    monkeypatch.setattr(npu.platform, "machine", lambda: "x86_64")
    installed_version = npu.importlib.metadata.version
    monkeypatch.setattr(
        npu.importlib.metadata,
        "version",
        lambda d: npu.VERSIONS.get(d) or installed_version(d),
    )
    monkeypatch.setattr(npu, "checkpoint_binding", lambda *args: (config, {}))
    monkeypatch.setattr(npu, "prepare_requests", lambda *args: (requests, [], limits))

    def convert(profile, source, work, cache_length, calibration):
        package(work, profile, capacity=cache_length)
        if changed_manifest:
            manifest.write_text("{}")

    monkeypatch.setattr(npu, "_convert", convert)
    out = tmp_path / "out"
    args = ("270m", checkpoint, tmp_path / "config", out, 256)
    if changed_manifest:
        with pytest.raises(ValueError, match="scoring context changed"):
            npu.export_checkpoint(*args)
        assert not out.exists()
    else:
        assert npu.export_checkpoint(*args) == out / npu.PACKAGE
        assert device.validate_export(out)[1] == requests
        assert len(npu.load_fixed_golden_rows(out / npu.SPLIT, required_rows=50)) == 50
        with pytest.raises(FileExistsError):
            npu.export_checkpoint(*args)


def test_prepared_requests_use_recorded_split_outside_training_and_sft_template(
    monkeypatch, tmp_path
):
    from ir_training.eval import prepared_contract
    from ir_training.models import registry

    train = tmp_path / "augmented"
    train.mkdir()
    (train / "manifest.json").write_text("{}")
    (tmp_path / "training_metadata.json").write_text(
        json.dumps(
            {
                "prepared_dataset_binding": {
                    "manifest_sha256": npu.file_sha256(train / "manifest.json")
                }
            }
        )
    )
    held = tmp_path / "prepared"
    cohorts = {}
    for name in ("golden32", "golden35", "bixby50"):
        path = held / f"{name}.jsonl"
        rows = [{"id": name}]
        if name == "bixby50":
            rows = list(npu.read_jsonl(npu.BIXBY / npu.SPLIT))
            for row in rows:
                row["messages"] = [{"role": "user", "content": row["response_text"]}]
        write_jsonl(path, rows)
        cohorts[name] = {"split_sha256": npu.file_sha256(path), "split_path": str(path)}
    (held / "manifest.json").write_text(
        json.dumps(
            {
                "splits": {
                    "bixby50": {
                        "output_sha256": cohorts["bixby50"]["split_sha256"],
                        "benchmark": npu.read_json(
                            npu.BIXBY / "benchmark_manifest.json"
                        ),
                    }
                }
            }
        )
    )
    cohorts["bixby50"]["manifest_sha256"] = npu.file_sha256(held / "manifest.json")
    config = {
        "model": {"max_output_tokens": 8, "max_context_tokens": 8192},
        "run": {"dataset_dir": str(train)},
        "golden_eval": {"max_input_tokens": 64},
        "training": {"max_seq_length": 64},
        "final_evaluation_datasets": cohorts,
    }

    def tokenize(text, *, add_special_tokens):
        assert text.startswith("saved prompt ") and not add_special_tokens
        return {"input_ids": [2, 17]}

    def render(row, *, tokenizer, include_assistant):
        assert tokenizer is tokenize and not include_assistant
        return "saved prompt " + row["messages"][-1]["content"]

    adapter = SimpleNamespace(format_example=render, load_tokenizer=lambda: tokenize)
    monkeypatch.setattr(registry, "create_adapter", lambda opts: adapter)
    monkeypatch.setattr(npu, "checked_preparation_manifest", lambda path: {})
    monkeypatch.setattr(
        prepared_contract,
        "verify_golden_preparation",
        lambda *args, **kwargs: cohorts["bixby50"],
    )
    monkeypatch.setattr(
        prepared_contract, "verify_loaded_evaluation_tokenizer", lambda *args: None
    )
    monkeypatch.setattr(npu, "training_calibration", lambda *args: ([], {}))
    requests, _, proof = npu.prepare_requests("270m", tmp_path, config, 128)
    assert [row["id"] for row in requests] == list(APPROVED_SOURCE_IDS)
    assert all(row["input_ids"] == [2, 17] for row in requests)
    assert proof["bixby50_split_sha256"] == cohorts["bixby50"]["split_sha256"]
    config["golden_eval"]["max_input_tokens"] = 1
    with pytest.raises(ValueError, match="prompt limit 1"):
        npu.prepare_requests("270m", tmp_path, config, 128)


def test_only_complete_native_holdout_reaches_canonical_scorer(tmp_path, monkeypatch):
    results, report = tmp_path / "results", tmp_path / "report"
    results.mkdir()
    (results / "backend.txt").write_text("qualcomm-htp\nmtp-off\n")
    index = "\n".join(f"{key}\t2\t1.5" for key in APPROVED_SOURCE_IDS)
    (results / "index.tsv").write_text(index)
    requests = [{"id": key, "input_ids": [1, 2]} for key in APPROVED_SOURCE_IDS]
    for key in APPROVED_SOURCE_IDS[:-1]:
        (results / f"{key}.out").write_text("raw output")
    with pytest.raises(ValueError, match="Incomplete native"):
        device.score_results(results, report, requests, {}, npu.BIXBY / npu.SPLIT)
    assert not report.exists()
    (results / f"{APPROVED_SOURCE_IDS[-1]}.out").write_text("raw output")
    monkeypatch.delenv("A2UI_ANDROID_REPAIR_RUNTIME", raising=False)
    device.score_results(
        results, report, requests, {"mtp_enabled": False}, npu.BIXBY / npu.SPLIT
    )
    assert json.loads((report / "aggregate_metrics.json").read_text(encoding="utf-8"))
    assert (
        len(
            (report / "scored_predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        == 50
    )
    rows = [
        json.loads(line)
        for line in (report / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [r["id"] for r in rows] == list(APPROVED_SOURCE_IDS)
    (results / "backend.txt").write_text("gpu\nmtp-off\n")
    with pytest.raises(ValueError, match="HTP evidence"):
        device.score_results(results, report, requests, {}, npu.BIXBY / npu.SPLIT)
