"""Opt-in student-feedback artifacts around the existing preparation and scorer.

No synthetic data is generated here. Probing calls the existing QAT evaluator;
feedback building never loads a student or contacts Muse. Frozen splits stay put.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ir_training.data.student_feedback import (
    VERSION, FeedbackOptions, build_feedback, canonical_json, fingerprint,
    select_probe_indices, text_hash,
)


@contextmanager
def offline_hf():
    """No implicit model/tokenizer downloads; restore the caller's environment."""
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def write_rows(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")


class IndexedRows(Sequence):
    """Keep byte offsets, not hundreds of thousands of repeated chat scaffolds."""
    def __init__(self, path: Path):
        self.path = path
        self.offsets = []
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            value = json.loads(stream.readline().decode("utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("Feedback inputs must be JSON objects")
        return value


def inventory(directory: Path) -> dict[str, str]:
    from ir_training.common.cache_store import assert_no_links
    assert_no_links(directory)
    files = sorted(directory.iterdir())
    if not files or any(not p.is_file() or p.is_symlink() for p in files):
        raise ValueError("A frozen artifact directory must contain regular files only")
    return {p.name: sha_file(p) for p in files}


def check_inventory(directory: Path, expected: dict[str, str]) -> None:
    if inventory(directory) != expected:
        raise ValueError(f"Artifact inventory changed: {directory}")


def new_output(output: Path, *protected: Path) -> None:
    from ir_training.common.cache_store import assert_no_links
    assert_no_links(output)
    for path in protected:
        path = path.resolve()
        if output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError("Output must be disjoint from all input artifacts")
    output.mkdir(parents=True, exist_ok=False)


def checked_prepared(directory: Path):
    """Reserve val, named holdouts and canonical/replaced sources transitively."""
    from ir_training.data.audit_filter import _identity_keys, _source_hashes, load_reserved_cohorts
    from ir_training.data.golden_replacement import source_identity_record
    from ir_training.eval.prepared_contract import checked_preparation_manifest
    from ir_training.common.config import repo_root
    from ir_training.pipeline.golden_training import GOLDENS
    directory = directory.resolve()
    files = inventory(directory)
    manifest = checked_preparation_manifest(directory)
    splits = manifest.get("splits") or {}
    if not {"train", "val"} <= splits.keys():
        raise ValueError("Separate checked train and val splits are required")
    for name, entry in splits.items():
        if Path(name).name != name or files.get(name + ".jsonl") != entry.get("output_sha256"):
            raise ValueError("Prepared split hash differs from manifest")
    for name in ("train", "val"):
        admitted, original = splits[name].get("accepted_rows"), splits[name].get("input_rows")
        if (type(admitted) is not int or type(original) is not int or original < 1
                or admitted < 1 or admitted / original < .5):
            raise ValueError("Feedback requires at least 50% prepared train/val retention; fix coverage first")
    paths = [directory / f"{name}.jsonl" for name in splits if name != "train"]
    paths.extend(repo_root() / item[0] for item in GOLDENS.values())
    reserved = load_reserved_cohorts(paths)
    rows = IndexedRows(directory / "train.jsonl")
    if not rows or len(rows) != splits["train"].get("accepted_rows"):
        raise ValueError("Prepared training row count differs from manifest")
    parents, owners, blocked, ids = list(range(len(rows))), {}, [], set()

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, row in enumerate(rows):
        if (i + 1) % 10000 == 0:
            print(f"Feedback source-family scan: {i+1}/{len(rows)}", flush=True)
        identity = source_identity_record(row)
        row_id = identity.get("row_id")
        if not row_id or not identity.get("source_id") or row_id in ids:
            raise ValueError("Feedback requires unique prepared row IDs and source identities")
        ids.add(row_id)
        keys, hashes = _identity_keys(identity), _source_hashes(row)
        augmentation = row.get("metadata", {}).get("augmentation") or {}
        # Include donor lineage even when a variant's source text/IDs differ.
        keys |= {str(augmentation[key]) for key in ("donor_id", "source_group_id") if augmentation.get(key)}
        blocked.append(bool(keys & reserved["identities"] or hashes & reserved["responses"])
                       or row.get("evaluation_only") is True or row.get("reference_available") is False)
        for key in sorted({"id:" + k for k in keys} | {"text:" + h for h in hashes}):
            if key in owners:
                parents[find(i)] = find(owners[key])
            else:
                owners[key] = i
    bad_families = {find(i) for i, flag in enumerate(blocked) if flag}
    if bad_families:
        raise ValueError("Prepared training overlaps a reserved evaluation source family")
    families = [str(find(i)) for i in range(len(rows))]
    check_inventory(directory, files)
    return rows, families, manifest, files, reserved["evidence"]


def loading_config(source_config: Path, checkpoint: Path, prepared: Path, output: Path,
                   *, seed=42, accumulation=16) -> dict:
    """Reuse verified SFT/QAT *loading* gates, not the GRPO optimizer/objective.

    The temporary method is required by that loader's contract. Feedback training
    writes its own experimental provenance and never claims to have run GRPO.
    """
    from ir_training.common.config import load_yaml
    cfg = deepcopy(load_yaml(source_config))
    if cfg.get("training", {}).get("method") != "qat_lora_sft":
        raise ValueError("Feedback v1 starts from a verified QAT LoRA SFT adapter")
    model_id = str(cfg.get("model", {}).get("model_id", "")).lower()
    family = "e2b" if "gemma-4-e2b" in model_id else "270m" if "gemma-3-270m" in model_id else None
    if family is None:
        raise ValueError("Feedback QAT loader currently supports E2B and 270M LoRA, not full-QAT models")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    cfg["run"] = {"id": output.name, "output_dir": str(output), "dataset_dir": str(prepared),
                  "purpose": "qat_lora_grpo_v1"}
    train = cfg["training"]
    for name in ("resume_from_checkpoint", "resume_policy", "deepspeed", "fsdp", "fsdp_config"):
        train.pop(name, None)
    cfg.pop("resume_from_checkpoint", None)
    cfg.pop("qat_mtp", None)
    train.update(method="qat_lora_grpo", seed=seed, logging_steps=1,
        per_device_train_batch_size=1, gradient_accumulation_steps=accumulation,
        distributed_backend="ddp", expected_world_size=world,
        expected_effective_batch_size=world * accumulation,
        ddp_broadcast_buffers=False, gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg["grpo"] = {"family": family, "sft_checkpoint": str(checkpoint),
                   "sft_training_config": str(source_config), "beta": 0, "num_generations": 4}
    return cfg


def make_scorer():
    from pipeline.genui_quality import (
        ensure_v5_4_validation_ready, load_reward_config_v5_4,
        prepare_source_context_v5_4, score_completion_group_v5_4,
    )
    from pipeline.genui_quality.acceptance_v5_4 import training_acceptance_v5_4
    from ir_training.eval.generate import build_prediction_record
    from ir_training.data.url_preprocess import restore_url_placeholders
    from ir_training.common.config import repo_root
    ensure_v5_4_validation_ready()
    cfg = load_reward_config_v5_4(repo_root() / "dataset/configs/genui_metric_v5_4.yaml")
    contexts = OrderedDict()

    def score(row, text):
        key = fingerprint(row)
        if key not in contexts:
            bound = build_prediction_record(row, "")
            urls = bound["url_map"]
            contract = bound.get("expected_ui_contract_v5_4")
            if contract is None:
                contract = bound.get("expected_ui_contract")
            contexts[key] = (prepare_source_context_v5_4(bound["response_text"],
                intent=bound.get("intent_bucket") or bound.get("intent"),
                assets=restore_url_placeholders(bound["assets"], urls),
                expected_ui_contract=restore_url_placeholders(contract, urls), config=cfg), urls)
            if len(contexts) > 32:
                contexts.popitem(last=False)
        context, urls = contexts[key]
        result = score_completion_group_v5_4([restore_url_placeholders(text, urls)], context,
                                            generation_mode=True, active_express=True)[0]
        decision = training_acceptance_v5_4(result)
        strict = (result.evidence.get("raw_express_envelope", {}).get("exact_single_express_block") is True
                  and all(result.normalization.get(name) is True for name in
                    ("native_syntax_valid", "native_catalog_valid", "production_valid", "strict_schema_valid")))
        blockers = list(decision["blocking_reasons"])
        if not strict:
            blockers.append("raw_express_invalid")
        return {"eligible": bool(decision["eligible"] and strict), "strict_valid": strict,
                "quality": result.quality_0_1, "blocking_reasons": list(dict.fromkeys(blockers)),
                "review_reasons": decision["review_reasons"], "metric_fingerprint": result.metric_fingerprint,
                "acceptance_policy": decision["policy_version"]}
    return score


@offline_hf()
def probe(prepared: Path, source_config: Path, checkpoint: Path, output: Path, *,
          max_rows=256, seed=42, max_new_tokens=2048, execute=False):
    """Greedy deployment-like failures, single process; no optimization or Muse."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Probe v1 is single-process; select one GPU with CUDA_VISIBLE_DEVICES")
    rows, families, manifest, source_files, reserved = checked_prepared(prepared)
    indices = select_probe_indices(rows, families, max_rows=max_rows, seed=seed)
    cfg = loading_config(source_config, checkpoint, prepared, output, seed=seed)
    from ir_training.train.qat_grpo_contract import verify_sft_adapter_lineage
    lineage = verify_sft_adapter_lineage(cfg)
    if lineage.get("verified") is not True:
        raise ValueError(f"Source QAT SFT lineage failed: {lineage}")
    plan = {"version": VERSION, "source_dir": str(prepared), "source_files": source_files,
            "source_config": str(source_config), "source_config_sha256": sha_file(source_config),
            "checkpoint": str(checkpoint), "checkpoint_files": inventory(checkpoint),
            "source_lineage": lineage, "reserved": reserved, "indices": indices,
            "producer_sha256": {"pipeline": sha_file(Path(__file__)),
                "decisions": sha_file(Path(__file__).parents[1] / "data/student_feedback.py")},
            "seed": seed, "max_new_tokens": max_new_tokens, "qat": True,
            "mode": "greedy_training_source_probe", "optimizer_steps": 0}
    if not execute:
        return {"status": "plan_only", **plan}
    import torch
    from transformers import set_seed
    from ir_training.models.registry import create_adapter
    from ir_training.train.prepared_binding import verify_tokenizer_binding
    from ir_training.eval.generate import generate_predictions, build_prediction_record
    from ir_training.train.grpo_runtime import prepared_prompt_messages, render_chat_prompt
    from ir_training.common.config import load_yaml
    if not torch.cuda.is_available():
        raise RuntimeError("QAT student probing requires CUDA")
    source = load_yaml(source_config)
    adapter = create_adapter(source["model"])
    tokenizer = adapter.load_tokenizer()
    verify_tokenizer_binding(prepared, tokenizer, source["model"], required=True)
    max_prompt = manifest["tokenizer"]["max_input_tokens"]
    context = source["model"].get("max_context_tokens")
    if type(max_new_tokens) is not int or max_new_tokens < 1 or not context or max_prompt + max_new_tokens > context:
        raise ValueError("Probe token budgets exceed the declared context")
    selected, requests = [], []
    for index in indices:
        row = rows[index]
        prompt = render_chat_prompt(tokenizer, prepared_prompt_messages(row), source["model"].get("chat_template_kwargs"))
        if adapter.format_example(row, tokenizer=tokenizer, include_assistant=False) != prompt:
            raise ValueError("Prepared and generation prompt boundaries differ")
        selected.append(row)
        requests.append({"train_index": index, "row_sha256": fingerprint(row), "prompt_sha256": text_hash(prompt)})
    new_output(output, prepared, checkpoint, source_config)
    write_rows(output / "inputs.jsonl", selected)
    write_rows(output / "requests.jsonl", requests)
    write_json(output / "plan.json", plan)
    set_seed(seed)
    count = generate_predictions(source, output / "inputs.jsonl", output / "predictions.jsonl",
        max_input_tokens=max_prompt, max_new_tokens=max_new_tokens, adapter_checkpoint=checkpoint, apply_qat=True)
    predictions = IndexedRows(output / "predictions.jsonl")
    if count != len(indices) or len(predictions) != len(indices):
        raise ValueError("Incomplete student probe; no completion receipt published")
    observations = []
    for request, row, prediction in zip(requests, selected, predictions, strict=True):
        bound = build_prediction_record(row, "")
        if (prediction.get("source_context_sha256") != bound["source_context_sha256"]
                or prediction.get("runtime", {}).get("prompt_sha256") != request["prompt_sha256"]):
            raise ValueError("Probe prediction source/prompt binding mismatch")
        observations.append({**request, "completion": prediction["generated_text"],
                             "stop_reason": prediction["runtime"]["stop_reason"]})
    write_rows(output / "observations.jsonl", observations)
    check_inventory(prepared, source_files)
    check_inventory(checkpoint, plan["checkpoint_files"])
    if sha_file(source_config) != plan["source_config_sha256"]:
        raise ValueError("Source config changed during probing")
    receipt = {"version": VERSION, "complete": True, "files": inventory(output)}
    write_json(output / "receipt.json", receipt)  # Commit marker written last.
    return {"status": "probe_complete", "rows": count, "output": str(output)}


def checked_artifact(directory: Path, marker: str):
    files = inventory(directory)
    saved = json.loads((directory / marker).read_text(encoding="utf-8"))
    if saved.get("version") != VERSION or saved.get("complete") is not True:
        raise ValueError("Incomplete or unsupported feedback artifact")
    files.pop(marker)
    required = ({"plan.json", "inputs.jsonl", "requests.jsonl", "predictions.jsonl", "observations.jsonl"}
                if marker == "receipt.json" else
                {"binding.json", "replay.json", "preferences.jsonl", "diagnostics.jsonl", "report.json"})
    if not required <= files.keys():
        raise ValueError("Feedback artifact is missing required evidence")
    if files != saved.get("files"):
        raise ValueError("Feedback artifact bytes changed")
    return saved


def checked_observations(probe_dir: Path, plan: dict, rows):
    """Rejoin raw predictions, request IDs and frozen inputs before mining."""
    from ir_training.eval.generate import build_prediction_record
    from ir_training.generation_policy import stop_express_completion
    observations = list(IndexedRows(probe_dir / "observations.jsonl"))
    requests = IndexedRows(probe_dir / "requests.jsonl")
    inputs = IndexedRows(probe_dir / "inputs.jsonl")
    predictions = IndexedRows(probe_dir / "predictions.jsonl")
    if any(len(items) != len(plan["indices"]) for items in (observations, requests, inputs, predictions)):
        raise ValueError("Incomplete or duplicated probe evidence")
    for index, observation, request, input_row, prediction in zip(
            plan["indices"], observations, requests, inputs, predictions, strict=True):
        row = rows[index]
        runtime = prediction.get("runtime") or {}
        raw = prediction.get("raw_generated_text")
        if (input_row != row or request.get("train_index") != index
                or request.get("row_sha256") != fingerprint(row) or not isinstance(raw, str)
                or runtime.get("prompt_sha256") != request.get("prompt_sha256")
                or prediction.get("source_context_sha256") != build_prediction_record(row, "")["source_context_sha256"]
                or prediction.get("generated_text") != stop_express_completion(raw)
                or observation != {**request, "completion": prediction.get("generated_text"),
                                   "stop_reason": runtime.get("stop_reason")}):
            raise ValueError("Probe prediction/request/source binding differs")
    return observations


def build(probe_dir: Path, output: Path, options: FeedbackOptions):
    checked_artifact(probe_dir, "receipt.json")
    plan = json.loads((probe_dir / "plan.json").read_text(encoding="utf-8"))
    current_producer = {"pipeline": sha_file(Path(__file__)),
                        "decisions": sha_file(Path(__file__).parents[1] / "data/student_feedback.py")}
    if plan.get("producer_sha256") != current_producer:
        raise ValueError("Feedback producer changed since the probe; do not silently reinterpret evidence")
    prepared = Path(plan["source_dir"])
    rows, families, manifest, files, reserved = checked_prepared(prepared)
    if files != plan["source_files"] or reserved != plan["reserved"]:
        raise ValueError("Prepared data or reserved cohorts changed since probing")
    check_inventory(Path(plan["checkpoint"]), plan["checkpoint_files"])
    if sha_file(Path(plan["source_config"])) != plan["source_config_sha256"]:
        raise ValueError("Probe source config changed")
    observations = checked_observations(probe_dir, plan, rows)
    result = build_feedback(rows, families, observations, score=make_scorer(), options=options)
    new_output(output, prepared, probe_dir, Path(plan["checkpoint"]), Path(plan["source_config"]))
    write_rows(output / "preferences.jsonl", result["preferences"])
    write_rows(output / "diagnostics.jsonl", result["diagnostics"])
    write_json(output / "replay.json", result["replay_indices"])
    write_json(output / "report.json", result["report"])
    binding = {"version": VERSION, "source_dir": str(prepared), "source_files": files,
               "reserved": reserved, "probe_dir": str(probe_dir), "probe_receipt_sha256": sha_file(probe_dir / "receipt.json"),
               "probe_plan": plan, "options": result["report"]["options"],
               "training_semantics": "Frozen original rows plus bounded replay indices; no target rewrites",
               "export_qualified": False}
    write_json(output / "binding.json", binding)
    check_inventory(prepared, files)
    checked_artifact(probe_dir, "receipt.json")
    write_json(output / "manifest.json", {"version": VERSION, "complete": True, "files": inventory(output)})
    return result["report"]


def load_feedback(directory: Path):
    checked_artifact(directory, "manifest.json")
    binding = json.loads((directory / "binding.json").read_text(encoding="utf-8"))
    probe_dir = Path(binding["probe_dir"])
    checked_artifact(probe_dir, "receipt.json")
    if sha_file(probe_dir / "receipt.json") != binding["probe_receipt_sha256"]:
        raise ValueError("Feedback refers to a changed probe")
    rows, families, manifest, files, reserved = checked_prepared(Path(binding["source_dir"]))
    if files != binding["source_files"] or reserved != binding["reserved"]:
        raise ValueError("Feedback source or holdouts changed")
    replay = json.loads((directory / "replay.json").read_text(encoding="utf-8"))
    preferences = list(IndexedRows(directory / "preferences.jsonl"))
    # Rescore the small probe, not the entire training set, before trusting a
    # published replay plan. This also rejects incompatible scorer revisions.
    expected = build_feedback(rows, families, checked_observations(probe_dir, binding["probe_plan"], rows),
                              score=make_scorer(), options=FeedbackOptions(**binding["options"]))
    if (replay != expected["replay_indices"] or preferences != expected["preferences"]
            or json.loads((directory / "report.json").read_text(encoding="utf-8")) != expected["report"]
            or list(IndexedRows(directory / "diagnostics.jsonl")) != expected["diagnostics"]):
        raise ValueError("Feedback decisions no longer match the existing admission policy")
    return rows, families, replay, preferences, binding, manifest
