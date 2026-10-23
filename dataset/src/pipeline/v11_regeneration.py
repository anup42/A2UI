"""Opt-in, immutable v11 Stage 3 regeneration; never called by training.

Input rows remain in their original split. Unselected lines are copied byte for
byte. Failed selected labels and unsafe sources are quarantined, never replaced
by stale supervision. Reports describe checks, not independent fact verification.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

from pipeline.training_augmentation import (
    DEFAULT_TEACHER,
    admission_errors,
    bounded_teacher_environment,
)
from pipeline.v11_regeneration_io import (
    ROOT,
    SplitInput,
    atomic_bytes,
    canonical,
    checked,
    digest,
    file_hash,
    safe_path,
    save,
)

VERSION = "v11-regeneration-v1"
DATASET = ROOT / "dataset"
SUPPLEMENT = DATASET / "prompts/v11_regeneration_v1.md"


def _guard():
    from pipeline import v11_regeneration_guard
    return v11_regeneration_guard


def _row_id(row: dict) -> str:
    value = row.get("id") or (row.get("metadata") or {}).get("ui_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Every input row requires a stable id")
    metadata = row.get("metadata") or {}
    if (row.get("evaluation_only") or row.get("reference_available") is False
            or metadata.get("benchmark") or metadata.get("benchmark_contract")
            or value.upper().startswith("BXP-")
            or any(name in value.lower() for name in ("golden32", "golden33", "golden35", "bixby50"))):
        raise ValueError("Golden/Bixby/evaluation-only rows are not regeneration inputs")
    return value


def _benchmark_exclusions():
    """Read only pinned input identities for exclusion, never teacher targets."""
    from ir_training.data.express_preparation import TASK_PREFIX
    from ir_training.data.golden_replacement import response_text
    from ir_training.data.url_preprocess import restore_url_placeholders
    from ir_training.pipeline.golden_training import GOLDENS
    identities, sources, files = set(), set(), {}
    for relative, _, expected in GOLDENS.values():
        path = ROOT / relative
        if file_hash(path) != expected:
            raise ValueError("Pinned benchmark exclusion source changed")
        files[relative] = expected
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                identities.update(str(value) for value in (row.get("id"), row.get("source_id"),
                    (row.get("metadata") or {}).get("source_id")) if value)
                texts = {response_text(row)}
                # Historical benchmark aliases can differ in masking. Exclude
                # both source forms without treating that as a relabeling input.
                for message in reversed(row.get("messages", [])):
                    text = message.get("content", "")
                    if message.get("role") == "user" and text.startswith(TASK_PREFIX):
                        texts.add(text[len(TASK_PREFIX):])
                        break
                mapping = _reference_map(row)
                texts.update(restore_url_placeholders(text, mapping) for text in tuple(texts))
                sources.update(digest(" ".join(text.split())) for text in texts)
    return identities, sources, files


def _config(*, teacher_model, max_samples, max_attempts, batch_size, max_tokens,
            prompt_max_tokens, context_tokens, seed):
    if any(type(value) is not int or value < 1 for value in
           (max_attempts, batch_size, max_tokens, prompt_max_tokens, context_tokens)):
        raise ValueError("Budgets, batch size and max attempts must be positive integers")
    if type(max_samples) is not int or max_samples < 0 or type(seed) is not int or seed < 0:
        raise ValueError("max_samples and seed must be nonnegative integers")
    if max_attempts > 5 or batch_size > 128:
        raise ValueError("Bounded regeneration permits at most 5 attempts and batch size 128")
    if max_tokens + prompt_max_tokens + 512 > context_tokens:
        raise ValueError("Prompt + output budgets + 512 safety tokens exceed server context")
    from utils.config import load_yaml
    specs = [item for item in load_yaml(DATASET / "configs/models.yaml").get("models", [])
             if item.get("name") == teacher_model]
    if len(specs) != 1 or specs[0].get("provider") != "local" or specs[0].get("model") != "meta-models/Muse-Glimmer-30B":
        raise ValueError("Select exactly one registered local Muse model; no remote fallback")
    supplement = SUPPLEMENT.read_text(encoding="utf-8").strip()
    if not supplement or "{response_text}" in supplement:
        raise ValueError("Missing or invalid opt-in regeneration supplement")
    paths = {Path(__file__), DATASET / "scripts/regenerate_v11_samples.py", ROOT / "training/data/train/v11/restore.py"}
    for directory in (DATASET / "src/pipeline", DATASET / "src/llm", DATASET / "src/utils",
                      ROOT / "training/src/ir_training/data", ROOT / "training/src/ir_training/eval"):
        paths.update(directory.rglob("*.py"))
    paths.update((DATASET / "schema").rglob("*.json"))
    paths.update((DATASET / "configs").rglob("*.yaml"))
    paths.update((DATASET / "configs").rglob("*.json"))
    paths.update((DATASET / "prompts").glob("*.md"))
    paths.add(DATASET / "configs/models.yaml")
    paths.add(ROOT / "training/data/quality/v11_review_holds_20261001.json")
    # Embedded callers can start more than one opt-in run in one interpreter.
    # Refresh the canonical catalogue so its cached contents match the bytes
    # bound below; resume and publication reject any subsequent policy change.
    from ir_training.data.release_review import review_policy
    review_policy.cache_clear()
    review_policy()
    # Store a hash, not potentially credential-bearing endpoints/environment.
    environment = {key: value for key, value in os.environ.items()
                   if key.startswith(("LOCAL_", "STAGE3_", "GENUI_", "A2UI_GENUI_"))}
    _, _, benchmark_files = _benchmark_exclusions()
    return {"version": VERSION, "teacher_model": teacher_model, "teacher_spec": specs[0],
            "max_samples": max_samples, "max_attempts": max_attempts, "batch_size": batch_size,
            "grpo_entity_extraction": True,
            "max_tokens": max_tokens, "prompt_max_tokens": prompt_max_tokens,
            "context_tokens": context_tokens, "seed": seed,
            "environment_sha256": digest(environment), "guard": _guard().guard_identity(),
            "benchmark_exclusion_hashes": benchmark_files,
            "code_prompt_schema_hashes": {path.relative_to(ROOT).as_posix(): file_hash(path) for path in sorted(paths)}}


def _scan(source: SplitInput, config: dict) -> tuple[list[dict], dict]:
    guard = _guard()
    benchmark_ids, benchmark_sources, _ = _benchmark_exclusions()
    selection = []
    ids = set()
    selected = checked_targets = deferred = held = 0
    for split in ("train", "val"):
        for line, raw, row in source.rows(split):
            row_id = _row_id(row)
            aliases = {row_id, str(row.get("source_id", "")), str((row.get("metadata") or {}).get("source_id", ""))}
            source_variants = {guard.source_text(row), _generation_source(row)}
            if aliases & benchmark_ids or any(digest(" ".join(text.split())) in benchmark_sources for text in source_variants):
                raise ValueError("Pinned Golden/Bixby source is excluded from regeneration")
            if row_id in ids:
                raise ValueError(f"Duplicate row id across train/val: {row_id}")
            ids.add(row_id)
            if len(ids) % 1000 == 0:
                print(f"v11 audit: rows={len(ids)}, selected={selected}, source_held={held}", file=sys.stderr)
            assigned = (row.get("metadata") or {}).get("assigned_split", split)
            if assigned != split:
                raise ValueError(f"Original assigned split mismatch: {row_id}")
            source_check = guard.assess_source(row)
            if source_check.get("eligible") is not True:
                status, reasons = "source_held", source_check.get("reasons", [])
                held += 1
            elif config["max_samples"] and selected >= config["max_samples"]:
                deferred += 1
                continue
            else:
                checked_targets += 1
                target_check = guard.assess_candidate(row, row.get("completion", row.get("a2ui_express")))
                if target_check.get("eligible") is True:
                    continue
                status, reasons = "pending", target_check.get("reasons", [])
                selected += 1
            selection.append({"id": row_id, "split": split, "line": line,
                              "row_sha256": hashlib.sha256(raw).hexdigest(),
                              "source_sha256": hashlib.sha256(guard.source_text(row).encode()).hexdigest(),
                              "status": status, "reasons": reasons})
    source.recheck_sidecars()
    _check_runtime_identity(config)
    return selection, {"input_rows": len(ids), "selected": selected, "source_held": held,
                       "targets_checked": checked_targets, "targets_deferred_by_limit": deferred,
                       "unchanged": len(ids) - selected - held}


def _check_runtime_identity(config):
    for relative, expected in config["code_prompt_schema_hashes"].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError("Bound code/prompt/schema/config changed during run; use a fresh output")
    if _guard().guard_identity() != config["guard"]:
        raise ValueError("Source/candidate guard changed during run")


def _state_path(output: Path, entry: dict) -> Path:
    return output / "journal" / (digest({"split": entry["split"], "id": entry["id"]}) + ".json")


def _load_state(output: Path, entry: dict, contract_hash: str) -> dict:
    path = _state_path(output, entry)
    if not path.exists():
        return {"entry": entry, "contract_sha256": contract_hash, "attempts": [], "status": entry["status"]}
    state = checked(path)
    if state.get("entry") != entry or state.get("contract_sha256") != contract_hash:
        raise ValueError("Row journal identity/source/selection mismatch")
    for attempt in state.get("attempts", []):
        artifact = safe_path(output / attempt["artifact"])
        if output not in artifact.parents or file_hash(artifact) != attempt["sha256"]:
            raise ValueError("Attempt evidence changed or escaped output")
        _verify_receipt(artifact)
    if state.get("status") == "accepted" and not isinstance(state.get("replacement"), dict):
        raise ValueError("Accepted journal is missing replacement")
    return state


def _verify_receipt(path: Path) -> dict:
    receipt = checked(path)
    names = {child.relative_to(path.parent).as_posix() for child in path.parent.rglob("*")
             if child.is_file() and child.name not in {"receipt.json", "receipt.json.pending"}}
    if names != set(receipt["files"]):
        raise ValueError("Sealed Stage 3 child inventory changed")
    for name, expected in receipt["files"].items():
        child = safe_path(path.parent / name)
        if path.parent not in child.parents or file_hash(child) != expected:
            raise ValueError("Sealed Stage 3 child evidence changed")
    return receipt


@contextmanager
def _exclusive(output: Path):
    """OS-owned lock releases on process death; a stale filename is harmless."""
    path = safe_path(output / "regeneration.lock")
    with path.open("a+b") as stream:
        if not path.stat().st_size:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("Another regeneration process owns this output directory") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _reference_map(row: dict) -> dict[str, str]:
    from ir_training.data.reference_binding import (
        _reference_map as checked_reference_map,
    )
    return checked_reference_map(row)


def _generation_source(row: dict) -> str:
    from ir_training.data.url_preprocess import restore_url_placeholders
    return restore_url_placeholders(_guard().source_text(row), _reference_map(row))


def _candidate_target(row: dict, candidate: dict) -> str:
    from ir_training.data.express_preparation import serialize_checked
    from ir_training.data.url_preprocess import restore_url_placeholders

    from pipeline.ir_formats import express
    from pipeline.stage3_genui import _MODEL_REFERENCE_RE, split_reference_suffix
    target = serialize_checked(candidate["a2ui_express"], "root-first")
    # Canonical Stage 3 restored real references. Rebind only the original
    # registered references to the immutable training-source placeholders.
    mapping = _reference_map(row)
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Ambiguous inverse reference aliases; manual target review required")
    reverse = {value: token for token, value in mapping.items()}

    def replace(match):
        quote = match.group("quote") or ""
        raw = match.group("quoted_local") or match.group(0)
        raw, trailing = (raw, "") if quote else split_reference_suffix(raw)
        return quote + reverse.get(raw, raw) + trailing + quote

    def remask(value):
        if isinstance(value, dict):
            return {key: remask(item) for key, item in value.items()}
        if isinstance(value, list):
            return [remask(item) for item in value]
        if isinstance(value, str):
            return reverse.get(value, _MODEL_REFERENCE_RE.sub(replace, value))
        return value

    graph = remask(target.graph)
    if restore_url_placeholders(graph, mapping) != target.graph:
        raise ValueError("Reference rebinding changed generated semantics")
    return serialize_checked(express.encode(graph, shorten_ids=False), "root-first").text


def _replacement(row: dict, target: str, evidence: dict) -> dict:
    from ir_training.data.express_preparation import prepare_row
    updated = deepcopy(row)
    updated["completion"] = target
    updated["a2ui_express"] = target
    updated["completion_targets"] = {"a2ui_express_v1": target}
    updated["messages"][-1]["content"] = target
    updated.setdefault("metadata", {})["v11_regeneration"] = evidence
    prepared, _, _ = prepare_row(updated, "root-first")
    if _guard().source_text(prepared) != _guard().source_text(row) or _row_id(prepared) != _row_id(row):
        raise ValueError("Regeneration changed immutable source or row identity")
    return prepared


@contextmanager
def _teacher_environment(config: dict):
    with bounded_teacher_environment():
        overrides = {"LOCAL_VLLM_MAX_OUTPUT_TOKENS": str(config["max_tokens"]),
                     "LOCAL_VLLM_BATCH_PARALLELISM": str(config["batch_size"]),
                     "LOCAL_VLLM_PARALLEL_REQUESTS": str(config["batch_size"]),
                     "STAGE3_ROLLING_DISPATCH": "0"}
        previous = {key: os.environ.get(key) for key in overrides}
        os.environ.update(overrides)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _round(output, batch, config, contract_hash, adapter, runner):
    from utils.rate_limit import RateLimiter

    from pipeline.cache import PromptCache
    if runner is None:
        from pipeline.stage3_genui import run_stage3
        runner = run_stage3
    round_key = digest([{ "id": entry["id"], "attempt": len(state["attempts"]) + 1}
                        for entry, _, state in batch])
    inflight = {state.get("inflight") for _, _, state in batch if state.get("inflight")}
    if len(inflight) > 1:
        raise ValueError("Recover one in-flight round at a time")
    directory = safe_path(output / next(iter(inflight))) if inflight else output / "attempts" / round_key
    if output not in directory.parents:
        raise ValueError("In-flight attempt path escaped output")
    if directory.exists():
        # An interrupted invocation is recoverable from its immutable outputs.
        saved = checked(directory / "inputs.json")
        if (saved.get("contract_sha256") != contract_hash
                or any(entry not in saved.get("rows", []) for entry, _, _ in batch)):
            raise ValueError("In-flight attempt input mismatch")
    else:
        directory.parent.mkdir(parents=True, exist_ok=True)
        # Promote only a fully prepared request directory. A kill while staging
        # leaves unreferenced forensic evidence, not an unrecoverable attempt.
        preparing = Path(tempfile.mkdtemp(prefix=".preparing-", dir=directory.parent))
        save(preparing / "inputs.json", {"contract_sha256": contract_hash, "rows": [entry for entry, _, _ in batch]})
        queries, responses = [], []
        for entry, row, state in batch:
            query = dict(row.get("metadata") or {})
            query.update(query_id=entry["id"], query_text=query.get("query_text", "Render the supplied response faithfully."))
            queries.append(query)
            response = {"query_id": entry["id"], "response_id": entry["id"], "n_idx": 1,
                        "response_text": _generation_source(row)}
            if state["attempts"]:
                response["stage3_repair_feedback"] = {"kind": "semantic_fidelity",
                    "review_reasons": state.get("reasons", entry["reasons"]),
                    "attempt": len(state["attempts"]), "max_attempts": config["max_attempts"] - 1}
            responses.append(response)
        for name, rows in (("queries.jsonl", queries), ("responses.jsonl", responses)):
            atomic_bytes(preparing / name, b"".join(canonical(row) + b"\n" for row in rows))
        save(preparing / "input_hashes.json", {name: file_hash(preparing / name) for name in ("queries.jsonl", "responses.jsonl")})
        os.rename(preparing, directory)
    for name, expected in checked(directory / "input_hashes.json").items():
        if file_hash(directory / name) != expected:
            raise ValueError("Attempt request evidence changed")
    for entry, _, state in batch:
        state["inflight"] = directory.relative_to(output).as_posix()
        save(_state_path(output, entry), state)
    fatal = None
    if not (directory / "started.json").exists():
        if adapter is None:
            adapter = _build_teacher(config)
        save(directory / "started.json", {"contract_sha256": contract_hash})
        logger = logging.getLogger("dataset.v11_regeneration")
        try:
            runner(queries_path=directory / "queries.jsonl", responses_path=directory / "responses.jsonl",
                   prompt_path=DATASET / "prompts/genui_gen_mobile_a2ui_express_v1.md",
                   prompt_supplement_path=SUPPLEMENT, adapter=adapter, genui_path=directory / "genui.jsonl",
                   grpo_entity_extraction=config["grpo_entity_extraction"],
                   schema_path=DATASET / "schema/canonical_ui_graph_v1.schema.json", artifacts_dir=directory / "artifacts",
                   candidates_per_response=1, max_repair_attempts=1, max_tokens=config["max_tokens"],
                   prompt_max_tokens=config["prompt_max_tokens"], seed=config["seed"],
                   rate_limiter=RateLimiter(0), cache=PromptCache(directory / "cache.jsonl", enabled=False),
                   logger=logger, batch_size=config["batch_size"], max_total=len(batch), max_attempts=1,
                   metric_version="v5_4", ir_formats=["a2ui_express_v1"], phase_invocation_id=VERSION + ":" + round_key)
        except Exception as exc:  # noqa: BLE001 - persist evidence before reporting infrastructure failure
            fatal = f"{type(exc).__name__}: {exc}"
    else:
        fatal = "Recovered interrupted attempt; missing records consume one bounded attempt"
    records = []
    malformed_lines = []
    genui = directory / "genui.jsonl"
    if genui.exists():
        for number, line in enumerate(genui.read_bytes().splitlines(), 1):
            try:
                record = json.loads(line)
                if (isinstance(record, dict) and isinstance(record.get("response_id"), str)
                        and record["response_id"]):
                    records.append(record)
                else:
                    malformed_lines.append(number)
            except ValueError:
                malformed_lines.append(number)
    receipt = {"files": {path.relative_to(directory).as_posix(): file_hash(path)
                         for path in directory.rglob("*") if path.is_file() and path.name not in {"receipt.json", "receipt.json.pending"}},
               "fatal_error": fatal, "malformed_record_lines": malformed_lines}
    if (directory / "receipt.json").exists():
        old = _verify_receipt(directory / "receipt.json")
        if old["files"] != receipt["files"]:
            raise ValueError("Sealed Stage 3 evidence changed")
        receipt = old
    else:
        save(directory / "receipt.json", receipt)
    grouped = {}
    for record in records:
        grouped.setdefault(record.get("response_id"), []).append(record)
    unknown = set(grouped) - {entry["id"] for entry in checked(directory / "inputs.json")["rows"]}
    for entry, row, state in batch:
        record_path = directory / "receipt.json"
        state["attempts"].append({"artifact": record_path.relative_to(output).as_posix(), "sha256": file_hash(record_path)})
        candidates = grouped.get(entry["id"], [])
        reasons = ["unknown_stage3_source_binding"] if unknown else ["missing_or_duplicate_stage3_candidate"]
        if malformed_lines:
            reasons.append("malformed_stage3_record")
        if len(candidates) == 1 and not unknown and not malformed_lines:
            candidate = candidates[0]
            try:
                reasons = admission_errors(candidate, _generation_source(row))
            except (AttributeError, KeyError, TypeError):
                reasons = ["malformed_stage3_record"]
            if not reasons:
                try:
                    target = _candidate_target(row, candidate)
                    assessment = _guard().assess_candidate(row, target)
                    reasons = assessment.get("reasons", []) if assessment.get("eligible") is not True else []
                    if assessment.get("eligible") is not True and not reasons:
                        reasons = ["candidate_guard_not_eligible"]
                    if not reasons:
                        evidence = {"version": VERSION, "contract_sha256": contract_hash,
                                    "original_row_sha256": entry["row_sha256"],
                                    "source_sha256": entry["source_sha256"], "attempt": state["attempts"][-1],
                                    "assessment": assessment, "original_split": entry["split"]}
                        state["replacement"] = _replacement(row, target, evidence)
                except (ValueError, KeyError, TypeError) as exc:
                    reasons = [f"replacement_validation:{exc}"]
        state["status"] = "accepted" if not reasons else "rejected"
        state["reasons"] = reasons
        state.pop("inflight", None)
        save(_state_path(output, entry), state)
    return receipt["fatal_error"]


def _publish(source, output, selection, contract_hash, summary):
    selected = {(entry["split"], entry["line"]): entry for entry in selection}
    counts = dict(summary, accepted=0, rejected=0)
    identities = {}
    quarantine = output / "quarantine.jsonl.pending"
    with quarantine.open("wb") as rejected:
        for split in ("train", "val"):
            pending = output / f"{split}.jsonl.pending"
            rows = 0
            with pending.open("wb") as stream:
                for line, raw, row in source.rows(split):
                    entry = selected.get((split, line))
                    if entry is None:
                        stream.write(raw)
                        rows += 1
                        continue
                    if hashlib.sha256(raw).hexdigest() != entry["row_sha256"]:
                        raise ValueError("Selected input bytes changed")
                    state = _load_state(output, entry, contract_hash)
                    if state["status"] == "accepted":
                        replacement = state["replacement"]
                        if _guard().assess_candidate(row, replacement["completion"]).get("eligible") is not True:
                            raise ValueError("Accepted candidate no longer passes admission")
                        stream.write(canonical(replacement) + b"\n")
                        rows += 1
                        counts["accepted"] += 1
                    else:
                        if state["status"] not in {"source_held", "rejected"}:
                            raise ValueError("Cannot publish a run with pending rows")
                        counts["rejected"] += state["status"] == "rejected"
                        rejected.write(canonical({"id": entry["id"], "split": split, "line": line,
                            "status": state["status"], "reasons": state.get("reasons", entry["reasons"]),
                            "original_row": row, "provenance": entry, "attempts": state["attempts"]}) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            if not rows:
                raise ValueError(f"No accepted {split} rows; incomplete dataset is not published")
            identities[f"{split}.jsonl"] = {"rows": rows, "bytes": pending.stat().st_size, "sha256": file_hash(pending)}
        rejected.flush()
        os.fsync(rejected.fileno())
    source.recheck_sidecars()
    pilot = bool(summary["targets_deferred_by_limit"])
    for name in ("train.jsonl", "val.jsonl", "quarantine.jsonl"):
        destination = "pilot_" + name if pilot and name != "quarantine.jsonl" else name
        os.replace(output / (name + ".pending"), output / destination)
    if pilot:
        identities = {"pilot_" + name: value for name, value in identities.items()}
    scaffold = source.directory / "prompt_scaffold.json"
    if scaffold.exists():
        atomic_bytes(output / scaffold.name, scaffold.read_bytes())
    manifest = {"policy": VERSION, "dataset": output.name, "status": "pilot_partial" if pilot else "complete",
                "training_input_ready": not pilot,
                "input_hashes": source.identities, "outputs": identities, "counts": counts,
                "resume_contract_sha256": contract_hash, "quarantine_sha256": file_hash(output / "quarantine.jsonl"),
                "source_quality_scope": "known_holds_and_declared_checks_not_independent_fact_verification",
                "untouched_rows": "byte_preserved", "split_policy": "preserve_original_train_val",
                "benchmarks_used": False, "training_start_generation": False}
    content = canonical(manifest) + b"\n"
    save(output / "publication.json", {"manifest_sha256": hashlib.sha256(content).hexdigest(),
                                      "contract_sha256": contract_hash})
    atomic_bytes(output / "manifest.json", content)
    return manifest


def regenerate(input_dir: Path, output_dir: Path, *, execute=False, resume=False,
               teacher_model=DEFAULT_TEACHER, max_samples=0, max_attempts=3, batch_size=8,
               max_tokens=12288, prompt_max_tokens=16000, context_tokens=32768, seed=123,
               adapter: Any = None, stage3_runner: Any = None) -> dict:
    """Verify/plan by default; only explicit execution constructs a teacher."""
    source = SplitInput(input_dir)
    output = safe_path(output_dir)
    if source.directory == output or source.directory in output.parents or output in source.directory.parents:
        raise ValueError("Input and output must be disjoint; the release is immutable")
    if resume and not execute:
        raise ValueError("--resume requires --execute")
    config = _config(teacher_model=teacher_model, max_samples=max_samples, max_attempts=max_attempts,
                     batch_size=batch_size, max_tokens=max_tokens, prompt_max_tokens=prompt_max_tokens,
                     context_tokens=context_tokens, seed=seed)
    if resume:
        contract = checked(output / "contract.json")
        if contract.get("config") != config:
            raise ValueError("Resume contract mismatch: code/prompt/config changed; use a fresh output")
        for split in ("train", "val"):
            for _ in source.rows(split):
                pass
        source.recheck_sidecars()
        if contract.get("inputs") != source.identities:
            raise ValueError("Resume contract mismatch: source changed; use a fresh output")
        selection, summary = contract["selection"], contract["summary"]
    else:
        selection, summary = _scan(source, config)
        contract = {"config": config, "inputs": source.identities, "selection": selection, "summary": summary}
    contract_hash = digest(contract)
    if not execute:
        return {"mode": "plan", **summary, "contract_sha256": contract_hash,
                "output_dir": str(output), "generation_started": False}
    if resume:
        if not output.is_dir() or checked(output / "contract.json") != contract:
            raise ValueError("Resume contract mismatch: source/selection/code/prompt/config changed; use a fresh output")
        for path in output.rglob("*"):
            safe_path(path)
    else:
        if output.exists():
            raise ValueError("Fresh execution requires a new output directory; use --resume only for exact continuation")
        output.mkdir(parents=True)
        (output / "journal").mkdir()
    with _exclusive(output):
        if not resume:
            save(output / "contract.json", contract)
        return _execute(source, output, selection, summary, config, contract_hash, adapter, stage3_runner)


def _execute(source, output, selection, summary, config, contract_hash, adapter, stage3_runner):
    if (output / "manifest.json").exists():
        publication = checked(output / "publication.json")
        if (publication.get("contract_sha256") != contract_hash
                or publication.get("manifest_sha256") != file_hash(output / "manifest.json")):
            raise ValueError("Published manifest changed")
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("resume_contract_sha256") != contract_hash:
            raise ValueError("Published manifest contract changed")
        for name, identity in manifest["outputs"].items():
            if file_hash(safe_path(output / name)) != identity["sha256"]:
                raise ValueError("Published dataset changed; refusing to replace it")
        if file_hash(output / "quarantine.jsonl") != manifest["quarantine_sha256"]:
            raise ValueError("Published quarantine changed")
        for entry in selection:
            state = _load_state(output, entry, contract_hash)
            if state["status"] not in {"accepted", "rejected", "source_held"} or state.get("inflight"):
                raise ValueError("Published candidate journal is missing or incomplete")
        scaffold = output / "prompt_scaffold.json"
        if "prompt_scaffold.json" in source.identities and file_hash(scaffold) != source.identities["prompt_scaffold.json"]:
            raise ValueError("Published prompt scaffold changed")
        return manifest
    selection_by_location = {(entry["split"], entry["line"]): entry for entry in selection}
    with _teacher_environment(config):
        batch = []
        for split in ("train", "val"):
            for line, raw, row in source.rows(split):
                entry = selection_by_location.get((split, line))
                if entry is None:
                    continue
                if hashlib.sha256(raw).hexdigest() != entry["row_sha256"]:
                    raise ValueError("Selected source changed before generation")
                state = _load_state(output, entry, contract_hash)
                if state["status"] in {"accepted", "source_held"}:
                    continue
                batch.append((entry, row, state))
                if len(batch) == config["batch_size"]:
                    adapter = _process_batch(output, batch, config, contract_hash, adapter, stage3_runner)
                    batch = []
        if batch:
            _process_batch(output, batch, config, contract_hash, adapter, stage3_runner)
    _check_runtime_identity(config)
    return _publish(source, output, selection, contract_hash, summary)


def _process_batch(output, batch, config, contract_hash, adapter, runner):
    for directory in sorted({state["inflight"] for _, _, state in batch if state.get("inflight")}):
        recovering = [item for item in batch if item[2].get("inflight") == directory]
        _round(output, recovering, config, contract_hash, adapter, runner)
    while pending := [item for item in batch if item[2]["status"] != "accepted"
                      and len(item[2]["attempts"]) < config["max_attempts"]]:
        if adapter is None:
            adapter = _build_teacher(config)
        fatal = _round(output, pending, config, contract_hash, adapter, runner)
        if fatal and not fatal.startswith("Recovered interrupted attempt"):
            raise RuntimeError(f"Stage 3 failed; durable evidence retained, resume with identical flags: {fatal}")
    return adapter


def _build_teacher(config):
    from llm.base import ModelSpec
    from llm.factory import build_adapter
    values = config["teacher_spec"]
    return build_adapter(ModelSpec(**{key: values[key] for key in
        ("name", "provider", "model", "supports_json_mode", "endpoint", "limits") if key in values}))
