"""CPU-only, semantics-checked preparation of existing Express supervision.

This module consumes completed Stage 3 targets; it never generates or repairs
model predictions. Both ordering experiments use exactly the same validation
and canonical emitter. IDs are deliberately retained, including shared edges.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping

from ir_training.common.config import repo_root


PREPARATION_VERSION = "2.0.0"
TARGET_FORMAT = "a2ui_express_v1"
TASK_PREFIX = "Create A2UI Express v1 GenUI IR for this response:\n\n"


class PreparationError(ValueError):
    def __init__(self, reason: str, detail: str):
        self.reason = reason
        super().__init__(detail)


@dataclass(frozen=True)
class SerializedTarget:
    text: str
    semantic_sha256: str
    component_types: tuple[str, ...]
    reference_kinds: tuple[str, ...]
    graph: dict[str, Any]


def _api():
    dataset_src = repo_root() / "dataset" / "src"
    if str(dataset_src) not in sys.path:
        sys.path.insert(0, str(dataset_src))
    from pipeline.ir_formats import active, express
    from pipeline.ir_formats.canonical import semantic_hash
    from pipeline.renderer_semantics import iter_renderer_references
    return active, express, semantic_hash, iter_renderer_references


@lru_cache(maxsize=1)
def _wire_validator():
    # Cache schema compilation for a large corpus. Missing dependencies or
    # schemas are fatal setup errors, not a reason to drop every source row.
    from jsonschema import Draft202012Validator
    schema = json.loads((repo_root() / "dataset/schema/genuicraft_a2ui_v1_wire.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _validate_wire(graph: dict[str, Any]) -> None:
    from pipeline.ir_formats import a2ui_wire
    from jsonschema import ValidationError
    try:
        payload = a2ui_wire.encode(graph, shorten_ids=False)
        _wire_validator().validate(payload)
    except ValidationError as exc:
        # This catches property types (e.g. Table.highlightColumns) that the
        # Express syntax/canonical reference validator alone does not check.
        from jsonschema.exceptions import best_match
        leaf = best_match([exc]) or exc
        location = "/".join(str(part) for part in leaf.absolute_path)
        # Re-check the known component type instead of reporting all unrelated
        # alternatives of anyComponent's oneOf (often an entire UI document).
        for message in payload:
            for component in message.get("updateComponents", {}).get("components", []):
                name = component.get("component")
                if name not in _wire_validator().schema["$defs"]["components"]:
                    continue
                validator = _wire_validator().evolve(schema={"$ref": f"#/$defs/components/{name}"})
                specific = best_match(validator.iter_errors(component))
                if specific is not None:
                    leaf = specific
                    location = f"component[{component['id']}]." + "/".join(str(part) for part in leaf.absolute_path)
                    break
            else:
                continue
            break
        # Nested union failures can include the entire document in message;
        # the original row already lives in quarantine, so keep detail small.
        detail = leaf.message
        if len(detail) > 400:
            detail = f"failed {leaf.validator} constraint at schema /" + "/".join(str(part) for part in leaf.absolute_schema_path)
        raise PreparationError("wire_schema_invalid", f"{location}: {detail}") from exc
    except ValueError as exc:
        raise PreparationError("wire_lowering_invalid", str(exc)) from exc


@lru_cache(maxsize=256)
def serialize_checked(text: str, ordering: str = "bottom-up") -> SerializedTarget:
    """Validate, order catalog-encoded assignments, then prove graph identity.

    `root-first` is the control for `bottom-up`: same accepted rows, emitter,
    IDs and semantics, with only assignment ordering different. Reachability
    must be 100%; detached definitions are quarantined rather than discarded.
    """
    if ordering not in {"bottom-up", "root-first"}:
        raise ValueError("ordering must be bottom-up or root-first")
    active, express, semantic_hash, references = _api()
    _wire_validator()
    try:
        source = active.decode_express_completion(text)
    except ValueError as exc:
        raise PreparationError("express_invalid", str(exc)) from exc
    elements = source["elements"]
    seen: set[str] = set()
    postorder: list[str] = []

    def visit(element_id: str) -> None:
        if element_id in seen:
            return
        seen.add(element_id)
        # Canonical validation has already rejected missing references and
        # reachable cycles using this same renderer inventory.
        for edge in references(elements[element_id]):
            visit(edge.target_id)
        postorder.append(element_id)

    visit(source["root"])
    detached = sorted(set(elements) - seen)
    if detached:
        raise PreparationError("unreachable_components", f"{len(detached)}/{len(elements)} detached definitions: {', '.join(detached)}")
    _validate_wire(source)
    before_hash = semantic_hash(source)
    # Use the catalog codec, including inline-component materialization and
    # every reference alias, never a regex that can match inside quoted text.
    try:
        encoded = express.encode(source, shorten_ids=False)
    except ValueError as exc:
        raise PreparationError("serializer_invalid", str(exc)) from exc
    lines = encoded.splitlines()
    state_lines = [line for line in lines[1:-1] if line.startswith("$")]
    assignments = {line.split("=", 1)[0]: line for line in lines[1:-1] if not line.startswith("$")}
    if set(assignments) != set(elements):
        raise PreparationError("serializer_definition_mismatch", "Canonical emitter changed component IDs")
    order = postorder if ordering == "bottom-up" else list(assignments)
    output = "\n".join([express.SENTINEL_OPEN, *state_lines, *(assignments[key] for key in order), express.SENTINEL_CLOSE])
    try:
        decoded = active.decode_express_completion(output)
    except ValueError as exc:
        raise PreparationError("serializer_invalid", str(exc)) from exc
    # Dict equality ignores assignment order but preserves IDs, every literal,
    # state/action payload, child order and edge attachment. Hash alone is a
    # weaker guarantee because it intentionally normalizes component IDs.
    if decoded != source or semantic_hash(decoded) != before_hash:
        raise PreparationError("serializer_semantic_mismatch", "Canonical emission changed target semantics")
    return SerializedTarget(
        output, before_hash,
        tuple(str(element["type"]) for element in elements.values()),
        tuple(edge.reference_kind for element in elements.values() for edge in references(element)),
        decoded,
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_prompt(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"{message['role'].title()}:\n{message['content']}" for message in messages) + "\n\nAssistant:\n"


def prepare_row(row: Mapping[str, Any], ordering: str) -> tuple[dict[str, Any], SerializedTarget, dict[str, Any]]:
    """Keep target aliases, few-shot messages and the text prompt in sync."""
    completion = row.get("completion")
    if not isinstance(completion, str):
        raise PreparationError("missing_completion", "Row requires a text completion")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2 or any(
        not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}
        or not isinstance(message.get("content"), str) for message in messages
    ):
        raise PreparationError("invalid_messages", "Row requires structured text chat messages")
    if messages[-1]["role"] != "assistant" or messages[-2]["role"] != "user":
        raise PreparationError("invalid_task_turns", "Final turns must be the task user and target assistant")
    if messages[-1]["content"].strip() != completion.strip():
        raise PreparationError("target_message_mismatch", "completion differs from final assistant target")
    task = messages[-2]["content"]
    if not task.startswith(TASK_PREFIX):
        raise PreparationError("task_prefix_mismatch", "Cannot establish immutable final task/source binding")
    response_text = task[len(TASK_PREFIX):]
    if not response_text.strip():
        raise PreparationError("missing_response_text", "Final task source is empty")
    if "response_text" in row and str(row["response_text"]).strip() != response_text.strip():
        raise PreparationError("response_text_mismatch", "Final user input differs from response_text")
    if row.get("target_format", TARGET_FORMAT) != TARGET_FORMAT:
        raise PreparationError("target_format_mismatch", "Only native Express targets are accepted")
    target = serialize_checked(completion, ordering)
    result = deepcopy(dict(row))
    for message in result["messages"]:
        if message["role"] == "assistant":
            message["content"] = serialize_checked(message["content"], ordering).text
    for key in ("a2ui_express",):
        if key in row and (not isinstance(row[key], str) or row[key].strip() != completion.strip()):
            raise PreparationError("target_alias_mismatch", f"{key} differs from completion")
        result[key] = target.text
    targets = row.get("completion_targets", {TARGET_FORMAT: completion})
    if not isinstance(targets, dict) or set(targets) != {TARGET_FORMAT} or not isinstance(targets[TARGET_FORMAT], str) or targets[TARGET_FORMAT].strip() != completion.strip():
        raise PreparationError("target_alias_mismatch", "completion_targets differs from the active completion")
    result["completion"] = target.text
    result["completion_targets"] = {TARGET_FORMAT: target.text}
    result["canonical_graph"] = deepcopy(target.graph)
    result["semantic_hash"] = target.semantic_sha256
    result["response_text"] = response_text
    result["target_format"] = TARGET_FORMAT
    result["prompt"] = _render_prompt(result["messages"][:-1])
    # The fixed task prefix is part of deployment parity; the source response
    # and target vary by row and must not be part of the shared scaffold hash.
    scaffold = {"messages": result["messages"][:-2], "task_prefix": TASK_PREFIX, "serialization_order": ordering}
    scaffold_hash = _sha(_json(scaffold))
    metadata = result.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise PreparationError("invalid_metadata", "metadata must be an object")
    metadata["express_preparation"] = {
        "version": PREPARATION_VERSION, "serialization_order": ordering,
        "source_completion_sha256": _sha(completion), "completion_sha256": _sha(target.text),
        "source_prompt_sha256": _sha(str(row.get("prompt", ""))),
        "prompt_sha256": _sha(result["prompt"]), "scaffold_sha256": scaffold_hash,
        "semantic_sha256": target.semantic_sha256, "root_reachability": 1.0,
        "id_repair_applied": False,
    }
    return result, target, {"sha256": scaffold_hash, **scaffold}


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _token_lengths(row: dict[str, Any], tokenizer: Any, template_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        full = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False, **template_kwargs)
        prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True, **template_kwargs)
    except Exception as exc:
        raise PreparationError("chat_template_invalid", str(exc)) from exc
    if not isinstance(full, str) or not isinstance(prompt, str) or not prompt or not full.startswith(prompt):
        raise PreparationError("chat_template_prefix_mismatch", "Generation prompt is not the exact prefix of the supervised chat")
    completion = full[len(prompt):]
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    completion_ids = list(tokenizer(completion, add_special_tokens=False)["input_ids"])
    full_ids = list(tokenizer(full, add_special_tokens=False)["input_ids"])
    if not prompt_ids or not completion_ids:
        raise PreparationError("empty_tokenized_turn", "Prompt and completion must both contain tokens")
    return {
        # SFT concatenates separately tokenized prompt and assistant suffix.
        "sequence_tokens": len(prompt_ids) + len(completion_ids),
        "full_chat_tokens": len(full_ids), "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(completion_ids), "prompt_token_ids_sha256": _sha(_json(prompt_ids)),
        "supervised_token_ids_sha256": _sha(_json(prompt_ids + completion_ids)),
    }


def prepare_splits(
    inputs: Mapping[str, Path], output_dir: Path, *, ordering: str = "root-first",
    tokenizer: Any | None = None, max_seq_length: int | None = None,
    max_input_tokens: int | None = None, chat_template_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish a complete new directory atomically; never overwrite a dataset.

    Invalid row objects are quarantined with reasons and original content.
    Malformed JSON, I/O failures, missing dependencies and empty accepted
    splits abort publication. A caller can safely retry in a new destination.
    """
    if ordering not in {"bottom-up", "root-first"}:
        raise ValueError("ordering must be bottom-up or root-first")
    for name, limit in (("max_seq_length", max_seq_length), ("max_input_tokens", max_input_tokens)):
        if limit is not None and (not isinstance(limit, int) or limit <= 0 or tokenizer is None):
            raise ValueError(f"{name} requires a positive integer and an explicit tokenizer")
    template_kwargs = dict(chat_template_kwargs or {})
    if {"tokenize", "add_generation_prompt"} & set(template_kwargs):
        raise ValueError("chat_template_kwargs cannot override tokenize/add_generation_prompt")
    if not inputs or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) or name.casefold() == "quarantine" for name in inputs):
        raise ValueError("Supply at least one input; split names must be simple identifiers")
    sources = {name: Path(path).resolve(strict=True) for name, path in inputs.items()}
    destination = output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    _api()
    _wire_validator()
    source_hashes = {name: _sha_file(path) for name, path in sources.items()}
    # Carry explicitly approved benchmark membership through token preparation.
    # Its raw hash is checked before any transformation; the resulting split has
    # a new preparation hash. Never silently relax ordinary Golden uniqueness.
    benchmarks = {}
    from ir_training.eval.golden_set import benchmark_contract_for_split
    for name, path in sources.items():
        if (path.parent / "benchmark_manifest.json").is_file() or (path.parent / "manifest.json").is_file():
            contract = benchmark_contract_for_split(path)
            if contract is not None:
                benchmarks[name] = contract
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.preparing-", dir=destination.parent))
    try:
        stats: dict[str, Any] = {}
        scaffolds: dict[str, Any] = {}
        with (temporary / "quarantine.jsonl").open("w", encoding="utf-8", newline="\n") as quarantine:
            for split, source in sources.items():
                counts: Counter[str] = Counter()
                reasons: Counter[str] = Counter()
                types: Counter[str] = Counter()
                kinds: Counter[str] = Counter()
                scaffold_counts: Counter[str] = Counter()
                token_maxima: dict[str, int] = {}
                accepted_ids = hashlib.sha256()
                source_ids = hashlib.sha256()
                with source.open(encoding="utf-8-sig") as stream, (temporary / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as output:
                    for line_number, line in enumerate(stream, start=1):
                        if not line.strip():
                            continue
                        counts["input_rows"] += 1
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"{source}:{line_number}: malformed JSON; no dataset published") from exc
                        source_ids.update(f"{line_number}:{_sha(_json(row))}\n".encode("utf-8"))
                        try:
                            if not isinstance(row, dict):
                                raise PreparationError("row_not_object", "Each JSONL row must be an object")
                            prepared, target, scaffold = prepare_row(row, ordering)
                            if tokenizer is not None:
                                lengths = _token_lengths(prepared, tokenizer, template_kwargs)
                                if max_seq_length is not None and lengths["sequence_tokens"] > max_seq_length:
                                    raise PreparationError("sequence_too_long", f"{lengths['sequence_tokens']} tokens > {max_seq_length}; complete row quarantined")
                                if max_input_tokens is not None and lengths["prompt_tokens"] > max_input_tokens:
                                    raise PreparationError("prompt_too_long", f"{lengths['prompt_tokens']} tokens > {max_input_tokens}; complete row quarantined")
                                prepared["metadata"]["express_preparation"]["tokenization"] = lengths
                                for key, value in lengths.items():
                                    if key.endswith("_tokens"):
                                        token_maxima[key] = max(token_maxima.get(key, 0), value)
                        except PreparationError as exc:
                            counts["quarantined_rows"] += 1
                            reasons[exc.reason] += 1
                            quarantine.write(_json({"split": split, "source_line": line_number, "reason": exc.reason, "detail": str(exc), "row": row}) + "\n")
                            continue
                        output.write(_json(prepared) + "\n")
                        counts["accepted_rows"] += 1
                        accepted_ids.update(f"{line_number}:{_sha(_json(row))}\n".encode("utf-8"))
                        types.update(target.component_types)
                        kinds.update(target.reference_kinds)
                        scaffolds[scaffold["sha256"]] = scaffold
                        scaffold_counts[scaffold["sha256"]] += 1
                if not counts["accepted_rows"]:
                    raise ValueError(f"Split {split} has no accepted rows; no dataset published")
                if _sha_file(source) != source_hashes[split]:
                    raise ValueError(f"Source changed while preparing {split}; no dataset published")
                stats[split] = {
                    "input_rows": counts["input_rows"], "accepted_rows": counts["accepted_rows"],
                    "quarantined_rows": counts["quarantined_rows"], "quarantine_reasons": dict(reasons),
                    "source_path": str(source), "source_sha256": source_hashes[split],
                    "output_sha256": _sha_file(temporary / f"{split}.jsonl"),
                    "source_rows_sha256": source_ids.hexdigest(), "accepted_source_rows_sha256": accepted_ids.hexdigest(),
                    "component_counts": dict(types), "renderer_reference_counts": dict(kinds),
                    "scaffold_counts": dict(scaffold_counts),
                    "max_accepted_token_lengths": token_maxima,
                }
                if split in benchmarks:
                    if counts["quarantined_rows"]:
                        raise ValueError("Golden preparation must preserve all declared occurrences; no partial benchmark published")
                    from ir_training.eval.golden_set import validate_benchmark_rows
                    from ir_training.common.jsonl import read_jsonl
                    validate_benchmark_rows(list(read_jsonl(temporary / f"{split}.jsonl")), benchmarks[split])
                    stats[split]["benchmark"] = benchmarks[split]
        from pipeline.ir_formats.common import codec_identity
        tracked_sources = [Path(__file__), repo_root() / "dataset/src/pipeline/ir_formats/express.py", repo_root() / "dataset/src/pipeline/ir_formats/canonical.py", repo_root() / "dataset/src/pipeline/renderer_semantics.py", repo_root() / "dataset/schema/genuicraft_a2ui_v1_wire.schema.json", repo_root() / "dataset/schema/genuicraft_a2ui_catalog_v1.json"]
        manifest = {
            "preparation_version": PREPARATION_VERSION, "ordering": ordering,
            "validation": {"strict_express": True, "wire_schema": True, "root_reachability": 1.0, "semantic_roundtrip": True, "id_repair": False},
            "codec_identity": codec_identity(),
            "implementation_sha256": {str(path.relative_to(repo_root())).replace("\\", "/"): _sha_file(path) for path in tracked_sources},
            "splits": stats, "scaffold_count": len(scaffolds),
            "prompt_parity": "Use prompt_scaffolds.json and prepared messages for HF and deployment. Tokenizer chat-template/token-ID parity still requires the selected tokenizer on the GPU/export PC.",
            "ab_comparison": "accepted_source_rows_sha256 must match between root-first and bottom-up runs for every split.",
        }
        if tokenizer is not None:
            vocabulary = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None
            manifest["tokenizer"] = {
                "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
                "class": type(tokenizer).__name__, "vocabulary_sha256": _sha(_json(vocabulary)) if vocabulary is not None else None,
                "chat_template_sha256": _sha(_json(getattr(tokenizer, "chat_template", None))),
                "chat_template_kwargs": template_kwargs,
                "bos_token_id": getattr(tokenizer, "bos_token_id", None),
                "eos_token_id": getattr(tokenizer, "eos_token_id", None),
                "pad_token_id": getattr(tokenizer, "pad_token_id", None),
                "add_special_tokens": False, "max_seq_length": max_seq_length, "max_input_tokens": max_input_tokens,
            }
        _write_json(temporary / "prompt_scaffolds.json", list(scaffolds.values()))
        manifest["prompt_scaffolds_sha256"] = _sha_file(temporary / "prompt_scaffolds.json")
        manifest["quarantine_sha256"] = _sha_file(temporary / "quarantine.jsonl")
        _write_json(temporary / "manifest.json", manifest)
        # Same-volume rename exposes all files together. Refuse an existing
        # destination even if another process creates it during preparation.
        if destination.exists():
            raise FileExistsError(f"Output directory appeared while preparing: {destination}")
        temporary.rename(destination)
        return manifest
    finally:
        # Only the fresh temporary sibling created above can be removed.
        if temporary.exists():
            shutil.rmtree(temporary)
