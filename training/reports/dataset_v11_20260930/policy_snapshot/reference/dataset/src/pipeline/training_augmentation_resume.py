"""Durable, fail-closed execution for dataset-owned augmentation.

The immutable contract describes logical slots; per-slot atomic checkpoints
reference immutable call artifacts. Root JSONL files are replaceable projections,
not caches used to decide which examples are valid. No teacher is constructed by
the read-only resume validator.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pipeline import training_augmentation as api

RESUME_SCHEMA = 1
FIDELITY_REGENERATION_POLICY = {
    "version": "augmentation-stage3-fidelity-v1",
    "max_regenerations": 2,
}
# Exact LF/CRLF identities from 84cfcf16. This is the only pre-policy durable
# contract allowed to continue under the bounded fidelity regeneration code.
PRE_FIDELITY_CODE_HASHES = {
    "src/pipeline/training_augmentation.py": {
        "dc69dc52e630985cadce8d53cf12d59283ab7dc89df558367a55d6d7d1e4416c",
        "55fc3c833c87c23e9b68e3953ff7b8e946bc30c29b8265329e11e67602362e19",
    },
    "src/pipeline/training_augmentation_resume.py": {
        "c6f51711c66216aa1ce96b698956ef8c76d38ad3a8cfffeaab909e4acaf82da8",
        "09ac09d717b12ae34f656c1011a1baf95a9abea15b4381dc6f0944d03a5824a6",
    },
    "src/pipeline/stage3_genui.py": {
        "213a09f7addf375f15b71f3ce7e507140c265c34ac0cbbcf2bc9dc16b72f90e5",
        "f93b44c9f8a9d6a5cb31863e64c86fa67868692f232b3dd29343f1083aefa090",
    },
    "src/pipeline/muse_prompt.py": {
        "d3590c3515dd1ae66e99eeffce81a402fdb1e0cad3d2640539f710e79149f6f9",
        "c89d615d39e1ebb2e03eb6c8a5bd0ece8cb8cce76392aae2c4f608a6aac6966e",
    },
}
PRE_FIDELITY_MUSE_PROMPT_HASHES = {
    "f829fc9a7d020755fda04612a1fc8472d4bb5d910b8f4e7f9f011ac911da1eaa",
    "bef0807bfbc7900ad9efaf952b9543cf9900ed16db3d3cb540ba9355df165bde",
}
FIDELITY_MUSE_PROMPT_HASHES = {
    "352b008cced299e01e5cbfbdde288c715dbb3aa4bebd5149a06bd454b4fd5980",
    "c8cb846b0e891c3ed17acd797061aaf2443805c53a1ea3f5492cc062347b7ae4",
}
LEGACY_CODE_HASHES = {
    "src/pipeline/training_augmentation.py": {"553f12c58348cdd60b0076ce6a6a473a7405f2de770e0099935db92eebc49e6c", "8ce8f27201495703b09b783e4ad6bca9d09fea6b1c87b74e7944d838d3304c90"},
    "scripts/generate_training_augmentations.py": {"a35a6154d00af1b50a02a519c4e4b89851649858c90f42026a5ad963c90f6846", "06d95ca166a9633b66d1a06f2ec9be0dccae7c65985fc9c17977220afa46c69e"},
    "src/pipeline/stage3_genui.py": {"005f8683ec30f1c8ee42f45303fb17680a5af3054bdc844a6262f2686833faaa", "5954d70aaae8bd8736c747b4a742ff387b72f96574ed369efacac160f4ad06d9"},
    "src/pipeline/genui_quality/acceptance_v5_4.py": {"aa612fc011b53c96d0c0e98fb7c540c0892181e0246410dc4805b3371e545ff9", "a34821a8eff06d072068182837e8a7de6e992442c6070a876509d8594e9a4390"},
}
PROJECTIONS = ("queries.jsonl", "responses.jsonl", "genui.jsonl", "accepted_genui.jsonl")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return api.text_hash(_canonical(value))


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")  # noqa: TRY004 - persisted contract validation
    return value


def _safe_path(path: Path) -> Path:
    path = Path(path).absolute()
    if any(_is_link(parent) for parent in (path, *path.parents)):
        raise ValueError(f"Symlink is not permitted: {path}")
    return path.resolve()


def _is_link(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _atomic_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic(path: Path, value: Any) -> None:
    _atomic_bytes(path, (_canonical(value) + "\n").encode("utf-8"))


def _rows(path: Path, values: list[dict[str, Any]]) -> None:
    _atomic_bytes(path, "".join(_canonical(value) + "\n" for value in values).encode("utf-8"))


def _envelope(value: dict[str, Any]) -> dict[str, Any]:
    return {"payload": value, "sha256": _hash(value)}


def _checked(path: Path) -> dict[str, Any]:
    value = _read(path)
    if not isinstance(value.get("payload"), dict) or value.get("sha256") != _hash(value["payload"]):
        raise ValueError(f"Checkpoint hash mismatch: {path}")
    return value["payload"]


def _configuration(donors_path, *, teacher_model, max_new_samples, seed, reference_bindings_path):
    if type(max_new_samples) is not int or not 1 <= max_new_samples <= 100000:
        raise ValueError("max_new_samples must be an integer in [1, 100000]")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    from llm.factory import load_model_specs
    from utils.config import load_yaml

    from pipeline.augmentation_references import VERSION as reference_version

    donors_path = _safe_path(donors_path)
    donors = api.load_donors(donors_path)
    specs = [spec for spec in load_model_specs(load_yaml(api.DATASET_ROOT / "configs/models.yaml")) if spec.name == teacher_model]
    if len(specs) != 1:
        raise ValueError(f"Expected exactly one registered teacher: {teacher_model}")
    spec = specs[0]
    if spec.name != api.DEFAULT_TEACHER or spec.provider != "local" or not str(spec.endpoint).startswith(("http://", "https://")):
        raise ValueError("This version requires the registered Muse HTTP teacher; no fallback is allowed")
    donor_hash = _digest(donors_path)
    bindings = {}
    binding_hash = None
    if reference_bindings_path is not None:
        path = _safe_path(reference_bindings_path)
        document = _read(path)
        if document.get("version") != 1 or document.get("donors_sha256") != donor_hash or not isinstance(document.get("bindings"), dict):
            raise ValueError("Reference bindings version or donors hash mismatch")
        donors_by_id = {donor["donor_id"]: donor for donor in donors}
        if set(document["bindings"]) != set(donors_by_id):
            raise ValueError("Reference bindings must cover exactly the supplied donors")
        for donor_id, item in document["bindings"].items():
            if donor_id not in donors_by_id or not isinstance(item, dict) or item.get("source_sha256") != api.text_hash(donors_by_id[donor_id]["response_text"]):
                raise ValueError("Reference binding donor identity/source hash mismatch")
            mapping = item.get("reference_map")
            if not isinstance(mapping, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items()):
                raise ValueError("Malformed reference binding map")
            bindings[donor_id] = mapping
        binding_hash = _digest(path)
    random.Random(seed).shuffle(donors)
    schedule = []
    for index, donor in enumerate(donors[:max_new_samples]):
        category = api.CATEGORIES[index % len(api.CATEGORIES)]
        schedule.append({"candidate_id": "aug_" + api.text_hash(f"{donor_hash}:{seed}:{index}:{category}")[:24],
                         "index": index, "category": category, "donor_id": donor["donor_id"],
                         "source_group_id": donor["source_group_id"], "donor_source_sha256": api.text_hash(donor["response_text"])})
    root = api.DATASET_ROOT
    code_paths = {root / "scripts/generate_training_augmentations.py"}
    # Conservative transitive identity: imports, validators, schemas and provider
    # configuration cannot silently drift while a partially completed run resumes.
    for folder in ("src/pipeline", "src/llm", "src/utils"):
        code_paths.update((root / folder).rglob("*.py"))
    code_paths.update((root / "schema").rglob("*.json"))
    code_paths.add(root / "configs/models.yaml")
    prompt = (root / "prompts/training_augmentation_v1.md").read_text(encoding="utf-8")
    with api.bounded_teacher_environment() as environment:
        effective_environment = dict(environment)
    # Unset optional switches are also bound, without recording any credentials.
    optional_environment = {key: value for key, value in os.environ.items()
                            if key.startswith(("STAGE3_", "GENUI_", "LOCAL_VLLM_", "LOCAL_MUSE_"))
                            and key not in effective_environment and not any(secret in key for secret in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    config = {"version": api.VERSION, "resume_schema": RESUME_SCHEMA, "reference_policy_version": reference_version,
              "donors_sha256": donor_hash, "reference_bindings_sha256": binding_hash,
              "teacher_model": teacher_model, "teacher_identity": asdict(spec), "seed": seed,
              "max_new_samples": max_new_samples, "categories": list(api.CATEGORIES), "schedule": schedule,
              "teacher_prompt_sha256": api.text_hash(prompt), "recipes_sha256": _hash(api.RECIPES),
              "code_hashes": {path.relative_to(root).as_posix(): _digest(path) for path in sorted(code_paths)},
              "stage3_prompt_sha256": _digest(root / "prompts/genui_gen_mobile_a2ui_express_v1.md"),
              "muse_stage3_prompt_sha256": _digest(root / "prompts/muse_stage3_quality_v2.md"),
              "contract_sha256": _digest(root / "schema/canonical_ui_graph_v1.schema.json"),
              "teacher_environment": effective_environment, "optional_environment_sha256": _hash(optional_environment),
              "teacher_endpoints": os.environ.get("LOCAL_VLLM_ENDPOINTS") or spec.endpoint,
              "fidelity_regeneration_policy": FIDELITY_REGENERATION_POLICY}
    return config, donors, bindings, spec, prompt


def _known_fidelity_resume_upgrade(saved: dict[str, Any], current: dict[str, Any]) -> bool:
    """Recognize only the exact 84cfcf16 contract plus this policy upgrade."""
    if "fidelity_regeneration_policy" in saved or current.get("fidelity_regeneration_policy") != FIDELITY_REGENERATION_POLICY:
        return False
    saved_fields = {key: value for key, value in saved.items()
                    if key not in ("code_hashes", "muse_stage3_prompt_sha256")}
    current_fields = {key: value for key, value in current.items()
                      if key not in ("code_hashes", "muse_stage3_prompt_sha256", "fidelity_regeneration_policy")}
    if (
        saved_fields != current_fields
        or saved.get("muse_stage3_prompt_sha256") not in PRE_FIDELITY_MUSE_PROMPT_HASHES
        or current.get("muse_stage3_prompt_sha256") not in FIDELITY_MUSE_PROMPT_HASHES
    ):
        return False
    saved_hashes = saved.get("code_hashes")
    current_hashes = current.get("code_hashes")
    if not isinstance(saved_hashes, dict) or not isinstance(current_hashes, dict) or set(saved_hashes) != set(current_hashes):
        return False
    changed = {path for path in saved_hashes if saved_hashes[path] != current_hashes[path]}
    if changed != set(PRE_FIDELITY_CODE_HASHES):
        return False
    return all(saved_hashes[path] in allowed for path, allowed in PRE_FIDELITY_CODE_HASHES.items())


def _legacy_validate(output: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = output / "manifest.json"
    if not path.is_file():
        raise ValueError("Cannot resume legacy generation without manifest.json: code identity cannot be proven; use a fresh output directory")
    manifest = _read(path)
    if manifest.get("version") != api.VERSION or manifest.get("status") not in ("failed", "completed") or "resume_schema" in manifest:
        raise ValueError("Only the recognized finalized legacy v1 generation can be upgraded")
    hashes = {key.replace("\\", "/"): value for key, value in manifest.get("code_hashes", {}).items()}
    if set(hashes) != set(LEGACY_CODE_HASHES) or any(value not in LEGACY_CODE_HASHES[key] for key, value in hashes.items()):
        raise ValueError("Legacy generation code hashes are not the recognized baseline")
    for field in ("version", "donors_sha256", "teacher_model", "teacher_identity", "seed", "max_new_samples", "categories", "teacher_prompt_sha256",
                  "teacher_environment", "teacher_endpoints"):
        if manifest.get(field) != config[field]:
            raise ValueError(f"Legacy generation contract mismatch: {field}")
    if manifest.get("teacher_prompt_sha256") != "ebf8131eb6bcfe21d0c8a5eaa4ca8f2d4119509bc452a2e28d242985a796b957":
        raise ValueError("Legacy source prompt is not the recognized baseline")
    # Exact known LF/CRLF byte identities only; do not normalize arbitrary code.
    for field, (source_hashes, target_hashes) in {
        "stage3_prompt_sha256": (
            {"f354f2e6e65cd18fa72984f5e3eff7c60b99ccec660c2710c970a7d8eab60b13", "75e28a4ef75496f61420c7a85c56fee8b3fd0d49660850b77e0757ae12eb7b09"},
            {"f354f2e6e65cd18fa72984f5e3eff7c60b99ccec660c2710c970a7d8eab60b13", "75e28a4ef75496f61420c7a85c56fee8b3fd0d49660850b77e0757ae12eb7b09"},
        ),
        "muse_stage3_prompt_sha256": (PRE_FIDELITY_MUSE_PROMPT_HASHES, FIDELITY_MUSE_PROMPT_HASHES),
        "contract_sha256": (
            {"0c5c211e0ddf05e0e6f22b803dee6050a2141d4723e341a75ff3b4dd69384e83", "c2eebcf8ab825015fde06155ddde9690bddf5350576efdea2330be96257b0ef5"},
            {"0c5c211e0ddf05e0e6f22b803dee6050a2141d4723e341a75ff3b4dd69384e83", "c2eebcf8ab825015fde06155ddde9690bddf5350576efdea2330be96257b0ef5"},
        ),
    }.items():
        if manifest.get(field) not in source_hashes or config[field] not in target_hashes:
            raise ValueError(f"Legacy generation contract mismatch: {field}")
    accepted = output / "accepted_genui.jsonl"
    if not accepted.is_file() or _digest(accepted) != manifest.get("accepted_genui_sha256"):
        raise ValueError("Legacy accepted projection hash mismatch")
    if manifest.get("accepted_rows") != len(api.read_jsonl(accepted)):
        raise ValueError("Legacy accepted row count mismatch")
    for name in (*PROJECTIONS, "augmentation_audit.jsonl"):
        if not (output / name).is_file():
            raise ValueError(f"Legacy generation missing required evidence: {name}")
    return {"manifest_sha256": _digest(path), "files": {name: _digest(output / name) for name in (*PROJECTIONS, "augmentation_audit.jsonl", "manifest.json")},
            "migration": "recognized-v1-reference-binding-only"}


def validate_generation_resume(donors_path, output_dir, *, teacher_model=api.DEFAULT_TEACHER,
                               max_new_samples=90, seed=123, reference_bindings_path=None):
    """Read-only preflight; never constructs a teacher or changes run artifacts."""
    config, _, _, _, _ = _configuration(donors_path, teacher_model=teacher_model, max_new_samples=max_new_samples,
                                         seed=seed, reference_bindings_path=reference_bindings_path)
    output = _safe_path(output_dir)
    if not output.is_dir():
        raise ValueError("Resume requires an existing generation directory")
    if any(_is_link(path) for path in output.rglob("*")):
        raise ValueError("Resume artifacts must not contain symlinks")
    contract_path = output / "resume_contract.json"
    if contract_path.exists():
        contract = _checked(contract_path)
        compatible_fidelity_upgrade = False
        if contract.get("configuration") != config:
            compatible_fidelity_upgrade = _known_fidelity_resume_upgrade(contract.get("configuration", {}), config)
        if contract.get("configuration") != config and not compatible_fidelity_upgrade:
            fields = sorted(key for key in config if contract.get("configuration", {}).get(key) != config[key])
            raise ValueError("Generation resume contract mismatch: " + ", ".join(fields))
        for name, digest in contract.get("legacy_origin", {}).get("files", {}).items():
            if _digest(output / "legacy_v1" / name) != digest:
                raise ValueError("Preserved legacy evidence hash mismatch")
        for path in (output / "candidates").glob("*.json"):
            state = _checked(path)
            if path.stem not in {slot["candidate_id"] for slot in config["schedule"]} or path.stem != state.get("candidate_id"):
                raise ValueError("Unknown candidate journal identity")
            if state.get("contract_sha256") != _hash(contract):
                raise ValueError("Candidate journal contract mismatch")
            for artifact in state.get("artifacts", []):
                target = output / artifact["path"]
                if target.resolve().is_relative_to(output) is False or _digest(target) != artifact["sha256"]:
                    raise ValueError("Candidate attempt artifact hash mismatch")
        for receipt_path in (output / "stage3_attempts").glob("*/receipt.json"):
            receipt = _checked(receipt_path)
            for name, digest in receipt["files"].items():
                target = receipt_path.parent / name
                if not target.resolve().is_relative_to(receipt_path.parent) or _digest(target) != digest:
                    raise ValueError("Stage 3 attempt artifact hash mismatch")
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            manifest = _read(manifest_path)
            if manifest.get("resume_contract_sha256") != _hash(contract):
                # A kill between publishing the migration contract and replacing
                # the old manifest is recoverable from the sealed legacy snapshot.
                if _digest(manifest_path) != contract.get("legacy_origin", {}).get("manifest_sha256"):
                    raise ValueError("Generation manifest contract mismatch")
            elif manifest.get("status") in ("completed", "failed"):
                for name, digest in manifest.get("projection_hashes", {}).items():
                    if _digest(output / name) != digest:
                        raise ValueError("Generation projection hash mismatch")
                for name, digest in manifest.get("journal_hashes", {}).items():
                    if _digest(output / "candidates" / name) != digest:
                        raise ValueError("Candidate journal hash mismatch")
        return {"legacy_upgrade": False, "compatible_fidelity_upgrade": compatible_fidelity_upgrade,
                "resume_contract_sha256": _hash(contract)}
    origin = _legacy_validate(output, config)
    return {"legacy_upgrade": True, "compatible_fidelity_upgrade": False,
            "resume_contract_sha256": _hash({"configuration": config, "legacy_origin": origin}), "legacy_origin": origin}


@contextmanager
def _lock(output: Path):
    """OS-held lock releases on process death; the inert file is safe to retain."""
    with (output / ".generation.lock").open("a+b") as stream:
        stream.seek(0)
        if stream.read(1) == b"":
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
            raise ValueError("Generation directory is already in use") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _audit(output: Path, value: dict[str, Any]) -> None:
    with (output / "augmentation_audit.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(_canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _save(output: Path, state: dict[str, Any]) -> None:
    _atomic(output / "candidates" / (state["candidate_id"] + ".json"), _envelope(state))


def _provenance(config, slot):
    return {"version": api.VERSION, "category": slot["category"], "donor_id": slot["donor_id"],
            "source_group_id": slot["source_group_id"], "seed": config["seed"], "candidate_seed": config["seed"] + slot["index"],
            "donor_source_sha256": slot["donor_source_sha256"], "teacher_model": config["teacher_model"],
            "teacher_prompt_sha256": config["teacher_prompt_sha256"], "synthetic": True,
            "source_review_scope": "model_assessment_not_independent_fact_verification"}


def _context(slot, donor, bindings):
    from pipeline.augmentation_references import build_reference_context
    references = build_reference_context(donor["response_text"], slot["candidate_id"], category=slot["category"],
                                         donor_reference_map=bindings.get(donor["donor_id"]))
    return {"category": slot["category"], "recipe": api.RECIPES[slot["category"]],
            "variant": slot["index"] // len(api.CATEGORIES),
            "donor_response_text": references["donor_response_text"], "references": references,
            "reference_policy": references["reference_policy"]}, references


def _donor_sources(donors, bindings):
    """Compare restored donor bytes too; masked originals remain untouched."""
    from pipeline.augmentation_references import (
        bind_generated_source,
        build_reference_context,
    )
    seen = {donor["response_text"].strip() for donor in donors}
    for donor in donors:
        try:
            context = build_reference_context(donor["response_text"], "donor-deduplication", donor_reference_map=bindings.get(donor["donor_id"]))
            restored, _ = bind_generated_source(context["donor_response_text"], context)
            seen.add(restored.strip())
        except ValueError:
            # Invalid/unbound donors are rejected and audited at their slot.
            continue
    return seen


def _validated_source(generated, category, references):
    from pipeline.augmentation_references import bind_generated_source
    source = generated.get("response_text")
    if not isinstance(source, str) or not source.strip() or len(source) > 60000:
        raise ValueError("invalid_generated_source")
    if generated.get("category") != category or generated.get("synthetic") is not True:
        raise ValueError("invalid_source_category_or_synthetic_provenance")
    return bind_generated_source(source, references)


def _artifact(output, state, name, value):
    relative = "source_attempts/" + state["candidate_id"] + "/" + name + ".json"
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("Attempt artifact would be overwritten")
    _atomic(path, value)
    state["artifacts"].append({"path": relative, "sha256": _digest(path)})


def _call(output, state, adapter, *, kind, prompt, system, seed):
    number = state.get("call_count", 0) + 1
    state["call_count"] = number
    state["pending_call"] = {"kind": kind, "number": number, "seed": seed}
    _save(output, state)
    request = {"prompt": prompt, "system": system, "temperature": 0.9 if kind == "source" else 0.0,
               "max_tokens": 8192 if kind == "source" else 4096, "seed": seed, "json_mode": False}
    try:
        result = adapter.generate(**request)
        recorded = {key: getattr(result, key, None) for key in ("text", "error", "completion_complete", "latency_ms", "input_tokens", "output_tokens", "finish_reason")}
    except BaseException as exc:
        _artifact(output, state, f"{number:06d}_{kind}", {"request": request, "raised": f"{type(exc).__name__}: {exc}"})
        state.pop("pending_call", None)
        _save(output, state)
        raise
    _artifact(output, state, f"{number:06d}_{kind}", {"request": request, "result": recorded})
    state[kind + "_generation"] = recorded
    state.pop("pending_call", None)
    _save(output, state)
    return api._json_result(SimpleNamespace(**recorded))


def _source_candidate(output, state, slot, donor, bindings, config, adapter, system, seen):
    state["attempted"] = True
    audit = {"candidate_id": slot["candidate_id"], "augmentation": state["augmentation"]}
    try:
        context, references = _context(slot, donor, bindings)
        context_text = json.dumps(context, ensure_ascii=False)
        if state["status"] != "source_generated":
            state["source_attempts"] = state.get("source_attempts", 0) + 1
            state["status"] = "source_pending"
            _save(output, state)
            attempt_seed = config["seed"] + slot["index"] + (state["source_attempts"] - 1) * len(config["schedule"])
            generated = _call(output, state, adapter, kind="source", prompt="TASK: GENERATE_SOURCE\n" + context_text,
                              system=system, seed=attempt_seed)
            source, evidence = _validated_source(generated, slot["category"], references)
            if source.strip() in seen:
                raise ValueError("duplicate_or_unchanged_source")
            state.update(generated=generated, response_text=source, reference_evidence=evidence, status="source_generated")
            state["augmentation"]["source_sha256"] = api.text_hash(source)
            state["augmentation"]["reference_binding"] = evidence
            _save(output, state)
        review = _call(output, state, adapter, kind="review",
                       prompt="TASK: REVIEW_SOURCE\n" + context_text + "\nCANDIDATE:\n" + json.dumps(state["generated"], ensure_ascii=False),
                       system=system, seed=config["seed"] + slot["index"])
        # A completed negative verdict invalidates the source; malformed/incomplete
        # review may be retried without paying for an already valid source again.
        try:
            api.validate_review(review, slot["category"])
        except ValueError:
            state["status"] = "source_rejected"
            raise
        state.update(review=review, status="source_approved")
        seen.add(state["response_text"].strip())
        audit["status"] = "source_approved"
    except Exception as exc:  # noqa: BLE001 - persist candidate-local provider and validation failures
        if state["status"] not in ("source_generated", "source_rejected"):
            state["status"] = "source_rejected"
        audit.update(status="source_rejected", reason=f"{type(exc).__name__}: {exc}")
    audit.update({key: state[value] for key, value in (("source_generation", "source_generation"), ("source_review", "review_generation")) if value in state})
    _save(output, state)
    _audit(output, audit)


def _response(state):
    response = {"query_id": state["candidate_id"], "response_id": state["candidate_id"], "n_idx": 1,
                "response_text": state["response_text"], "scenario_family_id": state["augmentation"]["source_group_id"]}
    if state.get("stage3_repair_feedback") is not None:
        response["stage3_repair_feedback"] = state["stage3_repair_feedback"]
    return response


def _query(state):
    return {"query_id": state["candidate_id"], "query_text": "Render this synthetic training example faithfully.",
            "scenario_family_id": state["augmentation"]["source_group_id"]}


def _restore_fidelity_rejection(state: dict[str, Any]) -> bool:
    if state.get("status") not in ("stage3_rejected", "stage3_fidelity_rejected"):
        return False
    row = state.get("last_stage3_record")
    reasons = api.fidelity_regeneration_reasons(row, state.get("response_text", "")) if isinstance(row, dict) else []
    if not reasons:
        return False
    changed = state.get("status") != "stage3_fidelity_rejected" or state.get("fidelity_review_reasons") != reasons
    state["status"] = "stage3_fidelity_rejected"
    state["fidelity_review_reasons"] = reasons
    state.setdefault("fidelity_regeneration_attempts", 0)
    return changed


def _prepare_fidelity_regeneration(state: dict[str, Any]) -> bool:
    if state.get("status") != "stage3_fidelity_rejected":
        return False
    maximum = FIDELITY_REGENERATION_POLICY["max_regenerations"]
    attempts = state.get("fidelity_regeneration_attempts", 0)
    reasons = state.get("fidelity_review_reasons")
    if type(attempts) is not int or attempts < 0 or not isinstance(reasons, list) or not reasons:
        raise ValueError("Malformed fidelity regeneration checkpoint")
    if attempts >= maximum:
        return False
    attempts += 1
    state["fidelity_regeneration_attempts"] = attempts
    state["stage3_repair_feedback"] = {
        "kind": "semantic_fidelity",
        "review_reasons": reasons,
        "attempt": attempts,
        "max_attempts": maximum,
    }
    return True


def _round_records(output, round_dir, states, *, fatal_error=None):
    round_contract = _checked(round_dir / "contract.json")
    for name, digest in round_contract["input_hashes"].items():
        if _digest(round_dir / name) != digest:
            raise ValueError("Stage 3 attempt input hash mismatch")
    genui = round_dir / "genui.jsonl"
    try:
        records = api.read_jsonl(genui) if genui.exists() else []
    except (OSError, ValueError) as exc:
        records = []
        fatal_error = f"{type(exc).__name__}: {exc}"
    # Seal before clearing any in-flight checkpoint. A kill during admission then
    # leaves either recoverable in-flight state or an already sealed full attempt.
    evidence = {path.relative_to(round_dir).as_posix(): _digest(path) for path in round_dir.rglob("*") if path.is_file() and path.name not in ("receipt.json", "receipt.json.pending")}
    receipt_path = round_dir / "receipt.json"
    if receipt_path.exists():
        receipt = _checked(receipt_path)
        if receipt["files"] != evidence:
            raise ValueError("Stage 3 sealed attempt evidence changed")
        fatal_error = fatal_error or receipt.get("fatal_error")
    else:
        _atomic(receipt_path, _envelope({"files": evidence, "fatal_error": fatal_error}))
    grouped = {}
    for record in records:
        grouped.setdefault(record.get("response_id"), []).append(record)
    for candidate_id, source_hash in round_contract["sources"].items():
        state = states[candidate_id]
        if state["augmentation"]["source_sha256"] != source_hash or state["contract_sha256"] != round_contract["contract_sha256"]:
            raise ValueError("Stage 3 attempt candidate/source contract mismatch")
        rows = grouped.pop(candidate_id, [])
        reasons = ["missing_or_duplicate_stage3_candidate"] if len(rows) != 1 else api.admission_errors(rows[0], state["response_text"])
        if len(rows) == 1:
            state["last_stage3_record"] = rows[0]
        if not reasons:
            state["accepted_record"] = dict(rows[0], augmentation=state["augmentation"])
            state["status"] = "accepted"
            state.pop("fidelity_review_reasons", None)
            state.pop("stage3_repair_feedback", None)
        else:
            state["status"] = "stage3_rejected"
            state.pop("stage3_repair_feedback", None)
            if len(rows) == 1:
                _restore_fidelity_rejection(state)
                if state["status"] == "stage3_fidelity_rejected":
                    history = state.setdefault("fidelity_regeneration_history", [])
                    history.append({"stage3_attempt": round_dir.relative_to(output).as_posix(),
                                    "review_reasons": state["fidelity_review_reasons"],
                                    "record_sha256": _hash(rows[0])})
            if state["status"] != "stage3_fidelity_rejected":
                state.pop("fidelity_review_reasons", None)
        state.pop("stage3_inflight", None)
        _save(output, state)
        _audit(output, {"candidate_id": candidate_id, "augmentation": state["augmentation"],
                        "status": state["status"], "reasons": reasons,
                        "fidelity_review_reasons": state.get("fidelity_review_reasons", []),
                        "fidelity_regeneration_attempts": state.get("fidelity_regeneration_attempts", 0),
                        "fatal_error": fatal_error, "stage3_attempt": round_dir.relative_to(output).as_posix()})
    for candidate_id in grouped:
        _audit(output, {"candidate_id": candidate_id, "status": "stage3_rejected", "reasons": ["unknown_source_binding"]})
    return fatal_error


def _run_stage3(output, states, pending, config, adapter, runner):
    from utils.rate_limit import RateLimiter

    from pipeline.cache import PromptCache
    if runner is None:
        from pipeline.stage3_genui import run_stage3
        runner = run_stage3
    root = output / "stage3_attempts"
    root.mkdir(exist_ok=True)
    number = max([int(path.name) for path in root.iterdir() if path.is_dir()] or [0]) + 1
    round_dir = root / f"{number:06d}"
    round_dir.mkdir()
    _rows(round_dir / "queries.jsonl", [_query(state) for state in pending])
    _rows(round_dir / "responses.jsonl", [_response(state) for state in pending])
    _atomic_bytes(round_dir / "genui.jsonl", b"")
    _atomic(round_dir / "contract.json", _envelope({"contract_sha256": pending[0]["contract_sha256"],
            "sources": {state["candidate_id"]: state["augmentation"]["source_sha256"] for state in pending},
            "input_hashes": {name: _digest(round_dir / name) for name in ("queries.jsonl", "responses.jsonl")}}))
    for state in pending:
        state["stage3_inflight"] = round_dir.relative_to(output).as_posix()
        _save(output, state)
    logger = logging.getLogger("dataset.training_augmentation." + api.text_hash(str(output)))
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    logger.addHandler(handler)
    fatal_error = None
    try:
        runner(queries_path=round_dir / "queries.jsonl", responses_path=round_dir / "responses.jsonl",
               prompt_path=api.DATASET_ROOT / "prompts/genui_gen_mobile_a2ui_express_v1.md", adapter=adapter,
               genui_path=round_dir / "genui.jsonl", schema_path=api.DATASET_ROOT / "schema/canonical_ui_graph_v1.schema.json",
               artifacts_dir=round_dir / "artifacts", candidates_per_response=1, max_repair_attempts=1,
               max_tokens=12288, prompt_max_tokens=16000, seed=config["seed"], rate_limiter=RateLimiter(0),
               cache=PromptCache(output / "cache/prompts.jsonl", enabled=False), logger=logger,
               batch_size=1, max_total=len(pending), max_attempts=1, metric_version="v5_4",
               ir_formats=["a2ui_express_v1"], phase_invocation_id=api.VERSION + ":" + str(config["seed"]) + ":" + str(number))
    except Exception as exc:  # noqa: BLE001 - keep completed prefix and diagnostics for retry
        fatal_error = f"{type(exc).__name__}: {exc}"
    finally:
        handler.close()
        logger.removeHandler(handler)
    return _round_records(output, round_dir, states, fatal_error=fatal_error)


def _publish(output, config, contract, states, *, fatal_error=None, running=False):
    approved_statuses = ("source_approved", "stage3_rejected", "stage3_fidelity_rejected", "accepted")
    approved = [state for state in states.values() if state["status"] in approved_statuses]
    accepted = [state["accepted_record"] for state in states.values() if state["status"] == "accepted"]
    _rows(output / "queries.jsonl", [_query(state) for state in approved])
    _rows(output / "responses.jsonl", [_response(state) for state in approved])
    _rows(output / "genui.jsonl", [state["last_stage3_record"] for state in states.values() if "last_stage3_record" in state])
    _rows(output / "accepted_genui.jsonl", accepted)
    manifest = {key: value for key, value in config.items() if key not in ("schedule", "recipes_sha256", "optional_environment_sha256")}
    manifest.update(resume_contract_sha256=_hash(contract), synthetic=True,
                    source_review_scope="model_assessment_not_independent_fact_verification",
                    attempted_rows=sum(bool(state.get("attempted")) for state in states.values()), accepted_rows=len(accepted),
                    coverage={category: {"attempted": 0, "source_approved": 0, "accepted": 0} for category in api.CATEGORIES},
                    accepted_genui_sha256=_digest(output / "accepted_genui.jsonl"))
    for state in states.values():
        coverage = manifest["coverage"][state["augmentation"]["category"]]
        coverage["attempted"] += bool(state.get("attempted"))
        coverage["source_approved"] += state["status"] in approved_statuses
        coverage["accepted"] += state["status"] == "accepted"
    manifest["status"] = "running" if running else "failed" if fatal_error or not accepted else "completed"
    manifest["error"] = fatal_error or ("No eligible augmentation candidates" if not accepted and not running else None)
    manifest["projection_hashes"] = {name: _digest(output / name) for name in (*PROJECTIONS, "augmentation_audit.jsonl")}
    manifest["journal_hashes"] = {path.name: _digest(path) for path in (output / "candidates").glob("*.json")}
    if "legacy_origin" in contract:
        manifest["legacy_upgrade"] = contract["legacy_origin"]
    _atomic(output / "manifest.json", manifest)
    return manifest


def _migrate(output, config, states, donors, bindings):
    """Audit-bound legacy sources may survive; unknown/unbound sources regenerate."""
    legacy = output / "legacy_v1"
    for name in (*PROJECTIONS, "augmentation_audit.jsonl", "manifest.json"):
        destination = legacy / name
        if destination.exists() and _digest(destination) != _digest(output / name):
            raise ValueError("Legacy migration snapshot collision")
        shutil.copyfile(output / name, destination)
    audits = api.read_jsonl(legacy / "augmentation_audit.jsonl")
    responses = {row["response_id"]: row for row in api.read_jsonl(legacy / "responses.jsonl")}
    accepted = {row["response_id"]: row for row in api.read_jsonl(legacy / "accepted_genui.jsonl")}
    for slot in config["schedule"]:
        state = states[slot["candidate_id"]]
        matching = [audit for audit in audits if audit.get("candidate_id") == slot["candidate_id"] and audit.get("status") == "source_approved"]
        if not matching:
            continue
        state["attempted"] = True
        try:
            if len(matching) != 1:
                raise ValueError("legacy_source_audit_not_unique")
            audit = matching[0]
            provenance = audit.get("augmentation", {})
            if any(provenance.get(key) != value for key, value in state["augmentation"].items()):
                raise ValueError("legacy_source_provenance_mismatch")
            _, references = _context(slot, donors[slot["donor_id"]], bindings)
            generated = api._json_result(SimpleNamespace(**audit["source_generation"]))
            review = api._json_result(SimpleNamespace(**audit["source_review"]))
            api.validate_review(review, slot["category"])
            source, evidence = _validated_source(generated, slot["category"], references)
            if source != generated["response_text"] or source != responses[slot["candidate_id"]]["response_text"] or api.text_hash(source) != provenance.get("source_sha256"):
                raise ValueError("legacy_source_binding_mismatch")
            state.update(generated=generated, response_text=source, review=review, reference_evidence=evidence,
                         source_generation=audit["source_generation"], review_generation=audit["source_review"], status="source_approved")
            state["augmentation"] = provenance
            row = accepted.get(slot["candidate_id"])
            if row is not None and row.get("augmentation") == provenance and not api.admission_errors(row, source):
                state.update(accepted_record=row, last_stage3_record=row, status="accepted")
            _audit(output, {"candidate_id": slot["candidate_id"], "status": "legacy_reused", "reused": state["status"]})
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            state["status"] = "source_rejected"
            _audit(output, {"candidate_id": slot["candidate_id"], "status": "legacy_regenerate", "reason": str(exc)})
        _save(output, state)


def generate(donors_path, output_dir, *, teacher_model=api.DEFAULT_TEACHER, max_new_samples=90, seed=123,
             adapter=None, stage3_runner=None, resume=False, reference_bindings_path=None):
    config, donors, bindings, spec, prompt = _configuration(donors_path, teacher_model=teacher_model,
        max_new_samples=max_new_samples, seed=seed, reference_bindings_path=reference_bindings_path)
    if adapter is not None and adapter.spec != spec:
        raise ValueError("Injected adapter must match the registered teacher")
    output = _safe_path(output_dir)
    if resume:
        preflight = validate_generation_resume(donors_path, output, teacher_model=teacher_model,
            max_new_samples=max_new_samples, seed=seed, reference_bindings_path=reference_bindings_path)
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Augmentation output directory must be empty; use --resume for a compatible run")
        preflight = {"legacy_upgrade": False, "compatible_fidelity_upgrade": False}
    output.mkdir(parents=True, exist_ok=True)
    with _lock(output):
        if resume:
            validate_generation_resume(donors_path, output, teacher_model=teacher_model,
                max_new_samples=max_new_samples, seed=seed, reference_bindings_path=reference_bindings_path)
        contract_path = output / "resume_contract.json"
        contract = {"configuration": config}
        if preflight["legacy_upgrade"]:
            contract["legacy_origin"] = preflight["legacy_origin"]
            (output / "legacy_v1").mkdir(exist_ok=True)
            # Snapshot before publishing a new contract or replacing any projection.
            for name, digest in contract["legacy_origin"]["files"].items():
                shutil.copyfile(output / name, output / "legacy_v1" / name)
                if _digest(output / "legacy_v1" / name) != digest:
                    raise ValueError("Legacy snapshot hash mismatch")
        if contract_path.exists():
            contract = _checked(contract_path)
        else:
            _atomic(contract_path, _envelope(contract))
        (output / "candidates").mkdir(exist_ok=True)
        (output / "augmentation_audit.jsonl").touch(exist_ok=True)
        states = {}
        donor_by_id = {donor["donor_id"]: donor for donor in donors}
        seen = _donor_sources(donors, bindings)
        for slot in config["schedule"]:
            candidate_id = slot["candidate_id"]
            path = output / "candidates" / (candidate_id + ".json")
            if path.exists():
                state = _checked(path)
                if state.get("candidate_id") != candidate_id or state.get("contract_sha256") != _hash(contract):
                    raise ValueError("Candidate journal identity mismatch")
                expected = _provenance(config, slot)
                if any(state.get("augmentation", {}).get(key) != value for key, value in expected.items()):
                    raise ValueError("Candidate journal provenance mismatch")
                if state["status"] in ("source_generated", "source_approved", "stage3_rejected",
                                       "stage3_fidelity_rejected", "accepted"):
                    _, references = _context(slot, donor_by_id[slot["donor_id"]], bindings)
                    source, _ = _validated_source(state["generated"], slot["category"], references)
                    if source != state["response_text"] or api.text_hash(source) != state["augmentation"].get("source_sha256") or source.strip() in seen:
                        raise ValueError("Candidate checkpoint source binding/uniqueness mismatch")
                    seen.add(source.strip())
                    if state["status"] != "source_generated":
                        api.validate_review(state["review"], slot["category"])
                    if state["status"] == "accepted" and (api.admission_errors(state["accepted_record"], source) or state["accepted_record"].get("augmentation") != state["augmentation"]):
                        raise ValueError("Saved accepted candidate no longer passes admission")
                    if _restore_fidelity_rejection(state):
                        _save(output, state)
            else:
                state = {"candidate_id": candidate_id, "contract_sha256": _hash(contract), "augmentation": _provenance(config, slot),
                         "status": "pending", "attempted": False, "artifacts": []}
                _save(output, state)
            states[candidate_id] = state
        if preflight["legacy_upgrade"]:
            _migrate(output, config, states, donor_by_id, bindings)
            seen.update(state["response_text"].strip() for state in states.values() if state["status"] in ("accepted", "source_approved"))
        # Initial manifest is durable before any network/provider work.
        _publish(output, config, contract, states, running=True)
        rounds = {state["stage3_inflight"] for state in states.values() if state.get("stage3_inflight")}
        for relative in sorted(rounds):
            _round_records(output, output / relative, states)
        needs_work = any(state["status"] != "accepted" for state in states.values())
        fatal_error = None
        with api.bounded_teacher_environment():
            if needs_work and adapter is None:
                from llm.factory import build_adapter
                adapter = build_adapter(spec)
            for slot in config["schedule"]:
                state = states[slot["candidate_id"]]
                if state["status"] not in ("accepted", "source_approved", "stage3_rejected", "stage3_fidelity_rejected"):
                    _source_candidate(output, state, slot, donor_by_id[slot["donor_id"]], bindings, config, adapter, prompt, seen)
            pending = []
            for state in states.values():
                if state["status"] in ("source_approved", "stage3_rejected") or _prepare_fidelity_regeneration(state):
                    pending.append(state)
            if pending:
                fatal_error = _run_stage3(output, states, pending, config, adapter, stage3_runner)
            while not fatal_error:
                retry = [state for state in states.values() if _prepare_fidelity_regeneration(state)]
                if not retry:
                    break
                fatal_error = _run_stage3(output, states, retry, config, adapter, stage3_runner)
        manifest = _publish(output, config, contract, states, fatal_error=fatal_error)
        if manifest["status"] != "completed":
            raise RuntimeError(f"Augmentation failed: {manifest['error']}; diagnostics: {output}")
        return manifest
