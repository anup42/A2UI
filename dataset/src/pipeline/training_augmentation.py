"""Dataset-owned, bounded synthetic source augmentation and Stage 3 labeling.

Training supplies already split training donors. This module never reads a
heldout dataset, donor targets, or edits a generated label. Model source review
is an assessment of synthetic coherence, never independent fact verification.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

VERSION = "training-augmentation-v1"
DEFAULT_TEACHER = "muse_glimmer_30b_sglang_reasoning_dflash"
DATASET_ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = (
    "multisection_completeness", "table_count_variations", "coherent_numeric_entities",
    "action_no_action_contrasts", "wording_formats", "missing_uncertain_facts",
    "length_position", "forms_rare_controls", "literal_media_references",
)
RECIPES = {
    "multisection_completeness": "Produce at least three meaningful sections, with facts in every section and an essential final section. Preserve all required facts across sections.",
    "table_count_variations": "Produce a coherent comparison or schedule table; vary row and column counts, explicit totals and singular/plural wording. All cells and counts must agree.",
    "coherent_numeric_entities": "Create synthetic substituted entities and numeric values. Recompute every dependent total, percentage, date/time constraint and summary; never perform isolated numeric replacement.",
    "action_no_action_contrasts": "Alternate by variant parity: even variants have explicit grounded Action: [Button: label] destination lines; odd variants are informational with no action requests, buttons or invented action destinations.",
    "wording_formats": "Rewrite the same information using a different natural wording and presentation (prose, bullets, headings or table), preserving semantic roles and exact required literals.",
    "missing_uncertain_facts": "Make selected facts explicitly unavailable, unknown, tentative or conditional. Keep qualifications and uncertainty visible; never replace missing facts with plausible invented answers.",
    "length_position": "Vary source length and ordering. Place required facts at both beginning and end and preserve middle sections. Odd variants are compact; even variants are longer with useful concrete details, not repetition.",
    "forms_rare_controls": "Describe a synthetic usable form with explicit labels, initial values, allowed options, required/optional status and constraints. Rotate among supported input, date/time, slider, checkbox and choice controls. State any submit operation as mock; never imply a real external capability.",
    "literal_media_references": "Stress exact punctuation, quotes, identifiers, Unicode, URLs and declared media references. Retain original supplied references verbatim when used; optional new references must be explicit synthetic placeholders. Never claim a media asset exists or was downloaded.",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def text_hash(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Malformed JSONL at {path.name}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path.name}:{line_number}")  # noqa: TRY004 - external JSON schema violation
            rows.append(row)
    return rows


def load_donors(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("No training donors supplied")
    ids: set[str] = set()
    sources: set[str] = set()
    for row in rows:
        for key in ("donor_id", "source_group_id", "response_text"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Donor requires nonempty {key}")
        if row.get("split") != "train":
            raise ValueError("Only explicit split='train' donors are permitted")
        source = row["response_text"].strip()
        if row["donor_id"] in ids or source in sources:
            raise ValueError("Duplicate donor ID or exact source text")
        if len(source) > 60000:
            raise ValueError("Donor source exceeds the 60000-character budget")
        ids.add(row["donor_id"])
        sources.add(source)
    # Deliberately discard donor labels, contracts, review claims and other fields.
    return [{key: row[key] for key in ("donor_id", "source_group_id", "response_text", "split")} for row in rows]


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _json_result(result: Any) -> dict[str, Any]:
    if result.error or result.completion_complete is not True:
        raise ValueError("teacher_completion_failed_or_incomplete")
    value = json.loads(result.text)
    if not isinstance(value, dict):
        raise ValueError("teacher_result_not_object")  # noqa: TRY004 - external JSON schema violation
    return value


def validate_review(value: dict[str, Any], category: str) -> None:
    if value.get("category") != category:
        raise ValueError("source_review_category_mismatch")
    for key in ("approved", "coherent", "category_satisfied", "synthetic_provenance_clear"):
        if value.get(key) is not True:
            raise ValueError(f"source_review_failed:{key}")
    if value.get("issues") != []:
        raise ValueError("source_review_has_issues_or_malformed_issues")


def admission_errors(row: dict[str, Any], response_text: str) -> list[str]:
    reasons = []
    if row.get("record_status") != "accepted":
        reasons.append("stage3_not_accepted")
    acceptance = row.get("training_acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("eligible") is not True:
        reasons.append("training_not_eligible")
    if not isinstance(acceptance, dict) or acceptance.get("blocking_reasons") != [] or acceptance.get("review_reasons") != []:
        reasons.append("training_blocking_or_review_reasons")
    validation = row.get("validation") or {}
    for field in ("schema_valid_strict", "standard_a2ui_valid"):
        if validation.get(field) is not True:
            reasons.append(field)
    gen = row.get("gen") or {}
    if gen.get("error") or gen.get("completion_complete") is not True:
        reasons.append("generation_failed_or_incomplete")
    if row.get("response_text") != response_text:
        reasons.append("source_binding_mismatch")
    if row.get("source_format") != "a2ui_express_v1" or not isinstance(row.get("a2ui_express"), str) or not row["a2ui_express"].strip():
        reasons.append("missing_express_completion")
    return reasons


@contextmanager
def bounded_teacher_environment():
    # Isolated dataset subprocess; restore settings for embedded/test callers.
    overrides = {
        "LOCAL_ALLOW_HTTP_ENDPOINT": "1", "LOCAL_STRICT_OFFLINE": "0",
        "LOCAL_VLLM_ENABLE_THINKING": "1", "LOCAL_MUSE_REQUIRE_REASONING": "1",
        "LOCAL_VLLM_SEND_CHAT_TEMPLATE_KWARGS": "1",
        "LOCAL_VLLM_TIMEOUT_SECONDS": "180", "LOCAL_VLLM_RETRY_CONNECTION_ERRORS": "0",
        "LOCAL_VLLM_RETRY_RESULT_ERRORS": "0", "LOCAL_VLLM_RETRY_MAX_SECONDS": "180",
        "LOCAL_VLLM_BATCH_PARALLELISM": "1", "LOCAL_VLLM_PARALLEL_REQUESTS": "1",
        "STAGE3_FINAL_REGEN_ATTEMPTS": "1", "STAGE3_RESPECT_CONFIG_PROMPT_MAX": "1",
        "LOCAL_VLLM_MAX_OUTPUT_TOKENS": "12288",
        "LOCAL_VLLM_REASONING_STRENGTH": os.environ.get("LOCAL_VLLM_REASONING_STRENGTH", "high"),
        "LOCAL_VLLM_SERVED_MODEL": os.environ.get("LOCAL_VLLM_SERVED_MODEL", "muse-glimmer"),
    }
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield overrides
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def generate_training_augmentations(
    donors_path: Path, output_dir: Path, *, teacher_model: str = DEFAULT_TEACHER,
    max_new_samples: int = 90, seed: int = 123, adapter: Any = None,
    stage3_runner: Any = None,
) -> dict[str, Any]:
    """Generate at most max_new_samples source candidates, returning an audit manifest.

    adapter/stage3_runner are dependency-injection seams for offline tests only.
    The CLI always resolves the exact registered teacher and existing Stage 3.
    """
    if type(max_new_samples) is not int or not 1 <= max_new_samples <= 100000:
        raise ValueError("max_new_samples must be an integer in [1, 100000]")
    donors_path, output_dir = Path(donors_path).resolve(), Path(output_dir).resolve()
    donors = load_donors(donors_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Augmentation output directory must be empty; use a fresh invocation directory")
    from llm.factory import build_adapter, load_model_specs
    from utils.config import load_yaml
    specs = [item for item in load_model_specs(load_yaml(DATASET_ROOT / "configs/models.yaml")) if item.name == teacher_model]
    if len(specs) != 1:
        raise ValueError(f"Expected exactly one registered teacher: {teacher_model}")
    spec = specs[0]
    if spec.name != DEFAULT_TEACHER or spec.provider != "local" or not str(spec.endpoint).startswith(("http://", "https://")):
        raise ValueError("This version requires the registered Muse HTTP teacher; no fallback is allowed")
    if adapter is not None and adapter.spec != spec:
        raise ValueError("Injected adapter must match the registered teacher")
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = DATASET_ROOT / "prompts/training_augmentation_v1.md"
    prompt = prompt_path.read_text(encoding="utf-8")
    files = {name: output_dir / name for name in ("queries.jsonl", "responses.jsonl", "genui.jsonl", "accepted_genui.jsonl", "augmentation_audit.jsonl")}
    for path in files.values():
        path.touch()
    code_paths = [Path(__file__), DATASET_ROOT / "scripts/generate_training_augmentations.py", DATASET_ROOT / "src/pipeline/stage3_genui.py", DATASET_ROOT / "src/pipeline/genui_quality/acceptance_v5_4.py"]
    manifest: dict[str, Any] = {
        "version": VERSION, "donors_sha256": sha256_bytes(donors_path.read_bytes()),
        "teacher_model": teacher_model, "teacher_identity": asdict(spec), "seed": seed,
        "max_new_samples": max_new_samples, "attempted_rows": 0, "accepted_rows": 0,
        "categories": list(CATEGORIES), "coverage": {category: {"attempted": 0, "source_approved": 0, "accepted": 0} for category in CATEGORIES},
        "teacher_prompt_sha256": text_hash(prompt),
        "code_hashes": {str(path.relative_to(DATASET_ROOT)): sha256_bytes(path.read_bytes()) for path in code_paths},
        "stage3_prompt_sha256": sha256_bytes((DATASET_ROOT / "prompts/genui_gen_mobile_a2ui_express_v1.md").read_bytes()),
        "muse_stage3_prompt_sha256": sha256_bytes((DATASET_ROOT / "prompts/muse_stage3_quality_v1.md").read_bytes()),
        "contract_sha256": sha256_bytes((DATASET_ROOT / "schema/canonical_ui_graph_v1.schema.json").read_bytes()),
        "synthetic": True, "source_review_scope": "model_assessment_not_independent_fact_verification",
        "status": "running",
    }
    approved: dict[str, dict[str, Any]] = {}
    seen = {donor["response_text"].strip() for donor in donors}
    random.Random(seed).shuffle(donors)
    fatal_error = None
    stage_logger = None
    try:
        with bounded_teacher_environment() as environment:
            manifest["teacher_environment"] = dict(environment)
            manifest["teacher_endpoints"] = os.environ.get("LOCAL_VLLM_ENDPOINTS") or spec.endpoint
            adapter = adapter if adapter is not None else build_adapter(spec)
            for index in range(min(max_new_samples, len(donors))):
                category = CATEGORIES[index % len(CATEGORIES)]
                donor = donors[index % len(donors)]
                candidate_id = "aug_" + text_hash(f"{manifest['donors_sha256']}:{seed}:{index}:{category}")[:24]
                provenance = {"version": VERSION, "category": category, "donor_id": donor["donor_id"],
                              "source_group_id": donor["source_group_id"], "seed": seed, "candidate_seed": seed + index,
                              "donor_source_sha256": text_hash(donor["response_text"]), "teacher_model": teacher_model,
                              "teacher_prompt_sha256": text_hash(prompt), "synthetic": True,
                              "source_review_scope": manifest["source_review_scope"]}
                audit: dict[str, Any] = {"candidate_id": candidate_id, "augmentation": provenance}
                manifest["attempted_rows"] += 1
                manifest["coverage"][category]["attempted"] += 1
                context = json.dumps({"category": category, "recipe": RECIPES[category], "variant": index // len(CATEGORIES),
                                      "donor_response_text": donor["response_text"]}, ensure_ascii=False)
                try:
                    source_call = adapter.generate(prompt="TASK: GENERATE_SOURCE\n" + context, system=prompt,
                                                   temperature=0.9, max_tokens=8192, seed=seed + index, json_mode=False)
                    audit["source_generation"] = {"text": source_call.text, "error": source_call.error,
                        "completion_complete": source_call.completion_complete, "latency_ms": source_call.latency_ms,
                        "input_tokens": source_call.input_tokens, "output_tokens": source_call.output_tokens}
                    generated = _json_result(source_call)
                    source = generated.get("response_text")
                    if not isinstance(source, str) or not source.strip() or len(source) > 60000:
                        raise ValueError("invalid_generated_source")
                    if generated.get("category") != category or generated.get("synthetic") is not True:
                        raise ValueError("invalid_source_category_or_synthetic_provenance")
                    if source.strip() in seen:
                        raise ValueError("duplicate_or_unchanged_source")
                    seen.add(source.strip())
                    provenance["source_sha256"] = text_hash(source)
                    review_call = adapter.generate(prompt="TASK: REVIEW_SOURCE\n" + context + "\nCANDIDATE:\n" + json.dumps(generated, ensure_ascii=False),
                                                   system=prompt, temperature=0.0, max_tokens=4096, seed=seed + index, json_mode=False)
                    audit["source_review"] = {"text": review_call.text, "error": review_call.error,
                        "completion_complete": review_call.completion_complete, "latency_ms": review_call.latency_ms,
                        "input_tokens": review_call.input_tokens, "output_tokens": review_call.output_tokens}
                    review = _json_result(review_call)
                    validate_review(review, category)
                    manifest["coverage"][category]["source_approved"] += 1
                    query = {"query_id": candidate_id, "query_text": "Render this synthetic training example faithfully.", "scenario_family_id": donor["source_group_id"]}
                    response = {"query_id": candidate_id, "response_id": candidate_id, "n_idx": 1,
                                "response_text": source, "scenario_family_id": donor["source_group_id"]}
                    _append(files["queries.jsonl"], query)
                    _append(files["responses.jsonl"], response)
                    approved[candidate_id] = {"response_text": source, "augmentation": dict(provenance)}
                    audit["status"] = "source_approved"
                except Exception as exc:  # noqa: BLE001 - retain an audit for any failed candidate/provider
                    audit.update(status="source_rejected", reason=f"{type(exc).__name__}: {exc}")
                _append(files["augmentation_audit.jsonl"], audit)
            if approved:
                from utils.rate_limit import RateLimiter

                from pipeline.cache import PromptCache
                if stage3_runner is None:
                    from pipeline.stage3_genui import run_stage3
                    stage3_runner = run_stage3
                stage_logger = logging.getLogger("dataset.training_augmentation." + text_hash(str(output_dir)))
                stage_logger.propagate = False
                stage_logger.setLevel(logging.INFO)
                handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
                handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
                stage_logger.addHandler(handler)
                stage3_runner(queries_path=files["queries.jsonl"], responses_path=files["responses.jsonl"],
                    prompt_path=DATASET_ROOT / "prompts/genui_gen_mobile_a2ui_express_v1.md", adapter=adapter,
                    genui_path=files["genui.jsonl"], schema_path=DATASET_ROOT / "schema/canonical_ui_graph_v1.schema.json",
                    artifacts_dir=output_dir / "artifacts", candidates_per_response=1, max_repair_attempts=1,
                    max_tokens=12288, prompt_max_tokens=16000, seed=seed, rate_limiter=RateLimiter(0),
                    cache=PromptCache(output_dir / "cache/prompts.jsonl", enabled=False), logger=stage_logger,
                    batch_size=1, max_total=len(approved), max_attempts=1, metric_version="v5_4",
                    ir_formats=["a2ui_express_v1"], phase_invocation_id=VERSION + ":" + str(seed))
    except Exception as exc:  # noqa: BLE001 - publish failure diagnostics before aborting the invocation
        fatal_error = f"{type(exc).__name__}: {exc}"
    finally:
        if stage_logger is not None:
            for handler in stage_logger.handlers:
                handler.close()
            stage_logger.handlers.clear()
    try:
        try:
            stage3_rows = read_jsonl(files["genui.jsonl"])
        except (OSError, ValueError) as exc:
            stage3_rows = []
            fatal_error = f"{type(exc).__name__}: {exc}"
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in stage3_rows:
            grouped.setdefault(str(row.get("response_id")), []).append(row)
        for candidate_id, candidate in approved.items():
            rows = grouped.pop(candidate_id, [])
            reasons = ["missing_or_duplicate_stage3_candidate"] if len(rows) != 1 else admission_errors(rows[0], candidate["response_text"])
            if not reasons and not fatal_error:
                row = dict(rows[0], augmentation=candidate["augmentation"])
                _append(files["accepted_genui.jsonl"], row)
                manifest["accepted_rows"] += 1
                manifest["coverage"][candidate["augmentation"]["category"]]["accepted"] += 1
            _append(files["augmentation_audit.jsonl"], {"candidate_id": candidate_id, "augmentation": candidate["augmentation"],
                "status": "stage3_rejected" if reasons or fatal_error else "accepted", "reasons": reasons,
                "fatal_error": fatal_error})
        for candidate_id in grouped:
            _append(files["augmentation_audit.jsonl"], {"candidate_id": candidate_id, "status": "stage3_rejected", "reasons": ["unknown_source_binding"]})
    except Exception as exc:  # noqa: BLE001 - manifest records malformed provider artifacts and admission failures
        fatal_error = f"{type(exc).__name__}: {exc}"
    manifest["accepted_genui_sha256"] = sha256_bytes(files["accepted_genui.jsonl"].read_bytes())
    manifest["status"] = "failed" if fatal_error or not manifest["accepted_rows"] else "completed"
    manifest["error"] = fatal_error or ("No eligible augmentation candidates" if not manifest["accepted_rows"] else None)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if manifest["status"] != "completed":
        raise RuntimeError(f"Augmentation failed: {manifest['error']}; diagnostics: {output_dir}")
    return manifest
