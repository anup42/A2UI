#!/usr/bin/env python3
"""Score and report a pulled GenUiTrainedBixby50Test run without inventing data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import statistics
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_SRC = REPO_ROOT / "training" / "src"
FROZEN_CORPUS = REPO_ROOT / "training" / "data" / "eval" / "bixby50_v1" / "bixby50.jsonl"
CORPUS_MANIFEST = FROZEN_CORPUS.with_name("benchmark_manifest.json")
ANDROID_CORPUS = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "genuicraft_bixby50.jsonl"
METRIC_VERSION = "v5_4"

if str(TRAINING_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINING_SRC))

from ir_training.eval.metrics import aggregate_scores, score_prediction


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
    )


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def int_or_none(value: Any) -> int | None:
    number = as_number(value)
    return int(number) if number is not None and number >= 0 and number.is_integer() else None


def median(values: Iterable[float | int | None]) -> float | None:
    observed = [float(value) for value in values if value is not None]
    return statistics.median(observed) if observed else None


def fmt_number(value: Any, digits: int = 2) -> str:
    number = as_number(value)
    if number is None:
        return "unavailable"
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.{digits}f}"


def frozen_index() -> list[dict[str, Any]]:
    rows = read_jsonl(FROZEN_CORPUS)
    if len(rows) != 50:
        raise ValueError(f"Frozen Bixby50 corpus has {len(rows)} rows, expected 50")
    ids = [str(row.get("id") or "") for row in rows]
    if any(not case_id for case_id in ids) or len(set(ids)) != 50:
        raise ValueError("Frozen Bixby50 corpus IDs are blank or duplicated")
    return rows


def expected_source(row: Mapping[str, Any]) -> dict[str, str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    return {
        "id": str(row["id"]),
        "query": str(metadata.get("original_query") or ""),
        "text": str(row.get("response_text") or ""),
        "domain": str(metadata.get("domain") or ""),
    }


def verify_source(path: Path, frozen: Mapping[str, Any]) -> tuple[bool, str | None]:
    if not path.is_file():
        return False, "source.json is missing"
    try:
        source = read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, f"source.json could not be parsed: {type(exc).__name__}: {exc}"
    if not isinstance(source, dict):
        return False, "source.json is not an object"
    expected = expected_source(frozen)
    differences = [name for name, value in expected.items() if source.get(name) != value]
    if differences:
        return False, "source.json differs from frozen fields: " + ", ".join(differences)
    return True, None


def result_metrics(result: Mapping[str, Any] | None) -> Mapping[str, Any]:
    value = result.get("metrics") if isinstance(result, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def render_training_prompt(system: str, initial_messages: list[dict[str, str]], user: str) -> str:
    parts = ["<bos><|turn>system\n", system.strip(), "<turn|>\n"]
    role_names = {"USER": "user", "MODEL": "model"}
    for message in initial_messages:
        role = message["role"].upper()
        if role not in role_names:
            raise ValueError(f"unsupported prompt role: {message['role']!r}")
        parts.extend(
            ["<|turn>", role_names[role], "\n", message["text"].strip(), "<turn|>\n"]
        )
    parts.extend(["<|turn>user\n", user.strip(), "<turn|>\n<|turn>model\n"])
    return "".join(parts)


def prompt_fields_from_json(prompt: Mapping[str, Any]) -> tuple[str, list[dict[str, str]], str]:
    system = prompt.get("system")
    user = prompt.get("user")
    messages = prompt.get("initialMessages")
    if not isinstance(system, str) or not isinstance(user, str) or not isinstance(messages, list):
        raise TypeError("prompt.json lacks string system/user or list initialMessages")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"prompt.json initialMessages[{index}] is not an object")
        role = message.get("role")
        text = message.get("text")
        if not isinstance(role, str) or not isinstance(text, str):
            raise TypeError(f"prompt.json initialMessages[{index}] lacks string role/text")
        normalized.append({"role": role.upper(), "text": text})
    return system, normalized, user


def prompt_fields_from_contract(
    prompt_document: Mapping[str, Any],
    response_text: str,
) -> tuple[str, list[dict[str, str]], str]:
    scaffold = prompt_document.get("scaffold")
    if not isinstance(scaffold, Mapping):
        raise TypeError("shared_prompt.json lacks scaffold")
    messages = scaffold.get("messages")
    task_prefix = scaffold.get("task_prefix")
    if not isinstance(messages, list) or len(messages) != 3 or not isinstance(task_prefix, str):
        raise ValueError("shared_prompt.json scaffold must contain three messages and task_prefix")
    expected_roles = ["system", "user", "assistant"]
    contents: list[str] = []
    for index, (message, expected_role) in enumerate(zip(messages, expected_roles, strict=True)):
        if not isinstance(message, Mapping):
            raise TypeError(f"shared_prompt.json message {index} is not an object")
        role = message.get("role")
        content = message.get("content")
        if role != expected_role or not isinstance(content, str):
            raise ValueError(
                f"shared_prompt.json message {index} must be a string {expected_role!r} message"
            )
        contents.append(content)
    return (
        contents[0],
        [
            {"role": "USER", "text": contents[1]},
            {"role": "MODEL", "text": contents[2]},
        ],
        task_prefix + response_text.strip(),
    )


def audit_rendered_prompt(
    case_dir: Path,
    result: Mapping[str, Any] | None,
    selected: bool,
    prompt_document: Mapping[str, Any],
    response_text: str,
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "status": "not_selected" if not selected else "unavailable",
        "expectedSource": None,
        "expectedRenderedPromptSha256": None,
        "promptRecordedExpectedRenderedPromptSha256": None,
        "nativeRenderedPromptSha256": None,
        "promptJsonMatchesContract": None,
        "promptRecordedHashMatchesExpected": None,
        "nativeHashMatchesExpected": None,
        "reason": None,
    }
    if not selected:
        return audit

    native_hash = result.get("renderedPromptSha256") if isinstance(result, Mapping) else None
    if isinstance(native_hash, str) and native_hash:
        audit["nativeRenderedPromptSha256"] = native_hash

    prompt_path = case_dir / "prompt.json"
    prompt: Mapping[str, Any] | None = None
    prompt_error: str | None = None
    if prompt_path.is_file():
        try:
            loaded = read_json(prompt_path)
            if not isinstance(loaded, Mapping):
                raise TypeError("prompt.json is not an object")
            prompt = loaded
            recorded_hash = prompt.get("expectedRenderedPromptSha256")
            if isinstance(recorded_hash, str) and recorded_hash:
                audit["promptRecordedExpectedRenderedPromptSha256"] = recorded_hash
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            prompt_error = f"prompt.json could not be used: {type(exc).__name__}: {exc}"
    else:
        prompt_error = "prompt.json is missing"

    expected_fields: tuple[str, list[dict[str, str]], str] | None = None
    try:
        expected_fields = prompt_fields_from_contract(prompt_document, response_text)
        audit["expectedSource"] = "shared_prompt.json plus frozen response"
    except (TypeError, ValueError):
        if prompt is not None:
            try:
                expected_fields = prompt_fields_from_json(prompt)
                audit["expectedSource"] = "BOS turn serialization reconstructed from prompt.json"
            except (TypeError, ValueError) as exc:
                prompt_error = str(exc)

    if expected_fields is None:
        audit["reason"] = prompt_error or "expected training prompt inputs are unavailable"
        return audit

    expected_hash = sha256_text(render_training_prompt(*expected_fields))
    audit["expectedRenderedPromptSha256"] = expected_hash

    prompt_mismatch = False
    if prompt is not None:
        try:
            saved_fields = prompt_fields_from_json(prompt)
            prompt_matches = saved_fields == expected_fields
            audit["promptJsonMatchesContract"] = prompt_matches
            prompt_mismatch = not prompt_matches
        except (TypeError, ValueError) as exc:
            prompt_error = str(exc)
            prompt_mismatch = True
        recorded_hash = audit["promptRecordedExpectedRenderedPromptSha256"]
        if recorded_hash is not None:
            recorded_matches = recorded_hash == expected_hash
            audit["promptRecordedHashMatchesExpected"] = recorded_matches
            prompt_mismatch = prompt_mismatch or not recorded_matches

    if audit["nativeRenderedPromptSha256"] is None:
        if prompt_mismatch:
            audit["status"] = "mismatch"
            audit["reason"] = prompt_error or "saved prompt provenance differs from the training contract"
        else:
            audit["reason"] = "native renderedPromptSha256 is unavailable"
        return audit

    native_matches = audit["nativeRenderedPromptSha256"] == expected_hash
    audit["nativeHashMatchesExpected"] = native_matches
    if native_matches and not prompt_mismatch:
        audit["status"] = "verified"
    else:
        audit["status"] = "mismatch"
        reasons = []
        if not native_matches:
            reasons.append("native rendered-prompt hash differs from the expected training prompt")
        if prompt_mismatch:
            reasons.append("saved prompt provenance differs from the training contract")
        audit["reason"] = "; ".join(reasons)
    return audit


def runtime_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    returned = [case for case in cases if case["returned_output"]]
    rates: list[float] = []
    weighted_seconds = 0.0
    weighted_tokens = 0
    input_total = 0
    output_total = 0
    input_observed = 0
    output_observed = 0
    cap_count = 0
    for case in cases:
        metrics = case["native_metrics"]
        input_tokens = int_or_none(metrics.get("inputTokens"))
        output_tokens = int_or_none(metrics.get("outputTokens"))
        rate = as_number(metrics.get("decodeTokensPerSecond"))
        if input_tokens is not None:
            input_total += input_tokens
            input_observed += 1
        if output_tokens is not None:
            output_total += output_tokens
            output_observed += 1
            cap_count += int(output_tokens >= 2_048)
        if rate is not None and rate > 0:
            rates.append(rate)
            if output_tokens is not None and output_tokens > 0:
                weighted_tokens += output_tokens
                weighted_seconds += output_tokens / rate

    cold = next((case for case in cases if case["first_in_run"]), None)
    warm = [case for case in returned if not case["first_in_run"]]
    weighted_rate = weighted_tokens / weighted_seconds if weighted_seconds > 0 else None
    return {
        "nativeDecodeTokensPerSecond": {
            "observedCases": len(rates),
            "median": statistics.median(rates) if rates else None,
            "min": min(rates) if rates else None,
            "max": max(rates) if rates else None,
            "tokenWeightedOverall": weighted_rate,
            "tokenWeightedOutputTokens": weighted_tokens if weighted_seconds > 0 else None,
            "tokenWeightedDecodeSeconds": weighted_seconds if weighted_seconds > 0 else None,
        },
        "tokens": {
            "inputTotal": input_total if input_observed else None,
            "inputObservedCases": input_observed,
            "outputTotal": output_total if output_observed else None,
            "outputObservedCases": output_observed,
            "outputAtOrAbove2048Count": cap_count if output_observed else None,
        },
        "latency": {
            "coldFirstCaseId": cold["id"] if cold else None,
            "coldFirstEndToEndMs": cold["elapsed_ms"] if cold else None,
            "coldFirstProviderCallMs": cold["provider_call_ms"] if cold else None,
            "warmReturnedOutputCases": len(warm),
            "warmMedianEndToEndMs": median(case["elapsed_ms"] for case in warm),
            "warmMedianProviderCallMs": median(case["provider_call_ms"] for case in warm),
        },
        "measurementNote": (
            "Native decode tokens/s excludes engine initialization and prompt prefill. "
            "End-to-end elapsedMs includes the converter/provider request path; the first case is cold."
        ),
    }


def source_fidelity_averages(scored: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[float]] = {}
    for row in scored:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else {}
        coverage = as_number(metrics.get("content_coverage"))
        if coverage is not None:
            buckets.setdefault("content_coverage", []).append(coverage)
        fidelity = metrics.get("fidelity_atomics_v5_4")
        if isinstance(fidelity, Mapping):
            for key, value in fidelity.items():
                number = as_number(value)
                if number is not None:
                    buckets.setdefault(str(key), []).append(number)
    return {
        "denominator": (
            "observed numeric values among scored outputs; null, absent, and inapplicable values "
            "are excluded from each displayed mean"
        ),
        "scoredCount": len(scored),
        "metrics": {
            key: {
                "average": sum(values) / len(values),
                "observedCount": len(values),
                "scoredCount": len(scored),
                "excludedAsUnavailableOrInapplicableCount": len(scored) - len(values),
            }
            for key, values in sorted(buckets.items())
        },
    }


def relative_href(path: Path, run_dir: Path) -> str:
    return quote(path.relative_to(run_dir).as_posix(), safe="/._-")


def artifact_link(path: Path, run_dir: Path, label: str) -> str:
    if not path.is_file():
        return ""
    href = html.escape(relative_href(path, run_dir), quote=True)
    return f'<a href="{href}">{html.escape(label)}</a>'


def prompt_parity_text(parity: Mapping[str, Any]) -> str:
    reported = int_or_none(parity.get("nativeHashReportedCount")) or 0
    if reported == 0:
        unavailable = int_or_none(parity.get("unavailableCount")) or 0
        return (
            "unavailable: no native rendered-prompt hash was recorded"
            + (f" ({unavailable} selected cases unavailable)" if unavailable else "")
        )
    verified = int_or_none(parity.get("verifiedCount")) or 0
    mismatches = int_or_none(parity.get("mismatchCount")) or 0
    unavailable = int_or_none(parity.get("unavailableCount")) or 0
    return (
        f"{verified}/{reported} reported hashes verified; "
        f"{mismatches} mismatches; {unavailable} selected cases unavailable"
    )


def gallery_html(
    run_dir: Path,
    cases: list[dict[str, Any]],
    counts: Mapping[str, Any],
    prompt_parity: Mapping[str, Any],
    model_audit: Path,
) -> str:
    cards: list[str] = []
    for case in cases:
        case_dir = run_dir / case["id"]
        source_html = html.escape(case["source_text"]).replace(" \n", "&#32;\n")
        links = [
            artifact_link(case_dir / "source.json", run_dir, "source.json"),
            artifact_link(case_dir / "output.express", run_dir, "output.express"),
            artifact_link(case_dir / "a2ui.json", run_dir, "a2ui.json"),
            artifact_link(case_dir / "screen.png", run_dir, "screen.png"),
            artifact_link(case_dir / "screen_scrolled.png", run_dir, "screen_scrolled.png"),
        ]
        screen = case_dir / "screen.png"
        image = (
            f'<a href="{html.escape(relative_href(screen, run_dir), quote=True)}">'
            f'<img loading="lazy" src="{html.escape(relative_href(screen, run_dir), quote=True)}" '
            f'alt="{html.escape(case["id"])} rendered screen"></a>'
            if screen.is_file()
            else '<div class="no-image">No screen artifact</div>'
        )
        error = case.get("error") or case.get("source_error") or case.get("scoring_error") or ""
        parity_status = str(case["prompt_provenance"]["status"])
        cards.append(
            '<article class="case">'
            f'<header><h2>{html.escape(case["id"])}</h2>'
            f'<span class="status">{html.escape(case["status"])}</span></header>'
            f'<p>{html.escape(case["domain"] or "Unknown domain")}</p>'
            f'<p>Native prompt hash: <strong>{html.escape(parity_status)}</strong></p>'
            f'<p class="links">{" · ".join(link for link in links if link) or "No case artifacts"}</p>'
            f'{image}'
            f'<details><summary>Frozen source response</summary><pre>{source_html}</pre></details>'
            f'<p class="error">{html.escape(str(error))}</p>'
            '</article>'
        )
    audit_note = (
        artifact_link(model_audit, run_dir, "model_audit.json")
        if model_audit.is_file()
        else "model_audit.json is absent; no model-audit conclusion is assumed."
    )
    count_labels = {
        "selected": "Selected",
        "completed": "Completed",
        "returnedOutput": "Returned output",
        "androidStrict": "Android strict",
        "pythonStrict": "Python strict",
        "rendered": "Rendered valid",
        "runtimeErrors": "Runtime errors",
        "scored": "Python scored",
    }
    count_line = " · ".join(
        f"{html.escape(count_labels.get(key, key))}: {html.escape(str(value))}"
        for key, value in counts.items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trained E2B v10 W4 · Bixby50</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f4f6fa;color:#182030}}main{{max-width:1500px;margin:auto;padding:24px}}
.summary{{background:white;border:1px solid #dce2ec;border-radius:12px;padding:16px;margin-bottom:18px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}}
.case{{background:white;border:1px solid #dce2ec;border-radius:12px;padding:14px;min-width:0}}header{{display:flex;align-items:center;justify-content:space-between;gap:8px}}h1,h2{{margin:.2em 0}}.status{{font:600 12px ui-monospace,monospace;background:#e8edf5;padding:4px 7px;border-radius:99px}}
img{{width:100%;height:auto;max-height:620px;object-fit:contain;background:#111;border-radius:8px}}.no-image{{height:180px;display:grid;place-items:center;background:#edf0f5;border-radius:8px;color:#5d6878}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;max-height:360px;overflow:auto;background:#f7f8fb;padding:10px;border-radius:7px}}.links,.error{{overflow-wrap:anywhere}}.error{{color:#a22}}
</style></head><body><main><h1>Trained E2B v10 W4 · Bixby50</h1>
<section class="summary"><p>{count_line}</p>
<p><strong>Native rendered-prompt parity:</strong> {html.escape(prompt_parity_text(prompt_parity))}</p>
<p>{audit_note}</p>
<p>No reference IR is available, and this report does not independently verify factual claims. Source-fidelity metrics compare the generated artifact with the supplied frozen response.</p></section>
<section class="grid">{''.join(cards)}</section></main></body></html>
"""


def markdown_report(
    run_dir: Path,
    config: Mapping[str, Any],
    counts: Mapping[str, Any],
    runtime: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    fidelity: Mapping[str, Any],
    provenance: Mapping[str, Any],
    status_counts: Mapping[str, int],
    model_audit: Path,
) -> str:
    native = runtime["nativeDecodeTokensPerSecond"]
    tokens = runtime["tokens"]
    latency = runtime["latency"]
    prompt_parity = provenance.get("nativeRenderedPromptParity")
    if not isinstance(prompt_parity, Mapping):
        prompt_parity = {}
    model_audit_text = (
        "[model_audit.json](model_audit.json) is available for separate inspection."
        if model_audit.is_file()
        else "`model_audit.json` is absent; this report makes no model-audit assumption."
    )
    fidelity_rows = []
    for key, value in fidelity["metrics"].items():
        fidelity_rows.append(
            f"| `{key}` | {fmt_number(value['average'], 4)} | "
            f"{value['observedCount']}/{value['scoredCount']} | "
            f"{value['excludedAsUnavailableOrInapplicableCount']} |"
        )
    if not fidelity_rows:
        fidelity_rows.append(
            f"| _No source-fidelity metric available_ | unavailable | "
            f"0/{fidelity.get('scoredCount', 0)} | {fidelity.get('scoredCount', 0)} |"
        )
    status_text = ", ".join(f"`{key}`={value}" for key, value in sorted(status_counts.items())) or "none"
    device = config.get("device") if isinstance(config.get("device"), Mapping) else {}
    model = config.get("model") if isinstance(config.get("model"), Mapping) else {}
    prompt = config.get("prompt") if isinstance(config.get("prompt"), Mapping) else {}
    runtime_config = config.get("runtime") if isinstance(config.get("runtime"), Mapping) else {}
    return f"""# Trained E2B v10 W4 Bixby50 report

## Run coverage

| Measure | Count |
|---|---:|
| Selected | {counts['selected']} |
| Completed | {counts['completed']} |
| Returned output | {counts['returnedOutput']} |
| Android strict valid | {counts['androidStrict']} |
| Python strict valid | {counts['pythonStrict']} |
| Rendered valid | {counts['rendered']} |
| Runtime errors | {counts['runtimeErrors']} |
| Python-scored outputs | {counts['scored']} |

Statuses: {status_text}.

The Android and Python strict counts come from separate implementations and are reported independently. A schema-valid artifact is not labelled a quality pass. No reference IR exists for this source-only holdout, and neither the scorer nor this report independently verifies the factual truth of the supplied response or generated UI.

## Performance

| Measure | Value |
|---|---:|
| Native decode tokens/s median | {fmt_number(native['median'])} |
| Native decode tokens/s min | {fmt_number(native['min'])} |
| Native decode tokens/s max | {fmt_number(native['max'])} |
| Native decode tokens/s, token-weighted overall | {fmt_number(native['tokenWeightedOverall'])} |
| Input tokens total | {fmt_number(tokens['inputTotal'], 0)} |
| Output tokens total | {fmt_number(tokens['outputTotal'], 0)} |
| Outputs at or above 2048 tokens | {fmt_number(tokens['outputAtOrAbove2048Count'], 0)} |
| Cold first end-to-end elapsed | {fmt_number(latency['coldFirstEndToEndMs'], 0)} ms |
| Cold first provider call | {fmt_number(latency['coldFirstProviderCallMs'], 0)} ms |
| Warm median end-to-end elapsed | {fmt_number(latency['warmMedianEndToEndMs'], 0)} ms |
| Warm median provider call | {fmt_number(latency['warmMedianProviderCallMs'], 0)} ms |

Native decode throughput excludes model initialization and prompt prefill. End-to-end elapsed time includes the converter/provider request path; the first case is the cold call. Warm medians include non-first cases that returned `output.express`.

## Source-fidelity metrics

These averages use scored outputs only. Each displayed mean excludes null, absent, and inapplicable atomics, and the table shows its observed/scored denominator. Runtime failures and cases without `output.express` are unavailable and do not contribute fabricated zero rewards. The official `aggregate_scores` fields below retain their training-compatible denominator unchanged.

| Metric | Observed-only average | Observed/scored | N/A excluded |
|---|---:|---:|---:|
{chr(10).join(fidelity_rows)}

The v5.4 aggregate contains {aggregate.get('count', 0)} scored outputs. `generation_reward_v5_4` average: {fmt_number(aggregate.get('generation_reward_v5_4'))}; `render_artifact_quality_v5_4` average: {fmt_number(aggregate.get('render_artifact_quality_v5_4'))}.

## Configuration and provenance

- Run: `{run_dir.name}`
- Profile: `{config.get('profile', 'unavailable')}`
- Model: `{model.get('basename', 'unavailable')}` ({fmt_number(model.get('sizeBytes'), 0)} bytes)
- Device: `{device.get('manufacturer', 'unavailable')} {device.get('model', 'unavailable')}`, Android SDK `{device.get('sdkInt', 'unavailable')}`, hardware `{device.get('hardware', 'unavailable')}`
- Runtime: GPU=`{runtime_config.get('accelerator') == 'GPU'}`, context `{runtime_config.get('maxContextTokens', 'unavailable')}`, output cap `{runtime_config.get('maxOutputTokens', 'unavailable')}`, thinking `{runtime_config.get('thinkingEnabled', 'unavailable')}`, MTP `{runtime_config.get('mtpEnabled', 'unavailable')}`, native metrics `{runtime_config.get('metricsEnabled', 'unavailable')}`
- Prompt contract SHA-256: `{prompt.get('contractSha256', 'unavailable')}`
- Prompt asset SHA-256 recorded/current match: `{provenance.get('promptAssetHashMatches')}`
- Prompt contract SHA-256 recorded/current match: `{provenance.get('promptContractHashMatches')}`
- Android corpus SHA-256 recorded/current match: `{provenance.get('androidCorpusHashMatches')}`
- Frozen evaluation corpus SHA-256/manifest match: `{provenance.get('frozenCorpusHashMatchesManifest')}`
- Verified case sources: `{provenance.get('verifiedSources')}/{counts['completed']}` completed cases
- Native rendered-prompt SHA-256 parity: **{prompt_parity_text(prompt_parity)}**
- Rendered-prompt expected-hash method: `{prompt_parity.get('expectedHashMethod', 'unavailable')}`
- Missing result IDs: `{', '.join(provenance.get('missingResultIds', [])) or 'none'}`
- Duplicate result IDs: `{', '.join(provenance.get('duplicateResultIds', [])) or 'none'}`
- Unexpected result IDs: `{', '.join(provenance.get('unexpectedResultIds', [])) or 'none'}`
- {model_audit_text}

See [per_case.csv](per_case.csv), [scored_predictions.jsonl](scored_predictions.jsonl), [aggregate_metrics.json](aggregate_metrics.json), and [gallery.html](gallery.html). Visual-quality conclusions remain a manual review step.
"""


def build_report(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    config = read_json(run_dir / "run_config.json", {})
    if not isinstance(config, dict):
        raise TypeError("run_config.json must be an object")
    raw_results = read_json(run_dir / "results.json", [])
    if not isinstance(raw_results, list) or any(not isinstance(row, dict) for row in raw_results):
        raise ValueError("results.json must be an array of objects")
    frozen_rows = frozen_index()
    official_ids = [str(row["id"]) for row in frozen_rows]
    official_set = set(official_ids)

    result_ids = [str(row.get("id") or "") for row in raw_results]
    result_counts = Counter(result_ids)
    duplicate_result_ids = sorted(case_id for case_id, count in result_counts.items() if case_id and count > 1)
    unexpected_result_ids = sorted(case_id for case_id in result_counts if case_id not in official_set)
    results_by_id = {
        case_id: next(row for row in raw_results if row.get("id") == case_id)
        for case_id in official_ids
        if result_counts[case_id] == 1
    }
    missing_result_ids = [case_id for case_id in official_ids if result_counts[case_id] == 0]

    configured_cases = config.get("cases")
    selected_ids = [str(value) for value in configured_cases] if isinstance(configured_cases, list) else [
        case_id for case_id in official_ids if result_counts[case_id] > 0
    ]
    selected_counts = Counter(selected_ids)
    duplicate_selected_ids = sorted(case_id for case_id, count in selected_counts.items() if count > 1)
    unexpected_selected_ids = sorted(case_id for case_id in selected_counts if case_id not in official_set)
    selected_set = set(selected_ids) & official_set

    prompt_config = config.get("prompt") if isinstance(config.get("prompt"), Mapping) else {}
    corpus_config = config.get("corpus") if isinstance(config.get("corpus"), Mapping) else {}
    prompt_asset = run_dir / "shared_prompt.json"
    prompt_document = read_json(prompt_asset, {})
    manifest = read_json(CORPUS_MANIFEST, {})
    android_corpus_hash = sha256_file(ANDROID_CORPUS)
    frozen_corpus_hash = sha256_file(FROZEN_CORPUS)
    provenance_errors: list[str] = []
    prompt_hash_matches = bool(
        prompt_asset.is_file()
        and prompt_config.get("assetSha256")
        and prompt_config.get("assetSha256") == sha256_file(prompt_asset)
    )
    prompt_contract_actual = (
        prompt_document.get("contract_sha256") if isinstance(prompt_document, Mapping) else None
    )
    prompt_contract_matches = bool(
        prompt_contract_actual
        and prompt_config.get("contractSha256") == prompt_contract_actual
    )
    android_hash_matches = bool(
        android_corpus_hash and corpus_config.get("sha256") == android_corpus_hash
    )
    frozen_hash_matches = bool(
        isinstance(manifest, Mapping)
        and frozen_corpus_hash
        and manifest.get("output_sha256") == frozen_corpus_hash
    )
    if not prompt_hash_matches:
        provenance_errors.append("shared_prompt.json does not match run_config prompt asset SHA-256")
    if not prompt_contract_matches:
        provenance_errors.append("shared_prompt.json contract SHA-256 does not match run_config")
    if not android_hash_matches:
        provenance_errors.append("run_config corpus SHA-256 does not match the repository Android corpus snapshot")
    if not frozen_hash_matches:
        provenance_errors.append("frozen Bixby50 JSONL does not match benchmark_manifest.json")

    cases: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    prompt_audits: list[dict[str, Any]] = []
    verified_sources = 0
    for frozen in frozen_rows:
        case_id = str(frozen["id"])
        case_dir = run_dir / case_id
        result = results_by_id.get(case_id)
        selected = case_id in selected_set
        expected = expected_source(frozen)
        source_ok, source_error = verify_source(case_dir / "source.json", frozen)
        completed = result is not None
        if source_ok and completed:
            verified_sources += 1
        if completed and not source_ok:
            provenance_errors.append(f"{case_id}: {source_error}")
        raw_path = case_dir / "output.express"
        returned_output = raw_path.is_file()
        native = result_metrics(result)
        prompt_audit = audit_rendered_prompt(
            case_dir=case_dir,
            result=result,
            selected=selected,
            prompt_document=prompt_document if isinstance(prompt_document, Mapping) else {},
            response_text=expected["text"],
        )
        prompt_audits.append({"id": case_id, **prompt_audit})
        if prompt_audit["status"] == "mismatch":
            provenance_errors.append(
                f"{case_id}: rendered-prompt provenance mismatch: {prompt_audit['reason']}"
            )
        status = (
            "duplicate_result"
            if result_counts[case_id] > 1
            else str(result.get("status") or "unknown")
            if result is not None
            else "missing_result"
            if selected
            else "not_selected"
        )
        scoring_error: str | None = None
        python_metrics: dict[str, Any] | None = None
        generated_text: str | None = None
        if returned_output and result_counts[case_id] == 1:
            generated_text = raw_path.read_text(encoding="utf-8-sig")
            if source_ok:
                try:
                    python_metrics = score_prediction(
                        response_text=str(frozen.get("response_text") or ""),
                        expected=None,
                        generated_text=generated_text,
                        metric_version=METRIC_VERSION,
                    )
                    scored_row = {
                        "id": case_id,
                        "source_id": case_id,
                        "response_text": str(frozen.get("response_text") or ""),
                        "generated_text": generated_text,
                        "expected": None,
                        "reference_available": False,
                        "evaluation_only": True,
                        "selection_role": "final_only_holdout",
                        "benchmark": frozen.get("metadata", {}).get("benchmark"),
                        "android": result,
                        "promptProvenance": prompt_audit,
                        "metrics": python_metrics,
                    }
                    scored.append(scored_row)
                except Exception as exc:  # noqa: BLE001 - one scorer failure must not hide other cases
                    scoring_error = f"{type(exc).__name__}: {exc}"
            else:
                scoring_error = "not scored because source.json did not match the frozen source"

        case = {
            "id": case_id,
            "domain": expected["domain"],
            "source_text": expected["text"],
            "selected": selected,
            "completed": completed,
            "status": status,
            "returned_output": returned_output,
            "android_strict": bool(result and result.get("strictValid") is True),
            "python_strict": bool(python_metrics and python_metrics.get("schema_valid_strict") is True),
            "rendered": bool(result and result.get("renderValid") is True),
            "source_verified": source_ok,
            "source_error": source_error,
            "error": str(result.get("error") or "") if result else "",
            "scoring_error": scoring_error,
            "elapsed_ms": as_number(result.get("elapsedMs")) if result else None,
            "first_in_run": bool(result and result.get("firstInRun") is True),
            "provider_call_ms": as_number(native.get("providerCallMs")),
            "native_metrics": dict(native),
            "python_metrics": python_metrics,
            "prompt_provenance": prompt_audit,
        }
        cases.append(case)

    selected_prompt_audits = [row for row in prompt_audits if row["status"] != "not_selected"]
    native_prompt_hash_count = sum(
        row["nativeRenderedPromptSha256"] is not None for row in selected_prompt_audits
    )
    prompt_parity = {
        "expectedHashMethod": (
            "SHA-256 of UTF-8 BOS/turn serialization from shared_prompt.json plus the frozen "
            "response; reconstructed prompt.json is the fallback"
        ),
        "selectedCount": len(selected_prompt_audits),
        "expectedHashAvailableCount": sum(
            row["expectedRenderedPromptSha256"] is not None for row in selected_prompt_audits
        ),
        "nativeHashReportedCount": native_prompt_hash_count,
        "verifiedCount": (
            sum(row["status"] == "verified" for row in selected_prompt_audits)
            if native_prompt_hash_count
            else None
        ),
        "mismatchCount": sum(row["status"] == "mismatch" for row in selected_prompt_audits),
        "unavailableCount": sum(row["status"] == "unavailable" for row in selected_prompt_audits),
        "cases": prompt_audits,
    }
    aggregate = aggregate_scores(scored)
    fidelity = source_fidelity_averages(scored)
    runtime = runtime_summary(cases)
    status_counts = Counter(case["status"] for case in cases)
    counts = {
        "selected": len(selected_set),
        "completed": sum(case["completed"] for case in cases),
        "returnedOutput": sum(case["returned_output"] for case in cases),
        "androidStrict": sum(case["android_strict"] for case in cases),
        "pythonStrict": sum(case["python_strict"] for case in cases),
        "rendered": sum(case["rendered"] for case in cases),
        "runtimeErrors": sum(case["status"] == "runtime_error" for case in cases),
        "scored": len(scored),
    }
    provenance = {
        "promptAssetSha256Recorded": prompt_config.get("assetSha256"),
        "promptAssetSha256Actual": sha256_file(prompt_asset),
        "promptAssetHashMatches": prompt_hash_matches,
        "promptContractSha256": prompt_config.get("contractSha256"),
        "promptContractSha256Actual": prompt_contract_actual,
        "promptContractHashMatches": prompt_contract_matches,
        "androidCorpusSha256Recorded": corpus_config.get("sha256"),
        "androidCorpusSha256Actual": android_corpus_hash,
        "androidCorpusHashMatches": android_hash_matches,
        "frozenCorpusSha256": frozen_corpus_hash,
        "frozenCorpusManifestSha256": manifest.get("output_sha256") if isinstance(manifest, Mapping) else None,
        "frozenCorpusHashMatchesManifest": frozen_hash_matches,
        "verifiedSources": verified_sources,
        "missingResultIds": missing_result_ids,
        "duplicateResultIds": duplicate_result_ids,
        "unexpectedResultIds": unexpected_result_ids,
        "duplicateSelectedIds": duplicate_selected_ids,
        "unexpectedSelectedIds": unexpected_selected_ids,
        "nativeRenderedPromptParity": prompt_parity,
        "errors": provenance_errors,
    }
    aggregate.update(
        {
            "metric_version": METRIC_VERSION,
            "counts": counts,
            "statusCountsAll50": dict(sorted(status_counts.items())),
            "runtime": runtime,
            "runConfig": {
                "profile": config.get("profile"),
                "model": config.get("model"),
                "runtime": config.get("runtime"),
                "device": config.get("device"),
                "prompt": config.get("prompt"),
                "corpus": config.get("corpus"),
            },
            "sourceFidelityScoredOnly": fidelity,
            "provenance": provenance,
            "reference_available": False,
            "factual_verification_performed": False,
            "report_note": (
                "No reference IR. Runtime failures have unavailable scores and do not contribute zero rewards. "
                "Displayed source-fidelity means exclude null/inapplicable atomics; official aggregate_scores "
                "fields are unchanged. Visual conclusions require manual review."
            ),
        }
    )

    write_jsonl(run_dir / "scored_predictions.jsonl", scored)
    write_json(run_dir / "aggregate_metrics.json", aggregate)
    csv_fields = [
        "id", "domain", "selected", "completed", "status", "returned_output",
        "android_strict", "python_strict", "rendered", "source_verified", "elapsed_ms",
        "provider_call_ms", "input_tokens", "output_tokens", "decode_tokens_per_second",
        "output_at_2048_cap", "generation_reward_v5_4", "render_artifact_quality_v5_4",
        "content_coverage", "rendered_prompt_parity", "expected_rendered_prompt_sha256",
        "native_rendered_prompt_sha256", "prompt_json_matches_contract", "prompt_parity_reason",
        "screen", "screen_scrolled", "error", "source_error", "scoring_error",
    ]
    csv_path = run_dir / "per_case.csv"
    csv_temp = csv_path.with_name(f".{csv_path.name}.tmp")
    with csv_temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for case in cases:
            native = case["native_metrics"]
            metrics = case["python_metrics"] or {}
            output_tokens = int_or_none(native.get("outputTokens"))
            prompt_audit = case["prompt_provenance"]
            case_dir = run_dir / case["id"]
            writer.writerow(
                {
                    "id": case["id"],
                    "domain": case["domain"],
                    "selected": case["selected"],
                    "completed": case["completed"],
                    "status": case["status"],
                    "returned_output": case["returned_output"],
                    "android_strict": case["android_strict"],
                    "python_strict": case["python_strict"],
                    "rendered": case["rendered"],
                    "source_verified": case["source_verified"],
                    "elapsed_ms": case["elapsed_ms"],
                    "provider_call_ms": case["provider_call_ms"],
                    "input_tokens": int_or_none(native.get("inputTokens")),
                    "output_tokens": output_tokens,
                    "decode_tokens_per_second": as_number(native.get("decodeTokensPerSecond")),
                    "output_at_2048_cap": (
                        output_tokens >= 2_048 if output_tokens is not None else None
                    ),
                    "generation_reward_v5_4": metrics.get("generation_reward_v5_4"),
                    "render_artifact_quality_v5_4": metrics.get("render_artifact_quality_v5_4"),
                    "content_coverage": metrics.get("content_coverage"),
                    "rendered_prompt_parity": prompt_audit["status"],
                    "expected_rendered_prompt_sha256": prompt_audit[
                        "expectedRenderedPromptSha256"
                    ],
                    "native_rendered_prompt_sha256": prompt_audit["nativeRenderedPromptSha256"],
                    "prompt_json_matches_contract": prompt_audit["promptJsonMatchesContract"],
                    "prompt_parity_reason": prompt_audit["reason"],
                    "screen": (case_dir / "screen.png").is_file(),
                    "screen_scrolled": (case_dir / "screen_scrolled.png").is_file(),
                    "error": case["error"],
                    "source_error": case["source_error"],
                    "scoring_error": case["scoring_error"],
                }
            )
    csv_temp.replace(csv_path)

    model_audit = run_dir / "model_audit.json"
    atomic_text(
        run_dir / "gallery.html",
        gallery_html(run_dir, cases, counts, prompt_parity, model_audit),
    )
    atomic_text(
        run_dir / "REPORT.md",
        markdown_report(
            run_dir,
            config,
            counts,
            runtime,
            aggregate,
            fidelity,
            provenance,
            status_counts,
            model_audit,
        ),
    )
    require_50_errors = []
    if set(selected_ids) != official_set or len(selected_ids) != 50:
        require_50_errors.append("run_config does not select each frozen Bixby50 ID exactly once")
    if missing_result_ids:
        require_50_errors.append(f"missing result IDs: {', '.join(missing_result_ids)}")
    if duplicate_result_ids:
        require_50_errors.append(f"duplicate result IDs: {', '.join(duplicate_result_ids)}")
    if unexpected_result_ids:
        require_50_errors.append(f"unexpected result IDs: {', '.join(unexpected_result_ids)}")
    if duplicate_selected_ids:
        require_50_errors.append(f"duplicate selected IDs: {', '.join(duplicate_selected_ids)}")
    if unexpected_selected_ids:
        require_50_errors.append(f"unexpected selected IDs: {', '.join(unexpected_selected_ids)}")
    return {
        "run": str(run_dir),
        "counts": counts,
        "statusCountsAll50": dict(sorted(status_counts.items())),
        "outputs": [
            "scored_predictions.jsonl",
            "aggregate_metrics.json",
            "REPORT.md",
            "per_case.csv",
            "gallery.html",
        ],
        "provenanceErrors": provenance_errors,
        "require50Errors": require_50_errors,
    }, provenance_errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score and report a pulled trained-E2B Bixby50 Android benchmark run.",
    )
    parser.add_argument("run_dir", type=Path, help="Pulled externalFiles/sdk_benchmark/<runId> directory")
    parser.add_argument(
        "--require-50",
        action="store_true",
        help="Exit nonzero for missing/duplicate/unexpected IDs after still writing every report artifact.",
    )
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error(f"run directory does not exist: {run_dir}")
    try:
        summary, provenance_errors = build_report(run_dir)
    except Exception as exc:  # noqa: BLE001 - CLI reports a concise diagnostic and nonzero status
        print(f"report_trained_bixby50.py: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failures = list(provenance_errors)
    if args.require_50:
        failures.extend(summary["require50Errors"])
    if failures:
        print("Report emitted with validation errors:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
