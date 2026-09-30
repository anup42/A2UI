"""Small, provider-neutral generation diagnostics; never executes a model."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from pipeline.renderer_semantics import iter_renderer_references
from llm.base import completion_metadata


def finish_reason(payload: Any) -> str | None:
    if isinstance(payload, Mapping):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            value = choices[0].get("finish_reason")
            return str(value) if value is not None else None
        candidates = payload.get("candidates")
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], Mapping):
            value = candidates[0].get("finishReason")
            return str(value) if value is not None else None
    return None


def incomplete_reason(payload: Any) -> str | None:
    reason = finish_reason(payload)
    if reason and reason.lower() in {"length", "max_tokens", "content_filter", "safety", "recitation"}:
        return f"incomplete_completion: finish_reason={reason}"
    return None


def compact_exception(exc: Exception, limit: int = 1200) -> str:
    """Avoid expanding an entire wire payload/schema in repair instructions."""
    pending = [exc]
    leaves: list[str] = []
    while pending and len(leaves) < 64:
        error = pending.pop(0)
        context = getattr(error, "context", None)
        if context:
            # Applicable leaf type/property errors are more useful than oneOf dumps.
            pending.extend(context)
            continue
        path = "/".join(str(part) for part in getattr(error, "absolute_path", ()))
        if getattr(error, "validator", None) == "const" and path.endswith("/component"):
            continue
        message = getattr(error, "message", None) or str(error)
        item = f"{path}: {message}" if path else message
        item = item[:300]
        if item not in leaves:
            leaves.append(item)
    leaves.sort(key=lambda item: -item.split(":", 1)[0].count("/"))
    return ("; ".join(leaves[:8]) or type(exc).__name__)[:limit]


def graph_acceptance_errors(graph: Mapping[str, Any]) -> list[str]:
    """Renderer-reference reachability and role completeness, without a size quota."""
    elements = graph.get("elements", {})
    seen: set[str] = set()
    pending = [graph.get("root")]
    while pending:
        element_id = pending.pop()
        if element_id in seen or element_id not in elements:
            continue
        seen.add(element_id)
        pending.extend(ref.target_id for ref in iter_renderer_references(elements[element_id]))
    disconnected = sorted(set(elements) - seen)
    errors = []
    if disconnected:
        errors.append("unreachable_components: " + ", ".join(disconnected[:30]))
    content_found = False
    for element_id in seen:
        element = elements[element_id]
        kind = str(element.get("type", "")).lower()
        props = element.get("props", {})
        if kind == "emailpreview":
            body = props.get("body")
            if not body or (isinstance(body, str) and not body.strip()) or (isinstance(body, list) and not any(str(value).strip() for value in body)):
                errors.append(f"incomplete_role: EmailPreview {element_id!r} requires body")
            else:
                content_found = True
        elif kind not in {"stack", "row", "column", "card", "list", "divider", "tabs", "modal"}:
            content_found = content_found or any(value not in (None, "", [], {}) for value in props.values())
        elif kind == "list" and props.get("items"):
            content_found = True
    if not content_found:
        errors.append("empty_ui_content: no reachable content or input component")
    return errors


def record_attempt(task: dict[str, Any], artifacts_dir: Path, *, phase: str,
                   prompt: str, system: str | None, seed: int, temperature: float,
                   max_tokens: int, text: str = "", raw: Any = None,
                   input_tokens: int = 0, output_tokens: int = 0,
                   latency_ms: float = 0.0, error: str | None = None,
                   reasoning_tokens: int | None = None,
                   reasoning_text: str | None = None,
                   reasoning_source: str | None = None,
                   cost_usd: float | None = None) -> dict[str, Any]:
    """Persist each provider attempt before parsing; retries never overwrite it."""
    attempts = task.setdefault("generation_attempts", [])
    invocation = task.setdefault("generation_invocation_id", uuid4().hex)
    attempt = {
        "attempt_index": len(attempts), "phase": phase,
        "phase_invocation_id": task.get("phase_invocation_id"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_hash": hashlib.sha256(task["response_text"].encode("utf-8")).hexdigest(),
        "prompt_hash": hashlib.sha256(((system or "") + "\n" + prompt).encode("utf-8")).hexdigest(),
        "completion_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "seed": seed, "temperature": temperature, "max_tokens": max_tokens,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens, "latency_ms": latency_ms, "cost_usd": cost_usd,
        "finish_reason": finish_reason(raw), "error": error,
        "completion_complete": False if error else completion_metadata(raw)[1],
    }
    if isinstance(reasoning_text, str) and reasoning_text.strip():
        attempt["reasoning_available"] = True
        attempt["reasoning_source"] = reasoning_source
    if isinstance(raw, Mapping) and isinstance(raw.get("a2ui_request_attempts"), list):
        attempt["http_request_attempts"] = raw["a2ui_request_attempts"]
    directory = artifacts_dir / "generation_attempts" / f"{task['ui_id']}_{invocation}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{len(attempts):03d}_{phase}.json"
    payload = {**attempt, "prompt": prompt, "system": system, "completion": text}
    if isinstance(reasoning_text, str) and reasoning_text.strip():
        payload["reasoning_text"] = reasoning_text
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    attempt["artifact"] = str(path.relative_to(artifacts_dir))
    attempts.append(attempt)
    return attempt


def attempt_summary(task: Mapping[str, Any]) -> dict[str, Any]:
    attempts = task.get("generation_attempts", [])
    return {
        "invocation_id": task.get("generation_invocation_id"),
        "attempt_count": len(attempts),
        "provider_call_count": sum(item["phase"] != "cache" for item in attempts),
        "input_tokens": sum(item["input_tokens"] for item in attempts),
        "output_tokens": sum(item["output_tokens"] for item in attempts),
        "latency_ms": sum(item["latency_ms"] for item in attempts),
        "reported_cost_usd": sum(item.get("cost_usd") or 0 for item in attempts) if any(item.get("cost_usd") is not None for item in attempts) else None,
        "cost_reporting_complete": bool(attempts) and all(item.get("cost_usd") is not None for item in attempts),
        "scope": "provider adapter calls; local HTTP retries are recorded when exposed; server-internal retries remain unknown",
    }
