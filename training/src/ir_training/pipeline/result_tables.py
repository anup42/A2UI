"""Small human-readable end-of-run views over the existing evidence manifests.

These helpers never score predictions, select winners or infer a zero score
from absent evidence. JSON manifests and per-case artifacts remain authoritative.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ir_training.common.progress import log

DEPLOYMENT_LABELS = ("checkpoint_best", "checkpoint_final", "merged", "w32", "w16", "w8", "w4")
LITERT_LABELS = ("w32", "w16", "w8", "w4")
REWARD_METRIC = "generation_reward_v5_4_avg"
UNIQUE_REWARD_METRIC = "unique_source_generation_reward_v5_4_avg"
STRICT_METRIC = "schema_valid_strict_rate"
COHORT_LABELS = {"golden32": "Golden32", "golden35": "Golden35", "bixby50": "Bixby50"}


def _cell(value: Any) -> str:
    # Names/error messages are manifest data, not Markdown or terminal controls.
    return " ".join(re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", str(value)).replace("|", "/").split())


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _cohorts(state: Mapping[str, Any], *, deployment: bool = False) -> list[tuple[str, int]]:
    """Follow a recorded plan; old two-cohort manifests stay two-cohort reports."""
    plan = _mapping(state.get("plan"))
    training = _mapping(plan.get("training")) if deployment else plan
    configured = _mapping(training.get("goldens"))
    if not configured and not deployment:
        for trial in plan.get("trials", []) or []:
            configured = _mapping(_mapping(_mapping(trial).get("plan")).get("goldens"))
            if configured:
                break
    if configured:
        cohorts = [(name, spec["rows"]) for name, spec in configured.items()
                   if isinstance(spec, Mapping) and type(spec.get("rows")) is int and spec["rows"] > 0]
        if cohorts:
            return cohorts
    result = [("golden32", 32), ("golden35", 35)]
    evidence = _mapping(state.get("results"))
    if ("selected_bixby50" in state or plan.get("bixby50_policy")
            or any(str(name).endswith("_bixby50") for name in evidence)
            or str(state.get("active_stage", "")).endswith("_bixby50")):
        result.append(("bixby50", 50))
    return result


def _table(headers: tuple[str, ...], rows: list[list[str]]) -> str:
    values = [[_cell(cell) for cell in row] for row in rows]
    widths = [max(len(header), *(len(row[index]) for row in values)) for index, header in enumerate(headers)]
    def line(row):
        return "| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |"
    return "\n".join([line(headers), line(["-" * width for width in widths]), *(line(row) for row in values)])


def _number(value: Any, *, percent: bool = False, compact: bool = False) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "unknown"
    if percent:
        return f"{100 * value:.1f}" if 0 <= value <= 1 else "unknown"
    return f"{value:.4g}" if compact else f"{value:.4f}"


def _metric(result: Mapping[str, Any], key: str, *, percent: bool = False) -> str:
    aggregate = result.get("aggregate")
    return _number(aggregate.get(key) if isinstance(aggregate, Mapping) else None, percent=percent)


def _rows(result: Mapping[str, Any], expected: int) -> str:
    count = result.get("row_count")
    return f"{count}/{expected}" if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else f"unknown/{expected}"


def _result_status(result: Mapping[str, Any], expected: int) -> str:
    status = result.get("status")
    if status in {"failed", "incomplete", "not_run", "running"}:
        return str(status).replace("_", " ")
    if result.get("row_count") != expected:
        return "incomplete"
    return "evaluated"


def render_experiment_results(state: Mapping[str, Any]) -> str:
    """Render every planned trial, including failed and unattempted trials."""
    plan = _mapping(state.get("plan"))
    trials = {item["name"]: item for item in state.get("trials", []) or [] if isinstance(item, Mapping) and "name" in item}
    planned = plan.get("trials", []) or list(trials.values())
    selected = state.get("selected_trial")
    holdouts = [(name, count) for name, count in _cohorts(state) if name != "golden32"]
    rows = []
    for spec in planned:
        if not isinstance(spec, Mapping) or "name" not in spec:
            continue
        name = spec["name"]
        result = trials.get(name, {})
        status = result.get("status", "not reported")
        if not result:
            if name == state.get("active_trial"):
                status = "failed" if state.get("status") == "failed" else "running"
            elif state.get("status") != "complete":
                status = "not run"
        parameters = _mapping(result.get("parameters", spec.get("parameters")))
        evaluations = _mapping(result.get("evaluations"))
        best, final = _mapping(evaluations.get("best_golden32")), _mapping(evaluations.get("final_golden32"))
        rows.append([
            name + (" [selected]" if name == selected else ""), status,
            "/".join(_number(parameters.get(key), compact=True) for key in ("learning_rate", "weight_decay", "warmup_ratio")),
            str(parameters.get("augmentation", "unknown")),
            _metric(best, UNIQUE_REWARD_METRIC) + " / " + _metric(final, UNIQUE_REWARD_METRIC),
            _metric(best, STRICT_METRIC, percent=True) + " / " + _metric(final, STRICT_METRIC, percent=True),
            _rows(best, 32) + " / " + _rows(final, 32),
            _number(result.get("elapsed_seconds"), compact=True),
        ])
    lines = ["# Hyperparameter tuning results", "", f"Run status: {_cell(state.get('status', 'unknown'))}", "",
             ("Golden32: 32 occurrences / 31 unique sources. B/F = best/final checkpoint. "
             "Reward v5.4 is shown on the 0-100 scale, averaged over 31 unique sources; "
             "strict-valid percentages and row counts use all 32 occurrences."), "",
             "Selection uses only the best-checkpoint unique-source Golden32 reward. "
             + " ".join(f"{COHORT_LABELS.get(name, name)} never selects a trial." for name, _ in holdouts), ""]
    if rows:
        lines.append(_table(("Trial", "Status", "LR/WD/warmup", "Augmentation", "Reward v5.4 B/F (31)",
                             "Strict-valid B/F % (32)", "Rows B/F", "Elapsed s"), rows))
    else:
        lines.append("No trial results were reported.")
    lines.extend(["", f"Locked winner: {_cell(selected) if selected else 'not selected'}."])
    evaluated_holdouts = [(name, count, state.get(f"selected_{name}")) for name, count in holdouts
                          if isinstance(state.get(f"selected_{name}"), Mapping)]
    if evaluated_holdouts:
        lines.extend(["", "Selected checkpoint holdout (not used for selection):", "",
                      _table(("Cohort", "Status", "Rows", "Reward v5.4 (0-100)", "Strict-valid %"),
                             [[COHORT_LABELS.get(name, name), _result_status(holdout, count), _rows(holdout, count),
                               _metric(holdout, REWARD_METRIC), _metric(holdout, STRICT_METRIC, percent=True)]
                              for name, count, holdout in evaluated_holdouts])])
    for name, _ in holdouts:
        if isinstance(state.get(f"selected_{name}"), Mapping):
            continue
        label = COHORT_LABELS.get(name, name)
        if plan.get("evaluate_selected_holdout") is False:
            lines.extend(["", f"{label}: intentionally deferred until the fresh full-training run; not evaluated during tuning."])
        elif selected:
            lines.extend(["", f"{label}: no verified result reported (failed or not yet evaluated); the winner remains locked."])
    if any(name == "bixby50" for name, _ in holdouts):
        lines.extend(["", "Bixby50 uses source-response scoring; no reference IR."])
    if state.get("error"):
        lines.extend(["", f"Failure: {_cell(state['error'])}"])
    lines.extend(["", "Unknown means missing/nonfinite evidence, not zero. Full metrics and artifacts remain in experiments_manifest.json and comparison.json (when complete)."])
    return "\n".join(lines) + "\n"


def render_deployment_results(state: Mapping[str, Any], *, tuning_state: Mapping[str, Any] | None = None) -> str:
    """Render all planned evaluation slots without filling missing scores."""
    results = _mapping(state.get("results"))
    active = state.get("active_stage")
    failed = state.get("status") == "failed"
    skip_litert_evaluation = _mapping(state.get("plan")).get("skip_litert_evaluation") is True
    cohorts = _cohorts(state, deployment=True)
    rows = []
    for label in DEPLOYMENT_LABELS:
        for cohort, count in cohorts:
            key = f"{label}_{cohort}"
            if skip_litert_evaluation and label in LITERT_LABELS:
                # An intentional omission is neither missing evidence nor a
                # successful evaluation. Never surface stale runtime scores
                # under a plan that explicitly disables native inference.
                rows.append([label, COHORT_LABELS.get(cohort, cohort), "skipped by request",
                             "n/a", "n/a", "n/a", "n/a"])
                continue
            result = _mapping(results.get(key))
            if result:
                status = _result_status(result, count)
            elif key == active:
                status = "failed" if failed else "running"
            elif active == f"export_{label}" and failed:
                status = "blocked: export failed"
            else:
                # A subprocess can fail after inference but before validated
                # evidence reaches the parent; do not claim it never ran.
                status = "not reported"
            rows.append([label, COHORT_LABELS.get(cohort, cohort), status,
                         _rows(result, count), _metric(result, REWARD_METRIC),
                         _metric(result, STRICT_METRIC, percent=True),
                         _metric(result, UNIQUE_REWARD_METRIC) if cohort == "golden32" else "n/a"])
    lines = ["# Deployment results", "", f"Run status: {_cell(state.get('status', 'unknown'))}", "",
             "Golden32 has 32 occurrences / 31 unique sources. Golden35 has 35 unique sources and is not used for tuning/checkpoint selection.", "",
             ("Reward v5.4 (0-100) and strict-valid % use all evaluated occurrences. "
             "Golden32 unique reward gives each of its 31 sources equal weight (selection metric)."), "",
             _table(("Model", "Cohort", "Status", "Rows", "Reward v5.4", "Strict-valid %", "G32 unique reward"), rows), "",
             ("Evaluated means inference/scoring completed, not a quality-threshold pass. Unknown means missing/nonfinite evidence, not zero. "
             "Not reported means no verified result is available; consult the manifest/logs for attempted stages.")]
    if skip_litert_evaluation:
        lines[4:4] = ["Mode: checkpoint testing plus export-only LiteRT variants (--skip-litert-evaluation). "
                      "Native runtime not validated: Vulkan preflight and LiteRT-LM inference/scoring were skipped by request. "
                      "Artifact validation does not establish runtime compatibility or model quality.", ""]
        exports = _mapping(state.get("exports"))
        export_rows = []
        for label in LITERT_LABELS:
            if _mapping(exports.get(label)):
                status = "exported; artifact validated"
            elif active == f"export_{label}":
                status = "failed" if failed else "running"
            else:
                status = "not reported"
            export_rows.append([label.upper(), status])
        lines.extend(["", "LiteRT export artifacts (separate from runtime evaluation):", "",
                      _table(("Variant", "Export status"), export_rows)])
    if any(name == "bixby50" for name, _ in cohorts):
        lines.extend(["", "Bixby50 is a held-out test cohort, never used for tuning/checkpoint selection. "
                      "Bixby50 uses source-response scoring; no reference IR."])
    if active and failed:
        lines.extend(["", f"Stopped at: {_cell(active)}."])
    if state.get("error"):
        lines.extend(["", f"Failure: {_cell(state['error'])}"])
    if tuning_state is not None:
        lines.extend(["", render_experiment_results(tuning_state).rstrip()])
    return "\n".join(lines) + "\n"


def _publish(text: str, path: Path) -> Path | None:
    # Reporting must not obscure the actual training/runtime exception. Console
    # output is still useful if the output volume became unwritable or full.
    written = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
        written = path
    except OSError as exc:
        log(f"Could not persist final result table at {path}: {exc}")
    print("\n" + text, flush=True)
    if written:
        log(f"Result table: {path}")
    return written


def publish_experiment_results(state: Mapping[str, Any], *, path: str | Path | None = None) -> Path | None:
    destination = Path(path) if path is not None else Path(state["plan"]["output_dir"]) / "comparison.md"
    return _publish(render_experiment_results(state), destination)


def publish_deployment_results(state: Mapping[str, Any], *, tuning_state: Mapping[str, Any] | None = None,
                               path: str | Path | None = None) -> Path | None:
    destination = Path(path) if path is not None else Path(state["plan"]["output_dir"]) / "deployment_results.md"
    return _publish(render_deployment_results(state, tuning_state=tuning_state), destination)
