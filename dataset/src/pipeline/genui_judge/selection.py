"""Deterministic population/stress selection for the single-Codex benchmark."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pipeline.genui_quality.identity_v5_4 import expected_contract_hash_v5_4
from pipeline.genui_quality.source_contract_v5_4 import (
    resolve_expected_ui_contract_v5_4,
)

from .protocol import (
    judge_instructions_path,
    judge_instructions_sha256,
    protocol_fingerprint,
    protocol_mapping,
)


DEFAULT_SELECTION_SEED = "a2ui-single-codex-judge-960-v2"
TARGET_INTENT_COUNT = 32
POPULATION_PER_INTENT = 20
STRESS_PER_INTENT = 10
CALIBRATION_PER_INTENT = 20
VALIDATION_PER_INTENT = 5
HOLDOUT_PER_INTENT = 5
BLIND_REPEAT_COUNT = 96


@dataclass(frozen=True)
class CandidateSummary:
    line_index: int
    ui_id: str
    query_id: str
    response_id: str
    intent_bucket: str
    record_sha256: str
    response_text_sha256: str
    source_length: int
    component_count: int
    max_table_columns: int
    reachable_types: tuple[str, ...]
    legacy_score: float | None
    v5_4_score: float | None
    active_caps: tuple[str, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _order_key(seed: str, *parts: Any) -> str:
    return _hash_text("|".join([seed, *(str(part) for part in parts)]))


def _as_finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _legacy_score(record: Mapping[str, Any]) -> float | None:
    for candidate in (
        record.get("legacy_overall_score"),
        (record.get("metrics") or {}).get("overall_score")
        if isinstance(record.get("metrics"), Mapping)
        else None,
    ):
        value = _as_finite_float(candidate)
        if value is not None:
            return value
    return None


def _active_cap_names(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    names: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            name = (
                item.get("name")
                or item.get("code")
                or item.get("reason")
                or item.get("kind")
            )
        else:
            name = item
        cleaned = str(name or "").strip()
        if cleaned:
            names.append(cleaned)
    return tuple(sorted(set(names)))


def _load_metric_scores(path: Path | None) -> dict[str, tuple[float, tuple[str, ...]]]:
    if path is None:
        return {}
    if path.is_dir():
        path = path / "scores.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    output: dict[str, tuple[float, tuple[str, ...]]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number} is not an object")
            ui_id = str(row.get("ui_id") or "").strip()
            breakdown = row.get("render_artifact_quality_v5_4")
            if not ui_id or not isinstance(breakdown, Mapping):
                continue
            score = _as_finite_float(
                breakdown.get("artifact_quality_0_100")
                if breakdown.get("artifact_quality_0_100") is not None
                else breakdown.get("quality_0_100")
            )
            if score is None:
                continue
            caps = _active_cap_names(
                row.get("active_caps")
                if row.get("active_caps") is not None
                else breakdown.get("active_caps")
            )
            output[ui_id] = (score, caps)
    return output


def _element_features(spec: Any) -> tuple[int, int, tuple[str, ...]]:
    if not isinstance(spec, Mapping):
        return 0, 0, ()
    elements = spec.get("elements")
    if not isinstance(elements, Mapping):
        return 0, 0, ()
    types: list[str] = []
    max_columns = 0
    for element in elements.values():
        if not isinstance(element, Mapping):
            continue
        element_type = str(element.get("type") or "").strip()
        if element_type:
            types.append(element_type)
        if element_type.casefold() == "table":
            props = element.get("props")
            if isinstance(props, Mapping):
                columns = props.get("columns")
                if isinstance(columns, Sequence) and not isinstance(
                    columns, (str, bytes)
                ):
                    max_columns = max(max_columns, len(columns))
    return len(elements), max_columns, tuple(sorted(set(types)))


def _iter_source_summaries(
    path: Path,
    *,
    metric_scores: Mapping[str, tuple[float, tuple[str, ...]]],
) -> Iterable[CandidateSummary]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_index, raw_line in enumerate(handle):
            stripped = raw_line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, Mapping):
                raise ValueError(f"{path}:{line_index + 1} is not an object")
            ui_id = str(record.get("ui_id") or "").strip()
            query_id = str(record.get("query_id") or "").strip()
            response_id = str(record.get("response_id") or "").strip()
            intent = str(
                record.get("intent_bucket") or record.get("intent") or ""
            ).strip()
            response_text = str(record.get("response_text") or "")
            if not ui_id or not query_id or not response_id or not intent:
                raise ValueError(
                    f"{path}:{line_index + 1} lacks benchmark identity fields"
                )
            component_count, max_columns, types = _element_features(
                record.get("genui_json")
            )
            metric = metric_scores.get(ui_id)
            yield CandidateSummary(
                line_index=line_index,
                ui_id=ui_id,
                query_id=query_id,
                response_id=response_id,
                intent_bucket=intent,
                record_sha256=_hash_text(stripped),
                response_text_sha256=_hash_text(response_text),
                source_length=len(response_text),
                component_count=component_count,
                max_table_columns=max_columns,
                reachable_types=types,
                legacy_score=_legacy_score(record),
                v5_4_score=metric[0] if metric else None,
                active_caps=metric[1] if metric else (),
            )


def _load_prior_anchor_ids(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    if path.is_dir():
        candidate = path / "selection_manifest.jsonl"
        path = candidate if candidate.exists() else path / "llm_groundtruth.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    output: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                continue
            intent = str(row.get("intent_bucket") or "").strip()
            ui_id = str(row.get("ui_id") or "").strip()
            if intent and ui_id:
                output[intent].append(ui_id)
    return {key: list(dict.fromkeys(value)) for key, value in output.items()}


def _score_band(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 65.0:
        return "below_65"
    if value < 75.0:
        return "65_74"
    if value < 85.0:
        return "75_84"
    if value < 92.0:
        return "85_91"
    if value < 96.0:
        return "92_95"
    return "96_100"


def _length_band(
    value: int,
    ordered_lengths: Sequence[int],
) -> str:
    if not ordered_lengths:
        return "source_length_unknown"
    rank = sum(1 for item in ordered_lengths if item < value)
    fraction = rank / max(1, len(ordered_lengths) - 1)
    if fraction <= 0.25:
        return "source_length_q1"
    if fraction >= 0.75:
        return "source_length_q4"
    return "source_length_mid"


def _stress_features(
    candidate: CandidateSummary,
    *,
    ordered_lengths: Sequence[int],
) -> set[str]:
    score = (
        candidate.v5_4_score
        if candidate.v5_4_score is not None
        else candidate.legacy_score
    )
    features = {
        f"score_band:{_score_band(score)}",
        _length_band(candidate.source_length, ordered_lengths),
    }
    for cap in candidate.active_caps:
        features.add(f"cap:{cap}")
    type_names = {value.casefold() for value in candidate.reachable_types}
    for name in (
        "table",
        "chart",
        "formula",
        "tabs",
        "modal",
        "image",
        "video",
        "audio",
        "emailpreview",
        "codeblock",
        "consolelog",
        "textfield",
        "checkbox",
        "slider",
        "choicepicker",
        "datetimeinput",
    ):
        if name in type_names:
            features.add(f"type:{name}")
    if candidate.max_table_columns >= 4:
        features.add("render_risk:wide_table")
    if candidate.max_table_columns >= 6:
        features.add("render_risk:very_wide_table")
    if candidate.component_count >= 40:
        features.add("complexity:large")
    elif candidate.component_count <= 12:
        features.add("complexity:small")
    else:
        features.add("complexity:medium")
    if (
        candidate.v5_4_score is not None
        and candidate.legacy_score is not None
        and abs(candidate.v5_4_score - candidate.legacy_score) >= 15.0
    ):
        features.add("metric_disagreement:15_plus")
    return features


def _choose_anchor(
    candidates: Sequence[CandidateSummary],
    prior_ids: Sequence[str],
    *,
    seed: str,
) -> CandidateSummary:
    by_id = {candidate.ui_id: candidate for candidate in candidates}
    available = [by_id[ui_id] for ui_id in prior_ids if ui_id in by_id]
    pool = available or list(candidates)
    return min(
        pool,
        key=lambda item: _order_key(seed, "anchor", item.ui_id),
    )


def _choose_stress(
    candidates: Sequence[CandidateSummary],
    *,
    anchor: CandidateSummary,
    excluded_ids: set[str],
    seed: str,
) -> list[CandidateSummary]:
    lengths = sorted(item.source_length for item in candidates)
    pool = [
        item
        for item in candidates
        if item.ui_id not in excluded_ids and item.ui_id != anchor.ui_id
    ]
    feature_map = {
        item.ui_id: _stress_features(item, ordered_lengths=lengths)
        for item in pool
    }
    frequency = Counter(
        feature
        for features in feature_map.values()
        for feature in features
    )
    covered = _stress_features(anchor, ordered_lengths=lengths)
    selected = [anchor]
    while len(selected) < STRESS_PER_INTENT:
        if not pool:
            raise ValueError(
                f"not enough stress candidates for {anchor.intent_bucket}"
            )

        def score(item: CandidateSummary) -> tuple[float, int]:
            features = feature_map[item.ui_id]
            novelty = sum(
                1.0 / math.sqrt(max(1, frequency[feature]))
                for feature in features
                if feature not in covered
            )
            return (
                novelty,
                -int(
                    _order_key(
                        seed, "stress", item.intent_bucket, item.ui_id
                    )[:16],
                    16,
                ),
            )

        chosen = max(pool, key=score)
        selected.append(chosen)
        covered.update(feature_map[chosen.ui_id])
        pool.remove(chosen)
    return selected


def _assign_split(
    candidates: Sequence[CandidateSummary],
    *,
    calibration_count: int,
    validation_count: int,
    seed: str,
    stratum: str,
) -> dict[str, str]:
    ordered = sorted(
        candidates,
        key=lambda item: _order_key(
            seed, "split", stratum, item.intent_bucket, item.query_id
        ),
    )
    output: dict[str, str] = {}
    for index, item in enumerate(ordered):
        if index < calibration_count:
            split = "calibration"
        elif index < calibration_count + validation_count:
            split = "validation"
        else:
            split = "holdout"
        output[item.ui_id] = split
    return output


def _packet_id(seed: str, ui_id: str, occurrence: int) -> str:
    return "p_" + _order_key(
        seed, "opaque_packet", ui_id, occurrence
    )[:24]


def _build_schedule(
    selected: Sequence[dict[str, Any]],
    *,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    anchor_rows = [row for row in selected if row["migration_anchor"]]
    if len(anchor_rows) != TARGET_INTENT_COUNT:
        raise AssertionError("pilot must contain one migration anchor per intent")
    anchors = sorted(
        anchor_rows,
        key=lambda row: _order_key(seed, "pilot", row["intent_bucket"]),
    )
    anchor_ids = {row["ui_id"] for row in anchors}
    remaining = sorted(
        [row for row in selected if row["ui_id"] not in anchor_ids],
        key=lambda row: _order_key(seed, "original_order", row["ui_id"]),
    )
    originals = anchors + remaining
    original_position = {
        row["ui_id"]: index for index, row in enumerate(originals)
    }
    eligible_random = [
        row
        for row in originals
        if row["ui_id"] not in anchor_ids
        and original_position[row["ui_id"]] <= len(originals) - 201
    ]
    random_repeats = sorted(
        eligible_random,
        key=lambda row: _order_key(seed, "random_repeat", row["ui_id"]),
    )[: BLIND_REPEAT_COUNT - TARGET_INTENT_COUNT]
    repeat_by_id = {row["ui_id"]: row for row in random_repeats}

    schedule: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    scheduled_repeats: set[str] = set()
    original_schedule_position: dict[str, int] = {}

    def append(row: Mapping[str, Any], occurrence: int) -> None:
        packet_id = _packet_id(seed, str(row["ui_id"]), occurrence)
        position = len(schedule)
        block_index = (
            0
            if position < 20
            else 1
            if position < TARGET_INTENT_COUNT
            else 2 + (position - TARGET_INTENT_COUNT) // 20
        )
        schedule.append(
            {
                "schedule_position": position,
                "packet_id": packet_id,
                "pilot": position < TARGET_INTENT_COUNT,
                "block_index": block_index,
                "milestone_index": position // 80,
            }
        )
        mapping.append(
            {
                "schedule_position": position,
                "packet_id": packet_id,
                "ui_id": row["ui_id"],
                "query_id": row["query_id"],
                "response_id": row["response_id"],
                "intent_bucket": row["intent_bucket"],
                "occurrence": occurrence,
                "repeat_of_packet_id": (
                    _packet_id(seed, str(row["ui_id"]), 0)
                    if occurrence
                    else None
                ),
            }
        )
        if occurrence == 0:
            original_schedule_position[str(row["ui_id"])] = position

    for original_index, row in enumerate(originals):
        append(row, 0)
        if original_index < 200:
            continue
        desired = math.floor(
            (original_index - 199)
            * len(random_repeats)
            / (len(originals) - 200)
        )
        while len(scheduled_repeats) < desired:
            eligible = [
                repeat_by_id[ui_id]
                for ui_id in repeat_by_id
                if ui_id not in scheduled_repeats
                and ui_id in original_schedule_position
                and len(schedule) - original_schedule_position[ui_id] >= 200
            ]
            if not eligible:
                break
            chosen = min(
                eligible,
                key=lambda item: _order_key(
                    seed, "repeat_order", item["ui_id"]
                ),
            )
            append(chosen, 1)
            scheduled_repeats.add(str(chosen["ui_id"]))

    for row in sorted(
        random_repeats,
        key=lambda item: _order_key(seed, "repeat_tail", item["ui_id"]),
    ):
        ui_id = str(row["ui_id"])
        if ui_id in scheduled_repeats:
            continue
        if len(schedule) - original_schedule_position[ui_id] < 200:
            raise AssertionError("blind repeat spacing is below 200 positions")
        append(row, 1)
        scheduled_repeats.add(ui_id)

    for row in anchors:
        ui_id = str(row["ui_id"])
        if len(schedule) - original_schedule_position[ui_id] < 200:
            raise AssertionError("anchor repeat spacing is below 200 positions")
        append(row, 1)
        scheduled_repeats.add(ui_id)

    if len(schedule) != len(selected) + BLIND_REPEAT_COUNT:
        raise AssertionError("unexpected judging schedule size")
    return schedule, mapping


def _read_selected_records(
    source_path: Path,
    line_indices: set[int],
) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with source_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_index, line in enumerate(handle):
            if line_index not in line_indices:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{source_path}:{line_index + 1} is not an object")
            records[line_index] = value
    if len(records) != len(line_indices):
        raise RuntimeError("failed to reread every selected source record")
    return records


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )


def build_benchmark_selection(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    metric_scores_path: str | Path | None = None,
    prior_v1_path: str | Path | None = None,
    seed: str = DEFAULT_SELECTION_SEED,
) -> dict[str, Any]:
    source_dir = Path(source_dir).resolve()
    source_path = source_dir / "genui.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"immutable benchmark output already exists: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    metric_scores = _load_metric_scores(
        Path(metric_scores_path).resolve()
        if metric_scores_path is not None
        else None
    )
    summaries = list(
        _iter_source_summaries(source_path, metric_scores=metric_scores)
    )
    by_intent: dict[str, list[CandidateSummary]] = defaultdict(list)
    for candidate in summaries:
        by_intent[candidate.intent_bucket].append(candidate)
    if len(by_intent) != TARGET_INTENT_COUNT:
        raise ValueError(
            f"expected {TARGET_INTENT_COUNT} intents, found {len(by_intent)}"
        )
    prior = _load_prior_anchor_ids(
        Path(prior_v1_path).resolve() if prior_v1_path is not None else None
    )
    selected_rows: list[dict[str, Any]] = []
    selected_candidates: list[CandidateSummary] = []
    for intent in sorted(by_intent):
        candidates = by_intent[intent]
        anchor = _choose_anchor(
            candidates, prior.get(intent, ()), seed=seed
        )
        population_pool = [
            item for item in candidates if item.ui_id != anchor.ui_id
        ]
        population = sorted(
            population_pool,
            key=lambda item: _order_key(
                seed, "population", intent, item.query_id
            ),
        )[:POPULATION_PER_INTENT]
        population_ids = {item.ui_id for item in population}
        stress = _choose_stress(
            candidates,
            anchor=anchor,
            excluded_ids=population_ids,
            seed=seed,
        )
        population_splits = _assign_split(
            population,
            calibration_count=14,
            validation_count=3,
            seed=seed,
            stratum="population",
        )
        stress_splits = _assign_split(
            stress,
            calibration_count=6,
            validation_count=2,
            seed=seed,
            stratum="stress",
        )
        for stratum, rows, splits in (
            ("population", population, population_splits),
            ("stress", stress, stress_splits),
        ):
            for candidate in rows:
                selected_candidates.append(candidate)
                selected_rows.append(
                    {
                        **asdict(candidate),
                        "reachable_types": list(candidate.reachable_types),
                        "active_caps": list(candidate.active_caps),
                        "selection_stratum": stratum,
                        "split": splits[candidate.ui_id],
                        "migration_anchor": candidate.ui_id == anchor.ui_id,
                        "selection_tie_break": _order_key(
                            seed, stratum, candidate.ui_id
                        ),
                    }
                )
    if len(selected_rows) != 960:
        raise AssertionError(f"expected 960 selected rows, got {len(selected_rows)}")
    if len({row["ui_id"] for row in selected_rows}) != 960:
        raise AssertionError("selection contains duplicate ui_id values")
    intent_counts = Counter(row["intent_bucket"] for row in selected_rows)
    if set(intent_counts.values()) != {30}:
        raise AssertionError(f"invalid per-intent counts: {intent_counts}")
    split_counts = Counter(row["split"] for row in selected_rows)
    if split_counts != {
        "calibration": 640,
        "validation": 160,
        "holdout": 160,
    }:
        raise AssertionError(f"invalid split counts: {split_counts}")
    source_splits: dict[str, set[str]] = defaultdict(set)
    for row in selected_rows:
        source_splits[str(row["query_id"])].add(str(row["split"]))
    split_leaks = sorted(
        source_id
        for source_id, splits in source_splits.items()
        if len(splits) != 1
    )
    if split_leaks:
        raise AssertionError(
            "source-grouped split leakage for query_id values: "
            + ", ".join(split_leaks[:10])
        )

    schedule, packet_map = _build_schedule(selected_rows, seed=seed)
    line_indices = {candidate.line_index for candidate in selected_candidates}
    records_by_line = _read_selected_records(source_path, line_indices)
    selected_by_ui = {row["ui_id"]: row for row in selected_rows}

    contracts: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    for candidate in sorted(
        selected_candidates,
        key=lambda item: selected_by_ui[item.ui_id]["selection_tie_break"],
    ):
        record = records_by_line[candidate.line_index]
        response_text = str(record.get("response_text") or "")
        assets = record.get("assets")
        resolution = resolve_expected_ui_contract_v5_4(
            response_text,
            intent=candidate.intent_bucket,
            assets=assets,
        )
        contract_hash = expected_contract_hash_v5_4(resolution.contract)
        contracts.append(
            {
                "ui_id": candidate.ui_id,
                "response_text_sha256": candidate.response_text_sha256,
                "expected_ui_contract": resolution.contract,
                "expected_ui_contract_hash": contract_hash,
                "expected_ui_contract_source": resolution.source,
                "resolution_errors": list(resolution.errors),
            }
        )
        selected_records.append(record)

    _write_jsonl(output_dir / "selection_manifest.jsonl", selected_rows)
    _write_jsonl(output_dir / "selected_genui.jsonl", selected_records)
    _write_jsonl(output_dir / "expected_contracts.jsonl", contracts)
    _write_jsonl(output_dir / "judge_schedule.jsonl", schedule)
    sealed_dir = output_dir / "sealed"
    sealed_dir.mkdir()
    _write_jsonl(sealed_dir / "packet_identity_map.jsonl", packet_map)
    (output_dir / "judge_protocol.json").write_text(
        json.dumps(
            {
                **protocol_mapping(),
                "protocol_fingerprint": protocol_fingerprint(),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "judge_instructions.md").write_text(
        judge_instructions_path().read_text(encoding="utf-8"),
        encoding="utf-8",
        newline="\n",
    )
    source_hash = _hash_file(source_path)
    manifest = {
        "schema_version": "genui_single_codex_selection.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "immutable": True,
        "source_dir": str(source_dir),
        "source_genui": str(source_path),
        "source_genui_sha256": source_hash,
        "source_record_count": len(summaries),
        "selection_seed": seed,
        "selected_count": len(selected_rows),
        "schedule_count": len(schedule),
        "blind_repeat_count": BLIND_REPEAT_COUNT,
        "intent_counts": dict(sorted(intent_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "stratum_counts": dict(
            sorted(Counter(row["selection_stratum"] for row in selected_rows).items())
        ),
        "migration_anchor_count": sum(
            bool(row["migration_anchor"]) for row in selected_rows
        ),
        "metric_scores_source": (
            str(Path(metric_scores_path).resolve())
            if metric_scores_path is not None
            else None
        ),
        "metric_score_coverage": sum(
            candidate.v5_4_score is not None for candidate in summaries
        ),
        "prior_v1_source": (
            str(Path(prior_v1_path).resolve())
            if prior_v1_path is not None
            else None
        ),
        "protocol_fingerprint": protocol_fingerprint(),
        "artifacts": {
            name: _hash_file(output_dir / name)
            for name in (
                "selection_manifest.jsonl",
                "selected_genui.jsonl",
                "expected_contracts.jsonl",
                "judge_schedule.jsonl",
                "judge_protocol.json",
                "judge_instructions.md",
            )
        },
        "judge_instructions_sha256": judge_instructions_sha256(),
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if _hash_file(source_path) != source_hash:
        raise RuntimeError("source genui.jsonl changed during selection")
    write_selection_coverage_audit(
        source_dir,
        output_dir,
        summaries=summaries,
    )
    return manifest


def write_selection_coverage_audit(
    source_dir: str | Path,
    benchmark_dir: str | Path,
    *,
    summaries: Sequence[CandidateSummary] | None = None,
) -> dict[str, Any]:
    source_dir = Path(source_dir).resolve()
    benchmark_dir = Path(benchmark_dir).resolve()
    destination = benchmark_dir / "selection_coverage_audit.json"
    if destination.exists():
        return json.loads(destination.read_text(encoding="utf-8"))
    if summaries is None:
        summaries = list(
            _iter_source_summaries(
                source_dir / "genui.jsonl",
                metric_scores={},
            )
        )
    selected = [
        json.loads(line)
        for line in (
            benchmark_dir / "selection_manifest.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    def type_counts(rows: Sequence[Any]) -> Counter[str]:
        return Counter(
            str(type_name).casefold()
            for row in rows
            for type_name in (
                row.reachable_types
                if isinstance(row, CandidateSummary)
                else row.get("reachable_types", ())
            )
        )

    population_types = type_counts(summaries)
    selected_types = type_counts(selected)
    stress_types = type_counts(
        [
            row
            for row in selected
            if row.get("selection_stratum") == "stress"
        ]
    )
    stress_rows = [
        row
        for row in selected
        if row.get("selection_stratum") == "stress"
    ]
    source_lengths = sorted(
        int(summary.source_length) for summary in summaries
    )
    q75_length = (
        source_lengths[int(0.75 * (len(source_lengths) - 1))]
        if source_lengths
        else 0
    )
    requested_modalities = (
        "table",
        "chart",
        "formula",
        "tabs",
        "modal",
        "image",
        "video",
        "audio",
    )
    audit = {
        "schema_version": "genui_judge_selection_coverage.v2",
        "source_record_count": len(summaries),
        "selected_record_count": len(selected),
        "requested_modalities": list(requested_modalities),
        "source_record_type_availability": {
            name: population_types[name] for name in requested_modalities
        },
        "selected_record_type_coverage": {
            name: selected_types[name] for name in requested_modalities
        },
        "stress_record_type_coverage": {
            name: stress_types[name] for name in requested_modalities
        },
        "stress_score_band_coverage": dict(
            sorted(
                Counter(
                    _score_band(
                        _as_finite_float(row.get("v5_4_score"))
                        if row.get("v5_4_score") is not None
                        else _as_finite_float(row.get("legacy_score"))
                    )
                    for row in stress_rows
                ).items()
            )
        ),
        "stress_render_risks": {
            "wide_table_4_plus_columns": sum(
                int(row.get("max_table_columns", 0)) >= 4
                for row in stress_rows
            ),
            "very_wide_table_6_plus_columns": sum(
                int(row.get("max_table_columns", 0)) >= 6
                for row in stress_rows
            ),
            "source_length_global_q75_plus": sum(
                int(row.get("source_length", 0)) >= q75_length
                for row in stress_rows
            ),
            "component_count_40_plus": sum(
                int(row.get("component_count", 0)) >= 40
                for row in stress_rows
            ),
            "migration_anchor_count": sum(
                bool(row.get("migration_anchor")) for row in stress_rows
            ),
        },
        "unavailable_in_source_population": [
            name
            for name in requested_modalities
            if population_types[name] == 0
        ],
        "missing_requirement_cap_coverage": dict(
            sorted(
                Counter(
                    cap
                    for row in selected
                    if row.get("selection_stratum") == "stress"
                    for cap in row.get("active_caps", ())
                    if str(cap).startswith("missing_")
                    or str(cap).startswith("partial_")
                ).items()
            )
        ),
        "limitations": (
            "A zero-availability renderer type cannot be selected without "
            "fabricating or manually rewriting a generated sample. Relevant "
            "source requirements remain represented through metric cap strata."
        ),
    }
    destination.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return audit


__all__ = [
    "BLIND_REPEAT_COUNT",
    "CALIBRATION_PER_INTENT",
    "DEFAULT_SELECTION_SEED",
    "HOLDOUT_PER_INTENT",
    "POPULATION_PER_INTENT",
    "STRESS_PER_INTENT",
    "TARGET_INTENT_COUNT",
    "VALIDATION_PER_INTENT",
    "build_benchmark_selection",
    "write_selection_coverage_audit",
]
