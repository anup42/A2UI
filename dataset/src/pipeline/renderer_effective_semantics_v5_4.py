"""Versioned renderer-effective component semantics for GenUI metric v5.4.

The functions in this module intentionally model the values consumed by the
production Android renderer, rather than rewarding a preferred FlatSpec
serialization. Historical metric versions do not import this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from . import flat_spec_contract
from .genui_quality import _core
from .genui_quality.evidence_v5_2 import resolve_renderer_path_v5_2


RENDERER_EFFECTIVE_SEMANTICS_VERSION = "5.4.0"
COMPONENT_CONTRACT_POLICY_VERSION = "5.4.0"
MEDIA_MAX_DEPTH = 5

_ROW_LIST_KEYS = ("cells", "values", "row", "data")
_MEDIA_OBJECT_KEYS = (
    "uri",
    "url",
    "src",
    "path",
    "value",
    "source",
    "image",
    "icon",
    "name",
)
_MEDIA_PROP_KEYS = {
    "image": ("url", "src", "image", "source", "name"),
    "icon": ("name", "icon", "source", "url", "src"),
    "video": ("url", "src", "source", "name"),
    "audio": ("url", "src", "source", "name"),
}
_CHART_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class EffectiveColumn:
    key: str
    label: str


@dataclass(frozen=True)
class EffectiveTable:
    columns: tuple[EffectiveColumn, ...]
    rows: tuple[tuple[str, ...], ...]
    source_kind: str
    complete: bool
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class EffectiveChart:
    columns: tuple[EffectiveColumn, ...]
    x_index: int | None
    y_index: int | None
    x_labels: tuple[str, ...]
    y_values: tuple[float, ...]
    y_display_values: tuple[str, ...]
    title: str
    chart_type: str
    dataset_identity: str
    complete: bool
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class EffectiveMedia:
    kind: str
    url: str
    complete: bool
    diagnostics: tuple[str, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _string_map(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {str(key): nested for key, nested in value.items()}


def _normalize_column_key(raw: str, fallback: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_./-]+", "_", raw.strip()).strip("_")
    return value or fallback


def _prettify_key(key: str) -> str:
    value = re.sub(r"\s+", " ", key.replace("_", " ").replace(".", " ").replace("/", " ")).strip()
    return " ".join(part[:1].upper() + part[1:].lower() for part in value.split()) or key


def _lookup_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().casefold())


def _parse_columns(value: Any) -> list[EffectiveColumn]:
    if not isinstance(value, list):
        return []
    columns: list[EffectiveColumn] = []
    for index, raw in enumerate(value):
        if isinstance(raw, str):
            label = raw.strip()
            if label:
                columns.append(
                    EffectiveColumn(
                        _normalize_column_key(label, f"col_{index + 1}"),
                        label,
                    )
                )
        elif isinstance(raw, Mapping):
            item = _string_map(raw) or {}
            label = str(item.get("label") or "").strip()
            key = str(item.get("key") or "").strip()
            final_label = label or _prettify_key(key)
            fallback = key or final_label
            if final_label or fallback:
                columns.append(
                    EffectiveColumn(
                        _normalize_column_key(fallback, f"col_{index + 1}"),
                        final_label or f"Column {index + 1}",
                    )
                )
    return columns


def _resolve_rows(
    props: Mapping[str, Any],
    state: Mapping[str, Any],
) -> tuple[list[Any], str]:
    direct = props.get("rows")
    if isinstance(direct, list):
        return list(direct), "rows"
    state_path = str(props.get("statePath") or "").strip()
    if state_path:
        value = resolve_renderer_path_v5_2(state, state_path)
        if isinstance(value, list):
            return list(value), "statePath"
    table = _string_map(props.get("table"))
    if table is not None and isinstance(table.get("rows"), list):
        return list(table["rows"]), "props.table.rows"
    return [], "missing"


def _resolve_columns(
    props: Mapping[str, Any],
    rows: Sequence[Any],
) -> tuple[list[EffectiveColumn], str]:
    direct = _parse_columns(props.get("columns"))
    if direct:
        return direct, "columns"
    table = _string_map(props.get("table"))
    nested = _parse_columns(table.get("columns") if table else None)
    if nested:
        return nested, "props.table.columns"
    for row in rows:
        mapping = _string_map(row)
        if mapping:
            return [
                EffectiveColumn(str(key), _prettify_key(str(key)))
                for key in mapping
            ], "inferred_mapping"
    for row in rows:
        if isinstance(row, list) and len(row) >= 2:
            return [
                EffectiveColumn(str(index), f"Column {index + 1}")
                for index in range(len(row))
            ], "inferred_positional"
    return [], "missing"


def _row_list(mapping: Mapping[str, Any]) -> list[Any] | None:
    for key in _ROW_LIST_KEYS:
        value = mapping.get(key)
        if isinstance(value, list):
            return list(value)
    nested = _string_map(mapping.get("row"))
    if nested is not None:
        for key in _ROW_LIST_KEYS:
            value = nested.get(key)
            if isinstance(value, list):
                return list(value)
    return None


def _path_value(row: Mapping[str, Any], key: str) -> Any:
    if key in row:
        return row[key]
    normalized = key.strip().strip("/")
    if not normalized:
        return None
    parts = (
        [part for part in normalized.split("/") if part]
        if "/" in normalized
        else [part for part in normalized.split(".") if part]
        if "." in normalized
        else []
    )
    if not parts:
        return None
    current: Any = row
    for part in parts:
        if isinstance(current, Mapping):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
        else:
            return None
    return current


def _column_ordinal(column: EffectiveColumn) -> int | None:
    pattern = re.compile(r"(?:^|[_\s-])(?:column|col)?[_\s-]*(\d+)$")
    for value in (column.key, column.label):
        token = value.strip().casefold()
        if token.isdigit() and int(token) > 0:
            return int(token) - 1
        match = pattern.search(token)
        if match and int(match.group(1)) > 0:
            return int(match.group(1)) - 1
    return None


def _display_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        mapping = _string_map(value) or {}
        return _display_text(
            mapping.get("text", mapping.get("label", mapping.get("value")))
        )
    if isinstance(value, list):
        return " / ".join(_display_text(item) for item in value).strip()
    return str(value).strip()


def _resolve_row(
    raw: Any,
    columns: Sequence[EffectiveColumn],
) -> tuple[str, ...]:
    mapping = _string_map(raw)
    if mapping is not None:
        values = _row_list(mapping)
        if values is not None:
            return tuple(
                _display_text(values[index] if index < len(values) else None)
                for index in range(len(columns))
            )
        entries = list(mapping.items())
        normalized = {
            _lookup_token(key): value
            for key, value in entries
            if _lookup_token(key)
        }

        def direct(column: EffectiveColumn) -> Any:
            for candidate in (
                column.key,
                column.label,
                _normalize_column_key(column.label, column.label),
            ):
                value = _path_value(mapping, candidate)
                if value is not None:
                    return value
            for token in (_lookup_token(column.key), _lookup_token(column.label)):
                if token in normalized:
                    return normalized[token]
            ordinal = _column_ordinal(column)
            if ordinal is not None and ordinal < len(entries):
                return entries[ordinal][1]
            return None

        direct_values = [direct(column) for column in columns]
        use_positional = (
            not any(value is not None for value in direct_values)
            and len(entries) >= len(columns)
        )
        return tuple(
            _display_text(
                entries[index][1]
                if use_positional and direct_values[index] is None
                else direct_values[index]
            )
            for index in range(len(columns))
        )
    if isinstance(raw, list):
        return tuple(
            _display_text(raw[index] if index < len(raw) else None)
            for index in range(len(columns))
        )
    return tuple(
        _display_text(raw) if index == 0 else ""
        for index in range(len(columns))
    )


def effective_table(
    props: Mapping[str, Any],
    state: Mapping[str, Any],
) -> EffectiveTable:
    rows, row_source = _resolve_rows(props, state)
    columns, column_source = _resolve_columns(props, rows)
    resolved = tuple(_resolve_row(row, columns) for row in rows)
    diagnostics: list[str] = []
    if not columns:
        diagnostics.append("table_columns_unresolved")
    if not rows:
        diagnostics.append("table_rows_empty")
    complete = bool(columns) and bool(rows) and all(
        len(row) == len(columns) for row in resolved
    )
    return EffectiveTable(
        columns=tuple(columns),
        rows=resolved,
        source_kind=f"{column_source}+{row_source}",
        complete=complete,
        diagnostics=tuple(diagnostics),
    )


def output_table_from_effective(
    element_id: str,
    element_type: str,
    table: EffectiveTable,
) -> _core.OutputTable:
    keys = [column.key for column in table.columns]
    return _core.OutputTable(
        element_id=element_id,
        element_type=element_type,
        headers=[column.label for column in table.columns],
        keys=keys,
        rows=[
            {
                key: row[index] if index < len(row) else ""
                for index, key in enumerate(keys)
            }
            for row in table.rows
        ],
    )


def _chart_column_index(
    columns: Sequence[EffectiveColumn],
    explicit_key: Any,
    fallback: int,
) -> int:
    token = _lookup_token(explicit_key)
    if token:
        for index, column in enumerate(columns):
            if token in {_lookup_token(column.key), _lookup_token(column.label)}:
                return index
    return min(max(0, fallback), max(0, len(columns) - 1))


def _parse_chart_number(value: str) -> float | None:
    match = _CHART_NUMBER.search(value)
    if match is None:
        return None
    try:
        result = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def effective_chart(
    props: Mapping[str, Any],
    state: Mapping[str, Any],
) -> EffectiveChart:
    raw_rows = props.get("rows")
    source_kind = "rows"
    if not isinstance(raw_rows, list):
        raw_rows = props.get("data")
        source_kind = "data"
    if not isinstance(raw_rows, list):
        raw_rows, source_kind = _resolve_rows(props, state)
    rows = list(raw_rows) if isinstance(raw_rows, list) else []
    columns, column_source = _resolve_columns(props, rows)
    diagnostics: list[str] = []
    if len(columns) < 2:
        diagnostics.append("chart_columns_insufficient")
        return EffectiveChart(
            tuple(columns),
            None,
            None,
            (),
            (),
            (),
            str(props.get("title") or props.get("label") or "").strip(),
            str(props.get("chartType") or props.get("type") or "").strip(),
            "",
            False,
            tuple(diagnostics),
        )
    x_index = _chart_column_index(columns, props.get("xKey"), 0)
    y_index = _chart_column_index(columns, props.get("yKey"), 1)
    if x_index == y_index:
        diagnostics.append("chart_axes_collide")
        return EffectiveChart(
            tuple(columns),
            x_index,
            y_index,
            (),
            (),
            (),
            str(props.get("title") or props.get("label") or "").strip(),
            str(props.get("chartType") or props.get("type") or "").strip(),
            "",
            False,
            tuple(diagnostics),
        )
    labels: list[str] = []
    numbers: list[float] = []
    displays: list[str] = []
    for raw in rows:
        resolved = _resolve_row(raw, columns)
        label = resolved[x_index].strip()
        display = resolved[y_index].strip()
        number = _parse_chart_number(display)
        if label and display and number is not None:
            labels.append(label)
            numbers.append(number)
            displays.append(display)
    if not labels:
        diagnostics.append("chart_points_empty")
    payload = {
        "columns": [asdict(column) for column in columns],
        "x_index": x_index,
        "y_index": y_index,
        "labels": labels,
        "display_values": displays,
    }
    return EffectiveChart(
        columns=tuple(columns),
        x_index=x_index,
        y_index=y_index,
        x_labels=tuple(labels),
        y_values=tuple(numbers),
        y_display_values=tuple(displays),
        title=str(props.get("title") or props.get("label") or "").strip(),
        chart_type=str(
            props.get("chartType") or props.get("type") or ""
        ).strip(),
        dataset_identity=_sha256(payload),
        complete=bool(labels),
        diagnostics=tuple(diagnostics),
    )


def output_table_from_chart(
    element_id: str,
    chart: EffectiveChart,
) -> _core.OutputTable | None:
    if chart.x_index is None or chart.y_index is None:
        return None
    x_column = chart.columns[chart.x_index]
    y_column = chart.columns[chart.y_index]
    return _core.OutputTable(
        element_id=element_id,
        element_type="Chart",
        headers=[x_column.label, y_column.label],
        keys=[x_column.key, y_column.key],
        rows=[
            {
                x_column.key: label,
                y_column.key: display,
            }
            for label, display in zip(
                chart.x_labels, chart.y_display_values
            )
        ],
    )


def canonical_media_kind(value: Any) -> str:
    token = str(value or "").strip().casefold().replace("_", "")
    return {
        "image": "image",
        "icon": "icon",
        "video": "video",
        "audio": "audio",
        "audioplayer": "audio",
    }.get(token, token)


def extract_media_url_token(value: Any, depth: int = 0) -> str | None:
    if depth > MEDIA_MAX_DEPTH or value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, Mapping):
        mapping = _string_map(value) or {}
        for key in _MEDIA_OBJECT_KEYS:
            resolved = extract_media_url_token(mapping.get(key), depth + 1)
            if resolved:
                return resolved
        return None
    if isinstance(value, list):
        for candidate in value:
            resolved = extract_media_url_token(candidate, depth + 1)
            if resolved:
                return resolved
    return None


def effective_media(
    element_type: Any,
    props: Mapping[str, Any],
) -> EffectiveMedia:
    kind = canonical_media_kind(element_type)
    keys = _MEDIA_PROP_KEYS.get(kind, ("url", "src", "source", "name"))
    url = ""
    for key in keys:
        resolved = extract_media_url_token(props.get(key))
        if resolved:
            url = resolved
            break
    diagnostics = () if url else ("media_url_unresolved",)
    return EffectiveMedia(kind, url, bool(url), diagnostics)


def renderer_effective_type_contract(
    spec: Mapping[str, Any],
    output: _core.OutputEvidence,
    evidence_result: Any,
    audit: _core.GraphAudit,
    config: Any,
) -> tuple[float, dict[str, Any]]:
    del output, evidence_result, config
    elements = spec.get("elements")
    elements = elements if isinstance(elements, Mapping) else {}
    state = spec.get("state")
    state = state if isinstance(state, Mapping) else {}
    scores: dict[str, float] = {}
    diagnostics: dict[str, list[str]] = {}
    for element_id in sorted(audit.reachable_ids):
        raw = elements.get(element_id)
        if not isinstance(raw, Mapping):
            scores[element_id] = 0.0
            diagnostics[element_id] = ["missing_element"]
            continue
        element_type = str(raw.get("type") or "")
        props = raw.get("props")
        props = props if isinstance(props, Mapping) else {}
        score = 1.0
        issues: list[str] = []
        if element_type == "Stack":
            direction = str(props.get("direction") or "").strip().casefold()
            if direction and direction not in {
                "row",
                "column",
                "horizontal",
                "vertical",
            }:
                score = 0.5
                issues.append("unsupported_stack_direction")
        elif element_type == "Table":
            model = effective_table(props, state)
            if not model.columns:
                score = 0.0
            elif not model.rows:
                score = 0.5
            issues.extend(model.diagnostics)
        elif element_type == "Chart":
            model = effective_chart(props, state)
            score = 1.0 if model.complete else 0.0
            issues.extend(model.diagnostics)
        elif element_type in {"Image", "Icon", "Video", "AudioPlayer"}:
            model = effective_media(element_type, props)
            score = 1.0 if model.complete else 0.0
            issues.extend(model.diagnostics)
        repeat = raw.get("repeat")
        if isinstance(repeat, Mapping):
            path = str(repeat.get("statePath") or "").strip()
            values = resolve_renderer_path_v5_2(state, path)
            if not path or not isinstance(values, list):
                score = min(score, 0.0)
                issues.append("repeat_state_not_list")
        scores[element_id] = score
        if issues:
            diagnostics[element_id] = issues
    value = sum(scores.values()) / len(scores) if scores else 0.0
    return value, {
        "policy_version": COMPONENT_CONTRACT_POLICY_VERSION,
        "per_element": scores,
        "diagnostics": diagnostics,
    }


def _effective_props(
    element_id: str,
    raw: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    element_type = str(raw.get("type") or "")
    props = raw.get("props")
    props = dict(props) if isinstance(props, Mapping) else {}
    if element_type == "Table":
        model = effective_table(props, state)
        return {
            "columns": [asdict(column) for column in model.columns],
            "rows": [list(row) for row in model.rows],
            **{
                key: props[key]
                for key in ("title", "domain", "presentation", "variant")
                if key in props
            },
        }
    if element_type == "Chart":
        model = effective_chart(props, state)
        table = output_table_from_chart(element_id, model)
        rows = table.rows if table is not None else []
        return {
            "title": model.title,
            "chartType": model.chart_type,
            "columns": [
                asdict(model.columns[index])
                for index in (model.x_index, model.y_index)
                if index is not None
            ],
            "rows": rows,
            "xKey": (
                model.columns[model.x_index].key
                if model.x_index is not None
                else ""
            ),
            "yKey": (
                model.columns[model.y_index].key
                if model.y_index is not None
                else ""
            ),
        }
    if element_type in {"Image", "Icon", "Video", "AudioPlayer"}:
        model = effective_media(element_type, props)
        return {
            "url": model.url,
            **{
                key: props[key]
                for key in (
                    "alt",
                    "description",
                    "title",
                    "accessibilityLabel",
                    "decorative",
                )
                if key in props
            },
        }
    return props


def renderer_effective_size_utility(
    spec: Mapping[str, Any],
    audit: _core.GraphAudit,
    source: _core.SourceContract,
    config: Any,
    evidence_result: Any,
) -> float:
    del evidence_result
    elements = spec.get("elements")
    elements = elements if isinstance(elements, Mapping) else {}
    state = spec.get("state")
    state = state if isinstance(state, Mapping) else {}
    payload = {
        "root": spec.get("root"),
        # Renderer-consumed state is already materialized into effective
        # Table/Chart/media evidence. Counting its serialization again would
        # make statePath and inline Android-equivalent representations score
        # differently. Unbound state remains in diagnostics, never positive
        # evidence.
        "state": {},
        "elements": {
            element_id: {
                "type": raw.get("type"),
                "props": _effective_props(element_id, raw, state),
                "children": list(raw.get("children") or []),
                **({"repeat": raw["repeat"]} if "repeat" in raw else {}),
                **({"visible": raw["visible"]} if "visible" in raw else {}),
                **({"on": raw["on"]} if "on" in raw else {}),
                **({"watch": raw["watch"]} if "watch" in raw else {}),
            }
            for element_id in sorted(audit.reachable_ids)
            for raw in [elements.get(element_id)]
            if isinstance(raw, Mapping)
        },
    }
    ratio = len(_canonical_json(payload)) / max(1, len(source.raw_text))
    return _core.low_is_good(
        ratio,
        config.good_json_to_source_ratio,
        config.bad_json_to_source_ratio,
    )


def renderer_effective_semantics_hash() -> str:
    probes = {
        "version": RENDERER_EFFECTIVE_SEMANTICS_VERSION,
        "table": asdict(
            effective_table(
                {
                    "columns": ["Name", "Value"],
                    "rows": [["A", 1]],
                },
                {},
            )
        ),
        "chart": asdict(
            effective_chart(
                {
                    "columns": ["Quarter", "Revenue"],
                    "rows": [["Q1", "$1,000"]],
                },
                {},
            )
        ),
        "media": asdict(
            effective_media(
                "Image",
                {"source": [{"uri": "https://example.invalid/a.png"}]},
            )
        ),
        "allowed_types": sorted(flat_spec_contract._ALLOWED_TYPES),  # type: ignore[attr-defined]
    }
    return _sha256(probes)


__all__ = [
    "COMPONENT_CONTRACT_POLICY_VERSION",
    "EffectiveChart",
    "EffectiveColumn",
    "EffectiveMedia",
    "EffectiveTable",
    "MEDIA_MAX_DEPTH",
    "RENDERER_EFFECTIVE_SEMANTICS_VERSION",
    "canonical_media_kind",
    "effective_chart",
    "effective_media",
    "effective_table",
    "extract_media_url_token",
    "output_table_from_chart",
    "output_table_from_effective",
    "renderer_effective_semantics_hash",
    "renderer_effective_size_utility",
    "renderer_effective_type_contract",
]
