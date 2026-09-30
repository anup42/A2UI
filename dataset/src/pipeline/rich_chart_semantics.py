"""Chart capability v2.1 data contract, mirrored by Android FlatChartModel.kt.

Only resolved, drawable data is evidence. Never invent a series, axis assignment,
box statistic or bubble size to make an incomplete chart pass admission.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import math
import re
from typing import Any, Mapping

from .renderer_capability import canonical_chart_subtype

_MISSING = {"", "-", "—", "–", "n/a", "na", "null", "none"}
_NUMBER = re.compile(r"^[\s$€£₹¥]*(?:[A-Z]{3}\s+)?([+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)(?:\s*[%°a-zA-Zµμ/²³$€£₹¥]*)$", re.ASCII)
_GROUPED = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[eE][+-]?\d+)?", re.ASCII)


def chart_number(value: str) -> float | None:
    match = _NUMBER.fullmatch(value.strip().replace("−", "-"))
    if not match:
        return None
    number = match[1]
    if "," in number and not _GROUPED.fullmatch(number):
        return None
    result = float(number.replace(",", ""))
    return result if math.isfinite(result) and abs(result) <= 1e150 else None


def _time(value: str) -> float | None:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:T.*)?", value):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return float(math.floor(parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).timestamp() * 1000))
    except (ValueError, OverflowError, OSError):
        return None


def resolve_chart(props: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    # Lazy import keeps the shared Table row/column resolver authoritative.
    from .renderer_effective_semantics_v5_4 import _resolve_rows, _resolve_columns, _resolve_row, _lookup_token, _sha256
    diagnostics: list[str] = []

    def fail(reason: str) -> None:
        if reason not in diagnostics:
            diagnostics.append(reason)

    raw_type = str(props.get("chartType") or "bar").strip().lower() or "bar"
    kind = canonical_chart_subtype(props.get("chartType"))
    if kind is None:
        fail("unsupported_chart_subtype")
        kind = raw_type
    raw_rows = props.get("rows")
    if not isinstance(raw_rows, list):
        raw_rows = props.get("data")
    if not isinstance(raw_rows, list):
        raw_rows, _ = _resolve_rows({**props, "statePath": props.get("statePath") or props.get("rowsPath") or props.get("dataPath")}, state)
    columns, _ = _resolve_columns(props, raw_rows)
    if len(columns) < 2:
        fail("chart_columns_insufficient")
    if len(raw_rows) > 512 or len(columns) > 64:
        fail("chart_resource_limit")
    rows = [_resolve_row(raw, columns) for raw in raw_rows[:512]]

    def column(key: Any, fallback: int) -> int:
        if key is None or not str(key).strip():
            return fallback
        for index, col in enumerate(columns):
            if _lookup_token(key) in {_lookup_token(col.key), _lookup_token(col.label)}:
                return index
        fail("chart_binding_missing")
        return -1

    def cell(row: tuple[str, ...], index: int) -> str:
        return row[index].strip() if 0 <= index < len(row) else ""

    x_index = column(props.get("xKey"), 0)
    labels = [cell(row, x_index) for row in rows]
    if any(not label for label in labels):
        fail("chart_x_missing")
    x_type = str(props.get("xType", "number" if kind in {"scatter", "bubble"} else "category")).lower()
    if x_type not in {"number", "time", "category"}:
        fail("chart_x_type_invalid")
    if kind in {"scatter", "bubble"} and x_type != "number":
        fail("chart_scatter_x_not_numeric")
    x_values = []
    for index, label in enumerate(labels):
        value = chart_number(label) if x_type == "number" else _time(label) if x_type == "time" else float(index)
        if value is None:
            fail("chart_x_invalid")
        x_values.append(float(index) if value is None else value)
    explicit = props.get("series")
    if "series" in props and not isinstance(explicit, list):
        fail("chart_series_invalid")
    single = kind in {"bar", "column", "pie", "donut", "scatter", "bubble", "funnel", "treemap"}
    if kind == "box":
        keys = props.get("boxKeys") or {}
        if not isinstance(keys, Mapping):
            keys = {}
        specs = []
        for role in ("min", "q1", "median", "q3", "max"):
            if keys.get(role) is None:
                fail("chart_box_keys_required")
            specs.append({"yKey": keys.get(role, role), "label": role})
    elif isinstance(explicit, list):
        specs = []
        for item in explicit:
            if not isinstance(item, Mapping) or item.get("yKey") is None or set(item) - {"yKey", "label", "type", "axis", "unit"}:
                fail("chart_series_invalid")
            specs.append(item if isinstance(item, Mapping) else {})
    elif kind == "combo":
        fail("chart_combo_series_required")
        specs = []
    elif props.get("yKey") is not None or single:
        specs = [{"yKey": props.get("yKey", columns[1].key if len(columns) > 1 else None)}]
    else:
        specs = [{"yKey": col.key} for index, col in enumerate(columns) if index != x_index and any(chart_number(cell(row, index)) is not None for row in rows)]
    if len(specs) > 16:
        fail("chart_resource_limit")
    series = []
    for spec in specs[:16]:
        if any(not isinstance(v, str) for v in spec.values()) or not str(spec.get('yKey') or '').strip():
            fail('chart_series_invalid')
        index = column(spec.get("yKey"), 1)
        if index == x_index:
            fail("chart_axes_collide")
        displays = [cell(row, index) for row in rows]
        values = [chart_number(display) for display in displays]
        if any(v is None and d.lower() not in _MISSING for v, d in zip(values, displays)):
            fail("chart_value_invalid")
        if not any(v is not None for v in values):
            fail("chart_series_empty")
        mark = str(spec.get("type", "" if kind == "combo" else kind)).lower()
        if (kind == "combo" and mark not in {"bar", "column", "line", "area", "scatter"}) or (kind != "combo" and "type" in spec and mark != kind):
            fail("chart_series_mark_invalid")
        axis = str(spec.get("axis", "left")).lower()
        if axis not in {"left", "right"} or (axis == "right" and kind not in {"combo", "line", "area", "scatter"}):
            fail("chart_axis_invalid")
        col = columns[index] if 0 <= index < len(columns) else None
        series.append({"key": col.key if col else "", "label": str(spec.get("label", col.label if col else "")), "column_index": index,
                       "mark": mark, "axis": axis, "unit": str(spec.get("unit", "")), "values": values, "displays": displays})
    if not series or not rows:
        fail("chart_points_empty")
    if len({s["column_index"] for s in series}) != len(series):
        fail("chart_series_duplicate")
    if single and kind != "scatter" and len(series) > 1:
        fail("chart_single_series_required")
    if "dual" in raw_type and {s["axis"] for s in series} != {"left", "right"}:
        fail("chart_dual_axes_required")
    if any(s['axis'] == 'right' for s in series) and not any(s['axis'] == 'left' for s in series):
        fail("chart_dual_axes_required")
    values = [v for s in series for v in s["values"]]
    if kind in {"pie", "donut", "treemap", "funnel", "radar", "box", "stackedarea", "stackedbar"} and any(v is None for v in values):
        fail("chart_complete_values_required")
    if kind in {"pie", "donut", "treemap", "funnel", "radar"} and any(v is not None and v < 0 for v in values):
        fail("chart_negative_weight")
    if kind in {"pie", "donut", "treemap", "funnel"} and sum(v or 0 for v in (series[0]["values"] if series else [])) <= 0:
        fail("chart_positive_total_required")
    if kind == "radar" and len(rows) < 3:
        fail("chart_radar_categories_insufficient")
    if kind == "box" and len(series) == 5:
        for r in range(len(rows)):
            numbers = [s["values"][r] for s in series]
            if any(v is None for v in numbers) or any(a > b for a, b in zip(numbers, numbers[1:])):
                fail("chart_box_order_invalid")
    size_index = column(props.get("sizeKey"), -1) if kind == "bubble" else None
    if kind == "bubble" and (size_index is None or size_index < 0 or size_index == x_index or any(s["column_index"] == size_index for s in series)):
        fail("chart_size_key_required")
    sizes = [chart_number(cell(row, size_index)) for row in rows] if kind == "bubble" else []
    if kind == "bubble" and any(v is None or v < 0 for v in sizes):
        fail("chart_size_invalid")
    orientation = str(props.get("orientation", "horizontal" if kind == "bar" else "vertical")).lower()
    if orientation not in {"horizontal", "vertical"} or (orientation == "horizontal" and kind not in {"bar", "column", "groupedbar", "stackedbar"}):
        fail("chart_orientation_invalid")
    if orientation == 'horizontal' and x_type != 'category':
        fail('chart_orientation_invalid')
    # The chart's visible data key preserves supplied ancillary cells (shares,
    # caveats, units) even when they are not selected as plotted measures.
    selected_indices = list(range(len(columns)))
    payload = {"columns": [asdict(col) for col in columns], "x_index": x_index, "x_values": x_values,
               "x_type": x_type, "labels": labels, "series": series, "sizes": sizes, "orientation": orientation, "rows": rows}
    return dict(columns=tuple(columns), x_index=x_index if 0 <= x_index < len(columns) else None,
                y_index=series[0]["column_index"] if series and 0 <= series[0]["column_index"] < len(columns) else None,
                x_labels=tuple(labels), y_values=tuple(series[0]["values"] if series else []),
                y_display_values=tuple(series[0]["displays"] if series else []),
                title=str(props.get("title") or props.get("label") or "").strip(), chart_type=kind,
                dataset_identity=_sha256(payload), complete=not diagnostics, diagnostics=tuple(diagnostics),
                series=tuple(series), x_values=tuple(x_values), x_type=x_type, sizes=tuple(sizes),
                selected_indices=tuple(selected_indices), selected_rows=tuple(tuple(cell(row, i) for i in selected_indices) for row in rows))
