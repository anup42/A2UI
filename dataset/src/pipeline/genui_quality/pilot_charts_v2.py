"""Source-bound chart specifications for the opt-in Stage 3 pilot."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass

from pipeline.renderer_capability import canonical_chart_subtype

from .pilot_layout_v2 import layout_text, table_cell_spelling
from .source_contract_v5_3 import _chart_payload

_DIRECTIVE = re.compile(
    r"(Chart Title|Chart Type|Title|X[- ]Axis(?:/Labels)?|Y[- ]Axis(?:/Values)?|Right Y[- ]Axis|Labels|Values):\s*(\S.*)",
    re.IGNORECASE,
)
_AXES = {"x axis": "x", "x axis/labels": "x", "y axis": "y", "y axis/values": "y", "right y axis": "right y"}
_PIPE_ROW = re.compile(r"\s*[-*+]\s+[^|\n]+\|[^\n]+")
_LEGACY_FIELDS = {"id", "required", "minimum_count", "interchangeable", "title", "chart_type", "source_section",
                  "x_fields", "y_fields"}


def _type(value: str) -> tuple[str | None, str]:
    token = re.sub(r"\s+chart$", "", value.casefold()).strip()
    orientation = ""
    if token in {"horizontal bar", "vertical bar"}:
        orientation, token = token.split()
    return canonical_chart_subtype(token.replace(" ", "")) if token else None, orientation


def chart_axis_key(label: str) -> str:
    # Role similarity tokenizes strings. A digest preserves case, signs,
    # punctuation and units in an explicitly supplied axis label.
    return hashlib.sha256(" ".join(label.split()).encode()).hexdigest()


def chart_axis_data_key(rows) -> str:
    cells = [[table_cell_spelling(str(cell)) for cell in row] for row in rows]
    return hashlib.sha256(json.dumps(cells, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class _ChartSpec:
    fields: dict[str, tuple[str, str]]
    rows: tuple[str, ...]
    ambiguous: bool

    @property
    def title(self) -> str:
        return self.fields.get("chart title", ("", ""))[1]

    @property
    def title_parts(self) -> tuple[str, str]:
        suffix = re.fullmatch(r"(.+?)\s+\(([^()]+ Chart)\)", self.title, re.IGNORECASE)
        return (suffix[1], suffix[2]) if suffix and _type(suffix[2])[0] else (self.title, "")

    @property
    def kind(self) -> tuple[str | None, str]:
        explicit = self.fields.get("chart type", ("", ""))[1]
        return _type(explicit or self.title_parts[1])

    @property
    def usable(self) -> bool:
        explicit = self.fields.get("chart type", ("", ""))[1]
        suffix = self.title_parts[1]
        return (bool(self.title) and not self.ambiguous and (not explicit or self.kind[0] is not None)
                and (not explicit or not suffix or _type(explicit)[0] == _type(suffix)[0]))


def chart_specs(source: str) -> tuple[_ChartSpec, ...]:
    """Read one contiguous specification at a time, outside code fences."""
    specs, fields, rows = [], {}, []
    in_spec = ambiguous = False
    fence = None
    for line in [*source.splitlines(), ""]:
        raw = layout_text(line)
        marker = re.match(r"^(`{3,}|~{3,})", line.lstrip())
        if marker:
            token = marker[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
        if not fence and not marker:
            if raw.casefold() == "chart data specification":
                in_spec = True
                continue
            match = _DIRECTIVE.fullmatch(raw)
            if match and (in_spec or fields or match[1].casefold() in {"chart type", "chart title"}):
                field = match[1].casefold().replace("-", " ")
                field = _AXES.get(field, "chart title" if field == "title" else field)
                ambiguous |= field in fields
                fields.setdefault(field, (line.strip(), match[2]))
                continue
            if fields and _PIPE_ROW.fullmatch(line):
                rows.append(line.strip())
                continue
        if fields:
            specs.append(_ChartSpec(fields, tuple(rows), ambiguous))
        fields, rows, in_spec, ambiguous = {}, [], False, False
    return tuple(specs)


def _declared_data(spec: _ChartSpec):
    """Return ordered visible categories and exact series cells when explicit."""
    fields = spec.fields
    if spec.rows:
        if "labels" in fields or "values" in fields or not {"x", "y"} <= fields.keys():
            return None
        labels, series = [], []
        for line in spec.rows:
            label, *cells = [cell.strip() for cell in layout_text(line).split("|")]
            pairs = [cell.partition(": ") for cell in cells]
            names = tuple(name for name, _, _ in pairs)
            if (not label or not names or len(set(names)) != len(names)
                    or any(not (name and sep and value) for name, sep, value in pairs)):
                return None
            if not series:
                series = [(name, []) for name in names]
            if names != tuple(name for name, _ in series):
                return None
            labels.append(label)
            for (_, values), (_, _, value) in zip(series, pairs):
                values.append(value)
        return labels, series
    if "labels" in fields and "values" in fields:
        labels, values = (re.split(r",\s+", fields[key][1]) for key in ("labels", "values"))
        if len(labels) == len(values) and all(labels) and all(values):
            return labels, [(None, values)]
    return None


def _data_matches(spec, candidate):
    data = _declared_data(spec)
    if data is None:
        return False
    labels, series = data
    actual = candidate.get("series", ())
    columns = candidate.get("pilot_series_columns", ())
    return (list(candidate.get("x_labels", ())) == labels and len(actual) == len(series)
            and all((name is None or (item["label"] == name and i < len(columns) and columns[i] == name))
                    and list(map(table_cell_spelling, item["displays"])) == list(map(table_cell_spelling, values))
                    for i, ((name, values), item) in enumerate(zip(series, actual))))


def _axis_columns(spec, tables):
    """Require a unique source table binding before retiring guessed axis keys."""
    if not {"x", "y"} <= spec.fields.keys():
        return None
    labels = [spec.fields[axis][1] for axis in ("x", "y")]
    matches = []
    for table in tables:
        headers = table.get("headers", ())
        selected = []
        for label in labels:
            stem = re.sub(r"\s+\([^()]+\)$", "", label)
            names = [header for header in headers if layout_text(header) in {label, stem}]
            if len(names) == 1:
                selected.append(names[0])
        rows = table.get("rows", ())
        if (len(selected) == 2 and selected[0] != selected[1] and rows
                and all(isinstance(row, (list, tuple)) and len(row) == len(headers) for row in rows)):
            x_index, y_index = (headers.index(header) for header in selected)
            matches.append((selected, chart_axis_data_key((row[x_index], row[y_index]) for row in rows)))
    return matches[0] if len(matches) == 1 else None


def bind_chart_specs(source, roles, tables):
    """Merge only proven title/type aliases of the same source declaration."""
    specs = chart_specs(source)
    charts = list(roles.get("chart", ()))
    type_specs = {}
    for spec in specs:
        if "chart type" in spec.fields:
            type_specs.setdefault(spec.kind[0], []).append(spec)
    bound = Counter()
    for spec in specs:
        if not spec.usable or sum(other.title_parts[0] == spec.title_parts[0] for other in specs) != 1:
            continue
        kind, orientation = spec.kind
        title = spec.title_parts[0]
        aliases = {title, spec.title, "Chart Title: " + title, _chart_payload(spec.title)["title"]}
        matched = [item for item in charts if item.get("title") in aliases]
        if len(type_specs.get(kind, ())) == 1 and "chart type" in spec.fields:
            matched.extend(item for item in charts if layout_text(item.get("title", "")).casefold() == "chart type"
                           and _type(item.get("chart_type", "").replace("_", " "))[0] == kind)
        if (not matched or any(set(item) - _LEGACY_FIELDS or item.get("minimum_count", 1) != 1
                               or not item.get("required", True) or item.get("interchangeable")
                               or (kind and item.get("chart_type") and
                                   _type(item["chart_type"].replace("_", " "))[0] != kind)
                               for item in matched)):
            continue
        requirement = {**matched[0], "title": title, "pilot_chart_title": chart_axis_key(title)}
        if kind:
            requirement["chart_type"] = kind
        if orientation:
            requirement["pilot_orientation"] = orientation
        for axis in ("x", "y", "right y"):
            if axis in spec.fields:
                requirement["pilot_axis_" + axis] = chart_axis_key(spec.fields[axis][1])
        declared_data = _declared_data(spec) is not None
        bindings = _axis_columns(spec, tables) if not declared_data else None
        if bindings or declared_data:
            requirement.pop("x_fields", None)
            requirement.pop("y_fields", None)
        if bindings:
            (x_key, y_key), data_key = bindings
            requirement.update(pilot_physical_x=[x_key], pilot_physical_y=[y_key], pilot_axis_data=data_key)
        if declared_data:
            requirement["pilot_chart_data"] = "exact chart data"
        first = charts.index(matched[0])
        charts = [item for item in charts if item not in matched]
        charts.insert(first, requirement)
        if "chart type" in spec.fields:
            bound[kind] += 1
    # A synthetic type-only role can also be shared by several distinct
    # titled declarations of that type. Retire it only after *all* such
    # declarations received their own requirements.
    charts = [item for item in charts if not (
        layout_text(item.get("title", "")).casefold() == "chart type" and set(item) <= _LEGACY_FIELDS
        and item.get("minimum_count", 1) == 1 and not item.get("interchangeable")
        and (kind := _type(item.get("chart_type", "").replace("_", " "))[0]) in bound
        and bound[kind] == len(type_specs[kind]))]
    return {**roles, "chart": charts} if charts else roles


def chart_role_candidates(source, expected, actual):
    specs = chart_specs(source)
    titles = {item["title"] for item in expected.get("chart", ()) if "pilot_chart_title" in item}
    charts = []
    candidates = actual.get("chart", ())
    for chart in candidates:
        matches = [spec for spec in specs if spec.usable and spec.title_parts[0] in titles
                   and chart.get("title") in {spec.title, spec.title_parts[0]}]
        if len(matches) == 1 and sum(other.get("title") in {matches[0].title, matches[0].title_parts[0]}
                                     for other in candidates) == 1:
            spec = matches[0]
            title = spec.title_parts[0]
            chart = {**chart, "title": title, "pilot_chart_title": chart_axis_key(title)}
            if _data_matches(spec, chart):
                chart["pilot_chart_data"] = "exact chart data"
        charts.append(chart)
    return {**actual, "chart": charts} if charts else actual


def declared_chart_row_texts(source):
    """A chart's own data rows are not independent generic entity records."""
    specs = chart_specs(source)
    occurrences = Counter(layout_text(line) for line in source.splitlines())
    return {layout_text(line) for spec in specs if spec.usable and _declared_data(spec) is not None
            and sum(other.title_parts[0] == spec.title_parts[0] for other in specs) == 1
            for line in spec.rows if occurrences[layout_text(line)] == 1}


def project_chart_fields(source, generic, ownership, charts):
    """Credit exact directives and cells from the same complete chart."""
    specs = chart_specs(source)
    rows, certificates = list(ownership), []
    available = {}
    for unit in generic:
        available.setdefault(layout_text(unit.raw), []).append(unit.raw)
    for chart in charts:
        matches = [spec for spec in specs if spec.usable and chart.get("title") in {spec.title, spec.title_parts[0]}]
        if len(matches) != 1:
            continue
        spec = matches[0]
        kind, orientation = spec.kind
        if ((kind and chart.get("chart_type") != kind)
                or (orientation and chart.get("pilot_orientation") != orientation)):
            continue
        element_id = chart["component_id"]
        group = [row for row in rows if row.get("component_id") == element_id and row.get("owner") == "source_chart"]
        if not group or len({row.get("render_instance") for row in group}) != 1:
            continue
        # A repeated chart element is never an independent source proof.
        if sum(other.get("title") in {spec.title, spec.title_parts[0]} for other in charts) != 1:
            continue
        data_match = _data_matches(spec, chart)
        originals = [raw for field, (raw, value) in spec.fields.items() if (
            field == "chart title" or field == "chart type" and kind
            or field in {"labels", "values"} and data_match
            or chart.get("pilot_axis_" + field) == chart_axis_key(value))]
        if data_match:
            originals.extend(spec.rows)
        end = rows.index(group[-1]) + 1
        additions = []
        for raw in originals:
            key = layout_text(raw)
            matching = available.get(key, [])
            if len(set(matching)) == 1:
                additions.append({**group[-1], "text": matching.pop(), "owner": "generic_visible_content"})
            certificates.append({"source_text": raw, "component_id": element_id, "proof": "exact_chart_field"})
        rows[end:end] = additions
    return rows, certificates
