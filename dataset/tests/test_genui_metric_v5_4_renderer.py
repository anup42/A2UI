from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from genui_metric_v5_4_support import root_spec  # noqa: E402
from pipeline.genui_quality import render_artifact_quality_v5_4  # noqa: E402
from pipeline.renderer_effective_semantics_v5_4 import (  # noqa: E402
    effective_chart,
    effective_media,
    effective_table,
)


def _table_signature(props: dict, state: dict | None = None) -> tuple:
    value = effective_table(props, state or {})
    return (
        tuple(column.label.casefold() for column in value.columns),
        value.rows,
        value.complete,
    )


def test_android_effective_table_forms_are_equivalent() -> None:
    expected = (
        ("city", "temp"),
        (("Paris", "21"), ("Rome", "25")),
        True,
    )
    cases = [
        (
            {
                "columns": [
                    {"key": "city", "label": "City"},
                    {"key": "temp", "label": "Temp"},
                ],
                "rows": [
                    {"city": "Paris", "temp": 21},
                    {"city": "Rome", "temp": 25},
                ],
            },
            {},
        ),
        (
            {
                "columns": [
                    {"key": "city", "label": "City"},
                    {"key": "temp", "label": "Temp"},
                ],
                "rows": [["Paris", 21], ["Rome", 25]],
            },
            {},
        ),
        (
            {
                "columns": ["city", "temp"],
                "rows": [["Paris", 21], ["Rome", 25]],
            },
            {},
        ),
        (
            {
                "table": {
                    "columns": ["city", "temp"],
                    "rows": [["Paris", 21], ["Rome", 25]],
                }
            },
            {},
        ),
        (
            {"columns": ["city", "temp"], "statePath": "rows"},
            {"rows": [["Paris", 21], ["Rome", 25]]},
        ),
        (
            {"columns": ["city", "temp"], "statePath": "/rows"},
            {"rows": [["Paris", 21], ["Rome", 25]]},
        ),
    ]
    assert [_table_signature(props, state) for props, state in cases] == [
        expected
    ] * len(cases)
    source = (
        "| City | Temp |\n"
        "|---|---|\n"
        "| Paris | 21 |\n"
        "| Rome | 25 |"
    )
    scores = [
        render_artifact_quality_v5_4(
            root_spec(
                {
                    "table": {
                        "type": "Table",
                        "props": props,
                        "children": [],
                    }
                },
                ["table"],
                state=state,
            ),
            source,
        ).quality_0_1
        for props, state in cases
    ]
    assert max(scores) - min(scores) <= 1e-12


def _chart_signature(props: dict, state: dict | None = None) -> tuple:
    value = effective_chart(props, state or {})
    return (
        value.x_labels,
        value.y_values,
        value.y_display_values,
        value.complete,
    )


def test_android_effective_chart_forms_and_fallback_are_equivalent() -> None:
    rows = [["Jan", 10], ["Feb", 20]]
    columns = ["month", "revenue"]
    expected = (("Jan", "Feb"), (10.0, 20.0), ("10", "20"), True)
    cases = [
        ({"columns": columns, "rows": rows}, {}),
        ({"columns": columns, "data": rows}, {}),
        ({"columns": columns, "statePath": "rows"}, {"rows": rows}),
        (
            {"table": {"columns": columns, "rows": rows}},
            {},
        ),
    ]
    assert [_chart_signature(props, state) for props, state in cases] == [
        expected
    ] * len(cases)
    scores = [
        render_artifact_quality_v5_4(
            root_spec(
                {
                    "chart": {
                        "type": "Chart",
                        "props": props,
                        "children": [],
                    }
                },
                ["chart"],
                state=state,
            ),
            "Revenue by month as a chart",
        ).quality_0_1
        for props, state in cases
    ]
    assert max(scores) - min(scores) <= 1e-12


@pytest.mark.parametrize(
    ("kind", "props", "expected"),
    [
        ("Image", {"source": {"image": {"src": "image.png"}}}, "image.png"),
        ("Video", {"source": {"value": {"url": "video.mp4"}}}, "video.mp4"),
        ("AudioPlayer", {"url": {"source": "audio.mp3"}}, "audio.mp3"),
    ],
)
def test_recursive_media_sources_match_android(
    kind: str, props: dict, expected: str
) -> None:
    value = effective_media(kind, props)
    assert value.complete
    assert value.url == expected


def test_renderer_inert_chart_metadata_has_zero_reward_effect() -> None:
    chart = {
        "type": "Chart",
        "props": {
            "columns": ["Month", "Revenue"],
            "rows": [["Jan", 10], ["Feb", 20]],
            "xKey": "Month",
            "yKey": "Revenue",
        },
        "children": [],
    }
    decorated = deepcopy(chart)
    decorated["props"].update(
        {
            "series": [{"name": "fake", "values": [999]}],
            "sourceText": "not consumed by Android",
            "legend": "not consumed by Android",
        }
    )
    source = "Revenue by month as a chart"
    baseline = render_artifact_quality_v5_4(
        root_spec({"chart": chart}, ["chart"]), source
    )
    mutated = render_artifact_quality_v5_4(
        root_spec({"chart": decorated}, ["chart"]), source
    )
    assert mutated.quality_0_1 == pytest.approx(
        baseline.quality_0_1, abs=1e-12
    )
