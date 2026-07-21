"""Renderer-semantic contract inventory and reachable evidence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ._core import (
    ALLOWED_ACTIONS,
    ALLOWED_TYPES,
    OutputAction,
    OutputEvidence,
    OutputTable,
    collect_output_evidence,
)


@dataclass(frozen=True)
class RendererSemanticContract:
    canonical_types: frozenset[str]
    aliases: Mapping[str, str]
    required_props: Mapping[str, tuple[str, ...]]
    actions: frozenset[str]
    state_pointer_props: tuple[str, ...]
    accessibility_props: tuple[str, ...]


RENDERER_SEMANTICS = RendererSemanticContract(
    canonical_types=ALLOWED_TYPES,
    aliases={
        "row": "Row",
        "column": "Column",
        "code": "CodeBlock",
        "code_block": "CodeBlock",
        "pre": "CodeBlock",
        "preformatted": "CodeBlock",
        "console": "ConsoleLog",
        "console_log": "ConsoleLog",
        "terminal": "ConsoleLog",
        "email_preview": "EmailPreview",
        "barchart": "Chart",
        "bar_chart": "Chart",
        "audio": "AudioPlayer",
    },
    required_props={
        "Text": ("text",),
        "Formula": ("text|latex",),
        "CodeBlock": ("code",),
        "ConsoleLog": ("code",),
        "EmailPreview": ("subject|title", "body"),
        "Table": ("columns", "statePath|rows"),
        "Chart": ("columns", "statePath|rows", "xKey", "yKey"),
        "Image": ("url",),
        "Icon": ("name",),
        "Video": ("url",),
        "AudioPlayer": ("url",),
        "Button": ("label", "on.press"),
        "TextField": ("label",),
        "CheckBox": ("label", "value"),
        "ChoicePicker": ("label", "options"),
        "Slider": ("min", "max", "value"),
        "DateTimeInput": ("value",),
    },
    actions=ALLOWED_ACTIONS,
    state_pointer_props=("statePath", "path", "watch", "repeat.statePath"),
    accessibility_props=(
        "label",
        "alt",
        "title",
        "accessibilityLabel",
        "contentDescription",
        "accessibility.label",
        "accessibility.decorative",
    ),
)


__all__ = [
    "OutputAction",
    "OutputEvidence",
    "OutputTable",
    "RENDERER_SEMANTICS",
    "RendererSemanticContract",
    "collect_output_evidence",
]
