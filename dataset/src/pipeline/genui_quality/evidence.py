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
    # Keep the frozen v4 scorer constants unchanged while reporting the full
    # renderer inventory introduced by capability manifest v2.
    canonical_types=ALLOWED_TYPES | frozenset({"Alert", "Checklist"}),
    aliases={
        "stack": "Stack",
        "column": "Stack",
        "row": "Stack",
        "list": "List",
        "card": "Card",
        "text": "Text",
        "formula": "Formula",
        "code": "CodeBlock",
        "codeblock": "CodeBlock",
        "code_block": "CodeBlock",
        "pre": "CodeBlock",
        "preformatted": "CodeBlock",
        "console": "ConsoleLog",
        "consolelog": "ConsoleLog",
        "console_log": "ConsoleLog",
        "terminal": "ConsoleLog",
        "logoutput": "ConsoleLog",
        "log_output": "ConsoleLog",
        "email": "EmailPreview",
        "mail": "EmailPreview",
        "emailpreview": "EmailPreview",
        "email_preview": "EmailPreview",
        "messagepreview": "EmailPreview",
        "message_preview": "EmailPreview",
        "table": "Table",
        "chart": "Chart",
        "barchart": "Chart",
        "bar_chart": "Chart",
        "image": "Image",
        "icon": "Icon",
        "video": "Video",
        "audioplayer": "AudioPlayer",
        "audio": "AudioPlayer",
        "divider": "Divider",
        "button": "Button",
        "tabs": "Tabs",
        "tab": "Tabs",
        "modal": "Modal",
        "textfield": "TextField",
        "textinput": "TextField",
        "textbox": "TextField",
        "input": "TextField",
        "checkbox": "CheckBox",
        "check": "CheckBox",
        "choicepicker": "ChoicePicker",
        "picker": "ChoicePicker",
        "dropdown": "ChoicePicker",
        "select": "ChoicePicker",
        "slider": "Slider",
        "datetimeinput": "DateTimeInput",
        "datetimepicker": "DateTimeInput",
        "dateinput": "DateTimeInput",
        "datepicker": "DateTimeInput",
        "alert": "Alert",
        "notice": "Alert",
        "messagecard": "Alert",
        "message_card": "Alert",
        "checklist": "Checklist",
        "check_list": "Checklist",
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
        "Alert": ("message|text",),
        "Checklist": ("items",),
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
