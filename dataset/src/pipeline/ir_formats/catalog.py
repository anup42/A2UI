"""Pinned source definitions for the GenUICraft A2UI Express profile.

Generated schemas and catalogs are derived from this module. Keep protocol,
grammar, catalog, and renderer identities explicit so upstream changes cannot
silently alter production generation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

A2UI_UPSTREAM_REPOSITORY = "a2ui-project/a2ui"
A2UI_UPSTREAM_COMMIT = "2276f8cc702eaeac25ffb05be85797b2a1205c74"
A2UI_EXPRESS_GRAMMAR_GIT_BLOB_SHA = "4f2492ae4600598d8b10e68fcd9f4292529dd653"
A2UI_PROTOCOL_VERSION = "v1.0"
A2UI_EXPRESS_VERSION = "genuicraft-express-v1"
GENUICRAFT_CATALOG_ID = "https://genui.samsung.com/a2ui/catalogs/genuicraft-mobile/v1"

COMPONENTS: dict[str, dict[str, Any]] = json.loads(r'''{
  "Alert": {
    "aliases": [
      "Alert",
      "alert",
      "notice",
      "messagecard",
      "message_card"
    ],
    "positional": [
      "message",
      "title",
      "tone",
      "timestamp"
    ],
    "defaults": {},
    "consumedProps": [
      "message",
      "text",
      "title",
      "tone",
      "timestamp",
      "source",
      "icon"
    ],
    "container": false,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "AudioPlayer": {
    "aliases": [
      "AudioPlayer",
      "audioplayer"
    ],
    "positional": [
      "url",
      "description",
      "posterUrl",
      "title"
    ],
    "defaults": {},
    "consumedProps": [
      "url",
      "src",
      "source",
      "name",
      "poster",
      "posterUrl",
      "thumbnail",
      "thumbnailUrl",
      "description",
      "title"
    ],
    "container": false,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "Button": {
    "aliases": [
      "Button",
      "button"
    ],
    "positional": [
      "label",
      "variant",
      "icon"
    ],
    "defaults": {},
    "consumedProps": [
      "label",
      "text",
      "variant",
      "icon",
      "accessibilityLabel",
      "contentDescription",
      "onClickLabel",
      "actionLabel"
    ],
    "container": false,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "Card": {
    "aliases": [
      "Card",
      "card"
    ],
    "positional": [
      "children",
      "title",
      "subtitle",
      "tone"
    ],
    "defaults": {},
    "consumedProps": [
      "title",
      "subtitle",
      "tone",
      "padding",
      "paddingHorizontal",
      "paddingVertical",
      "margin",
      "marginHorizontal",
      "marginVertical"
    ],
    "container": true,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "Chart": {
    "aliases": [
      "Chart",
      "chart",
      "barchart",
      "bar_chart"
    ],
    "positional": [
      "chartType",
      "columns",
      "statePath",
      "rows",
      "title",
      "subtitle"
    ],
    "defaults": {
      "chartType": "bar"
    },
    "consumedProps": [
      "chartType",
      "title",
      "subtitle",
      "yLabel",
      "columns",
      "rows",
      "statePath",
      "rowsPath",
      "dataPath",
      "xKey",
      "yKey"
    ],
    "container": false,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "CheckBox": {
    "aliases": [
      "CheckBox",
      "checkbox"
    ],
    "positional": [
      "label",
      "value",
      "statePath"
    ],
    "defaults": {},
    "consumedProps": [
      "label",
      "value",
      "statePath",
      "accessibilityLabel",
      "contentDescription"
    ],
    "container": false,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "Checklist": {
    "aliases": [
      "Checklist",
      "checklist",
      "check_list"
    ],
    "positional": [
      "items",
      "title",
      "disclaimer",
      "source"
    ],
    "defaults": {},
    "consumedProps": [
      "title",
      "items",
      "disclaimer",
      "source"
    ],
    "container": false,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "ChoicePicker": {
    "aliases": [
      "ChoicePicker",
      "choicepicker"
    ],
    "positional": [
      "label",
      "options",
      "value",
      "statePath"
    ],
    "defaults": {},
    "consumedProps": [
      "label",
      "value",
      "statePath",
      "options",
      "accessibilityLabel",
      "contentDescription"
    ],
    "container": false,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "CodeBlock": {
    "aliases": [
      "CodeBlock",
      "codeblock",
      "code",
      "code_block",
      "pre",
      "preformatted"
    ],
    "positional": [
      "code",
      "language",
      "title"
    ],
    "defaults": {},
    "consumedProps": [
      "code",
      "language",
      "title"
    ],
    "container": false,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "ConsoleLog": {
    "aliases": [
      "ConsoleLog",
      "consolelog",
      "console",
      "console_log",
      "terminal",
      "logoutput",
      "log_output"
    ],
    "positional": [
      "code",
      "language",
      "title"
    ],
    "defaults": {},
    "consumedProps": [
      "code",
      "language",
      "title"
    ],
    "container": false,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "DateTimeInput": {
    "aliases": [
      "DateTimeInput",
      "datetimeinput"
    ],
    "positional": [
      "label",
      "value",
      "mode",
      "placeholder",
      "statePath"
    ],
    "defaults": {},
    "consumedProps": [
      "label",
      "value",
      "statePath",
      "mode",
      "placeholder",
      "accessibilityLabel",
      "contentDescription"
    ],
    "container": false,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "Divider": {
    "aliases": [
      "Divider",
      "divider"
    ],
    "positional": [],
    "defaults": {},
    "consumedProps": [],
    "container": false,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "EmailPreview": {
    "aliases": [
      "EmailPreview",
      "emailpreview",
      "email_preview"
    ],
    "positional": [
      "subject",
      "body",
      "from",
      "to",
      "date",
      "title"
    ],
    "defaults": {},
    "consumedProps": [
      "from",
      "to",
      "cc",
      "bcc",
      "subject",
      "body",
      "date",
      "timestamp",
      "attachments",
      "title"
    ],
    "container": false,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "Formula": {
    "aliases": [
      "Formula",
      "formula"
    ],
    "positional": [
      "latex",
      "title",
      "result",
      "display"
    ],
    "defaults": {},
    "consumedProps": [
      "latex",
      "text",
      "title",
      "subtitle",
      "result",
      "display"
    ],
    "container": false,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "Icon": {
    "aliases": [
      "Icon",
      "icon"
    ],
    "positional": [
      "name",
      "size",
      "tint"
    ],
    "defaults": {},
    "consumedProps": [
      "name",
      "icon",
      "source",
      "url",
      "src",
      "size",
      "iconSize",
      "tint",
      "accessibilityLabel",
      "contentDescription",
      "decorative"
    ],
    "container": false,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "Image": {
    "aliases": [
      "Image",
      "image"
    ],
    "positional": [
      "url",
      "alt",
      "fit",
      "width",
      "height"
    ],
    "defaults": {
      "fit": "fill"
    },
    "consumedProps": [
      "url",
      "src",
      "source",
      "name",
      "fit",
      "contentScale",
      "alt",
      "fallbackUrl",
      "height",
      "width",
      "accessibilityLabel",
      "contentDescription",
      "decorative"
    ],
    "container": false,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "List": {
    "aliases": [
      "List",
      "list"
    ],
    "positional": [
      "children",
      "items"
    ],
    "defaults": {},
    "consumedProps": [
      "items"
    ],
    "container": true,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "Modal": {
    "aliases": [
      "Modal",
      "modal"
    ],
    "positional": [
      "trigger",
      "content",
      "title"
    ],
    "defaults": {},
    "consumedProps": [
      "trigger",
      "content",
      "title"
    ],
    "container": false,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "Slider": {
    "aliases": [
      "Slider",
      "slider"
    ],
    "positional": [
      "label",
      "value",
      "min",
      "max",
      "step",
      "statePath"
    ],
    "defaults": {},
    "consumedProps": [
      "label",
      "value",
      "statePath",
      "min",
      "max",
      "step",
      "accessibilityLabel",
      "contentDescription"
    ],
    "container": false,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "Stack": {
    "aliases": [
      "Stack",
      "stack"
    ],
    "positional": [
      "children",
      "direction",
      "gap",
      "align",
      "justify",
      "wrap"
    ],
    "defaults": {
      "direction": "vertical",
      "gap": "md",
      "align": "start",
      "justify": "start",
      "wrap": "nowrap"
    },
    "consumedProps": [
      "direction",
      "gap",
      "spacing",
      "space",
      "align",
      "justify",
      "wrap",
      "padding",
      "paddingHorizontal",
      "paddingVertical",
      "margin",
      "marginHorizontal",
      "marginVertical"
    ],
    "container": true,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "Table": {
    "aliases": [
      "Table",
      "table"
    ],
    "positional": [
      "columns",
      "statePath",
      "rows",
      "title",
      "domain",
      "preferredPresentation"
    ],
    "defaults": {},
    "consumedProps": [
      "title",
      "domain",
      "preferredPresentation",
      "presentation",
      "columns",
      "rows",
      "statePath",
      "rowsPath",
      "dataPath",
      "primaryColumn",
      "highlightColumns",
      "numericColumns",
      "entityMedia"
    ],
    "container": false,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "Tabs": {
    "aliases": [
      "Tabs",
      "tabs"
    ],
    "positional": [
      "tabs",
      "activeTabId"
    ],
    "defaults": {},
    "consumedProps": [
      "tabs",
      "activeTabId"
    ],
    "container": false,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "Text": {
    "aliases": [
      "Text",
      "text"
    ],
    "positional": [
      "text",
      "variant"
    ],
    "defaults": {
      "variant": "body"
    },
    "consumedProps": [
      "text",
      "variant",
      "heading",
      "accessibilityLabel",
      "contentDescription",
      "decorative"
    ],
    "container": false,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "TextField": {
    "aliases": [
      "TextField",
      "textfield"
    ],
    "positional": [
      "label",
      "value",
      "statePath",
      "placeholder"
    ],
    "defaults": {},
    "consumedProps": [
      "label",
      "value",
      "statePath",
      "placeholder",
      "accessibilityLabel",
      "contentDescription"
    ],
    "container": false,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  },
  "Video": {
    "aliases": [
      "Video",
      "video"
    ],
    "positional": [
      "url",
      "posterUrl",
      "description",
      "title"
    ],
    "defaults": {},
    "consumedProps": [
      "url",
      "src",
      "source",
      "name",
      "poster",
      "posterUrl",
      "thumbnail",
      "thumbnailUrl",
      "description",
      "title"
    ],
    "container": false,
    "allowAdditionalProps": true,
    "metadata": [
      "children",
      "repeat",
      "visible",
      "on",
      "watch"
    ]
  }
}''')
ACTIONS: dict[str, dict[str, Any]] = json.loads(r'''{
  "openUrl": {
    "positional": [
      "url"
    ],
    "allowAdditionalParams": false
  },
  "setState": {
    "positional": [
      "statePath",
      "value"
    ],
    "allowAdditionalParams": false
  },
  "pushState": {
    "positional": [
      "statePath",
      "value",
      "clearStatePath"
    ],
    "allowAdditionalParams": false
  },
  "removeState": {
    "positional": [
      "statePath",
      "index"
    ],
    "allowAdditionalParams": false
  },
  "validateForm": {
    "positional": [
      "statePath",
      "resultStatePath"
    ],
    "allowAdditionalParams": false
  },
  "emitEvent": {
    "positional": [
      "name",
      "context",
      "wantResponse",
      "responsePath"
    ],
    "allowAdditionalParams": false
  }
}''')

# Express accepts Row/Column as catalog aliases while the canonical renderer
# continues to receive Stack plus an explicit direction.
for _alias, _direction in (("Row", "horizontal"), ("Column", "vertical")):
    _descriptor = deepcopy(COMPONENTS["Stack"])
    _descriptor["aliases"] = [_alias, _alias.lower()]
    _descriptor["positional"] = [
        key for key in _descriptor.get("positional", ()) if key != "direction"
    ]
    _descriptor["defaults"] = {
        key: value
        for key, value in _descriptor.get("defaults", {}).items()
        if key != "direction"
    }
    _descriptor["canonicalType"] = "Stack"
    _descriptor["implicitProps"] = {"direction": _direction}
    COMPONENTS[_alias] = _descriptor

ACTION_POSITIONAL: dict[str, tuple[str, ...]] = {
    name: tuple(str(value) for value in descriptor.get("positional", ()))
    for name, descriptor in ACTIONS.items()
}


def catalog_payload() -> dict[str, Any]:
    return {
        "catalogId": GENUICRAFT_CATALOG_ID,
        "protocolVersion": A2UI_PROTOCOL_VERSION,
        "expressVersion": A2UI_EXPRESS_VERSION,
        "upstream": {
            "repository": A2UI_UPSTREAM_REPOSITORY,
            "commit": A2UI_UPSTREAM_COMMIT,
            "grammarGitBlobSha": A2UI_EXPRESS_GRAMMAR_GIT_BLOB_SHA,
        },
        "components": deepcopy(COMPONENTS),
        "actions": deepcopy(ACTIONS),
    }


def catalog_identity_hash() -> str:
    encoded = json.dumps(
        catalog_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "A2UI_UPSTREAM_REPOSITORY",
    "A2UI_UPSTREAM_COMMIT",
    "A2UI_EXPRESS_GRAMMAR_GIT_BLOB_SHA",
    "A2UI_PROTOCOL_VERSION",
    "A2UI_EXPRESS_VERSION",
    "GENUICRAFT_CATALOG_ID",
    "COMPONENTS",
    "ACTIONS",
    "ACTION_POSITIONAL",
    "catalog_payload",
    "catalog_identity_hash",
]
