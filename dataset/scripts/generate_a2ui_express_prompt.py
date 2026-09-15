#!/usr/bin/env python3
"""Generate the one model-facing A2UI Express contract.

The contract is derived from the pinned grammar, strict catalog, Express
inference profile, and shared quality policy. Dataset and shared Android assets
are byte-identical so syntax/signature drift is testable. The on-device Gemma
prompt is intentionally specialized for the smaller mobile runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GRAMMAR = ROOT / "specification" / "inference_formats" / "express" / "Express.g4"
CATALOG = ROOT / "dataset" / "schema" / "genuicraft_a2ui_catalog_v1.json"
PROFILE = ROOT / "dataset" / "schema" / "genuicraft_a2ui_express_profile_v1.json"
QUALITY = ROOT / "dataset" / "prompts" / "a2ui_express_quality_policy_v1.md"
OUTPUTS = (
    ROOT / "dataset" / "prompts" / "genui_gen.md",
    ROOT / "dataset" / "prompts" / "genui_gen_mobile_a2ui_express_v1.md",
    ROOT / "android" / "app" / "src" / "main" / "assets" / "pipeline_prompts" / "genui_gen.md",
    ROOT / "android" / "app" / "src" / "main" / "assets" / "pipeline_prompts" / "genui_gen_a2ui_express_v1.md",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signature(name: str, descriptor: dict) -> str:
    positional = descriptor.get("positional") or []
    if not positional:
        return f"{name}()"
    return f"{name}({', '.join(str(item) for item in positional)})"


def _property_type(schema: dict) -> str:
    if "enum" in schema:
        return "|".join(json.dumps(value, ensure_ascii=False) for value in schema["enum"])
    if "oneOf" in schema:
        literals = [item for item in schema["oneOf"] if isinstance(item, dict) and "enum" in item]
        if literals:
            return _property_type(literals[0])
    ref = schema.get("$ref", "").rsplit("/", 1)[-1]
    if ref:
        return {"dynamicString": "string", "dynamicNumber": "number", "dynamicBoolean": "boolean",
                "dynamicArray": "array", "dynamicObject": "object", "dynamicValue": "JSON value"}.get(ref, ref)
    kind = schema.get("type", "JSON value")
    if kind == "array":
        return "array<" + _property_type(schema.get("items", {})) + ">"
    return str(kind)


def _typed_properties(name: str, descriptor: dict, catalog: dict) -> str:
    catalog_components = catalog.get("components", {})
    canonical = catalog_components.get(name) or catalog_components.get(descriptor.get("aliasFor")) or {}
    properties = canonical.get("schema", {}).get("properties", {})
    names = list(dict.fromkeys((descriptor.get("positional") or []) + (canonical.get("consumedProps") or [])))
    entries = [f"{key}: " + ("component references" if key in {"children", "trigger", "content"} else _property_type(properties[key]))
               for key in names if key in properties]
    return "; ".join(entries) or "no properties"


def render_prompt() -> str:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    components = profile.get("components") or catalog.get("components") or {}
    actions = profile.get("actions") or {}
    quality = QUALITY.read_text(encoding="utf-8").strip()
    component_lines = "\n".join(
        f"- {_signature(str(name), descriptor)}\n  Types: {_typed_properties(str(name), descriptor, catalog)}"
        for name, descriptor in sorted(components.items())
        if isinstance(descriptor, dict)
    )
    action_lines = "\n".join(
        f"- {_signature(str(name), descriptor)}"
        for name, descriptor in sorted(actions.items())
        if isinstance(descriptor, dict) and not descriptor.get("aliasFor")
    )
    source_hashes = (
        f"grammar={_sha(GRAMMAR)} catalog={_sha(CATALOG)} "
        f"profile={_sha(PROFILE)} quality={_sha(QUALITY)}"
    )
    return f"""# A2UI Express v1 generated model contract

<!-- Generated from pinned grammar/catalog/profile/quality policy: {source_hashes} -->

You convert the supplied response into one rich, lossless GenUICraft A2UI
Express v1 program. A2UI Express is an assignment DSL, not JSON, HTML, JSX,
CSS, or a FlatSpec object. Preserve the complete canonical UI semantics;
syntax optimization must never remove meaningful UI content or interactions.

Response:
{{response_text}}

## Strict output contract

- Return exactly one `<a2ui>...</a2ui>` block, with no prose, JSON, markdown
  fences, or trailing content.
- Assign the root component to reserved variable `root`; every reference must
  resolve, and every useful assignment must be reachable from `root`.
- Use one assignment per line. Child lists contain assigned identifiers or
  inline calls, never bare component type names.
- Use positional arguments only while unambiguous; after a named argument is
  used, use named arguments. `_` may skip only an optional final positional
  slot. Omit trailing catalog defaults when semantics are unchanged.
- Use explicit named properties, children, repeat, visible, watch, and action
  arguments. Opaque `_props`, `_children`, `_repeat`, `_visible`, `_on`, and
  `_watch` bags are forbidden for new output.
- Event values must be action calls such as `Event("name",{{}})` or
  `openUrl("https://...")`, never quoted URLs/event names by themselves.
- Use `$={{...}}` or `$/path=value` for state and valid data bindings.
- Reject the temptation to invent URLs, paths, values, or filler components.
- URL/icon/image placeholders are STRING LITERALS, including their brackets:
  `Icon(url="[ICON_URL_1]")`, `Image("[IMAGE_URL_1]","Source image")`,
  `openUrl("[ACTION_URL_1]")`. Never use bare `ICON_URL_1` or `[ICON_URL_1]`.
  Copy the exact supplied token; names here are examples, not extra assets.
- Match the catalog types below. Use `wrap="wrap"` or `wrap="nowrap"`, never a
  boolean; gap is an enum string; width/height/padding are numbers.
- Typed visibility is `visible=true`, `visible=false`, or
  `visible={{path:"/consent_agreed"}}`. A quoted expression is a string and is
  invalid for boolean visibility. State declarations and bindings are different.
- A complete EmailPreview, compact Table, or focused control may use fewer than
  five components. Preserve its required source content; never pad node counts.
- `List(items=["First step","Second step"])` holds text rows. Text/link maps
  may use `text`, `title`, `label` and `url`, `href`, `link` or `source`.
  Put rich components in `children=[...]`; never put `Text(...)` calls in items.

## Pinned catalog signatures

Types describe literal values; dynamic types also accept a typed binding object.
Enums must use exactly a listed spelling. Optional omitted arguments retain defaults.

{component_lines}

## Pinned actions

{action_lines}

## Shared quality policy

{quality}

## Syntax example (do not copy its facts)

<a2ui>
root=Column([title,details,action],gap="md")
title=Text("Result","h2")
details=Table(["Detail","Value"],rows=[["Status","Ready"]],title="Details",domain="status",preferredPresentation="table")
action=Button("Continue","primary",onPress=Event("continue",{{}},true,"/result"))
</a2ui>

The pipeline supplies the response in the user message and restores approved
URL/local-asset placeholders only after strict parsing and compilation.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = render_prompt().encode("utf-8")
    different: list[str] = []
    for path in OUTPUTS:
        existing = path.read_bytes() if path.exists() else None
        if existing != content:
            different.append(str(path.relative_to(ROOT)).replace("\\", "/"))
            if not args.check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
    if different and args.check:
        print("A2UI Express prompt drift:", *different, sep="\n- ")
        return 1
    print(("A2UI Express prompt verified" if args.check else "A2UI Express prompt generated") + f" ({len(OUTPUTS)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
