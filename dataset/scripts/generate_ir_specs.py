#!/usr/bin/env python3
"""Regenerate pinned A2UI Express catalog, wire schema, grammar, and manifest."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATASET_SRC = ROOT / "dataset" / "src"
if str(DATASET_SRC) not in sys.path:
    sys.path.insert(0, str(DATASET_SRC))

from pipeline.ir_formats.catalog import (  # noqa: E402
    ACTIONS,
    ACTION_POSITIONAL,
    A2UI_EXPRESS_GRAMMAR_GIT_BLOB_SHA,
    A2UI_EXPRESS_VERSION,
    A2UI_PROTOCOL_VERSION,
    A2UI_UPSTREAM_COMMIT,
    A2UI_UPSTREAM_REPOSITORY,
    COMPONENTS,
    GENUICRAFT_CATALOG_ID,
    catalog_identity_hash,
    catalog_payload,
)

SCHEMA_DIR = ROOT / "dataset" / "schema"
GRAMMAR_PATH = ROOT / "specification" / "inference_formats" / "express" / "Express.g4"
PROFILE_PATH = SCHEMA_DIR / "genuicraft_a2ui_express_profile_v1.json"
CANONICAL_GRAPH_PATH = SCHEMA_DIR / "canonical_ui_graph_v1.schema.json"
WIRE_SCHEMA_PATH = SCHEMA_DIR / "genuicraft_a2ui_v1_wire.schema.json"
CATALOG_PATH = SCHEMA_DIR / "genuicraft_a2ui_catalog_v1.json"
PROMPT_PATH = ROOT / "dataset" / "prompts" / "genui_gen_mobile_a2ui_express_v1.md"
PROMPT_MIRROR_PATH = ROOT / "dataset" / "prompts" / "genui_gen.md"
PROMPT_GENERATOR_PATH = ROOT / "dataset" / "scripts" / "generate_a2ui_express_prompt.py"
QUALITY_POLICY_PATH = ROOT / "dataset" / "prompts" / "a2ui_express_quality_policy_v1.md"
ANDROID_PROMPT_PATHS = (
    ROOT / "android" / "app" / "src" / "main" / "assets" / "pipeline_prompts" / "genui_gen.md",
    ROOT / "android" / "app" / "src" / "main" / "assets" / "pipeline_prompts" / "genui_gen_a2ui_express_v1.md",
    ROOT / "android" / "app" / "src" / "main" / "assets" / "pipeline_prompts" / "genui_gen_gemma_litert.md",
)
PY_COMPILER_PATH = ROOT / "dataset" / "src" / "pipeline" / "ir_formats" / "express.py"
PY_ACTIVE_BOUNDARY_PATH = ROOT / "dataset" / "src" / "pipeline" / "ir_formats" / "active.py"
PY_WIRE_COMPILER_PATH = ROOT / "dataset" / "src" / "pipeline" / "ir_formats" / "a2ui_wire.py"
KOTLIN_COMPILER_PATH = ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "samsung" / "genuicraft" / "pipeline" / "A2uiExpressCodec.kt"
KOTLIN_WIRE_COMPILER_PATH = ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "samsung" / "genuicraft" / "pipeline" / "A2uiWireCodec.kt"
MIGRATION_PATH = ROOT / "dataset" / "scripts" / "migrate_legacy_dataset_to_a2ui_express.py"
REFERENCE_INVENTORY_PATH = ROOT / "dataset" / "tests" / "fixtures" / "flat_spec_reference_inventory_v1.json"
CONFORMANCE_CORPUS_PATH = ROOT / "dataset" / "tests" / "fixtures" / "a2ui_express_conformance_v1.json"

PINNED_GRAMMAR = r'''/**
 * ANTLR4 grammar for the A2UI Express language.
 * Pinned from a2ui-project/a2ui commit
 * 2276f8cc702eaeac25ffb05be85797b2a1205c74.
 * Upstream Git blob SHA: 4f2492ae4600598d8b10e68fcd9f4292529dd653.
 */
grammar Express;

program : statement* EOF ;
statement : assignment | expression ;
assignment : (identifier | path) '=' expression ;
expression : array | map | path | check | call | variable | literal ;
array : '[' (expression (',' expression)* ','?)? ']' ;
map : '{' (map_entry (',' map_entry)* ','?)? '}' ;
map_entry : (identifier | string) ':' expression ;
path : PATH ;
check : CHECK ('(' (expression (',' expression)* ','?)? ')')? ;
call : identifier '(' (arg (',' arg)* ','?)? ')' ;
arg : named_arg | expression ;
named_arg : identifier '=' expression ;
variable : '_' | identifier ;
literal : string | NUMBER | BOOLEAN | 'null' ;
identifier : IDENTIFIER ;
string : RAW_TRIPLE_STRING | TRIPLE_STRING | RAW_STRING | STANDARD_STRING ;

RAW_TRIPLE_STRING : [rR] '"""' .*? '"""' ;
TRIPLE_STRING     : '"""' ( '\\' . | ~'\\' )*? '"""' ;
RAW_STRING        : [rR] '"' ~[\r\n"]* '"' ;
STANDARD_STRING   : '"' ( '\\' . | ~'\\' )*? '"' ;
PATH : '$' [a-zA-Z0-9_/]* ;
CHECK : '?' [a-zA-Z_] [a-zA-Z0-9_]* ;
NUMBER : '-'? [0-9]+ ('.' [0-9]+)? ;
BOOLEAN : 'true' | 'false' ;
IDENTIFIER : [a-zA-Z_] [a-zA-Z0-9_]* ;
COMMENT : ( '#' | '//' ) ~[\r\n]* -> skip ;
BLOCK_COMMENT : '/*' .*? '*/' -> skip ;
SEMICOLON : ';' -> skip ;
WS : [ \t\r\n]+ -> skip ;
'''


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def catalog_document() -> dict[str, Any]:
    """Load the catalog and apply the generated strict property contract.

    The checked-in catalog is the reviewable source of component names and
    renderer property inventory.  The generated type definitions below keep
    the standard catalog closed while still allowing the pinned A2UI dynamic
    binding/function-call forms used by the native renderer.
    """
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    catalog.setdefault("$schema", "https://json-schema.org/draft/2020-12/schema")
    catalog.setdefault("$id", str(catalog.get("catalogId") or "https://genui.samsung.com/a2ui/catalogs/genuicraft-mobile/v1"))
    catalog["$defs"] = _catalog_defs()
    components = catalog.get("components")
    if isinstance(components, dict):
        for name, descriptor in components.items():
            if not isinstance(descriptor, dict):
                continue
            descriptor["allowAdditionalProps"] = False
            schema = descriptor.get("schema")
            if not isinstance(schema, dict):
                continue
            schema["additionalProperties"] = False
            properties = schema.setdefault("properties", {})
            for property_name in list(properties):
                if property_name == "visible":
                    properties[property_name] = {"$ref": "#/$defs/dynamicBoolean"}
                    continue
                if property_name == "repeat":
                    properties[property_name] = {"$ref": "#/$defs/repeat"}
                    continue
                if property_name in {"on", "watch"}:
                    properties[property_name] = {"$ref": "#/$defs/actionMap"}
                    continue
                if property_name in {"id", "component", "children"}:
                    continue
                properties[property_name] = _property_schema(str(property_name), str(name))
    return catalog


def _binding_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["path"],
        "properties": {
            "path": {
                "type": "string",
                "pattern": r"^/.*$",
            }
        },
    }


def _function_call_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["call"],
        "properties": {
            "call": {"type": "string", "minLength": 1},
            "args": {"type": "object"},
            "returnType": {
                "type": "string",
                "enum": ["string", "number", "boolean", "array", "object", "any", "void"],
            },
        },
    }


def _catalog_defs() -> dict[str, Any]:
    return {
        "dataBinding": _binding_schema(),
        "functionCall": _function_call_schema(),
        "action": {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "params"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["openUrl", "setState", "pushState", "removeState", "validateForm", "emitEvent"],
                },
                "params": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "url": {"type": "string"},
                        "statePath": {"type": "string"},
                        "value": {},
                        "index": {"type": ["integer", "string"]},
                        "clearStatePath": {"type": "string"},
                        "resultStatePath": {"type": "string"},
                        "name": {"type": "string"},
                        "context": {"type": "object"},
                        "wantResponse": {"type": "boolean"},
                        "responsePath": {"type": "string"},
                    },
                },
            },
        },
        "actionMap": {
            "type": "object",
            "additionalProperties": {"$ref": "#/$defs/action"},
        },
        "repeat": {
            "type": "object",
            "additionalProperties": False,
            "required": ["statePath"],
            "properties": {
                "statePath": {"type": "string", "pattern": "^/.*"},
                "key": {"type": "string", "minLength": 1},
                "template": {"type": "string", "minLength": 1},
                "itemTemplate": {"type": "string", "minLength": 1},
                "child": {"type": "string", "minLength": 1},
            },
        },
        "dynamicString": {
            "oneOf": [
                {"type": "string"},
                {"$ref": "#/$defs/dataBinding"},
                {"$ref": "#/$defs/functionCall"},
            ]
        },
        "dynamicNumber": {
            "oneOf": [
                {"type": "number"},
                {"type": "string"},
                {"$ref": "#/$defs/dataBinding"},
                {"$ref": "#/$defs/functionCall"},
            ]
        },
        "dynamicBoolean": {
            "oneOf": [
                {"type": "boolean"},
                {"$ref": "#/$defs/dataBinding"},
                {"$ref": "#/$defs/functionCall"},
            ]
        },
        "dynamicArray": {
            "oneOf": [
                {"type": "array"},
                {"$ref": "#/$defs/dataBinding"},
                {"$ref": "#/$defs/functionCall"},
            ]
        },
        "dynamicObject": {
            # Data bindings and function calls are objects too, so they
            # intentionally overlap the generic object branch.
            "anyOf": [
                {"type": "object"},
                {"$ref": "#/$defs/dataBinding"},
                {"$ref": "#/$defs/functionCall"},
            ]
        },
        "dynamicValue": {
            # A dataBinding/functionCall is also a valid object value; using
            # oneOf incorrectly rejects otherwise valid binding payloads.
            "anyOf": [
                {"type": ["string", "number", "boolean", "null"]},
                {"type": "array"},
                {"type": "object"},
                {"$ref": "#/$defs/dataBinding"},
                {"$ref": "#/$defs/functionCall"},
            ]
        },
    }


_STRING_PROPERTIES = {
    "accessibilityLabel", "actionLabel", "activeTabId", "align", "alt", "ariaLabel",
    "contentDescription", "contentScale", "date", "description", "domain", "fit", "from",
    "gap", "heading", "height", "icon", "iconSize", "justify", "label", "language", "latex",
    "message", "mode", "name", "placeholder", "poster", "posterUrl", "preferredPresentation",
    "presentation", "primaryColumn", "role", "semanticRole", "size", "source", "src", "statePath",
    "subtitle", "subject", "template", "text", "thumbnail", "thumbnailUrl", "timestamp", "title",
    "to", "tone", "url", "variant", "wrap", "xKey", "yKey", "yLabel",
}
_NUMBER_PROPERTIES = {"aspectRatio", "flex", "height", "iconSize", "max", "min", "padding", "paddingHorizontal", "paddingVertical", "margin", "marginHorizontal", "marginVertical", "size", "step", "width"}
_BOOLEAN_PROPERTIES = {"decorative", "display"}
_ARRAY_PROPERTIES = {"attachments", "columns", "entityMedia", "highlightColumns", "items", "numericColumns", "options", "rows", "tabs"}
_OBJECT_PROPERTIES = {"accessibility", "checks", "data", "style"}
_ENUM_PROPERTIES = {
    "direction": ("vertical", "horizontal"),
    "fit": ("contain", "cover", "fill", "none", "scale-down"),
    "gap": ("none", "sm", "md", "lg", "xl"),
    "align": ("start", "center", "end", "stretch"),
    "justify": ("start", "center", "end", "stretch", "spaceAround", "spaceBetween", "spaceEvenly"),
    "wrap": ("nowrap", "wrap"),
}


def _property_schema(name: str, component: str | None = None) -> dict[str, Any]:
    """Return a typed dynamic property schema for a catalog field."""
    del component  # reserved for component-specific enum profiles
    if name in _ENUM_PROPERTIES:
        return {
            "oneOf": [
                {"type": "string", "enum": list(_ENUM_PROPERTIES[name])},
                {"$ref": "#/$defs/dataBinding"},
                {"$ref": "#/$defs/functionCall"},
            ]
        }
    if name in _ARRAY_PROPERTIES:
        ref = "dynamicArray"
    elif name in _OBJECT_PROPERTIES:
        ref = "dynamicObject"
    elif name in _NUMBER_PROPERTIES:
        ref = "dynamicNumber"
    elif name in _BOOLEAN_PROPERTIES:
        ref = "dynamicBoolean"
    elif name in _STRING_PROPERTIES:
        ref = "dynamicString"
    else:
        ref = "dynamicValue"
    return {"$ref": f"#/$defs/{ref}"}


def wire_schema() -> dict[str, Any]:
    # The wire schema is generated from the strict GenUICraft catalog while
    # keeping the message envelopes identical to the vendored upstream v0.9
    # server-to-client contract.  The catalog is the only project-specific
    # part; no FlatSpec-shaped envelope is accepted here.
    catalog = catalog_document()
    components: dict[str, Any] = {}
    for name, descriptor in catalog.get("components", {}).items():
        schema = deepcopy(descriptor.get("schema"))
        if not isinstance(schema, dict):
            continue
        properties = schema.setdefault("properties", {})
        if "children" in properties:
            properties["children"] = {"$ref": "#/$defs/childList"}
        # Repeat metadata is lowered to the standard ChildList object.  The
        # remaining renderer metadata is retained as catalog-defined fields,
        # but action values use the upstream event/functionCall shape.
        properties.pop("repeat", None)
        for key in ("on", "watch"):
            if key in properties:
                properties[key] = {
                    "type": "object",
                    "additionalProperties": {"$ref": "#/$defs/action"},
                }
        components[str(name)] = schema

    component_refs = [
        {"$ref": f"#/$defs/components/{name}"}
        for name in sorted(components)
    ]
    message_refs = [
        {"$ref": "#/$defs/createSurface"},
        {"$ref": "#/$defs/updateComponents"},
        {"$ref": "#/$defs/updateDataModel"},
        {"$ref": "#/$defs/deleteSurface"},
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://a2ui.org/specification/v0_9/server_to_client.json",
        "title": "GenUICraft A2UI v0.9 Server-to-Client Messages",
        "description": (
            "The pinned upstream A2UI v0.9 message envelope with the strict "
            "GenUICraft catalog component definitions."
        ),
        "oneOf": message_refs + [
            {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/$defs/message"},
            }
        ],
        "$defs": {
            **deepcopy(catalog.get("$defs", {})),
            "message": {"oneOf": message_refs},
            "childList": {
                "oneOf": [
                    {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["componentId", "path"],
                        "properties": {
                            "componentId": {"type": "string", "minLength": 1},
                            "path": {"type": "string", "minLength": 1},
                            "key": {"type": "string", "minLength": 1},
                        },
                    },
                ]
            },
            "action": {
                "oneOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["event"],
                        "properties": {
                            "event": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["name"],
                                "properties": {
                                    "name": {"type": "string", "minLength": 1},
                                    "context": {"type": "object"},
                                    "wantResponse": {"type": "boolean"},
                                    "responsePath": {"type": "string"},
                                },
                            }
                        },
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["functionCall"],
                        "properties": {
                            "functionCall": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["call"],
                                "properties": {
                                    "call": {"type": "string", "minLength": 1},
                                    "args": {"type": "object"},
                                    "returnType": {"type": "string"},
                                },
                            }
                        },
                    },
                ]
            },
            "components": components,
            "createSurface": {
                "type": "object",
                "additionalProperties": False,
                "required": ["version", "createSurface"],
                "properties": {
                    "version": {"const": A2UI_PROTOCOL_VERSION},
                    "createSurface": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["surfaceId", "catalogId"],
                        "properties": {
                            "surfaceId": {"type": "string", "minLength": 1},
                            "catalogId": {
                                "const": GENUICRAFT_CATALOG_ID,
                                "type": "string",
                            },
                            "theme": {"type": "object"},
                            "sendDataModel": {"type": "boolean"},
                        },
                    },
                },
            },
            "updateComponents": {
                "type": "object",
                "additionalProperties": False,
                "required": ["version", "updateComponents"],
                "properties": {
                    "version": {"const": A2UI_PROTOCOL_VERSION},
                    "updateComponents": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["surfaceId", "components"],
                        "properties": {
                            "surfaceId": {"type": "string", "minLength": 1},
                            "components": {
                                "type": "array",
                                "minItems": 1,
                                "contains": {
                                    "type": "object",
                                    "required": ["id"],
                                    "properties": {"id": {"const": "root"}},
                                },
                                "items": {"$ref": "#/$defs/anyComponent"},
                            },
                        },
                    },
                },
            },
            "updateDataModel": {
                "type": "object",
                "additionalProperties": False,
                "required": ["version", "updateDataModel"],
                "properties": {
                    "version": {"const": A2UI_PROTOCOL_VERSION},
                    "updateDataModel": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["surfaceId"],
                        "properties": {
                            "surfaceId": {"type": "string", "minLength": 1},
                            "path": {"type": "string"},
                            "value": {},
                        },
                    },
                },
            },
            "deleteSurface": {
                "type": "object",
                "additionalProperties": False,
                "required": ["version", "deleteSurface"],
                "properties": {
                    "version": {"const": A2UI_PROTOCOL_VERSION},
                    "deleteSurface": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["surfaceId"],
                        "properties": {
                            "surfaceId": {"type": "string", "minLength": 1}
                        },
                    },
                },
            },
            "anyComponent": {"oneOf": component_refs},
        },
    }


def generated_files() -> dict[Path, bytes]:
    catalog = catalog_document()
    return {
        CATALOG_PATH: _json_bytes(catalog),
        PROFILE_PATH: PROFILE_PATH.read_bytes(),
        CANONICAL_GRAPH_PATH: CANONICAL_GRAPH_PATH.read_bytes(),
        WIRE_SCHEMA_PATH: _json_bytes(wire_schema()),
        PROMPT_PATH: PROMPT_PATH.read_bytes(),
        PROMPT_MIRROR_PATH: PROMPT_MIRROR_PATH.read_bytes(),
        PROMPT_GENERATOR_PATH: PROMPT_GENERATOR_PATH.read_bytes(),
        QUALITY_POLICY_PATH: QUALITY_POLICY_PATH.read_bytes(),
        **{path: path.read_bytes() for path in ANDROID_PROMPT_PATHS},
        PY_COMPILER_PATH: PY_COMPILER_PATH.read_bytes(),
        PY_ACTIVE_BOUNDARY_PATH: PY_ACTIVE_BOUNDARY_PATH.read_bytes(),
        PY_WIRE_COMPILER_PATH: PY_WIRE_COMPILER_PATH.read_bytes(),
        KOTLIN_COMPILER_PATH: KOTLIN_COMPILER_PATH.read_bytes(),
        KOTLIN_WIRE_COMPILER_PATH: KOTLIN_WIRE_COMPILER_PATH.read_bytes(),
        MIGRATION_PATH: MIGRATION_PATH.read_bytes(),
        REFERENCE_INVENTORY_PATH: REFERENCE_INVENTORY_PATH.read_bytes(),
        CONFORMANCE_CORPUS_PATH: CONFORMANCE_CORPUS_PATH.read_bytes(),
        GRAMMAR_PATH: PINNED_GRAMMAR.encode("utf-8"),
    }


def manifest_for(files: dict[Path, bytes]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for path, content in files.items():
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        entry: dict[str, Any] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
        if path == GRAMMAR_PATH:
            entry["upstreamGitBlobSha"] = A2UI_EXPRESS_GRAMMAR_GIT_BLOB_SHA
        details[rel] = entry
    return {
        "manifestVersion": "2.0.0",
        "upstreamRepository": A2UI_UPSTREAM_REPOSITORY,
        "upstreamCommit": A2UI_UPSTREAM_COMMIT,
        "protocolVersion": A2UI_PROTOCOL_VERSION,
        "expressVersion": A2UI_EXPRESS_VERSION,
        "catalogVersion": "genuicraft-a2ui-catalog-v1",
        "profileVersion": "genuicraft-a2ui-express-profile-v1",
        "canonicalGraphSchemaVersion": "canonical-ui-graph-v1",
        "catalogId": GENUICRAFT_CATALOG_ID,
        "catalogIdentityHash": catalog_document().get("catalogIdentityHash"),
        "tokenizer": {
            "status": "blocked",
            "name": "deployed_gemma_tokenizer_unavailable",
            "benchmarkFallback": "regex_lexical_estimate_v1",
        },
        "files": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail when generated files differ")
    args = parser.parse_args()
    files = generated_files()
    manifest_path = SCHEMA_DIR / "genuicraft_ir_formats.manifest.json"
    all_files = {**files, manifest_path: _json_bytes(manifest_for(files))}
    differences: list[str] = []
    for path, content in all_files.items():
        existing = path.read_bytes() if path.exists() else None
        if existing != content:
            differences.append(str(path.relative_to(ROOT)))
            if not args.check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
    if args.check and differences:
        print("Generated IR artifacts are stale:", *differences, sep="\n- ")
        return 1
    print("IR artifacts " + ("verified" if args.check else "generated") + f": {len(all_files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
