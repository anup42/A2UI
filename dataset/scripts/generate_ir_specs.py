#!/usr/bin/env python3
"""Regenerate pinned Compact IR/A2UI catalog, schemas, and manifest."""

from __future__ import annotations

import argparse
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
    COMPACT_IR_VERSION,
    COMPONENTS,
    GENUICRAFT_CATALOG_ID,
    catalog_identity_hash,
    catalog_payload,
)

SCHEMA_DIR = ROOT / "dataset" / "schema"
GRAMMAR_PATH = ROOT / "specification" / "inference_formats" / "express" / "Express.g4"

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


def compact_schema() -> dict[str, Any]:
    component_names = sorted({*COMPONENTS, "Row", "Column"})
    action_names = sorted(ACTIONS)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://genui.samsung.com/specification/compact-ir/v2/schema.json",
        "title": "GenUICraft Compact IR v2",
        "type": "object",
        "additionalProperties": False,
        "required": ["v", "r", "e"],
        "properties": {
            "v": {"const": COMPACT_IR_VERSION},
            "r": {"type": "string", "minLength": 1},
            "s": {"type": "object", "additionalProperties": True},
            "e": {
                "type": "object",
                "minProperties": 1,
                "additionalProperties": {"$ref": "#/$defs/element"},
            },
        },
        "$defs": {
            "action": {
                "type": "object",
                "additionalProperties": False,
                "required": ["action"],
                "properties": {
                    "action": {"type": "string", "enum": action_names},
                    "params": {"type": "object", "additionalProperties": True},
                },
            },
            "actionOrList": {
                "oneOf": [
                    {"$ref": "#/$defs/action"},
                    {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/action"}},
                ]
            },
            "element": {
                "type": "object",
                "additionalProperties": False,
                "required": ["t"],
                "properties": {
                    "t": {"type": "string", "enum": component_names},
                    "p": {"type": "object", "additionalProperties": True},
                    "c": {"type": "array", "items": {"type": "string", "minLength": 1}},
                    "x": {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {
                            "statePath": {"type": "string", "minLength": 1},
                            "path": {"type": "string", "minLength": 1},
                            "key": {"type": "string"},
                            "template": {"type": "string"},
                            "itemTemplate": {"type": "string"},
                            "child": {"type": "string"},
                        },
                        "anyOf": [
                            {"required": ["statePath"]},
                            {"required": ["path"]},
                            {"required": ["template"]},
                            {"required": ["itemTemplate"]},
                            {"required": ["child"]},
                        ],
                    },
                    "z": {},
                    "o": {"type": "object", "additionalProperties": {"$ref": "#/$defs/actionOrList"}},
                    "w": {"type": "object", "additionalProperties": {"$ref": "#/$defs/actionOrList"}},
                },
            },
        },
    }


def catalog_document() -> dict[str, Any]:
    payload = catalog_payload()
    payload["components"] = {
        name: {
            **descriptor,
            "schema": {
                "type": "object",
                "additionalProperties": True,
                "required": ["id", "component"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "component": {"const": name},
                    "children": {"type": "array", "items": {"type": "string"}},
                    "repeat": {"type": "object", "additionalProperties": True},
                    "visible": {},
                    "on": {"type": "object", "additionalProperties": True},
                    "watch": {"type": "object", "additionalProperties": True},
                },
            },
        }
        for name, descriptor in payload["components"].items()
    }
    payload["catalogIdentityHash"] = catalog_identity_hash()
    return payload


def wire_schema() -> dict[str, Any]:
    component = {
        "type": "object",
        "required": ["id", "component"],
        "additionalProperties": True,
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "component": {"type": "string", "enum": sorted(COMPONENTS)},
        },
    }
    message_properties = {"version": {"const": A2UI_PROTOCOL_VERSION}}
    create = {
        "type": "object",
        "additionalProperties": False,
        "required": ["surfaceId"],
        "properties": {
            "surfaceId": {"type": "string", "minLength": 1},
            "catalogId": {"type": "string", "const": GENUICRAFT_CATALOG_ID},
            "rootId": {"type": "string", "minLength": 1},
            "components": {"type": "array", "minItems": 1, "items": component},
            "dataModel": {"type": "object", "additionalProperties": True},
            "sendDataModel": {"type": "boolean"},
        },
    }
    update_components = {
        "type": "object",
        "additionalProperties": False,
        "required": ["surfaceId", "components"],
        "properties": {
            "surfaceId": {"type": "string", "minLength": 1},
            "components": {"type": "array", "minItems": 1, "items": component},
        },
    }
    update_data = {
        "type": "object",
        "additionalProperties": False,
        "required": ["surfaceId", "value"],
        "properties": {
            "surfaceId": {"type": "string", "minLength": 1},
            "path": {"type": "string"},
            "value": {},
        },
    }
    delete = {
        "type": "object",
        "additionalProperties": False,
        "required": ["surfaceId"],
        "properties": {"surfaceId": {"type": "string", "minLength": 1}},
    }
    defs = {
        "create": {"type": "object", "additionalProperties": False, "required": ["version", "createSurface"], "properties": {**message_properties, "createSurface": create}},
        "updateComponents": {"type": "object", "additionalProperties": False, "required": ["version", "updateComponents"], "properties": {**message_properties, "updateComponents": update_components}},
        "updateDataModel": {"type": "object", "additionalProperties": False, "required": ["version", "updateDataModel"], "properties": {**message_properties, "updateDataModel": update_data}},
        "deleteSurface": {"type": "object", "additionalProperties": False, "required": ["version", "deleteSurface"], "properties": {**message_properties, "deleteSurface": delete}},
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://genui.samsung.com/specification/a2ui/v1/agent_to_renderer.json",
        "title": "GenUICraft A2UI v1 Wire Messages",
        "oneOf": [
            {"$ref": "#/$defs/create"},
            {"$ref": "#/$defs/updateComponents"},
            {"$ref": "#/$defs/updateDataModel"},
            {"$ref": "#/$defs/deleteSurface"},
            {"type": "array", "minItems": 1, "items": {"oneOf": [
                {"$ref": "#/$defs/create"},
                {"$ref": "#/$defs/updateComponents"},
                {"$ref": "#/$defs/updateDataModel"},
                {"$ref": "#/$defs/deleteSurface"},
            ]}},
        ],
        "$defs": defs,
    }


def generated_files() -> dict[Path, bytes]:
    return {
        SCHEMA_DIR / "genui_compact_ir_v2.schema.json": _json_bytes(compact_schema()),
        SCHEMA_DIR / "genuicraft_a2ui_catalog_v1.json": _json_bytes(catalog_document()),
        SCHEMA_DIR / "genuicraft_a2ui_v1_wire.schema.json": _json_bytes(wire_schema()),
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
        "manifestVersion": "1.1.0",
        "upstreamRepository": A2UI_UPSTREAM_REPOSITORY,
        "upstreamCommit": A2UI_UPSTREAM_COMMIT,
        "protocolVersion": A2UI_PROTOCOL_VERSION,
        "compactIrVersion": COMPACT_IR_VERSION,
        "expressVersion": A2UI_EXPRESS_VERSION,
        "catalogId": GENUICRAFT_CATALOG_ID,
        "catalogIdentityHash": catalog_identity_hash(),
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
