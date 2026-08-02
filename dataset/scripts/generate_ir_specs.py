#!/usr/bin/env python3
"""Regenerate pinned A2UI Express catalog, wire schema, grammar, and manifest."""

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
PY_COMPILER_PATH = ROOT / "dataset" / "src" / "pipeline" / "ir_formats" / "express.py"
KOTLIN_COMPILER_PATH = ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "samsung" / "genuicraft" / "pipeline" / "A2uiExpressCodec.kt"
MIGRATION_PATH = ROOT / "dataset" / "scripts" / "migrate_legacy_dataset_to_a2ui_express.py"

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
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def wire_schema() -> dict[str, Any]:
    return json.loads(WIRE_SCHEMA_PATH.read_text(encoding="utf-8"))


def generated_files() -> dict[Path, bytes]:
    return {
        CATALOG_PATH: CATALOG_PATH.read_bytes(),
        PROFILE_PATH: PROFILE_PATH.read_bytes(),
        CANONICAL_GRAPH_PATH: CANONICAL_GRAPH_PATH.read_bytes(),
        WIRE_SCHEMA_PATH: WIRE_SCHEMA_PATH.read_bytes(),
        PROMPT_PATH: PROMPT_PATH.read_bytes(),
        PY_COMPILER_PATH: PY_COMPILER_PATH.read_bytes(),
        KOTLIN_COMPILER_PATH: KOTLIN_COMPILER_PATH.read_bytes(),
        MIGRATION_PATH: MIGRATION_PATH.read_bytes(),
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
        "catalogIdentityHash": json.loads(CATALOG_PATH.read_text(encoding="utf-8")).get("catalogIdentityHash"),
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
