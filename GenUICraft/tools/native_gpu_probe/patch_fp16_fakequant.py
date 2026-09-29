#!/usr/bin/env python3
"""Prepare guarded OpenCL FP16 fake-quant arithmetic diagnostic patches.

The input is a captured OpenCL source directory. The output contains flat
``<fnv1a64>.before.cl`` and ``<fnv1a64>.after.cl`` pairs for the source hook.
An optional ``--gemv-patch-dir`` composes the existing float-product GEMV
diagnostic with this one, always using the *captured original* as ``before``.

Only recognized FAKE_QUANT blocks are changed. Kernel arguments, half image
reads/writes, packed weights and their dequantization remain unchanged. Scalar
coefficients arrive as half arguments and thus remain half-rounded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


COEFF = r"shared_half4_\d+\.[xyzw]"
IDENT = r"[A-Za-z_]\w*"
WS = r"[ \t]*(?:\r?\n[ \t]*)*"
BLOCK = re.compile(
    rf"(?P<indent>[ \t]*)half4 clamped_value = min\(\(half4\)\((?P<hi>{COEFF})\), "
    rf"max\(\(half4\)\((?P<lo>{COEFF})\), (?P<input>{IDENT})\)\);"
    rf"(?P<ws1>{WS})half4 quantized_value = round\(\(clamped_value - "
    rf"\(half4\)\((?P<base>{COEFF})\)\) \* \(half4\)\((?P<inv>{COEFF})\)\);"
    rf"(?P<ws2>{WS})half4 dequantized_value = quantized_value \* "
    rf"\(half4\)\((?P<step>{COEFF})\) \+ \(half4\)\((?P<base2>{COEFF})\);"
    rf"(?P<ws3>{WS})(?P<dest>{IDENT}) = dequantized_value;"
)
EARLY_HALF_CAST = re.compile(
    r"\bhalf4(?P<space>[ \t]+)(?P<name>res_value)(?P<assign>[ \t]*=[ \t]*)"
    r"convert_half4\((?P<acc>r_sp\d+_s\d+)\);"
)
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def fnv1a64(data: bytes) -> str:
    value = FNV_OFFSET
    for byte in data:
        value ^= byte
        value = (value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_checked(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"refusing to overwrite changed file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def patch_fakequant(source: str, *, combined_gemv: bool) -> tuple[str, dict[str, int], str | None]:
    """Return patched source and counts, or fail closed on unfamiliar syntax."""
    candidates = source.count("half4 quantized_value = round(")
    if candidates == 0:
        return source, {}, "no_recognized_fakequant"
    blocks = list(BLOCK.finditer(source))
    if len(blocks) != candidates:
        return source, {}, f"unrecognized_fakequant_block:{len(blocks)}/{candidates}"
    for match in blocks:
        if match["lo"] != match["base"] or match["base"] != match["base2"]:
            return source, {}, "fakequant_zero_point_mismatch"
        # The generated assignment targets are half4 temporaries. Keep their
        # storage boundary at the same point, after float dequantization.
        if not re.search(rf"\bhalf4[ \t]+{re.escape(match['dest'])}\b", source):
            return source, {}, f"unrecognized_fakequant_destination:{match['dest']}"

    early_casts = list(EARLY_HALF_CAST.finditer(source)) if combined_gemv else []
    if combined_gemv and early_casts:
        if "float4 r_sp" not in source:
            return source, {}, "gemv_accumulator_not_float"
        if sum(match["input"] == "res_value" for match in blocks) != len(early_casts):
            return source, {}, "early_half_cast_not_one_to_one_fakequant_input"

    def replace(match: re.Match[str]) -> str:
        c = lambda name: f"(float4)(convert_float({match[name]}))"
        # OpenCL C requires convert_float4 for half4 -> float4. The GEMV
        # result is already float4 when its premature half cast is removed.
        input_value = (
            match["input"] if combined_gemv and early_casts and match["input"] == "res_value"
            else f"convert_float4({match['input']})"
        )
        return (
            f"{match['indent']}float4 clamped_value = min({c('hi')}, "
            f"max({c('lo')}, {input_value}));"
            f"{match['ws1']}float4 quantized_value = round((clamped_value - "
            f"{c('base')}) * {c('inv')});"
            f"{match['ws2']}float4 dequantized_value = quantized_value * "
            f"{c('step')} + {c('base2')};"
            f"{match['ws3']}{match['dest']} = convert_half4(dequantized_value);"
        )

    patched = BLOCK.sub(replace, source)
    if combined_gemv and early_casts:
        patched = EARLY_HALF_CAST.sub(
            lambda m: f"float4{m['space']}{m['name']}{m['assign']}{m['acc']};",
            patched,
        )
        if len(EARLY_HALF_CAST.findall(patched)):
            return source, {}, "remaining_early_half_cast"
    if patched.count("float4 quantized_value = round(") != candidates:
        return source, {}, "patched_fakequant_count_mismatch"
    if "half4 quantized_value = round(" in patched:
        return source, {}, "remaining_half_fakequant"
    return patched, {
        "float_fakequant_blocks": candidates,
        "removed_premature_gemv_half_casts": len(early_casts),
    }, None


def gemv_lookup(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("hash_rule") != "FNV-1a 64 of exact original UTF-8 source bytes":
        raise ValueError("GEMV manifest has an unexpected hash rule")
    result = {}
    for entry in manifest["patches"]:
        key = entry["fnv1a64"]
        before = (path / entry["before_file"]).read_bytes()
        after = (path / entry["after_file"]).read_bytes()
        if fnv1a64(before) != key or sha256(before) != entry["sha256_before"]:
            raise ValueError(f"GEMV before hash mismatch: {key}")
        if sha256(after) != entry["sha256_after"]:
            raise ValueError(f"GEMV after hash mismatch: {key}")
        result[key] = {"entry": entry, "before": before, "after": after}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captured_kernel_dir", type=Path)
    parser.add_argument("output_patch_dir", type=Path)
    parser.add_argument("--gemv-patch-dir", type=Path)
    args = parser.parse_args()
    source_dir = args.captured_kernel_dir.resolve()
    output_dir = args.output_patch_dir.resolve()
    if not source_dir.is_dir():
        parser.error(f"not a source directory: {source_dir}")
    if output_dir == source_dir:
        parser.error("output directory must differ from captured kernel directory")
    files = sorted(source_dir.glob("*.cl"))
    if not files:
        parser.error(f"no .cl files found in {source_dir}")
    gemv = gemv_lookup(args.gemv_patch_dir.resolve() if args.gemv_patch_dir else None)
    manifest: dict[str, object] = {
        "version": 1,
        "purpose": "OpenCL source-hook diagnostic, not a model or runtime fix",
        "hash_rule": "FNV-1a 64 of exact original UTF-8 source bytes",
        "captured_kernel_dir": str(source_dir),
        "combined_gemv_patch_dir": str(args.gemv_patch_dir.resolve()) if args.gemv_patch_dir else None,
        "precision_limit": "shared_half4 coefficients remain half-rounded; only fake-quant arithmetic is float",
        "patches": [],
        "exclusions": [],
    }
    captured_keys = set()
    for path in files:
        original = path.read_bytes()
        key = fnv1a64(original)
        captured_keys.add(key)
        if key in gemv and gemv[key]["before"] != original:
            raise ValueError(f"GEMV source does not match capture: {path.name}")
        base = gemv[key]["after"] if key in gemv else original
        try:
            source = base.decode("utf-8")
        except UnicodeDecodeError:
            manifest["exclusions"].append({"source_file": path.name, "fnv1a64": key, "reason": "not_utf8"})
            continue
        patched, edits, reason = patch_fakequant(source, combined_gemv=key in gemv)
        if reason and reason != "no_recognized_fakequant":
            # Never substitute only the GEMV edit if a fake-quant block was
            # recognized but could not be transformed safely.
            manifest["exclusions"].append({"source_file": path.name, "fnv1a64": key, "reason": reason})
            continue
        if key in gemv:
            edits = {**gemv[key]["entry"]["edits"], **edits}
        if patched == original.decode("utf-8"):
            manifest["exclusions"].append({"source_file": path.name, "fnv1a64": key, "reason": reason or "no_change"})
            continue
        after = patched.encode("utf-8")
        before_name, after_name = f"{key}.before.cl", f"{key}.after.cl"
        write_checked(output_dir / before_name, original)
        write_checked(output_dir / after_name, after)
        manifest["patches"].append({
            "source_file": path.name,
            "fnv1a64": key,
            "before_file": before_name,
            "after_file": after_name,
            "sha256_before": sha256(original),
            "sha256_after": sha256(after),
            "edits": edits,
        })
    missing = set(gemv) - captured_keys
    if missing:
        raise ValueError(f"GEMV manifest has {len(missing)} sources missing from capture")
    manifest["patched_count"] = len(manifest["patches"])
    manifest["excluded_count"] = len(manifest["exclusions"])
    write_checked(output_dir / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    print(f"{manifest['patched_count']} patched, {manifest['excluded_count']} excluded; manifest: {output_dir / 'manifest.json'}")
    return 0 if manifest["patched_count"] else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
