#!/usr/bin/env python3
"""Prepare guarded OpenCL FP16 GEMV -> FP32-accumulator diagnostic patches.

Input is a directory of captured OpenCL source files. The script never changes
those files. For each recognized half-accumulator GEMV it writes the exact
original to ``<fnv1a64>.before.cl`` and its patch to ``<fnv1a64>.after.cl``
under the output directory. The hash is of the exact original UTF-8 bytes
passed to clCreateProgramWithSource.

This isolates dot-product arithmetic: half tensor storage, decoded half weights,
fused quantization, kernel arguments, and final half writes remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


HALF_ACC = re.compile(
    r"\bhalf4(?P<space>[ \t]+)(?P<name>r_sp\d+_s\d+)"
    r"(?P<assign>[ \t]*=[ \t]*)\(half4\)\(0\.0f\)"
)
HALF_LOCAL = re.compile(r"\b__local(?P<space>[ \t]+)half4(?P<after>[ \t]+temp\[)")
PRODUCT = re.compile(
    r"(?P<lhs>\br_sp\d+_s\d+[ \t]*\+=[ \t]*)"
    r"(?P<value>v\d+\.[xyzw])[ \t]*\*[ \t]*(?P<weight>w\d+)"
    r"(?P<end>[ \t]*;)"
)
DOT = re.compile(
    r"(?P<lhs>\br_sp\d+_s\d+\.[xyzw][ \t]*\+=[ \t]*)"
    r"dot\([ \t]*(?P<value>v\d+)[ \t]*,[ \t]*(?P<weight>w\d+)[ \t]*\)"
    r"(?P<end>[ \t]*;)"
)
ACC_LINE = re.compile(r"\br_sp\d+_s\d+\b")
REDUCTION = re.compile(r"\br_sp\d+_s\d+[ \t]*\+=[ \t]*temp\[")
TEMP_STORE = re.compile(r"\btemp\[.*\][ \t]*=[ \t]*r_sp\d+_s\d+\b")
OUTPUT_CAST = re.compile(r"\bconvert_half4\([ \t]*r_sp\d+_s\d+[ \t]*\)")


def fnv1a64(data: bytes) -> str:
    value = 0xCBF29CE484222325
    for byte in data:
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def patch_source(source: str) -> tuple[str | None, dict[str, int], str | None]:
    """Return patched source, edit counts, or a specific exclusion reason."""
    if "MAIN_FUNCTION(" not in source or "weights_buffer" not in source:
        return None, {}, "not_generated_weighted_kernel"
    declarations = list(HALF_ACC.finditer(source))
    if not declarations:
        return None, {}, "no_half_r_sp_accumulator"
    if "#pragma OPENCL EXTENSION cl_khr_fp16 : enable" not in source:
        return None, {}, "missing_expected_fp16_pragma"

    # Fail closed on an unfamiliar use of the accumulator instead of silently
    # changing only part of a future kernel variant.
    for line in source.splitlines():
        if not ACC_LINE.search(line):
            continue
        if any(
            pattern.search(line)
            for pattern in (HALF_ACC, PRODUCT, DOT, REDUCTION, TEMP_STORE, OUTPUT_CAST)
        ):
            continue
        return None, {}, "unrecognized_accumulator_use"

    patch, product_count = PRODUCT.subn(
        lambda m: (
            f"{m['lhs']}convert_float({m['value']}) * "
            f"convert_float4({m['weight']}){m['end']}"
        ),
        source,
    )
    patch, dot_count = DOT.subn(
        lambda m: (
            f"{m['lhs']}dot(convert_float4({m['value']}), "
            f"convert_float4({m['weight']})){m['end']}"
        ),
        patch,
    )
    if product_count + dot_count == 0:
        return None, {}, "no_recognized_product"

    patch, local_count = HALF_LOCAL.subn(
        lambda m: f"__local{m['space']}float4{m['after']}", patch
    )
    patch, declaration_count = HALF_ACC.subn(
        lambda m: (
            f"float4{m['space']}{m['name']}"
            f"{m['assign']}(float4)(0.0f)"
        ),
        patch,
    )
    if declaration_count != len(declarations):
        return None, {}, "declaration_count_mismatch"
    if "__local half4 temp[" in patch or HALF_ACC.search(patch):
        return None, {}, "remaining_half_accumulator"
    if PRODUCT.search(patch) or DOT.search(patch):
        return None, {}, "remaining_half_product"
    if len(OUTPUT_CAST.findall(patch)) < declaration_count:
        return None, {}, "missing_explicit_half_output_cast"
    if source == patch:
        return None, {}, "no_change"

    return patch, {
        "float4_accumulator_declarations": declaration_count,
        "float4_local_arrays": local_count,
        "float_scalar_vector_products": product_count,
        "float_dot_products": dot_count,
    }, None


def write_checked(path: Path, data: bytes) -> None:
    """Permit idempotent reruns, but never overwrite a different prior patch."""
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"refusing to overwrite changed file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captured_kernel_dir", type=Path)
    parser.add_argument("output_patch_dir", type=Path)
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

    manifest: dict[str, object] = {
        "version": 1,
        "purpose": "OpenCL source-hook diagnostic, not a model or runtime fix",
        "hash_rule": "FNV-1a 64 of exact original UTF-8 source bytes",
        "captured_kernel_dir": str(source_dir),
        "patches": [],
        "exclusions": [],
    }
    seen_hashes: dict[str, bytes] = {}
    for path in files:
        raw = path.read_bytes()
        try:
            source = raw.decode("utf-8")
        except UnicodeDecodeError:
            manifest["exclusions"].append(
                {"source_file": path.name, "reason": "not_utf8"}
            )
            continue
        key = fnv1a64(raw)
        if key in seen_hashes and seen_hashes[key] != raw:
            raise ValueError(f"FNV collision at {path.name}: {key}")
        seen_hashes[key] = raw
        patch, edits, reason = patch_source(source)
        if reason:
            manifest["exclusions"].append(
                {
                    "source_file": path.name,
                    "fnv1a64": key,
                    "sha256_before": hashlib.sha256(raw).hexdigest(),
                    "reason": reason,
                }
            )
            continue
        assert patch is not None
        patched = patch.encode("utf-8")
        before_name = f"{key}.before.cl"
        after_name = f"{key}.after.cl"
        write_checked(output_dir / before_name, raw)
        write_checked(output_dir / after_name, patched)
        manifest["patches"].append(
            {
                "source_file": path.name,
                "fnv1a64": key,
                "before_file": before_name,
                "after_file": after_name,
                "sha256_before": hashlib.sha256(raw).hexdigest(),
                "sha256_after": hashlib.sha256(patched).hexdigest(),
                "edits": edits,
            }
        )
    manifest["patched_count"] = len(manifest["patches"])
    manifest["excluded_count"] = len(manifest["exclusions"])
    data = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    write_checked(output_dir / "manifest.json", data)
    print(
        f"{manifest['patched_count']} patched, {manifest['excluded_count']} "
        f"excluded; manifest: {output_dir / 'manifest.json'}"
    )
    return 0 if manifest["patched_count"] else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
