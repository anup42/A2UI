#!/usr/bin/env python3
"""Add the LiteRT-LM JNI dependency without rewriting Android ELF metadata.

The upstream Android sampler has unresolved LiteRT runtime symbols when it is
loaded from an AAR.  A normal patchelf --add-needed operation rewrites the
Android-specific dynamic metadata in some patchelf versions and produces a
library that Android rejects because its hash tables are no longer valid.

This sampler already has a DT_SONAME entry whose string is longer than the
dependency name.  Reusing that string slot and changing only DT_SONAME to
DT_NEEDED preserves the packed Android relocation/hash sections byte-for-byte.
The sampler is loaded by its filename, so retaining a SONAME is unnecessary.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


ELF64_EHDR = struct.Struct("<16sHHIQQQIHHHHHH")
ELF64_PHDR = struct.Struct("<IIQQQQQQ")
ELF64_DYN = struct.Struct("<qQ")

PT_LOAD = 1
PT_DYNAMIC = 2
DT_NULL = 0
DT_NEEDED = 1
DT_STRTAB = 5
DT_STRSZ = 10
DT_SONAME = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dependency", default="libLiteRtRuntimeBuiltin.so")
    parser.add_argument(
        "--rename-soname",
        help="Safely rename the existing DT_SONAME without changing ELF tags.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = bytearray(args.input.read_bytes())
    if len(data) < ELF64_EHDR.size or data[:4] != b"\x7fELF":
        raise SystemExit("input is not an ELF file")

    ident = data[:16]
    if ident[4] != 2 or ident[5] != 1:
        raise SystemExit("only little-endian ELF64 is supported")

    ehdr = ELF64_EHDR.unpack_from(data, 0)
    (
        _, _, _, _, _, phoff, _, _, _, phentsize, phnum, _, _, _
    ) = ehdr
    if phentsize != ELF64_PHDR.size:
        raise SystemExit(f"unexpected program-header size: {phentsize}")

    dynamic_offset = None
    dynamic_size = None
    load_segments = []
    for index in range(phnum):
        phdr = ELF64_PHDR.unpack_from(data, phoff + index * phentsize)
        p_type, _, p_offset, p_vaddr, _, p_filesz, _, _ = phdr
        if p_type == PT_LOAD:
            load_segments.append((p_offset, p_vaddr, p_filesz))
        elif p_type == PT_DYNAMIC:
            dynamic_offset = p_offset
            dynamic_size = p_filesz
    if dynamic_offset is None or dynamic_size is None:
        raise SystemExit("PT_DYNAMIC was not found")

    entries = []
    for offset in range(dynamic_offset, dynamic_offset + dynamic_size, ELF64_DYN.size):
        tag, value = ELF64_DYN.unpack_from(data, offset)
        entries.append((offset, tag, value))
        if tag == DT_NULL:
            break

    strtab_vaddr = next((value for _, tag, value in entries if tag == DT_STRTAB), None)
    strsz = next((value for _, tag, value in entries if tag == DT_STRSZ), None)
    soname_entry = next(((offset, value) for offset, tag, value in entries if tag == DT_SONAME), None)
    if strtab_vaddr is None or strsz is None or soname_entry is None:
        raise SystemExit("missing DT_STRTAB, DT_STRSZ, or DT_SONAME")

    strtab_offset = None
    for p_offset, p_vaddr, p_filesz in load_segments:
        if p_vaddr <= strtab_vaddr < p_vaddr + p_filesz:
            strtab_offset = p_offset + (strtab_vaddr - p_vaddr)
            break
    if strtab_offset is None:
        raise SystemExit("DT_STRTAB is not inside a PT_LOAD segment")

    soname_entry_offset, soname_string_offset = soname_entry
    string_start = strtab_offset + soname_string_offset
    string_end = data.find(b"\0", string_start, strtab_offset + strsz)
    if string_end < 0:
        raise SystemExit("DT_SONAME string is not NUL-terminated")
    old_name = bytes(data[string_start:string_end])
    new_name_text = args.rename_soname if args.rename_soname is not None else args.dependency
    new_name = new_name_text.encode("ascii")
    if len(new_name) > len(old_name):
        raise SystemExit(
            f"dependency name is longer than reusable SONAME slot: "
            f"{len(new_name)} > {len(old_name)}"
        )
    if not old_name:
        raise SystemExit("DT_SONAME slot is empty")

    # Change only the existing string bytes for a SONAME rename. The dynamic
    # table and all Android packed relocation/hash data remain untouched.
    if args.rename_soname is not None:
        replacement = new_name + b"\0"
        data[string_start:string_end + 1] = replacement.ljust(len(old_name) + 1, b"\0")
    else:
        # Change only the dynamic tag and the existing string bytes. The
        # original DT_NULL remains in place, and all Android packed
        # relocation/hash data is left untouched.
        struct.pack_into("<q", data, soname_entry_offset, DT_NEEDED)
        replacement = new_name + b"\0"
        data[string_start:string_end + 1] = replacement.ljust(len(old_name) + 1, b"\0")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    if args.rename_soname is not None:
        print(f"renamed DT_SONAME '{old_name.decode(errors='replace')}' to '{args.rename_soname}'")
    else:
        print(f"reused DT_SONAME '{old_name.decode(errors='replace')}' as DT_NEEDED '{args.dependency}'")
    print(f"wrote {args.output} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
