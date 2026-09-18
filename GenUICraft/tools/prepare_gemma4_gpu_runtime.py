#!/usr/bin/env python3
"""Reproduce the ARM64 LiteRT GPU runtime packaged by GenUICraft.

Every downloaded artifact and final native library is SHA-256 pinned. The only
binary modification replaces the upstream sampler's DT_SONAME entry with a
DT_NEEDED entry for libLiteRt.so. It does not move sections or rewrite Android
packed relocations/hash tables.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import io
import struct
import urllib.request
import zipfile
from pathlib import Path


LITERT_AAR_URL = (
    "https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/"
    "litert/2.2.0/litert-2.2.0.aar"
)
LITERT_AAR_SHA256 = "624518d72f8a249711a19e9901f480e74f823ca7818260a739cb2c023024807c"
LITERT_SO_ENTRY = "jni/arm64-v8a/libLiteRt.so"
LITERT_SO_SHA256 = "97355a36cb8ac7628cf407773291e98da79f3ef184cc43cb0e57dedf5f0c0637"

LITERT_LM_TAG = "v0.16.1"
LITERT_LM_MEDIA_ROOT = (
    "https://media.githubusercontent.com/media/google-ai-edge/LiteRT-LM/"
    f"{LITERT_LM_TAG}/prebuilt/android_arm64"
)
OPENCL_NAME = "libLiteRtOpenClAccelerator.so"
OPENCL_SHA256 = "dfdcb6a551dc78a9dab88883ad6e8e2d9b090ecda383a5cea3731c6b9561d244"
SAMPLER_NAME = "libLiteRtTopKOpenClSampler.so"
SAMPLER_SHA256 = "4404dc68786460602685cab62ddfa29035e9cfc38bb4550dec15abaaa1302a82"
PATCHED_SAMPLER_SHA256 = "743371dcc1ee57d4e2d14e6d50075c424f6ba0df554f6bb94a8b0afc1e76e253"

LITERT_LM_LICENSE_URL = (
    f"https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/{LITERT_LM_TAG}/LICENSE"
)
LITERT_LM_LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
LITERT_LICENSE_SHA256 = "71c6915d04265772a0339bed47276942c678b45cc01534210ebe6984fd1aec65"
LITERT_NOTICE_SHA256 = "2d4d617ff3047813c1b4bfd66dc3c95b2352b401546c6e58a33b7c885091372a"

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


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verified(data: bytes, expected: str, label: str) -> bytes:
    actual = sha256(data)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: expected {expected}, found {actual}")
    return data


def download(url: str, expected: str, label: str) -> bytes:
    last_failure: Exception | None = None
    for _ in range(3):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "GenUICraft-runtime-preparer/1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return verified(response.read(), expected, label)
        except (OSError, EOFError, http.client.HTTPException) as failure:
            last_failure = failure
    raise RuntimeError(f"could not download {label} after three attempts") from last_failure


def patch_sampler(upstream: bytes) -> bytes:
    data = bytearray(upstream)
    if len(data) < ELF64_EHDR.size or data[:4] != b"\x7fELF":
        raise RuntimeError("upstream sampler is not an ELF file")
    if data[4] != 2 or data[5] != 1:
        raise RuntimeError("upstream sampler is not little-endian ELF64")

    ehdr = ELF64_EHDR.unpack_from(data, 0)
    phoff, phentsize, phnum = ehdr[5], ehdr[9], ehdr[10]
    if phentsize != ELF64_PHDR.size:
        raise RuntimeError(f"unexpected ELF program-header size: {phentsize}")

    dynamic_offset = None
    dynamic_size = None
    load_segments: list[tuple[int, int, int]] = []
    for index in range(phnum):
        phdr = ELF64_PHDR.unpack_from(data, phoff + index * phentsize)
        p_type, _, p_offset, p_vaddr, _, p_filesz, _, _ = phdr
        if p_type == PT_LOAD:
            load_segments.append((p_offset, p_vaddr, p_filesz))
        elif p_type == PT_DYNAMIC:
            dynamic_offset, dynamic_size = p_offset, p_filesz
    if dynamic_offset is None or dynamic_size is None:
        raise RuntimeError("sampler has no PT_DYNAMIC segment")

    entries: list[tuple[int, int, int]] = []
    for offset in range(dynamic_offset, dynamic_offset + dynamic_size, ELF64_DYN.size):
        tag, value = ELF64_DYN.unpack_from(data, offset)
        entries.append((offset, tag, value))
        if tag == DT_NULL:
            break
    strtab_vaddr = next((v for _, tag, v in entries if tag == DT_STRTAB), None)
    strsz = next((v for _, tag, v in entries if tag == DT_STRSZ), None)
    soname = next(((o, v) for o, tag, v in entries if tag == DT_SONAME), None)
    if strtab_vaddr is None or strsz is None or soname is None:
        raise RuntimeError("sampler is missing DT_STRTAB, DT_STRSZ, or DT_SONAME")

    strtab_offset = next(
        (
            offset + (strtab_vaddr - vaddr)
            for offset, vaddr, size in load_segments
            if vaddr <= strtab_vaddr < vaddr + size
        ),
        None,
    )
    if strtab_offset is None:
        raise RuntimeError("sampler DT_STRTAB is outside PT_LOAD")

    entry_offset, string_offset = soname
    start = strtab_offset + string_offset
    end = data.find(b"\0", start, strtab_offset + strsz)
    if end < 0:
        raise RuntimeError("sampler DT_SONAME is not NUL-terminated")
    old_name = bytes(data[start:end])
    expected_soname = SAMPLER_NAME.encode("ascii")
    if old_name != expected_soname:
        raise RuntimeError(f"unexpected sampler DT_SONAME: {old_name!r}")

    dependency = b"libLiteRt.so"
    struct.pack_into("<q", data, entry_offset, DT_NEEDED)
    data[start : end + 1] = (dependency + b"\0").ljust(len(old_name) + 1, b"\0")
    return verified(bytes(data), PATCHED_SAMPLER_SHA256, "patched sampler")


def write_verified(path: Path, data: bytes, expected: str) -> None:
    verified(data, expected, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jni-dir",
        type=Path,
        default=project / "genuicraft/src/main/jniLibs/arm64-v8a",
    )
    parser.add_argument(
        "--license-dir",
        type=Path,
        default=project / "genuicraft/src/main/assets/genuicraft_licenses",
    )
    args = parser.parse_args()

    aar = download(LITERT_AAR_URL, LITERT_AAR_SHA256, "LiteRT 2.2.0 AAR")
    with zipfile.ZipFile(io.BytesIO(aar)) as archive:
        litert_so = verified(archive.read(LITERT_SO_ENTRY), LITERT_SO_SHA256, LITERT_SO_ENTRY)
        litert_license = verified(
            archive.read("LICENSE"), LITERT_LICENSE_SHA256, "LiteRT LICENSE"
        )
        litert_notice = verified(
            archive.read("THIRD_PARTY_NOTICE.txt"),
            LITERT_NOTICE_SHA256,
            "LiteRT third-party notice",
        )

    opencl = download(
        f"{LITERT_LM_MEDIA_ROOT}/{OPENCL_NAME}", OPENCL_SHA256, OPENCL_NAME
    )
    sampler = download(
        f"{LITERT_LM_MEDIA_ROOT}/{SAMPLER_NAME}", SAMPLER_SHA256, SAMPLER_NAME
    )
    patched_sampler = patch_sampler(sampler)
    litert_lm_license = download(
        LITERT_LM_LICENSE_URL, LITERT_LM_LICENSE_SHA256, "LiteRT-LM LICENSE"
    )

    write_verified(args.jni_dir / "libLiteRt.so", litert_so, LITERT_SO_SHA256)
    write_verified(args.jni_dir / OPENCL_NAME, opencl, OPENCL_SHA256)
    write_verified(args.jni_dir / SAMPLER_NAME, patched_sampler, PATCHED_SAMPLER_SHA256)
    write_verified(
        args.license_dir / "LiteRT-LICENSE.txt", litert_license, LITERT_LICENSE_SHA256
    )
    write_verified(
        args.license_dir / "LiteRT-THIRD-PARTY-NOTICE.txt",
        litert_notice,
        LITERT_NOTICE_SHA256,
    )
    write_verified(
        args.license_dir / "LiteRT-LM-LICENSE.txt",
        litert_lm_license,
        LITERT_LM_LICENSE_SHA256,
    )
    print(f"Prepared pinned ARM64 GPU runtime in {args.jni_dir}")


if __name__ == "__main__":
    main()
