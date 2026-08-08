#!/usr/bin/env python3
"""Audit small retained constants between two public Safetensors checkpoints.

The exact-topology compiler replaces the quantized FC/embedding inventory but
retains norms, layer scalars, and other non-inventory constants from the
released mobile package.  This audit checks whether a public dense training
seed exposes byte-identical counterparts for those retained language-model
constants without downloading the complete remote checkpoint.

Only bounded HTTP range responses are accepted. A server that ignores the
Range header is rejected before its response body is consumed. Quantizer
observer min/max tensors are excluded because they are training metadata, not
runtime-retained learned constants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import get_hf_file_metadata, hf_hub_url

MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_RANGE_GROUP_BYTES = 1024 * 1024
DEFAULT_PREFIX = "model.language_model."
DEFAULT_EXCLUDED_SUFFIXES = (
    ".input_max",
    ".input_min",
    ".output_max",
    ".output_min",
)
CONTENT_RANGE_RE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", re.IGNORECASE)


class RetainedConstantAuditError(RuntimeError):
    """Raised when a bounded retained-constant audit cannot be trusted."""


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _read_local_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise RetainedConstantAuditError(f"Safetensors header is truncated: {path}")
        header_size = int.from_bytes(raw_size, "little", signed=False)
        if not 0 < header_size <= MAX_HEADER_BYTES:
            raise RetainedConstantAuditError(
                f"Unsafe Safetensors header size {header_size}: {path}"
            )
        raw_header = handle.read(header_size)
    if len(raw_header) != header_size:
        raise RetainedConstantAuditError(f"Safetensors header is truncated: {path}")
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetainedConstantAuditError(
            f"Could not parse Safetensors header {path}: {exc}"
        ) from exc
    if not isinstance(header, dict):
        raise RetainedConstantAuditError(f"Safetensors header is not an object: {path}")
    return header_size, header


def _metadata_with_retry(repo_id: str, filename: str, retries: int) -> Any:
    errors: list[str] = []
    for attempt in range(max(1, retries)):
        try:
            return get_hf_file_metadata(hf_hub_url(repo_id, filename))
        except Exception as exc:  # noqa: BLE001 - network failures are summarized
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt + 1 < retries:
                time.sleep(min(4.0, 0.5 * (2**attempt)))
    raise RetainedConstantAuditError(
        f"Could not resolve Hugging Face metadata after {retries} attempts: {errors}"
    )


def _bounded_range_get(
    location: str,
    *,
    begin: int,
    end: int,
    retries: int,
    timeout_seconds: int,
    max_bytes: int = MAX_RANGE_GROUP_BYTES,
) -> bytes:
    if begin < 0 or end < begin:
        raise RetainedConstantAuditError(f"Invalid HTTP range [{begin}, {end}].")
    expected_size = end - begin + 1
    if max_bytes < 1 or expected_size > max_bytes:
        raise RetainedConstantAuditError(
            f"Refusing a range of {expected_size} bytes; limit is {max_bytes}."
        )
    errors: list[str] = []
    for attempt in range(max(1, retries)):
        response = None
        try:
            response = requests.get(
                location,
                headers={
                    "Range": f"bytes={begin}-{end}",
                    "Accept-Encoding": "identity",
                },
                stream=True,
                timeout=timeout_seconds,
            )
            content_range = response.headers.get("Content-Range") or ""
            match = CONTENT_RANGE_RE.fullmatch(content_range.strip())
            if response.status_code != 206 or match is None:
                raise RetainedConstantAuditError(
                    "Remote server did not honor the bounded byte range: "
                    f"status={response.status_code}, Content-Range={content_range!r}."
                )
            observed_begin, observed_end = int(match.group(1)), int(match.group(2))
            if observed_begin != begin or observed_end != end:
                raise RetainedConstantAuditError(
                    "Remote Content-Range differs from the requested range: "
                    f"requested={begin}-{end}, observed={observed_begin}-{observed_end}."
                )
            payload = response.raw.read(expected_size + 1)
            if len(payload) != expected_size:
                raise RetainedConstantAuditError(
                    f"Remote range returned {len(payload)} bytes; expected {expected_size}."
                )
            return payload
        except Exception as exc:  # noqa: BLE001 - bounded retries are reported
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt + 1 < retries:
                time.sleep(min(4.0, 0.5 * (2**attempt)))
        finally:
            if response is not None:
                response.close()
    raise RetainedConstantAuditError(
        f"Could not read remote range {begin}-{end} after {retries} attempts: {errors}"
    )


def _read_remote_header(
    location: str, *, retries: int, timeout_seconds: int
) -> tuple[int, dict[str, Any]]:
    raw_size = _bounded_range_get(
        location,
        begin=0,
        end=7,
        retries=retries,
        timeout_seconds=timeout_seconds,
    )
    header_size = int.from_bytes(raw_size, "little", signed=False)
    if not 0 < header_size <= MAX_HEADER_BYTES:
        raise RetainedConstantAuditError(
            f"Unsafe remote Safetensors header size {header_size}."
        )
    raw_header = _bounded_range_get(
        location,
        begin=8,
        end=7 + header_size,
        retries=retries,
        timeout_seconds=timeout_seconds,
        max_bytes=MAX_HEADER_BYTES,
    )
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetainedConstantAuditError(
            f"Could not parse remote Safetensors header: {exc}"
        ) from exc
    if not isinstance(header, dict):
        raise RetainedConstantAuditError("Remote Safetensors header is not an object.")
    return header_size, header


def _entry_schema(entry: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
    try:
        dtype = str(entry["dtype"])
        shape = tuple(int(value) for value in entry["shape"])
        offsets = tuple(int(value) for value in entry["data_offsets"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RetainedConstantAuditError(f"Invalid Safetensors entry: {entry}") from exc
    if len(offsets) != 2 or offsets[0] < 0 or offsets[1] <= offsets[0]:
        raise RetainedConstantAuditError(f"Invalid Safetensors data offsets: {entry}")
    return dtype, shape


def _select_entries(
    header: dict[str, Any],
    *,
    prefix: str,
    max_rank: int,
    excluded_suffixes: tuple[str, ...] = DEFAULT_EXCLUDED_SUFFIXES,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for name, raw_entry in header.items():
        if name == "__metadata__" or not str(name).startswith(prefix):
            continue
        if any(str(name).endswith(suffix) for suffix in excluded_suffixes):
            continue
        if not isinstance(raw_entry, dict):
            raise RetainedConstantAuditError(f"Invalid tensor entry for {name!r}.")
        _, shape = _entry_schema(raw_entry)
        if len(shape) <= max_rank:
            selected[str(name)] = raw_entry
    return selected


def _coalesce_remote_ranges(
    entries: dict[str, dict[str, Any]],
    *,
    data_start: int,
    max_group_bytes: int = MAX_RANGE_GROUP_BYTES,
) -> list[dict[str, Any]]:
    ordered = sorted(
        (
            int(entry["data_offsets"][0]),
            int(entry["data_offsets"][1]),
            name,
        )
        for name, entry in entries.items()
    )
    groups: list[dict[str, Any]] = []
    for relative_begin, relative_end, name in ordered:
        begin = data_start + relative_begin
        end = data_start + relative_end
        if (
            groups
            and groups[-1]["end_exclusive"] == begin
            and end - groups[-1]["begin"] <= max_group_bytes
        ):
            groups[-1]["end_exclusive"] = end
            groups[-1]["items"].append(
                {"name": name, "begin": begin, "end_exclusive": end}
            )
        else:
            groups.append(
                {
                    "begin": begin,
                    "end_exclusive": end,
                    "items": [
                        {"name": name, "begin": begin, "end_exclusive": end}
                    ],
                }
            )
    return groups


def _fetch_remote_entries(
    location: str,
    groups: list[dict[str, Any]],
    *,
    workers: int,
    retries: int,
    timeout_seconds: int,
) -> dict[str, bytes]:
    observed: dict[str, bytes] = {}

    def fetch(group: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
        payload = _bounded_range_get(
            location,
            begin=int(group["begin"]),
            end=int(group["end_exclusive"]) - 1,
            retries=retries,
            timeout_seconds=timeout_seconds,
        )
        return group, payload

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(fetch, group) for group in groups]
        for future in as_completed(futures):
            group, payload = future.result()
            group_begin = int(group["begin"])
            for item in group["items"]:
                relative_begin = int(item["begin"]) - group_begin
                relative_end = int(item["end_exclusive"]) - group_begin
                observed[str(item["name"])] = payload[relative_begin:relative_end]
    return observed


def _read_local_entries(
    path: Path,
    header_size: int,
    entries: dict[str, dict[str, Any]],
) -> dict[str, bytes]:
    data_start = 8 + header_size
    observed: dict[str, bytes] = {}
    with path.open("rb") as handle:
        for name in sorted(entries):
            begin, end = (int(value) for value in entries[name]["data_offsets"])
            handle.seek(data_start + begin)
            payload = handle.read(end - begin)
            if len(payload) != end - begin:
                raise RetainedConstantAuditError(
                    f"Local tensor is truncated: {name!r} in {path}"
                )
            observed[name] = payload
    return observed


def _aggregate_digest(
    entries: dict[str, dict[str, Any]], values: dict[str, bytes]
) -> str:
    digest = hashlib.sha256()
    for name in sorted(entries):
        dtype, shape = _entry_schema(entries[name])
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "little"))
        digest.update(encoded_name)
        digest.update(dtype.encode("ascii"))
        digest.update(json.dumps(shape).encode("ascii"))
        digest.update(values[name])
    return digest.hexdigest()


def audit(
    *,
    remote_repo: str,
    remote_filename: str,
    local_safetensors: str | Path,
    local_model_id: str | None = None,
    prefix: str = DEFAULT_PREFIX,
    max_rank: int = 1,
    workers: int = 8,
    retries: int = 5,
    timeout_seconds: int = 30,
    hash_local: bool = False,
) -> dict[str, Any]:
    local_path = Path(local_safetensors).expanduser().resolve()
    if not local_path.is_file() or local_path.stat().st_size <= 0:
        raise RetainedConstantAuditError(
            f"Local Safetensors checkpoint is missing or empty: {local_path}"
        )
    metadata = _metadata_with_retry(remote_repo, remote_filename, retries)
    location = str(getattr(metadata, "location", "") or "")
    if not location:
        raise RetainedConstantAuditError("Hugging Face metadata has no download location.")

    remote_header_size, remote_header = _read_remote_header(
        location, retries=retries, timeout_seconds=timeout_seconds
    )
    local_header_size, local_header = _read_local_header(local_path)
    remote_selected = _select_entries(
        remote_header, prefix=prefix, max_rank=max_rank
    )
    if not remote_selected:
        raise RetainedConstantAuditError(
            f"No remote tensors matched prefix={prefix!r}, max_rank={max_rank}."
        )

    schema_mismatches: list[dict[str, Any]] = []
    comparable: dict[str, dict[str, Any]] = {}
    for name, remote_entry in remote_selected.items():
        local_entry = local_header.get(name)
        if not isinstance(local_entry, dict):
            schema_mismatches.append({"name": name, "reason": "missing_local_tensor"})
            continue
        remote_schema = _entry_schema(remote_entry)
        local_schema = _entry_schema(local_entry)
        if remote_schema != local_schema:
            schema_mismatches.append(
                {
                    "name": name,
                    "reason": "dtype_or_shape_mismatch",
                    "remote": {
                        "dtype": remote_schema[0],
                        "shape": list(remote_schema[1]),
                    },
                    "local": {
                        "dtype": local_schema[0],
                        "shape": list(local_schema[1]),
                    },
                }
            )
            continue
        comparable[name] = remote_entry

    groups = _coalesce_remote_ranges(
        comparable, data_start=8 + remote_header_size
    )
    remote_values = _fetch_remote_entries(
        location,
        groups,
        workers=workers,
        retries=retries,
        timeout_seconds=timeout_seconds,
    )
    local_comparable = {name: local_header[name] for name in comparable}
    local_values = _read_local_entries(
        local_path, local_header_size, local_comparable
    )

    mismatches: list[dict[str, Any]] = []
    exact_count = 0
    for name in sorted(comparable):
        remote_value = remote_values[name]
        local_value = local_values[name]
        if remote_value == local_value:
            exact_count += 1
            continue
        first_difference = next(
            (
                index
                for index, (left, right) in enumerate(zip(remote_value, local_value))
                if left != right
            ),
            min(len(remote_value), len(local_value)),
        )
        mismatches.append(
            {
                "name": name,
                "size_bytes": len(remote_value),
                "first_differing_byte": first_difference,
                "remote_sha256": hashlib.sha256(remote_value).hexdigest(),
                "local_sha256": hashlib.sha256(local_value).hexdigest(),
            }
        )

    selected_bytes = sum(len(value) for value in remote_values.values())
    all_exact = bool(
        comparable
        and not schema_mismatches
        and not mismatches
        and exact_count == len(remote_selected)
    )
    return {
        "ok": all_exact,
        "training_executed": False,
        "scope": {
            "remote": f"{remote_repo}/{remote_filename}",
            "local": str(local_path),
            "prefix": prefix,
            "max_rank": max_rank,
            "excluded_suffixes": list(DEFAULT_EXCLUDED_SUFFIXES),
            "interpretation": (
                "Exact bytes prove compatibility only for public checkpoint tensors "
                "selected by this audit. They do not map those tensors into a compiled "
                "LiteRT graph or recover Google's private QAT/export recipe."
            ),
        },
        "remote": {
            "repo_id": remote_repo,
            "filename": remote_filename,
            "file_size_bytes": int(getattr(metadata, "size", 0) or 0),
            "etag": str(getattr(metadata, "etag", "") or ""),
            "commit_hash": str(getattr(metadata, "commit_hash", "") or ""),
            "header_size_bytes": remote_header_size,
        },
        "local": {
            "path": str(local_path),
            "model_id": str(local_model_id or "") or None,
            "file_size_bytes": local_path.stat().st_size,
            "header_size_bytes": local_header_size,
            "sha256": _sha256_file(local_path) if hash_local else None,
        },
        "selection": {
            "remote_selected_tensor_count": len(remote_selected),
            "comparable_tensor_count": len(comparable),
            "range_request_count": len(groups),
            "downloaded_tensor_bytes": selected_bytes,
            "schema_mismatch_count": len(schema_mismatches),
            "schema_mismatches": schema_mismatches,
        },
        "comparison": {
            "exact_tensor_count": exact_count,
            "value_mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "remote_aggregate_sha256": _aggregate_digest(comparable, remote_values),
            "local_aggregate_sha256": _aggregate_digest(
                local_comparable, local_values
            ),
            "all_selected_tensors_exact": all_exact,
        },
        "private_qat_recipe_recovered": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare bounded retained constants against a remote HF checkpoint."
    )
    parser.add_argument("--remote-repo", required=True)
    parser.add_argument("--remote-filename", default="model.safetensors")
    parser.add_argument("--local-safetensors", required=True)
    parser.add_argument(
        "--local-model-id",
        help="Public model id represented by --local-safetensors (recorded as provenance).",
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--max-rank", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--hash-local", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.max_rank < 0:
        parser.error("--max-rank cannot be negative")
    if args.workers < 1 or args.retries < 1 or args.timeout_seconds < 1:
        parser.error("--workers, --retries, and --timeout-seconds must be positive")
    try:
        result = audit(
            remote_repo=args.remote_repo,
            remote_filename=args.remote_filename,
            local_safetensors=args.local_safetensors,
            local_model_id=args.local_model_id,
            prefix=args.prefix,
            max_rank=args.max_rank,
            workers=args.workers,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
            hash_local=args.hash_local,
        )
    except (OSError, RetainedConstantAuditError) as exc:
        parser.error(str(exc))
        return 2
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["ok"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
