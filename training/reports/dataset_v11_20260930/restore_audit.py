#!/usr/bin/env python3
"""Verify or restore the supplied audit files, including compressed ledgers."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile


def verify_or_restore(bundle: Path, output: Path | None = None) -> int:
    bundle = bundle.resolve()
    if output is not None and output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    package = json.loads((bundle / 'source_package_manifest.json').read_text(encoding='utf-8'))
    imported = json.loads((bundle / 'repository_import.json').read_text(encoding='utf-8'))
    compressed = imported['compressed_audit_ledgers']
    prefix = 'training/reports/dataset_v11_20260930/'
    entries = {key[len(prefix):]: value for key, value in package['files'].items() if key.startswith(prefix)}
    temporary = None
    if output is not None:
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=output.name + '.partial-', dir=output.parent))
    try:
        for relative, expected in entries.items():
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError(f'Invalid audit path: {relative}')
            source = bundle / relative
            archive = compressed.get(relative)
            if archive:
                source = bundle / archive['compressed_path']
                if not source.resolve().is_relative_to(bundle):
                    raise ValueError(f'Invalid compressed path: {relative}')
                with source.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                if source.stat().st_size != archive['compressed_bytes'] or digest != archive['compressed_sha256']:
                    raise ValueError(f'Compressed audit checksum mismatch: {relative}')
            elif not source.resolve().is_relative_to(bundle):
                raise ValueError(f'Invalid audit path: {relative}')
            destination = temporary / relative if temporary else None
            if destination:
                destination.parent.mkdir(parents=True, exist_ok=True)
            digest, size = hashlib.sha256(), 0
            with (gzip.open(source, 'rb') if archive else source.open('rb')) as stream:
                out = destination.open('wb') if destination else None
                try:
                    while block := stream.read(4 * 1024 * 1024):
                        digest.update(block)
                        size += len(block)
                        if out:
                            out.write(block)
                finally:
                    if out:
                        out.close()
            if size != expected['bytes'] or digest.hexdigest() != expected['sha256']:
                raise ValueError(f'Original audit checksum mismatch: {relative}')
        if temporary:
            temporary.rename(output)
        return len(entries)
    except Exception:
        if temporary:
            print(f'Incomplete audit retained for diagnosis: {temporary}')
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument('--verify-only', action='store_true')
    options.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    count = verify_or_restore(Path(__file__).resolve().parent, args.output_dir)
    print(f'Verified {count} supplied audit files' + (f'; restored to {args.output_dir}' if args.output_dir else ''))


if __name__ == '__main__':
    main()
