from __future__ import annotations

import shutil
from pathlib import Path


def copy_hf_bundle(source_dir: str | Path, output_dir: str | Path) -> Path:
    source = Path(source_dir)
    out = Path(output_dir)
    if not source.exists():
        raise FileNotFoundError(str(source))
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(source, out)
    return out
