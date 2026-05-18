from __future__ import annotations

import subprocess
from pathlib import Path


def current_commit(cwd: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None
