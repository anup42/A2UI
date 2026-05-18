from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    resolved = Path(path)
    if not resolved.exists():
        return
    with resolved.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                yield item


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with resolved.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def load_by_key(path: str | Path, key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        value = row.get(key)
        if isinstance(value, str) and value:
            out[value] = row
    return out
