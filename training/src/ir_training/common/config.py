from __future__ import annotations

from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def training_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    anchor = base or training_root()
    return (anchor / path).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError("PyYAML is required. Install training/requirements-training.txt") from exc
    resolved = resolve_path(path, Path.cwd()) if not Path(path).is_absolute() else Path(path)
    with resolved.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML object: {resolved}")
    return data
