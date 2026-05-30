from __future__ import annotations

import argparse
import os
from pathlib import Path


def _variants(model_id: str) -> list[str]:
    raw = (model_id or "").strip().strip("/")
    if not raw:
        return []
    leaf = raw.rsplit("/", 1)[-1]
    variants = [
        raw,
        raw.replace("/", "--"),
        raw.replace("/", "__"),
        raw.replace("/", "_"),
        leaf,
        leaf.replace("_", "-"),
        leaf.replace("-", "_"),
    ]
    lowered = [item.lower() for item in variants]
    upper_b = [item.replace("-31b-", "-31B-") for item in variants + lowered]
    deduped: list[str] = []
    for item in variants + lowered + upper_b:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def _candidate_paths(root: Path, model_id: str) -> list[Path]:
    candidates: list[Path] = []
    for item in _variants(model_id):
        path = root / Path(item)
        candidates.append(path)
    return candidates


def resolve_model(root_values: list[str], model_id: str) -> Path | None:
    roots: list[Path] = []
    for raw in root_values:
        for part in (raw or "").split(os.pathsep):
            normalized = os.path.expandvars(os.path.expanduser(part.strip()))
            if normalized:
                roots.append(Path(normalized))

    for root in roots:
        if not root.exists():
            continue
        if (root / "config.json").is_file():
            return root
        exact = root / Path(model_id.strip("/"))
        if exact.is_dir() and (exact / "config.json").is_file():
            return exact
        for path in _candidate_paths(root, model_id):
            if path.is_dir() and (path / "config.json").is_file():
                return path
        for path in _candidate_paths(root, model_id):
            if path.is_dir():
                return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve a local model folder from shared model root(s).")
    parser.add_argument("--model", required=True, help="Model id from dataset/configs/models.yaml, e.g. google/gemma-4-31b-it")
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Model root directory. Can be repeated; each value may also contain os.pathsep-separated roots.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print diagnostics on failure.",
    )
    args = parser.parse_args()

    env_roots = [
        os.environ.get("MODEL_ROOT", ""),
        os.environ.get("LOCAL_MODEL_ROOT", ""),
        os.environ.get("A2UI_MODEL_ROOT", ""),
        os.environ.get("GEMMA4_MODEL_ROOT", ""),
        os.environ.get("QWEN_MODEL_ROOT", ""),
        os.environ.get("DEEPSEEK_MODEL_ROOT", ""),
    ]
    resolved = resolve_model([*args.root, *env_roots], args.model)
    if resolved:
        print(resolved)
        return 0
    if not args.quiet:
        roots = [item for item in [*args.root, *env_roots] if item]
        print(f"Could not resolve model '{args.model}' from roots: {roots}", file=__import__("sys").stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
