from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _parse_semver(version: str) -> tuple[int, int, int]:
    parts = version.strip().split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid semver '{version}'. Expected MAJOR.MINOR.PATCH")
    try:
        major, minor, patch = (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as exc:
        raise ValueError(f"Invalid semver '{version}'. Expected integer parts") from exc
    return major, minor, patch


def _bump(version: str, part: str) -> str:
    major, minor, patch = _parse_semver(version)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"Unsupported bump part: {part}")


def _load_components_yaml(path: Path) -> dict:
    if not path.exists():
        return {"release": "0.0.0", "components": {}, "compatibility": {}}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {"release": "0.0.0", "components": {}, "compatibility": {}}
    raw.setdefault("release", "0.0.0")
    raw.setdefault("components", {})
    raw.setdefault("compatibility", {})
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Bump dataset release/component versions")
    parser.add_argument("--release", type=str, default=None, help="Set explicit release version (MAJOR.MINOR.PATCH)")
    parser.add_argument("--bump", choices=["major", "minor", "patch"], default=None, help="Bump release by semver part")
    parser.add_argument(
        "--component",
        action="append",
        default=[],
        help="Set component version as name=version. Repeatable.",
    )
    parser.add_argument(
        "--set-compat",
        action="append",
        default=[],
        help="Set compatibility key as key=value. Repeatable.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing files")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    version_path = root / "VERSION"
    components_path = root / "versions" / "components.yaml"

    current_release = version_path.read_text(encoding="utf-8").strip() if version_path.exists() else "0.0.0"
    if not current_release:
        current_release = "0.0.0"

    if args.release and args.bump:
        raise SystemExit("Use either --release or --bump, not both")

    next_release = current_release
    if args.release:
        _parse_semver(args.release)
        next_release = args.release
    elif args.bump:
        next_release = _bump(current_release, args.bump)

    doc = _load_components_yaml(components_path)
    doc["release"] = next_release

    for entry in args.component:
        if "=" not in entry:
            raise SystemExit(f"Invalid --component '{entry}'. Use name=version")
        name, version = entry.split("=", 1)
        name = name.strip()
        version = version.strip()
        if not name or not version:
            raise SystemExit(f"Invalid --component '{entry}'. Use name=version")
        doc["components"][name] = version

    for entry in args.set_compat:
        if "=" not in entry:
            raise SystemExit(f"Invalid --set-compat '{entry}'. Use key=value")
        key, value = entry.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise SystemExit(f"Invalid --set-compat '{entry}'. Use key=value")
        doc["compatibility"][key] = value

    if args.dry_run:
        print(f"VERSION: {current_release} -> {next_release}")
        print(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
        return

    version_path.write_text(next_release + "\n", encoding="utf-8")
    components_path.parent.mkdir(parents=True, exist_ok=True)
    components_path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(f"Updated {version_path}")
    print(f"Updated {components_path}")


if __name__ == "__main__":
    main()
