"""Reject experimental Compose flow-layout references in a published SDK AAR.

This checks the compiled artifact, including generated composable lambdas. It is
a regression guard for the known FlowRow binary break, not a general guarantee
of compatibility with every Compose version. Run device smoke tests separately.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile


FORBIDDEN_TYPES = (
    "androidx/compose/foundation/layout/FlowLayoutKt",
    "androidx/compose/foundation/layout/FlowRowOverflow",
    "androidx/compose/foundation/layout/FlowColumnOverflow",
    "androidx/compose/foundation/layout/FlowRowScope",
    "androidx/compose/foundation/layout/FlowColumnScope",
)


def inspect_aar(path: Path) -> dict:
    aar_bytes = path.read_bytes()
    findings = []
    class_count = 0
    with zipfile.ZipFile(io.BytesIO(aar_bytes)) as aar:
        jars = [name for name in aar.namelist() if name == "classes.jar" or
                (name.startswith("libs/") and name.endswith(".jar"))]
        if "classes.jar" not in jars:
            raise ValueError("AAR has no classes.jar")
        for jar_name in jars:
            with zipfile.ZipFile(io.BytesIO(aar.read(jar_name))) as jar:
                for entry in jar.infolist():
                    if not entry.filename.endswith(".class"):
                        continue
                    class_count += 1
                    bytecode = jar.read(entry)
                    matches = [name for name in FORBIDDEN_TYPES if name.encode() in bytecode]
                    if matches:
                        findings.append({"jar": jar_name, "class": entry.filename,
                                         "references": matches})
    return {"aar": path.name, "sha256": hashlib.sha256(aar_bytes).hexdigest(),
            "class_count": class_count, "passed": not findings, "findings": findings}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aar", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = inspect_aar(args.aar)
    output = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "findings"}))
    if result["findings"]:
        print(f'{len(result["findings"])} classes reference incompatible Compose flow APIs.')
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
