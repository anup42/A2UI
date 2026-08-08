from __future__ import annotations

import platform
import sys
from importlib import metadata
from typing import Any


# Exact versions used for the host graph rebuilds and Android GPU parity runs
# recorded in docs/gemma4_e2b_qat_mtp_knowledge.md. A different version is not
# automatically wrong, but it is a different unverified conversion toolchain.
TESTED_EDGE_EXPORT_TOOLCHAIN = {
    "litert-torch": "0.9.3",
    "ai-edge-quantizer": "0.8.0",
    "ai-edge-litert": "2.1.6",
    "litert-lm-builder": "0.15.0",
    "flatbuffers": "25.12.19",
    "numpy": "2.4.4",
    "scipy": "1.17.1",
    "ml-dtypes": "0.5.4",
    "safetensors": "0.8.0",
}


def edge_export_toolchain_report() -> dict[str, Any]:
    """Describe and compare the installed public conversion environment."""

    packages: list[dict[str, Any]] = []
    for name, expected in TESTED_EDGE_EXPORT_TOOLCHAIN.items():
        try:
            observed = metadata.version(name)
        except metadata.PackageNotFoundError:
            observed = None
        packages.append(
            {
                "name": name,
                "expected": expected,
                "observed": observed,
                "match": observed == expected,
            }
        )
    exact = all(item["match"] for item in packages)
    return {
        "contract_version": 1,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": sys.platform,
        "packages": packages,
        "tested_versions_match": exact,
        "interpretation": (
            "Exact tested versions are reproduced. Graph, package, and device "
            "promotion gates are still required."
            if exact
            else "This is an unverified toolchain version set. Exact topology "
            "and Android gates may still prove compatibility, but do not reuse "
            "the prior toolchain claim."
        ),
    }
