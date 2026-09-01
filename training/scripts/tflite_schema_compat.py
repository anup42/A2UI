"""Compatibility loader for generated TFLite FlatBuffer schema classes.

The public AI Edge LiteRT wheel ships the generated schema as one module,
``ai_edge_litert.schema_py_generated``.  Older tooling and the parity scripts
use the equivalent split ``tflite.<ClassName>`` modules.  Keep the scripts
usable with either installation without adding a second schema wheel.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any


def schema_module(class_name: str) -> Any:
    """Return a module-like object exposing the requested generated class."""

    try:
        return importlib.import_module(f"tflite.{class_name}")
    except ImportError:
        schema = importlib.import_module("ai_edge_litert.schema_py_generated")
        generated = getattr(schema, class_name)
        # The split wheel exposes module-level helpers such as
        # ``BufferStart`` alongside ``Buffer.BufferStart``.  Mirror both
        # shapes so callers can keep using the legacy generated-module API.
        attributes = {
            name: getattr(generated, name)
            for name in dir(generated)
            if not name.startswith("_")
        }
        attributes.update(
            {
                name: getattr(schema, name)
                for name in dir(schema)
                if name.startswith(class_name)
            }
        )
        attributes[class_name] = generated
        return SimpleNamespace(**attributes)
