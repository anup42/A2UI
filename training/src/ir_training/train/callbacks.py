from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TrainingMetadataCallback:
    """Small callback-like helper for writing immutable run metadata."""

    def __init__(self, output_dir: str | Path, metadata: dict[str, Any]) -> None:
        self.output_dir = Path(output_dir)
        self.metadata = metadata

    def write(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "training_metadata.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
