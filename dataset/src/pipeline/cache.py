from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from llm.base import extract_reasoning_metadata


@dataclass
class CacheEntry:
    text: str
    raw: Any
    reasoning_text: Optional[str] = None
    reasoning_source: Optional[str] = None
    reasoning_tokens: Optional[int] = None


class PromptCache:
    def __init__(self, path: Path, enabled: bool | None = None):
        self.path = path
        self.enabled = _prompt_cache_enabled() if enabled is None else bool(enabled)
        self._cache: dict[str, CacheEntry] = {}
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = obj.get("prompt_hash")
                if key:
                    reasoning_text = obj.get("reasoning_text")
                    reasoning_source = obj.get("reasoning_source")
                    reasoning_tokens = obj.get("reasoning_tokens")
                    if not reasoning_text:
                        reasoning_text, reasoning_source, reasoning_tokens = (
                            extract_reasoning_metadata(obj.get("raw"))
                        )
                    self._cache[key] = CacheEntry(
                        text=obj.get("text", ""),
                        raw=obj.get("raw"),
                        reasoning_text=reasoning_text,
                        reasoning_source=reasoning_source,
                        reasoning_tokens=reasoning_tokens,
                    )

    def get(self, prompt_hash: str) -> Optional[CacheEntry]:
        if not self.enabled:
            return None
        return self._cache.get(prompt_hash)

    def set(
        self,
        prompt_hash: str,
        text: str,
        raw: Any,
        reasoning_text: Optional[str] = None,
        reasoning_source: Optional[str] = None,
        reasoning_tokens: Optional[int] = None,
    ) -> None:
        if not self.enabled:
            return
        if prompt_hash in self._cache:
            return
        entry = {"prompt_hash": prompt_hash, "text": text, "raw": raw}
        if reasoning_text:
            entry["reasoning_text"] = reasoning_text
            entry["reasoning_source"] = reasoning_source
            entry["reasoning_tokens"] = reasoning_tokens
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._cache[prompt_hash] = CacheEntry(
            text=text,
            raw=raw,
            reasoning_text=reasoning_text,
            reasoning_source=reasoning_source,
            reasoning_tokens=reasoning_tokens,
        )


def _prompt_cache_enabled() -> bool:
    raw = (
        os.environ.get("A2UI_ENABLE_PROMPT_CACHE")
        or os.environ.get("DATASET_ENABLE_PROMPT_CACHE")
        or ""
    ).strip().lower()
    return raw in {"1", "true", "yes", "on"}
