import hashlib
import re
from typing import Iterable

_punct_re = re.compile(r"[^a-z0-9\s]")
_space_re = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = _punct_re.sub(" ", lowered)
    lowered = _space_re.sub(" ", lowered)
    return lowered.strip()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(prefix: str, idx: int, width: int = 6) -> str:
    return f"{prefix}_{idx:0{width}d}"


def hash_list(items: Iterable[str]) -> str:
    h = hashlib.sha256()
    for item in items:
        h.update(item.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()
