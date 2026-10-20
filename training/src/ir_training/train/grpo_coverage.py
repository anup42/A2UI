"""Read-only per-rank data exposure telemetry, separate from reward/selection."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def dataset_coverage(rows: Any) -> dict:
    intents, lengths = Counter(), Counter()
    sources = set()
    for row in rows:
        sources.add(str(row.get("source_id") or hashlib.sha256(str(row.get("response_text", "")).encode()).hexdigest()))
        intents[str(row.get("intent_bucket") or "unknown")] += 1
        length = (row.get("prompt_provenance") or {}).get("prompt_tokens")
        bucket = "unknown" if length is None else str((int(length) // 1024) * 1024) + ".." + str((int(length) // 1024 + 1) * 1024 - 1)
        lengths[bucket] += 1
    return {"rows": len(rows), "unique_sources": len(sources), "intent_rows": dict(intents),
            "prompt_token_buckets": dict(lengths), "scope": "prepared data, before sampling; no synthetic rows"}


def with_coverage(reward, output: Path, rank: int):
    seen, counts, intents = defaultdict(set), Counter(), defaultdict(Counter)

    def tracked(*args, **kwargs):
        values = reward(*args, **kwargs)
        n = len(values)
        def vector(key, default):
            raw = kwargs.get(key)
            if raw is None:
                return [default] * n
            if isinstance(raw, str):
                return [raw] * n
            if len(raw) != n:
                raise ValueError(f"GRPO coverage metadata does not align: {key}")
            return raw
        sources = vector("source_id", None)
        prompts = vector("response_text", "")
        for source, text, intent, mode in zip(sources, prompts, vector("intent_bucket", "unknown"),
                                             vector("grpo_rollout_mode", "unknown"), strict=True):
            key = str(mode)
            seen[key].add(hashlib.sha256(str(source or text).encode()).hexdigest())
            counts[key] += 1
            intents[key][str(intent or "unknown")] += 1
        report = {"rank": rank, "step": getattr(kwargs.get("trainer_state"), "global_step", None),
                  "scope": "rank-local lifetime candidate exposure; do not sum unique counts across ranks",
                  "modes": {mode: {"candidates": counts[mode], "unique_sources": len(ids),
                                   "intent_candidates": dict(intents[mode])}
                            for mode, ids in seen.items()}}
        path = output / f"grpo_source_exposure.rank{rank}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return values

    tracked.__name__ = getattr(reward, "__name__", "qat_grpo_reward")
    return tracked
