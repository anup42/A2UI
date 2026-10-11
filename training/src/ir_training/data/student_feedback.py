"""Train-only replay and preference decisions; no models, teacher calls or repairs.

The integration layer supplies source-family IDs and the *existing* v5.4 scorer.
This module never accepts an externally supplied quality/eligibility label.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any, Callable, Mapping, Sequence

VERSION = "student_feedback_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FeedbackOptions:
    seed: int = 42
    max_extra_token_fraction: float = 0.10
    max_replays_per_family: int = 1
    min_quality_gap: float = 0.05

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if type(self.max_replays_per_family) is not int or not 1 <= self.max_replays_per_family <= 3:
            raise ValueError("max_replays_per_family must be in [1, 3]")
        for name in ("max_extra_token_fraction", "min_quality_gap"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.max_extra_token_fraction <= 0.5:
            raise ValueError("max_extra_token_fraction must be in (0, 0.5]")
        if not 0 < self.min_quality_gap <= 1:
            raise ValueError("min_quality_gap must be in (0, 1]")


def sequence_tokens(row: Mapping[str, Any]) -> int:
    value = row.get("metadata", {}).get("express_preparation", {}).get("tokenization", {}).get("sequence_tokens")
    if type(value) is not int or value < 1:
        raise ValueError("A checked prepared sequence_tokens count is required")
    return value


def checked_score(value: Mapping[str, Any]) -> dict[str, Any]:
    """Scorer failures must not become low scores or manufactured negatives."""
    value = dict(value)
    if type(value.get("eligible")) is not bool or type(value.get("strict_valid")) is not bool:
        raise ValueError("Scorer must return explicit eligibility and strict validity")
    quality = value.get("quality")
    if isinstance(quality, bool) or not isinstance(quality, Real) or not math.isfinite(quality) or not 0 <= quality <= 1:
        raise ValueError("Scorer quality must be finite in [0, 1]")
    for name in ("blocking_reasons", "review_reasons"):
        reasons = value.get(name)
        if not isinstance(reasons, list) or any(not isinstance(reason, str) or not reason for reason in reasons):
            raise ValueError("Scorer reasons must be lists of nonempty strings")
    if not isinstance(value.get("metric_fingerprint"), str) or not value["metric_fingerprint"]:
        raise ValueError("Scorer fingerprint is required")
    if value["eligible"] and (not value["strict_valid"] or value["blocking_reasons"] or value["review_reasons"]):
        raise ValueError("Inconsistent scorer admission decision")
    return value


def select_probe_indices(rows: Sequence[Mapping[str, Any]], families: Sequence[str], *,
                         max_rows: int, seed: int) -> list[int]:
    """One source-family representative; round-robin intent/length strata."""
    if len(rows) != len(families) or not rows:
        raise ValueError("Nonempty rows and aligned source families are required")
    if type(max_rows) is not int or max_rows < 1 or type(seed) is not int:
        raise ValueError("Invalid probe row budget/seed")
    candidates = sorted(range(len(rows)), key=lambda i: fingerprint([seed, fingerprint(rows[i])]))
    seen: set[str] = set()
    buckets: dict[tuple[str, int], deque[int]] = defaultdict(deque)
    for i in candidates:
        if families[i] in seen:
            continue
        seen.add(families[i])
        intent = str(rows[i].get("intent_bucket") or rows[i].get("metadata", {}).get("intent_bucket") or "unknown")
        buckets[intent, sequence_tokens(rows[i]) // 1024].append(i)
    result = []
    while buckets and len(result) < max_rows:
        for key in sorted(list(buckets)):
            result.append(buckets[key].popleft())
            if not buckets[key]:
                del buckets[key]
            if len(result) == max_rows:
                break
    return result


def build_feedback(rows: Sequence[Mapping[str, Any]], families: Sequence[str],
                   observations: Sequence[Mapping[str, Any]], *,
                   score: Callable[[Mapping[str, Any], str], Mapping[str, Any]],
                   options: FeedbackOptions) -> dict[str, Any]:
    """Rescore positives and actual student outputs on the exact same source.

    Truncated/unknown-stop outputs can identify replay cases but are never DPO
    negatives. Review-only failures need an explicit quality margin; an accepted
    alternative UI is never negative, regardless of its score or serialization.
    """
    if not rows or len(families) != len(rows):
        raise ValueError("Nonempty rows and aligned source families are required")
    lengths = [sequence_tokens(row) for row in rows]
    budget = math.floor(sum(lengths) * options.max_extra_token_fraction)
    records, hard, seen = [], {}, set()
    metric = None
    for observation in observations:
        index = observation.get("train_index")
        if type(index) is not int or not 0 <= index < len(rows) or index in seen:
            raise ValueError("Probe indices must be unique and belong to the bound train split")
        seen.add(index)
        row = rows[index]
        if observation.get("row_sha256") != fingerprint(row):
            raise ValueError("Student observation does not match the original training row")
        candidate = observation.get("completion")
        prompt_hash = observation.get("prompt_sha256")
        if not isinstance(candidate, str) or not isinstance(prompt_hash, str) or len(prompt_hash) != 64 or any(c not in "0123456789abcdef" for c in prompt_hash):
            raise ValueError("Student completion and bound prompt hash are required")
        chosen = row.get("completion")
        if not isinstance(chosen, str) or not chosen.strip():
            raise ValueError("Prepared reference completion is required")
        positive = checked_score(score(row, chosen))
        negative = checked_score(score(row, candidate))
        if positive["metric_fingerprint"] != negative["metric_fingerprint"]:
            raise ValueError("Chosen/rejected metric fingerprints differ")
        if metric is not None and metric != positive["metric_fingerprint"]:
            raise ValueError("Scorer changed during feedback construction")
        metric = positive["metric_fingerprint"]
        gap = positive["quality"] - negative["quality"]
        reasons = negative["blocking_reasons"] + negative["review_reasons"]
        failed = not negative["eligible"]
        status = ("reference_not_eligible" if not positive["eligible"] else
                  "student_accepted" if not failed else "hard_example")
        record = {"train_index": index, "row_sha256": fingerprint(row), "family": families[index],
                  "prompt_sha256": prompt_hash, "candidate_sha256": text_hash(candidate),
                  "stop_reason": observation.get("stop_reason"), "chosen": positive,
                  "student": negative, "quality_gap": gap, "status": status}
        records.append(record)
        if status != "hard_example":
            continue
        # No generic generation-error exceptions are swallowed by the caller.
        # A missing reason indicates a broken scorer rather than a useful case.
        if not reasons:
            raise ValueError("Ineligible student output has no diagnostic reason")
        pair_ok = (candidate.strip() != chosen.strip() and bool(candidate.strip())
                   and observation.get("stop_reason") in {"eos_token", "closing_sentinel"}
                   and (bool(negative["blocking_reasons"]) or gap >= options.min_quality_gap))
        record["pair_eligible"] = pair_ok
        record["pair_exclusion"] = (None if pair_ok else "truncated_empty_identical_or_ambiguous")
        hard[index] = (record, candidate)

    # Balance failure categories before deterministic tie-breaking. Replays are
    # bounded by *tokens*, not merely row count; long cases are not truncated.
    buckets: dict[str, deque[int]] = defaultdict(deque)
    for index in sorted(hard, key=lambda i: fingerprint([options.seed, fingerprint(rows[i])])):
        record, _ = hard[index]
        reasons = record["student"]["blocking_reasons"] + record["student"]["review_reasons"]
        buckets[sorted(reasons)[0]].append(index)
    replay, spent, exposures = [], 0, Counter()
    while buckets:
        for category in sorted(list(buckets)):
            index = buckets[category].popleft()
            if not buckets[category]:
                del buckets[category]
            family = families[index]
            if exposures[family] >= options.max_replays_per_family or spent + lengths[index] > budget:
                continue
            replay.append(index)
            exposures[family] += 1
            spent += lengths[index]
    preferences = []
    for index in replay:
        record, candidate = hard[index]
        if record["pair_eligible"]:
            pair = {"train_index": index, "row_sha256": fingerprint(rows[index]),
                    "prompt_sha256": record["prompt_sha256"], "chosen": rows[index]["completion"],
                    "rejected": candidate, "rejected_stop_reason": record["stop_reason"],
                    "quality_gap": record["quality_gap"], "metric_fingerprint": metric,
                    "reasons": record["student"]["blocking_reasons"] + record["student"]["review_reasons"]}
            pair["pair_id"] = fingerprint(pair)
            preferences.append(pair)
    report = {"version": VERSION, "options": asdict(options), "metric_fingerprint": metric,
              "original_rows": len(rows), "observed_rows": len(records), "hard_rows": len(hard),
              "replay_rows": len(replay), "preference_pairs": len(preferences),
              "original_sequence_tokens": sum(lengths), "replay_sequence_tokens": spent,
              "extra_sequence_token_budget": budget, "status_counts": dict(Counter(r["status"] for r in records)),
              "failure_counts": dict(Counter(reason for r in records for reason in
                    r["student"]["blocking_reasons"] + r["student"]["review_reasons"])),
              "targets_modified": False, "evaluation_sources_used": False,
              "quality_claim": "Automatic training evidence only; post-training/native gain is unmeasured."}
    return {"report": report, "replay_indices": replay, "preferences": preferences, "diagnostics": records}


def exposure_schedule(rows, families, replay_indices, *, total_examples, seed=42,
                      max_hard_token_fraction=0.10, max_hard_exposures_per_family=3):
    """A matched finite-run control schedule with bounded hard replacements.

    Keep example/optimizer counts identical across A/B/C. Exact token counts can
    differ and are reported. Hard exposure is capped relative to the control's
    full sequence tokens; it is not an unbounded weighted sampler.
    """
    import random
    if (type(total_examples) is not int or total_examples < 1 or not rows or len(families) != len(rows)
            or type(seed) is not int or type(max_hard_exposures_per_family) is not int
            or not 1 <= max_hard_exposures_per_family <= 10):
        raise ValueError("Invalid exposure schedule budget")
    if (isinstance(max_hard_token_fraction, bool) or not isinstance(max_hard_token_fraction, Real)
            or not math.isfinite(max_hard_token_fraction) or not 0 < max_hard_token_fraction <= .5):
        raise ValueError("Hard token fraction must be finite in (0, .5]")
    if len(set(replay_indices)) != len(replay_indices) or any(type(i) is not int or not 0 <= i < len(rows) for i in replay_indices):
        raise ValueError("Replay indices must be unique training indices")
    rng = random.Random(seed)
    baseline = []
    while len(baseline) < total_examples:
        cycle = list(range(len(rows)))
        rng.shuffle(cycle)
        baseline.extend(cycle[:total_examples - len(baseline)])
    lengths = [sequence_tokens(row) for row in rows]
    control_tokens = sum(lengths[i] for i in baseline)
    budget = math.floor(control_tokens * max_hard_token_fraction)
    replayed, hard_mask = list(baseline), [False] * total_examples
    positions = list(range(total_examples))
    rng.shuffle(positions)
    used, spent, cursor = Counter(), 0, 0
    candidates = list(replay_indices)
    rng.shuffle(candidates)
    for _ in range(max_hard_exposures_per_family):
        for index in candidates:
            family = families[index]
            if used[family] >= max_hard_exposures_per_family or spent + lengths[index] > budget or cursor == len(positions):
                continue
            position = positions[cursor]
            cursor += 1
            replayed[position] = index
            hard_mask[position] = True
            used[family] += 1
            spent += lengths[index]
    return {"baseline_indices": baseline, "replay_indices": replayed, "hard_mask": hard_mask,
            "report": {"total_examples": total_examples, "hard_examples": sum(hard_mask),
                       "control_sequence_tokens": control_tokens,
                       "replay_sequence_tokens": sum(lengths[i] for i in replayed),
                       "hard_sequence_tokens": spent, "hard_token_budget": budget,
                       "hard_token_fraction_of_control": spent / control_tokens,
                       "max_hard_exposures_per_family": max_hard_exposures_per_family,
                       "exact_token_budgets_matched": False}}
