#!/usr/bin/env python3
"""Benchmark lossless GenUICraft IR encodings on JSON/JSONL corpora.

The script always reports exact UTF-8 bytes and characters. Token counts use,
in priority order, an explicitly supplied SentencePiece model, a local
HuggingFace tokenizer, tiktoken when requested/available, or a deterministic
regex lexical estimate. The report labels the tokenizer mode so estimates are
never presented as deployed-model measurements.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Callable, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
DATASET_SRC = ROOT / "dataset" / "src"
if str(DATASET_SRC) not in sys.path:
    sys.path.insert(0, str(DATASET_SRC))

from pipeline.flat_spec_contract import coerce_and_validate  # noqa: E402
from pipeline.ir_formats import (  # noqa: E402
    A2UI_EXPRESS_V1,
    A2UI_V1_WIRE,
    COMPACT_IR_V2,
    FLAT_SPEC_V1,
    codec_identity,
    decode_to_flat_spec,
    encode_from_flat_spec,
    semantic_hash,
    serialized_text,
)

FORMATS = (FLAT_SPEC_V1, COMPACT_IR_V2, A2UI_EXPRESS_V1, A2UI_V1_WIRE)
DEFAULT_CORPUS = ROOT / "dataset" / "tests" / "fixtures" / "intent_flat_specs_v2.json"


@dataclass(frozen=True)
class TokenCounter:
    name: str
    exact_for_deployed_model: bool
    count: Callable[[str], int]


def _regex_count(text: str) -> int:
    # Stable fallback that does not collapse minified JSON to one whitespace token.
    return len(re.findall(r"[\w]+|[^\w\s]", text, flags=re.UNICODE))


def _build_token_counter(args: argparse.Namespace) -> TokenCounter:
    if args.sentencepiece_model:
        import sentencepiece as spm  # type: ignore

        processor = spm.SentencePieceProcessor(model_file=str(Path(args.sentencepiece_model)))
        return TokenCounter(
            name=f"sentencepiece:{Path(args.sentencepiece_model).name}",
            exact_for_deployed_model=True,
            count=lambda text: len(processor.encode(text, out_type=int)),
        )

    if args.hf_tokenizer:
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise SystemExit(f"transformers is required for --hf-tokenizer: {exc}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.hf_tokenizer,
            local_files_only=not args.allow_tokenizer_download,
            use_fast=True,
        )
        return TokenCounter(
            name=f"huggingface:{args.hf_tokenizer}",
            exact_for_deployed_model=True,
            count=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
        )

    if args.tiktoken_encoding:
        try:
            import tiktoken  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise SystemExit(f"tiktoken is required for --tiktoken-encoding: {exc}")
        encoding = tiktoken.get_encoding(args.tiktoken_encoding)
        return TokenCounter(
            name=f"tiktoken:{args.tiktoken_encoding}",
            exact_for_deployed_model=False,
            count=lambda text: len(encoding.encode(text)),
        )

    return TokenCounter(
        name="regex_lexical_estimate_v1",
        exact_for_deployed_model=False,
        count=_regex_count,
    )


def _looks_like_flat_spec(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("root"), str)
        and isinstance(value.get("elements"), Mapping)
    )


def _walk_specs(value: Any, path: str = "$") -> Iterable[tuple[str, Mapping[str, Any]]]:
    if _looks_like_flat_spec(value):
        yield path, value
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_specs(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_specs(item, f"{path}[{index}]")


def _read_corpus(path: Path) -> list[tuple[str, Mapping[str, Any]]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[tuple[str, Mapping[str, Any]]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            rows.extend((f"line:{line_number}:{subpath}", spec) for subpath, spec in _walk_specs(value))
        return rows
    value = json.loads(path.read_text(encoding="utf-8"))
    return list(_walk_specs(value))


def _portable_source_path(path: Path) -> str:
    """Keep reports reproducible without disclosing the checkout location."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _pct(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(percentile / 100 * len(ordered)) - 1))
    return float(ordered[rank])


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "total": float(sum(values)),
        "mean": float(statistics.mean(values)) if values else 0.0,
        "median": float(statistics.median(values)) if values else 0.0,
        "p95": _pct(values, 95),
        "min": float(min(values)) if values else 0.0,
        "max": float(max(values)) if values else 0.0,
    }


def _reduction(reference: float, candidate: float) -> float:
    return 0.0 if reference <= 0 else 1.0 - candidate / reference


def benchmark(
    corpus: list[tuple[str, Mapping[str, Any]]],
    token_counter: TokenCounter,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for source_name, raw_spec in corpus:
        validated = coerce_and_validate(raw_spec)
        if not validated.is_valid or validated.spec is None:
            failures.append({"source": source_name, "error": validated.error or "invalid FlatSpec"})
            continue
        canonical = validated.spec
        expected_hash = semantic_hash(canonical)
        component_count = len(canonical.get("elements", {}))
        encoded_text: dict[str, str] = {}
        roundtrip: dict[str, bool] = {}
        for format_id in FORMATS:
            payload = encode_from_flat_spec(canonical, format_id, shorten_ids=True)
            text = serialized_text(payload)
            decoded = decode_to_flat_spec(payload, format_hint=format_id).flat_spec
            ok = semantic_hash(decoded) == expected_hash
            encoded_text[format_id] = text
            roundtrip[format_id] = ok
            if not ok:
                failures.append({"source": source_name, "error": f"semantic mismatch for {format_id}"})

        flat_chars = len(encoded_text[FLAT_SPEC_V1])
        flat_bytes = len(encoded_text[FLAT_SPEC_V1].encode("utf-8"))
        flat_tokens = token_counter.count(encoded_text[FLAT_SPEC_V1])
        for format_id in FORMATS:
            text = encoded_text[format_id]
            chars = len(text)
            byte_count = len(text.encode("utf-8"))
            tokens = token_counter.count(text)
            rows.append(
                {
                    "source": source_name,
                    "semantic_hash": expected_hash,
                    "component_count": component_count,
                    "format": format_id,
                    "roundtrip_ok": roundtrip[format_id],
                    "chars": chars,
                    "bytes": byte_count,
                    "tokens": tokens,
                    "char_reduction_vs_flat": _reduction(flat_chars, chars),
                    "byte_reduction_vs_flat": _reduction(flat_bytes, byte_count),
                    "token_reduction_vs_flat": _reduction(flat_tokens, tokens),
                }
            )

    by_format: dict[str, dict[str, Any]] = {}
    for format_id in FORMATS:
        selected = [row for row in rows if row["format"] == format_id]
        chars = [float(row["chars"]) for row in selected]
        byte_counts = [float(row["bytes"]) for row in selected]
        tokens = [float(row["tokens"]) for row in selected]
        char_reductions = [float(row["char_reduction_vs_flat"]) for row in selected]
        token_reductions = [float(row["token_reduction_vs_flat"]) for row in selected]
        by_format[format_id] = {
            "samples": len(selected),
            "roundtrip_passed": sum(1 for row in selected if row["roundtrip_ok"]),
            "characters": _summarize(chars),
            "utf8_bytes": _summarize(byte_counts),
            "tokens": _summarize(tokens),
            "character_reduction_vs_flat": _summarize(char_reductions),
            "token_reduction_vs_flat": _summarize(token_reductions),
        }

    valid_compact_formats = [
        format_id
        for format_id in (COMPACT_IR_V2, A2UI_EXPRESS_V1)
        if by_format[format_id]["roundtrip_passed"] == by_format[format_id]["samples"]
    ]
    preferred = min(
        valid_compact_formats,
        key=lambda item: by_format[item]["tokens"]["total"],
        default=None,
    )
    report = {
        "report_version": "genuicraft_ir_benchmark.v1",
        "corpus_samples_discovered": len(corpus),
        "corpus_samples_benchmarked": len({row["source"] for row in rows}),
        "failures": failures,
        "tokenizer": {
            "name": token_counter.name,
            "exact_for_deployed_model": token_counter.exact_for_deployed_model,
            "note": (
                "Exact for the supplied tokenizer model."
                if token_counter.exact_for_deployed_model
                else "Deterministic estimate; rerun with the deployed Gemma tokenizer for production token counts."
            ),
        },
        "codec_identity": codec_identity(),
        "formats": by_format,
        "preferred_compact_format_by_token_total": preferred,
        "selection_guard": "Only formats with 100% semantic round-trip pass are eligible; UI component count is unchanged.",
    }
    return report, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=[DEFAULT_CORPUS])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--sentencepiece-model")
    parser.add_argument("--hf-tokenizer")
    parser.add_argument("--allow-tokenizer-download", action="store_true")
    parser.add_argument("--tiktoken-encoding")
    args = parser.parse_args()

    corpus: list[tuple[str, Mapping[str, Any]]] = []
    for input_path in args.inputs or [DEFAULT_CORPUS]:
        source_path = _portable_source_path(input_path)
        for subpath, spec in _read_corpus(input_path):
            corpus.append((f"{source_path}:{subpath}", spec))
    if not corpus:
        raise SystemExit("No FlatSpec objects were discovered in the supplied corpus")

    counter = _build_token_counter(args)
    report, rows = benchmark(corpus, counter)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
