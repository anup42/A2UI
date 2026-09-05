#!/usr/bin/env python3
"""Run a LiteRT-LM A2UI Express package with the training conversation shape."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "dataset" / "src"))

from ir_training.eval.generate import _extract_user_text, _extract_url_map
from ir_training.generation_policy import sha256_text, stop_express_completion


PREFACE = [
    {
        "role": "user",
        "content": (
            "Create A2UI Express v1 GenUI IR for this response:\n\n"
            "A small travel checklist with a title and two items."
        ),
    },
    {
        "role": "model",
        "content": (
            "<a2ui>\n"
            "root=Column([a,b])\n"
            'a=Text("Travel Checklist","h1")\n'
            "b=List([c,d])\n"
            'c=Text("Passport")\n'
            'd=Text("Charger")\n'
            "</a2ui>"
        ),
    },
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-prompt-file")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--response-file")
    source.add_argument("--response-jsonl")
    source.add_argument("--prepared-jsonl", help="Use the exact saved training conversation for prompt parity")
    parser.add_argument("--query-id")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=32)
    parser.add_argument("--max-num-tokens", type=int, default=4096)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preprocess-training-urls", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def _load_response(args: argparse.Namespace) -> str:
    if args.response_file:
        return Path(args.response_file).read_text(encoding="utf-8")
    if not args.query_id:
        raise ValueError("--query-id is required with --response-jsonl")
    with Path(args.response_jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("query_id")) == args.query_id:
                return str(row["response_text"])
    raise ValueError(f"query_id not found: {args.query_id}")


def _extract_text(response: Any) -> str:
    content = response.get("content", []) if isinstance(response, dict) else []
    return "".join(
        item.get("text", "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    )


def _prepared_conversation(row: dict[str, Any]) -> tuple[str, list[dict[str, str]], str]:
    _extract_user_text(row)  # Assert source/task identity before model execution.
    messages = list(row.get("messages") or [])
    if messages and messages[-1].get("role") == "assistant":
        messages.pop()
    if not messages or messages[0].get("role") != "system" or messages[-1].get("role") != "user":
        raise ValueError("Prepared row must have a system prompt and a final user task")
    if any(item.get("role") not in {"user", "assistant"} or not isinstance(item.get("content"), str) for item in messages[1:]):
        raise ValueError("Prepared conversation contains unsupported roles or non-text content")
    preface = [{"role": "model" if item["role"] == "assistant" else "user", "content": item["content"]}
               for item in messages[1:-1]]
    return str(messages[0]["content"]), preface, str(messages[-1]["content"])


def _validate_output(output: str) -> tuple[bool, str | None]:
    from pipeline.ir_formats import validate_express_completion, compile_express_to_wire
    result = validate_express_completion(output)
    if not result.raw_valid:
        return False, "; ".join(result.errors)
    try:
        compile_express_to_wire(result.canonical_graph)
    except Exception as exc:
        return False, f"wire:{type(exc).__name__}:{exc}"
    return True, None


def main() -> int:
    args = _parse_args()
    import litert_lm

    prepared_row = None
    url_map: dict[str, Any] = {}
    if args.prepared_jsonl:
        from ir_training.common.jsonl import read_jsonl
        matches = [row for row in read_jsonl(args.prepared_jsonl)
                   if args.query_id and args.query_id in {str(row.get("id")), str(row.get("query_id"))}]
        if len(matches) != 1:
            raise ValueError("--prepared-jsonl requires --query-id matching exactly one row id or query_id")
        if args.preprocess_training_urls:
            raise ValueError("Prepared rows already own URL preprocessing; do not process them again")
        prepared_row = matches[0]
        system_prompt, preface, target = _prepared_conversation(prepared_row)
        response_text = _extract_user_text(prepared_row)
        url_map = _extract_url_map(prepared_row)
        if args.system_prompt_file and Path(args.system_prompt_file).read_text(encoding="utf-8").strip() != system_prompt.strip():
            raise ValueError("System prompt file differs from the prepared training row")
    else:
        if not args.system_prompt_file:
            raise ValueError("--system-prompt-file is required without --prepared-jsonl")
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
        response_text = _load_response(args).strip()
        preface = PREFACE
    if args.preprocess_training_urls and prepared_row is None:
        from ir_training.data.url_preprocess import preprocess_training_urls

        processed = preprocess_training_urls(response_text, {})
        response_text = processed.response_text
        url_map = processed.url_map

    if prepared_row is None:
        target = "Create A2UI Express v1 GenUI IR for this response:\n\n" + response_text
    backend = (
        litert_lm.Backend.CPU(thread_count=args.cpu_threads)
        if args.backend == "cpu"
        else litert_lm.Backend.GPU()
    )
    sampler = litert_lm.SamplerConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    with litert_lm.Engine(
        args.model,
        backend=backend,
        max_num_tokens=args.max_num_tokens,
        enable_benchmark=True,
    ) as engine:
        with engine.create_conversation(
            system_message=system_prompt,
            messages=preface,
            sampler_config=sampler,
            max_output_tokens=args.max_output_tokens,
        ) as conversation:
            response = conversation.send_message(target)
            raw_output = _extract_text(response)
            output = stop_express_completion(raw_output).strip()
            benchmark = conversation.get_benchmark_info()

    valid, validation_error = _validate_output(output)
    raw_valid, raw_validation_error = _validate_output(raw_output)

    restored_output = output
    if url_map:
        from ir_training.data.url_preprocess import restore_url_placeholders

        restored_output = restore_url_placeholders(output, url_map)
    result = {
        "backend": args.backend.upper(),
        "input_response": response_text,
        "url_map": url_map,
        "output": output,
        "raw_output": raw_output,
        "raw_valid_a2ui_express": raw_valid,
        "raw_validation_error": raw_validation_error,
        "serving_stop_applied": raw_output.strip() != output,
        "stop_scope": "postdecode_only; native early stopping and token parity require runtime validation",
        "native_stop_reason": "unavailable_from_this_LiteRT_binding",
        "prepared_row_id": prepared_row.get("id") if prepared_row else None,
        "prompt_source": "prepared_messages" if prepared_row else "reconstructed_legacy_preface",
        "prompt_messages_sha256": sha256_text(json.dumps({"system": system_prompt, "preface": preface, "target": target}, ensure_ascii=False, sort_keys=True)),
        "response_text_sha256": sha256_text(response_text),
        "restored_output": restored_output,
        "valid_a2ui_express": valid,
        "validation_error": validation_error,
        "prefill_tokens": benchmark.last_prefill_token_count,
        "prefill_tokens_per_second": benchmark.last_prefill_tokens_per_second,
        "decode_tokens": benchmark.last_decode_token_count,
        "decode_tokens_per_second": benchmark.last_decode_tokens_per_second,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if validation_error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
