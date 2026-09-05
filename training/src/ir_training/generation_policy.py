"""Shared, auditable A2UI serving termination without graph or ID repair.

The scanner consumes decoded generated tokens only. A close tag inside a quoted
Express literal (or inside the prompt's demonstration) must not terminate a run.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral
from typing import Any, Sequence

EXPRESS_STOP = "</a2ui>"
POLICY_VERSION = "a2ui-envelope-v1"


def preserve_generation_eos(
    model: Any, tokenizer: Any, extra_eos_ids: Sequence[int] | None = None
) -> list[int]:
    """Union valid native end-of-turn IDs and tokenizer EOS, never hardcode IDs.

    Retain the returned IDs before Trainer construction, then pass them back via
    ``extra_eos_ids`` after any Trainer tokenizer/model alignment.
    """
    generation = getattr(model, "generation_config", None)
    config = getattr(model, "config", None)
    sources = [
        extra_eos_ids,
        getattr(generation, "eos_token_id", None),
        getattr(config, "eos_token_id", None),
        getattr(getattr(config, "text_config", None), "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ]
    try:
        vocab_size = len(tokenizer)
    except (TypeError, AttributeError):
        vocab_size = getattr(tokenizer, "vocab_size", None)
    ids: list[int] = []
    for source in sources:
        for value in source if isinstance(source, (list, tuple, set)) else [source]:
            if isinstance(value, bool) or not isinstance(value, Integral):
                continue
            token_id = int(value)
            if token_id < 0 or (vocab_size is not None and token_id >= vocab_size):
                continue
            if token_id not in ids:
                ids.append(token_id)
    if not ids:
        raise ValueError("No valid generation EOS IDs in model/tokenizer; verify the local model bundle.")
    if generation is not None:
        generation.eos_token_id = list(ids)
    return ids


def closing_sentinel_end(text: str, stop_strings: Sequence[str] | None = None) -> int | None:
    """Return the first unquoted stop boundary, including its complete tag."""
    wanted = tuple(stop_strings) if stop_strings is not None else (EXPRESS_STOP,)
    if any(not isinstance(value, str) or not value for value in wanted):
        raise ValueError("stop_strings must contain nonempty strings")
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in ('"', "'"):
            quote = char
        else:
            for sentinel in wanted:
                if text.startswith(sentinel, index):
                    return index + len(sentinel)
        index += 1
    return None


def stop_express_completion(text: str, stop_strings: Sequence[str] | None = None) -> str:
    boundary = closing_sentinel_end(text, stop_strings)
    return text if boundary is None else text[:boundary]


def build_stopping_criteria(
    tokenizer: Any, prompt_length: int | Sequence[int], stop_strings: Sequence[str] | None = None
) -> Any:
    """Per-sequence HF stopping; prompt_length is the padded input width.

    Decode the generated suffix each step, so tokenizer-boundary splits and
    escaped quotes are handled correctly. Deliberately fail instead of silently
    disabling requested termination when the installed runtime is incompatible.
    """
    from transformers import StoppingCriteria, StoppingCriteriaList
    import torch

    wanted = list(stop_strings) if stop_strings is not None else [EXPRESS_STOP]
    if not wanted:
        return StoppingCriteriaList([])
    closing_sentinel_end("", wanted)

    class ExpressEnvelopeCriteria(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
            del scores, kwargs
            widths = ([prompt_length] * len(input_ids)
                      if isinstance(prompt_length, Integral) else list(prompt_length))
            if len(widths) != len(input_ids):
                raise ValueError("Stop-policy prompt lengths do not match generated batch")
            done = [closing_sentinel_end(tokenizer.decode(ids[int(width):],
                        skip_special_tokens=True), wanted) is not None
                    for ids, width in zip(input_ids, widths)]
            return torch.tensor(done, dtype=torch.bool, device=input_ids.device)

    return StoppingCriteriaList([ExpressEnvelopeCriteria()])


def _token_list(ids: Any) -> list[int]:
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return [int(value) for value in ids]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generation_diagnostics(
    tokenizer: Any, generated_ids: Any, *, eos_token_ids: Sequence[int],
    max_new_tokens: int, prompt_text: str | None = None, input_ids: Any = None,
    stop_strings: Sequence[str] | None = None,
) -> dict[str, Any]:
    ids = _token_list(generated_ids)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    wanted = list(stop_strings) if stop_strings is not None else [EXPRESS_STOP]
    boundary = closing_sentinel_end(text, wanted)
    terminal_id = ids[-1] if ids else None
    reason = ("closing_sentinel" if boundary is not None else
              "eos_token" if terminal_id in eos_token_ids else
              "max_new_tokens" if len(ids) >= max_new_tokens else "unknown")
    policy = {"version": POLICY_VERSION, "eos_token_ids": list(eos_token_ids),
              "stop_strings": wanted, "stop_scope": "generated_tokens_outside_quoted_literals",
              "max_new_tokens": int(max_new_tokens), "add_special_tokens": False}
    result: dict[str, Any] = {
        "generation_policy": policy,
        "generation_policy_sha256": sha256_text(json.dumps(policy, sort_keys=True)),
        "stop_reason": reason, "terminal_token_id": terminal_id,
        "output_tokens": len(ids), "generated_token_ids_sha256": sha256_text(json.dumps(ids)),
        "raw_output_scope": "tokens_actually_returned_by_runtime; continuation_after_stop_unobserved",
        "chat_template_sha256": sha256_text(str(getattr(tokenizer, "chat_template", "") or "")),
    }
    if prompt_text is not None:
        result["prompt_sha256"] = sha256_text(prompt_text)
    if input_ids is not None:
        prompt_ids = _token_list(input_ids)
        result.update(input_tokens=len(prompt_ids), input_token_ids_sha256=sha256_text(json.dumps(prompt_ids)))
    return result
