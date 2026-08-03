from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "src"))
sys.path.insert(0, str(ROOT / "dataset" / "src"))

from ir_training.data.chat_templates import build_messages, build_prompt
from pipeline.ir_formats.active import validate_express_completion


def test_training_messages_include_a_valid_native_express_example() -> None:
    messages = build_messages("system", "Show delivery status", target_format="a2ui_express_v1")

    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    validate_express_completion(messages[2]["content"])
    assert messages[-1]["content"].endswith("Show delivery status")


def test_plain_prompt_includes_the_same_few_shot_before_the_request() -> None:
    prompt = build_prompt("system", "Show delivery status", target_format="a2ui_express_v1")

    assert prompt.index('root=Column([a,b])') < prompt.index("Show delivery status")
    assert prompt.endswith("Assistant:\n")
