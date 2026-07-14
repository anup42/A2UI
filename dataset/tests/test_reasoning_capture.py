import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm.base import extract_reasoning_metadata, split_reasoning_from_text
from pipeline.cache import PromptCache


class ReasoningCaptureTests(unittest.TestCase):
    def test_extracts_vllm_reasoning_content(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "Preserve the table rows.",
                        "content": '{"root":"main","elements":{}}',
                    }
                }
            ],
            "usage": {"completion_tokens_details": {"reasoning_tokens": 17}},
        }

        text, source, tokens = extract_reasoning_metadata(payload)

        self.assertEqual(text, "Preserve the table rows.")
        self.assertEqual(source, "message.reasoning_content")
        self.assertEqual(tokens, 17)

    def test_extracts_gemini_thought_parts_without_final_json(self) -> None:
        payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"thought": True, "text": "Use weather cards."},
                            {"text": '{"root":"main","elements":{}}'},
                        ]
                    }
                }
            ],
            "usageMetadata": {"thoughtsTokenCount": 9},
        }

        text, source, tokens = extract_reasoning_metadata(payload)

        self.assertEqual(text, "Use weather cards.")
        self.assertEqual(source, "candidate.thought_parts")
        self.assertEqual(tokens, 9)

    def test_splits_inline_thinking_from_final_json(self) -> None:
        reasoning, final_text = split_reasoning_from_text(
            '<think>Keep all headings.</think>\n{"root":"main","elements":{}}'
        )

        self.assertEqual(reasoning, "Keep all headings.")
        self.assertEqual(final_text, '{"root":"main","elements":{}}')

    def test_splits_gemma_reasoning_markers(self) -> None:
        reasoning, final_text = split_reasoning_from_text(
            '<|channel>thought\nUse a compact table.<channel|>\n{"root":"main","elements":{}}'
        )

        self.assertEqual(reasoning, "Use a compact table.")
        self.assertEqual(final_text, '{"root":"main","elements":{}}')

    def test_prompt_cache_preserves_reasoning_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "prompt_cache.jsonl"
            cache = PromptCache(cache_path, enabled=True)
            cache.set(
                "hash-1",
                '{"root":"main","elements":{}}',
                {"choices": []},
                "Keep the response structure.",
                "message.reasoning_content",
                12,
            )

            reloaded = PromptCache(cache_path, enabled=True).get("hash-1")

            self.assertIsNotNone(reloaded)
            assert reloaded is not None
            self.assertEqual(reloaded.reasoning_text, "Keep the response structure.")
            self.assertEqual(reloaded.reasoning_source, "message.reasoning_content")
            self.assertEqual(reloaded.reasoning_tokens, 12)
            line = json.loads(cache_path.read_text(encoding="utf-8").strip())
            self.assertIn("reasoning_text", line)


if __name__ == "__main__":
    unittest.main()
