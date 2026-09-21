from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from ir_training.data.express_preparation import TASK_PREFIX, prepare_splits
from ir_training.data.golden_replacement import read_rows_strict, serialize_rows
from ir_training.data.shared_prompt import create_shared_prompt_contract

ROOT = Path(__file__).resolve().parents[1]


class _LengthTokenizer:
    name_or_path = "deterministic-length-tokenizer"
    chat_template = "test roles"
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def __init__(self, prompt_length: int = 5120):
        self.prompt_length = prompt_length

    def get_vocab(self):
        return {"fixture": 0}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False
        rendered = "".join(f"{item['role']}:\n{item['content']}\n" for item in messages)
        return rendered + ("assistant:\n" if add_generation_prompt else "")

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if "SHORT_FIXTURE" in text or "short_fixture" in text:
            length = 20
        elif "LONG_FIXTURE" in text:
            length = 3000
        elif "long_fixture" in text:
            length = 2000
        elif text.endswith("assistant:\n"):
            length = self.prompt_length
        else:
            length = 6000
        return {"input_ids": range(length)}


def _supervised_rows(tmp_path: Path) -> dict[str, Path]:
    donor = read_rows_strict(
        ROOT / "data/eval/golden32_archive_repeat_v1/golden32.jsonl"
    )[0]
    result = {}
    for split in ("train", "val"):
        rows = []
        for kind in ("SHORT_FIXTURE", "LONG_FIXTURE"):
            row = deepcopy(donor)
            identity = f"{split}-{kind.lower()}"
            row.update(
                id=identity,
                source_id=identity,
                response_text=f"{kind} independent source",
                metadata={"query_id": identity},
                completion=f'<a2ui>\nroot=Text("{identity}")\n</a2ui>',
            )
            row["a2ui_express"] = row["completion"]
            row["completion_targets"] = {"a2ui_express_v1": row["completion"]}
            row["messages"][-2]["content"] = TASK_PREFIX + row["response_text"]
            row["messages"][-1]["content"] = row["completion"]
            rows.append(row)
        path = tmp_path / f"{split}.jsonl"
        path.write_bytes(serialize_rows(rows))
        result[split] = path
    return result


def _all_sources(tmp_path: Path) -> dict[str, Path]:
    return {
        **_supervised_rows(tmp_path),
        "golden32": ROOT / "data/eval/golden32_archive_repeat_v1/golden32.jsonl",
        "golden35": ROOT / "data/eval/golden35_v1/golden35.jsonl",
        "bixby50": ROOT / "data/eval/bixby50_v1/bixby50.jsonl",
    }


def test_preparation_separates_training_sequence_and_all_evaluation_prompt_limits(tmp_path):
    destination = tmp_path / "prepared"
    manifest = prepare_splits(
        _all_sources(tmp_path),
        destination,
        tokenizer=_LengthTokenizer(prompt_length=5120),
        max_seq_length=4096,
        max_input_tokens=5120,
        shared_prompt=create_shared_prompt_contract(),
        evaluation_splits={"golden32", "golden35", "bixby50"},
    )

    for split in ("train", "val"):
        evidence = manifest["splits"][split]
        assert evidence["accepted_rows"] == 1
        assert evidence["quarantined_rows"] == 1
        assert evidence["quarantine_reasons"] == {"sequence_too_long": 1}
        assert evidence["max_accepted_token_lengths"]["sequence_tokens"] == 40
    for split, rows in (("golden32", 32), ("golden35", 35), ("bixby50", 50)):
        evidence = manifest["splits"][split]
        assert evidence["accepted_rows"] == rows
        assert evidence["quarantined_rows"] == 0
        assert evidence["max_accepted_token_lengths"]["prompt_tokens"] == 5120
        if split != "bixby50":
            assert evidence["max_accepted_token_lengths"]["sequence_tokens"] > 4096
    assert manifest["splits"]["bixby50"]["reference_available"] is False
    assert manifest["splits"]["bixby50"]["evaluation_only"] is True
    assert manifest["splits"]["bixby50"]["target_validation"] == "not_applicable_source_only"
    assert manifest["tokenizer"]["max_seq_length"] == 4096
    assert manifest["tokenizer"]["max_input_tokens"] == 5120


@pytest.mark.parametrize("cohort", ["golden32", "golden35", "bixby50"])
def test_each_strict_evaluation_cohort_rejects_prompt_above_5120(tmp_path, cohort):
    sources = _all_sources(tmp_path)
    with pytest.raises(ValueError, match=f"Split {cohort} has no accepted rows"):
        prepare_splits(
            {"train": sources["train"], "val": sources["val"], cohort: sources[cohort]},
            tmp_path / f"prepared-{cohort}",
            tokenizer=_LengthTokenizer(prompt_length=5121),
            max_seq_length=4096,
            max_input_tokens=5120,
            shared_prompt=create_shared_prompt_contract(),
            evaluation_splits={cohort},
        )
