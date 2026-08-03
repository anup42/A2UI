import re
from typing import List, Tuple, Dict, Any

from datasets import load_dataset


ROLE_PATTERN = re.compile(r"(?:^|\n\n)(Human|Assistant):\s*")


def parse_hh_text(text: str) -> List[Tuple[str, str]]:
    text = text.strip()
    matches = list(ROLE_PATTERN.finditer(text))
    if not matches:
        raise ValueError(f"Could not parse HH text:\n{text[:300]}")

    turns = []
    for i, match in enumerate(matches):
        role = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        turns.append((role, content))
    return turns


def hh_to_explicit_dpo(example: Dict[str, Any]) -> Dict[str, Any]:
    try:
        chosen_turns = parse_hh_text(example["chosen"])
        rejected_turns = parse_hh_text(example["rejected"])

        if len(chosen_turns) < 2 or len(rejected_turns) < 2:
            return {"valid": False, "prompt": "", "chosen": "", "rejected": ""}

        if chosen_turns[-1][0] != "Assistant" or rejected_turns[-1][0] != "Assistant":
            return {"valid": False, "prompt": "", "chosen": "", "rejected": ""}

        chosen_prefix = chosen_turns[:-1]
        rejected_prefix = rejected_turns[:-1]

        if chosen_prefix != rejected_prefix:
            return {"valid": False, "prompt": "", "chosen": "", "rejected": ""}

        prompt = "\n\n".join([f"{role}: {content}" for role, content in chosen_prefix]).strip()
        chosen_resp = chosen_turns[-1][1].strip()
        rejected_resp = rejected_turns[-1][1].strip()

        if not prompt or not chosen_resp or not rejected_resp:
            return {"valid": False, "prompt": "", "chosen": "", "rejected": ""}

        if chosen_resp == rejected_resp:
            return {"valid": False, "prompt": "", "chosen": "", "rejected": ""}

        return {
            "valid": True,
            "prompt": prompt,
            "chosen": chosen_resp,
            "rejected": rejected_resp,
        }
    except Exception:
        return {"valid": False, "prompt": "", "chosen": "", "rejected": ""}


def load_and_prepare_datasets(dataset_name: str, train_samples: int, eval_samples: int, seed: int):
    raw_train = load_dataset(dataset_name, split="train")
    raw_test = load_dataset(dataset_name, split="test")

    train_ds = raw_train.map(hh_to_explicit_dpo)
    eval_ds = raw_test.map(hh_to_explicit_dpo)

    train_ds = train_ds.filter(lambda x: x["valid"])
    eval_ds = eval_ds.filter(lambda x: x["valid"])

    keep_cols = ["prompt", "chosen", "rejected"]
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
    eval_ds = eval_ds.remove_columns([c for c in eval_ds.column_names if c not in keep_cols])

    train_ds = train_ds.shuffle(seed=seed).select(range(min(train_samples, len(train_ds))))
    eval_ds = eval_ds.shuffle(seed=seed).select(range(min(eval_samples, len(eval_ds))))

    assert "prompt" in train_ds.column_names
    assert "chosen" in train_ds.column_names
    assert "rejected" in train_ds.column_names

    return train_ds, eval_ds