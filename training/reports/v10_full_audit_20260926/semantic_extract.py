"""Extract chosen current v10 pairs for close reading without changing the dataset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "training/outputs/datasets/full_data_archive_recovered_v10"
OUT = Path(__file__).resolve().parent / "semantic_selected_pairs.jsonl"
TRAIN = {27, 137, 599, 704, 753, 831, 967, 1584, 1702, 2228, 2328, 3955, 4785, 4932, 4964, 8319, 8877, 8952, 12148, 12400, 14292, 14450, 22037, 29947, 30894, 37244, 39477, 40233, 41986, 45772, 46851, 48799, 62577, 78969, 81069, 81518, 87790}
VAL = {1423, 1753}


def main() -> None:
    with OUT.open("w", encoding="utf-8") as output:
        for split, selected in (("train", TRAIN), ("val", VAL)):
            with (DATA / f"{split}.jsonl").open(encoding="utf-8") as stream:
                for line_no, line in enumerate(stream, 1):
                    if line_no not in selected:
                        continue
                    row = json.loads(line)
                    pair = {"split": split, "line": line_no, "id": row["id"], "source_id": row["source_id"],
                            "line_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                            "source": row["messages"][-2]["content"].split("\n\n", 1)[-1], "target": row["completion"]}
                    output.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
