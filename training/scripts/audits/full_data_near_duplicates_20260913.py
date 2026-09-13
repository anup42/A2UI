"""Bounded lexical source-family leakage audit; no training input modifications."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path
import re
import time

from full_data_inventory_20260913 import REPO, TASK_PREFIX, normalized, physical_lines, sha, write_json

WORD = re.compile(r"\w+", re.UNICODE)


def words(text):
    return WORD.findall(text.casefold())


def trigrams(tokens):
    return set(zip(tokens, tokens[1:], tokens[2:]))


def source_and_target(row):
    messages = row["messages"]
    task = messages[-2]["content"]
    if messages[-2]["role"] != "user" or not task.startswith(TASK_PREFIX) or messages[-1]["role"] != "assistant":
        raise ValueError("Unexpected final task/answer envelope")
    return task[len(TASK_PREFIX):], messages[-1]["content"]


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("C:/Users/anupk/Downloads/training_data"))
    parser.add_argument("--output-dir", type=Path, default=REPO/"training/reports/full_data_audit_20260913/near_duplicates")
    args = parser.parse_args()
    started = time.monotonic()
    validation, word_df, file_hashes = [], Counter(), {}
    digest = hashlib.sha256()
    for line, (offset, raw, encoding) in enumerate(physical_lines(args.source_dir/"val.jsonl"), 1):
        digest.update(raw)
        row = json.loads(raw.decode(encoding).removeprefix("\ufeff"))
        source, target = source_and_target(row)
        tokens = words(source)
        validation.append({"line":line,"source":source,"tokens":tokens,"word_set":set(tokens),"shingles":trigrams(tokens),
                           "source_sha256":sha(source),"source_normalized_sha256":sha(normalized(source)),"target_sha256":sha(target),
                           "byte_offset":offset,"byte_length":len(raw)})
        word_df.update(set(tokens))
    file_hashes["val"] = digest.hexdigest()
    anchors = defaultdict(set)
    anchor_counts = []
    for index, record in enumerate(validation):
        tokens = record["tokens"]
        candidates = list(enumerate(zip(tokens,tokens[1:],tokens[2:])))
        selected = set()
        for quarter in range(4):
            low, high = len(candidates)*quarter//4, len(candidates)*(quarter+1)//4
            ranked = sorted(candidates[low:high], key=lambda item:(-sum(math.log((len(validation)+1)/(1+word_df[word])) for word in item[1]),item[0]))
            used_positions = []
            for position, shingle in ranked:
                if shingle in selected or any(abs(position-used)<3 for used in used_positions):
                    continue
                selected.add(shingle)
                used_positions.append(position)
                if len(used_positions) >= 6:
                    break
        for shingle in selected:
            anchors[shingle].add(index)
        anchor_counts.append(len(selected))
    print(f"Retrieval index: {len(validation)} validation sources, {len(anchors)} distinct rare trigram anchors",flush=True)
    counts, retained = Counter(), []
    digest = hashlib.sha256()
    for train_line, (offset,raw,encoding) in enumerate(physical_lines(args.source_dir/"train.jsonl"),1):
        digest.update(raw)
        row = json.loads(raw.decode(encoding).removeprefix("\ufeff"))
        source, target = source_and_target(row)
        tokens = words(source)
        shingles = trigrams(tokens)
        hits = Counter()
        for shingle in shingles:
            for index in anchors.get(shingle, ()):
                hits[index] += 1
        counts["training_rows_scanned"] += 1
        counts["anchor_hit_candidate_pairs"] += len(hits)
        for index, shared_anchors in hits.items():
            if shared_anchors < 2:
                continue
            counts["two_anchor_candidate_pairs"] += 1
            val = validation[index]
            length_ratio = min(len(tokens),len(val["tokens"]))/max(1,len(tokens),len(val["tokens"]))
            if length_ratio < .65:
                counts["candidate_pairs_below_length_ratio_065"] += 1
                continue
            score = jaccard(shingles,val["shingles"])
            counts["candidate_pairs_shingle_scored"] += 1
            if score < .60:
                continue
            sequence_ratio = SequenceMatcher(None,tokens,val["tokens"],autojunk=False).ratio()
            exact = sha(normalized(source)) == val["source_normalized_sha256"]
            retained.append({"train_line":train_line,"val_line":val["line"],"train_source_sha256":sha(source),
                             "val_source_sha256":val["source_sha256"],"exact_normalized_source":exact,
                             "target_text_equal":sha(target)==val["target_sha256"],"shared_anchors":shared_anchors,
                             "word_trigram_jaccard":round(score,6),"word_set_jaccard":round(jaccard(set(tokens),val["word_set"]),6),
                             "ordered_word_sequence_ratio":round(sequence_ratio,6),"word_length_ratio":round(length_ratio,6),
                             "train_source_chars":len(source),"val_source_chars":len(val["source"]),
                             "train_byte_offset":offset,"train_byte_length":len(raw),
                             "val_byte_offset":val["byte_offset"],"val_byte_length":val["byte_length"]})
        if train_line % 25000 == 0:
            print(f"Scanned {train_line:,} train rows; {len(retained)} pairs retained at trigram Jaccard >=0.60; {time.monotonic()-started:.1f}s",flush=True)
    file_hashes["train"] = digest.hexdigest()
    retained.sort(key=lambda item:(-item["word_trigram_jaccard"],-item["ordered_word_sequence_ratio"],item["val_line"],item["train_line"]))
    exact = [pair for pair in retained if pair["exact_normalized_source"]]
    nonexact = [pair for pair in retained if not pair["exact_normalized_source"]]
    threshold_counts = []
    for shingle_min, sequence_min in ((.95,.98),(.90,.95),(.80,.90),(.70,.85),(.60,.0)):
        subset = [pair for pair in nonexact if pair["word_trigram_jaccard"]>=shingle_min and pair["ordered_word_sequence_ratio"]>=sequence_min]
        threshold_counts.append({"word_trigram_jaccard_min":shingle_min,"ordered_word_sequence_ratio_min":sequence_min,
                                 "nonexact_pairs":len(subset),"distinct_train_rows":len({pair["train_line"] for pair in subset}),
                                 "distinct_val_rows":len({pair["val_line"] for pair in subset}),
                                 "distinct_val_rows_not_already_exact":len({pair["val_line"] for pair in subset}-{pair["val_line"] for pair in exact})})
    report = {"schema_version":1,"completed_at_utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":round(time.monotonic()-started,2),
              "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"source_hashes":file_hashes,"counts":dict(counts),
              "validation_rows":len(validation),"retrieval_anchor_count":len(anchors),"anchors_per_val_min":min(anchor_counts),"anchors_per_val_max":max(anchor_counts),
              "exact_pairs":len(exact),"exact_distinct_val_rows":len({pair["val_line"] for pair in exact}),"nonexact_pairs_retained":len(nonexact),
              "thresholds":threshold_counts,"known_pair_40160_209":[pair for pair in retained if pair["train_line"]==40160 and pair["val_line"]==209],
              "method":"Unicode casefolded word tokens (regex \\w+), 3-word shingles. Per validation source, select up to six high inverse-validation-document-frequency trigrams from each source quarter, with starts at least 3 words apart within a quarter (up to24 anchors). Stream every training source, retrieve pairs sharing >=2 distinct anchors, discard word-length ratios<0.65, calculate full word-shingle Jaccard and retain >=0.60. Calculate exact normalized-source equality, word-set Jaccard, and ordered-token SequenceMatcher(autojunk=False) for retained pairs. Fixed few-shot turns excluded.",
              "limits":"Complete file scan, but bounded lexical retrieval, not exhaustive all-pairs or semantic equivalence detection. Anchor selection may miss paraphrases, rewritten entities, reordered content, or very short shared segments. Case/punctuation differences are intentionally normalized by word tokenization. High overlap indicates source-family review candidates, not automatically identical task meaning. Counts are lower bounds for this method, not all possible leakage. No external embeddings, network, or model inference used."}
    write_json(args.output_dir/"summary.json",report)
    write_json(args.output_dir/"pairs.json",{"exact":exact,"nonexact":nonexact})
    print(json.dumps({"exact_pairs":len(exact),"nonexact_pairs_retained":len(nonexact),"thresholds":threshold_counts,"elapsed_seconds":report["elapsed_seconds"]},indent=2),flush=True)


if __name__ == "__main__":
    main()
