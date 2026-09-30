"""Require complete source wording for letter/document supervision."""

import re

from ir_training.data.archive_final_review import bound_content_strings
from ir_training.data.archive_recovery import _normalized_phrase

POLICY = "messages-archive-letter-review-v8-20260913"
LETTER_START = re.compile(
    r"^\s*(?:Subject\s*:|Dear\b|To whom it may concern)", re.IGNORECASE
)


def letter_gaps(source, graph):
    if not LETTER_START.search(source):
        return []
    target_tokens = _normalized_phrase("\n".join(bound_content_strings(graph))).split()
    gaps = []
    for index, block in enumerate(re.split(r"\n\s*\n", source), 1):
        lines = [
            line
            for line in block.splitlines()
            if not re.match(
                r"^\s*(?:Media\s*:|Action\s*:|Sources?\s*:|Quick Actions)",
                line,
                re.IGNORECASE,
            )
        ]
        prose = re.sub(r"^\s*Subject\s*:\s*", "", " ".join(lines), flags=re.IGNORECASE)
        words = _normalized_phrase(prose).split()
        if len(words) < 4:
            continue
        # Ordered subsequence tolerates extra UI headings, punctuation, and
        # chunk boundaries. It does not approve a shortened/reworded letter.
        position = 0
        for token in target_tokens:
            if position < len(words) and token == words[position]:
                position += 1
        if position != len(words):
            gaps.append(
                {
                    "block": index,
                    "source_words": len(words),
                    "matched_ordered_prefix_words": position,
                    "source_excerpt": prose[:200],
                }
            )
    return gaps
