# Bounded source-family train/validation leakage audit

The audit streamed all 150,292 training sources against a lexical retrieval index built from all 910 validation sources. It took 98.39 seconds in one Python process, used only the standard library, did not perform model inference, and did not modify either input file. Both full-file SHA-256 hashes match the main inventory.

## Result

- Recovered all **28 exact normalized-source train/validation pairs** already identified by the complete hash inventory.
- Identified **four additional nonexact source pairs**, all involving **validation line 209**. This adds **one source-family review case**, not four independent validation cases.
- No nonexact pair in the retrieved candidates met both trigram Jaccard >= 0.80 and ordered-word sequence ratio >= 0.90. Three pairs met trigram Jaccard >= 0.70 and sequence ratio >= 0.85. The fourth is a looser same-template/family match.

| Training line | Validation line | Word-trigram Jaccard | Ordered-word sequence ratio | Exact target text? |
|---:|---:|---:|---:|---|
| 40,160 | 209 | 0.765823 | 0.946996 | Yes |
| 4,697 | 209 | 0.709877 | 0.939502 | No |
| 51,046 | 209 | 0.709877 | 0.939502 | No |
| 19,609 | 209 | 0.641618 | 0.868056 | No |

## Manual review of all four additional pairs

These sources are variants of the same resignation-letter family. Validation line 209 and training line 40,160 differ only in a handful of phrases: for example, “current” versus “outstanding,” “assist” versus “support,” and wording of the final well-wishes. Their final target text is identical. This confirms that exact source hashes alone miss this source-family overlap.

Training line 4,697 is another close wording variant. Line 51,046 also changes a duration from three to four, so it is **not factually identical** despite extensive overlap. Line 19,609 changes the professional role and department and the handover details; it is a **same-template/family candidate**, not the same exact task. These distinctions matter: do not erase substantive fact differences or automatically replace all four targets with one canonical string.

For a validation set intended to measure generalization beyond already-seen source families, keep this letter family within one split and replace the validation occurrence with an independent source. For alternative benchmark designs that intentionally permit known templates, state that policy explicitly and do not describe the split as source-family-independent.

## Reproducible method

1. Extract only the final user-task response, excluding the fixed few-shot example.
2. Case-fold and tokenize with Unicode `\w+`; construct sets of three-word shingles.
3. For each validation source, select up to six high inverse-validation-document-frequency shingle anchors in each of four source quarters, with starts at least three words apart within each quarter. This produced 21,209 distinct anchors and 16–24 anchors per validation source.
4. Stream every training source. Retrieve candidates sharing at least two distinct anchors. Discard candidates whose word-count length ratio is below 0.65.
5. Compute full word-trigram Jaccard for the remaining 122,926 candidates. Retain scores >= 0.60, then calculate word-set Jaccard, normalized-source equality and ordered-token `SequenceMatcher(autojunk=False)` similarity. Report exact matches separately.

The retrieval stage encountered 2,014,621 one-anchor candidate pairs and 816,764 pairs with at least two anchors. Of these, 693,838 were rejected by the length-ratio prefilter. Thirty-two pairs survived the final shingle threshold: 28 exact and four nonexact.

## Limits

This is a complete **file scan with bounded lexical candidate retrieval**, not an exhaustive all-pairs semantic comparison. It can miss paraphrases, large entity replacements, reordered content, short shared subdocuments, or containment relationships rejected by the length-ratio filter. Word tokenization intentionally ignores case and punctuation; similarity does not prove identical task meaning. Therefore, the result is evidence of one additional source-family overlap, **not proof that all other validation sources are independent**.

The audit measures lexical relatedness only. It does not assess the quality of the four targets, replace the main DSL/semantic audit, or establish the generator's original intent. Retain original run and source-family metadata during future generation so that split independence can be enforced directly instead of inferred afterward.

## Reproduction

Run `python training/scripts/audits/full_data_near_duplicates_20260913.py` from the repository root. The script writes `summary.json` with method, hashes, counts and conservative thresholds, plus `pairs.json` with every retained pair's line numbers, byte offsets and scores. It neither writes repaired training rows nor changes train/validation assignments.
