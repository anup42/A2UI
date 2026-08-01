# GACJ v3 design decisions

## Strengthening choices

1. **Criterion reference instead of free score interpolation.** The model
   chooses observable anchor levels; host code computes all scores.
2. **Exactly five criteria per dimension.** Equal 0.2 budgets make the base
   dimension score naturally land on a five-point grid.
3. **Independent severity ledger.** A major central failure cannot be hidden by
   four unrelated strong criteria.
4. **No silent missing-evidence normalization.** Missing required evidence
   yields an incomplete score.
5. **Raw and policy scores are separate.** This preserves continuous variation
   for calibration and avoids hiding improvements inside policy bands.
6. **Closed packet and judgment schemas.** Unexpected identity, metric, or score
   fields are rejected.
7. **Frozen two-pass blinding.** Screenshot evaluation remains sealed before
   source release.
8. **Model identity enforcement.** A model change requires a new version.
9. **Fresh repeat schedule.** The 96 supplied samples become 192 opaque
   occurrences; paired copies are placed in different task blocks.
10. **Legacy replay is explicitly synthetic.** It verifies formulas but cannot
    masquerade as a fresh reliability result.

## Absolute-score meaning

- 100: reference-quality under the frozen rubric;
- 75: production-ready with minor/localized defects;
- 50: usable but not production-ready;
- 25: severely deficient;
- 0: broken or no usable evidence.

The score is criterion-referenced. It is not a percentile, probability, or
percentage of factual correctness. Equal-interval human interpretation still
requires independent human calibration.
