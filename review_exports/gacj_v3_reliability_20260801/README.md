# GACJ v3 independent blind reliability run

This artifact contains the post-run host outputs for the fresh Codex GACJ v3
benchmark. It contains 96 complete repeat pairs, 384 validated pass records,
and four task IDs. Sealed repeat identities and screenshot assets are omitted.

## Result

The run is complete but does not pass the frozen reliability gate:

- ICC(A,1): `-0.0078057526` (required `>= 0.85`)
- MAE: `16.690625` (required `<= 5.0`)
- Signed bias: `13.378125` (required absolute value `< 2.0`)
- Complete pairs: `96`
- Same-task pair violations: `0`
- Recommendation: `remain provisional`

The independent task blocks show substantial score drift. The first 96
scheduled occurrences average 72.32, while the final 32 average 92.92. This
is evidence of judge/task drift, not evidence that the metric or generated UI
quality changed across the schedule.

The `criterion_v3_same_task_reliability.json` file is an earlier same-task
consistency diagnostic and must not be used as the reliability result. Its
perfect repeat agreement was rejected because all passes used one task context.

The persisted judge identifier is the constant run label
`gacj-v3-fresh-judge`; the host did not expose a platform model name. Treat
this as a Codex-task reliability audit, not a model-version calibration claim.
