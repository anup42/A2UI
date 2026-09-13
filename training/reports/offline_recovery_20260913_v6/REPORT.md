# Intermediate v6 review — superseded by v9

The completed v6 export contained 115,798 training and 2,365 validation rows
(118,163 candidates), with 33,039 rows quarantined. It excluded 36 reserved
Golden-related source rows and rebuilt validation using source families.

The follow-up sample review found that train line 4,693 still omitted multiple
paragraphs after a clipped word was repaired. It also found an unnecessary
relabeling of a secondary source button in train line 419. These are reasons
to continue to v7, not to call the v6 content fully verified.

v7 reconstructs affected button labels from original rows, applies a stricter
missing-label rule, checks long prose blocks separately, filters failures,
and rebuilds validation again. Later stages add letter and join-boundary review.
Use the [final v9 report](../offline_recovery_20260913_v9/REPORT.md)
and [runbook](../../docs/messages_archive_final_review.md).

The machine-readable v6 [summary](summary.json) remains as intermediate
evidence. Original inputs and the v6 output copy remain unchanged.
