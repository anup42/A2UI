The saved R32 Bixby50 MTP-on replay emits a nonempty preview before document completion for 32 of 50 cases. The remaining 18 lack a complete valid root or its component contract. Four also fail terminal compilation; fourteen become accepted only through terminal `GENERATED_DSL_REPAIR`.

| No-preview category | Count | Cases |
| --- | ---: | --- |
| Complete state followed by an unfinished root list; no component signatures arrive | 11 | 006, 024, 029, 030, 032, 037, 044, 045, 048, 049, 050 |
| Only a 59-character unfinished root list; terminal baseline rejects | 4 | 018, 027, 033, 036 |
| Truncated state literal with no root or Table signature | 1 | 010 |
| Bare `Text(...)` call with no root assignment | 1 | 038 |
| Unsupported `ColumnWave(...)` root, with a truncated final Button action | 1 | 040 |

Case numbers use the `BXP-` prefix. None of the eighteen has a Markdown fence or BOM wrapper. Wrapper normalization would therefore add no coverage in this corpus. Publishing their state or standalone calls would require creating a root, guessing a catalog type, or choosing presentation before the model supplies it. The live parser should keep holding these cases and retain the existing terminal repair decision.

The replay uses cumulative 32-character chunks from `GenUICraft/validation/20260930_r32_bixby50_gpu_fp32`, selecting the MTP-on corpus. It measures character readiness, not native generation latency, perceived speed, or painted-frame timing. The compiler leaves raw text and final `compileWithRepair` behavior unchanged.

[Replay results](bixby50_replay.json) and [all eighteen diagnoses with repository-relative raw-output references](bixby50_no_preview_diagnosis.json) contain the evidence and exact case paths.
