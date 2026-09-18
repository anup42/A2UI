# Bixby50 table presentation expectations

These expectations were classified from the supplied Markdown and pinned before examining new prompt-variant outputs. The v9 revalidated baseline was inspected only to verify extraction and record its choices; it is not a layout oracle. This is a source-shape and saved-selection audit, not a validation of the fixture's factual claims or a substitute for rendering the UI.

The corpus contains **23 tables in 23 cases**, with **27 cases containing no source table**. All 23 extracted source tables match the v9 baseline's ordered columns and rows exactly. Source fixture SHA-256: `fc46aa381957bed206f9e0f53e28edbb21b5096ca093985ad184fda73faaea0a`, also recorded as `source_corpus.sha256` in `table_expectations.json`.

## Source-based preferences

| Cases | Source shape | Preferred presentation | Sensible domain choices |
| --- | --- | --- | --- |
| 001 | Forecast dates with attributes | Cards; table acceptable | weather preferred; generic/schedule acceptable |
| 003, 007, 025 | Train services, sights/hours, ordered study days | Cards; complete table acceptable | schedule, comparison, generic |
| 006, 008, 011, 012, 013, 019, 026, 044, 045, 049 | Complete entity profiles per row | Cards; complete table acceptable | comparison/generic |
| 016 | Release title/platform/date/description records | Cards; table acceptable | schedule/comparison/generic |
| 041 | Country entry-rule profiles | Cards; table acceptable | comparison/generic; booking is a broad travel equivalent requiring normal render checks |
| 050 | Climate-signal pattern/effect profiles | Cards; table acceptable | comparison/weather/generic, retaining non-forecast fields |
| 029, 030, 037 | **Feature matrices:** features in rows, compared entities in columns | **Table**; replacing it with row cards needs review | comparison/generic |
| 032 | Terms, definitions, and effects | Table or complete definition cards | generic preferred; comparison acceptable |
| 034 | Eight short food/serving/protein records | Table preferred for aligned numeric lookup; cards acceptable | generic/comparison |
| 047 | Battery chemistry tradeoffs including safety | Table preferred for cross-property comparison; complete cards acceptable | comparison/generic |

The last two are intentional exceptions to a blanket entity-row-to-card rule. All 23 cases have individual rationales and exact source cells in `table_expectations.json`. There are no source-table expectations for 002, 004, 005, 009, 010, 014, 015, 017, 018, 020, 021, 022, 023, 024, 027, 028, 031, 033, 035, 036, 038, 039, 040, 042, 043, 046, or 048. A missing output for any of those cases is still a failure, not a skipped table audit.

## What the script checks

`tools/audit_prompt_tables.py` uses only Python's standard library and reads saved run directories. It never calls ADB, a model, Gradle, or the network.

- Each planned case must have a saved compiled output, including cases with zero expected tables. Planned IDs come from `--cases`, otherwise the run's experiment/config files, otherwise all 50 expectations. A truncated results list cannot quietly reduce the denominator.
- Tables are matched by **ordered columns and rows**, never by component ID or unchanged container layout. Literal profile escaping is decoded exactly once. Canonical scalar text is compared first; a second fingerprint permits paired inline Markdown emphasis/code markers and whitespace differences. Numeric values, units, citations, URLs, punctuation, row order, and column order are not discarded.
- Missing, altered, unreachable, extra, or duplicated tables are reported separately from presentation choices. Changes to cells with matching headers produce an `altered_table` entry and bounded cell-diff evidence. A completely unmatched replacement is reported as missing plus unexpected instead of claiming a speculative match.
- For content-preserving matches, domain and presentation are classified as preferred, acceptable alternatives, implicit, or requiring review. Baseline choices and whether selection changed are recorded without demanding baseline equality. Unspecified hints require renderer review because the script does not imitate all native renderer heuristics.
- Source run failures, missing/duplicate/unexpected result IDs, and unavailable outputs remain explicit. A render-failure record with a valid JSON output can still receive useful table-content diagnostics, but the run is not labeled wholly successful.

By default, the command exits nonzero for content/output/evidence failures. `--strict-choices` additionally makes implicit or questionable selections produce a nonzero exit. A valid complete table used instead of preferred entity cards remains an accepted alternative; the preference counts expose that tradeoff.

## Usage

From the GenUICraft repo:

```powershell
python tools/audit_prompt_tables.py --self-test
python tools/audit_prompt_tables.py ..\tmp\genuicraft_20260918\gemma50_v9_revalidated --output table_audit.json
python tools/audit_prompt_tables.py PATH_TO_VARIANT_1 PATH_TO_VARIANT_2 --output variants_table_audit.json
python tools/audit_prompt_tables.py PATH_TO_RUN --cases BXP-003,BXP-030,BXP-032,BXP-037 --strict-choices
```

The script chooses `revalidated.output.a2ui.json`, then `output.a2ui.json`, then byte-preserved `source.output.a2ui.json`. Every selected file and its SHA-256 are recorded. Reports are tagged `saved_table_selection_audit` with zero model calls and are not inference successes.

## Validation performed

- Script self-checks passed for ID-independent matching, formatting equivalence, changed numeric facts, wrong domains, feature-matrix presentation review, missing/unreachable tables, missing outputs in zero-table cases, escaped pipes/backticks, code-fence exclusion, and one-layer literal decoding.
- v9 revalidated baseline: 50/50 outputs present, 23 exact table-content matches, no table-content failures; 9 preferred presentations and 14 accepted alternatives; 19 preferred and 4 accepted domain choices.
- v7 original Gemma baseline: same 23 exact matches and choice distribution. This is a saved-output audit, not a new inference run.
- Saved `gauss50_v10`: 50/50 outputs present, 17 exact matches and 6 formatting-equivalent matches, no table-content failures; 6 preferred presentations and 17 accepted alternatives.

These results show that the checker accepts useful variation while exposing presentation tradeoffs. They do not establish actual on-screen routing, lack of clipping, or readability. Replay changed/suspicious choices and inspect long cells, units, row/column reachability, and feature relationships before declaring UI quality improved.
