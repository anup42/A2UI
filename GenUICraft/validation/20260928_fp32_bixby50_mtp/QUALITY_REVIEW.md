# Completed paired cases

Reviewed 2026-09-28: BXP-001 through BXP-012, GPU FP32 with MTP off/on, from completed batches `group_01` through `group_03`. The run was stopped at the user's request after 27 completed generations: 15 MTP-off and 12 MTP-on. This final review covers the 24 outputs forming 12 matched pairs. The three extra MTP-off cases BXP-013 through BXP-015 are unpaired and excluded from comparative counts; interrupted MTP-on BXP-013 is excluded. No new inference, renderer replay, device operations, or runtime changes were performed for this review.

**Neither mode consistently wins on content preservation.** MTP on is worse on weather units and baggage exceptions, better on the route-generation failure, and mixed on Jaipur: its raw graph fails, but the repaired graph exposes more source content than the raw-valid MTP-off output. Syntax validity, successful rendering, and faithful source preservation are separate outcomes.

| Matched-12 outcome, derived from Android result metadata | MTP off | MTP on |
|---|---:|---:|
| Raw strict-valid output | 10/12 | 8/12 |
| Generated-output repair used | 2/12 | 4/12 |
| Confirmed captured-render status | 11/12 | 11/12 |
| Source fallback used | 0/12 | 0/12 |

These counts do not define semantic accuracy. BXP-002 is capture-unconfirmed in both modes: the harness records `renderValid=false` after a case-label accessibility timeout, with `rendererContractValid=true` and no screenshot. This does not establish a renderer failure. Several captured cases still have important source statements outside the reachable graph.

## Method and evidence scope

Compared each `source.json` with raw `output.express`, the generated or repaired `a2ui.json`, and `result.json`. Followed the component graph from `root`, resolving reachable Table rows through `updateDataModel`/`statePath`; inspected weather screenshots to verify the percentage-unit loss on screen. Compiler canonicalization renames element IDs, so raw IDs below refer to `output.express`, not necessarily the same letters in `a2ui.json`. All 12 paired off/on source objects were verified identical.

This review assesses preservation of the supplied frozen response. It does not independently verify the truth or currency of travel, weather, pricing, or other source claims. Except for the inspected weather screenshots, reachable content denotes the graph representation, not a complete visual audit of every screen and scroll position.

## Case findings

| Case | MTP off | MTP on | Evidence-balanced judgment |
|---|---|---|---|
| **001 Weather** | Three dates, high/low temperatures, rain percentages and planning tips preserved. Drops citations `[7][14]`. | Same facts except `57%/55%/55%` become bare numeric `57/55/55`; column label does not supply `%`. Same missing citations. | **On worse:** screenshots confirm visible “Rain Probability 57” versus off “57%”. Both are raw-valid. |
| **002 AQI** | Root references only `a,b,c,d`; PM2.5, citations `[5][7]`, and all health/outdoor guidance are present in unreachable `e..i`. | Byte-identical raw output and the same unreachable content. | **Equal, materially incomplete.** Both raw-valid and capture-unconfirmed; the orphan-content defect is independent of the capture timeout. |
| **003 Trains** | Preserves train numbers, stations, times, durations, classes and uncertainty. Drops `[3][11][10]`; adds an unsupported “View Arrival Times” button. | References undefined `l` from `j`: raw-invalid. Repair materializes all three rows and notes, but replaces descriptive column labels with field keys and changes order. Same missing citations. | **On worse structurally; repaired facts broadly comparable.** Do not count its rendered repair as raw validity. |
| **004 Baggage** | Numeric weights, sizes and charges preserved. Root includes `a..m`, leaving `n` (route/channel-dependent price caveat) unreachable. All 11 source citation markers removed. | Root only includes `a..l`: additionally hides `m`, which explains higher UpFront/premium allowances. Same missing caveat and citations. | **On worse:** an additional substantive fare exception is hidden despite raw validity. |
| **005 Metro** | Operating-hours subtree appears twice. Ticket/interchange headings are reachable without their details; ticket options, 1/3/5-day passes, interchange facts and rider guide are hidden. An invented `[ACTION_URL_1]` button exists but is unreachable. | Avoids duplicated hours and exposes the RV Road interchange fact. Still hides ticket/pass details, Majestic interchange and rider guide. Adds a generic source note retaining five detached markers; `[5][8][4]` remain missing. | **On slightly better reachability, still severely incomplete.** Detached citations do not preserve claim association. |
| **006 Routes** | Raw-invalid: `m` references undefined `q`. Repair preserves route rows, distances, time ranges, road/stop details and prose, but attaches invented contract text and long grammar/catalog/profile/quality hashes as reachable text. Descriptive table labels become keys. | Raw-valid. Same substantive route rows and prose, without the contract/hash leakage; two empty headings added. | **On better.** Both drop introductory citations `[2][4][11]`; citations inside route rows survive. Off used 1,054 output tokens versus on 753. |
| **007 Sights** | Table facts, opening hours, fee-availability caveats and citations `[6][2][1]` preserved. Step 2 (`g`) repeats via root and nested column; final comparative recommendation (`i`) is unreachable. | Same duplicated step and hidden conclusion; step 3 changes placement but stays reachable. | **Materially equal, incomplete.** Both raw-valid; table contents are not missing merely because SDK warnings say so. |
| **008 Restaurants** | Raw-invalid: root references undefined `d`. | Raw-invalid: column `d` references undefined `f`. | **Equivalent after repair.** Both recover all restaurant rows, costs, opening schedules, citations and summary. Both lose descriptive labels, notably “Approx. cost per person” becomes `cost`, weakening the price qualifier. |
| **009 Jaipur** | Raw-valid but repeats C-Scheme block (`h`). Its downside (`n`), entire Vaishali section (`o..t`) and Vaishali downside (`u`) are unreachable. Visible `[3][6]` retained; `[4]` hidden. | Raw-invalid due to undefined `v,w,x`. Repair removes repeated/dangling references and attaches both Vaishali sections, exposing its access/transport/atmosphere/downside. C-Scheme downside was never generated and cannot be recovered. Drops all `[3][6][4]`. | **Mixed:** off wins raw validity and two citations; on repair exposes substantially more material content. Neither is faithful. |
| **010 Earbuds** | All three prices, ANC/playback figures, trade-offs, recommendation, citations `[6][2][7]`, and the live-pricing uncertainty caveat are preserved in reachable text/table data. | Same material preservation; minor punctuation and column-label pluralization differences. | **Materially equal; no substantive omission found.** Both raw-valid. |
| **011 Phones** | Preserves all three models, indicative prices and price caveats, display/battery/charging figures, support policies, buyer-fit descriptions, all 12 citation markers and quick picks. | Same literal table rows and recommendations; adds `battery` to `numericColumns` without changing its source strings. | **Materially equal; no substantive omission found in the graph.** Both raw-valid. No claim of pixel-level parity from the presentation hint. |
| **012 Notes apps** | Raw-valid. Preserves all three comparison rows, including OneNote's 5 GB limit, and all best-fit statements. Drops `[1]`, repeats Simplenote advice (`g`), and hides the final caveat that detailed free-tier export rules are incompletely specified (`i`). | Raw output lacks closing `</a2ui>` and references undefined `c`. Repair recovers all comparison rows and best-fit statements once, with simplified table labels. Drops `[1]` and never generates the final source-coverage caveat. | **Mixed:** off wins raw validity; on repair avoids duplicated advice. Both lose the important uncertainty caveat and one citation. |

### Direct evidence

For every case, `source.json`, `result.json`, and `a2ui.json` sit beside the linked raw output.

| Case | MTP off raw output | MTP on raw output |
|---|---|---|
| 001 | [off](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_01_mtp_off/BXP-001/output.express) | [on](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_01_mtp_on/BXP-001/output.express) |
| 002 | [off](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_01_mtp_off/BXP-002/output.express) | [on](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_01_mtp_on/BXP-002/output.express) |
| 003 | [off](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_01_mtp_off/BXP-003/output.express) | [on](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_01_mtp_on/BXP-003/output.express) |
| 004 | [off](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_01_mtp_off/BXP-004/output.express) | [on](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_01_mtp_on/BXP-004/output.express) |
| 005 | [off](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_01_mtp_off/BXP-005/output.express) | [on](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_01_mtp_on/BXP-005/output.express) |
| 006 | [off](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_02_mtp_off/BXP-006/output.express) | [on](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_02_mtp_on/BXP-006/output.express) |
| 007 | [off](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_02_mtp_off/BXP-007/output.express) | [on](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_02_mtp_on/BXP-007/output.express) |
| 008 | [off](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_02_mtp_off/BXP-008/output.express) | [on](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_02_mtp_on/BXP-008/output.express) |
| 009 | [off](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_02_mtp_off/BXP-009/output.express) | [on](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_02_mtp_on/BXP-009/output.express) |
| 010 | [off](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_02_mtp_off/BXP-010/output.express) | [on](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_02_mtp_on/BXP-010/output.express) |
| 011 | [off](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_03_mtp_off/BXP-011/output.express) | [on](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_03_mtp_on/BXP-011/output.express) |
| 012 | [off](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_03_mtp_off/BXP-012/output.express) | [on](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_03_mtp_on/BXP-012/output.express) |

Weather screenshot verification: [off percentage](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_01_mtp_off/BXP-001/screen.png), [on missing percentage unit](C:/Users/anupk/Documents/git/A2UI/GenUICraft/validation/20260928_fp32_bixby50_mtp/batches/group_01_mtp_on/BXP-001/screen.png).

## Scoring interpretation verified against source and live Python probes

1. **v5.4 resolves reachable table state.** `evidence_v5_4.py:555–557` starts traversal at root; `:364` invokes `effective_table(props, self.state)`. `renderer_effective_semantics_v5_4.py:149–160` resolves direct rows or `statePath`; `:328–345` resolves keyed columns and row values. The inherited walker in `evidence_v5_3.py:733–740` evaluates visibility. Thus SDK literal-only warnings about state-backed rows must not be treated as measured omissions.
2. **The report scores raw generation.** `GenUICraft/tools/report_trained_bixby50.py:735–744` reads `output.express` and calls `score_prediction` without a repaired candidate. Android repaired `a2ui.json` is a separate artifact. A rendered repair does not raise a raw parse failure's v5.4 generation reward above zero.
3. **Exact-value and table atomics have different scope.** `_v5_3.py:675–690`, reused by v5.4, computes exact-number/date/unit fidelity from generic non-table source/output units. Table facts use the table channel. BXP-001 off receives exact-value fidelity 0 because dropped `[7][14]` are the numeric values in its generic prose, while its weather table fidelity is 1.0. Do not label that zero as loss of weather numbers.
4. **Legacy `content_coverage` is not visible-content accuracy.** `dataset/src/pipeline/metrics.py:1230–1238` searches unique ASCII word substrings anywhere in serialized canonical JSON, including unreachable components/state. The report includes this diagnostic alongside v5.4 atomics; label it legacy lexical coverage.
5. **Denominators differ.** Report source-fidelity means (`report_trained_bixby50.py:379–405`) exclude missing/null/inapplicable atomics, including atomics unavailable after parsing fails. The raw generation-reward aggregate includes returned parse-failing outputs as zero. Cases without returned output have unavailable scores, not fabricated zero rewards. Always show observed/scored denominators and runtime completion counts separately.
6. **Scores are composite heuristics, not percent-correct claims.** They do not fact-check source truth, certify citation-to-claim association, or independently prove screenshot layout/clipping/usability. `action_and_source_link_fidelity` is null for these bare `[n]` citation examples; it is not a standalone citation-preservation rate. Preserve manual examples alongside scores.

Read-only probes of the same scorer entry point used by the report:

| Raw output | v5.4 reward / 100 | Relevant atomic or limit |
|---|---:|---|
| BXP-001 off | 91.1536 | Table fidelity 1.000 |
| BXP-001 on | 90.0311 | Table fidelity 0.872; percentage loss detected |
| BXP-006 on | 89.2566 | Table fidelity 1.000; route values recognized |
| BXP-007 on | 70.0000 | Table fidelity 1.000; reachability 0.9 triggers a 0.7 cap |
| BXP-010 on | 95.7280 | Exact values 1.000; visible-content multiset 0.9873 |
| BXP-003 / BXP-008 on | 0.0000 | Raw parse failure; repaired rendering does not substitute |

These selected probes are not the complete aggregate. This stopped paired subset supports concrete defect comparisons, not a general claim that MTP improves or degrades quality across the entire holdout. Unpaired or unfinished cases must not be included as successes or failures in the paired comparison.
