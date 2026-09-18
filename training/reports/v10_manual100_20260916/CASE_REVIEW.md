# Manual review of all 100 cases

Every source response and complete A2UI Express target was read. Verdicts are manually authored, not emitted by the checks. `train:line` and `val:line` are 1-based v10 file positions; `original` is the archive coordinate.

KEEP means no material defect found in this bounded review, not verified true/safe in every domain. REPAIR and REJECT both mean withhold as-is; REJECT means rebuilding or external verification rather than a bounded known edit.

## Case 001: Hypothyroidism follow-up — KEEP

- v10: `train:465`; original: `train:17193`
- Record: `archive-train-017193-0b54fc7eec716fc4`
- [Full source and target](SAMPLES.md#case-001)

All five physician questions, four tracking rows, three icons and four action/source references are preserved. Source medical advice is not clinically certified.

## Case 002: 5k pace calculation — REPAIR

- v10: `train:2295`; original: `train:47052`
- Record: `archive-train-047052-619972e8e004ea5d`
- [Full source and target](SAMPLES.md#case-002)

Source and target say 10.4% speed increase; 25/22.5 - 1 = 11.11%. The paces and rounded speeds are otherwise consistent. Correct the source percentage and regenerate the paired target.

Tags: `source_arithmetic`

## Case 003: JFK-LHR ranking — REPAIR

- v10: `train:3156`; original: `train:136019`
- Record: `archive-train-136019-8e17c5cc6e61e115`
- [Full source and target](SAMPLES.md#case-003)

Option 2/3/1 references have no option-ID column and conflict with displayed row order. Name the airlines explicitly or preserve original IDs. Timings/durations also need source timezone/date clarification; do not invent them.

Tags: `source_ambiguous_option_ids`, `source_time_metadata`

## Case 004: Video-editing laptops — KEEP

- v10: `train:3369`; original: `train:50822`
- Record: `archive-train-050822-84db70d11a7ff49a`
- [Full source and target](SAMPLES.md#case-004)

All three model rows, recommendations, budget caveats and two actions are retained. Generic model assumptions are scenario data, not verified current benchmarks.

## Case 005: Brake-noise diagnostic guide — KEEP

- v10: `train:4393`; original: `train:59836`
- Record: `archive-train-059836-54757ec9693ad95b`
- [Full source and target](SAMPLES.md#case-005)

All five diagnostic rows, five repair-process steps, three cost lines, budget caveat and source/action references are preserved; no unsafe target-only instruction added.

## Case 006: Apartment rent reminder — KEEP

- v10: `train:4554`; original: `train:78255`
- Record: `archive-train-078255-fabb0b9da2aa74b2`
- [Full source and target](SAMPLES.md#case-006)

Amount, due date, payment terms and three distinct payment/lease actions are preserved; historical scenario date is not itself an error.

## Case 007: 1970s progressive-rock albums — KEEP

- v10: `train:5425`; original: `train:51599`
- Record: `archive-train-051599-c5fcaf67b7badc0e`
- [Full source and target](SAMPLES.md#case-007)

Three album recommendations, years/tracks, explanatory prose, comparison matrix and four references are retained; aesthetic descriptions are not treated as verified factual rankings.

## Case 008: Vegan dinner plan — KEEP

- v10: `train:7902`; original: `train:91278`
- Record: `archive-train-091278-2c435eddc4f199ff`
- [Full source and target](SAMPLES.md#case-008)

Three courses, all 13 grocery items and the correctly summed $80 total, preparation details and two actions are preserved.

## Case 009: January wellness plan — KEEP

- v10: `train:7953`; original: `train:121588`
- Record: `archive-train-121588-6e13484514bdedf8`
- [Full source and target](SAMPLES.md#case-009)

Four daily targets, four roadmap rows, three tracking methods with their media/actions and both quick actions are retained.

## Case 010: Wildlife camera budget — REJECT

- v10: `train:9062`; original: `train:48765`
- Record: `archive-train-048765-007ef1ce0c5a43bf`
- [Full source and target](SAMPLES.md#case-010)

Source assigns an AI processing unit to the original Sony Alpha 1; Sony documents its addition in Alpha 1 II. The $4,000 body-plus-lens recommendation also lists no lens/kit totals and already asks to stretch the budget. Rebuild the recommendation from verified model generations and dated complete kit prices; not claiming every possible used/discount kit is over budget.

Tags: `source_product_spec`, `source_budget_feasibility`

Evidence: [Sony Alpha 1 II announcement (20 November 2024)](https://www.sony.com.hk/press/pdf/20241120_e.pdf). The manufacturer describes the AI processing unit as an addition to Alpha 1 II. This supports the generation-mixup finding; no live kit-price comparison was completed.

## Case 011: Mumbai heat/hydration card — KEEP

- v10: `train:12097`; original: `train:18310`
- Record: `archive-train-018310-f17d8cd4aacb5d24`
- [Full source and target](SAMPLES.md#case-011)

All weather readings, heat-index caveats, hydration schedule/table and two actions are preserved. Clinical suitability and historical weather truth are outside this conversion review.

## Case 012: Tokyo typhoon alert — KEEP

- v10: `train:12224`; original: `train:8928`
- Record: `archive-train-008928-276687714acd2520`
- [Full source and target](SAMPLES.md#case-012)

Alert date, wind, risk, four impact rows and emergency/source links faithfully retained; treated as a supplied scenario, not a verified historical alert.

## Case 013: Space strategy board games — REJECT

- v10: `train:12794`; original: `train:54230`
- Record: `archive-train-054230-9b385ca6ce4016f5`
- [Full source and target](SAMPLES.md#case-013)

Source mixes real space titles with unsupported space versions/themes (Terra Mystica, Root, Scythe) and an unverified Void Century entry. A faithful target still teaches unreliable recommendations; verify and regenerate the source list.

Tags: `source_product_theme`, `source_unverified_entities`

Evidence: [Leder Games: Root](https://ledergames.com/products/root-a-game-of-woodland-might-and-right). Publisher describes a woodland/forest game. A named space variant in the sample was not substantiated; absence of verification is not proof a fan mod cannot exist.

Evidence: [Feuerland: Terra Mystica rules](https://www.feuerland-spiele.de/fileadmin/game/Terra_Mystica/Terra_Mystica_rules_EN_Web.pdf). Publisher rules describe fantasy factions and home terrain; the source's planetary presentation requires rebuilding or an explicit supported adaptation.

Evidence: [Stonemaier: Scythe](https://europe.stonemaiergames.com/products/scythe). Publisher places Scythe in alternate-history 1920s Europa; the sample's space-mod framing is unverified.

## Case 014: Quarterly review meeting — KEEP

- v10: `train:14433`; original: `train:78362`
- Record: `archive-train-078362-d44158713911d4d8`
- [Full source and target](SAMPLES.md#case-014)

All attendees, three time slots, goals/tasks/deliverables and calendar/template actions are retained.

## Case 015: Ankle return-to-jogging plan — REPAIR

- v10: `train:14666`; original: `train:140420`
- Record: `archive-train-140420-3a4f77a7f977ec01`
- [Full source and target](SAMPLES.md#case-015)

Source roadmap introduces jogging in Week 7, while safety/frequency text authorizes impact/jogging during Weeks 6 and 7. Target copies the inconsistency. Resolve with qualified source review before regenerating; do not choose a medical schedule automatically.

Tags: `source_internal_contradiction`, `medical_source_review`

## Case 016: Compound-interest video script — KEEP

- v10: `train:15227`; original: `train:3484`
- Record: `archive-train-003484-4cf94e5623194e74`
- [Full source and target](SAMPLES.md#case-016)

All five timed audio/visual sections and the four-stage production roadmap survive intact. Simple-interest example is internally consistent; promotional investment language is source-authored.

## Case 017: Dell thermal troubleshooting — KEEP

- v10: `train:15801`; original: `train:33709`
- Record: `archive-train-033709-47d30d97318affe9`
- [Full source and target](SAMPLES.md#case-017)

Five checklist steps, three software sections, physical-maintenance instructions and four references retained. Hardware-dependent tuning claims have not been benchmark-certified.

## Case 018: Veterinary radio advertisement — KEEP

- v10: `train:16898`; original: `train:45083`
- Record: `archive-train-045083-5b34158345fb8d56`
- [Full source and target](SAMPLES.md#case-018)

All four timed segments, voiceover/SFX/music directions, spoken call-to-action and production timeline are retained. Fictional/source business details are not verified real-world claims.

## Case 019: Biology-to-data-science roadmap — KEEP

- v10: `train:20562`; original: `train:30434`
- Record: `archive-train-030434-1db481798c534eab`
- [Full source and target](SAMPLES.md#case-019)

All four phases and detailed implementation paragraphs/actions/source links retained; math arrow normalized correctly.

## Case 020: Essay deadline alert — KEEP

- v10: `train:24797`; original: `train:82725`
- Record: `archive-train-082725-43557754d8fbf26d`
- [Full source and target](SAMPLES.md#case-020)

Urgency, assignment/date/time, six-hour remainder and all three actions preserved without invented fields.

## Case 021: Broadway ticket choices — KEEP

- v10: `train:25666`; original: `train:539`
- Record: `archive-train-000539-9e8f64ce9ab5460a`
- [Full source and target](SAMPLES.md#case-021)

Five show rows and correct two-ticket totals, row/section labels, inventory caveat, three booking actions and four other links retained.

## Case 022: Vienna opera budget — KEEP

- v10: `train:26139`; original: `train:98944`
- Record: `archive-train-098944-a3aa9b9a7a0baee8`
- [Full source and target](SAMPLES.md#case-022)

Three seat-price totals and budget statuses are correct and preserved. Shared action placeholder already exists in the source; distinct real booking destinations cannot be inferred.

## Case 023: Daily productivity schedule — KEEP

- v10: `train:26742`; original: `train:149753`
- Record: `archive-train-149753-2021e52f3e091f9b`
- [Full source and target](SAMPLES.md#case-023)

Six intervals preserve the eight activity hours plus two buffer hours; all seven icons are referenced and both actions retained.

## Case 024: Sydney court reminder — KEEP

- v10: `train:27143`; original: `train:88903`
- Record: `archive-train-088903-17d06ce401a47785`
- [Full source and target](SAMPLES.md#case-024)

Case number, attachment filename, date/time, AEDT, arrival guidance and three actions preserved.

## Case 025: Seafood and wine pairings — KEEP

- v10: `train:27310`; original: `train:44613`
- Record: `archive-train-044613-a1651c847c19c8be`
- [Full source and target](SAMPLES.md#case-025)

Seven pairings with full four-column meanings, three strategy paragraphs and four action/source links retained.

## Case 026: Senior PM onboarding — KEEP

- v10: `train:28780`; original: `train:8203`
- Record: `archive-train-008203-e8564067288dbe10`
- [Full source and target](SAMPLES.md#case-026)

Three roadmap phases and all 12 detailed tasks retained with their five icons and two bound actions; malformed source link wrapper is safely normalized.

## Case 027: New York-Tokyo travel budget — REPAIR

- v10: `train:30879`; original: `train:70200`
- Record: `archive-train-070200-045f4128b34bdd39`
- [Full source and target](SAMPLES.md#case-027)

Source says listed options fit a $3,000 two-person flight budget, then allocates only $2,400 to flights plus $600 transit. JAL/ANA totals ($2,700/$2,800) exceed that allocation. Clarify total-trip versus flight-only budget and which options actually fit; target copies the ambiguity.

Tags: `source_budget_scope`

## Case 028: Wearable comparison matrix — REPAIR

- v10: `train:30894`; original: `train:29166`
- Record: `archive-train-029166-93fb5e6cec231445`
- [Full source and target](SAMPLES.md#case-028)

Table j transposes entities into rows but retains device names as column labels: device is labelled Step Counter, accuracy Fitness Watch, battery Medical ECG Wearable. The focus field is unprojected. Regenerate a correctly oriented, source-faithful matrix; strict validity does not catch this.

Tags: `target_table_column_semantics`, `target_hidden_table_field`

## Case 029: Friday deep-work schedule — KEEP

- v10: `train:32320`; original: `train:63467`
- Record: `archive-train-063467-9c812553a8cded2b`
- [Full source and target](SAMPLES.md#case-029)

All six contiguous time slots, five table columns, prioritization explanations and two actions retained.

## Case 030: NDA alert mockup — KEEP

- v10: `train:35687`; original: `train:107112`
- Record: `archive-train-107112-7416605ec8641f94`
- [Full source and target](SAMPLES.md#case-030)

All document attributes/status rows, the mockup and SSO assumptions, urgency and three actions retained.

## Case 031: Bali spa comparison — KEEP

- v10: `train:37041`; original: `train:4814`
- Record: `archive-train-004814-79d5491473e0246f`
- [Full source and target](SAMPLES.md#case-031)

Three options are correctly compared against duration/budget, and Spa C's services, date, two-person total and actions are fully retained.

## Case 032: Europa science-fiction story — KEEP

- v10: `train:37670`; original: `train:107289`
- Record: `archive-train-107289-401137b64b9979b3`
- [Full source and target](SAMPLES.md#case-032)

All six story paragraphs and four fictional infrastructure details are preserved; fantastical worldbuilding is not misclassified as factual misinformation.

## Case 033: DocuSign field-placement guide — REPAIR

- v10: `train:39477`; original: `train:135286`
- Record: `archive-train-135286-e0ce756d9979e5d7`
- [Full source and target](SAMPLES.md#case-033)

Target z keeps tag names but omits their positions: Signature in the main signature block, Date Signed adjacent, Initials at the bottom of each page. Restore these three short source instructions through regeneration.

Tags: `target_short_instruction_omission`

## Case 034: $100k savings calculation — REPAIR

- v10: `train:39624`; original: `train:76867`
- Record: `archive-train-076867-6b5b4396c58a9325`
- [Full source and target](SAMPLES.md#case-034)

Source's stated annuity formula/rate/139 deposits do not reproduce $570.06 monthly. Recalculate under the explicit timing/rate convention, then update principal, interest and explanatory percentages together; target copies the source numbers.

Tags: `source_arithmetic`

## Case 035: New York-London flight selection — REPAIR

- v10: `train:40262`; original: `train:129773`
- Record: `archive-train-129773-3a2a1e2c07fa4418`
- [Full source and target](SAMPLES.md#case-035)

All three rows have the same 12-hour departure/arrival clock gap on the same route/date but claim 10h/6h/8h durations. The fastest-flight selection is therefore not trustworthy without correcting source date/time/duration data.

Tags: `source_time_contradiction`

## Case 036: 12-week 5k program — REPAIR

- v10: `train:41986`; original: `train:40335`
- Record: `archive-train-040335-8c9b17fe96ee266f`
- [Full source and target](SAMPLES.md#case-036)

Target shortens the interval header and loses 'Repeat 4-6 times' plus the Saturday association. Source also places Dec 1, 2024 race day in a Saturday column although that date is Sunday and Sunday is listed as full rest. Restore qualifiers and resolve race-day exception.

Tags: `target_header_qualifier_omission`, `source_calendar_contradiction`

## Case 037: Two-hour study playlist — KEEP

- v10: `train:42478`; original: `train:62752`
- Record: `archive-train-062752-5bfec7345ff80c9d`
- [Full source and target](SAMPLES.md#case-037)

All ten tracks with contiguous 120-minute timing, genre/description, three phases, instrumental qualifier and four links retained.

## Case 038: Toronto emergency insulin pharmacies — REJECT

- v10: `train:43299`; original: `train:1214`
- Record: `archive-train-001214-770d2d9ec34b649d`
- [Full source and target](SAMPLES.md#case-038)

Source asserts all three named pharmacies are verified, open 24 hours and stock emergency insulin, but the archive contains no verification timestamp, live stock evidence or recoverable URL map. Target is faithful. Those consequential claims could not be confirmed in this audit; quarantine pending actual pharmacy/source verification, not a declaration that each store is nonexistent.

Tags: `source_unverified_medical_availability`

## Case 039: London job-offer comparison — KEEP

- v10: `train:43406`; original: `train:122171`
- Record: `archive-train-122171-e125854032e165ca`
- [Full source and target](SAMPLES.md#case-039)

Five comparison rows and full recommendation/assumptions preserved. Annual time figures are explicitly rough estimates; no target-only arithmetic introduced.

## Case 040: Rain-audio fade tools — KEEP

- v10: `train:44153`; original: `train:101378`
- Record: `archive-train-101378-6fd4371a6eb51171`
- [Full source and target](SAMPLES.md#case-040)

All four tool rows, three recommendations, capability limitation, six action destinations and source links preserved.

## Case 041: Python data-analysis roadmap — KEEP

- v10: `train:45106`; original: `train:101725`
- Record: `archive-train-101725-9d3b8b0504560bc0`
- [Full source and target](SAMPLES.md#case-041)

All four phases and complete skills/deliverables, three tool buttons and three source references retained; abbreviated button labels remain unambiguous.

## Case 042: Pho preparation guide — KEEP

- v10: `train:45306`; original: `train:28904`
- Record: `archive-train-028904-5bb402e49dff4e26`
- [Full source and target](SAMPLES.md#case-042)

All four cooking stages, timing matrix and serving instructions retained through the Instructions tab; source food-safety suitability is not independently certified.

## Case 043: $5,200 budget distribution — KEEP

- v10: `train:46653`; original: `train:66247`
- Record: `archive-train-066247-36a7c6c2296f2d74`
- [Full source and target](SAMPLES.md#case-043)

All allocations and expense/savings details preserved; fixed costs sum to $2,050, wants to $1,560 and savings to $1,040 as stated.

## Case 044: Needs-budget utilization — KEEP

- v10: `train:46656`; original: `train:89643`
- Record: `archive-train-089643-02d69703ff808b41`
- [Full source and target](SAMPLES.md#case-044)

All allocations/totals and yearly savings are correct and preserved. Optional minor polish: 80.7692% is printed as 80.7% rather than rounding to 80.8%; no material change to the budget.

Tags: `minor_rounding_note`

## Case 045: $4,500 allocation strategy — KEEP

- v10: `train:47564`; original: `train:98441`
- Record: `archive-train-098441-0f7228ae662119db`
- [Full source and target](SAMPLES.md#case-045)

50/30/20 amounts and $10,800 annual savings are correct; both tables, examples, caveats and four links retained.

## Case 046: Used family-SUV comparison — REPAIR

- v10: `train:48799`; original: `train:45383`
- Record: `archive-train-045383-6db30ec4724fe2dc`
- [Full source and target](SAMPLES.md#case-046)

Target drops the short third option descriptors, notably Mazda's 'Best-in-class handling', which is absent elsewhere. Restore the full option fields rather than only the first two pipe-separated fields. Vehicle-year specifications also remain unverified.

Tags: `target_short_option_omission`

## Case 047: Tokyo-flight jazz queue — KEEP

- v10: `train:52192`; original: `train:123310`
- Record: `archive-train-123310-22b72816d37f4e9c`
- [Full source and target](SAMPLES.md#case-047)

All 12 artists/albums in alphabetical order, moods, links and narrative preserved. v10's escaped literal state pointer now resolves. Claimed 12-hour total cannot be confirmed without edition durations.

## Case 048: Wildlife kits under $2,500 — REPAIR

- v10: `train:55219`; original: `train:79381`
- Record: `archive-train-079381-0cd68c6064704b38`
- [Full source and target](SAMPLES.md#case-048)

Canon RF 100-400mm is described as weather-sealed/Full Body-Lens; Canon's own specification says dust/weather-resistant construction: None. Correct that source claim and regenerate the target. Fuji partial-body sealing and all live kit prices also need verification; not asserted as independently disproved.

Tags: `source_product_spec`

Evidence: [Canon RF100-400mm F5.6-8 IS USM specifications](https://downloads.canon.com/DMSD/rf100-400f5.6-8-isusm/RF100-400mm-F5.6-8-IS-USM_Downloadable-Spec-Sheet_V1.0.pdf). Manufacturer specifies no dust/weather-resistant construction, contradicting the sample's weather-sealed lens claim.

## Case 049: Portfolio daily movers — KEEP

- v10: `train:56755`; original: `train:39612`
- Record: `archive-train-039612-22e1f73dae2be14e`
- [Full source and target](SAMPLES.md#case-049)

Three holdings, percentages, largest-gainer/loser explanation and two links retained; internally consistent supplied snapshot.

## Case 050: Student oil-painting supplies — KEEP

- v10: `train:56759`; original: `train:515`
- Record: `archive-train-000515-b08829e0c2237bbd`
- [Full source and target](SAMPLES.md#case-050)

All ten colors, three brush sizes/uses, canvas advice and correctly summed $92 budget retained. Optional spelling polish: source/target say 'Odorsless'.

Tags: `minor_spelling_note`

## Case 051: Sales employee efficiency — REPAIR

- v10: `train:57251`; original: `train:29368`
- Record: `archive-train-029368-a23c02ca98937300`
- [Full source and target](SAMPLES.md#case-051)

Arithmetic is correct but source calls revenue/leads a 'conversion rate'. The displayed units are USD/lead, not a fraction of leads converting. Rename metric/formula/insights to revenue per lead or supply actual conversion counts before regeneration.

Tags: `source_metric_definition`

Evidence: [Google Ads: Conversion rate definition](https://support.google.com/google-ads/answer/2684489?hl=en). Conversion rate uses conversion counts divided by eligible interactions, not currency per lead. The source's USD/lead dimensional mismatch is independently evident.

## Case 052: Electrical safety audit — KEEP

- v10: `train:57394`; original: `train:108265`
- Record: `archive-train-108265-b08c82892103b3ad`
- [Full source and target](SAMPLES.md#case-052)

Five diagnostic rows, three costed professional-remediation options and all action/source links retained. No target-only risky electrical procedure introduced.

## Case 053: Chair delivery status — KEEP

- v10: `train:58529`; original: `train:18122`
- Record: `archive-train-018122-a474888daf4c9a05`
- [Full source and target](SAMPLES.md#case-053)

Order ID, delivery ETA, four progress states and both actions preserved.

## Case 054: Degree versus bootcamp guide — KEEP

- v10: `train:61089`; original: `train:116508`
- Record: `archive-train-116508-f65a76cda0c74ee8`
- [Full source and target](SAMPLES.md#case-054)

All seven comparison features, both route recommendations and all decision-framework caveats retained; program eligibility claims not comprehensively verified.

## Case 055: Home-purchase cost example — KEEP

- v10: `train:61604`; original: `train:124801`
- Record: `archive-train-124801-8a5dc4dcff507779`
- [Full source and target](SAMPLES.md#case-055)

Formula inputs, cost components and totals are consistent to displayed precision and fully retained; scenario finance assumptions not personalized advice.

## Case 056: Tokyo neighborhood comparison — KEEP

- v10: `train:62918`; original: `train:74345`
- Record: `archive-train-074345-a5eff7b5bb0c16e1`
- [Full source and target](SAMPLES.md#case-056)

Both neighborhoods, all four metrics, six profile details and action/source references preserved; property prices/safety ratings remain unverified scenario inputs.

## Case 057: New York LLC guide — KEEP

- v10: `train:63180`; original: `train:138391`
- Record: `archive-train-138391-d5a1882c50f0734c`
- [Full source and target](SAMPLES.md#case-057)

Four phases, six checklist rows and all official-link bindings retained. This is a conversion pass, not certification that every legal deadline/exception is covered.

## Case 058: Singapore rental yields — KEEP

- v10: `train:63961`; original: `train:114898`
- Record: `archive-train-114898-3edb7562c56f4dd7`
- [Full source and target](SAMPLES.md#case-058)

All prices, rents, correctly calculated gross yields and ranking preserved. Optional notation polish: the 0.33 spread is percentage points, not a relative-percent increase.

Tags: `minor_metric_notation_note`

## Case 059: Patient-recovery snapshot — KEEP

- v10: `train:64672`; original: `train:34825`
- Record: `archive-train-034825-07f4c2405c202741`
- [Full source and target](SAMPLES.md#case-059)

Day 1/3/7 observations, three proposed milestones and four action labels are fully preserved; clinical assumptions and claimed linearity are not independently certified.

## Case 060: 2018 Camry maintenance — REJECT

- v10: `train:65564`; original: `train:129027`
- Record: `archive-train-129027-e8e186ed4b1c642f`
- [Full source and target](SAMPLES.md#case-060)

Source mixes an engine-unspecified oil recommendation with a 24-month roadmap spanning 30,000 miles despite a stated 10,000-12,000 miles/year, plus blanket parts-replacement advice. Needs model/engine-specific maintenance-source verification and regeneration, not a target-only edit.

Tags: `source_time_contradiction`, `source_vehicle_maintenance`

Evidence: [Toyota 2018 Camry warranty and maintenance guide](https://assets.sia.toyota.com/publications/en/omms-s/T-MMS-18Camry/pdf/T-MMS-18Camry.pdf). Manufacturer distinguishes operating conditions and refers to the owner's manual for oil grade/viscosity. Supports requiring engine-specific maintenance verification; the sample's 30,000-mile/two-year conflict is independently visible in its own numbers.

## Case 061: Rome-Florence travel comparison — KEEP

- v10: `train:65713`; original: `train:129562`
- Record: `archive-train-129562-a647fca286596a03`
- [Full source and target](SAMPLES.md#case-061)

Both transport options, two-adult prices, durations, rental cost caveats and ZTL/parking guidance are preserved, with the train image actually bound to an Image component. No live price or local-regulation certification.

## Case 062: Morning audio automation instructions — KEEP

- v10: `train:67174`; original: `train:7444`
- Record: `archive-train-007444-6f5cde2ef24e868c`
- [Full source and target](SAMPLES.md#case-062)

Both times, volumes, audio selections, three configuration instructions and all action/icon references are retained. These are setup instructions, not evidence of an automation actually being created.

## Case 063: Generic product comparison — KEEP

- v10: `train:67187`; original: `train:42196`
- Record: `archive-train-042196-3a4936d4646dafaa`
- [Full source and target](SAMPLES.md#case-063)

All three products' prices, ratings, review counts and the source's balanced recommendation are retained. The original query/budget cap is unavailable, so a different preferred product cannot be inferred.

## Case 064: Cairo-Oslo January comparison — KEEP

- v10: `train:68092`; original: `train:146181`
- Record: `archive-train-146181-1dd663f4c34a7c39`
- [Full source and target](SAMPLES.md#case-064)

Weather table, correctly rounded Celsius/Fahrenheit values, rainfall, daylight and packing guidance are preserved. Historical/climatological inputs were not independently certified.

## Case 065: Photosynthesis and respiration explanation — REPAIR

- v10: `train:68181`; original: `train:134249`
- Record: `archive-train-134249-4968f514c32c9bc9`
- [Full source and target](SAMPLES.md#case-065)

Source calls energy a giant loop/cycle and describes oxygen-using cellular respiration as universal to all living things. Clarify that matter cycles but energy flows/dissipates, and distinguish aerobic respiration from anaerobic processes before regenerating the faithful target.

Tags: `source_science_concept`

Evidence: [OpenStax: Biogeochemical Cycles](https://openstax.org/books/concepts-biology/pages/20-2-biogeochemical-cycles). Matter is recycled, while energy moves directionally through ecosystems and dissipates as heat.

Evidence: [OpenStax: Cellular Respiration](https://openstax.org/books/microbiology/pages/8-3-cellular-respiration). Anaerobic respiration uses electron acceptors other than oxygen; oxygen-dependent respiration is not universal.

## Case 066: Real-estate QR solution roadmap — KEEP

- v10: `train:68935`; original: `train:134995`
- Record: `archive-train-134995-3a8550cb8193e17b`
- [Full source and target](SAMPLES.md#case-066)

Four implementation phases, all tasks/deliverables, three component specifications including their short third fields, and every action/resource are retained.

## Case 067: Monthly budget variance — KEEP

- v10: `train:69481`; original: `train:38181`
- Record: `archive-train-038181-39eae062b32c36b3`
- [Full source and target](SAMPLES.md#case-067)

All four expense rows, the variance formula and summary are retained. Budget 1,950, actual 2,030 and net variance -80 reconcile.

## Case 068: Portfolio rebalancing — KEEP

- v10: `train:70836`; original: `train:88910`
- Record: `archive-train-088910-fee3b6873add7730`
- [Full source and target](SAMPLES.md#case-068)

All three assets, target calculations and the 300 cash transfer split into 200 stocks/100 bonds reconcile and are preserved. Optional notation polish: 1.8 percentage-point cash overweight, not a relative percentage.

Tags: `minor_metric_notation_note`

## Case 069: VS Code Python setup — REPAIR

- v10: `train:71639`; original: `train:127138`
- Record: `archive-train-127138-454111fa0e33558b`
- [Full source and target](SAMPLES.md#case-069)

Conversion preserves the complete code block, but the source includes obsolete python.formatting.provider configuration and implies defaultInterpreterPath changes always switch an already selected interpreter. Update to supported settings and explicit interpreter selection guidance; verify against current Microsoft documentation.

Tags: `source_obsolete_configuration`

Evidence: [Microsoft: Python settings reference](https://code.visualstudio.com/docs/python/settings-reference). defaultInterpreterPath is consulted on first workspace load; changing it after selecting an interpreter does not switch the selected interpreter.

Evidence: [Microsoft: migration to Python tool extensions](https://github.com/microsoft/vscode-python/wiki/Migration-to-Python-Tools-Extensions). python.formatting.provider is a removed/deprecated setting; formatter-extension settings should be used.

## Case 070: Madrid home-security plan — KEEP

- v10: `train:72259`; original: `train:24445`
- Record: `archive-train-024445-476b87bd5ab12e10`
- [Full source and target](SAMPLES.md#case-070)

Four camera placements, six sensors, all seven budget rows and four implementation phases are preserved. Hardware totals exactly 1,200 euros; no live equipment/security-coverage certification.

## Case 071: Leaking water-heater assessment — REPAIR

- v10: `train:72949`; original: `train:69697`
- Record: `archive-train-069697-55f3a3eb88988c8c`
- [Full source and target](SAMPLES.md#case-071)

Source tells the reader to touch hot-outlet fittings without a burn precaution and says to begin saving if the tank shell leaks, without making immediate safe isolation/professional inspection clear. Target is faithful, but a qualified/manufacturer-grounded safety rewrite is needed; do not invent hardware-specific shutoff instructions.

Tags: `source_safety_caveat`

Evidence: [Rheem: water-heater warning signs](https://www.rheem.com/water-heating/articles/the-ultimate-guide-to-water-heater-noises-whats-normal-and-what-isnt/). Manufacturer advises stopping use and professional inspection for dripping/leak-related warning signs. This supports the audit's safety-review hold, not model-specific DIY instructions.

## Case 072: Twelve-week strength program — KEEP

- v10: `train:73859`; original: `train:2045`
- Record: `archive-train-002045-a1f6acf6bdd84bf2`
- [Full source and target](SAMPLES.md#case-072)

All three phases, ten exercises with sets/repetitions, progression conditions, rest and RPE guidance are preserved. Suitability for a particular beginner/medical condition is not certified.

## Case 073: Job-offer cash comparison — REPAIR

- v10: `train:74282`; original: `train:18468`
- Record: `archive-train-018468-af93a9d71ca51e52`
- [Full source and target](SAMPLES.md#case-073)

The opening claims Offer C has the highest potential total cash, yet the correct table/final verdict give B 92,000 and C 89,600. Correct the contradictory opening in the source, then regenerate the target; the arithmetic rows themselves are correct.

Tags: `source_internal_contradiction`

## Case 074: Plumbing appointment confirmation — KEEP

- v10: `train:74523`; original: `train:139864`
- Record: `archive-train-139864-a60612b439f005cf`
- [Full source and target](SAMPLES.md#case-074)

Exact date, arrival window, technician, address, confirmation wording and reschedule action are retained. Treated as supplied scenario data, not proof of a real booking.

## Case 075: PhD-defense roadmap — KEEP

- v10: `train:74559`; original: `train:133864`
- Record: `archive-train-133864-85ee8528289b1bd2`
- [Full source and target](SAMPLES.md#case-075)

All six phases, dates, tasks, deliverables, buffer week and actions are preserved; no query-specific university procedure is inferred.

## Case 076: Renter budget allocation — KEEP

- v10: `train:75220`; original: `train:107139`
- Record: `archive-train-107139-1355b9c9f258010f`
- [Full source and target](SAMPLES.md#case-076)

All allocations, two conditions and all three renter-needs descriptions/statuses are preserved. 2,250 + 1,350 + 900 = 4,500; escaped JSON-pointer paths correctly address slash-prefixed state keys.

## Case 077: Stock-tracking spreadsheet guide — REPAIR

- v10: `train:76930`; original: `train:10747`
- Record: `archive-train-010747-3c2ae8aa758d40b0`
- [Full source and target](SAMPLES.md#case-077)

Source defines columns Date/Ticker/Closing Price/Daily % Change but then gives =(B2-C2)/C2, which would subtract a price from the ticker text under that layout. GOOGLEFINANCE price is also labeled a closing price without a historical-date/close request. Correct the source's column/formula mapping and quote type, then regenerate.

Tags: `source_formula_reference`, `source_metric_definition`

Evidence: [Google: GOOGLEFINANCE](https://support.google.com/docs/answer/3093281?hl=en). price without historical dates is a real-time quote (potentially delayed); historical close and closeyest are separate attributes. This is distinct from the source's internally mismatched B/C column formula.

## Case 078: Fictional Vortex-9 manual — KEEP

- v10: `train:77671`; original: `train:133563`
- Record: `archive-train-133563-56d193f1cd0dac84`
- [Full source and target](SAMPLES.md#case-078)

All fictional calibration values, sarcastic prose, diagnostic rows and actions are retained. Do not label explicit sci-fi absurdity as a factual hallucination; without the original query the requested tone cannot be judged. Consider genre tagging for training balance.

Tags: `genre_scope_note`

## Case 079: Senior diabetes wellness routine — REPAIR

- v10: `train:78317`; original: `train:147515`
- Record: `archive-train-147515-841d30fb0733b1ad`
- [Full source and target](SAMPLES.md#case-079)

Target is faithful, but the source presents fixed twice-daily medication times and monitoring as a tailored NHS-style routine without the medicine, prescription or clinician context in the archive. Require that context or a qualified rewrite explicitly deferring medication timing to the prescription; do not automatically choose a dosing schedule.

Tags: `source_needs_clinical_context`

Evidence: [NHS: Treatment for type 2 diabetes](https://www.nhs.uk/conditions/type-2-diabetes/treatment/). Medication type/timing depends on prescribed treatment and care-team instructions. Missing prescription context is a verification gap; the audit does not prove the sample's exact times wrong for every possible patient.

## Case 080: Needs-budget review — KEEP

- v10: `train:78334`; original: `train:65849`
- Record: `archive-train-065849-13331d19d7352962`
- [Full source and target](SAMPLES.md#case-080)

All five expenses, exact 2,250 total, 2,600 cap and 350 surplus, formula and actions are preserved. Subscription categorization is treated as supplied scenario input.

## Case 081: Sequential-discount calculation — KEEP

- v10: `train:78812`; original: `train:128321`
- Record: `archive-train-128321-d800694fb01c1f22`
- [Full source and target](SAMPLES.md#case-081)

120 x 0.8 x 0.9 = 86.40, savings 33.60 and effective discount 28% are correct; all variables, stages, explanations and actions are retained.

## Case 082: Sole-proprietorship versus LLC — REPAIR

- v10: `train:78969`; original: `train:6013`
- Record: `archive-train-006013-483c4a5ddba86725`
- [Full source and target](SAMPLES.md#case-082)

Target creates a separate entity column but labels it Liability, alongside the actual Liability column. Correct the generated column semantics (Business structure/Entity vs Liability); legal generalizations also need jurisdiction-specific review before being treated as advice.

Tags: `target_table_column_semantics`

## Case 083: Osaka restaurant selection — KEEP

- v10: `train:79642`; original: `train:89514`
- Record: `archive-train-089514-a64ff2ffa33f7389`
- [Full source and target](SAMPLES.md#case-083)

All three supplied options, price tiers, atmosphere and suitability plus the selected restaurant's reasons and actions are retained. Supplied restaurant profiles are not independently verified listings.

## Case 084: Monthly expense visualization — REPAIR

- v10: `train:81069`; original: `train:106355`
- Record: `archive-train-106355-ab1a6c7fb2ad5a98`
- [Full source and target](SAMPLES.md#case-084)

Source explicitly specifies a pie chart titled Monthly Expense Distribution; target has tables and text only, no Chart. The data is preserved but the requested visual role is absent. Current semantic gate only detects a narrower line-chart pattern. Minor rounding in food share is secondary.

Tags: `target_requested_chart_missing`

## Case 085: Auckland home-sale plan — REPAIR

- v10: `train:81518`; original: `train:134350`
- Record: `archive-train-134350-9636322a4c50dac5`
- [Full source and target](SAMPLES.md#case-085)

All dates, four investments and estimates are preserved, but the Sources section and provider names TradeMe Property, Realestate.co.nz and QV New Zealand disappear. Reusing the URLs in generically labeled action buttons preserves navigation tokens, not source attribution.

Tags: `target_source_attribution_omission`

## Case 086: Financial-planner meeting preparation — KEEP

- v10: `train:82286`; original: `train:116386`
- Record: `archive-train-116386-30052df8ca49eda8`
- [Full source and target](SAMPLES.md#case-086)

All seven document rows, three discussion options including their short purpose labels, and three named resources/actions are preserved.

## Case 087: Bulk QR-scanning design — KEEP

- v10: `train:87061`; original: `train:101778`
- Record: `archive-train-101778-d16d74fc286e9a3e`
- [Full source and target](SAMPLES.md#case-087)

Four phases, all tasks, 2-second cooldown, HUD text and actions/source labels are preserved. Zero-latency/never-freezes language is an aspirational design claim, not a measured implementation guarantee.

Tags: `performance_claim_note`

## Case 088: Paris hotel comparison — REPAIR

- v10: `train:87790`; original: `train:49929`
- Record: `archive-train-049929-410cf9b7f25a6cbb`
- [Full source and target](SAMPLES.md#case-088)

All prices, totals and recommendation logic match, but target adds bookingUrl/actionLabel for Hotel C although the source supplies booking actions only for A and B. Static table extraction projects only declared columns here: this is unsupported latent action metadata, not a demonstrated visible Book Hotel C button. Remove unsupported metadata through the pipeline/regeneration; do not invent a destination.

Tags: `target_unsupported_action_metadata`

## Case 089: AP Biology study plan — REPAIR

- v10: `train:88065`; original: `train:106733`
- Record: `archive-train-106733-5a0c0e82d980dd76`
- [Full source and target](SAMPLES.md#case-089)

Target is faithful, but source conflates the inner mitochondrial membrane with the proton-gradient location for photosynthesis. Distinguish mitochondrial inner membrane (respiration) from thylakoid membrane (photosynthesis) and regenerate; exam-date assumptions cannot be checked without year/query.

Tags: `source_science_concept`

Evidence: [OpenStax: Light-Dependent Reactions](https://openstax.org/books/biology-2e/pages/8-2-the-light-dependent-reactions-of-photosynthesis). Photosynthetic proton accumulation is across the thylakoid membrane, analogous to but distinct from the mitochondrial membrane used in respiration.

## Case 090: Sales-leadership resume examples — REPAIR

- v10: `train:88465`; original: `train:5673`
- Record: `archive-train-005673-dc4d5b9d89b64f2b`
- [Full source and target](SAMPLES.md#case-090)

Target preserves all five bullets, but source says market share increased 12% as measured by new-logo acquisition alone. Customer acquisition and market share are different metrics; obtain the market denominator or rewrite as supported customer growth, without inventing resume achievements.

Tags: `source_metric_definition`

## Case 091: Berlin DevOps transition plan — KEEP

- v10: `val:127`; original: `train:69436`
- Record: `archive-train-069436-c75a97d2275030b5`
- [Full source and target](SAMPLES.md#case-091)

All three roadmap phases, nine project tasks, market-alignment prose and actions/source labels are preserved. Salary/employer prerequisites are unverified scenario advice, not validated hiring facts.

## Case 092: Bank-deposit eligibility — KEEP

- v10: `val:191`; original: `train:103893`
- Record: `archive-train-103893-89f70e27db0bfc8c`
- [Full source and target](SAMPLES.md#case-092)

Eligibility, 15,000 minimum-deposit shortfall and one-year 420 interest/10,420 balance are internally correct and preserved. P x APY x t is only appropriate here for the specified one-year period, not a general compound-growth formula.

## Case 093: Temperature conversion — KEEP

- v10: `val:249`; original: `train:39795`
- Record: `archive-train-039795-a3f77523cc792d5c`
- [Full source and target](SAMPLES.md#case-093)

All three temperature conversions and formula are correct and preserved. This audit verifies conversion of supplied inputs, not that the temperatures are authentic historical NYC observations.

## Case 094: RPG narrative comparison — KEEP

- v10: `val:423`; original: `train:103216`
- Record: `archive-train-103216-12be6f0f456e426c`
- [Full source and target](SAMPLES.md#case-094)

All four games, twelve scores, four profiles, rationale and links are retained. Ratings are subjective; no independent industry-standard scoring method is supplied. Prefer an explicit subjective/illustrative label if used as factual evaluation data.

Tags: `subjective_rating_provenance_note`

## Case 095: Account-executive onboarding — KEEP

- v10: `val:521`; original: `train:8045`
- Record: `archive-train-008045-1e8c9a1bcd41a81f`
- [Full source and target](SAMPLES.md#case-095)

All three milestone rows, twelve detailed tasks and two actions are retained, including counts and dates.

## Case 096: Iceland winter-parka recommendations — REJECT

- v10: `val:702`; original: `train:79549`
- Record: `archive-train-079549-4628aed5e7e60ce0`
- [Full source and target](SAMPLES.md#case-096)

Source provides unsupported exact -15/-12/-10 C ratings, an underspecified Helly Hansen model and the unverified name Patagonia Tres Peaks. Patagonia's located manufacturer page identifies Tres 3-in-1, but that is not proof they are the same item. Rebuild using identified products and documented performance conditions; do not infer nonexistence from a failed lookup.

Tags: `source_product_identity`, `source_unsupported_safety_rating`

Evidence: [Patagonia: Men's Tres 3-in-1 Parka](https://www.patagonia.com/product/mens-tres-3-in-1-parka/28389-NENA.html). Located manufacturer product is Tres 3-in-1, not the sample's Tres Peaks wording. The audit did not establish a matching product or verify any of its claimed exact minimum temperatures.

## Case 097: Japan entry-document checklist — REPAIR

- v10: `val:770`; original: `train:86743`
- Record: `archive-train-086743-7bd193c65528f0ad`
- [Full source and target](SAMPLES.md#case-097)

Target faithfully reproduces the source's must-present list, but Visit Japan Web is not mandatory. Separate optional digital processing from required entry documents and avoid guaranteeing admission; verify jurisdiction/date-specific guidance before regenerating.

Tags: `source_optional_requirement`

Evidence: [Consulate-General of Japan in Sydney: visa FAQ](https://www.sydney.au.emb-japan.go.jp/document/english/visa_info/visa-faq-feb-2023.pdf). Visit Japan Web is recommended rather than mandatory. The source's must-present list incorrectly makes the QR registration a requirement.

## Case 098: Berlin apartment ranking — KEEP

- v10: `val:774`; original: `train:24653`
- Record: `archive-train-024653-e6e3d515c2ec539c`
- [Full source and target](SAMPLES.md#case-098)

All three apartments, sizes, rents, commute times, rank priorities and option attributes are preserved. Ranking matches the stated primary commute-time criterion.

## Case 099: Gym-attendance analysis — KEEP

- v10: `val:1168`; original: `train:113299`
- Record: `archive-train-113299-fd016be6795e8da8`
- [Full source and target](SAMPLES.md#case-099)

All 14 dated rows preserved; 9 successes + 5 misses, 64.29%, a March 8-10 three-day streak and 4.5 weekly sessions reconcile. Every missed day is followed by attendance.

## Case 100: Package transit durations — REPAIR

- v10: `val:1753`; original: `train:70367`
- Record: `archive-train-070367-6372194920d5ddc8`
- [Full source and target](SAMPLES.md#case-100)

Total 49.25 hours is correct, but source/target call the 18.75-hour second interval the longest. First interval (May 10 08:00 to May 11 14:30) is 30.5 hours. Correct the comparison claim, preserving all timestamps.

Tags: `source_time_contradiction`
