# v10 dataset: fresh manual audit of 100 samples

Audit date: 2026-09-16. Dataset: `training/outputs/datasets/full_data_archive_recovered_v10`.

## Conclusion

**69 keep, 26 repair, 5 reject/regenerate.** v10 is structurally clean in this sample, but it is **not yet semantically clean enough to describe as fully quality-approved**. This is an audit judgment, not a measured model score or a statement that the remaining dataset has the same defect rate.

I manually read the complete source response and complete A2UI Express target for each of the 100 cases. Automated checks supplemented that reading; they did not produce the manual verdicts.

| Disposition | Train sample | Validation sample | Total |
| --- | ---: | ---: | ---: |
| KEEP: no material defect found within this review | 62 | 7 | **69** |
| REPAIR: identified correction/review path; withhold as-is | 24 | 2 | **26** |
| REJECT: rebuild or substantiate the current pair before reuse | 4 | 1 | **5** |
| Total reviewed | **90** | **10** | **100** |

KEEP does not certify every real-world claim, clinical recommendation, legal rule, price, or link. REPAIR does not mean an automatic offline repair is already safe or has been performed. REJECT means reject this version for training, **not delete the original or declare it permanently unrecoverable**.

## What was sampled and checked

- Population: **91,115 train + 1,862 validation = 92,977 records**.
- Fixed random seed: **2026091603**. Uniform reservoir sampling without replacement within each split: 90 train and 10 validation.
- Previously reviewed 100 + 10 source families/original source hashes were excluded to avoid simply rechecking cases used to develop v10. This excluded 69 current train rows and 9 validation rows; eligible populations were 91,046 and 1,853 respectively.
- All 100 selected record IDs and source-family IDs are distinct. Exact selections and exclusions are saved in [checks.json](checks.json).
- **100/100 passed** strict preparation, identity verification and effective semantic-hash checks. Existing warning, paragraph-gap, letter-gap and v10 semantic-policy checks flagged **0/100**. That is a pass for these checks, not every possible repository quality scorer.
- Ten sampled records carried v10 repair metadata. The remaining 90 were not changed by that specific v10 repair pass; they may have older recovery transformations.
- All 100 **lack the original user query and original query ID**. This is therefore a response-to-UI audit, not a complete original-query-answer relevance audit.
- 52 records have a nonempty current recovered URL map; 48 do not. All still list missing historical original-URL-map provenance. I checked symbolic source/target bindings, not every live destination or media fetch.
- The original train/validation files, manifest and eight other listed dataset artifacts were rehashed after the audit: **unchanged**. See [integrity_after.json](integrity_after.json).

Validation is intentionally oversampled, and previously reviewed families were excluded. Do not multiply the sample percentages by 92,977, use them as population accuracy, or directly compare them with the previous v9 sample as a controlled experiment.

## Source quality versus UI conversion quality

| Manual finding | Count | Meaning |
| --- | ---: | --- |
| No target-conversion issue identified | 92 | May still contain a flawed source answer |
| A2UI target issue | 8 | Seven content/presentation problems and one unsupported latent-metadata problem |
| Source-answer issue or verification concern | 24 | Arithmetic, contradictions, factual claims or missing safety/provenance context |
| Both source and target issue | 1 | Case 036; already included in the two preceding counts |
| Unique samples requiring attention | **31** | 8 + 24 - 1 |

Important: a faithful conversion of a wrong answer is not necessarily a bad response-to-UI mapping. The converter should not learn to silently rewrite the user's supplied facts. For source-quality fixes, correct/verify the source and regenerate its paired target; for target-only fixes, preserve the source.

## Eight A2UI target findings

All coordinates below are **v10 file line numbers**, not original archive row numbers. Full original coordinates and source/target text are in the linked case review.

| Case | v10 row | Finding | Required handling |
| --- | --- | --- | --- |
| 028 | train:30894 | Wearable rows were transposed, but device names remained column labels. The `focus` values remain in state without a declared display column. | Regenerate a correctly labeled matrix/entity layout with all three focus values visible. |
| 033 | train:39477 | DocuSign guide loses short placement details: main signature block, adjacent to signature, bottom of each page. | Preserve short instructions, not only long prose. |
| 036 | train:41986 | Running-plan header loses `Repeat 4-6 times` and the Saturday qualifier. Source also puts a 2024-12-01 race into a Saturday schedule even though that date is Sunday. | Restore header semantics and resolve the source's race-day exception. |
| 046 | train:48799 | SUV option loses `Best-in-class handling` from the Mazda option's short third field. | Preserve meaningful pipe-separated option fields. |
| 082 | train:78969 | Entity column is labeled `Liability`, duplicating the actual Liability column. | Preserve column meaning when extracting entity names from source cells. |
| 084 | train:81069 | Explicit pie-chart request and title have no Chart component; tables alone remain. | Preserve the requested visual role, using the supported contract. |
| 085 | train:81518 | Source-provider names disappear: TradeMe Property, Realestate.co.nz, QV New Zealand. Their URL tokens survive only in generic action buttons. | Preserve attribution labels as well as URL identity. |
| 088 | train:87790 | Target invents `bookingUrl`/`actionLabel` for Hotel C, which had no source booking action. | Remove unsupported action metadata via generation/pipeline logic. Static extraction does not demonstrate a visible Hotel C button here. |

The first row's missing `focus` data is not rescued by being stored in state: native direct-table extraction projects declared columns. Conversely, the Hotel C finding is explicitly **latent unsupported data**, not a claim that a button was observed on a device.

## Source-answer findings

### Clear internal arithmetic and consistency examples

- **002, train:2295:** 5 km time falls from 25 to 22.5 minutes. Speed increases **11.11%**, not the source's 10.4%.
- **034, train:39624:** Applying the sample's own savings formula gives **569.27/month**, not 570.06. The stated 570.06 payment would produce approximately **100,139.47**, not exactly 100,000, under that formula. Deposit timing and rate convention must remain explicit.
- **035, train:40262:** Three New York-London options all show a 12-hour clock gap but claim 10-, 6- and 8-hour durations. A common route/date timezone offset cannot reconcile all three.
- **051, train:57251:** Revenue divided by leads has units **USD/lead**; the source calls it conversion rate. Rename the metric or obtain actual conversion counts; do not invent counts. [Definition reference](https://support.google.com/google-ads/answer/2684489?hl=en).
- **073, train:74282:** Introduction claims Offer C has the highest total cash; its own correct numbers are **B 92,000 > C 89,600 > A 89,250**.
- **077, train:76930:** Suggested sheet columns are Date/Ticker/Closing Price/Change, but `(B2-C2)/C2` uses the ticker column as a number. The `price` attribute is also not a historical closing-price request. [Google documentation](https://support.google.com/docs/answer/3093281?hl=en).
- **100, val:1753:** The source calls the 18.75-hour interval the longest, but the other interval is **30.5 hours**. Total 49.25 hours is correct.

Other repair holds cover ambiguous flight option IDs/budget scope (003, 027), contradictory ankle-recovery timing (015), science explanation clarity (065, 089), unsupported metric interpretation in resume examples (090), and missing prescription context (079). The latter is a **verification concern**, not proof that every possible patient would have different medication times. All case-specific reasons are in [CASE_REVIEW.md](CASE_REVIEW.md).

### Targeted external checks

- **048:** Canon explicitly lists no dust/weather-resistant construction for the RF100-400mm; the sample says it is weather-sealed. [Manufacturer specifications](https://downloads.canon.com/DMSD/rf100-400f5.6-8-isusm/RF100-400mm-F5.6-8-IS-USM_Downloadable-Spec-Sheet_V1.0.pdf).
- **069:** Microsoft documents the obsolete formatter setting and the fact that editing `defaultInterpreterPath` does not switch an already selected interpreter. [Migration guide](https://github.com/microsoft/vscode-python/wiki/Migration-to-Python-Tools-Extensions), [settings reference](https://code.visualstudio.com/docs/python/settings-reference).
- **071:** Leaking-water-heater advice needs immediate safety/professional-inspection guidance, not just replacement-budget planning. [Manufacturer guidance](https://www.rheem.com/water-heating/articles/the-ultimate-guide-to-water-heater-noises-whats-normal-and-what-isnt/).
- **097:** Visit Japan Web is optional/recommended, not a mandatory document as the sample's list implies. [Japanese consular FAQ](https://www.sydney.au.emb-japan.go.jp/document/english/visa_info/visa-faq-feb-2023.pdf).

These were targeted checks of suspicious claims, not an exhaustive fact check of the 100 sources. Full references and precisely bounded findings are in [external_evidence.json](external_evidence.json).

### Five reject/regenerate recommendations

| Case | v10 row | Why a local mechanical fix is insufficient |
| --- | --- | --- |
| 010 | train:9062 | Camera recommendation mixes Sony Alpha 1/Alpha 1 II capabilities and does not substantiate complete kits against the 4,000 budget. Rebuild verified model/kit comparisons. [Sony announcement](https://www.sony.com.hk/press/pdf/20241120_e.pdf). |
| 013 | train:12794 | Space-game comparison mixes base games with unverified space variants/adaptations. Publishers describe [Root as woodland](https://ledergames.com/products/root-a-game-of-woodland-might-and-right) and [Scythe as alternate-history 1920s](https://europe.stonemaiergames.com/products/scythe). Do not invent replacement games. |
| 038 | train:43299 | Emergency-insulin locations/hours/stock are called verified without usable verification evidence. Audit could not substantiate these consequential claims. This is not proof every listed pharmacy is nonexistent. |
| 060 | train:65564 | Claimed two-year/10-12k miles-per-year schedule spans 30k miles and gives engine-unspecified maintenance/replacement guidance. Verify against the [2018 Camry maintenance guide](https://assets.sia.toyota.com/publications/en/omms-s/T-MMS-18Camry/pdf/T-MMS-18Camry.pdf) and exact vehicle. |
| 096 | val:702 | Parka identities and exact minimum-temperature claims are unsubstantiated. Located [Patagonia product](https://www.patagonia.com/product/mens-tres-3-in-1-parka/28389-NENA.html) is Tres 3-in-1; do not assume it is the sample's Tres Peaks or transfer ratings blindly. |

## What the current filters miss

These are observations from the current working-tree code, not fixes applied by this audit:

1. **Short semantic content:** `source_units()` ignores pipe-containing lines and only includes prose of at least 45 characters/eight words. It can miss small but essential instructions and option fields (033, 046).
2. **Header meaning and entity association:** Word presence alone does not establish which device/column a value belongs to (028, 082), or preserve header repetition counts (036).
3. **General hidden fields:** Hidden-table checking currently targets `alert...`/`date...` keys, not an omitted `focus` column (028).
4. **Chart coverage:** The v10 gate is restricted to a line-chart pattern; the explicit pie chart in 084 escapes it.
5. **Citation attribution:** Presence of the referenced token is checked, but the human-readable source/provider label can disappear (085).
6. **Unsupported latent content:** Extra row action metadata can pass while all source tokens remain valid (088).
7. **Source truth:** Factual correctness, arithmetic relationships, contradictory recommendations and live medical stock claims are not established by schema or source-to-target fidelity.

Code anchors: [semantic gates](../../src/ir_training/data/archive_semantic_review.py), [native table projection](../../../android/app/src/main/java/com/samsung/genuicraft/renderer/flat/compose/table/FlatTableModelExtraction.kt), [table render-model extraction](../../../android/app/src/main/java/com/samsung/genuicraft/renderer/flat/compose/FlatDirectTableRender.kt).

## Recommended next work

1. Add regression fixtures for the eight target findings and generalize the checks above. Match semantic fields and entity associations, not global word overlap. Do not make every paraphrase or every hidden implementation field a rejection.
2. Add bounded arithmetic/unit/date consistency checks for structurally recognizable calculations. Escalate ambiguous rates, time zones, missing IDs and unspecified assumptions rather than guessing.
3. Maintain a separate source-verification queue for product, legal, clinical and live-availability claims. Correct a source only with evidence, and regenerate the paired target through Stage 3; do not manually edit IR labels. If Stage 3 cannot run, hold these cases out rather than claiming repair completion.
4. Preserve original queries, source timestamps, generator provenance and complete URL maps in future data. These would resolve ambiguities this archive cannot.
5. Check semantic/template diversity before adding more augmentation. Several accepted samples repeat simple budget/roadmap structures. No exact-near-duplicate population estimate was performed in this audit.
6. Freeze repaired data, then audit a **new unseen sample** and evaluate actual model checkpoints. Keep the Golden sets out of training/validation and retain Golden35 for final holdout evaluation. This audit produced no Golden scores.

Minor rounding/spelling and subjective/fictional content were not automatically rejected. Examples kept with notes include 044, 050, 058, 068, 078, 087 and 094. This avoids inflating the rejection count by treating every stylistic choice or mock scenario as a factual failure.

## Evidence files and reproduction

- [CASE_REVIEW.md](CASE_REVIEW.md): manual reason and disposition for every one of the 100 cases.
- [SAMPLES.md](SAMPLES.md): complete unchanged source responses and A2UI Express targets.
- [samples.jsonl](samples.jsonl): portable audit copies, line numbers, hashes, checks, components and available current URL maps.
- [manual_verdicts.json](manual_verdicts.json): hand-authored verdicts; array fields are case, disposition, subject, rationale, tags.
- [summary.json](summary.json): counts, split breakdown, issue case lists and recalculated arithmetic.
- [checks.json](checks.json): fixed seed, selected rows, exclusion identities, checks and pre-audit dataset hashes.
- [integrity_after.json](integrity_after.json): matching post-audit artifact hashes.
- [external_evidence.json](external_evidence.json): targeted primary-source references and limitations.

To rebuild the generated case report from the packaged evidence (does not edit data):

```powershell
python training/reports/v10_manual100_20260916/build_report.py
```

Add `--verify-dataset` if the same v10 dataset is present locally. The original sampler is `tmp/review_v10_random100_20260916.py`; its previous-audit exclusions and final selection are embedded in `checks.json`, so the selected sample is independently identifiable without those temporary folders.

No dataset records, filtering code, model checkpoints, Golden sets or unrelated work were changed by this audit. No training, device rendering or live action tests were run. Do not use structural passes or these judgments as a guarantee of failure-free training or improved Golden scores.
