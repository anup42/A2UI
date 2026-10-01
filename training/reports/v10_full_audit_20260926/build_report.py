"""Build the consolidated audit from completed read-only evidence."""
import collections, csv, gzip, json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]

def read(name):return json.loads((OUT/name).read_text(encoding='utf8'))
def pct(a,b):return f'{a:,} ({a/b*100:.2f}%)'

def main():
    s=read('structure_summary.json');l=read('leakage_counts.json');sv=read('serving_distribution.json');bc=read('bixby_source_distribution.json');sem=read('semantic_candidates.json')
    c=s['counts']['all'];n=c['rows'];st=s['stats']['all']
    # Slice statistics preserve full-corpus counts while avoiding huge row dumps in the report.
    cohorts=collections.defaultdict(collections.Counter);largest=[];extra=collections.Counter()
    rows_path=OUT/'structure_rows.jsonl'
    opener=lambda:gzip.open(OUT/'structure_rows.jsonl.gz','rt',encoding='utf8') if not rows_path.exists() else rows_path.open(encoding='utf8')
    with opener() as f, (OUT/'row_metrics.csv').open('w',newline='',encoding='utf8') as cf:
        keys=['split','line','id','valid','source_tokens','target_tokens','archived_sequence_tokens','deployed_sequence_tokens','component_count','depth']
        w=csv.DictWriter(cf,fieldnames=keys);w.writeheader()
        for raw in f:
            r=json.loads(raw);w.writerow({k:r.get(k) for k in keys})
            extra['deployed_prompt_over4096']+=r['deployed_prompt_tokens']>4096
            extra['archive_prompt_over4096']+=r['archived_prompt_tokens']>4096
            extra['any_current4096_gate']+=r['deployed_prompt_tokens']>4096 or r['deployed_sequence_tokens']>4096
            extra['sdk_input_chars_over120000']+=r['target_chars']>120000
            extra['sdk_components_over1024']+=r.get('component_count',0)>1024
            extra['graph_depth_over64']+=r.get('depth',0)>64
            for domain in r.get('table_domains',{}):
                cohorts[domain]['rows']+=1
                cohorts[domain]['over4096']+=r['deployed_sequence_tokens']>4096
            if r['target_tokens']>2048:largest.append({k:r[k] for k in keys if k in r})
    (OUT/'length_slices.json').write_text(json.dumps({'table_domain_rows':cohorts,'targets_over2048':largest,'additional_counts':extra},indent=2)+'\n',encoding='utf8')
    metrics='\n'.join(f"| {label} | {st[k]['p50']:,.0f} | {st[k]['p95']:,.0f} | {st[k]['p99']:,.0f} | {st[k]['max']:,} |" for label,k in [('Source only','source_tokens'),('Target only','target_tokens'),('Archived generation prompt','archived_prompt_tokens'),('Deployed generation prompt','deployed_prompt_tokens'),('Archived full sequence','archived_sequence_tokens'),('Deployed-prompt full sequence','deployed_sequence_tokens'),('Components','component_count'),('Graph depth','depth')])
    distribution='\n'.join(f"| {label} | {pct(c[k],n)} | {bc[b]}/50 |" for label,k,b in [('Markdown headings','source_has_markdown_heading','markdown_heading'),('Leading-pipe table text','source_has_pipe_table','pipe_table'),('Media scaffolding','source_has_media','media'),('Action/Quick Actions scaffolding','source_has_action','action'),('URL placeholders','source_has_url_placeholder','placeholder'),('Numeric citation markers','source_has_numeric_citation','numeric_citation'),('HTTP(S) substrings','source_has_live_url','live_url')])
    budgets='\n'.join(f"| {budget:,} | {pct(c[f'archived_sequence_over_{budget}'],n)} | {pct(c[f'deployed_sequence_over_{budget}'],n)} |" for budget in [4096,6144,8192])
    components='\n'.join(f"| {k} | {v:,} | {pct(s['counts']['all:rows_with_component'].get(k,0),n)} |" for k,v in sorted(s['counts']['all:component_counts'].items(),key=lambda x:-x[1]))
    domains='\n'.join(f"| {k} | {v:,} | {cohorts[k]['rows']:,} | {pct(cohorts[k]['over4096'],cohorts[k]['rows'])} |" for k,v in sorted(s['counts']['all:table_domains'].items(),key=lambda x:-x[1]))
    candidates='\n'.join(f"| {k.split(':',1)[1]} | {v:,} |" for k,v in sem['flagged_rows'].items() if k.startswith('all:'))
    flags={k:v for k,v in c.items() if k.startswith('flag:')}
    chart_rows=s['counts']['all:rows_with_component'].get('Chart',0)
    chart_note=('There are **zero supervised Chart targets** in the entire corpus, although the active contract supports Chart and the source scan finds 494 explicit chart-specification candidates without one. This is a concrete missing capability in the labels, not merely class imbalance.' if chart_rows==0 else f'Only {chart_rows:,} rows supervise Chart components; compare this coverage with the explicit chart requests retrieved by the semantic scan.')
    dup=l['all_rows_census']['raw']
    report=f'''# v10 complete dataset audit — 26 September 2026

## Assessment

**Do not treat unchanged v10 as a quality-approved dataset for the current Bixby/Perplexity converter.** Its syntax/graph integrity is strong, but source-to-target omissions, a pronounced production-input mismatch, prompt-budget expansion, and incomplete model lineage need attention before another long training run. More epochs alone do not address those issues.

This audit freshly processes **{n:,} records: 91,115 train + 1,862 validation**, from `training/outputs/datasets/full_data_archive_recovered_v10`. The restored files match the tracked `training/data/train/v10` bundle byte-for-byte. This audit did not change original datasets, training code, model weights or app code. It covers base v10 before any optional augmentation; no augmented rows were generated or audited. No new model training, export or device inference was performed.

“Complete” means every row received the enumerated machine checks. It does **not** mean every source fact or rendered screen was manually certified. Semantic evidence combines full-corpus candidate checks, 20 newly sampled complete-pair reviews, targeted examples, and verification that the previous audit's known defects remain present.

## Highest-priority findings

| Priority | Finding | Training/runtime consequence | Recommended handling |
|---|---|---|---|
| P1 | Synthetic media/action/URL scaffolding dominates v10; numeric citations are nearly absent. | If v10 is used unchanged, its supervision differs substantially from the measured Bixby50 response style. | Add a separate development cohort resembling real production responses, with faithful generated targets, natural prose, tables, citations without URLs, and no forced media/actions. Keep Bixby50 out of SFT. |
| P1 | Current preparation replaces the old 4,766-character system prompt with the 10,553-character deployed contract. | Many valid rows exceed 4,096 tokens under the local proxy; current preparation is designed to quarantine such rows. Actual retention requires the HF manifest. | Bind the exact training tokenizer and prepared manifest, measure retention, then choose a budget/prompt supported by the app. Never truncate targets to make them fit. |
| P1 | Confirmed targets omit schedules, passenger fields, short instructions, requested charts and entity relationships. | The labels teach incomplete conversions even though they compile successfully. | Add semantic regression gates, repair the generator/source where necessary, regenerate Stage 3 targets, and quarantine unresolved pairs. Do not manually rewrite target IR. |
| P1 | Training uses URL placeholders; the trained SDK takes raw text and has no request-scoped URL mask/restore map. | A correct learned placeholder cannot become the original clickable URL without a binding; raw URL inputs differ from supervision. | Make training and serving use the same reversible reference policy; test generated links and missing-reference behavior. |
| P1 | Remote prepared-v10 variants and the tested checkpoints lack a locally verified common training receipt. | Current raw-v10 or prompt checks cannot prove what the newer checkpoint actually learned. | Bind source hashes, prepared data, tokenizer/template, prompt, training config, checkpoint and export in one manifest. |
| P2 | Identical sources have multiple different canonical targets. | Alternative layouts can be legitimate. Variants need fidelity review for possible omissions/actions, and repeated sources receive extra weight. | Compare fidelity within each source group; retain validated variants intentionally or choose a consistently generated target. |
| P2 | Source arithmetic/consistency errors remain. | Faithful targets can still display bad information; this is distinct from conversion failure. | Correct and verify the source, then regenerate its pair. Do not train the converter to silently change supplied facts. |
| P2 | One valid target meets the app's sequential-ID repetition condition. | The current guard can stop a faithful generation before its children are defined. | Refine serving stop detection and regression-test the complete valid target and genuine loops. |
| P2 | All rows lack original query/intent/generator lineage; Bixby50 has prior prompt-development exposure. | Domain balancing, source truth checks and unbiased generalization claims are limited. | Preserve provenance for new data; add a genuinely unseen final cohort and report results by source family/domain. |

## 1. Integrity and structural validity

| Check | Fresh result |
|---|---:|
| Parsed corpus records | {n:,}/{n:,} |
| Strict Express/canonical graph + complete reachability + wire-schema + semantic serialization round-trip | {c['valid']:,}/{n:,} passed |
| Structural failures | {c.get('invalid',0):,} |
| Message/source/assistant consistency, empty content, envelope, replacement-character or special-token flags | {sum(flags.values()):,} flags |
| Distinct archived system/few-shot scaffolds | {len(s['counts']['all:scaffold_sha'])} |
| Effective source/target hash metadata mismatches | 0 |
| Train/validation assigned-split mismatches | 0 |

The full-corpus structural pass uses the production parser, reference inventory, serializer and unchanged wire-schema assertions. Its compiled schema path is checked against the production validator on Golden originals and deliberately invalid copies; the detailed parity receipt is [schema_parity.json](schema_parity.json). This is structural/compiler validation, **not 92,977 Android screenshot tests**.

All archived messages have matching final input/output bindings. There is no evidence here of prompt tokens accidentally used as labels: the current HF path masks the prompt with `-100`, supervises the completion, pads labels with `-100`, and rejects overflow instead of truncating. See [pipeline review](pipeline_review.md) for code anchors and focused tests.

## 2. Production distribution mismatch

The same lexical tests were applied to training+validation and the frozen 50 Bixby responses. These count recognizable source formatting, not semantic domains. The URL-placeholder regex in this table recognizes URL namespaces; the broader reference census also covers placeholders of other kinds. HTTP substring matching finds 42 rows, whereas the reference census requires a URL word boundary and counts 17. Neither check certifies a working destination; table matching here requires a leading pipe and is not a count of every tabular source.

| Source feature | v10 ({n:,} rows) | Bixby50 |
|---|---:|---:|
{distribution}

This is strong evidence of a **distribution gap**, not proof of a particular checkpoint's causal failure. v10 largely teaches conversion of sources already organized as titled sections, media descriptors and action blocks. The measured Bixby50 cohort contains numeric citations in 48/50 responses and media/action scaffolding or literal URLs in 0/50. Regeneration should preserve those original response forms rather than append synthetic icon/action sections simply to resemble v10.

The reference/provenance audit separately found **92,911 rows with reference placeholders**, **48,108 rows with nonempty current URL maps**, and **44,803 placeholder-containing rows with empty current maps**. That leaves **305,649 distinct source-placeholder occurrences counted per row without bindings**. Current maps bind URLs encountered during normalization; they do not reconstruct missing historical asset lineage. All records declare missing original query, generator and asset provenance. See [reference census](reference_counts.json) and [leakage/provenance review](leakage_review.md).

## 3. Prompt and sequence-length budget

Token counts below use the locally extracted SentencePiece vocabulary, SHA-256 `{s['tokenizer_sha256']}`, plus explicit BOS and the deployed non-thinking Gemma text template. They are exact for that local counting procedure and a **proxy for the missing remote HF training-tokenizer receipt**. They are not claims about the final accepted row count of an unobserved GPU run. The corpus-wide sequence measurements use stored target text. An independent probe compared 503 deterministic rows spanning both splits against canonical preparation and separately encoded prompt/completion: all token deltas were zero, with no threshold disagreement. This sample check does not substitute for the actual remote tokenizer. See [token probe](pipeline_token_probe.json).

| Quantity | Median | P95 | P99 | Maximum |
|---|---:|---:|---:|---:|
{metrics}

| Full-sequence limit | Over limit with archived prompt | Over limit with deployed prompt |
|---|---:|---:|
{budgets}

The current Golden launcher checks the **normalized** prompt and full supervised sequence against 4,096 tokens. It quarantines the whole row. Consequently, the raw bundle's 91,115 training rows must not be presented as the effective training set. A larger nominal model context does not override the launcher's `max_seq_length`.

At the 4,096 full-sequence threshold, the local proxy leaves **19,276 training rows and 385 validation rows** fitting that budget, before other preparation gates. These are hypothetical retention counts, not a verified GPU training receipt.

The deployed generation prompt alone exceeds 4,096 tokens in **{extra['deployed_prompt_over4096']:,} rows**. Sequence and prompt thresholds are separate settings and both must be included in the exact-tokenizer preparation report.

For a concrete example, train:1 is structurally valid and measures **3,265 tokens with the archived prompt versus 4,757 with the deployed prompt** under this local tokenizer. The same source/target pair moves from below to above the 4,096 threshold simply because the scaffold changes.

Only **{pct(c['target_over_2048'],n)}** stored targets exceed the SDK's 2,048 output-token budget under this tokenizer. The long repetitive outputs observed on-device therefore cannot be explained simply by claiming that most gold targets require more than 2,048 tokens. Source fidelity, output stopping, model fitting and export/runtime parity remain separate checks.

The archived prompt is consistent across the dataset; the current SDK asset matches the current shared builder. The risk is using or comparing different prepared versions without receipts, not random prompt mixing within this raw v10 corpus.

## 4. Confirmed semantic problems

The new fixed-seed complete-pair sample contains 18 train + 2 validation rows, excluding the previous 100 sampled rows: **16 KEEP, 3 REPAIR, 1 REJECT/REGENERATE**. KEEP means no material defect found in that review, not external factual certification. This small, split-stratified review must not be extrapolated to all {n:,} records. The previous 100-row review's 69/26/5 dispositions are historical evidence; their eight target defects are still present in the current corpus.

| v10 coordinate | Confirmed problem | Type |
|---|---|---|
| train:1702 | Detailed four-week activity schedule is lost; only the broader roadmap survives. | Target content loss |
| train:37244 | Passenger name, flight, seat and gate details are dropped. | Target field loss |
| train:14450 | Property restrictions/features are omitted. | Target qualifier loss |
| train:4964 | Detailed itinerary descriptions disappear. | Target content loss |
| train:60428 | Five source cells sit under four headers; target discards alert-checklist instructions. | Malformed source table plus target loss |
| train:69525 | Explicit chart title and axes produce tables/prose without a Chart component. | Visual-role loss |
| train:64773 | Values sum to 80,500, source says 73,500, percentages sum to 109.6%; target repeats errors and omits two specified charts. | Source contradiction plus target loss |
| train:73806 | Claims 450 remains after meeting all allocation targets even though targets total the entire 3,500 income. | Source contradiction |
| train:30894 | Transposed wearable table mislabels entities and hides `focus` values in state. | Entity/visible-field mismatch; previous finding |
| train:87790 | Hotel C receives unsupported booking/action metadata. | Latent invented action data; previous finding |

Full exact snippets, source identities, fresh verdicts, retained old findings and false-positive examples are in [semantic review](semantic_review.md), [fresh 20 pairs](semantic_fresh20.jsonl), and [fresh verdicts](semantic_fresh20_verdicts.json).

Full-corpus screening found these **review candidates**, not certified failure counts; categories overlap:

| Heuristic | Flagged rows |
|---|---:|
{candidates}

Exact-match tests can flag harmless reformatting (`10,000` vs `10000`), equivalent paraphrases, or conceptual chart discussion. Do not automatically discard all flagged rows. Conversely, source tokens being present somewhere in state do not prove that the right entity or visible table column shows them.

## 5. Duplicates, splits and benchmark exposure

- **{dup['unique_groups']:,} unique exact sources** across {n:,} rows.
- **{dup['repeated_groups']:,} repeated-source groups**, involving **{dup['rows_in_repeated_groups']:,} rows** and **{dup['extra_rows_beyond_one_per_group']:,} extra rows** beyond one per source. Each group has different raw targets and multiple stored `effective_semantic_sha256` hashes. A different graph hash alone does not prove a factual contradiction; layout choice changes it too.
- **0 exact duplicate targets.** Split-family identity is preserved; exact, whitespace-normalized, reference-insensitive and original-source signatures found **0 cross-split groups**.
- Bounded lexical search scored **134,009 candidate train/validation pairs** and found **0** meeting its trigram-Jaccard threshold of 0.60. This is not an exhaustive semantic-paraphrase guarantee.
- No direct source matches to Golden35, the 31 unique accepted Golden32 sources, or Bixby50; rejected Golden source identities were also checked where available. Golden32's repeated case is not an extra independent test source.
- Bixby50 was previously used for prompt development, as its experiment README acknowledges. Absence of SFT row leakage does not make it an untouched generalization benchmark.

See [leakage/provenance report](leakage_review.md), [machine counts](leakage_counts.json), and [repeated-source examples](duplicate_source_examples.csv).

## 6. Representation and serving compatibility

The full-corpus component inventory is below. Rows may contain several component types, so row percentages must not be summed.

| Component | Instances | Rows containing it |
|---|---:|---:|
{components}

{chart_note}

Table-domain metadata is a layout hint and does not recover missing original intent labels. Rows can contain multiple domains. The final column shows how prompt expansion could bias retention under a 4,096 limit (proxy tokenizer and stored targets).

`formula`, `playlist` and `restaurants` are explicitly supported renderer domains; their presence is not an unsupported-enum defect. Compiler size limits are checked separately from domain routing, and current Python validation does not substitute for on-device visual inspection.

| Table domain | Tables | Rows containing domain | Rows over 4,096 with deployed prompt |
|---|---:|---:|---:|
{domains}

**Repetition guard conflict:** train:6416 (`archive-train-109871-f518eb549644b798`) has a valid screenplay target with `l=Column([m,n,o,...,af,ag,ah],"sm")`. These are 22 distinct sequential children with actual Text definitions, not a model loop. A streaming callback exposing that list before the closing sentinel meets `GenUiOutputRepetitionGuard`'s 20-reference condition. If the entire completed document arrives in one callback, the closing-tag bypass prevents the stop. Full-text testing alone would therefore miss this risk. One such complete-list candidate was found across the corpus; this check does not claim all possible partial-token/prefix false stops are enumerated or that a device failure was observed. See [serving census](serving_distribution.json) and [pipeline review](pipeline_review.md).

Long repeated literals mostly include valid shared dates/tasks; their count is not a measured decoder-loop rate. Likewise, raw long digit strings can be IDs or phone numbers. The recent model's repeated zeros, changed train numbers and altered percentages cannot be attributed to a specific training row without additional evidence.

## 7. Relation to the recent Fold7 results

The [previous five-case comparison](../../../GenUICraft/validation/20260925_fold7_more_epochs_comparison/REPORT.md) records 0/5 raw strict passes and 0/5 mechanical source-integrity passes for both checkpoints, although generated-output repair rendered 5/5 each. Those are prior device measurements, not new measurements from this audit. This corpus contains valid complete targets, so malformed generated DSL is not explained by widespread malformed gold labels. The observed omissions and citation losses are consistent with the data's verified weaknesses, but the repeated IDs, corrupted numbers and syntax failures also require checking exact training-prompt lineage, dense-checkpoint behavior and export/runtime parity. No supplied training manifest independently verifies the newer model's epoch count or exact prepared dataset.

## 8. Recommended sequence before another long run

1. **Freeze the experiment contract:** identify the exact input bundle, prepared splits, tokenizer/chat template, prompt hash, source-code revision and export pipeline used for each compared checkpoint. Use the current manifest-required launcher.
2. **Fix target fidelity first:** add regressions for the confirmed lost fields/charts/entity bindings and malformed source tables. Regenerate through the corrected Stage 3 pipeline; keep uncertain pairs out of the candidate set. Source factual repairs must update the source first.
3. **Close the serving gap:** adopt the same reversible URL/citation normalization in training and SDK, and cover absent URLs, plain text, markdown and mixed Unicode/numeric formats. Do not invent URLs for numeric citation markers.
4. **Measure the real prepared data:** with the actual HF tokenizer, report accepted/quarantined counts by source family, domain, input length and target length after prompt normalization. Select a budget compatible with deployment; preserve complete closing tags and graphs.
5. **Balance and deduplicate intentionally:** audit same-source variants, weight source families fairly, reduce forced icon/action scaffolding, and add representative production-style development cases independent of Bixby50.
6. **Run a short controlled model evaluation:** compare raw strict validity, repaired validity, numeric/citation/entity fidelity, visual-role coverage and rendered output. Separately evaluate dense checkpoint, quantized export, and on-device generation with MTP off/on to locate regressions. Rendering success through repair/fallback is not model accuracy.
7. **Scale epochs only after those gates improve.** Keep an unseen final evaluation set, and compare checkpoints on the same prompts, tokenizer, decoding settings and device.

## Evidence and reproducibility

- [Structural/token summary](structure_summary.json); [compact per-row CSV](row_metrics.csv); [length/domain slices](length_slices.json).
- [Compressed complete per-row evidence](structure_rows.jsonl.gz); [final integrity and code fingerprints](audit_manifest.json).
- [Pipeline source review](pipeline_review.md); [semantic evidence](semantic_review.md); [split/provenance evidence](leakage_review.md).
- Reproduce structure: `python training/reports/v10_full_audit_20260926/audit_structure.py --workers 12`. Requires the local extracted tokenizer path named in that script plus repository Python dependencies. It writes report artifacts only; interrupted runs can use `--resume`.
- Reproduce serving census: `python training/reports/v10_full_audit_20260926/audit_serving.py`.
- Other audit scripts and their deterministic seeds/methods live beside this report. The existing archived manual100 is retained unchanged.
- Dataset train SHA-256: `{s['input_sha256']['train']}`.
- Dataset validation SHA-256: `{s['input_sha256']['val']}`.

The audit does not establish that v10 is the sole cause of the observed E2B failures. Exact preparation lineage and dense-versus-export inference remain necessary to separate dataset quality from optimization, quantization and serving behavior.
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf8')
    print('Wrote REPORT.md and row_metrics.csv; structural rows',n)

if __name__=='__main__':main()
