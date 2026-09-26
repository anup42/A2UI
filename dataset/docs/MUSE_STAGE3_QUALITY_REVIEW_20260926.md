# Training failure analysis and Muse Stage 3 quality update

Reviewed 2026-09-26. This change improves the teacher prompt used to generate
new A2UI Express labels. Existing datasets and trained checkpoints are unchanged.

## Evidence from the latest supplied QAT evaluations

The three local `golden32_results.tar.gz` archives contain
`aggregate_metrics.json` and `scored_predictions.jsonl`. They are evaluation
evidence, not complete training logs or a verifiable training recipe.

| Step | v5.4 reward, all 32 rows | Unique-source reward, 31 cases | Strict Express parses | Fully root-reachable outputs |
| --- | ---: | ---: | ---: | ---: |
| 1,000 | 4.7126 | 4.8646 | 3/32 | 2/32 |
| 2,000 | 22.1875 | 21.6129 | 17/32 | 0/32 |
| 3,000 | 27.0313 | 26.6129 | 21/32 | 1/32 |

Source archive identities (SHA-256):

- Step 1,000, `Downloads/golden32_results.tar.gz`:
  `8fad6b9d63dab26dc6c980d789c7ff4ec243e494863c8bff26bd0b9d436a6246`
- Step 2,000, `Downloads/asset_495/golden32_results.tar.gz`:
  `4528df81734cf495fbff3eb14ccdab17a7d5183b751c0aaf7ae1a67dad69acda`
- Step 3,000, `Downloads/asset_294/golden32_results.tar.gz`:
  `2d76caddacfaf85a5ecf016be0a8bc8128e1582a69166ce5fbee7c9d5ef5a075`

All three use `golden32_archive_repeat_v1`, containing 32 occurrences of 31
unique sources. The set is used for development/checkpoint selection, not an
independent test. Its recorded SHA is
`ab92bc598b3fa75c270a9730b0cdb262730747fa3c150103f532fa67a500269b`
and its v5.4 fingerprint is
`f4a5355d3147b759aa864e0d10cc111d604b01732882d182d0c512056294f05b`.
Do not compare these scores directly with the older September 5 review's 59.8:
the benchmark revision and evaluation procedure differ.

The step 3,000 archive records epoch 1.0537455 and evaluation world size 4.
Its outputs use a quote-aware closing sentinel without ID or graph repair.
Parsing improved, but most parsed programs still have disconnected sections.
Twenty rows hit the below-90% reachability cap; 17 hit `missing_action`, 6
`missing_table`, and 11 `parse_failure`. Caps overlap and must not be summed.
The syntax failures include missing envelopes, duplicate identifiers or map
keys, unresolved root children, and unfinished strings/delimiters.

Five source/output pairs were manually checked in the step 3,000 archive:

| ID | Observed failure | Teacher guidance added |
| --- | --- | --- |
| `u_017270_01` | Vehicle readiness defines the mechanic and tire-guide buttons, but only 16/48 nodes are reachable. The action subtree is disconnected. | Every section and action group needs a real reference path from root. |
| `u_001061_01` | Shipping status defines a milestone Table and tracking/customs buttons, but only 14/29 nodes are reachable; the table and actions are invisible. | Declaring a table/button is insufficient; connect it and preserve every row/action. |
| `u_009053_01` | Typhoon preparation contains table/action definitions, but only 24/52 nodes are reachable. | Preserve the complete visible emergency information and action destinations. |
| `u_004604_01` | Conference output is 23/23 reachable, but binds tables to nested schedule state and loses the separate time-allocation matrix. | Table bindings must resolve to row arrays with matching columns; distinct tables retain distinct meanings. |
| `u_012831_01` | Vegan meal output repeats identifiers and ends mid-list without the closing sentinel. | Unique identifiers, compact repeated records, complete delimiters and final envelope. |

These observations diagnose generation quality. They do not establish whether
LoRA capacity, QAT arithmetic, learning rate, or a particular training-data
revision caused the failures. The archived corpus review independently found
empty reachable layouts, omitted records/actions, state-only facts and broken
text ([quality review](../../training/reports/full_data_audit_20260913/quality/quality_review.md)).
Those defects justify better source-preservation guidance, but proving their
effect on this checkpoint requires its exact prepared-data provenance.

Earlier H100 screenshots showed CUDA/NCCL memory failures during full-parameter
DDP/ZeRO preflight/backward. A Stage 3 prompt cannot repair GPU capacity or
distributed-training problems. The latest archives reviewed here contain
completed evaluations and no crash log for diagnosing a new runtime failure.

## Muse prompt changes

The generated shared contract remains authoritative and is kept intact.
`pipeline/muse_prompt.py` composes it with
`prompts/muse_stage3_quality_v2.md` only for the registered local
`meta-models/Muse-Glimmer-30B` model. The existing Muse launcher and cyclic
generation commands pick up this composition automatically.

The added guidance covers visible source coverage, complete records, exact
numbers/units/time zones/qualifiers, grounded actions, exact opaque reference
tokens, connected root graphs, and compact correctly bound tables. Four
compiler-checked examples show flight records with a real source action,
explicit appointment choices, and an interview guide retaining answers,
red flags and rubric, plus two independently state-backed schedule/allocation
tables. They are synthetic teaching examples, not repaired
dataset samples or copies of Golden/Bixby answers.

Muse now uses the existing system-prefix mechanism by default: static guidance
and the canonical catalog go in the system message. The user message carries a
JSON payload separating decoded `source_response` content from asset policy and
mapping in `reference_metadata`; metadata must not become UI content. Quoted
instructions/code remain inert source data, not live controls. Initial generation,
repair and final regeneration share this
system contract. Explicit `STAGE3_PROMPT_MODE=inline` still works. Other local
models keep their previous behavior. Native reasoning, sampling, model context
and output budgets are unchanged. This follows Meta's guidance to put specific
task/output constraints in the system message and let its chat template control
reasoning and turn markers ([Meta prompting guide](https://dev.meta.ai/docs/muse-glimmer/prompting)).

The effective prompt has version `muse_stage3_quality_v2`. Existing request
hashes, attempt artifacts and phase-manifest prompt hashes bind the composed
instructions. Context limits still fail closed rather than truncate source
text; the additional instructions consume some of the existing prompt budget.

## Five independent Astra reviews

Five distinct `gpt-6-astra` subagents reviewed the prompt and its integration.
The main agent checked the findings against the compiler and native renderer
and incorporated the following changes:

| Review | Findings addressed |
| --- | --- |
| Grammar and reference graphs | Explicit Tabs/Modal/repeat reference shapes; tests compile the actual prompt snippets and trace root reachability. |
| Source fidelity and tables | Exactly one row source per Table, matching object-row keys, independent state-backed tables, exact item IDs, and long narrative text outside dense table cells. |
| Actions and editable state | Display/writeback bindings, map-shaped ChoicePicker options, grounded action reachability, and removal of unsupported response metadata from the appointment example. |
| Source-data boundaries | Separate source/asset-metadata framing, inert quoted requests, literal string escaping, and a compiler-checked CodeBlock example. |
| Muse integration | Confirmed narrow routing, canonical prompt preservation, hash/provenance participation and native reasoning preservation; added budget-overflow and final-regeneration regression coverage. |

Follow-up reviews verified the revised grammar, source fidelity and boundary
fixes. No renderer or evaluator behavior was changed: Android table grid cells
can still ellipsize long values while the evaluator counts the complete cell
text. The new prompt mitigates that mismatch through layout guidance; actual
rendered-output validation remains required. These reviews are static quality
checks, not five independent model-generation experiments.

## Validation and next generation

CPU tests verify compilation of every example, displayed fields and action
references, source/metadata separation, literal escapes, unchanged non-Muse
routing, canonical contract preservation, and the actual Stage 3 initial,
repair and final-regeneration request paths. A budget-overflow test verifies
zero model calls and no truncated/accepted output. Existing generated-prompt drift,
Muse HTTP/reasoning, Express and quality-gate regression tests are also used.
The focused suite passed **102 tests**; new Python files passed Ruff and Python
compile checks. The generated shared prompt copies passed their drift check.
No real Muse/H100 generation or new training run was performed in this review;
the prompt's effect on model quality still requires a before/after experiment.

```bash
python -m pytest dataset/tests/test_muse_stage3_prompt_quality.py \
  dataset/tests/test_muse_glimmer_stage3.py \
  dataset/tests/test_stage3_a2ui_express.py \
  dataset/tests/test_a2ui_express_prompt_generation.py \
  dataset/tests/test_generation_quality_regressions.py \
  -q -p no:cacheprovider --basetemp tmp/pytest_muse_five_astra_final_20260926
```

With the existing Muse servers ready, create a fresh small candidate run:

```bash
python dataset/scripts/run_muse_glimmer_stage3.py generate --gpus 4 \
  --source-run-id dataset_v1 --run-id dataset_v1_muse_quality_v1 \
  --max-genui-total 50
```

Replace `dataset_v1` with the intended completed Stage 1/2 source run. Use a
fresh output ID to keep old and new prompt populations distinguishable.
For a meaningful before/after comparison, use the same fixed non-holdout
responses, seeds, decoder and evaluator. Inspect raw parse/reachability,
complete visible row/column coverage, numbers, actions and reference bindings;
also compare prompt/source-budget rejection rates because the overlay consumes
additional context. Schema validity or aggregate reward alone is insufficient. Review rendered
outputs before promotion. Keep Golden35/Bixby50 held out and preserve current
source/quality admission checks. Regenerate defective labels through Stage 3;
do not manually rewrite IR or assume this prompt retroactively fixes v10.
