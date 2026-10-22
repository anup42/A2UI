# GRPO list-to-card field extraction

## Correctness fix

The v5.4 content assignment previously compared one source bullet against
each separate output field. Faithful field extraction could therefore lower
content-unit fidelity and output precision. Visible `Card.title` and
`Card.subtitle` text was also omitted from the evidence collector.

The v5.4 scorer now collects those renderer-visible headers and tracks each
runtime Card occurrence, including repeated templates. For content assignment
only, adjacent Text/header fields in the same Card are treated as one unit
when their **ordered tokens exactly reproduce a complete source unit**.
Nested and sibling Cards cannot pool their fields. Source multiplicity still
uses the existing certified one-to-one assignment.

This is conservative certification, not a new semantic parser: field
reordering, paraphrases, or removed labels do not qualify for this grouping
and continue through the existing scorer. No arbitrary Card-count bonus or
unconditional precision exemption was added. Wrapping intact prose in a Card
does not receive an extraction bonus.

The original ungrouped evidence remains authoritative for visible-word
multisets, exact numbers/dates/units, duplication, references, tables, media,
and other semantic roles. Hidden/unreachable content and unused state do not
earn credit. Missing, invented, duplicated, or cross-entity swapped fields
do not receive exact grouping credit. Dynamic evidence and matching budgets,
strict validity, admission thresholds, and QAT reward weights are unchanged.
The legacy v5.3 scoring path does not enable the new grouping.

## Prompt and compatibility

The canonical Express quality policy now distinguishes entity records from
plain steps/checklists. It includes a source-grounded example extracting two
hotel bullets into separate cards and readable fields, while retaining compact
Table/native-card representations where appropriate. Its four generated
dataset/Android prompt copies remain synchronized. Existing chat templates,
training algorithms, optimizer settings, model/export logic, and immutable
prepared prompt snapshots are unchanged.

Fresh preparation picks up the updated prompt and its hash. Existing prepared
datasets/resumes retain their recorded prompt; do not overwrite their snapshots
to retrofit the new instructions. Prepare a fresh dataset/run to train with the
new prompt. Normal token-budget checks still apply to the longer prompt.

Evidence/scoring policy versions and the v5.4 source fingerprint include this
fix. Re-evaluate both the SFT baseline and candidate checkpoints with the same
new scorer before comparing scores; do not compare old and new scalar scores
as if they used the same metric implementation. Already-running processes and
checkpoints are not retroactively updated, and updating code alone does not
retrain an existing model.

The QAT extraction preference reserves 2% of quality for every candidate whose
source contains eligible entity records. Only strictly valid, fully faithful
field extraction earns that reserve back. A candidate with a fidelity defect
cannot escape the reserve and outrank an otherwise equivalent faithful response
merely by omitting a record. Existing quality caps and fidelity checks remain
authoritative; perfect faithful extraction scores are unchanged. This reward
implementation change is bound by the recorded code fingerprint. Rebuild
preferences or start a fresh run to use it; do not rewrite old run evidence.

## Regression coverage

Synthetic CPU tests cover separate Card fields and titles, repeated/shared
render occurrences, nested Cards, exact values and entity associations,
omissions/additions/duplication, hidden and unreachable evidence, strict raw
validity, ordinary Lists, prompt synchronization, and legacy scoring behavior.
These tests establish scorer/prompt correctness, not measured GPU training
quality or Android inference accuracy; a fresh training/evaluation run remains
necessary to verify the learned behavior.

Validated on 2026-10-21: **456 tests passed**, covering the v5.3/v5.4 metric
suites, QAT GRPO reward/pipeline/training/export contracts, GRPO auditing and
experiment controls, and shared/frozen prompt contracts. The focused Card
evidence and reward suites contain 14 and 23 tests respectively. Compilation,
scoped Ruff checks, generated-prompt synchronization, and `git diff --check`
also passed. No GPU training or device inference was run.
