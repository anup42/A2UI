"""Package the completed, independently verified repaired dataset audit."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(r'D:\git\GenUI-LM')
WORK = ROOT/'working_dir/dataset_v11_20260930'
AUDIT = ROOT/'training/reports/dataset_v11_20260930'
REFERENCE = Path(r'D:\git\anup-code\A2UI')

def read(path):
    return json.loads(path.read_text(encoding='utf8'))

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

verification = read(AUDIT/'verification.json')
if verification['status'] != 'verified':
    raise ValueError('Independent verification must finish before packaging')
cleaning = read(AUDIT/'cleaning_summary.json')
stats = read(AUDIT/'statistics.json')
counts = cleaning['new_cleaning_counts']
base = cleaning['base_repairs']
new_val = stats['v11s']['val']['rows']

evidence = AUDIT/'evidence'
evidence.mkdir(exist_ok=True)
for name in ('download_provenance.json', 'repaired_pilot_selection.json',
             'repaired_pilot_summary.json', 'repaired_regression_tests.log',
             'reassessment_policy_evidence.json', 'v10_hold_repair_assessment.json',
             'retained_spot_review.json', 'repaired_retained_spot_review.json'):
    shutil.copyfile(WORK/name, evidence/name)
shutil.copyfile(WORK/'projected/archive_manifest.json', AUDIT/'archive_manifest.json')
shutil.copyfile(ROOT/'training/reports/dataset_v11_conservative_20260930/requirements-audit.txt', AUDIT/'requirements-audit.txt')

# Preserve the exact policy code and schemas, not only hashes to mutable checkouts.
fingerprints = {**cleaning['implementation_sha256'], **cleaning['reassessment_reference_implementation_sha256']}
for relative, expected in fingerprints.items():
    source = REFERENCE/relative
    if sha(source) != expected:
        raise ValueError('Reference changed after verification: '+relative)
    dest = AUDIT/'policy_snapshot/reference'/relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
for source_name, expected in cleaning['benchmark_sha256'].items():
    source = Path(source_name)
    if sha(source) != expected:
        raise ValueError('Reserved benchmark changed after verification')
    dest = AUDIT/'policy_snapshot/reference'/source.relative_to(REFERENCE)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
for pattern in ('*space_v11*.py', 'repair_v11_base.py', 'project_space_archive.py'):
    for source in (ROOT/'training/scripts').glob(pattern):
        dest = AUDIT/'policy_snapshot/importer'/source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
for source in (ROOT/'training/tests').glob('test_space_v11*.py'):
    dest = AUDIT/'policy_snapshot/tests'/source.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
for name in ('apply_live_snapshot.py', 'snapshot_live_genui.py', 'pilot_repaired_import.py', 'package_repaired_release.py'):
    shutil.copyfile(WORK/name, AUDIT/'policy_snapshot/importer'/name)

notes = f'''Dataset v11 / v11s — repaired release, 30 September 2026

The complete source inventory is 155,000 queries, 148,915 responses and 131,400
Stage 3 records (one malformed record). 17,516 response IDs lack a Stage 3 record.
Missing Stage 3 targets were not invented or counted as repaired training pairs.

NEW DATA
Retained: {stats['v11s']['combined']['rows']:,}; unchanged by the repair chain: {counts['KEEP']:,};
mechanically repaired: {counts['REPAIR']:,}; quarantined: {counts['QUARANTINE']:,}.
Of retained new rows, {counts.get('recovered_generator_review_rows', 0):,} were originally
generator-ineligible and passed the recorded reassessment. Original generator
flags remain in every row. This overlaps the repair count; it is not an
additional set of rows. Original generator rejection counts are diagnostic,
not measured defect rates. The deliberately stratified 354-row pilot is not a
random accuracy study. Full output counts come from the complete corpus.

REPAIR POLICY
Reuse of v10 reference rebinding/pruning, encoding and split-word recovery,
source-grounded text/table/state-path repairs, plus narrow uniquely anchored
source-literal prose restoration. Known reference identities cannot be swapped.
Each retained edit has exact before/after differences, source hashes and repair
proofs. Repeated repair must be stable. Every retained new row was independently
replayed from its original Stage 3 projection during verification.
An initial verifier comparison treated in-memory tuple evidence as different
from the same JSON arrays on disk. All ten repaired v10 rows already replayed
byte-for-byte. The comparison was fixed to use the persisted JSON representation;
tests still reject changed values and array order. No dataset bytes changed.

Generic review flags can clear when renderer-visible evidence preserves source
units and numeric anchors. Specific action/media/table/role/dynamic findings
require fresh v5.4 checks and remain held when unresolved. Native normal Card
titles count as visible; arbitrary invisible state and ignored Table titles do
not. This is a heuristic representation-fidelity policy: it does not prove all
added non-numeric target phrases, factual correctness, or final device rendering.

V10
Original v10 remains byte-identical at 92,977 rows. Its later manual findings
identified 34 rows in 31 families. {base['recovered_rows']} rows were repaired and recovered;
{base['remaining_held_rows']} rows in {base['remaining_held_families']} families remain excluded from v11.
Four recovered cases correct internal arithmetic or unit labels only:
case 2: speed improvement 100/9 percent (11.1 percent rounded);
case 51: revenue per lead, not conversion rate;
case 73: option B wins using the stated table totals;
case 100: longest duration 30.5 hours, with total 49.25 hours.
Other recovered cases restore source-bound table/field/attribution content or
remove unsupported latent action metadata. Exact proofs are in
known_v10_repairs.json. Medical/legal/product/source-verification ambiguities and
unsupported pie-chart semantics were not guessed. Unmodified v10 rows retain
their original bytes and all retained v10 rows retain their original split.

EVALUATION
v11s/val contains {new_val:,} new-only held-out rows and is byte-identical to the
new suffix of v11/val. v11s/train is already contained in v11/train; do not use
it as an unseen evaluation set after training on v11. New duplicate source/target
pairs and detected v10/Golden overlaps are excluded. Same/near sources, repeated
queries and scenario families stay in one split. The near-source search uses
the bounded lexical retrieval and thresholds inherited from v10, not exhaustive
semantic equivalence. Golden32, Golden35 and Bixby50 remain reserved.

TRAINING
These are model-independent A2UI Express v1 train.jsonl/val.jsonl pairs with
response_text, completion and messages. No samples were truncated. Tokenizer
preparation must run for the chosen model and context length. Generation token
telemetry is not a measurement of training tokens. Model training, performance
evaluation, device rendering and independent source fact checking were not run.

PREVIOUS CANDIDATE
The earlier strict generator-eligibility build retained 13,372 new rows. It is
preserved in v11_conservative/v11s_conservative with its separate audit. Those
files are historical comparison artifacts. v11 and v11s are this repaired release.

See REPORT.html and statistics.json for complete train/val/combined counts,
length percentiles, components, origins, intents, difficulty, modality, reference
kinds, source review status and available teacher telemetry. Reason counts and
component presence counts can overlap. verification.json records scope/results.
The historically_repaired flag means any recorded repair, including this release;
repaired_in_v11 identifies the repairs newly applied for v11 specifically.
'''
(AUDIT/'REVIEW_NOTES.txt').write_text(notes, encoding='utf8')

old = (ROOT/'training/reports/dataset_v11_conservative_20260930/REPRODUCE.txt').read_text(encoding='utf8')
old = old.replace('4. Prepare and merge:', '''4. Apply hash-bound repairs to the held v10 families:
   training/scripts/repair_v11_base.py
   --policy-repo D:\\git\\anup-code\\A2UI
   --prior-work-dir D:\\git\\GenUI-LM\\working_dir\\dataset_v11_20260930\\base_review
   --work-dir D:\\git\\GenUI-LM\\working_dir\\dataset_v11_20260930\\base_repaired

5. Prepare, repair, reassess and merge:''')
old = old.replace('training/scripts/prepare_space_v11.py new', 'training/scripts/prepare_space_v11_repaired.py new')
old = old.replace('5. Independently verify', '6. Independently verify')
old = old.replace('training/scripts/verify_space_v11.py', 'training/scripts/verify_space_v11_repaired.py')
# Preserve the original base review command; change only downstream work paths.
head, tail = old.split('5. Prepare, repair, reassess and merge:', 1)
tail = tail.replace('dataset_v11_20260930\\base_review', 'dataset_v11_20260930\\base_repaired')
old = head+'5. Prepare, repair, reassess and merge:'+tail
head, tail = old.split('5. Prepare, repair, reassess and merge:', 1)
tail = tail.replace('--workers 8', '--workers 16')
old = head+'5. Prepare, repair, reassess and merge:'+tail
old = old.replace('6. Regression checks (11 tests):', '7. Regression checks (see evidence/repaired_regression_tests.log):')
old = old.replace('-p test_space_v11_import.py', '-p test_space_v11*.py')
old = old.replace('(267 rows)', f'({new_val:,} rows)')
old = old.replace('See REVIEW_NOTES.txt for conservative eligibility selection, reviewer limitations,', 'See REVIEW_NOTES.txt for source-proven repairs, reassessment and reviewer limits,')
old += '\nExact reference policies, schemas, YAML configs, importer scripts and tests are preserved in policy_snapshot. The original reference checkout is read-only. Final output hashes are in each dataset manifest.\n'
(AUDIT/'REPRODUCE.txt').write_text(old, encoding='utf8')

summary = ['Verified dataset comparison', '', 'Dataset | Train | Validation | Total']
for name in ('v10','v11','v11s'):
    summary.append(f"{name} | {stats[name]['train']['rows']:,} | {stats[name]['val']['rows']:,} | {stats[name]['combined']['rows']:,}")
(AUDIT/'SUMMARY.txt').write_text('\n'.join(summary)+'\n', encoding='utf8')
for name in ('v11','v11s'):
    info = f'''{name}: independently verified model-independent training dataset.
Train: {stats[name]['train']['rows']:,}; validation: {stats[name]['val']['rows']:,}.
Full audit: {AUDIT}
Input format: A2UI Express v1, consistent with v10; source response_text and target completion.
Use run.dataset_dir in GenUI-LM or --input-dir with the A2UI training launcher.
Allow model-specific tokenizer preparation to run; do not use as --prepared-input-dir.
v11s/val is the new-only holdout in v11/val. v11s/train is part of v11/train.
Source factual and device rendering checks have not been certified by this build.
'''
    (ROOT/'training/data/train'/name/'README.txt').write_text(info, encoding='utf8')
manifest = {str(p.relative_to(AUDIT)).replace('\\','/'):sha(p) for p in sorted(AUDIT.rglob('*')) if p.is_file() and p.name not in ('candidates.jsonl','audit_files_sha256.json')}
(AUDIT/'audit_files_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
print('\n'.join(summary))
