# Single-Codex GenUI Judge Benchmark v2

## Scope and authority

This benchmark uses the interactive Codex runtime as the sole visual judge. It
does not call Azure, Vertex, OpenAI API, or any other judge service. Its labels
must be published as `provisional_single_codex_groundtruth`; they are neither
human ground truth nor human calibration.

The source response is reference truth. The judge evaluates whether the native
Android UI visibly represents that response. It must not grade the response's
factual accuracy or writing quality.

## Immutable benchmark layout

The benchmark root is named
`llm_groundtruth_native_960_<UTC timestamp>_v2`. Important artifacts are:

- `selection_manifest.json[l]`: exact deterministic sample membership,
  population/stress stratum, source-grouped split, and migration anchors.
- `selected_genui.jsonl`: an immutable 960-row copy selected from the source.
- `expected_contracts.jsonl`: deterministic v5.4 contracts and fingerprints.
- `judge_schedule.jsonl`: 960 originals plus 96 blinded repeats.
- `sealed/packet_identity_map.jsonl`: private packet-to-source mapping.
- `native_provenance.json`: APK/install, renderer, schema, and device identity.
- `native_capture/batches/`: immutable capture batches of at most 50.
- `native_capture_manifest.jsonl`: screenshot hashes and failure classes.
- `packets/`: blinded screenshot-only and source-conditioned packet surfaces.
- `raw_judgments.jsonl`: durable pass-level Codex output.
- `judgments_by_packet.jsonl`: exact host-computed R, U, and J by packet.
- `finalized_groundtruth.jsonl`: exactly 960 unique finalized samples.
- `analysis/`: distributions, reliability, calibration, plots, and report.

The source `genui.jsonl` is read-only and its SHA-256 is verified before and
after selection and scoring.

## Selection

First create a compact current-v5.4 index for stress sampling. This index has no
native evidence and is not the final v5.4 audit:

```powershell
python dataset/scripts/run_single_codex_judge_v2.py `
  precompute-selection-v5-4 `
  --source-dir <source-run> `
  --output-dir <immutable-selection-index> `
  --workers 8
```

Then select exactly 30 rows for each of 32 intents: 20 deterministic
population-random and 10 stress rows. Stress selection includes one prior-v1
anchor per intent and greedily covers score bands, caps, long sources, wide
tables, charts, formulas, Tabs, Modal, media, and complex layouts.

```powershell
python dataset/scripts/run_single_codex_judge_v2.py select `
  --source-dir <source-run> `
  --output-dir <benchmark-root> `
  --metric-scores <immutable-selection-index> `
  --prior-v1 <prior-v1-root>
```

Per intent, the split is 20 calibration, 5 validation, and 5 untouched
holdout. `query_id` is the source-grouping key. The selector rejects any
cross-split source leakage.

## Native capture safety and parity

Build and install only the checkout-matched `JudgeCapture` variant using the
Android build/install skill:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  C:\Users\anupk\.codex\skills\android-build-install\scripts\build_install_android.ps1 `
  -Root <repo>\android -Module :app -Variant JudgeCapture -Serial <serial>
```

The variant application ID is `com.samsung.genuicraft.judgecapture`; every
embedded API/service credential is overridden to an empty string. Do not
uninstall, clear, downgrade, launch, or instrument
`com.samsung.genuicraft`.

Capture with:

```powershell
python dataset/scripts/capture_single_codex_judge_native.py `
  --benchmark-dir <benchmark-root> `
  --serial <serial> `
  --apk-path <repo>\android\app\build\outputs\apk\judgeCapture\app-judgeCapture.apk
```

The host verifies that the installed base APK SHA-256 equals the checkout
build. It records commit, dirty-source status, renderer-source inventory,
schemas, device, density, font scale, locale, theme, viewport, and screenshot
hashes. It captures the initial viewport at compact, 700dp, and 900dp widths,
the compact full-height page, and up to four bounded alternate renderer
states. Display overrides are restored in `finally`.

Each infrastructure failure is retried twice. Three failed attempts stop the
run. A screenshot-confirmed renderer failure is candidate evidence and all ten
judge dimensions must be zero; it is never converted into an infrastructure
failure.

## Two-pass judging

Build opaque packets only after native evidence is complete:

```powershell
python dataset/scripts/run_single_codex_judge_v2.py build-packets `
  --benchmark-dir <benchmark-root>
```

The host task has already seen source-run and metric context, so it must not
act as the judge. Initialize a neutral workspace whose path contains no
generator or model name, then use fresh Codex tasks for judging:

```powershell
python dataset/scripts/run_single_codex_judge_v2.py seal-implementation `
  --benchmark-dir <benchmark-root>
python dataset/scripts/run_single_codex_judge_v2.py init-blind-workspace `
  --benchmark-dir <benchmark-root> `
  --workspace-dir <neutral-opaque-workspace>
```

Every batch preparation/import rechecks this implementation fingerprint and
stops if judging code, schemas, packet inputs, or native evidence changed.

For each block, release only its screenshot surface:

```powershell
python dataset/scripts/run_single_codex_judge_v2.py prepare-blind-batch `
  --benchmark-dir <benchmark-root> --start 0 --count 20 `
  --pass-type screenshot_only
```

The fresh judge task reloads the rubric in that capsule, opens only the listed
screenshot packets, scores dimensions 3–10, and writes ordered JSONL to the
capsule's `result_path`. Import it immediately:

```powershell
python dataset/scripts/run_single_codex_judge_v2.py import-blind-batch `
  --benchmark-dir <benchmark-root> --start 0 --count 20 `
  --pass-type screenshot_only --expected-task-id <codex-task-id>
```

Only a successful host import unseals the matching source-conditioned packets:

```powershell
python dataset/scripts/run_single_codex_judge_v2.py prepare-blind-batch `
  --benchmark-dir <benchmark-root> --start 0 --count 20 `
  --pass-type source_conditioned
```

The same fresh judge task then scores dimensions 1–2 and saves the second
JSONL result. Import it with the same command and
`--pass-type source_conditioned`.

For an unattended block, the host may run
`watch_single_codex_judge_block.py` after the Codex task ID is known. It waits
for a valid screenshot result, imports it with that exact task ID, releases
the source packet, and imports the source result. The watcher never releases
pass two early and writes logs only on the host side.

Scores are 0–100 in increments of 5. Common anchors are 0 broken/absent,
25 major failure, 50 materially partial, 75 good with noticeable defects, and
100 no material defect. Confidence is recorded but never changes a score.

The per-batch capsule never exposes generator identity, the source-run path,
FlatSpec JSON, filenames with model names, legacy/v5.4 scores, or earlier
judgments. Source-conditioned packet files do not exist in the capsule until
all screenshot-only rows in that block are durably recorded.

Write one JSON object per line in exact packet order. The host stores the exact
task ID, available model identifier, timestamp, pass type, and packet position
outside the blinded packet. After import, the score-bearing block file moves
to the sealed host archive and is removed from the neutral capsule so later
tasks cannot inspect previous judgments. After the source pass is imported,
the capsule also removes its packet JSON and screenshot hard links; later
tasks cannot file-match a blind repeat against an earlier capsule.

The host computes:

\[
R=(0.18S_1+0.14S_2)/0.32
\]

\[
U=(0.14S_3+0.12S_4+0.10S_5+0.10S_6+
0.08S_7+0.04S_8+0.05S_9+0.05S_{10})/0.68
\]

\[
J=\sum_{i=1}^{10}w_iS_i
\]

where weights are exactly
`[0.18, 0.14, 0.14, 0.12, 0.10, 0.10, 0.08, 0.04, 0.05, 0.05]`.

Complete the 32-packet one-per-intent pilot and freeze the rubric before any
post-pilot judgment:

```powershell
python dataset/scripts/run_single_codex_judge_v2.py freeze-protocol `
  --benchmark-dir <benchmark-root>
```

Every 80 completed scheduled packets, write a handoff and start a fresh Codex
task:

```powershell
python dataset/scripts/run_single_codex_judge_v2.py write-handoff `
  --benchmark-dir <benchmark-root> --milestone 80
```

## Repeats and adjudication

The schedule contains 32 repeated intent anchors and 64 deterministic random
repeats, each at least 200 positions after its original. Codex is not told
which packets repeat.

After repeats are complete:

```powershell
python dataset/scripts/run_single_codex_judge_v2.py repeat-analysis `
  --benchmark-dir <benchmark-root>
```

A fresh blinded adjudication packet is required when composite absolute
difference is at least 10, a dimension differs by at least 20, either judgment
confidence is below 0.75, or composites cross a 25-point anchor band. Judge
these packets in a fresh Codex task through
`prepare-blind-adjudication-batch` and
`import-blind-adjudication-batch`, using the same screenshot-first/source-
second release sequence. The supplemental capsule contains only fresh opaque
packet IDs and no repeat metadata. Final adjudicated dimensions are the per-
dimension median of the original, repeat, and third judgment. Otherwise, the
original remains final; repeats measure reliability rather than silently
averaging away drift.

If the available Codex model identifier changes, stop and create a new judge
protocol version. Do not merge model versions.

## Fresh metric audit and analysis

Once captures are final, recompute v5.4 with native evidence:

```powershell
python dataset/scripts/run_single_codex_judge_v2.py rescore-v5-4 `
  --benchmark-dir <benchmark-root>
```

Finalize and analyze:

```powershell
python dataset/scripts/run_single_codex_judge_v2.py finalize `
  --benchmark-dir <benchmark-root>
python dataset/scripts/run_single_codex_judge_v2.py analyze `
  --benchmark-dir <benchmark-root>
```

Analysis reports population/stress/combined and macro/micro intent
distributions; repeat ICC(A,1), Pearson, Spearman, MAE, signed bias, and
Bland–Altman limits; milestone drift; legacy/v5.4 correlations against R, U,
and J; Kendall, RMSE, paired-bootstrap intervals, calibration curves, and
score-band examples.

Identity, affine, and isotonic mappings are fit on 640 calibration rows.
Validation has 160 rows. Isotonic is selected only when it improves validation
MAE by at least 0.5 points over affine. The selected mapping is evaluated once
on the 160-row holdout.

Do not optimize metric weights from these labels. If repeat ICC is below 0.85,
repeat MAE exceeds 5, absolute bias is at least 2, or adjusted milestone drift
exceeds 3, do not publish the label set. Refresh the frozen rubric and rescore
the identified milestone plus repeat anchors under a new protocol version.
