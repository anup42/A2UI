# Fast evaluation patch integration review

The supplied `fast-eval-b0516a6.patch` was reviewed and integrated over
`a71aee07854d90169de613c0c9eae680b0136b19` (native chart support), preserving
the newer chart contract. All 19 modified-file preimages matched the stated
base, `9cc1cfef0fb72f80fad2e78c4b1054d0b6b0176e`. The patch applied without
conflicts; its 46 paths are confined to `training/`.

- Original patch commit: `b0516a6ab87baa78c4159528b22b3407a1aca044`.
- Patch SHA-256: `4f8e16cce27ce13bb2efaccab5d5f3ba13c4a0ef51d05af1bd2a124107002dcd`.
- Retained features: scoped effective-weight QAT caching, precise BF16 SRQ
  compilation, worker compiler-cache isolation, host/native latency reporting,
  persistent LiteRT GPU evaluation, and separate Android post-repair scores.
- Raw Golden32 source-macro scores remain the checkpoint selector. Rejected
  repairs stay in the evaluation denominator; source text/reference targets
  are never passed into repair.

## Corrections made during review

| Finding | Resolution and regression coverage |
|---|---|
| Kotlin `println` used the host stdout encoding; the real Windows bridge failed UTF-8 decoding on non-ASCII output. | Write bridge responses through an explicit UTF-8 writer. Real JVM tests preserve Hindi, Korean, Japanese, and emoji. |
| Graphics-process evidence was queried only when no compute allocation existed. An allowed CUDA context could hide a Vulkan context on another GPU. | Inspect both inventories, merge evidence per device, and reject any observed unallocated GPU. Tests cover mixed-device rejection and same-device deduplication. |
| A tensor-valued LoRA scale created in inference mode has no mutation version counter. The cache accessed that counter outside its bypass guard. | Bypass and clear cached weights when parameters or scales lack version counters; verify exact outputs across scale mutation. |
| The repair source fingerprint omitted the trained converter that supplies preprocessing and repair policy. | Include `GenUiTrainedConverter.kt` and test invalidation when its policy changes. |
| Malformed bridge timing, row indices, empty accepted output, and rejection kinds were insufficiently checked. | Reject boolean/nonfinite/negative timings, non-integer indices, empty accepted output, and invalid rejection evidence before aggregation. |
| Existing documentation punctuation was corrupted by an encoding conversion. | Restore the existing document text before appending the new section. Use explicit UTF-8 for benchmark input reads and make the repair tests independently importable. |

The JVM bridge also accepts a valid radar chart without repair using the current
SDK catalog. This verifies compiler integration, not a physical-device render.

## Supplied benchmark evidence

The other files in this directory are historical evidence supplied with the
patch. GPU timings and quality scores were **not rerun** during this Windows
integration. They predate the chart-support commit and subsequent fixes above.
Their recorded source hashes describe the original measured code, not this
integrated checkout. Do not present them as freshly validated performance or
quality of the current branch.

The supplied JSON, CSV, and log contents are retained unchanged. Trailing table
padding was removed from text reports; benchmark values were not changed.
The supplied HF parity report covers 32 occurrences / 31 unique sources.
Its HF Run E checkpoint and LiteRT Run D export are different models, so the
cross-runtime scores do not establish model parity. Historical launch PIDs and
pipeline-restoration entries do not establish current remote job status.

## Local validation

- Initial patch regression suite: 289 passed, 3 skipped, including real tiny
  CPU Gemma worker subprocesses.
- Corrected cache, LiteRT process checks, and real Android repair bridge suite:
  101 passed. The bridge was built from current Kotlin sources with locally
  cached compiler dependencies; no APK or model artifact was built.
- Additional QAT, callback, stopping, scoring, deployment, chart, and v11
  regressions: 255 passed, 3 skipped; one TensorBoard integration test initially
  required the missing local TensorBoard dependency.
- TensorBoard verification uses a temporary environment inheriting existing
  CPU PyTorch, with the declared `tensorboard>=2.17` dependency installed only
  there (resolved TensorBoard 2.21.0). Follow-up TensorBoard, official pipeline,
  and result-table regressions: **168 passed, 1 skipped**, resolving the missing
  dependency failure above.
- Python syntax compilation, critical Ruff checks, and Bash syntax checks pass.
- Staged whitespace checks pass. Exactly the 46 supplied patch files and this
  review are included; all eight pre-existing Android edits remain byte-identical
  and outside the commit. No release data, model weights, or caches are changed.

Test suites overlap; their counts must not be added as unique test counts.
This host has CPU PyTorch 2.13.0 and no CUDA runtime. Production-model inference,
GPU performance/parity, training, and physical-device validation were not run.
