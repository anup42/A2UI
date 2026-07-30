# Blinded Single-Codex judge task procedure v2

This task is a blinded judge, not the benchmark host. Work only from the
neutral batch manifest supplied in the task message and files directly listed
by that manifest. Do not inspect the repository, host benchmark, other batch
directories, sealed files, filesystem links, or conversation history for
sample identity or earlier scores.

Stop without scoring and report a protocol violation if any supplied path,
filename, or packet reveals generator identity, a source-run name, FlatSpec
JSON, legacy/v5 scores, or a previous judgment.

For every block:

1. Read the on-disk judge instructions and rubric named by the screenshot-only
   manifest.
2. Obtain the exact current Codex task ID and available model identifier from
   runtime request metadata.
3. Inspect every listed screenshot-only packet and all of its native images.
   Use the overview for orientation and original PNGs when text, clipping,
   state, or viewport behavior is ambiguous.
4. Score only dimensions 3–10 and write exact ordered JSONL to `result_path`.
   Use `apply_patch`, then parse and count-check the file. Only after it passes,
   write `{"ready":true}` to the manifest's `ready_path`. Do not open source
   content before this marker is complete.
5. Wait for the host bridge to create `source_conditioned_manifest.json`.
   The host creates it only after the screenshot pass validates and is durably
   archived.
6. Reload the same frozen rubric, inspect each newly released source packet,
   expected contract, and the same screenshots, then score only dimensions
   1–2. Write, parse, and count-check the second ordered JSONL result, then
   write its `{"ready":true}` marker.
7. Stop after both files are saved. Do not calculate R, U, J, correlations, or
   metric comparisons.

Every score is a multiple of 5 in [0,100]. Confidence is in [0,1] and never
changes a score. Include specific evidence, specific visible defects,
genuinely unavailable items only, and a concise rationale for each packet.
Use UTC ISO-8601 timestamps; the screenshot timestamp must precede the source
timestamp. A same-APK candidate render failure receives zero on every
dimension.
