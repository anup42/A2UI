# Progressive native rendering validation — 2026-10-08

10/10 measured conversions completed; 5/5 streaming cases reached an early Compose frame boundary.

SDK 0.6.0: **478 unit tests passed**; **3 deterministic Fold7 device tests passed**. The device cases cover early native preview, recreation/final handoff, failure/cancellation clearing, and streaming off. [Original full SDK build/test log](sdk_build_tests.txt) preserves the full-suite run; the test-count aggregate was captured before targeted replays rewrote generated XML.

Pipeline bridge: **9 unit tests passed** ([log](pipeline_bridge_tests.txt)). Connected progressive-preview checks exercise the SDK demo; this report does not claim a separate Pipeline/IR early-preview device test.

Bixby source checks: **14 JavaScript / 18 host / 5 manager tests passed**. The [22-entry changed-file handoff](../../artifacts/Bixby_GenUICraft_0.6.0_Streaming_ChangedFiles.zip) is source/package evidence; a Bixby application build/device run is not claimed.

Saved Bixby50 replay: **32/50 early previews**, **14 accepted only by terminal generated-DSL repair**, **4 terminal rejects**. Replay measures character readiness rather than native latency.

## Native comparison

Artifact status: **complete**, 5/5 paired cases.

| Case | Raw equal | Final equal | First frame boundary on (s) | Preview updates | Session wall on / off (s) | Wall change | Converter on / off (s) | Decode on / off (tok/s) | Init on / off (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| BXP-001 · Weather | yes | yes | 10.077 | 5 | 15.054 / 16.280 | -7.5% | 14.023 / 15.373 | 42.02 / 37.51 | 4.962 / 5.247 |
| BXP-003 · Rail travel | yes | yes | 15.298 | 6 | 24.195 / 24.188 | +0.0% | 23.194 / 23.140 | 34.56 / 34.27 | 5.053 / 5.017 |
| BXP-004 · Airline baggage | yes | yes | 10.950 | 13 | 21.478 / 21.144 | +1.6% | 20.532 / 20.105 | 36.17 / 36.34 | 5.633 / 5.315 |
| BXP-008 · Restaurants | yes | yes | 17.012 | 5 | 24.061 / 23.869 | +0.8% | 23.013 / 22.801 | 32.39 / 34.23 | 4.973 / 5.422 |
| BXP-011 · Smartphones | yes | yes | 19.196 | 7 | 27.099 / 28.407 | -4.6% | 26.040 / 27.300 | 32.69 / 29.69 | 6.349 / 6.367 |

Raw bytes match in **5/5 pairs**; final Express bytes match in **5/5 pairs**. Matching hashes establish byte parity, not source fidelity.

Median first frame boundary: **15.298 s**. Median lead before the same on-run completed state: **7.903 s**. Median observed session wall: **24.061 s on / 23.869 s off**. Median paired wall change: **0.03%**.

Identical raw-and-final pairs: **5/5**; their median paired wall change is **0.03%**. One observation per arm and changing initialization times prevent this small run from establishing a general rendering-overhead rate.

Enabled final snapshots matched the returned document in **5/5 cases**. Disabled cases emitted no snapshots in **5/5 cases**.

## Timing and evidence boundaries

- `firstPreviewFrameElapsedMs` is a Compose frame-boundary observation, not verified drawing or physical screen presentation. First UI may be heading/text rather than a completed card; later revisions add content. The warmup live PNG shows a heading while generation remains active.
- `sessionWallElapsedMs` ends at the observed completed state before final screenshot capture. It includes native initialization and metrics cleanup; observation polling is 40 ms. Converter elapsed is reported separately.
- MTP metrics finalization closes its engine. Initialization flags and durations distinguish cold setup from reuse; a warmup does not make subsequent metrics-on calls warm.
- Measured pairs do not take screenshots during inference. The warmup live capture is excluded from measured pairs.
- A nonempty preview does not establish source fidelity; final compilation/repair and failure status remain authoritative.

All measured arms cold-initialized: **True**. Runtime log: **11 GPU FP16+MTP startup lines**, **0 explicit cleanup lines**; maximum Q/DQ rejected counter: **0**. The filtered log records initialization/policy evidence; it does not isolate cleanup duration.

Runtime evidence: [native/fold7_fp16_mtp_20261008/runtime.log](native/fold7_fp16_mtp_20261008/runtime.log)

Device: **SM-F966B**; precision: **FP16_CORRECTED**; MTP requested: **True**. Root-verified corrected-model SHA-256: `7f01bdf1c6ba9bdf658e57c75001fc35a42edad88238bc5dab0a5f7dcf3de373`.

[Gallery](index.html) · [Summary and full hashes](summary.json) · [SDK verification manifest](verification.json) · [Deterministic device log](device_state_tests.txt) · [Native device log](device_native_tests.txt) · [Replay](bixby50_replay.json) · [No-preview diagnosis](bixby50_no_preview_diagnosis.md)

Regenerate after archiving evidence with `python build_report.py`. Missing native inputs remain pending. This script does not run models, builds, or device commands.
