# Current trained E2B inference profile — 2026-10-11

Start with [Findings](FINDINGS.md) for the diagnosis and recommended changes, or open [the HTML report](index.html) for charts and every measured request. [REPORT.md](REPORT.md), [per_request.csv](per_request.csv), and [summary.json](summary.json) contain the generated measurements.

## Scope

- Connected Fold8 SM-F971B and Flip8 SM-F776U; both report SM8850, with different engineering Android builds.
- Identical SHA-256-verified QAT GRPO 20261009 E2B model, corrected FP16 GPU, LiteRT-LM 0.16.1, GenUICraft 0.6.35.
- Three frozen Bixby50 cases: BXP-001 weather, BXP-003 trains, BXP-031 long checklist. This is a latency investigation, not a 50-case accuracy evaluation.
- Per phone: two cold/reinitialized train requests, six warm MTP-on requests, six warm MTP-off requests, and four warm streaming-preview requests. **36 measured requests total**, plus **six separately reported startup warmups**.
- Both phones also replayed three real generated documents twice: **12 final render checks passed**. Screenshots were captured after the measured intervals and visually checked.
- No production app/library inference behavior was changed. Added instrumentation and offline reporting tools only. Network/Perplexity response time is outside these measurements.

Most requests did not remain at Android thermal status 0 despite bounded cooling. Only three of 36 meet the strict-cooled definition. The report retains all observations and reports thermal, clock, memory, output-length, run-order and display-state differences. It does not turn them into an isolated hardware or MTP speedup estimate.

## Evidence layout

`<serial>/runs/<arm>/device/` contains ordered inputs, raw model output, final DSL/A2UI, native metrics, warmups and teardown. Adjacent host receipts and commands record instrumentation completion. `telemetry.jsonl` and `<serial>/resource_samples.jsonl` contain sampled GPU/thermal and CPU/memory conditions. Raw broad device inventories and logcat captures are retained locally for investigation, rather than included in the versioned evidence bundle.

`<serial>/render_timing/` contains final successful UI replay results and screenshots. The first Flip8 replay is preserved in `render_timing_initial_probe_failed/`: the test incorrectly required object identity for an equal repeated document. Compose structural-equality state legitimately retained the prior object. The assertion was corrected to content equality, the probe was rebuilt, and all six checks were rerun successfully. This was a test assertion error, not a renderer failure; the failed attempt is excluded from final timing aggregates.

See [provenance](provenance.json), [code audit](CODE_AUDIT.md), [native trace interpretation](NATIVE_ANALYSIS.md), [resource analysis](RESOURCE_ANALYSIS.md), and [condition events](conditions_events.json). Device/API timings do not establish per-kernel time or achieved GPU memory bandwidth.

## Reproduce

Build/install the Android `DebugAndroidTest` variant on an explicitly selected serial. Use a new output directory for every run. A warm MTP-on control is:

```powershell
adb -s <serial> shell am instrument -w -r `
  -e class com.samsung.genuicraft.InferenceBottleneckProbeTest#measureTrainedConverterBottlenecks `
  -e modelPath <corrected-model-absolute-device-path> `
  -e outputDir <unique-run-leaf> `
  -e cases BXP-001,BXP-003,BXP-031 -e repeats 2 `
  -e reuse true -e mtp true -e streaming false `
  -e cooldownMs 15000 -e thermalCooldownTimeoutMs 90000 `
  com.samsung.genuicraft.test/androidx.test.runner.AndroidJUnitRunner
```

Exact commands for the other arms are saved beside each host receipt. Pull `Android/data/com.samsung.genuicraft/files/sdk_inference_bottleneck/<leaf>` into the matching `runs/<arm>/device` directory. The probe itself defers metrics-driven engine destruction to measure reuse; this is an experimental control, not the default app behavior.

Regenerate the offline tables and charts from the repository root:

```powershell
python -B GenUICraft/tools/report_inference_bottlenecks.py --root GenUICraft/validation/20261011_inference_profile --final `
  --expected-arm R37L20059SY/baseline_cold_mtp_on `
  --expected-arm R37L20059SY/controls_warm_mtp_on `
  --expected-arm R37L20059SY/controls_warm_mtp_off `
  --expected-arm R37L20059SY/controls_stream_mtp_on `
  --expected-arm R3GL203AKSF/baseline_cold_mtp_on `
  --expected-arm R3GL203AKSF/controls_warm_mtp_on `
  --expected-arm R3GL203AKSF/controls_warm_mtp_off `
  --expected-arm R3GL203AKSF/controls_stream_mtp_on
python -B GenUICraft/tools/plot_inference_bottlenecks.py --root GenUICraft/validation/20261011_inference_profile
```

The plot generator requires matplotlib and numpy. No device or model access is needed for offline report generation.
