# Android post-repair Golden scoring

`evaluate_predictions` preserves all existing raw scores and adds a separately
labelled score for the UI that the trained Android profile accepts after repair.
The HF and LiteRT Golden CLI scripts accept `--android-repair-runtime`; all
callers also honor `A2UI_ANDROID_REPAIR_RUNTIME`.

Build the bridge once (Java 17+ required):

```bash
python training/scripts/setup_android_repair.py --output-dir /path/to/repair-runtime \
  --java /path/to/java --download-dependencies
export A2UI_ANDROID_REPAIR_RUNTIME=/path/to/repair-runtime/android_repair_runtime.json
```

The bridge compiles the actual Kotlin `GenUiCompiler` and repair/codec sources
from `GenUICraft`, and processes the entire cohort in one JVM. Runtime manifests
bind the source digest and fail on stale code. Rebuild after changing SDK code.
The manifest uses absolute paths; rebuild or relocate its paths when moving hosts.

The policy matches `GenUiSession.TRAINED_E2B_V10_W4`: trim the generated output,
`sourceText=null`, `allowSourceTextFallback=false`, `allowGeneratedDslRepair=true`.
The repair process never receives source responses, expected IR, or benchmark
labels. The complete Android repair implementation is used, including bounded
syntax fixes, graph recovery, data binding recovery and damaged generated state
recovery. The legacy FlatSpec detector is extracted verbatim from its SDK source
because its migration-only file otherwise pulls in Compose; production parsers
and repair are compiled unchanged.

Per-row `android_repair` records acceptance, repair kind, diagnostics and timing.
`android_repaired_text` and `android_repaired_metrics` describe the accepted UI.
Rejected rows score empty UI and remain in the full cohort denominator.
`android_repaired_diagnostics` contains the aggregate; flat
`android_repaired_generation_reward_v5_4` and other numeric aliases support
score tables. Minimal TensorBoard charts include repair score, validity and
latency. Render acceptance is separate from source fidelity, which remains part
of the normal v5.4 score.

`android_repair.batch_wall_seconds` includes JVM startup and JSON transport;
`repair_compute_seconds`, mean/p50/p95/max timings measure Kotlin work. The first
row includes class loading/JIT warmup. `scoring_wall_seconds` measures the Python
score passes separately. These are not GPU inference times.

If no runtime is configured, `android_repair_available=false` and a diagnostic
explain that repair was not run; no fabricated after-repair score is published.

Run parity and score tests with the runtime enabled:

```bash
PYTHONPATH=training/src:dataset/src python -m pytest \
  training/tests/test_android_repair_scoring.py training/tests/test_express_metrics.py
```
