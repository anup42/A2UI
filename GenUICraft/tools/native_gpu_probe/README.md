# Native LiteRT-LM diagnostic probe

`probe.cc` runs one frozen text prompt through the pinned LiteRT-LM C API on
Android arm64. It does not use the GenUICraft app. The prompt file is read as
binary UTF-8 and passed directly to a session with prompt templating disabled;
the probe does not prepend `<bos>` or alter whitespace. The native session may
insert its own BOS token. The output in `<prefix>.raw.txt` is the C API response
text without trimming or repair.

The run enables LiteRT-LM benchmark counters, disables speculative decoding,
and uses top-p sampling with `top_k=1`, `top_p=1`, `temperature=0`, and seed 42.
Context and output limits default to 8192 and 2048 tokens. `--force-f32`
requests FP32 **activations** through the C API; it does not change the model's
quantized weights. Each run has a distinct `<prefix>.cache` directory to keep
backend and precision controls separate.

Run:

```text
probe --model MODEL.litertlm --prompt PROMPT.txt --output-prefix OUT/PREFIX \
  --backend cpu|gpu [--force-f32] [--sampler-backend default|cpu] \
  [--score-target TARGET.txt] \
  [--max-context 8192] [--max-output 2048]
```

`--score-target TARGET.txt` switches the probe from generation to fixed-target
text scoring. It reads the target as exact UTF-8 bytes, prefills the same prompt,
and calls the distributed C API's `litert_lm_session_run_text_scoring` with one
target and token lengths requested. This is useful for comparing GPU FP16 and
FP32 with identical prompt and continuation bytes, before sampled outputs can
diverge. Use the same target file and model for each precision run. The probe
copies the target to `<prefix>.scored_target.txt`; it does not create
`<prefix>.raw.txt` in scoring mode. The original decode path is unchanged when
`--score-target` is absent.

Scoring mode writes the returned aggregate score to `target_score` in
`<prefix>.metrics.json`. When the runtime supplies per-token scores, it also
writes one float per line to `<prefix>.target_token_scores.txt`; the status and
count are recorded in `target_token_scores_status` and
`target_token_score_count`. An unavailable per-token array is reported as
`unavailable`, while a missing aggregate score or null scoring response is a
probe error with the stage and reason in metrics. Nonfinite scores are retained
in the token-score sidecar and counted in metrics. The score is the runtime's
raw scoring value; compare the same target across controls without assuming a
particular sign or normalization. `decode_wall_ms` remains null and
`score_wall_ms` records the scoring call in this mode.

`<prefix>.target_token_ids.txt` is the C API tokenizer's independent encoding
of the target. Its count is recorded separately from the runtime's scored token
length and per-token score count. Tokenization at the prompt/target boundary,
native BOS handling, and score-array availability may differ, so the file is
diagnostic evidence rather than a guaranteed index map for the score array.
These scores measure forced target tokens, not the full vocabulary logits,
sampled IDs, or intermediate layer tensors. If the pinned backend does not
support scoring, the probe exits with an explicit error; an exported C API
symbol alone does not prove runtime support for every model/backend.

`<prefix>.metrics.json` records process ID, requested controls, input/output
sizes, wall times, and native benchmark counts/rates when available. The token
ID sidecars are produced by calling the C API tokenizer on the supplied prompt
and returned output. Retokenized output IDs are not a direct capture of the
decode sequence, and tokenizer input IDs may omit a BOS inserted by the native
session. The C API does not expose top-k logits; optional selected-token scores
are written as a separate sidecar when available.

Build from the repository root with the Android NDK and the pinned C API
distribution already unpacked under `.tmp/r64_gpu_controls/c_api`:

```powershell
$ndkClang = 'C:\Users\anupk\AppData\Local\Android\Sdk\ndk\28.2.13676358\toolchains\llvm\prebuilt\windows-x86_64\bin\clang++.exe'
$nativeCapi = '.tmp\r64_gpu_controls\c_api'
& $ndkClang --target=aarch64-linux-android28 -std=c++17 -O2 -static-libstdc++ `
  -I "$nativeCapi\include" GenUICraft\tools\native_gpu_probe\probe.cc `
  -L "$nativeCapi\lib\android_arm64" '-l:liblitert-lm.so' -ldl `
  '-Wl,-rpath,$ORIGIN' -o .tmp\r64_gpu_controls\device\probe
```

The `-l:liblitert-lm.so` form matters because the distributed shared library
has no SONAME: passing its full path to the linker records that host path in
`DT_NEEDED`, which fails on Android. Verify `DT_NEEDED` names
`liblitert-lm.so` and `RUNPATH` is `$ORIGIN` before pushing the probe and
library together.

`--sampler-backend cpu` is a diagnostic control implemented in
`sampler_bridge.h` against the pinned native library ABI. It changes the
sampler selected inside engine and session settings while leaving the model
backend requested by `--backend` intact. The bridge fails closed if its ABI
checks do not match; it is not a public LiteRT-LM C API feature and should not
be copied into production code.

Use `run_controls.py` for the controlled experiment rather than invoking the
bridge against an arbitrary library. It checks the native library and model
hashes on the device before each variant, saves the full command and probe hash,
and scores the raw output without repair. The library comes from the official
[v0.16.0 C API distribution](https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.16.0/litert_lm_c_api-0.1.0.zip).
The distribution SHA-256 is
`f0f3ae7b5730af783d1f018f7ad9a8de20c25fedf01af4e35fc11d4382246f7d`.
Runtime dependency hashes, frozen prompts, results and version caveats are in
the [experiment report](../../validation/20260928_r64_gpu_controls/REPORT.md).

After staging the probe, its native dependencies and prompts under
`/data/local/tmp/r64_gpu_controls`, set its executable mode with
`adb -s R3GL203AKSF shell chmod 755 /data/local/tmp/r64_gpu_controls/probe`.
Run the controls sequentially, using a fresh label to avoid overwriting evidence:

```powershell
python GenUICraft/tools/native_gpu_probe/run_controls.py --variant gpu_default --label repeat_gpu_default
python GenUICraft/tools/native_gpu_probe/run_controls.py --variant gpu_cpu_sampler --label repeat_gpu_cpu_sampler
python GenUICraft/tools/native_gpu_probe/run_controls.py --variant gpu_fp32 --label repeat_gpu_fp32
```

The diagnostic currently pins BXP-001 and BXP-003 and the rank-64 QAT-compatible
model hash. It is intentionally not a general model benchmark. The host script
pulls only output/metrics/token IDs, not the compiled GPU caches. Raw output
token IDs are **retokenized**, not a direct capture of sampled IDs. The final
source also contains a defensive null-pointer check added after the successful
device controls; that final source was compiled with `-Wall -Wextra`.

## Experimental rank-64 RoPE lookup repack

`patch_rope_lookup.py` and `verify_rope_lookup.py` are **experimental diagnostic
tools, not a deployment exporter or an SDK precision gate**. They accept only
the rank-64 QAT-compatible input package with SHA-256
`de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62`.
The repacker replaces the four assistant graphs' 16 RoPE SIN/COS operations with
exact-position GATHERs into precomputed FLOAT32 sin/cos tables for positions
`[0, 8192)`. It leaves the MTP drafter section untouched, so diagnostic device
runs must keep MTP disabled. The package only gets a TFLite description marker;
it does **not** get production LiteRT-LM root metadata. The lookup has a hard
8192-position exclusive bound even when run with GPU FP32.

Install `ai-edge-litert`, `flatbuffers`, and `numpy` in the Python environment,
then run these from the A2UI repository root using an existing local copy of
the pinned input. The destination directory must already exist, and each
destination must be a new path. A complete package needs about 2.7 GB of free
space beyond the input. `--plan-only` validates and reports package geometry
without writing one.

```powershell
$inputModel = '.tmp\fp16_rootcause\r64.litertlm'
$outDir = '.tmp\fp16_export_audit'
python GenUICraft/tools/native_gpu_probe/patch_rope_lookup.py $inputModel "$outDir\rope_lookup.litertlm" --plan-only
python GenUICraft/tools/native_gpu_probe/patch_rope_lookup.py $inputModel "$outDir\rope_lookup.litertlm"
python GenUICraft/tools/native_gpu_probe/verify_rope_lookup.py $inputModel "$outDir\rope_lookup.litertlm"

$frontSize = (Get-Content "$outDir\rope_lookup.litertlm.report.json" | ConvertFrom-Json).front_size
python GenUICraft/tools/native_gpu_probe/patch_rope_lookup.py $inputModel "$outDir\repack_control.litertlm" --repack-only --match-front-size $frontSize
python GenUICraft/tools/native_gpu_probe/verify_rope_lookup.py $inputModel "$outDir\repack_control.litertlm" --repack-only
```

The verifier checks every original inline buffer byte, a complete verbatim
copy of the original target model backing those buffers, all non-target
sections including MTP, graph operator counts, and the sidecar SHA-256 values.
The causal packaging control has the same external-buffer/table layout and
front length but retains the original SIN/COS operators. Neither diagnostic
package should be selected by production SDK FP16 gating.

The separate [LiteRT-LM v0.17.1 mixed-accumulation patch](litert_lm_v0.17.1_mixed_accum.patch)
is a small, **untested** upstream C++ proposal. It adds an explicit opt-in
`GpuConfig` flag and selects LiteRT enum value 3 *after* `SetCommonGpuOptions`,
which otherwise overwrites GPU precision. The current C API and Kotlin wrapper
do not expose this new flag, so the patch alone cannot change the packaged app
runtime. LiteRT documents FP32 accumulation only for supported matrix
operators; generated kernels and output quality still need device validation.
