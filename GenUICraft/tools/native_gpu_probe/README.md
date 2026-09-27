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
  [--max-context 8192] [--max-output 2048]
```

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
