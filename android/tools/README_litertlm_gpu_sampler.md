# LiteRT-LM Android GPU sampler and GPU-compatible model export

The LiteRT-LM Android AAR contains the GPU executor, but it does not bundle the
optional OpenCL Top-K sampler. Without this library, LiteRT-LM falls back to its
static sampler on Adreno devices.

The packaged library is the official LiteRT-LM v0.16.1 ARM64 sampler. The
upstream sampler has an unresolved `kLiteRtRuntimeBuiltin` symbol when loaded
through the Android AAR, so this app packages a small compatibility shim built
against the official LiteRT 2.2.0 Android runtime and the matching
`libc++_shared.so`. The patched sampler depends on the shim rather than on the
private JNI library.

Source:

`https://github.com/google-ai-edge/LiteRT-LM/raw/refs/tags/v0.16.1/prebuilt/android_arm64/libLiteRtTopKOpenClSampler.so`

The unpatched source SHA-256 is:

`4404DC68786460602685CAB62DDFA29035E9CFC38BB4550DEC15ABAAA1302A82`

The minimally patched library SHA-256 (with a `DT_NEEDED` entry for
`libLiteRtRuntimeBuiltin.so`) is:

`1E20C81D0FE2DC8F25A26929632564CDF1E3758D3662E315748606A70E274719`

The patch is applied by `android/tools/patch_litertlm_gpu_sampler_elf.py`.
It reuses the existing, longer `DT_SONAME` string slot as a `DT_NEEDED`
entry, so the Android-specific hash and relocation tables are not rewritten.

It is stored at:

`android/app/src/main/jniLibs/arm64-v8a/libLiteRtTopKOpenClSampler.so`

The companion files are:

- `libLiteRtRuntimeBuiltin.so`: LiteRT 2.2.0 compatibility shim exporting
  `kLiteRtRuntimeBuiltin`.
- `libLiteRt.so` and `libLiteRtClGlAccelerator.so`: the matching LiteRT 2.2.0
  runtime and OpenCL delegate.
- `libc++_shared.so`: the NDK C++ runtime required by the shim.

For the patched v0.16.1 provider path, place
`litertlm-android-0.16.1-gpu-fixed-with-provider-v6.aar` under the repository's
`working_dir/`. Gradle uses that AAR when present and otherwise falls back to
the published LiteRT-LM 0.15.0 dependency so a clean checkout remains
buildable.

## Model export requirement

Use `training/scripts/export_litert_gpu_compat.py` for custom Gemma 3 270M
exports. The script removes StableHLO composite boundaries for both
`odml.rms_norm` and attention softmax, leaving primitive LiteRT operators that
the Adreno delegate evaluates correctly. Removing only the softmax boundary is
not sufficient: the custom graph can report full delegation while still
returning invalid sampled token `0` values.

Before installing a candidate, inspect its graph and confirm that
`STABLEHLO_COMPOSITE` is absent. The Android parity probe must also complete
without `Invalid decode and sample result` warnings.
