#!/usr/bin/env bash
# Build in an isolated upstream checkout. No APK or GPU project is rebuilt.
set -euo pipefail
: "${ANDROID_NDK_HOME:?Set ANDROID_NDK_HOME to an Android NDK r28b or later}"
: "${QAIRT_ROOT:?Set QAIRT_ROOT to the matching Qualcomm SDK version directory containing lib/}"
[[ $# == 1 ]] || { echo 'Usage: bash training/scripts/build_sm8850_npu_probe.sh NEW_RUNTIME_DIR' >&2; exit 2; }
repo=$(cd "$(dirname "$0")/../.." && pwd)
out=$(realpath -m "$1")
[[ ! -e "$out" ]] || { echo "Output already exists: $out" >&2; exit 1; }
qnn=$(cd "$QAIRT_ROOT/lib" && pwd)
libs=(libQnnHtp.so libQnnHtpPrepare.so libQnnSystem.so libQnnHtpV81Stub.so)
for lib in "${libs[@]}"; do test -s "$qnn/aarch64-android/$lib"; done
skel="$qnn/hexagon-v81/unsigned/libQnnHtpV81Skel.so"
test -s "$skel"
mkdir -p "$(dirname "$out")"
work=$(mktemp -d "$(dirname "$out")/.sm8850-build.XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT
GIT_LFS_SKIP_SMUDGE=1 git clone --quiet --depth 1 --branch v0.17.0 \
  https://github.com/google-ai-edge/LiteRT-LM.git "$work/lm"
cd "$work/lm"
[[ $(git rev-parse HEAD) == e9fd8c53ff968071774206163027dd84bedfe925 ]]
git lfs pull --include='prebuilt/android_arm64/*.so'
cp -r "$repo/training/native/sm8850" a2ui_probe
bazel=${BAZEL:-bazelisk}
"$bazel" build --config=android_arm64 //a2ui_probe:bixby50_npu \
  @litert//litert/vendors/qualcomm/dispatch:dispatch_api_so
mkdir "$work/runtime"
cp bazel-bin/a2ui_probe/bixby50_npu bazel-bin/external/litert/litert/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so \
  prebuilt/android_arm64/*.so "$skel" "$work/runtime/"
for lib in "${libs[@]}"; do cp "$qnn/aarch64-android/$lib" "$work/runtime/"; done
mv -T "$work/runtime" "$out"
echo "Android SM8850 runtime built in $out"
