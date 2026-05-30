#!/usr/bin/env bash
set -euo pipefail

# Build and install libnss_wrapper without root access.
# This fixes Apptainer/Singularity failures like:
#   FATAL: Couldn't determine user account information: user: unknown userid <uid>
#
# Usage on the Slurm/login machine:
#   bash dataset/scripts/install_user_nss_wrapper.sh
# Then rerun:
#   bash dataset/scripts/run_qwen36_vllm_cu128_apptainer.sh

NSS_WRAPPER_VERSION="${NSS_WRAPPER_VERSION:-1.1.16}"
NSS_WRAPPER_PREFIX="${NSS_WRAPPER_PREFIX:-${HOME}/.local/a2ui-nss-wrapper}"
NSS_WRAPPER_BUILD_ROOT="${NSS_WRAPPER_BUILD_ROOT:-${HOME}/.cache/a2ui_nss_wrapper_build}"
NSS_WRAPPER_URL="${NSS_WRAPPER_URL:-https://ftp.samba.org/pub/cwrap/nss_wrapper-${NSS_WRAPPER_VERSION}.tar.gz}"
A2UI_BYPASS_SSL="${A2UI_BYPASS_SSL:-1}"

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required to build nss_wrapper." >&2
  echo "Try: module avail cmake && module load cmake" >&2
  exit 1
fi

if ! command -v tar >/dev/null 2>&1; then
  echo "tar is required to extract nss_wrapper." >&2
  exit 1
fi

download_file() {
  local url="$1"
  local out="$2"
  if command -v curl >/dev/null 2>&1; then
    if [[ "${A2UI_BYPASS_SSL}" = "1" ]]; then
      curl -k -L --retry 3 --retry-delay 2 -o "${out}" "${url}"
    else
      curl -L --retry 3 --retry-delay 2 -o "${out}" "${url}"
    fi
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    if [[ "${A2UI_BYPASS_SSL}" = "1" ]]; then
      wget --no-check-certificate -O "${out}" "${url}"
    else
      wget -O "${out}" "${url}"
    fi
    return
  fi
  echo "curl or wget is required to download nss_wrapper." >&2
  exit 1
}

mkdir -p "${NSS_WRAPPER_BUILD_ROOT}" "${NSS_WRAPPER_PREFIX}"

archive="${NSS_WRAPPER_BUILD_ROOT}/nss_wrapper-${NSS_WRAPPER_VERSION}.tar.gz"
src_dir="${NSS_WRAPPER_BUILD_ROOT}/nss_wrapper-${NSS_WRAPPER_VERSION}"
build_dir="${NSS_WRAPPER_BUILD_ROOT}/build"

if [[ ! -f "${archive}" ]]; then
  echo "Downloading ${NSS_WRAPPER_URL}"
  download_file "${NSS_WRAPPER_URL}" "${archive}"
fi

rm -rf "${src_dir}" "${build_dir}"
tar -xzf "${archive}" -C "${NSS_WRAPPER_BUILD_ROOT}"

if [[ ! -d "${src_dir}" ]]; then
  echo "Expected source folder not found: ${src_dir}" >&2
  echo "Archive contents:" >&2
  tar -tzf "${archive}" | head -20 >&2
  exit 1
fi

cmake \
  -S "${src_dir}" \
  -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${NSS_WRAPPER_PREFIX}" \
  -DUNIT_TESTING=OFF

cmake --build "${build_dir}" --parallel "${MAX_JOBS:-4}"
cmake --install "${build_dir}"

lib_path="$(find "${NSS_WRAPPER_PREFIX}" -type f -name 'libnss_wrapper.so' | head -n 1)"
if [[ -z "${lib_path}" || ! -f "${lib_path}" ]]; then
  echo "libnss_wrapper.so was not installed under ${NSS_WRAPPER_PREFIX}" >&2
  exit 1
fi

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/a2ui_nss_wrapper_test.XXXXXX")"
trap 'rm -rf "${tmp_dir}"' EXIT
uid="$(id -u)"
gid="$(id -g)"
home_dir="${HOME:-/tmp}"
user_name="${USER:-a2ui_user}"
group_name="$(id -gn 2>/dev/null || echo a2ui_group)"

cat > "${tmp_dir}/passwd" <<EOF
root:x:0:0:root:/root:/bin/bash
${user_name}:x:${uid}:${gid}:A2UI synthetic user:${home_dir}:/bin/bash
EOF

cat > "${tmp_dir}/group" <<EOF
root:x:0:
${group_name}:x:${gid}:
EOF

if command -v getent >/dev/null 2>&1; then
  LD_PRELOAD="${lib_path}" \
  NSS_WRAPPER_PASSWD="${tmp_dir}/passwd" \
  NSS_WRAPPER_GROUP="${tmp_dir}/group" \
  getent passwd "${uid}" >/dev/null
fi

echo
echo "Installed libnss_wrapper:"
echo "  ${lib_path}"
echo
echo "The Apptainer launcher auto-detects this path. If needed, export it explicitly:"
echo "  export A2UI_NSS_WRAPPER_LIB=${lib_path}"
