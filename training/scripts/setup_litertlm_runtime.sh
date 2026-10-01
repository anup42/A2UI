#!/usr/bin/env bash
# Create an isolated Linux LiteRT-LM runtime; never install a system GPU driver.
set -euo pipefail

usage() {
  echo "Usage: $0 RUNTIME_DIR [--python PYTHON] [--offline-runtime ENV_DIR] [--vulkan-loader-deb FILE] [--nvidia-gl-deb FILE]" >&2
  exit 2
}
[[ $# -ge 1 ]] || usage
runtime_dir=$(realpath -m "$1")
shift
python_bin=python3
offline_runtime=
vulkan_deb=
nvidia_deb=
while [[ $# -gt 0 ]]; do
  [[ $# -ge 2 ]] || usage
  case "$1" in
    --python) python_bin=$2 ;;
    --offline-runtime) offline_runtime=$2 ;;
    --vulkan-loader-deb) vulkan_deb=$2 ;;
    --nvidia-gl-deb) nvidia_deb=$2 ;;
    *) usage ;;
  esac
  shift 2
done
[[ "$runtime_dir" != / && "$runtime_dir" != "$HOME" ]] || usage
mkdir -p "$runtime_dir" "$runtime_dir/tmp" "$runtime_dir/cache/pip" "$runtime_dir/cache/xdg"
export TMPDIR="$runtime_dir/tmp" PIP_CACHE_DIR="$runtime_dir/cache/pip" XDG_CACHE_HOME="$runtime_dir/cache/xdg"
export PYTHONDONTWRITEBYTECODE=1
if [[ ! -x "$runtime_dir/venv/bin/python" ]]; then
  "$python_bin" -m venv "$runtime_dir/venv"
fi
if [[ -n "$offline_runtime" ]]; then
  # The pinned runtime wheel has no Python dependencies. Copy its installed
  # package and metadata into the new environment when PyPI is unreachable.
  "$offline_runtime/bin/python" - "$runtime_dir/venv/bin/python" <<'PY'
import importlib.metadata, json, pathlib, shutil, subprocess, sys
dist = importlib.metadata.distribution('litert-lm-api')
if dist.version != '0.17.0':
    raise SystemExit('Offline source must contain litert-lm-api==0.17.0')
target = pathlib.Path(subprocess.check_output([
    sys.argv[1], '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'
], text=True).strip())
source = pathlib.Path(dist.locate_file(''))
for name in ('litert_lm', 'litert_lm_api-0.17.0.dist-info'):
    shutil.copytree(source / name, target / name, dirs_exist_ok=True)
print(json.dumps({'offline_source': str(source), 'target': str(target), 'version': dist.version}))
PY
else
  "$runtime_dir/venv/bin/python" -m pip install 'litert-lm-api==0.17.0'
fi
if [[ -n "$vulkan_deb" ]]; then
  dpkg-deb -x "$vulkan_deb" "$runtime_dir/vulkan"
fi
if [[ -n "$nvidia_deb" ]]; then
  installed_version=$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' /proc/driver/nvidia/version | head -n 1)
  package_version=$(dpkg-deb -f "$nvidia_deb" Version)
  package_name=$(dpkg-deb -f "$nvidia_deb" Package)
  [[ -n "$installed_version" && "$package_version" == "$installed_version"-* && "$package_name" == libnvidia-gl-* ]] || {
    echo "Refusing mismatched graphics libraries: kernel=$installed_version package=$package_name/$package_version" >&2
    exit 1
  }
  dpkg-deb -x "$nvidia_deb" "$runtime_dir/nvidia"
  # Headless NVIDIA uses EGL. GLX can fail vkCreateInstance despite resolved
  # shared libraries; no X server or system ICD modification is necessary.
  "$runtime_dir/venv/bin/python" - "$runtime_dir" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
icd = json.loads((root / 'nvidia/usr/share/vulkan/icd.d/nvidia_icd.json').read_text())
icd['ICD']['library_path'] = 'libEGL_nvidia.so.0'
(root / 'nvidia_egl_icd.json').write_text(json.dumps(icd, indent=2) + '\n')
PY
fi
{
  printf 'export A2UI_LITERT_RUNTIME_DIR=%q\n' "$runtime_dir"
  cat <<'SH'
export TMPDIR="$A2UI_LITERT_RUNTIME_DIR/tmp"
export XDG_CACHE_HOME="$A2UI_LITERT_RUNTIME_DIR/cache/xdg"
export PIP_CACHE_DIR="$A2UI_LITERT_RUNTIME_DIR/cache/pip"
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$A2UI_LITERT_RUNTIME_DIR/vulkan/usr/lib/x86_64-linux-gnu:$A2UI_LITERT_RUNTIME_DIR/nvidia/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
if [[ -f "$A2UI_LITERT_RUNTIME_DIR/nvidia_egl_icd.json" ]]; then
  export VK_ICD_FILENAMES="$A2UI_LITERT_RUNTIME_DIR/nvidia_egl_icd.json"
fi
SH
} > "$runtime_dir/activate.sh"
source "$runtime_dir/activate.sh"
"$runtime_dir/venv/bin/python" -m pip freeze > "$runtime_dir/requirements.lock.txt"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
"$runtime_dir/venv/bin/python" "$script_dir/run_litertlm_gpu.py" --preflight --report "$runtime_dir/preflight.json"
echo "Source $runtime_dir/activate.sh; pass --runtime-python $runtime_dir/venv/bin/python to evaluate_litertlm_on_golden.py."
