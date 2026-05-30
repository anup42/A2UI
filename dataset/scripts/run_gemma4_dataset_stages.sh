#!/usr/bin/env bash
set -euo pipefail

# Run dataset Stage 1/2/3/4/5 against a running Gemma4 31B vLLM
# OpenAI-compatible endpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GEMMA4_ENABLE_REASONING="${GEMMA4_ENABLE_REASONING:-0}"
if [[ "${GEMMA4_ENABLE_REASONING,,}" =~ ^(1|true|yes|y|on)$ ]]; then
  MODEL_NAME="${MODEL_NAME:-gemma4_31b_vllm_reasoning}"
  export LOCAL_VLLM_ENABLE_THINKING="${LOCAL_VLLM_ENABLE_THINKING:-1}"
else
  MODEL_NAME="${MODEL_NAME:-gemma4_31b_vllm}"
  export LOCAL_VLLM_ENABLE_THINKING="${LOCAL_VLLM_ENABLE_THINKING:-0}"
fi

RUN_ID="${RUN_ID:-dataset_gemma4_31b_vllm_v0}"
MODEL_ROOT="${MODEL_ROOT:-${LOCAL_MODEL_ROOT:-${A2UI_MODEL_ROOT:-${GEMMA4_MODEL_ROOT:-}}}}"
RATE_LIMIT_QPS="${RATE_LIMIT_QPS:-0.2}"
STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-1}"
RENDER_WORKERS="${RENDER_WORKERS:-1}"
MAX_QUERIES_TOTAL="${MAX_QUERIES_TOTAL:-}"
MAX_RESPONSES_TOTAL="${MAX_RESPONSES_TOTAL:-}"
MAX_GENUI_TOTAL="${MAX_GENUI_TOTAL:-}"
MAX_GENERATION_TOTAL="${MAX_GENERATION_TOTAL:-}"
GENERATION_CYCLE_SIZE="${GENERATION_CYCLE_SIZE:-${MAX_GENERATION_CYCLE_SIZE:-500}}"
K_QUERIES_PER_INTENT="${K_QUERIES_PER_INTENT:-}"
STAGE="${STAGE:-all}"

if [[ -n "${MAX_GENERATION_TOTAL}" ]]; then
  MAX_QUERIES_TOTAL="${MAX_GENERATION_TOTAL}"
  MAX_RESPONSES_TOTAL="${MAX_GENERATION_TOTAL}"
  MAX_GENUI_TOTAL="${MAX_GENERATION_TOTAL}"
fi

export LOCAL_ALLOW_HTTP_ENDPOINT="${LOCAL_ALLOW_HTTP_ENDPOINT:-1}"
export LOCAL_STRICT_OFFLINE="${LOCAL_STRICT_OFFLINE:-0}"
export LOCAL_VLLM_STRIP_THINKING="${LOCAL_VLLM_STRIP_THINKING:-1}"
export LOCAL_VLLM_SEND_CHAT_TEMPLATE_KWARGS="${LOCAL_VLLM_SEND_CHAT_TEMPLATE_KWARGS:-0}"
export LOCAL_VLLM_TIMEOUT_SECONDS="${LOCAL_VLLM_TIMEOUT_SECONDS:-600}"
export LOCAL_VLLM_MAX_OUTPUT_TOKENS="${LOCAL_VLLM_MAX_OUTPUT_TOKENS:-8192}"
export LOCAL_STAGE3_PROMPT_MAX_TOKENS="${LOCAL_STAGE3_PROMPT_MAX_TOKENS:-8192}"
export A2UI_QUERY_MAX_TOKENS="${A2UI_QUERY_MAX_TOKENS:-8192}"
export A2UI_RESPONSE_MAX_TOKENS="${A2UI_RESPONSE_MAX_TOKENS:-8192}"
export A2UI_GENUI_MAX_TOKENS="${A2UI_GENUI_MAX_TOKENS:-8192}"
export A2UI_GENUI_PROMPT_MAX_TOKENS="${A2UI_GENUI_PROMPT_MAX_TOKENS:-8192}"
export A2UI_STAGE3_PROMPT_FILE="${A2UI_STAGE3_PROMPT_FILE:-prompts/genui_gen_mobile_flatspec_ondevice_v3.md}"
export A2UI_STAGE25_BATCH_SIZE="${A2UI_STAGE25_BATCH_SIZE:-${GENERATION_CYCLE_SIZE}}"
export A2UI_QUERY_TEMPERATURE="${A2UI_QUERY_TEMPERATURE:-1.0}"
export A2UI_RESPONSE_TEMPERATURES="${A2UI_RESPONSE_TEMPERATURES:-1.0}"
export A2UI_GENUI_TEMPERATURE="${A2UI_GENUI_TEMPERATURE:-0.7}"
export A2UI_GENUI_REPAIR_TEMPERATURE="${A2UI_GENUI_REPAIR_TEMPERATURE:-0.2}"
export A2UI_GENUI_FINAL_REGEN_TEMPERATURE="${A2UI_GENUI_FINAL_REGEN_TEMPERATURE:-0.7}"
export LOCAL_VLLM_TOP_P="${LOCAL_VLLM_TOP_P:-0.95}"
export LOCAL_VLLM_TOP_K="${LOCAL_VLLM_TOP_K:-64}"
export LOCAL_VLLM_REPETITION_PENALTY="${LOCAL_VLLM_REPETITION_PENALTY:-1.0}"
export GEMMA4_MODEL_ID="${GEMMA4_MODEL_ID:-google/gemma-4-31b-it}"
export MODEL_ROOT
export LOCAL_MODEL_ROOT="${LOCAL_MODEL_ROOT:-${MODEL_ROOT}}"
export A2UI_MODEL_ROOT="${A2UI_MODEL_ROOT:-${MODEL_ROOT}}"
export GEMMA4_MODEL_ROOT="${GEMMA4_MODEL_ROOT:-${MODEL_ROOT}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export A2UI_DISABLE_SSL_VERIFY="${A2UI_DISABLE_SSL_VERIFY:-1}"
if [[ "${A2UI_DISABLE_SSL_VERIFY}" = "1" ]]; then
  export CURL_CA_BUNDLE=""
  export REQUESTS_CA_BUNDLE=""
  export SSL_CERT_FILE=""
  export PYTHONHTTPSVERIFY=0
  export GIT_SSL_NO_VERIFY=1
  export HF_HUB_DISABLE_SSL_VERIFICATION=1
fi

ensure_vllm_served_model() {
  if [[ "${A2UI_SKIP_VLLM_MODEL_CHECK:-0}" = "1" ]]; then
    export LOCAL_VLLM_SERVED_MODEL="${LOCAL_VLLM_SERVED_MODEL:-${GEMMA4_MODEL_ID}}"
    return 0
  fi

  local expected="${LOCAL_VLLM_SERVED_MODEL:-${GEMMA4_MODEL_ID}}"
  local models_url="${LOCAL_VLLM_MODELS_URL:-http://127.0.0.1:8000/v1/models}"
  local served
  served="$(python - "${models_url}" "${expected}" <<'PY'
import json
import sys
import urllib.request

models_url = sys.argv[1]
expected = sys.argv[2]
try:
    with urllib.request.urlopen(models_url, timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
except Exception as exc:
    print(f"Could not read local vLLM models from {models_url}: {exc}", file=sys.stderr)
    raise SystemExit(2)

ids = [
    item.get("id")
    for item in payload.get("data", [])
    if isinstance(item, dict) and isinstance(item.get("id"), str)
]
if expected in ids:
    print(expected)
    raise SystemExit(0)
for model_id in ids:
    if model_id.lower() == expected.lower():
        print(model_id)
        raise SystemExit(0)

print(
    "Local vLLM is running, but the requested model is not served.\n"
    f"  requested: {expected}\n"
    f"  served: {ids}\n"
    "Fix: restart vLLM with dataset/scripts/run_gemma4_vllm_python.sh, "
    "or export LOCAL_VLLM_SERVED_MODEL to one of the served ids.",
    file=sys.stderr,
)
raise SystemExit(3)
PY
)" || return $?

  export LOCAL_VLLM_SERVED_MODEL="${served}"
  echo "Using local vLLM served model: ${LOCAL_VLLM_SERVED_MODEL}"
}

case "${STAGE}" in
  1|2|3|all|cyclic)
    ensure_vllm_served_model
    ;;
esac

run_stage() {
  local stage="$1"
  shift
  local max_queries="${MAX_QUERIES_TOTAL}"
  local max_responses="${MAX_RESPONSES_TOTAL}"
  local max_genui="${MAX_GENUI_TOTAL}"
  if [[ -v A2UI_STAGE_MAX_QUERIES_TOTAL ]]; then
    max_queries="${A2UI_STAGE_MAX_QUERIES_TOTAL}"
  fi
  if [[ -v A2UI_STAGE_MAX_RESPONSES_TOTAL ]]; then
    max_responses="${A2UI_STAGE_MAX_RESPONSES_TOTAL}"
  fi
  if [[ -v A2UI_STAGE_MAX_GENUI_TOTAL ]]; then
    max_genui="${A2UI_STAGE_MAX_GENUI_TOTAL}"
  fi
  local max_args=()
  if [[ -n "${max_queries}" ]]; then
    max_args+=(--max_queries_total "${max_queries}")
  fi
  if [[ -n "${max_responses}" ]]; then
    max_args+=(--max_responses_total "${max_responses}")
  fi
  if [[ -n "${max_genui}" ]]; then
    max_args+=(--max_genui_total "${max_genui}")
  fi
  local k_queries="${K_QUERIES_PER_INTENT}"
  if [[ "${stage}" = "1" && -z "${k_queries}" && -n "${max_queries}" ]]; then
    local intents
    intents="$(intent_count)"
    if (( intents > 0 )); then
      k_queries=$(( (max_queries + intents - 1) / intents ))
    fi
  fi
  if [[ -n "${k_queries}" ]]; then
    max_args+=(--k_queries_per_intent "${k_queries}")
  fi
  echo "Running Stage ${stage} with ${MODEL_NAME}, run_id=${RUN_ID}, reasoning=${GEMMA4_ENABLE_REASONING}"
  echo "Sampling: query_temp=${A2UI_QUERY_TEMPERATURE} response_temps=${A2UI_RESPONSE_TEMPERATURES} genui_temp=${A2UI_GENUI_TEMPERATURE} top_p=${LOCAL_VLLM_TOP_P} top_k=${LOCAL_VLLM_TOP_K} repetition_penalty=${LOCAL_VLLM_REPETITION_PENALTY}"
  if [[ "${stage}" = "3" ]]; then
    echo "Stage 2.5 domain batch size=${A2UI_STAGE25_BATCH_SIZE}"
  fi
  python dataset/src/main.py \
    --stage "${stage}" \
    --model "${MODEL_NAME}" \
    --run_id "${RUN_ID}" \
    --rate_limit_qps "${RATE_LIMIT_QPS}" \
    "${max_args[@]}" \
    "$@"
}

resolve_run_file() {
  local file_name="$1"
  python - "${RUN_ID}" "${file_name}" <<'PY'
from pathlib import Path
import sys

repo = Path.cwd()
sys.path.insert(0, str(repo / "dataset" / "src"))
from pipeline.storage import get_run_paths  # noqa: E402
from utils.config import load_yaml  # noqa: E402

run_id = sys.argv[1]
file_name = sys.argv[2]
root = repo / "dataset"
cfg = load_yaml(root / "configs" / "run.yaml").get("run", {})
output_dir = Path(cfg.get("output_dir", "data/runs"))
if not output_dir.is_absolute():
    output_dir = root / output_dir
paths = get_run_paths(output_dir, run_id, cfg.get("artifact_dir", "artifacts"))
print(paths.run_dir / file_name)
PY
}

jsonl_count() {
  local path="$1"
  python - "${path}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.exists():
    print(0)
    raise SystemExit(0)
count = 0
with path.open("r", encoding="utf-8", errors="replace") as handle:
    for line in handle:
        if line.strip():
            count += 1
print(count)
PY
}

intent_count() {
  python - <<'PY'
from pathlib import Path
import sys

repo = Path.cwd()
sys.path.insert(0, str(repo / "dataset" / "src"))
from utils.config import load_yaml  # noqa: E402

root = repo / "dataset"
cfg = load_yaml(root / "configs" / "run.yaml").get("run", {})
intents_file = root / cfg.get("intents_file", "intents.info")
count = 0
for line in intents_file.read_text(encoding="utf-8").splitlines():
    if line.strip():
        count += 1
print(count)
PY
}

run_limited_stage() {
  local stage="$1"
  local max_queries="$2"
  local max_responses="$3"
  local max_genui="$4"
  shift 4
  A2UI_STAGE_MAX_QUERIES_TOTAL="${max_queries}" \
  A2UI_STAGE_MAX_RESPONSES_TOTAL="${max_responses}" \
  A2UI_STAGE_MAX_GENUI_TOTAL="${max_genui}" \
    run_stage "${stage}" "$@"
}

ensure_stage_count() {
  local stage="$1"
  local path="$2"
  local target="$3"
  shift 3
  local current before after delta
  current="$(jsonl_count "${path}")"
  while (( current < target )); do
    before="${current}"
    delta=$(( target - current ))
    case "${stage}" in
      1)
        run_limited_stage 1 "${delta}" "" "" "$@"
        ;;
      2)
        run_limited_stage 2 "" "${delta}" "" "$@"
        ;;
      3)
        run_limited_stage 3 "" "" "${delta}" --genui_batch_size "${STAGE3_BATCH_SIZE}" "$@"
        ;;
      *)
        echo "Internal error: unsupported cyclic stage ${stage}" >&2
        exit 1
        ;;
    esac
    after="$(jsonl_count "${path}")"
    echo "Cyclic Stage ${stage} progress ${after}/${target} created=$(( after - before ))"
    if (( after <= before )); then
      echo "Cyclic Stage ${stage} made no progress toward target=${target}; current=${after}" >&2
      exit 1
    fi
    current="${after}"
  done
}

run_cyclic_generation() {
  if [[ -z "${MAX_GENERATION_TOTAL}" ]]; then
    echo "MAX_GENERATION_TOTAL is required for cyclic generation." >&2
    exit 1
  fi
  if ! [[ "${MAX_GENERATION_TOTAL}" =~ ^[0-9]+$ ]] || (( MAX_GENERATION_TOTAL <= 0 )); then
    echo "MAX_GENERATION_TOTAL must be a positive integer; got '${MAX_GENERATION_TOTAL}'." >&2
    exit 1
  fi
  if ! [[ "${GENERATION_CYCLE_SIZE}" =~ ^[0-9]+$ ]] || (( GENERATION_CYCLE_SIZE <= 0 )); then
    echo "GENERATION_CYCLE_SIZE must be a positive integer; got '${GENERATION_CYCLE_SIZE}'." >&2
    exit 1
  fi

  local queries_path responses_path domains_path genui_path intents k_per_intent
  queries_path="$(resolve_run_file "queries.jsonl")"
  responses_path="$(resolve_run_file "responses.jsonl")"
  domains_path="$(resolve_run_file "domains.jsonl")"
  genui_path="$(resolve_run_file "genui.jsonl")"
  intents="$(intent_count)"
  if (( intents <= 0 )); then
    echo "No intents found; cannot compute k_queries_per_intent." >&2
    exit 1
  fi
  k_per_intent=$(( (MAX_GENERATION_TOTAL + intents - 1) / intents ))
  if [[ -n "${K_QUERIES_PER_INTENT}" ]]; then
    if ! [[ "${K_QUERIES_PER_INTENT}" =~ ^[0-9]+$ ]] || (( K_QUERIES_PER_INTENT <= 0 )); then
      echo "K_QUERIES_PER_INTENT must be a positive integer; got '${K_QUERIES_PER_INTENT}'." >&2
      exit 1
    fi
    if (( K_QUERIES_PER_INTENT > k_per_intent )); then
      k_per_intent="${K_QUERIES_PER_INTENT}"
    fi
  fi
  export K_QUERIES_PER_INTENT="${k_per_intent}"

  echo "Cyclic generation enabled: run_id=${RUN_ID} total=${MAX_GENERATION_TOTAL} cycle_size=${GENERATION_CYCLE_SIZE} k_queries_per_intent=${K_QUERIES_PER_INTENT}"
  local cycle=0
  while true; do
    local q r d g floor target
    q="$(jsonl_count "${queries_path}")"
    r="$(jsonl_count "${responses_path}")"
    d="$(jsonl_count "${domains_path}")"
    g="$(jsonl_count "${genui_path}")"
    if (( q >= MAX_GENERATION_TOTAL && r >= MAX_GENERATION_TOTAL && g >= MAX_GENERATION_TOTAL )); then
      echo "Cyclic generation complete: queries=${q} responses=${r} domains=${d} genui=${g}"
      break
    fi

    floor="${q}"
    if (( r < floor )); then floor="${r}"; fi
    if (( g < floor )); then floor="${g}"; fi
    target=$(( floor + GENERATION_CYCLE_SIZE ))
    if (( target > MAX_GENERATION_TOTAL )); then
      target="${MAX_GENERATION_TOTAL}"
    fi
    cycle=$(( cycle + 1 ))
    echo "Cyclic generation cycle=${cycle} target=${target} current queries=${q} responses=${r} domains=${d} genui=${g}"

    ensure_stage_count 1 "${queries_path}" "${target}"
    ensure_stage_count 2 "${responses_path}" "${target}"
    ensure_stage_count 3 "${genui_path}" "${target}"
  done
}

if [[ -n "${MAX_GENERATION_TOTAL}" && ( "${STAGE}" = "all" || "${STAGE}" = "cyclic" ) ]]; then
  run_cyclic_generation
  exit 0
fi

case "${STAGE}" in
  1)
    run_stage 1
    ;;
  2)
    run_stage 2
    ;;
  3)
    run_stage 3 --genui_batch_size "${STAGE3_BATCH_SIZE}"
    ;;
  4)
    run_stage 4 --render_workers "${RENDER_WORKERS}"
    ;;
  5)
    run_stage 5
    ;;
  all)
    run_stage 1
    run_stage 2
    run_stage 3 --genui_batch_size "${STAGE3_BATCH_SIZE}"
    run_stage 4 --render_workers "${RENDER_WORKERS}"
    run_stage 5
    ;;
  cyclic)
    run_cyclic_generation
    ;;
  *)
    echo "Unknown STAGE=${STAGE}; use 1, 2, 3, 4, 5, all, or cyclic" >&2
    exit 1
    ;;
esac
