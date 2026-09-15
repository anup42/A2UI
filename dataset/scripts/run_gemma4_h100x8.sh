#!/usr/bin/env bash
set -euo pipefail

# CPU-safe by default: print the plan. Explicit `servers` or `generate` runs
# belong on the Linux 8xH100 server only. No training command is invoked.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-plan}"
export H100_TP_SIZE="${H100_TP_SIZE:-2}"
case "${H100_TP_SIZE}" in 1|2|4|8) ;; *) echo 'H100_TP_SIZE must be 1, 2, 4 or 8.' >&2; exit 2 ;; esac
replicas=$((8 / H100_TP_SIZE))
export REPLICA_TP_SIZE="${H100_TP_SIZE}"
export REPLICA_GPU_IDS="${REPLICA_GPU_IDS:-0,1,2,3,4,5,6,7}"
read -r -a selected_gpus <<< "${REPLICA_GPU_IDS//,/ }"
if (( ${#selected_gpus[@]} != 8 )); then
  echo 'This profile requires exactly eight GPU IDs in REPLICA_GPU_IDS.' >&2; exit 2
fi
export REPLICA_ENV_FILE="${REPLICA_ENV_FILE:-/tmp/a2ui_gemma4_h100x8_tp${H100_TP_SIZE}.env}"
export VLLM_ENDPOINTS_ENV_FILE="${REPLICA_ENV_FILE}"
export REPLICA_LOG_DIR="${REPLICA_LOG_DIR:-/tmp/a2ui_gemma4_h100x8}"
export A2UI_VLLM_SERVER_SCRIPT="${A2UI_VLLM_SERVER_SCRIPT:-${SCRIPT_DIR}/run_gemma4_vllm_python.sh}"
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_BASE_PORT="${VLLM_BASE_PORT:-8000}"
export VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
export VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"
export VLLM_ASYNC_SCHEDULING="${VLLM_ASYNC_SCHEDULING:-1}"
export VLLM_LIMIT_MM_PER_PROMPT="${VLLM_LIMIT_MM_PER_PROMPT:-'{"image":0,"audio":0,"video":0}'}"
export VLLM_CLEAN_STALE_PROCESSES="${VLLM_CLEAN_STALE_PROCESSES:-0}"
export VLLM_MAX_RESTARTS="${VLLM_MAX_RESTARTS:-2}"
export GEMMA4_SPECULATIVE_MODE="${GEMMA4_SPECULATIVE_MODE:-off}"
export GEMMA4_ENABLE_REASONING="${GEMMA4_ENABLE_REASONING:-0}"
export LOCAL_VLLM_ENABLE_THINKING="${LOCAL_VLLM_ENABLE_THINKING:-${GEMMA4_ENABLE_REASONING}}"
export LOCAL_VLLM_SEND_CHAT_TEMPLATE_KWARGS="${LOCAL_VLLM_SEND_CHAT_TEMPLATE_KWARGS:-1}"
export LOCAL_VLLM_REQUESTS_PER_SERVER="${LOCAL_VLLM_REQUESTS_PER_SERVER:-32}"
if ! [[ "${LOCAL_VLLM_REQUESTS_PER_SERVER}" =~ ^[1-9][0-9]*$ ]]; then
  echo 'LOCAL_VLLM_REQUESTS_PER_SERVER must be positive.' >&2; exit 2
fi
export LOCAL_VLLM_BATCH_PARALLELISM="${LOCAL_VLLM_BATCH_PARALLELISM:-$((replicas * LOCAL_VLLM_REQUESTS_PER_SERVER))}"
export LOCAL_VLLM_PARALLEL_REQUESTS="${LOCAL_VLLM_BATCH_PARALLELISM}"
export LOCAL_VLLM_TIMEOUT_SECONDS="${LOCAL_VLLM_TIMEOUT_SECONDS:-1800}"
export LOCAL_VLLM_RETRY_MAX_SECONDS="${LOCAL_VLLM_RETRY_MAX_SECONDS:-180}"
export LOCAL_VLLM_RETRY_RESULT_ERRORS="${LOCAL_VLLM_RETRY_RESULT_ERRORS:-0}"
export A2UI_MAX_ATTEMPTS="${A2UI_MAX_ATTEMPTS:-2}"
export A2UI_MAX_REPAIR_ATTEMPTS="${A2UI_MAX_REPAIR_ATTEMPTS:-1}"
export STAGE3_FINAL_REGEN_ATTEMPTS="${STAGE3_FINAL_REGEN_ATTEMPTS:-1}"
export RATE_LIMIT_QPS="${RATE_LIMIT_QPS:-0}"
export A2UI_CALL_SLEEP_SECONDS="${A2UI_CALL_SLEEP_SECONDS:-0}"
export STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-8}"
export A2UI_STAGE1_INTENT_BATCH_SIZE="${A2UI_STAGE1_INTENT_BATCH_SIZE:-32}"
export A2UI_STAGE1_INTENT_CYCLE_SIZE="${A2UI_STAGE1_INTENT_CYCLE_SIZE:-0}"
export STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-${LOCAL_VLLM_BATCH_PARALLELISM}}"
export STAGE2_RESPONSE_BATCH_SIZE="${STAGE2_RESPONSE_BATCH_SIZE:-1}"
export STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-${LOCAL_VLLM_BATCH_PARALLELISM}}"
export A2UI_QUERY_MAX_TOKENS="${A2UI_QUERY_MAX_TOKENS:-4096}"
export A2UI_RESPONSE_MAX_TOKENS="${A2UI_RESPONSE_MAX_TOKENS:-4096}"
export A2UI_GENUI_MAX_TOKENS="${A2UI_GENUI_MAX_TOKENS:-4096}"
export LOCAL_VLLM_MAX_OUTPUT_TOKENS="${LOCAL_VLLM_MAX_OUTPUT_TOKENS:-8192}"
export STAGE3_CONTEXT_SAFETY_TOKENS="${STAGE3_CONTEXT_SAFETY_TOKENS:-512}"
for token_setting in VLLM_MAX_MODEL_LEN A2UI_GENUI_MAX_TOKENS STAGE3_CONTEXT_SAFETY_TOKENS; do
  if ! [[ "${!token_setting}" =~ ^[0-9]+$ ]]; then
    echo "${token_setting} must be a nonnegative integer." >&2; exit 2
  fi
done
prompt_budget=$((VLLM_MAX_MODEL_LEN - A2UI_GENUI_MAX_TOKENS - STAGE3_CONTEXT_SAFETY_TOKENS))
if (( prompt_budget <= 0 )); then
  echo 'Context must exceed the Stage 3 output budget plus safety reserve.' >&2; exit 2
fi
export A2UI_GENUI_PROMPT_MAX_TOKENS="${A2UI_GENUI_PROMPT_MAX_TOKENS:-${prompt_budget}}"
export LOCAL_STAGE3_PROMPT_MAX_TOKENS="${LOCAL_STAGE3_PROMPT_MAX_TOKENS:-${A2UI_GENUI_PROMPT_MAX_TOKENS}}"
export STAGE3_RESPECT_CONFIG_PROMPT_MAX="${STAGE3_RESPECT_CONFIG_PROMPT_MAX:-1}"
export A2UI_DISABLE_SSL_VERIFY="${A2UI_DISABLE_SSL_VERIFY:-0}"
export DATASET_OFFLINE_MODE="${DATASET_OFFLINE_MODE:-1}"
export STAGE2_KEEP_UNRESOLVED_MEDIA="${STAGE2_KEEP_UNRESOLVED_MEDIA:-1}"
export STAGE2_REAL_ASSET_RETRY_ENABLED="${STAGE2_REAL_ASSET_RETRY_ENABLED:-0}"
export STAGE="${STAGE:-cyclic}"
export GENERATION_CYCLE_SIZE="${GENERATION_CYCLE_SIZE:-2000}"
export GENUI_STAGE3_AGGREGATE_EVERY="${GENUI_STAGE3_AGGREGATE_EVERY:-1000}"
export MAX_GENERATION_TOTAL="${MAX_GENERATION_TOTAL:-10000}"

print_plan() {
  printf '8xH100 80GB candidate profile (benchmark before scaling)\n'
  printf 'replicas=%s tensor_parallel_per_replica=%s\n' "${replicas}" "${H100_TP_SIZE}"
  printf 'dtype=%s context=%s sequences_per_server=%s scheduled_tokens=%s\n' "${VLLM_DTYPE}" "${VLLM_MAX_MODEL_LEN}" "${VLLM_MAX_NUM_SEQS}" "${VLLM_MAX_NUM_BATCHED_TOKENS}"
  printf 'client_parallelism=%s requests_per_server=%s rate_limit=%s sleep=%s\n' "${LOCAL_VLLM_BATCH_PARALLELISM}" "${LOCAL_VLLM_REQUESTS_PER_SERVER}" "${RATE_LIMIT_QPS}" "${A2UI_CALL_SLEEP_SECONDS}"
  printf 'queries_per_prompt=%s stage2_query_wave=%s responses_per_query=%s stage3_wave=%s\n' "${STAGE1_BATCH_SIZE}" "${STAGE2_BATCH_SIZE}" "${STAGE2_RESPONSE_BATCH_SIZE}" "${STAGE3_BATCH_SIZE}"
  printf 'thinking=%s speculative=%s assets_offline=%s keep_asset_references=%s\n' "${GEMMA4_ENABLE_REASONING}" "${GEMMA4_SPECULATIVE_MODE}" "${DATASET_OFFLINE_MODE}" "${STAGE2_KEEP_UNRESOLVED_MEDIA}"
  printf 'output_budgets: query=%s response=%s ui=%s; target=%s cycle=%s\n' "${A2UI_QUERY_MAX_TOKENS}" "${A2UI_RESPONSE_MAX_TOKENS}" "${A2UI_GENUI_MAX_TOKENS}" "${MAX_GENERATION_TOTAL}" "${GENERATION_CYCLE_SIZE}"
  printf 'stage3_prompt_cap=%s safety_reserve=%s repairs=%s final_regenerations=%s transport_attempts=%s\n' "${A2UI_GENUI_PROMPT_MAX_TOKENS}" "${STAGE3_CONTEXT_SAFETY_TOKENS}" "${A2UI_MAX_REPAIR_ATTEMPTS}" "${STAGE3_FINAL_REGEN_ATTEMPTS}" "${A2UI_MAX_ATTEMPTS}"
  printf 'Server shell: GEMMA4_MODEL_PATH=/absolute/model/path bash %s servers\n' "${BASH_SOURCE[0]}"
  printf 'Client shell: RUN_ID=unique_run_id bash %s generate\n' "${BASH_SOURCE[0]}"
}

validate_endpoint_count() {
  local endpoint_list="${LOCAL_VLLM_ENDPOINTS//,/ }"
  endpoint_list="${endpoint_list//;/ }"
  local configured_endpoints=()
  read -r -a configured_endpoints <<< "${endpoint_list}"
  if (( ${#configured_endpoints[@]} != replicas )); then
    echo "Expected ${replicas} endpoints for TP=${H100_TP_SIZE}; found ${#configured_endpoints[@]}. Clear stale LOCAL_VLLM_ENDPOINTS or use the matching replica environment file." >&2
    exit 2
  fi
}
if [[ -n "${LOCAL_VLLM_ENDPOINTS:-}" ]]; then validate_endpoint_count; fi

case "${MODE}" in
  plan|--dry-run) print_plan; exit 0 ;;
  servers|generate) ;;
  *) echo 'Use plan, servers, or generate.' >&2; exit 2 ;;
esac
if [[ "$(uname -s)" != "Linux" ]]; then
  echo 'Execution is supported only on the Linux GPU server; use plan on this PC.' >&2
  exit 2
fi
print_plan
if [[ "${MODE}" == "servers" ]]; then
  if [[ -z "${GEMMA4_MODEL_PATH:-}${GEMMA4_MODEL_ROOT:-}${MODEL_ROOT:-}" ]]; then
    echo 'Set GEMMA4_MODEL_PATH or MODEL_ROOT to the existing local model.' >&2; exit 2
  fi
  exec bash "${SCRIPT_DIR}/run_gemma4_vllm_replicas.sh"
fi
if [[ -z "${RUN_ID:-}" ]]; then
  echo 'Set a distinct RUN_ID for this configuration; existing audited data must remain unchanged.' >&2; exit 2
fi
if [[ ! -f "${REPLICA_ENV_FILE}" && -z "${LOCAL_VLLM_ENDPOINTS:-}" ]]; then
  echo "Start the servers with the same H100_TP_SIZE first; missing ${REPLICA_ENV_FILE}." >&2; exit 2
fi
if [[ -z "${LOCAL_VLLM_ENDPOINTS:-}" ]]; then
  # Generated by the selected replica supervisor, with topology-specific path.
  source "${REPLICA_ENV_FILE}"
fi
validate_endpoint_count
exec bash "${SCRIPT_DIR}/run_gemma4_dataset_stages_multi_vllm.sh"
