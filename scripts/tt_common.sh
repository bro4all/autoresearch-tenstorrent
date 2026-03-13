#!/usr/bin/env bash
set -euo pipefail

tt_in_container() {
  [[ -f "/.dockerenv" ]]
}

tt_list_timeout_seconds() {
  printf '%s\n' "${AUTORESEARCH_TT_LIST_TIMEOUT_SECS:-10}"
}

tt_set_visible_devices() {
  if [[ -z "${TT_VISIBLE_DEVICES:-}" && -n "${TT_METAL_VISIBLE_DEVICES:-}" ]]; then
    export TT_VISIBLE_DEVICES="${TT_METAL_VISIBLE_DEVICES}"
  fi
  export TT_VISIBLE_DEVICES="${TT_VISIBLE_DEVICES:-${AUTORESEARCH_TT_VISIBLE_DEVICES:-0}}"
  export TT_METAL_VISIBLE_DEVICES="${TT_VISIBLE_DEVICES}"
  export PJRT_DEVICE="${PJRT_DEVICE:-TT}"
  export XLA_STABLEHLO_COMPILE="${XLA_STABLEHLO_COMPILE:-1}"
}

tt_reset_device_id() {
  local first
  IFS=',' read -r first _ <<<"${TT_VISIBLE_DEVICES}"
  printf '%s\n' "${AUTORESEARCH_TT_RESET_DEVICE:-${first:-0}}"
}

tt_reset_and_wait() {
  if ! command -v tt-smi >/dev/null 2>&1; then
    echo "tt-smi is not available in this environment; skipping TT device reset." >&2
    return 1
  fi
  local device wait_secs smi_timeout
  device="$(tt_reset_device_id)"
  wait_secs="${AUTORESEARCH_TT_RESET_WAIT_SECS:-30}"
  smi_timeout="${AUTORESEARCH_TT_SMI_TIMEOUT_SECS:-30}"
  echo "Resetting Tenstorrent device ${device} and waiting ${wait_secs}s for link retraining..." >&2
  timeout "${smi_timeout}" tt-smi --reset "${device}" >/dev/null
  export AUTORESEARCH_TT_HOST_RESET_DONE=1
  sleep "${wait_secs}"
}

tt_maybe_host_reset() {
  if tt_in_container; then
    return
  fi
  if [[ "${AUTORESEARCH_TT_RESET_BEFORE_RUN:-0}" != "1" ]]; then
    return
  fi
  tt_reset_and_wait
}

tt_management_healthy() {
  if ! command -v tt-smi >/dev/null 2>&1; then
    echo "tt-smi is not available in this environment." >&2
    return 1
  fi
  if [[ ! -e /dev/tenstorrent/0 ]]; then
    echo "/dev/tenstorrent/0 is not present on this host." >&2
    return 1
  fi
  local log_file status
  log_file="$(mktemp)"
  set +e
  timeout "$(tt_list_timeout_seconds)" tt-smi -ls >"${log_file}" 2>&1
  status=$?
  set -e
  if [[ "${status}" -eq 0 ]] && grep -q "Wormhole" "${log_file}"; then
    rm -f "${log_file}"
    return 0
  fi
  cat "${log_file}" >&2
  rm -f "${log_file}"
  return 1
}

tt_host_preflight() {
  if tt_in_container; then
    return
  fi
  tt_set_visible_devices
  if tt_management_healthy; then
    return
  fi
  if [[ "${AUTORESEARCH_TT_PREFLIGHT_RESET_ON_FAIL:-1}" != "1" ]]; then
    echo "TT preflight failed and reset-on-fail is disabled." >&2
    return 1
  fi
  local retries attempt
  retries="${AUTORESEARCH_TT_PREFLIGHT_RETRIES:-1}"
  attempt=0
  while true; do
    echo "TT host preflight failed; attempting reset recovery ${attempt}/${retries}." >&2
    tt_reset_and_wait || return 1
    if tt_management_healthy; then
      return
    fi
    if [[ "${attempt}" -ge "${retries}" ]]; then
      echo "TT host preflight failed after reset retries." >&2
      return 1
    fi
    attempt=$((attempt + 1))
  done
}

tt_log_has_recoverable_init_error() {
  local log_file
  log_file="$1"
  grep -Eq \
    'Read unexpected run_mailbox value from core|Timeout waiting for Ethernet core service remote IO request|Fabric Router Sync: Timeout|Detected dispatch kernels still running but failed to complete an early exit' \
    "${log_file}"
}

tt_run_with_recovery() {
  local max_retries attempt log_file status
  max_retries="${AUTORESEARCH_TT_INIT_RETRIES:-1}"
  attempt=0
  while true; do
    log_file="$(mktemp)"
    set +e
    "$@" 2>&1 | tee "${log_file}"
    status="${PIPESTATUS[0]}"
    set -e
    if [[ "${status}" -eq 0 ]]; then
      rm -f "${log_file}"
      return 0
    fi
    if [[ "${attempt}" -ge "${max_retries}" ]] || ! tt_log_has_recoverable_init_error "${log_file}"; then
      rm -f "${log_file}"
      return "${status}"
    fi
    rm -f "${log_file}"
    attempt=$((attempt + 1))
    echo "TT init failed with a recoverable dispatch/fabric error; retry ${attempt}/${max_retries} after reset." >&2
    if ! tt_reset_and_wait; then
      echo "Unable to reset TT device here; returning the original TT init failure." >&2
      return "${status}"
    fi
  done
}
