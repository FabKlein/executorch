#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set_cortex_m_float_backend_for_dtype() {
  local dtype="$1"
  case "${dtype}" in
    float32)
      CORTEX_M_FLOAT_BUILD_ARGS=(--cmsis_nn_enable_f32)
      ;;
    float16)
      CORTEX_M_FLOAT_BUILD_ARGS=(--cmsis_nn_enable_f16)
      ;;
    both|all)
      CORTEX_M_FLOAT_BUILD_ARGS=(--cmsis_nn_enable_f32 --cmsis_nn_enable_f16)
      ;;
    *)
      echo "Unsupported Cortex-M float dtype selection: ${dtype}" >&2
      return 1
      ;;
  esac
}

cortex_m_build_dir_for_toolchain() {
  local et_build_root="$1"
  local toolchain="$2"
  case "${toolchain}" in
    arm-none-eabi-gcc) printf '%s' "${et_build_root}/cmake-out" ;;
    armclang) printf '%s' "${et_build_root}/cmake-out-armclang" ;;
    clang) printf '%s' "${et_build_root}/cmake-out-clang" ;;
    *)
      echo "Unsupported Cortex-M toolchain for float capabilities: ${toolchain}" >&2
      return 1
      ;;
  esac
}

set_cortex_m_float_capabilities_from_build() {
  local et_build_root="$1"
  local toolchain="$2"
  local build_dir
  local capabilities_file

  build_dir="$(cortex_m_build_dir_for_toolchain "${et_build_root}" "${toolchain}")"
  capabilities_file="${build_dir}/backends/cortex_m/float_capabilities.json"
  if [[ ! -f "${capabilities_file}" ]]; then
    echo "Missing Cortex-M float capability artifact: ${capabilities_file}" >&2
    echo "Build ExecuTorch first so the Cortex-M backend can publish its float support." >&2
    return 1
  fi

  export EXECUTORCH_CORTEX_M_FLOAT_CAPABILITIES_FILE="${capabilities_file}"
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_logged_command() {
  local label="$1"
  shift

  if [[ "${CORTEX_M_TEST_VERBOSE:-0}" == "1" ]]; then
    echo "${label}:"
    print_command "$@"
    "$@"
    return
  fi

  local log_file
  log_file="$(mktemp)"
  echo "${label}..."
  if "$@" >"${log_file}" 2>&1; then
    rm -f "${log_file}"
    echo "${label}: done"
    return
  fi

  local status=$?
  echo "${label}: failed" >&2
  cat "${log_file}" >&2
  rm -f "${log_file}"
  return "${status}"
}

print_selected_ops_summary() {
  local selected_ops_yaml="$1"
  ET_SELECTED_OPS_YAML="${selected_ops_yaml}" python - <<'PY'
import os
from pathlib import Path
import yaml

path = Path(os.environ["ET_SELECTED_OPS_YAML"])
data = yaml.safe_load(path.read_text()) or {}
ops = sorted((data.get("operators") or {}).keys())

def print_group(title: str, prefix: str) -> None:
    group = [op for op in ops if op.startswith(prefix)]
    print(f"{title}:")
    if not group:
        print("  (none)")
        return
    for op in group:
        print(f"  {op}")

print_group("Lowered Cortex-M operators", "cortex_m::")
print_group("Residual aten operators", "aten::")
print_group("Residual dim_order operators", "dim_order_ops::")
PY
}
