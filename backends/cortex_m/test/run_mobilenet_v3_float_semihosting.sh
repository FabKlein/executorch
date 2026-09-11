#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
et_root_dir=$(cd "${script_dir}/../../.." && pwd)
et_root_dir=$(realpath "${et_root_dir}")

mode="executorch"
dtype="float32"
target="ethos-u55-128"
timeout="3600"
toolchain="arm-none-eabi-gcc"
et_build_root=""
toolchain_bin=""
cmsis_nn_local_path=""
sample_index=""
sample_seed="0"
verbose="0"

default_build_root_for_toolchain() {
  case "${1}" in
    arm-none-eabi-gcc) printf '%s' "${et_root_dir}/arm_test_gcc15" ;;
    armclang) printf '%s' "${et_root_dir}/arm_test_ac6" ;;
    clang) printf '%s' "${et_root_dir}/arm_test_clang" ;;
    *)
      echo "Unsupported toolchain: ${1}" >&2
      exit 1
      ;;
  esac
}

check_toolchain() {
  case "${toolchain}" in
    arm-none-eabi-gcc)
      if ! command -v arm-none-eabi-gcc >/dev/null 2>&1; then
        echo "arm-none-eabi-gcc not found. GCC >= 14 is required for Cortex-M float flows. Use --toolchain_bin=/path/to/bin or update PATH." >&2
        exit 1
      fi
      local major
      major="$(arm-none-eabi-gcc -dumpversion | cut -d. -f1)"
      if [[ -z "${major}" || "${major}" -lt 14 ]]; then
        echo "Detected arm-none-eabi-gcc $(arm-none-eabi-gcc -dumpversion), but GCC >= 14 is required for Cortex-M float flows." >&2
        exit 1
      fi
      ;;
    armclang)
      if ! command -v armclang >/dev/null 2>&1; then
        echo "armclang not found. Set --toolchain_bin=/path/to/ArmCompilerforEmbedded/bin." >&2
        exit 1
      fi
      ;;
    clang)
      if ! command -v clang >/dev/null 2>&1; then
        echo "clang not found. Set --toolchain_bin=/path/to/plain/LLVM/bin." >&2
        exit 1
      fi
      if [[ -z "${CLANG_TOOLCHAIN_ROOT:-}" ]]; then
        echo "CLANG_TOOLCHAIN_ROOT is not set. Use --toolchain_bin=/path/to/plain/LLVM/bin." >&2
        exit 1
      fi
      ;;
    *)
      echo "Unsupported toolchain: ${toolchain}" >&2
      exit 1
      ;;
  esac
}

resolve_runner_executable() {
  local runner_dir="$1"
  if [[ -x "${runner_dir}/arm_executor_runner" ]]; then
    printf '%s' "${runner_dir}/arm_executor_runner"
  elif [[ -f "${runner_dir}/arm_executor_runner.elf" ]]; then
    printf '%s' "${runner_dir}/arm_executor_runner.elf"
  else
    echo "Runner executable not found in ${runner_dir}" >&2
    exit 1
  fi
}

help() {
  echo "Usage: $(basename "$0") [options]"
  echo "Note: GCC >= 14 is required when --toolchain=arm-none-eabi-gcc."
  echo "Options:"
  echo "  --mode=pytorch|executorch|all      Run host PyTorch only, semihosting only, or both. Default: ${mode}"
  echo "  --dtype=float32|float16            Demo dtype. Default: ${dtype}"
  echo "  --toolchain=arm-none-eabi-gcc|armclang|clang"
  echo "                                     Backend toolchain. Default: ${toolchain}"
  echo "  --sample_index=<INT>               Imagenette validation index. Default: random from --sample_seed"
  echo "  --sample_seed=<INT>                Seed for random sample choice. Default: ${sample_seed}"
  echo "  --target=<TARGET>                  FVP target. Default: ${target}"
  echo "  --timeout=<SEC>                    FVP timeout. Default: ${timeout}"
  echo "  --et_build_root=<PATH>             Build root. Default depends on toolchain."
  echo "  --toolchain_bin=<PATH>             Optional toolchain bin directory to prepend to PATH."
  echo "  --cmsis_nn_local_path=<PATH>       Optional local CMSIS-NN checkout passed to build_executorch.sh."
  echo "  --verbose                          Print full build command output."
  exit 0
}

for arg in "$@"; do
  case $arg in
    -h|--help) help ;;
    --mode=*) mode="${arg#*=}" ;;
    --dtype=*) dtype="${arg#*=}" ;;
    --toolchain=*) toolchain="${arg#*=}" ;;
    --sample_index=*) sample_index="${arg#*=}" ;;
    --sample_seed=*) sample_seed="${arg#*=}" ;;
    --target=*) target="${arg#*=}" ;;
    --timeout=*) timeout="${arg#*=}" ;;
    --et_build_root=*) et_build_root="${arg#*=}" ;;
    --toolchain_bin=*) toolchain_bin="${arg#*=}" ;;
    --cmsis_nn_local_path=*) cmsis_nn_local_path="${arg#*=}" ;;
    --verbose) verbose="1" ;;
    *)
      echo "Unknown argument: ${arg}"
      exit 1
      ;;
  esac
done

if [[ -z "${et_build_root}" ]]; then
  et_build_root="$(default_build_root_for_toolchain "${toolchain}")"
fi

run_dir="${et_build_root}/mobilenet_v3_small_${dtype}_semihosting_run"
runner_dir="${et_build_root}/mobilenet_v3_small_semihosting_runner"
export_script="backends/cortex_m/test/models/export_float_mobilenet_v3_demo.py"

cd "${et_root_dir}"
source venv/bin/activate
caller_pythonpath="${PYTHONPATH:-}"
if [[ -n "${toolchain_bin}" ]]; then
  export PATH="${toolchain_bin}:${PATH}"
fi
source examples/arm/arm-scratch/setup_path.sh
if [[ -n "${toolchain_bin}" ]]; then
  export PATH="${toolchain_bin}:${PATH}"
fi
if [[ "${toolchain}" == "armclang" ]]; then
  export AC6_TOOLCHAIN="${toolchain_bin}"
elif [[ "${toolchain}" == "clang" ]]; then
  export CLANG_TOOLCHAIN_ROOT="${toolchain_bin}"
fi
export PYTHONPATH="${et_root_dir}/src${caller_pythonpath:+:${caller_pythonpath}}"
source backends/cortex_m/test/float_backend_env.sh
check_toolchain
set_cortex_m_float_backend_for_dtype "${dtype}"
export CORTEX_M_TEST_VERBOSE="${verbose}"

mkdir -p "${et_build_root}" "${run_dir}"
rm -f "${run_dir}"/out-*.bin "${run_dir}"/i*.bin "${run_dir}"/expected-*.bin "${run_dir}"/run_meta.json "${run_dir}"/selected_ops.yaml

sample_args=(--sample_seed "${sample_seed}")
if [[ -n "${sample_index}" ]]; then
  sample_args+=(--sample_index "${sample_index}")
fi

if [[ "${mode}" == "pytorch" ]]; then
  python "${export_script}" -o "${run_dir}" --dtype "${dtype}" --mode pytorch "${sample_args[@]}"
  exit 0
fi

build_executorch_args=(
  --toolchain="${toolchain}"
  --et_build_root="${et_build_root}"
  "${CORTEX_M_FLOAT_BUILD_ARGS[@]}"
)
if [[ -n "${cmsis_nn_local_path}" ]]; then
  build_executorch_args+=(--cmsis_nn_local_path="${cmsis_nn_local_path}")
fi
run_logged_command "Build ExecuTorch" \
  backends/arm/scripts/build_executorch.sh "${build_executorch_args[@]}"
set_cortex_m_float_capabilities_from_build "${et_build_root}" "${toolchain}"

python "${export_script}" -o "${run_dir}" --dtype "${dtype}" --mode prepare "${sample_args[@]}"

pte_file="${run_dir}/mobilenet_v3_small_$([[ "${dtype}" == "float32" ]] && printf '%s' 'f32' || printf '%s' 'f16')_demo.pte"

python codegen/tools/gen_oplist.py \
  --model_file_path="${pte_file}" \
  --output_path="${run_dir}/selected_ops.yaml" >/dev/null
print_selected_ops_summary "${run_dir}/selected_ops.yaml"

export ET_MV3_SELECTED_OPS_YAML="${run_dir}/selected_ops.yaml"
select_ops_list="$(python - <<'PY'
import os
from pathlib import Path
import yaml

data = yaml.safe_load(Path(os.environ["ET_MV3_SELECTED_OPS_YAML"]).read_text())
ops = list(data.get("operators", {}).keys())
print(",".join(sorted(ops)))
PY
)"

runner_build_args=(
  --toolchain="${toolchain}"
  --et_build_root="${et_build_root}"
  --pte=semihosting
  --target="${target}"
  --system_config=Ethos_U55_High_End_Embedded
  --memory_mode=Shared_Sram
  --output="${runner_dir}"
  --select_ops_list="${select_ops_list}"
  "${CORTEX_M_FLOAT_BUILD_ARGS[@]}"
  --extra_build_flags=-DET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE=83886080
)
if [[ -n "${cmsis_nn_local_path}" ]]; then
  runner_build_args+=(--cmsis_nn_local_path="${cmsis_nn_local_path}")
fi
run_logged_command "Build executor runner" \
  backends/arm/scripts/build_executor_runner.sh "${runner_build_args[@]}"

runner_exe="$(resolve_runner_executable "${runner_dir}")"

fvp_cmd=(
  FVP_Corstone_SSE-300_Ethos-U55
  -C ethosu.num_macs=128
  -C mps3_board.visualisation.disable-visualisation=1
  -C mps3_board.telnetterminal0.start_telnet=0
  -C mps3_board.uart0.out_file=-
  -C cpu0.semihosting-enable=1
  -C cpu0.semihosting-stack_base=0
  -C cpu0.semihosting-heap_limit=0
  -C "cpu0.semihosting-cwd=${run_dir}"
  -C "ethosu.extra_args=--fast"
  -C "cpu0.semihosting-cmd_line=executor_runner -m $(basename "${pte_file}") -o out -i i0.bin"
  -a "${runner_exe}"
  --timelimit "${timeout}"
)
echo "FVP command:"
print_command "${fvp_cmd[@]}"
"${fvp_cmd[@]}" | tee "${run_dir}/run.log"

export ET_MV3_RUN_DIR="${run_dir}"
export ET_MV3_DTYPE="${dtype}"
python - <<'PY'
import json
import numpy as np
import os
from pathlib import Path
from torchvision.models import MobileNet_V3_Small_Weights

run_dir = Path(os.environ["ET_MV3_RUN_DIR"])
dtype_name = os.environ["ET_MV3_DTYPE"]
dtype = np.float16 if dtype_name == "float16" else np.float32
meta = json.loads((run_dir / "run_meta.json").read_text())
expected = np.fromfile(run_dir / "expected-0.bin", dtype=dtype).astype(np.float32)
got = np.fromfile(run_dir / "out-0.bin", dtype=dtype).astype(np.float32)

np.save(run_dir / "ref_output.npy", expected)
np.save(run_dir / "got_output.npy", got)

categories = MobileNet_V3_Small_Weights.DEFAULT.meta["categories"]
fvp_top1 = categories[int(got.argmax())]

def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def print_topk(name: str, logits: np.ndarray, k: int = 10) -> None:
    probs = softmax(logits)
    topk_idx = np.argsort(-probs)[:k]
    print(f"{name} top-{k} probabilities:")
    for rank, idx in enumerate(topk_idx.tolist(), start=1):
        print(f"  {rank:2d}. {categories[idx]}: {probs[idx]:.6f}")

print(f"Sample index     : {meta['sample_index']}")
print(f"True label       : {meta['true_label']}")
print(f"PyTorch top-1    : {meta['predicted_label']}")
print(f"FVP top-1        : {fvp_top1}")
print(f"sum(got)         : {got.sum()}")
print(f"sum(ref)         : {expected.sum()}")
print(f"max_abs          : {np.max(np.abs(got - expected))}")
print(f"mean_abs         : {np.mean(np.abs(got - expected))}")
print_topk("PyTorch", expected)
print_topk("FVP", got)
print(f"Saved arrays     : {run_dir/'ref_output.npy'} , {run_dir/'got_output.npy'}")
PY
