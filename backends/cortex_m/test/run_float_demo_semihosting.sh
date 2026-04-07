#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
et_root_dir=$(cd "${script_dir}/../../.." && pwd)
et_root_dir=$(realpath "${et_root_dir}")

demo="softmax"
dtype="float16"
target="ethos-u55-128"
timeout="120"
et_build_root="${et_root_dir}/arm_test_gcc15"
toolchain_bin=""
cmsis_nn_local_path=""
runner_dir="${et_build_root}/arm_semihosting_executor_runner_corstone-300"
conv_input_scale="1.0"

check_arm_gcc() {
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
}

help() {
  echo "Usage: $(basename "$0") [options]"
  echo "Note: GCC >= 14 is required for Cortex-M float flows."
  echo "Options:"
  echo "  --demo=softmax|transpose|combo|avg_pool|max_pool|linear|activation_tanh|activation_hardswish|conv_x3|conv_bias|depthwise_padding|depthwise_bias|conv_transpose_basic|conv_transpose_bias   Demo to run. Default: ${demo}"
  echo "  --dtype=float16|float32          Demo dtype. Default: ${dtype}"
  echo "  --target=<TARGET>                FVP target. Default: ${target}"
  echo "  --timeout=<SEC>                  FVP timeout. Default: ${timeout}"
  echo "  --et_build_root=<PATH>           GCC build root. Default: ${et_build_root}"
  echo "  --toolchain_bin=<PATH>           Optional GCC toolchain bin directory to prepend to PATH."
  echo "  --cmsis_nn_local_path=<PATH>     Optional local CMSIS-NN checkout passed to build_executorch.sh."
  echo "  --conv_input_scale=<FLOAT>       Input scale for conv demos. Default: ${conv_input_scale}"
  exit 0
}

for arg in "$@"; do
  case $arg in
    -h|--help) help ;;
    --demo=*) demo="${arg#*=}" ;;
    --dtype=*) dtype="${arg#*=}" ;;
    --target=*) target="${arg#*=}" ;;
    --timeout=*) timeout="${arg#*=}" ;;
    --et_build_root=*) et_build_root="${arg#*=}" ;;
    --toolchain_bin=*) toolchain_bin="${arg#*=}" ;;
    --cmsis_nn_local_path=*) cmsis_nn_local_path="${arg#*=}" ;;
    --conv_input_scale=*) conv_input_scale="${arg#*=}" ;;
    *)
      echo "Unknown argument: ${arg}"
      exit 1
      ;;
  esac
done

if [[ "${dtype}" != "float16" && "${dtype}" != "float32" ]]; then
  echo "Unsupported dtype: ${dtype}"
  exit 1
fi

case "${demo}" in
  softmax)
    export_script="backends/cortex_m/test/models/export_float_softmax_demo.py"
    pte_base="softmax"
    ;;
  avg_pool)
    export_script="backends/cortex_m/test/models/export_float_avg_pool2d_demo.py"
    pte_base="avg_pool2d"
    ;;
  max_pool)
    export_script="backends/cortex_m/test/models/export_float_max_pool2d_demo.py"
    pte_base="max_pool2d"
    ;;
  linear)
    export_script="backends/cortex_m/test/models/export_float_linear_demo.py"
    pte_base="linear"
    ;;
  activation_tanh)
    export_script="backends/cortex_m/test/models/export_float_activation_demo.py"
    pte_base="activation_tanh"
    ;;
  activation_hardswish)
    export_script="backends/cortex_m/test/models/export_float_activation_demo.py"
    pte_base="activation_hardswish"
    ;;
  conv_x3|conv_bias|depthwise_padding|depthwise_bias)
    export_script="backends/cortex_m/test/models/export_float_conv_demo.py"
    pte_base="${demo}"
    ;;
  conv_transpose_basic|conv_transpose_bias)
    export_script="backends/cortex_m/test/models/export_float_conv_transpose_demo.py"
    pte_base="${demo}"
    ;;
  transpose)
    export_script="backends/cortex_m/test/models/export_float_transpose_demo.py"
    pte_base="transpose"
    ;;
  combo)
    export_script="backends/cortex_m/test/models/export_float_elementwise_combo_demo.py"
    pte_base="float_elementwise_combo"
    ;;
  *)
    echo "Unsupported demo: ${demo}"
    exit 1
    ;;
esac

runner_dir="${et_build_root}/arm_semihosting_executor_runner_corstone-300"
run_dir="${et_build_root}/${demo}_${dtype}_semihosting_run"

cd "${et_root_dir}"
source venv/bin/activate
if [[ -n "${toolchain_bin}" ]]; then
  export PATH="${toolchain_bin}:${PATH}"
fi
source examples/arm/arm-scratch/setup_path.sh
if [[ -n "${toolchain_bin}" ]]; then
  export PATH="${toolchain_bin}:${PATH}"
fi
export PYTHONPATH="${et_root_dir}/src"
source backends/cortex_m/test/float_backend_env.sh
check_arm_gcc
set_cortex_m_float_backend_for_dtype "${dtype}"

mkdir -p "${et_build_root}" "${run_dir}"
rm -f "${run_dir}"/out-*.bin "${run_dir}"/i*.bin "${run_dir}"/expected-*.bin "${run_dir}"/run_meta.json

build_executorch_args=(
  --et_build_root="${et_build_root}"
  "${CORTEX_M_FLOAT_BUILD_ARGS[@]}"
  --devtools
)
if [[ -n "${cmsis_nn_local_path}" ]]; then
  build_executorch_args+=(--cmsis_nn_local_path="${cmsis_nn_local_path}")
fi
backends/arm/scripts/build_executorch.sh "${build_executorch_args[@]}"
set_cortex_m_float_capabilities_from_build "${et_build_root}" "arm-none-eabi-gcc"

  python "${export_script}" \
  -o "${run_dir}" \
  --dtype "${dtype}" \
  $([[ "${demo}" == "activation_tanh" ]] && printf '%s' "--kind tanh") \
  $([[ "${demo}" == "activation_hardswish" ]] && printf '%s' "--kind hardswish") \
  $([[ "${demo}" == "conv_x3" || "${demo}" == "conv_bias" || "${demo}" == "depthwise_padding" || "${demo}" == "depthwise_bias" ]] && printf '%s' "--variant ${demo} --input_scale ${conv_input_scale}") \
  $([[ "${demo}" == "conv_transpose_basic" ]] && printf '%s' "--variant basic") \
  $([[ "${demo}" == "conv_transpose_bias" ]] && printf '%s' "--variant bias")

suffix="f32"
if [[ "${dtype}" == "float16" ]]; then
  suffix="f16"
fi
pte_file="${run_dir}/${pte_base}_${suffix}_demo.pte"

export ET_RUN_DIR="${run_dir}"
export ET_RUN_DEMO="${demo}"
export ET_RUN_DTYPE="${dtype}"
export ET_CONV_INPUT_SCALE="${conv_input_scale}"

python - <<'PY'
import json
import os
from pathlib import Path

import torch

run_dir = Path(os.environ["ET_RUN_DIR"])
demo = os.environ["ET_RUN_DEMO"]
dtype_name = os.environ["ET_RUN_DTYPE"]
dtype = torch.float16 if dtype_name == "float16" else torch.float32


def write_storage_bytes(path: Path, tensor: torch.Tensor) -> None:
    path.write_bytes(bytes(tensor.untyped_storage()))


def write_channels_last_bytes(path: Path, tensor: torch.Tensor) -> None:
    write_storage_bytes(path, tensor.contiguous(memory_format=torch.channels_last))


meta = {"demo": demo, "dtype": dtype_name, "input_count": 0}
conv_input_scale = float(os.environ.get("ET_CONV_INPUT_SCALE", "1.0"))

if demo == "softmax":
    x = torch.linspace(-4.0, 4.0, steps=16, dtype=dtype)
    out = torch.softmax(x, dim=-1)
    write_storage_bytes(run_dir / "expected-0.bin", out)
elif demo == "transpose":
    x = torch.linspace(-0.5, 0.5, steps=2 * 3 * 4 * 2, dtype=dtype).reshape(2, 3, 4, 2)
    out = x.permute(0, 3, 1, 2).contiguous()
    write_storage_bytes(run_dir / "i0.bin", x.contiguous())
    write_storage_bytes(run_dir / "expected-0.bin", out)
    meta["input_count"] = 1
elif demo == "avg_pool":
    x = torch.linspace(-3.0, 5.0, steps=1 * 2 * 4 * 4, dtype=dtype).reshape(1, 2, 4, 4)
    x = x.contiguous(memory_format=torch.channels_last)
    out = torch.nn.functional.avg_pool2d(
        x,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        ceil_mode=False,
        count_include_pad=False,
    )
    write_storage_bytes(run_dir / "i0.bin", x)
    write_storage_bytes(run_dir / "expected-0.bin", out)
    meta["input_count"] = 1
elif demo == "max_pool":
    x = torch.linspace(-3.0, 5.0, steps=1 * 2 * 4 * 4, dtype=dtype).reshape(1, 2, 4, 4)
    x = x.contiguous(memory_format=torch.channels_last)
    out = torch.nn.functional.max_pool2d(
        x,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        dilation=(1, 1),
        ceil_mode=False,
    )
    write_storage_bytes(run_dir / "i0.bin", x)
    write_storage_bytes(run_dir / "expected-0.bin", out)
    meta["input_count"] = 1
elif demo == "combo":
    x = torch.linspace(-0.75, 0.75, steps=1 * 8 * 5 * 5, dtype=dtype).reshape(1, 8, 5, 5)
    y = torch.linspace(0.25, 1.25, steps=1 * 8 * 5 * 5, dtype=dtype).reshape(1, 8, 5, 5)
    bias = torch.linspace(-0.5, 0.5, steps=8, dtype=dtype).reshape(1, 8, 1, 1)
    x = x.contiguous(memory_format=torch.channels_last)
    y = y.contiguous(memory_format=torch.channels_last)
    bias = bias.contiguous(memory_format=torch.channels_last)
    out = torch.minimum(torch.maximum((x + y) * y, bias), x)
    write_storage_bytes(run_dir / "i0.bin", x)
    write_storage_bytes(run_dir / "i1.bin", y)
    write_storage_bytes(run_dir / "i2.bin", bias)
    write_storage_bytes(run_dir / "expected-0.bin", out)
    meta["input_count"] = 3
elif demo == "linear":
    x = torch.linspace(-1.5, 1.5, steps=2 * 2 * 4, dtype=dtype).reshape(2, 2, 4)
    weight = torch.linspace(-1.0, 1.0, steps=3 * 4, dtype=dtype).reshape(3, 4)
    bias = torch.linspace(-0.25, 0.25, steps=3, dtype=dtype)
    out = torch.nn.functional.linear(x, weight, bias)
    write_storage_bytes(run_dir / "i0.bin", x.contiguous())
    write_storage_bytes(run_dir / "expected-0.bin", out.contiguous())
    meta["input_count"] = 1
elif demo == "activation_tanh":
    x = torch.linspace(-4.0, 4.0, steps=16, dtype=dtype).reshape(4, 4)
    out = torch.tanh(x)
    write_storage_bytes(run_dir / "i0.bin", x.contiguous())
    write_storage_bytes(run_dir / "expected-0.bin", out.contiguous())
    meta["input_count"] = 1
elif demo == "activation_hardswish":
    x = torch.linspace(-5.0, 5.0, steps=1 * 2 * 3 * 4, dtype=dtype).reshape(1, 2, 3, 4)
    out = torch.nn.functional.hardswish(x)
    write_storage_bytes(run_dir / "i0.bin", x.contiguous())
    write_storage_bytes(run_dir / "expected-0.bin", out.contiguous())
    meta["input_count"] = 1
elif demo == "conv_x3":
    x = torch.linspace(-1.0 * conv_input_scale, 1.0 * conv_input_scale, steps=1 * 3 * 8 * 8, dtype=dtype).reshape(1, 3, 8, 8)
    x = x.contiguous(memory_format=torch.channels_last)
    convs = []
    for idx in range(3):
        conv = torch.nn.Conv2d(3, 3, 3, padding=1, bias=False).to(dtype)
        with torch.no_grad():
            conv.weight.copy_(
                torch.linspace(
                    -1.0 + idx * 0.2,
                    1.0 + idx * 0.2,
                    steps=conv.weight.numel(),
                    dtype=torch.float32,
                ).reshape_as(conv.weight)
            )
        convs.append(conv)
    out = x
    for conv in convs:
        out = conv(out)
    write_storage_bytes(run_dir / "i0.bin", x)
    write_channels_last_bytes(run_dir / "expected-0.bin", out)
    meta["input_count"] = 1
elif demo == "conv_bias":
    x = torch.linspace(-3.0 * conv_input_scale, 3.0 * conv_input_scale, steps=1 * 5 * 10 * 10, dtype=dtype).reshape(1, 5, 10, 10)
    x = x.contiguous(memory_format=torch.channels_last)
    conv = torch.nn.Conv2d(5, 4, (1, 2), bias=True).to(dtype)
    with torch.no_grad():
        conv.weight.copy_(
            torch.linspace(
                -1.0,
                1.0,
                steps=conv.weight.numel(),
                dtype=torch.float32,
            ).reshape_as(conv.weight)
        )
        conv.bias.copy_(torch.linspace(-0.25, 0.25, steps=conv.bias.numel(), dtype=torch.float32))
    out = conv(x)
    write_storage_bytes(run_dir / "i0.bin", x)
    write_channels_last_bytes(run_dir / "expected-0.bin", out)
    meta["input_count"] = 1
elif demo == "depthwise_padding":
    x = torch.linspace(-2.0 * conv_input_scale, 2.0 * conv_input_scale, steps=1 * 2 * 5 * 5, dtype=dtype).reshape(1, 2, 5, 5)
    x = x.contiguous(memory_format=torch.channels_last)
    conv = torch.nn.Conv2d(2, 2, 5, padding=2, groups=2, bias=False).to(dtype)
    with torch.no_grad():
        conv.weight.copy_(
            torch.linspace(
                -1.0,
                1.0,
                steps=conv.weight.numel(),
                dtype=torch.float32,
            ).reshape_as(conv.weight)
        )
    out = conv(x)
    write_storage_bytes(run_dir / "i0.bin", x)
    write_channels_last_bytes(run_dir / "expected-0.bin", out)
    meta["input_count"] = 1
elif demo == "depthwise_bias":
    x = torch.linspace(-3.0 * conv_input_scale, 3.0 * conv_input_scale, steps=1 * 3 * 6 * 6, dtype=dtype).reshape(1, 3, 6, 6)
    x = x.contiguous(memory_format=torch.channels_last)
    conv = torch.nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=True).to(dtype)
    with torch.no_grad():
        conv.weight.copy_(
            torch.linspace(
                -1.0,
                1.0,
                steps=conv.weight.numel(),
                dtype=torch.float32,
            ).reshape_as(conv.weight)
        )
        conv.bias.copy_(torch.linspace(-0.25, 0.25, steps=conv.bias.numel(), dtype=torch.float32))
    out = conv(x)
    write_storage_bytes(run_dir / "i0.bin", x)
    write_channels_last_bytes(run_dir / "expected-0.bin", out)
    meta["input_count"] = 1
elif demo == "conv_transpose_basic":
    x = torch.linspace(1.0, 5.0, steps=1 * 2 * 5 * 5, dtype=dtype).reshape(1, 2, 5, 5)
    x = x.contiguous(memory_format=torch.channels_last)
    conv_t = torch.nn.ConvTranspose2d(2, 4, 3, bias=False).to(dtype)
    with torch.no_grad():
        conv_t.weight.copy_(
            torch.linspace(
                -1.0,
                1.0,
                steps=conv_t.weight.numel(),
                dtype=torch.float32,
            ).reshape_as(conv_t.weight)
        )
    out = conv_t(x)
    write_storage_bytes(run_dir / "i0.bin", x)
    write_channels_last_bytes(run_dir / "expected-0.bin", out)
    meta["input_count"] = 1
elif demo == "conv_transpose_bias":
    x = torch.linspace(-20.0, 20.0, steps=1 * 4 * 6 * 6, dtype=dtype).reshape(1, 4, 6, 6)
    x = x.contiguous(memory_format=torch.channels_last)
    conv_t = torch.nn.ConvTranspose2d(4, 8, 3, bias=True).to(dtype)
    with torch.no_grad():
        conv_t.weight.copy_(
            torch.linspace(
                -1.0,
                1.0,
                steps=conv_t.weight.numel(),
                dtype=torch.float32,
            ).reshape_as(conv_t.weight)
        )
        conv_t.bias.copy_(torch.linspace(-0.25, 0.25, steps=conv_t.bias.numel(), dtype=torch.float32))
    out = conv_t(x)
    write_storage_bytes(run_dir / "i0.bin", x)
    write_channels_last_bytes(run_dir / "expected-0.bin", out)
    meta["input_count"] = 1
else:
    raise SystemExit(f"Unsupported demo: {demo}")

(run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
PY

build_executorch_args=(
  --toolchain="${toolchain}"
  --et_build_root="${et_build_root}"
  "${CORTEX_M_FLOAT_BUILD_ARGS[@]}"
  --devtools
)
if [[ -n "${cmsis_nn_local_path}" ]]; then
  build_executorch_args+=(--cmsis_nn_local_path="${cmsis_nn_local_path}")
fi
backends/arm/scripts/build_executorch.sh "${build_executorch_args[@]}"
set_cortex_m_float_capabilities_from_build "${et_build_root}" "${toolchain}"

runner_build_args=(
  --toolchain="${toolchain}"
  --et_build_root="${et_build_root}"
  --pte=semihosting
  --target="${target}"
  --system_config=Ethos_U55_High_End_Embedded
  --memory_mode=Shared_Sram
  --output="${runner_dir}"
  --extra_build_flags=-DET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE=83886080
)
if [[ -n "${cmsis_nn_local_path}" ]]; then
  runner_build_args+=(--cmsis_nn_local_path="${cmsis_nn_local_path}")
fi
backends/arm/scripts/build_executor_runner.sh "${runner_build_args[@]}"

cmd_line="executor_runner -m $(basename "${pte_file}") -o out"
if [[ -f "${run_dir}/i0.bin" ]]; then
  cmd_line+=" -i i0.bin"
fi
if [[ -f "${run_dir}/i1.bin" ]]; then
  cmd_line+=" -i i1.bin"
fi
if [[ -f "${run_dir}/i2.bin" ]]; then
  cmd_line+=" -i i2.bin"
fi

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
  -C "cpu0.semihosting-cmd_line=${cmd_line}"
  -a "${runner_dir}/arm_executor_runner"
  --timelimit "${timeout}"
)
echo "FVP command:"
print_command "${fvp_cmd[@]}"
"${fvp_cmd[@]}"

python - <<'PY'
import json
import numpy as np
import os
from pathlib import Path

run_dir = Path(os.environ["ET_RUN_DIR"])
meta = json.loads((run_dir / "run_meta.json").read_text())
dtype = np.float16 if meta["dtype"] == "float16" else np.float32
out_path = run_dir / "out-0.bin"
ref_path = run_dir / "expected-0.bin"
if not out_path.exists():
    raise SystemExit(f"Missing output file: {out_path}")
got = np.fromfile(out_path, dtype=dtype)
ref = np.fromfile(ref_path, dtype=dtype)
print("demo      :", meta["demo"])
print("dtype     :", meta["dtype"])
print("output_file:", out_path)
print("expected  :", ref)
print("got       :", got)
print("sum(got)  :", got.astype(np.float32).sum())
print("sum(ref)  :", ref.astype(np.float32).sum())
print("max_abs   :", np.max(np.abs(got.astype(np.float32) - ref.astype(np.float32))))
PY
