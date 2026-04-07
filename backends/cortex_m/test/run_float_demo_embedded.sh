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
et_build_root="${et_root_dir}/arm_test_gcc15_embedded"
toolchain_bin=""
cmsis_nn_local_path=""

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
  echo "  --demo=softmax|transpose|combo   Demo to run. Default: ${demo}"
  echo "  --dtype=float16|float32          Demo dtype. Default: ${dtype}"
  echo "  --target=<TARGET>                FVP target. Default: ${target}"
  echo "  --timeout=<SEC>                  FVP timeout. Default: ${timeout}"
  echo "  --et_build_root=<PATH>           GCC build root. Default: ${et_build_root}"
  echo "  --toolchain_bin=<PATH>           Optional GCC toolchain bin directory to prepend to PATH."
  echo "  --cmsis_nn_local_path=<PATH>     Optional local CMSIS-NN checkout passed to build_executorch.sh."
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
  softmax|transpose|combo)
    ;;
  *)
    echo "Unsupported demo: ${demo}"
    exit 1
    ;;
esac

run_dir="${et_build_root}/${demo}_${dtype}_embedded_run"
runner_dir="${et_build_root}/${demo}_${dtype}_embedded_runner"

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
rm -f "${run_dir}"/expected-0.bin "${run_dir}"/run_meta.json "${run_dir}"/run.log

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

export ET_EMBED_RUN_DIR="${run_dir}"
export ET_EMBED_DEMO="${demo}"
export ET_EMBED_DTYPE="${dtype}"

python - <<'PY'
import json
import os
from pathlib import Path

import torch

import executorch.backends.cortex_m.ops.operators  # noqa: F401
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.exir import EdgeCompileConfig
from executorch.extension.export_util.utils import export_to_edge
from executorch.extension.export_util.utils import save_pte_program

run_dir = Path(os.environ["ET_EMBED_RUN_DIR"])
demo = os.environ["ET_EMBED_DEMO"]
dtype_name = os.environ["ET_EMBED_DTYPE"]
dtype = torch.float16 if dtype_name == "float16" else torch.float32


def write_storage_bytes(path: Path, tensor: torch.Tensor) -> None:
    path.write_bytes(bytes(tensor.contiguous().untyped_storage()))


class EmbeddedSoftmaxDemo(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.register_buffer("x", torch.linspace(-4.0, 4.0, steps=16, dtype=dtype))

    def forward(self) -> torch.Tensor:
        return torch.softmax(self.x, dim=-1)


class EmbeddedTransposeDemo(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        x = torch.linspace(-0.5, 0.5, steps=2 * 3 * 4 * 2, dtype=dtype).reshape(2, 3, 4, 2)
        self.register_buffer("x", x)

    def forward(self) -> torch.Tensor:
        return self.x.permute(0, 3, 1, 2)


class EmbeddedComboDemo(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        x = torch.linspace(-0.75, 0.75, steps=1 * 8 * 5 * 5, dtype=dtype).reshape(1, 8, 5, 5)
        y = torch.linspace(0.25, 1.25, steps=1 * 8 * 5 * 5, dtype=dtype).reshape(1, 8, 5, 5)
        bias = torch.linspace(-0.5, 0.5, steps=8, dtype=dtype).reshape(1, 8, 1, 1)
        self.register_buffer("x", x.contiguous(memory_format=torch.channels_last))
        self.register_buffer("y", y.contiguous(memory_format=torch.channels_last))
        self.register_buffer("bias", bias.contiguous(memory_format=torch.channels_last))

    def forward(self) -> torch.Tensor:
        return torch.minimum(torch.maximum((self.x + self.y) * self.y, self.bias), self.x)


if demo == "softmax":
    model = EmbeddedSoftmaxDemo(dtype).eval()
    expected = model().detach().cpu()
    pte_name = f"softmax_{'f16' if dtype == torch.float16 else 'f32'}_embedded_demo"
elif demo == "transpose":
    model = EmbeddedTransposeDemo(dtype).eval()
    expected = model().detach().cpu().contiguous()
    pte_name = f"transpose_{'f16' if dtype == torch.float16 else 'f32'}_embedded_demo"
elif demo == "combo":
    model = EmbeddedComboDemo(dtype).eval()
    expected = model().detach().cpu().contiguous(memory_format=torch.channels_last)
    pte_name = f"float_elementwise_combo_{'f16' if dtype == torch.float16 else 'f32'}_embedded_demo"
else:
    raise SystemExit(f"Unsupported demo: {demo}")

edge_program = export_to_edge(
    model,
    (),
    edge_compile_config=EdgeCompileConfig(_check_ir_validity=False),
)
edge_program._edge_programs["forward"] = CortexMPassManager(
    edge_program.exported_program()
).transform()
program = edge_program.to_executorch()
save_pte_program(program, pte_name, str(run_dir))

write_storage_bytes(run_dir / "expected-0.bin", expected)
(run_dir / "run_meta.json").write_text(
    json.dumps(
        {
            "demo": demo,
            "dtype": dtype_name,
            "pte_name": f"{pte_name}.pte",
            "numel": expected.numel(),
        },
        indent=2,
    )
)
PY

pte_file="$(python - <<'PY'
import json
import os
from pathlib import Path
run_dir = Path(os.environ["ET_EMBED_RUN_DIR"])
meta = json.loads((run_dir / "run_meta.json").read_text())
print(run_dir / meta["pte_name"])
PY
)"

runner_build_args=(
  --toolchain="${toolchain}"
  --et_build_root="${et_build_root}"
  --pte="${pte_file}"
  --target="${target}"
  --output="${runner_dir}"
)
if [[ -n "${cmsis_nn_local_path}" ]]; then
  runner_build_args+=(--cmsis_nn_local_path="${cmsis_nn_local_path}")
fi
backends/arm/scripts/build_executor_runner.sh "${runner_build_args[@]}"

backends/arm/scripts/run_fvp.sh \
  --elf="${runner_dir}/arm_executor_runner" \
  --target="${target}" \
  --timeout="${timeout}" | tee "${run_dir}/run.log"

python - <<'PY'
import json
import numpy as np
import os
import re
from pathlib import Path

run_dir = Path(os.environ["ET_EMBED_RUN_DIR"])
meta = json.loads((run_dir / "run_meta.json").read_text())
dtype = np.float16 if meta["dtype"] == "float16" else np.float32
expected = np.fromfile(run_dir / "expected-0.bin", dtype=dtype)
log_text = (run_dir / "run.log").read_text()

matches = re.findall(r"Output\[0\]\[(\d+)\]: \((?:float|half)\) ([^\s]+)", log_text)
if not matches:
    raise SystemExit("No parsed output values found in run.log; ET_LOG_DUMP_OUTPUT may be missing.")

got = np.zeros(meta["numel"], dtype=np.float32)
for index_str, value_str in matches:
    got[int(index_str)] = float(value_str)

expected_f32 = expected.astype(np.float32)
print("demo      :", meta["demo"])
print("dtype     :", meta["dtype"])
print("expected  :", expected)
print("got       :", got)
print("sum(got)  :", got.sum())
print("sum(ref)  :", expected_f32.sum())
print("max_abs   :", np.max(np.abs(got - expected_f32)))
PY
