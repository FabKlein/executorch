#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
REBUILD_RUNNERS=0
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

for arg in "$@"; do
  case "$arg" in
    --rebuild-runners)
      REBUILD_RUNNERS=1
      ;;
    --toolchain_bin=*)
      toolchain_bin="${arg#*=}"
      ;;
    --cmsis_nn_local_path=*)
      cmsis_nn_local_path="${arg#*=}"
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--rebuild-runners] [--toolchain_bin=/path/to/bin] [--cmsis_nn_local_path=/path/to/CMSIS-NN]" >&2
      exit 1
      ;;
  esac
done

cd "$ROOT_DIR"

if [[ ! -d venv ]]; then
  echo "Expected virtual environment at $ROOT_DIR/venv" >&2
  exit 1
fi

source venv/bin/activate
if [[ -n "${toolchain_bin}" ]]; then
  export PATH="${toolchain_bin}:${PATH}"
fi
source examples/arm/arm-scratch/setup_path.sh
if [[ -n "${toolchain_bin}" ]]; then
  export PATH="${toolchain_bin}:${PATH}"
fi
export PYTHONPATH="$ROOT_DIR/src"
if [[ -n "${cmsis_nn_local_path}" ]]; then
  export CMSIS_NN_LOCAL_PATH="${cmsis_nn_local_path}"
fi

check_arm_gcc

if [[ "$REBUILD_RUNNERS" -eq 1 ]]; then
  src/executorch/backends/arm/test/setup_testing.sh
fi

pytest --config-file=backends/arm/test/pytest.ini backends/cortex_m/test
