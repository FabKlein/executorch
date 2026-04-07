# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
BUILD_SCRIPT = REPO_ROOT / "backends/arm/scripts/build_executorch.sh"


def _require_arm_gcc() -> Path:
    gcc = shutil.which("arm-none-eabi-gcc")
    if gcc is None:
        pytest.skip("arm-none-eabi-gcc not found on PATH")

    version_text = subprocess.check_output(
        [gcc, "-dumpfullversion", "-dumpversion"], text=True
    ).strip()
    major = int(version_text.split(".", maxsplit=1)[0])
    if major < 14:
        pytest.skip(
            f"Cortex-M float capability tests require GCC >= 14, got {version_text}"
        )
    return Path(gcc)


def _require_cmsis_nn_local_path() -> Path:
    cmsis_nn_local_path = os.environ.get("CMSIS_NN_LOCAL_PATH", "").strip()
    if not cmsis_nn_local_path:
        pytest.skip(
            "Set CMSIS_NN_LOCAL_PATH to a float-enabled local CMSIS-NN checkout"
        )

    cmsis_nn_root = Path(cmsis_nn_local_path)
    if not cmsis_nn_root.is_dir():
        pytest.skip(
            f"CMSIS_NN_LOCAL_PATH does not point to a directory: {cmsis_nn_root}"
        )

    if not (cmsis_nn_root / "Include" / "arm_nnfunctions_flt.h").is_file():
        pytest.skip(
            "CMSIS_NN_LOCAL_PATH does not look like a float-enabled CMSIS-NN "
            f"checkout: {cmsis_nn_root}"
        )

    return cmsis_nn_root


def _build_root_name(enable_f32: bool, enable_f16: bool) -> str:
    if enable_f32 and enable_f16:
        return "gcc_f32_on_f16_on"
    if enable_f32:
        return "gcc_f32_on_f16_off"
    if enable_f16:
        return "gcc_f32_off_f16_on"
    return "gcc_f32_off_f16_off"


@pytest.fixture(scope="session")
def gcc_path() -> Path:
    return _require_arm_gcc()


@pytest.fixture(scope="session")
def cmsis_nn_local_path() -> Path:
    return _require_cmsis_nn_local_path()


@pytest.fixture(scope="session")
def build_roots(
    tmp_path_factory: pytest.TempPathFactory,
    gcc_path: Path,
    cmsis_nn_local_path: Path,
) -> dict[tuple[bool, bool], Path]:
    roots: dict[tuple[bool, bool], Path] = {}
    gcc_bin = str(gcc_path.parent)
    combos = (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    )

    for enable_f32, enable_f16 in combos:
        build_root = tmp_path_factory.mktemp(
            _build_root_name(enable_f32, enable_f16), numbered=False
        )
        cmd = [
            "bash",
            str(BUILD_SCRIPT),
            "--toolchain=arm-none-eabi-gcc",
            f"--et_build_root={build_root}",
            f"--cmsis_nn_local_path={cmsis_nn_local_path}",
        ]
        if enable_f32:
            cmd.append("--cmsis_nn_enable_f32")
        if enable_f16:
            cmd.append("--cmsis_nn_enable_f16")

        env = os.environ.copy()
        env["PATH"] = f"{gcc_bin}:{env.get('PATH', '')}"
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        roots[(enable_f32, enable_f16)] = build_root

    return roots


def _capabilities_file(build_root: Path) -> Path:
    return (
        build_root / "cmake-out" / "backends" / "cortex_m" / "float_capabilities.json"
    )


def _assert_capability_artifact(
    build_root: Path, enable_f32: bool, enable_f16: bool
) -> None:
    capabilities_path = _capabilities_file(build_root)
    assert (
        capabilities_path.is_file()
    ), f"Missing float capability artifact: {capabilities_path}"
    capabilities = json.loads(capabilities_path.read_text())
    assert capabilities == {
        "enable_f32": enable_f32,
        "enable_f16": enable_f16,
    }


def _run_mobilenet_prepare(
    build_root: Path, dtype: str
) -> subprocess.CompletedProcess[str]:
    capabilities_path = _capabilities_file(build_root)
    script = """
import torch
import executorch.backends.cortex_m.ops.operators  # noqa: F401
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.exir import EdgeCompileConfig
from executorch.extension.export_util.utils import export_to_edge
from torchvision.models import mobilenet_v3_small

dtype_name = __import__("sys").argv[1]
dtype = torch.float16 if dtype_name == "float16" else torch.float32
model = mobilenet_v3_small(weights=None).eval()
model = model.half() if dtype == torch.float16 else model.float()
sample = torch.randn(1, 3, 224, 224, dtype=dtype).contiguous(
    memory_format=torch.channels_last
)
edge_program = export_to_edge(
    model,
    (sample,),
    edge_compile_config=EdgeCompileConfig(_check_ir_validity=False),
)
edge_program._edge_programs["forward"] = CortexMPassManager(
    edge_program.exported_program()
).transform()
print("OK")
"""

    env = os.environ.copy()
    env["EXECUTORCH_CORTEX_M_FLOAT_CAPABILITIES_FILE"] = str(capabilities_path)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-c", script, dtype],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("enable_f32", "enable_f16", "dtype", "should_succeed"),
    (
        (False, False, "float32", False),
        (False, False, "float16", False),
        (True, False, "float32", True),
        (True, False, "float16", False),
        (False, True, "float32", False),
        (False, True, "float16", True),
        (True, True, "float32", True),
        (True, True, "float16", True),
    ),
)
def test_mobilenet_v3_float_capability_matrix(
    build_roots: dict[tuple[bool, bool], Path],
    enable_f32: bool,
    enable_f16: bool,
    dtype: str,
    should_succeed: bool,
) -> None:
    build_root = build_roots[(enable_f32, enable_f16)]
    _assert_capability_artifact(build_root, enable_f32, enable_f16)

    result = _run_mobilenet_prepare(build_root, dtype)

    if should_succeed:
        assert result.returncode == 0, result.stdout
        assert "OK" in result.stdout, result.stdout
    else:
        assert result.returncode != 0, result.stdout
        assert (
            f"requires Cortex-M CMSIS-NN {dtype} support" in result.stdout
        ), result.stdout
