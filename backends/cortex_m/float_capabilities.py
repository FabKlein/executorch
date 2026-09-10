# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build-generated capability gate for Cortex-M float lowering."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from executorch.exir.dialects._ops import ops as exir_ops

_CAPABILITY_CACHE_KEY: tuple[str, int] | None = None
_CAPABILITY_CACHE_VALUE: "CortexMFloatCapabilities | None" = None
_MISSING_ARTIFACT_WARNED = False


@dataclass(frozen=True)
class CortexMFloatCapabilities:
    enable_f32: bool
    enable_f16: bool

    def is_float_dtype_enabled(self, dtype: torch.dtype) -> bool:
        if dtype == torch.float32:
            return self.enable_f32
        if dtype == torch.float16:
            return self.enable_f16
        return True

    def require_float_dtype_enabled(self, dtype: torch.dtype, op_name: str) -> None:
        if self.is_float_dtype_enabled(dtype):
            return
        dtype_name = (
            "float32"
            if dtype == torch.float32
            else "float16" if dtype == torch.float16 else str(dtype)
        )
        raise RuntimeError(
            f"{op_name} requires Cortex-M CMSIS-NN {dtype_name} support, but "
            "the build-generated capability artifact reports it as disabled. "
            "Rebuild the backend with matching float support."
        )


def get_cortex_m_float_capabilities() -> CortexMFloatCapabilities:
    global _CAPABILITY_CACHE_KEY, _CAPABILITY_CACHE_VALUE, _MISSING_ARTIFACT_WARNED

    artifact_path = os.environ.get("EXECUTORCH_CORTEX_M_FLOAT_CAPABILITIES_FILE")
    disabled = CortexMFloatCapabilities(enable_f32=False, enable_f16=False)
    if not artifact_path:
        if not _MISSING_ARTIFACT_WARNED:
            warnings.warn(
                "EXECUTORCH_CORTEX_M_FLOAT_CAPABILITIES_FILE is unset; "
                "defaulting Cortex-M float lowering to f32=disabled, "
                "f16=disabled.",
                stacklevel=2,
            )
            _MISSING_ARTIFACT_WARNED = True
        return disabled

    artifact = Path(artifact_path)
    if not artifact.exists():
        warnings.warn(
            f"Cortex-M float capability artifact is missing: {artifact}. "
            "Defaulting float lowering to disabled.",
            stacklevel=2,
        )
        return disabled

    stat = artifact.stat()
    cache_key = (str(artifact.resolve()), stat.st_mtime_ns)
    if _CAPABILITY_CACHE_KEY == cache_key and _CAPABILITY_CACHE_VALUE is not None:
        return _CAPABILITY_CACHE_VALUE

    data = json.loads(artifact.read_text())
    _CAPABILITY_CACHE_KEY = cache_key
    _CAPABILITY_CACHE_VALUE = CortexMFloatCapabilities(
        enable_f32=bool(data.get("enable_f32", False)),
        enable_f16=bool(data.get("enable_f16", False)),
    )
    return _CAPABILITY_CACHE_VALUE


def get_optional_cortex_m_float_op(op_name: str):
    """Return a registered Cortex-M float op, or ``None`` when disabled."""

    try:
        return getattr(exir_ops.edge.cortex_m, op_name).default
    except AttributeError:
        return None
