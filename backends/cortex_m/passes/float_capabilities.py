# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pass-time capability gate for Cortex-M float lowering.

Reads the build-generated Cortex-M float capability artifact and exposes a
typed helper used by the export passes. This keeps float lowering aligned with
the backend that was actually compiled, instead of relying on ad hoc wrapper
environment variables.
"""

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
    # These flags model what the Cortex-M backend was built to support, not
    # what the incoming model happens to use.
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
            f"{op_name} requires Cortex-M CMSIS-NN {dtype_name} support, but the build-generated "
            "Cortex-M float capability artifact reports that dtype as disabled. Rebuild the backend "
            "with the matching float support enabled before running the Cortex-M export pipeline."
        )


def get_cortex_m_float_capabilities() -> CortexMFloatCapabilities:
    # Keep capability lookup in one place so all passes share the same backend
    # view. The build-generated artifact is the single source of truth; if it
    # is absent we conservatively assume float lowering is disabled.
    global _CAPABILITY_CACHE_KEY
    global _CAPABILITY_CACHE_VALUE
    global _MISSING_ARTIFACT_WARNED

    artifact_path = os.environ.get("EXECUTORCH_CORTEX_M_FLOAT_CAPABILITIES_FILE")
    if not artifact_path:
        if not _MISSING_ARTIFACT_WARNED:
            warnings.warn(
                "EXECUTORCH_CORTEX_M_FLOAT_CAPABILITIES_FILE is unset; "
                "defaulting Cortex-M float lowering to f32=disabled, "
                "f16=disabled. Export wrappers should point this at the "
                "build-generated float_capabilities.json artifact.",
                stacklevel=2,
            )
            _MISSING_ARTIFACT_WARNED = True
        return CortexMFloatCapabilities(enable_f32=False, enable_f16=False)

    artifact = Path(artifact_path)
    if not artifact.exists():
        warnings.warn(
            "EXECUTORCH_CORTEX_M_FLOAT_CAPABILITIES_FILE points to a missing "
            f"artifact: {artifact}. Defaulting Cortex-M float lowering to "
            "f32=disabled, f16=disabled.",
            stacklevel=2,
        )
        return CortexMFloatCapabilities(enable_f32=False, enable_f16=False)

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
    """Return the Cortex-M op packet's default overload if it was registered.

    Disabled float dtypes are removed from the generated Cortex-M operator
    schema set, so passes must treat these lookups as optional when matching or
    selecting replacement ops.
    """

    try:
        return getattr(exir_ops.edge.cortex_m, op_name).default
    except AttributeError:
        return None
