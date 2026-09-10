# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility imports for the former pass-local capability module."""

from executorch.backends.cortex_m.float_capabilities import (  # noqa: F401
    CortexMFloatCapabilities,
    get_cortex_m_float_capabilities,
    get_optional_cortex_m_float_op,
)

__all__ = [
    "CortexMFloatCapabilities",
    "get_cortex_m_float_capabilities",
    "get_optional_cortex_m_float_op",
]
