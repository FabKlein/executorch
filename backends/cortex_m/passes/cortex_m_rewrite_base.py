# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Optional

import torch
from executorch.backends.cortex_m.float_capabilities import (
    CortexMFloatCapabilities,
    get_cortex_m_float_capabilities,
)
from executorch.backends.cortex_m.passes.passes_utils import is_channels_last
from executorch.exir.pass_base import ExportPass, ProxyValue
from torch.fx.node import Argument

NHWC_DIM_ORDER = [0, 2, 3, 1]
FLOAT_FUSED_ACTIVATION_TAG = "cortex_m_float_activation"


class CortexMRewriteBase(ExportPass):
    """Shared base for Cortex-M call_operator rewrite passes.

    These passes replace one op target/argument list at a time while the
    ExportPass interpreter replays the graph:

        aten op + args + metadata
                 |
                 v
        _get_*_replacement()
                 |
                 v
        cortex_m op + normalized args

    Rewrites that need to create constants or insert extra FX nodes still live
    in ConvertToCortexMPass instead.
    """

    def __init__(self, capabilities: Optional[CortexMFloatCapabilities] = None) -> None:
        super().__init__()
        self.capabilities = capabilities or get_cortex_m_float_capabilities()

    def _require_float_dtype(self, dtype: torch.dtype, op_name: str) -> None:
        self.capabilities.require_float_dtype_enabled(dtype, op_name)

    def _to_physical_order(self, logical_pad: list[int], tensor_data) -> list[int]:
        # PyTorch pad lists are expressed in logical tensor dimensions.  CMSIS-NN
        # kernels consume the physical memory order, so channels-last tensors
        # need NCHW logical pad values remapped to NHWC physical positions.
        #
        #   logical NCHW dims:   [N, C, H, W]
        #   physical NHWC dims:  [N, H, W, C]
        if not is_channels_last(tensor_data):
            return logical_pad
        return [logical_pad[NHWC_DIM_ORDER[i]] for i in range(4)]

    def _get_default_float_activation_bounds(
        self, dtype: torch.dtype
    ) -> tuple[float, float]:
        if dtype == torch.float16:
            finfo = torch.finfo(torch.float16)
            return float(finfo.min), float(finfo.max)
        finfo = torch.finfo(torch.float32)
        return float(finfo.min), float(finfo.max)

    def _get_float_activation_bounds(
        self, dtype: torch.dtype, node_meta: Optional[dict]
    ) -> tuple[float, float]:
        # Activation fusion stores the clamp bounds on the producer metadata:
        #
        #   producer -> relu/hardtanh
        #
        # becomes:
        #
        #   producer(meta["custom"][FLOAT_FUSED_ACTIVATION_TAG] = (min, max))
        #
        # The eventual cortex_m op receives those bounds as explicit arguments.
        default_min, default_max = self._get_default_float_activation_bounds(dtype)
        if not node_meta:
            return default_min, default_max

        custom_meta = node_meta.get("custom", {})
        bounds = custom_meta.get(FLOAT_FUSED_ACTIVATION_TAG)
        if bounds is None:
            return default_min, default_max

        activation_min, activation_max = bounds
        activation_min = (
            default_min if activation_min is None else float(activation_min)
        )
        activation_max = (
            default_max if activation_max is None else float(activation_max)
        )
        activation_min = max(default_min, activation_min)
        activation_max = min(default_max, activation_max)
        return activation_min, activation_max

    def _to_int_pair(
        self, value: Argument, default: Optional[tuple[int, int]]
    ) -> tuple[int, int]:
        if value is None:
            assert default is not None, "Expected default sequence for normalization"
            return (default[0], default[1])
        try:
            int_pair = cast(tuple[int, int], value)
            return int_pair
        except Exception:
            raise ValueError(f"Expected a tuple of two integers, got {value}")

    def _unwrap_argument(self, arg: Argument) -> Argument:
        # ExportPass call_operator receives ProxyValue wrappers, but many scalar
        # parameters (alpha, dim, keepdim, etc.) are easier to validate as the
        # underlying Python values.
        if isinstance(arg, ProxyValue):
            return arg.data
        return arg

    def _to_bool(self, value: Argument, default: bool) -> bool:
        if value is None:
            return default
        try:
            bool_value = cast(bool, value)
            return bool_value
        except Exception:
            raise ValueError(f"Expected a boolean value, got {value}")
