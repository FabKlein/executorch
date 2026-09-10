# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Float activation → cortex_m::activation_f{32,16} rewrite pass."""

from typing import Dict, Optional

import torch
from executorch.backends.cortex_m.float_activation_constants import (
    CMSIS_FLOAT_ACT_HARDSIGMOID,
    CMSIS_FLOAT_ACT_HARDSWISH,
    CMSIS_FLOAT_ACT_HARDTANH,
    CMSIS_FLOAT_ACT_LEAKY_RELU,
    CMSIS_FLOAT_ACT_RELU,
    CMSIS_FLOAT_ACT_RELU6,
    CMSIS_FLOAT_ACT_SIGMOID,
    CMSIS_FLOAT_ACT_TANH,
)
from executorch.backends.cortex_m.passes.cortex_m_rewrite_base import CortexMRewriteBase
from executorch.backends.cortex_m.float_capabilities import (
    get_optional_cortex_m_float_op,
)
from executorch.backends.cortex_m.passes.passes_utils import is_channels_last
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import NodeMetadata, ProxyValue
from torch.fx.node import Argument


class FloatActivationRewritePass(CortexMRewriteBase):
    """Rewrite standalone float activations to cortex_m::activation_f{32,16}.

    This pass handles direct activation nodes that survived earlier fusion:

        aten.relu(x)          -> cortex_m::activation_f*(x, RELU)
        aten.hardtanh(x,0,6)  -> cortex_m::activation_f*(x, RELU6)

    Decomposed arithmetic forms such as hardswish are handled separately by
    CollapseFloatActivationDecompositionPass.
    """

    def _get_float_activation_replacement(self, op, args):
        input_tensor = args[0].data
        input_dtype = getattr(input_tensor, "dtype", None)
        if input_dtype not in (torch.float32, torch.float16):
            return op, args

        replacement_op = (
            exir_ops.edge.cortex_m.activation_f32.default
            if input_dtype == torch.float32
            else exir_ops.edge.cortex_m.activation_f16.default
        )
        self._require_float_dtype(
            input_dtype,
            (
                "cortex_m::activation_f32"
                if input_dtype == torch.float32
                else "cortex_m::activation_f16"
            ),
        )

        if op == exir_ops.edge.aten.relu.default:
            return replacement_op, (args[0], CMSIS_FLOAT_ACT_RELU, 0.0)
        if op == exir_ops.edge.aten.tanh.default:
            return replacement_op, (args[0], CMSIS_FLOAT_ACT_TANH, 0.0)
        if op == exir_ops.edge.aten.sigmoid.default:
            return replacement_op, (args[0], CMSIS_FLOAT_ACT_SIGMOID, 0.0)
        if op == exir_ops.edge.aten.hardswish.default:
            return replacement_op, (args[0], CMSIS_FLOAT_ACT_HARDSWISH, 0.0)
        if op == exir_ops.edge.aten.hardsigmoid.default:
            return replacement_op, (args[0], CMSIS_FLOAT_ACT_HARDSIGMOID, 0.0)
        if op == exir_ops.edge.aten.leaky_relu.default:
            negative_slope = self._unwrap_argument(args[1]) if len(args) > 1 else 0.01
            return replacement_op, (
                args[0],
                CMSIS_FLOAT_ACT_LEAKY_RELU,
                float(negative_slope),
            )
        if op == exir_ops.edge.aten.hardtanh.default:
            min_val = float(self._unwrap_argument(args[1])) if len(args) > 1 else -1.0
            max_val = float(self._unwrap_argument(args[2])) if len(args) > 2 else 1.0
            # CMSIS-NN exposes ReLU6 as an activation mode.  Keep arbitrary
            # hardtanh ranges as aten unless they were fused into a producer
            # earlier by FloatOpRewritePass.
            if min_val == 0.0 and max_val == 6.0:
                return replacement_op, (args[0], CMSIS_FLOAT_ACT_RELU6, 0.0)
            if min_val == -1.0 and max_val == 1.0:
                return replacement_op, (args[0], CMSIS_FLOAT_ACT_HARDTANH, 0.0)
        if op == exir_ops.edge.aten.clamp.default:
            min_arg = self._unwrap_argument(args[1]) if len(args) > 1 else None
            max_arg = self._unwrap_argument(args[2]) if len(args) > 2 else None
            if min_arg is None or max_arg is None:
                return op, args
            min_val = float(min_arg)
            max_val = float(max_arg)
            if min_val == -1.0 and max_val == 1.0:
                return replacement_op, (args[0], CMSIS_FLOAT_ACT_HARDTANH, 0.0)

        return op, args

    def call_operator(
        self,
        op: EdgeOpOverload,
        args: tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        match op:
            case (
                exir_ops.edge.aten.relu.default
                | exir_ops.edge.aten.tanh.default
                | exir_ops.edge.aten.sigmoid.default
                | exir_ops.edge.aten.hardswish.default
                | exir_ops.edge.aten.hardsigmoid.default
                | exir_ops.edge.aten.leaky_relu.default
                | exir_ops.edge.aten.hardtanh.default
                | exir_ops.edge.aten.clamp.default
            ):
                op, args = self._get_float_activation_replacement(op, args)

        result = super().call_operator(op, args, {}, meta)

        activation_targets = tuple(
            target
            for target in (
                get_optional_cortex_m_float_op("activation_f32"),
                get_optional_cortex_m_float_op("activation_f16"),
            )
            if target is not None
        )
        if op in activation_targets:
            # Most activations are layout-preserving.  When the input is a 4D
            # channels-last fake tensor, keep that metadata on the output node so
            # later convolution/layout passes still see:
            #
            #   NHWC producer -> activation -> NHWC consumer
            #
            # rather than losing layout information at the activation boundary.
            input_arg = args[0]
            input_tensor = getattr(input_arg, "data", None)
            if (
                input_tensor is not None
                and getattr(input_tensor, "ndim", 0) == 4
                and is_channels_last(input_tensor)
            ):
                result.node.meta["val"] = input_tensor

        return result
