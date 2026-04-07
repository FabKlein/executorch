# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Float pool / reduction → cortex_m rewrite pass.

Handles: max_pool2d, avg_pool2d, mean.dim (→ avg_pool2d), softmax.
"""

from typing import Dict, Optional

import torch
from executorch.backends.cortex_m.passes.cortex_m_rewrite_base import CortexMRewriteBase
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import NodeMetadata, ProxyValue
from torch.fx.node import Argument


class FloatPoolRewritePass(CortexMRewriteBase):
    """Rewrite float pooling / reduction ops to cortex_m equivalents.

    The pass keeps only the CMSIS-NN-friendly cases:

        aten.max_pool2d / avg_pool2d  -> cortex_m pool op
        aten.mean(x, dim=[2,3], keepdim=True)
                                  -> cortex_m avg_pool2d over HxW
        aten.softmax(dim=last)    -> cortex_m softmax

    Unsupported options intentionally fall back to aten so the caller can see
    the residual op instead of silently changing semantics.
    """

    def _get_float_max_pool2d_replacement(self, args):
        input_tensor = args[0].data
        input_dtype = input_tensor.dtype
        if input_dtype not in (torch.float32, torch.float16):
            return exir_ops.edge.aten.max_pool2d.default, args

        kernel_size = self._to_int_pair(args[1], None)
        stride_arg = args[2] if len(args) > 2 else None
        stride = self._to_int_pair(stride_arg, kernel_size)
        padding_arg = args[3] if len(args) > 3 else None
        padding = self._to_int_pair(padding_arg, (0, 0))
        dilation_arg = args[4] if len(args) > 4 else None
        dilation = self._to_int_pair(dilation_arg, (1, 1))
        ceil_mode_arg = args[5] if len(args) > 5 else False
        ceil_mode = self._to_bool(ceil_mode_arg, False)

        # CMSIS-NN pool wrappers here implement the common non-dilated,
        # floor-mode variant.  Leave uncommon variants in aten form.
        if dilation != (1, 1) or ceil_mode:
            return exir_ops.edge.aten.max_pool2d.default, args

        replacement_op = (
            exir_ops.edge.cortex_m.max_pool2d_f32.default
            if input_dtype == torch.float32
            else exir_ops.edge.cortex_m.max_pool2d_f16.default
        )
        self._require_float_dtype(
            input_dtype,
            (
                "cortex_m::max_pool2d_f32"
                if input_dtype == torch.float32
                else "cortex_m::max_pool2d_f16"
            ),
        )
        return replacement_op, (args[0], kernel_size, stride, padding)

    def _get_float_avg_pool2d_replacement(self, args, meta):
        input_dtype = args[0].data.dtype
        if input_dtype not in (torch.float32, torch.float16):
            return exir_ops.edge.aten.avg_pool2d.default, args

        kernel_size = self._to_int_pair(args[1], None)
        stride_arg = args[2] if len(args) > 2 else None
        stride = self._to_int_pair(stride_arg, kernel_size)
        padding_arg = args[3] if len(args) > 3 else None
        padding = self._to_int_pair(padding_arg, (0, 0))
        ceil_mode_arg = args[4] if len(args) > 4 else False
        ceil_mode = self._to_bool(ceil_mode_arg, False)
        count_include_pad_arg = args[5] if len(args) > 5 else True
        count_include_pad = self._to_bool(count_include_pad_arg, True)
        divisor_override = args[6] if len(args) > 6 else None
        divisor_override_val = self._unwrap_argument(divisor_override)

        # The backend average-pool op divides by the valid window size.  Padding
        # included in the divisor or explicit divisor_override would need a
        # different numerical contract, so keep those as aten.
        if (
            ceil_mode
            or (count_include_pad and padding != (0, 0))
            or divisor_override_val is not None
        ):
            return exir_ops.edge.aten.avg_pool2d.default, args

        replacement_op = (
            exir_ops.edge.cortex_m.avg_pool2d_f32.default
            if input_dtype == torch.float32
            else exir_ops.edge.cortex_m.avg_pool2d_f16.default
        )
        self._require_float_dtype(
            input_dtype,
            (
                "cortex_m::avg_pool2d_f32"
                if input_dtype == torch.float32
                else "cortex_m::avg_pool2d_f16"
            ),
        )
        return replacement_op, (args[0], kernel_size, stride, padding)

    def _get_float_mean_dim_replacement(self, args, meta):
        input_arg = args[0]
        input_tensor = input_arg.data
        if input_tensor.dtype not in (torch.float32, torch.float16):
            return exir_ops.edge.aten.mean.dim, args
        if input_tensor.dim() != 4:
            return exir_ops.edge.aten.mean.dim, args

        dims_arg = args[1] if len(args) > 1 else None
        dims_val = self._unwrap_argument(dims_arg)
        if dims_val is None:
            return exir_ops.edge.aten.mean.dim, args

        dims = [int(d) % input_tensor.dim() for d in list(dims_val)]
        if sorted(dims) != [2, 3]:
            return exir_ops.edge.aten.mean.dim, args

        keepdim_arg = args[2] if len(args) > 2 else False
        keepdim = self._to_bool(keepdim_arg, False)
        if not keepdim:
            return exir_ops.edge.aten.mean.dim, args

        dtype_arg = args[3] if len(args) > 3 else None
        dtype_val = self._unwrap_argument(dtype_arg)
        if dtype_val is not None and dtype_val != input_tensor.dtype:
            return exir_ops.edge.aten.mean.dim, args

        # Global-average-pool lowering used by MCU image/KWS models:
        #
        #   aten.mean(x[N,C,H,W], dim=[2,3], keepdim=True)
        #
        # becomes:
        #
        #   cortex_m::avg_pool2d(x, kernel=(H,W), stride=(H,W), padding=0)
        #
        # Layout metadata may already describe the tensor as NHWC physically, but
        # the logical reduction dims are still H/W in the Edge IR contract.
        kernel_size = (int(input_tensor.shape[-2]), int(input_tensor.shape[-1]))
        avg_pool_args = (
            input_arg,
            kernel_size,
            kernel_size,
            (0, 0),
            False,
            False,
            None,
        )
        return self._get_float_avg_pool2d_replacement(avg_pool_args, meta)

    def _get_float_softmax_replacement(self, args):
        input_tensor = args[0].data
        input_dtype = input_tensor.dtype
        if input_dtype not in (torch.float32, torch.float16):
            return exir_ops.edge.aten._softmax.default, args

        dim = self._unwrap_argument(args[1])
        half_to_float = args[2] if len(args) > 2 else False
        if half_to_float:
            return exir_ops.edge.aten._softmax.default, args

        rank = len(input_tensor.shape)
        positive_dim = dim if dim >= 0 else dim + rank
        if positive_dim != rank - 1:
            return exir_ops.edge.aten._softmax.default, args

        # CMSIS-NN softmax operates across the innermost contiguous dimension,
        # so only last-dimension softmax is lowered here.
        if input_dtype == torch.float32:
            self._require_float_dtype(input_dtype, "cortex_m::softmax_f32")
            replacement_op = exir_ops.edge.cortex_m.softmax_f32.default
        else:
            self._require_float_dtype(input_dtype, "cortex_m::softmax_f16")
            replacement_op = exir_ops.edge.cortex_m.softmax_f16.default

        return replacement_op, (args[0], int(dim))

    # -- pass entry point ------------------------------------------------------

    def call_operator(
        self,
        op: EdgeOpOverload,
        args: tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        match op:
            case exir_ops.edge.aten.max_pool2d.default:
                op, args = self._get_float_max_pool2d_replacement(args)
            case exir_ops.edge.aten.avg_pool2d.default:
                op, args = self._get_float_avg_pool2d_replacement(args, meta)
            case exir_ops.edge.aten.mean.dim:
                op, args = self._get_float_mean_dim_replacement(args, meta)
            case exir_ops.edge.aten._softmax.default:
                op, args = self._get_float_softmax_replacement(args)

        return super().call_operator(op, args, {}, meta)
