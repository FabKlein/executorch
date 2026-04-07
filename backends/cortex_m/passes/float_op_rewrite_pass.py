# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Float op → cortex_m rewrite pass.

Handles: float add/mul/linear/pad, dtype-agnostic min/max/permute,
float batch-norm, and the float activation-fusion pre-pass that tags
preceding ops with fused activation bounds before erasing the activation node.
"""

import sys
from typing import cast, Dict, Optional

import torch
from executorch.backends.cortex_m.passes.cortex_m_rewrite_base import (
    CortexMRewriteBase,
    FLOAT_FUSED_ACTIVATION_TAG,
    NHWC_DIM_ORDER,
)
from executorch.backends.cortex_m.passes.passes_utils import (
    is_channel_broadcast,
    is_channels_last,
    is_float_depthwise_conv,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import NodeMetadata, ProxyValue
from torch.fx.node import Argument
from torch.fx.passes.infra.pass_base import PassResult


class FloatOpRewritePass(CortexMRewriteBase):
    """Rewrite float ops to cortex_m equivalents.

    Covers add, mul, linear, pad, min/max, permute, and batch norm.
    Also contains the float activation-fusion pre-pass: when a fuseable
    activation (relu, hardtanh) immediately follows a fuseable producer
    (add, mul, conv), the activation is erased and the producer is tagged
    with clamped output bounds that downstream float-op passes consume.
    """

    _FLOAT_FUSEABLE_OPS = {
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.convolution.default,
    }
    _FLOAT_FUSEABLE_ACTIVATIONS = {
        exir_ops.edge.aten.relu.default,
        exir_ops.edge.aten.hardtanh.default,
    }

    # -- activation fusion helpers (graph-level pre-pass) --------------------

    def _supports_float_binary_replacement(self, op, args) -> bool:
        # Only fuse activations into binary ops that this pass can lower later.
        # Channel-broadcast add/mul is supported only for NHWC tensors because
        # the backend interprets a 1D RHS as channel data.
        #
        #   lhs[N,H,W,C] + rhs[1,1,1,C] -> cortex_m::add_f*
        lhs = args[0].meta.get("val")
        rhs = args[1].meta.get("val")
        if lhs is None or rhs is None:
            return False
        if lhs.dtype != rhs.dtype or lhs.dtype not in (torch.float32, torch.float16):
            return False
        channel_broadcast = is_channel_broadcast(lhs, rhs)
        if lhs.shape != rhs.shape and not channel_broadcast:
            return False
        if channel_broadcast and (
            not is_channels_last(lhs) or not is_channels_last(rhs)
        ):
            return False
        if op == exir_ops.edge.aten.add.Tensor:
            alpha = args[2] if len(args) > 2 else 1
            try:
                return float(alpha) == 1.0
            except (TypeError, ValueError):
                return False
        return True

    def _supports_float_convolution_replacement(self, conv_node: torch.fx.Node) -> bool:
        # This mirrors the eligibility check in ConvertToCortexMPass.  We must
        # know the producer can become a cortex_m conv before erasing the
        # following activation and storing its clamp bounds on the conv metadata.
        args = conv_node.args
        if len(args) < 9:
            return False
        input_tensor = (
            args[0].meta.get("val") if isinstance(args[0], torch.fx.Node) else None
        )
        weight_tensor = (
            args[1].meta.get("val") if isinstance(args[1], torch.fx.Node) else None
        )
        output_tensor = conv_node.meta.get("val")
        if input_tensor is None or weight_tensor is None or output_tensor is None:
            return False
        dtype = getattr(input_tensor, "dtype", None)
        if dtype not in (torch.float32, torch.float16):
            return False
        if output_tensor.dtype != dtype or weight_tensor.dtype != dtype:
            return False
        if input_tensor.ndim != 4 or output_tensor.ndim != 4 or weight_tensor.ndim != 4:
            return False
        if not is_channels_last(input_tensor):
            return False
        transposed = bool(self._unwrap_argument(args[6]))
        groups = int(self._unwrap_argument(args[8]))
        if transposed:
            return groups == 1
        out_channels = weight_tensor.shape[0]
        in_channels = weight_tensor.shape[1] * groups
        return groups == 1 or is_float_depthwise_conv(in_channels, out_channels, groups)

    def _supports_float_fused_replacement(self, preceding_op: torch.fx.Node) -> bool:
        if preceding_op.target in (
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.mul.Tensor,
        ):
            return self._supports_float_binary_replacement(
                preceding_op.target, preceding_op.args
            )
        if preceding_op.target == exir_ops.edge.aten.convolution.default:
            return self._supports_float_convolution_replacement(preceding_op)
        return False

    def _get_output_min_max_from_activation(
        self, activation_node: torch.fx.Node
    ) -> Optional[tuple[float, float]]:
        if activation_node.target == exir_ops.edge.aten.relu.default:
            return (0.0, sys.float_info.max)
        if activation_node.target == exir_ops.edge.aten.hardtanh.default:
            if len(activation_node.args) > 2:
                try:
                    return (
                        float(activation_node.args[1]),
                        float(activation_node.args[2]),
                    )
                except (TypeError, ValueError):
                    return None
            return (-1.0, 1.0)
        return None

    # -- op replacement methods ------------------------------------------------

    def _get_float_add_replacement(self, args, meta):
        lhs = args[0].data
        rhs = args[1].data
        if lhs.dtype != rhs.dtype:
            return exir_ops.edge.aten.add.Tensor, args
        channel_broadcast = is_channel_broadcast(lhs, rhs)
        if lhs.shape != rhs.shape and not channel_broadcast:
            return exir_ops.edge.aten.add.Tensor, args
        if channel_broadcast and (
            not is_channels_last(lhs) or not is_channels_last(rhs)
        ):
            return exir_ops.edge.aten.add.Tensor, args
        alpha = self._unwrap_argument(args[2]) if len(args) > 2 else 1
        if float(alpha) != 1.0:
            return exir_ops.edge.aten.add.Tensor, args
        activation_min, activation_max = self._get_float_activation_bounds(
            lhs.dtype, meta.data
        )
        if lhs.dtype == torch.float32:
            self._require_float_dtype(lhs.dtype, "cortex_m::add_f32")
            replacement_op = exir_ops.edge.cortex_m.add_f32.default
        elif lhs.dtype == torch.float16:
            self._require_float_dtype(lhs.dtype, "cortex_m::add_f16")
            replacement_op = exir_ops.edge.cortex_m.add_f16.default
        else:
            return exir_ops.edge.aten.add.Tensor, args
        return replacement_op, (args[0], args[1], activation_min, activation_max)

    def _get_float_mul_replacement(self, args, meta):
        lhs = args[0].data
        rhs = args[1].data
        if lhs.dtype != rhs.dtype:
            return exir_ops.edge.aten.mul.Tensor, args
        channel_broadcast = is_channel_broadcast(lhs, rhs)
        if lhs.shape != rhs.shape and not channel_broadcast:
            return exir_ops.edge.aten.mul.Tensor, args
        if channel_broadcast and (
            not is_channels_last(lhs) or not is_channels_last(rhs)
        ):
            return exir_ops.edge.aten.mul.Tensor, args
        activation_min, activation_max = self._get_float_activation_bounds(
            lhs.dtype, meta.data
        )
        if lhs.dtype == torch.float32:
            self._require_float_dtype(lhs.dtype, "cortex_m::mul_f32")
            replacement_op = exir_ops.edge.cortex_m.mul_f32.default
        elif lhs.dtype == torch.float16:
            self._require_float_dtype(lhs.dtype, "cortex_m::mul_f16")
            replacement_op = exir_ops.edge.cortex_m.mul_f16.default
        else:
            return exir_ops.edge.aten.mul.Tensor, args
        return replacement_op, (args[0], args[1], activation_min, activation_max)

    def _get_float_linear_replacement(self, args, meta):
        input_tensor = args[0].data
        weight_tensor = args[1].data
        bias_tensor = args[2].data if len(args) > 2 and args[2] is not None else None

        if input_tensor.dtype not in (torch.float32, torch.float16):
            return exir_ops.edge.aten.linear.default, args
        if weight_tensor.dtype != input_tensor.dtype:
            return exir_ops.edge.aten.linear.default, args
        if input_tensor.dim() < 1 or len(weight_tensor.shape) != 2:
            return exir_ops.edge.aten.linear.default, args
        if input_tensor.shape[-1] != weight_tensor.shape[1]:
            return exir_ops.edge.aten.linear.default, args
        if bias_tensor is not None:
            if bias_tensor.dtype != input_tensor.dtype:
                return exir_ops.edge.aten.linear.default, args
            if (
                len(bias_tensor.shape) != 1
                or bias_tensor.shape[0] != weight_tensor.shape[0]
            ):
                return exir_ops.edge.aten.linear.default, args

        activation_min, activation_max = self._get_float_activation_bounds(
            input_tensor.dtype, meta.data
        )
        replacement_op = (
            exir_ops.edge.cortex_m.linear_f32.default
            if input_tensor.dtype == torch.float32
            else exir_ops.edge.cortex_m.linear_f16.default
        )
        self._require_float_dtype(
            input_tensor.dtype,
            (
                "cortex_m::linear_f32"
                if input_tensor.dtype == torch.float32
                else "cortex_m::linear_f16"
            ),
        )
        return replacement_op, (
            args[0],
            args[1],
            args[2] if len(args) > 2 else None,
            # False means "ordinary OI weight layout". ConvertToCortexMPass may
            # later rewrite constant weights to the packed layout and flip this
            # flag to True:
            #
            #   aten.linear(x, w[O,I]) -> cortex_m::linear(..., packed=False)
            #   late pack pass         -> cortex_m::linear(..., packed=True)
            False,
            0,
            activation_min,
            activation_max,
        )

    def _get_minimum_replacement(self, args, meta):
        if args[0].data.dtype not in (
            torch.int8,
            torch.int32,
            torch.float32,
            torch.float16,
        ):
            return exir_ops.edge.aten.minimum.default, args
        if args[0].data.dtype in (torch.float32, torch.float16):
            self._require_float_dtype(args[0].data.dtype, "cortex_m::minimum")
        return exir_ops.edge.cortex_m.minimum.default, args

    def _get_maximum_replacement(self, args, meta):
        if args[0].data.dtype not in (torch.int8, torch.float32, torch.float16):
            return exir_ops.edge.aten.maximum.default, args
        if args[0].data.dtype in (torch.float32, torch.float16):
            self._require_float_dtype(args[0].data.dtype, "cortex_m::maximum")
        return exir_ops.edge.cortex_m.maximum.default, args

    def _get_permute_replacement(self, args, meta):
        if args[0].data.dtype not in (torch.int8, torch.float32, torch.float16):
            return exir_ops.edge.aten.permute_copy.default, args
        if args[0].data.dtype in (torch.float32, torch.float16):
            self._require_float_dtype(args[0].data.dtype, "cortex_m::transpose")
        rank = len(args[0].data.shape)
        perms = [p % rank for p in args[1]]
        args = (args[0], perms)
        return exir_ops.edge.cortex_m.transpose.default, args

    def _get_float_pad_replacement(self, args):
        input_data = args[0].data
        dtype = getattr(input_data, "dtype", None)
        if dtype not in (torch.float32, torch.float16):
            return exir_ops.edge.aten.constant_pad_nd.default, args

        padding = self._unwrap_argument(args[1])
        pad_value_raw = self._unwrap_argument(args[2]) if len(args) > 2 else 0.0
        pad_value = float(pad_value_raw)

        rank = len(input_data.shape)
        assert 1 <= rank <= 4, f"cortex_m pad: expected rank in [1, 4], got {rank}"
        n_pairs = len(padding) // 2
        assert (
            len(padding) % 2 == 0 and n_pairs <= rank
        ), f"cortex_m pad: invalid padding length {len(padding)} for rank {rank}"

        pre_pad = [0, 0, 0, 0]
        post_pad = [0, 0, 0, 0]
        for i in range(n_pairs):
            # aten.constant_pad_nd lists padding from the innermost dimension
            # outward: [W_left, W_right, H_top, H_bottom, ...].  Normalize this
            # into fixed rank-4 logical NCHW vectors before the base helper maps
            # them to physical NHWC order when needed.
            dim_4d = 3 - i
            pre_pad[dim_4d] = int(padding[2 * i])
            post_pad[dim_4d] = int(padding[2 * i + 1])

        pre_pad = self._to_physical_order(pre_pad, input_data)
        post_pad = self._to_physical_order(post_pad, input_data)

        replacement_op = (
            exir_ops.edge.cortex_m.pad_f32.default
            if dtype == torch.float32
            else exir_ops.edge.cortex_m.pad_f16.default
        )
        self._require_float_dtype(
            dtype,
            "cortex_m::pad_f32" if dtype == torch.float32 else "cortex_m::pad_f16",
        )
        return replacement_op, (args[0], pre_pad, post_pad, pad_value)

    def _get_float_native_batch_norm_replacement(
        self, args: tuple[Argument, ...]
    ) -> tuple[EdgeOpOverload, tuple[Argument, ...]] | None:
        # aten._native_batch_norm_legit_no_training returns a tuple:
        #
        #   (normalized, saved_mean, saved_invstd)
        #
        # The graph usually consumes only getitem(..., 0).  call_operator records
        # the original BN arguments on the tuple node, and call_getitem below
        # replaces getitem(0) with the single-output cortex_m batch-norm op.
        input_arg = args[0]
        if isinstance(input_arg, ProxyValue):
            input_tensor = input_arg.data
        elif hasattr(input_arg, "meta") and "val" in input_arg.meta:
            input_tensor = input_arg.meta["val"]
        else:
            return None
        input_dtype = getattr(input_tensor, "dtype", None)
        if input_dtype not in (torch.float32, torch.float16):
            return None
        if getattr(input_tensor, "ndim", None) not in (2, 4):
            return None

        weight = args[1]
        bias = args[2]
        running_mean = args[3]
        running_var = args[4]
        eps = float(self._unwrap_argument(args[6]))

        if weight is None or bias is None:
            return None

        replacement_op = (
            exir_ops.edge.cortex_m.batch_norm_native_f32.default
            if input_dtype == torch.float32
            else exir_ops.edge.cortex_m.batch_norm_native_f16.default
        )
        self._require_float_dtype(
            input_dtype,
            (
                "cortex_m::batch_norm_native_f32"
                if input_dtype == torch.float32
                else "cortex_m::batch_norm_native_f16"
            ),
        )
        return replacement_op, (
            args[0],
            weight,
            bias,
            running_mean,
            running_var,
            eps,
        )

    # -- pass entry points -----------------------------------------------------

    def call(self, graph_module: torch.fx.GraphModule):
        changed = False
        for activation_node in list(graph_module.graph.nodes):
            # Fuse simple activation consumers before interpreter replay:
            #
            #   producer -> relu/hardtanh -> users
            #
            # becomes:
            #
            #   producer(meta clamp bounds) -> users
            #
            # The producer replacement method then materializes those bounds as
            # explicit activation_min/activation_max arguments.
            if activation_node.op != "call_function":
                continue
            if activation_node.target not in self._FLOAT_FUSEABLE_ACTIVATIONS:
                continue
            preceding_op = activation_node.args[0]
            if not isinstance(preceding_op, torch.fx.Node):
                continue
            if preceding_op.op != "call_function":
                continue
            if preceding_op.target not in self._FLOAT_FUSEABLE_OPS:
                continue
            if len(preceding_op.users) != 1:
                continue
            if not self._supports_float_fused_replacement(preceding_op):
                continue
            output_min_max = self._get_output_min_max_from_activation(activation_node)
            if output_min_max is None:
                continue
            custom_meta = preceding_op.meta.setdefault("custom", {})
            custom_meta[FLOAT_FUSED_ACTIVATION_TAG] = output_min_max
            activation_node.replace_all_uses_with(preceding_op)
            graph_module.graph.erase_node(activation_node)
            changed = True

        if changed:
            graph_module.recompile()

        result = super().call(graph_module)
        return PassResult(result.graph_module, changed or result.modified)

    def call_operator(
        self,
        op: EdgeOpOverload,
        args: tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        match op:
            case exir_ops.edge.aten.add.Tensor:
                op, args = self._get_float_add_replacement(args, meta)
            case exir_ops.edge.aten.mul.Tensor:
                op, args = self._get_float_mul_replacement(args, meta)
            case exir_ops.edge.aten.linear.default:
                op, args = self._get_float_linear_replacement(args, meta)
            case exir_ops.edge.aten.minimum.default:
                op, args = self._get_minimum_replacement(args, meta)
            case exir_ops.edge.aten.maximum.default:
                op, args = self._get_maximum_replacement(args, meta)
            case exir_ops.edge.aten.permute_copy.default:
                op, args = self._get_permute_replacement(args, meta)
            case exir_ops.edge.aten.constant_pad_nd.default:
                op, args = self._get_float_pad_replacement(args)

        # Preserve kwargs for pass-through ops.  This matters for Edge layout
        # helpers such as _clone_dim_order(dim_order=[...]); dropping dim_order
        # turns the clone into preserve_format and can make a later view_copy
        # observe channels-last strides instead of the requested default order.
        result = super().call_operator(op, args, kwargs, meta)

        if op == exir_ops.edge.aten._native_batch_norm_legit_no_training.default:
            # See _get_float_native_batch_norm_replacement: the actual rewrite
            # happens when the interpreter sees getitem(..., 0).
            custom_meta = result.node.meta.setdefault("custom", {})
            custom_meta["cortex_m_native_batch_norm_args"] = args

        return result

    def call_getitem(
        self, value: ProxyValue, key: int, meta: NodeMetadata
    ) -> ProxyValue:
        # Complete the batch-norm tuple rewrite:
        #
        #   bn_tuple = aten._native_batch_norm_legit_no_training(...)
        #   y = operator.getitem(bn_tuple, 0)
        #
        # becomes:
        #
        #   y = cortex_m::batch_norm_native_f*(...)
        #
        # Other tuple outputs are not lowered because inference graphs should not
        # consume saved training statistics.
        if (
            key == 0
            and value.node.target
            == exir_ops.edge.aten._native_batch_norm_legit_no_training.default
        ):
            resolved_args = value.node.meta.get("custom", {}).get(
                "cortex_m_native_batch_norm_args"
            )
            if resolved_args is None:
                return super().call_getitem(value, key, meta)
            replacement = self._get_float_native_batch_norm_replacement(
                cast(tuple[Argument, ...], tuple(resolved_args))
            )
            if replacement is not None:
                op, args = replacement
                return super().call_operator(op, args, {}, meta)

        return super().call_getitem(value, key, meta)
