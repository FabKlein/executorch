# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    get_param_tensor,
    is_param_node,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export.graph_signature import InputKind

from .passes_utils import is_channels_last
from .weight_packing import pack_nt_t_weights_to_nt_n_packed, unpack_nt_n_packed_weights


class BypassFlattenForLinearPass(ExportPass):
    """
    Replace clone_dim_order/view_copy flatten materialization before Cortex-M
    linear with a direct NHWC linear input when safe.

    Pattern:
      4D NHWC tensor
        -> dim_order_ops._clone_dim_order(optional)
        -> aten.view_copy([N, -1] or [N, C*H*W])
        -> cortex_m.linear_*

    The CMSIS-NN float fully-connected API already accepts NHWC input directly,
    so the extra layout clone + flatten copy is pure overhead.
    """

    def __init__(self, exported_program) -> None:
        super().__init__()
        self.exported_program = exported_program

    @staticmethod
    def _nchw_flatten_index(c: int, h: int, w: int, height: int, width: int) -> int:
        return (c * height + h) * width + w

    @classmethod
    def _make_nhwc_to_nchw_permutation(
        cls, channels: int, height: int, width: int
    ) -> torch.Tensor:
        indices = [
            cls._nchw_flatten_index(c, h, w, height, width)
            for h in range(height)
            for w in range(width)
            for c in range(channels)
        ]
        return torch.tensor(indices, dtype=torch.int64)

    @classmethod
    def _will_be_channels_last_after_pool_lowering(cls, node: torch.fx.Node) -> bool:
        """Predict a safe pre-pool-lowering flatten bypass.

        This pass normally requires channels-last metadata on the flatten
        source.  For classifier tails like:

            aten.max_pool2d -> aten.relu -> clone/default -> view_copy -> linear

        running after FloatPoolRewritePass is too late because the pool fake
        output has already become channels-last and the intermediate view_copy
        can fail fake execution.  The pattern is still safe to bypass before
        pool lowering: once max_pool2d is rewritten to cortex_m::max_pool2d_*,
        the runtime tensor will be NHWC, so we reorder the FC weights for
        direct NHWC input now and remove the view_copy.
        """

        if node.op != "call_function":
            return False
        if node.target == exir_ops.edge.aten.max_pool2d.default:
            return True
        if node.target in (
            exir_ops.edge.aten.relu.default,
            exir_ops.edge.dim_order_ops._clone_dim_order.default,
        ):
            input_node = node.args[0] if node.args else None
            return isinstance(
                input_node, torch.fx.Node
            ) and cls._will_be_channels_last_after_pool_lowering(input_node)
        return False

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False

        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            target_str = str(node.target)
            if (
                "cortex_m.linear_f32.default" not in target_str
                and "cortex_m.linear_f16.default" not in target_str
            ):
                continue

            input_node = node.args[0]
            weights_node = node.args[1]
            if not isinstance(input_node, torch.fx.Node) or not isinstance(
                weights_node, torch.fx.Node
            ):
                continue
            if (
                input_node.target
                == exir_ops.edge.dim_order_ops._clone_dim_order.default
            ):
                candidate = input_node.args[0]
                if not isinstance(candidate, torch.fx.Node):
                    continue
                input_node = candidate
            if input_node.target != exir_ops.edge.aten.view_copy.default:
                continue

            flatten_source = input_node.args[0]
            if not isinstance(flatten_source, torch.fx.Node):
                continue

            original_input = flatten_source
            if (
                flatten_source.op == "call_function"
                and flatten_source.target
                == exir_ops.edge.dim_order_ops._clone_dim_order.default
            ):
                candidate = flatten_source.args[0]
                if not isinstance(candidate, torch.fx.Node):
                    continue
                original_input = candidate

            input_val = original_input.meta.get("val")
            view_val = input_node.meta.get("val")
            weights_val = weights_node.meta.get("val")
            output_val = node.meta.get("val")
            if any(v is None for v in (input_val, view_val, weights_val, output_val)):
                continue

            try:
                if view_val.ndim != 2 or output_val.ndim != 2:
                    continue
                if weights_val.ndim not in (1, 2):
                    continue
            except Exception:
                continue

            if input_val.ndim == 2 and tuple(input_val.shape) == tuple(view_val.shape):
                # Simple no-op flatten case:
                #
                #   2D input -> view_copy(same shape) -> linear
                #
                # Just bypass the redundant view.
                node.args = (original_input, *node.args[1:])
                modified = True
                continue

            try:
                if input_val.ndim != 4:
                    continue
                if not is_channels_last(
                    input_val
                ) and not self._will_be_channels_last_after_pool_lowering(
                    original_input
                ):
                    continue
            except Exception:
                continue

            batch = int(input_val.shape[0])
            flat_features = int(math.prod(input_val.shape[1:]))
            weight_is_packed = bool(node.args[3]) if len(node.args) > 3 else False
            packed_out_features = int(node.args[4]) if len(node.args) > 4 else 0
            out_features = (
                packed_out_features if weight_is_packed else int(weights_val.shape[0])
            )
            expected_view_shape = (batch, flat_features)
            expected_out_shape = (batch, out_features)

            if tuple(view_val.shape) != expected_view_shape:
                continue
            if not weight_is_packed and int(weights_val.shape[1]) != flat_features:
                continue
            if tuple(output_val.shape) != expected_out_shape:
                continue

            channels = int(input_val.shape[1])
            height = int(input_val.shape[2])
            width = int(input_val.shape[3])

            if not is_param_node(self.exported_program, weights_node):
                continue

            original_weight = get_param_tensor(self.exported_program, weights_node)
            if original_weight is None:
                continue

            permutation = self._make_nhwc_to_nchw_permutation(channels, height, width)
            if weight_is_packed:
                # After bypassing:
                #
                #   NHWC tensor -> linear
                #
                # instead of:
                #
                #   NHWC tensor -> flatten(NCHW order) -> linear
                #
                # the linear weight must be reordered so it still observes the
                # same logical feature order. Packed weights need one extra
                # round-trip through:
                #
                #   packed -> unpacked [O, I] -> reorder I -> repack
                unpacked_weight = unpack_nt_n_packed_weights(
                    original_weight, out_features, flat_features
                )
                reordered_unpacked_weight = unpacked_weight.index_select(1, permutation)
                reordered_weight = pack_nt_t_weights_to_nt_n_packed(
                    reordered_unpacked_weight,
                    out_features,
                    flat_features,
                )
            else:
                # Standard rank-2 weights only need the feature-dimension
                # permutation:
                #
                #   [O, I_flattened] -> reorder columns -> [O, I_nhwc_direct]
                reordered_weight = original_weight.index_select(1, permutation)

            with graph.inserting_after(weights_node):
                reordered_weight_node = create_constant_placeholder(
                    self.exported_program,
                    graph,
                    node.name + "_weight_nhwc_flatten_order",
                    InputKind.PARAMETER,
                    reordered_weight,
                )

            new_args = (original_input, reordered_weight_node, *node.args[2:])
            node.args = new_args
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()

        return PassResult(graph_module, modified)
