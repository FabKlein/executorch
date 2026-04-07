#!/usr/bin/env python3
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class RemoveRedundantCloneDimOrderPass(ExportPass):
    """
    Remove backend graph no-op dim-order copy nodes.

    Some float Cortex-M graphs still contain _clone_dim_order or
    _to_dim_order_copy nodes whose input/output layout metadata is identical.
    Those nodes do not change values, shape, dim order, or stride and are pure
    overhead. Keep this cleanup local to the Cortex-M backend instead of
    relaxing generic runtime behavior.
    """

    _INT8_PAYLOAD_COPY_OPS = {
        torch.ops.aten.clone.default,
        torch.ops.aten.detach_.default,
        torch.ops.aten.lift_fresh_copy.default,
    }

    @classmethod
    def _is_int8_payload_copy(cls, node: torch.fx.Node) -> bool:
        source_fn_stack = node.meta.get("source_fn_stack") or []
        return any(
            source in cls._INT8_PAYLOAD_COPY_OPS
            or (
                isinstance(source, str)
                and source.startswith(("clone", "detach", "lift_fresh_copy"))
            )
            for _, source in source_fn_stack
        )

    @staticmethod
    def _is_redundant_layout_copy(
        node: torch.fx.Node, input_node: torch.fx.Node
    ) -> bool:
        input_val = input_node.meta.get("val")
        output_val = node.meta.get("val")
        if input_val is None or output_val is None:
            return False
        if not (
            hasattr(input_val, "dim_order")
            and hasattr(output_val, "dim_order")
            and hasattr(input_val, "dtype")
            and hasattr(output_val, "dtype")
            and hasattr(input_val, "stride")
            and hasattr(output_val, "stride")
            and hasattr(input_val, "shape")
            and hasattr(output_val, "shape")
        ):
            return False

        # In quantized graphs, a _clone_dim_order node may be the explicit int8
        # payload op under test (for example aten.clone lowered through
        # shared-qspec quantization), or an index/value constant. Preserve
        # non-float copies by default and only prune the known backend no-op
        # shape ``view_copy -> _clone_dim_order`` that can appear before linear.
        if input_val.dtype not in (torch.float16, torch.float32):
            if not (
                input_val.dtype == torch.int8
                and input_node.target == exir_ops.edge.aten.view_copy.default
                and not RemoveRedundantCloneDimOrderPass._is_int8_payload_copy(node)
            ):
                return False

        return (
            tuple(input_val.shape) == tuple(output_val.shape)
            and tuple(input_val.dim_order()) == tuple(output_val.dim_order())
            and tuple(input_val.stride()) == tuple(output_val.stride())
        )

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        changed = False

        for node in list(graph.nodes):
            if node.op != "call_function":
                continue

            input_node = node.args[0]
            if not isinstance(input_node, torch.fx.Node):
                continue

            is_clone = (
                node.target == exir_ops.edge.dim_order_ops._clone_dim_order.default
            )
            is_to_copy = (
                node.target == exir_ops.edge.dim_order_ops._to_dim_order_copy.default
            )
            if not (is_clone or is_to_copy):
                continue

            if is_clone:
                requested_dim_order = node.kwargs.get("dim_order")
                if requested_dim_order:
                    continue
            elif is_to_copy:
                requested_dim_order = node.kwargs.get("dim_order")
                if (
                    requested_dim_order is not None
                    and not self._is_redundant_layout_copy(node, input_node)
                ):
                    continue

            if not self._is_redundant_layout_copy(node, input_node):
                continue

            node.replace_all_uses_with(input_node)
            graph.erase_node(node)
            changed = True

        if changed:
            graph.eliminate_dead_code()
            graph_module.recompile()

        return PassResult(graph_module, changed)
