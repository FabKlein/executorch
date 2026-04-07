# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, NodeMetadata, PassResult, ProxyValue

from torch._ops import OpOverload
from torch.fx.node import Argument


class DecomposeMeanPass(ExportPass):
    """
    Rewrites adaptive pooling and spatial mean forms into AvgPool2d-friendly
    classifier tails that CMSIS-NN can lower.
    """

    @staticmethod
    def _unwrap_dims(dim_arg: Argument) -> list[int] | None:
        if dim_arg is None:
            return None
        if isinstance(dim_arg, (list, tuple)):
            return [int(d) for d in dim_arg]
        return None

    def _rewrite_global_mean_chain(
        self, graph_module: torch.fx.GraphModule
    ) -> tuple[torch.fx.GraphModule, bool]:
        graph = graph_module.graph
        modified = False

        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or node.target != exir_ops.edge.aten.mean.dim
                or len(node.args) < 2
            ):
                continue

            second_dims = self._unwrap_dims(cast(Argument, node.args[1]))
            second_keepdim = bool(node.args[2]) if len(node.args) > 2 else False
            if second_dims != [2] or second_keepdim:
                continue

            prev_mean = node.args[0]
            if not isinstance(prev_mean, torch.fx.Node):
                continue
            if (
                prev_mean.op != "call_function"
                or prev_mean.target != exir_ops.edge.aten.mean.dim
                or len(prev_mean.args) < 2
            ):
                continue

            first_dims = self._unwrap_dims(cast(Argument, prev_mean.args[1]))
            first_keepdim = (
                bool(prev_mean.args[2]) if len(prev_mean.args) > 2 else False
            )
            # Match the common classifier tail:
            #
            #   [N, C, H, W]
            #      -> mean(dim=[3], keepdim=False)
            #      -> mean(dim=[2], keepdim=False)
            #      -> [N, C]
            #
            # and normalize it to:
            #
            #   [N, C, H, W]
            #      -> avg_pool2d(kernel=[H, W], stride=[H, W])
            #      -> view_copy([N, C])
            if first_dims not in ([2], [3]) or first_keepdim:
                continue

            original_input = prev_mean.args[0]
            if not isinstance(original_input, torch.fx.Node):
                continue
            input_val = original_input.meta.get("val")
            if input_val is None or len(input_val.shape) != 4:
                continue

            # Use one global avg-pool over the full spatial size so downstream
            # Cortex-M pooling lowering can recognize and replace it with the
            # backend pool op.
            kernel_size = [int(input_val.shape[-2]), int(input_val.shape[-1])]
            avg_args = (
                original_input,
                kernel_size,
                kernel_size,
                [0, 0],
                False,
                False,
                None,
            )

            with graph.inserting_before(node):
                avg_node = graph.create_node(
                    "call_function", exir_ops.edge.aten.avg_pool2d.default, avg_args
                )
                avg_val = exir_ops.edge.aten.avg_pool2d.default(
                    input_val, kernel_size, kernel_size, [0, 0], False, False, None
                )
                avg_node.meta = dict(node.meta)
                avg_node.meta["val"] = avg_val

                # The chained mean form returns [N, C], so finish the rewrite by
                # dropping the trailing 1x1 spatial dimensions.
                output_shape = [int(input_val.shape[0]), int(input_val.shape[1])]
                view_node = graph.create_node(
                    "call_function",
                    exir_ops.edge.aten.view_copy.default,
                    (avg_node, output_shape),
                )
                view_node.meta = dict(node.meta)
                view_node.meta["val"] = exir_ops.edge.aten.view_copy.default(
                    avg_val, output_shape
                )

            node.replace_all_uses_with(view_node)
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()
        return graph_module, modified

    def call_operator(
        self,
        op: OpOverload,
        args: tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op == torch.ops.aten.adaptive_avg_pool2d.default:
            input_tensor = cast(ProxyValue, args[0]).to_tensor()
            shape = input_tensor.shape
            stride = [1, 1]
            kernel_size = [shape[-2], shape[-1]]

            new_args = (args[0], kernel_size, stride, [0, 0], 0, 0)

            adaptive_output = torch.ops.aten.adaptive_avg_pool2d.default(
                input_tensor, *args[1:]
            )
            avg_pool_output = torch.ops.aten.avg_pool2d.default(
                input_tensor, *new_args[1:]
            )

            if adaptive_output.shape == avg_pool_output.shape:
                new_op = torch.ops.aten.avg_pool2d.default
                return super().call_operator(new_op, new_args, kwargs, meta)

        if op == torch.ops.aten.mean.dim:
            decomposed = self._mean_dim_to_avg_pool2d(args, kwargs, meta)
            if decomposed is not None:
                return decomposed

        return super().call_operator(op, args, kwargs, meta)

    def _mean_dim_to_avg_pool2d(
        self,
        args: tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue | None:
        """A mean over both spatial dimensions of NCHW is an average pool
        covering the whole plane. Any other reduction is left alone."""
        # A mean over a constant arrives as a bare tensor rather than a proxy.
        if not isinstance(args[0], ProxyValue):
            return None

        input_tensor = args[0].to_tensor()
        # The rank matters as well as the dims: a 3-D mean([-2, -1]) normalizes
        # to the same pair and is not a spatial reduction.
        if input_tensor.dim() != 4:
            return None

        # The kernel size and the view both take the shape as literals, which a
        # symbolic dimension cannot supply.
        if any(not isinstance(d, int) for d in input_tensor.shape):
            return None

        dims = args[1]
        if not isinstance(dims, (list, tuple)) or not all(
            isinstance(d, int) for d in dims
        ):
            return None
        if sorted(cast(int, d) % 4 for d in dims) != [2, 3]:
            return None

        # dtype= would change the accumulation type, which avg_pool2d cannot do.
        if kwargs.get("dtype") is not None:
            return None

        n, c, h, w = input_tensor.shape
        pooled = super().call_operator(
            torch.ops.aten.avg_pool2d.default,
            (args[0], [h, w], [1, 1], [0, 0], False, False),
            {},
            meta,
        )

        keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        if keepdim:
            return pooled
        # avg_pool2d keeps the spatial dimensions; a mean without keepdim drops
        # them.
        return super().call_operator(
            torch.ops.aten.view.default, (pooled, [n, c]), {}, meta
        )

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        # First do the graph-level mean-chain normalization, then reuse the
        # per-op adaptive_avg_pool2d/spatial mean decomposition.
        graph_module, chain_modified = self._rewrite_global_mean_chain(graph_module)
        result = super().call(graph_module)
        return PassResult(result.graph_module, chain_modified or result.modified)
