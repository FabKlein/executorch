# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unified dim-order normalization for the Cortex-M backend.

All layout fixups live here instead of being spread across per-pattern passes.
The pass is safe to run multiple times (idempotent) because each handler only
fires when its specific pattern is present.

The pass manager invokes it at two pipeline positions:

  * **Early** (before op substitution): catches alias_copy layout drift and
    default-order float convolution inputs that originate from the exported
    graph.

  * **Late** (after BN fold / layout-sensitive transforms): catches
    as_strided_copy and view_copy patterns that are only introduced by
    intermediate rewrites.
"""

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

_NHWC_DIM_ORDER = (0, 2, 3, 1)
_DEFAULT_DIM_ORDER = (0, 1, 2, 3)


def _get_dim_order(val: torch.Tensor) -> tuple[int, ...] | None:
    try:
        return tuple(val.dim_order())
    except Exception:
        return None


def _make_clone_node(
    graph: torch.fx.Graph,
    anchor: torch.fx.Node,
    input_node: torch.fx.Node,
    dim_order: list[int],
    meta_val: torch.Tensor,
    *,
    insert_after: bool = False,
) -> torch.fx.Node:
    ctx = (
        graph.inserting_after(anchor)
        if insert_after
        else graph.inserting_before(anchor)
    )
    with ctx:
        clone = graph.create_node(
            "call_function",
            exir_ops.edge.dim_order_ops._clone_dim_order.default,
            args=(input_node,),
            kwargs={"non_blocking": False, "dim_order": dim_order},
        )
        clone.meta = input_node.meta.copy()
        clone.meta["val"] = meta_val
    return clone


class NormalizeDimOrderPass(ExportPass):
    """Ensure tensors have the memory layout that Cortex-M backend ops expect.

    Handles four layout-normalization patterns in a single graph walk:

    1. **alias_copy layout preservation** — ``aten.alias_copy`` can
       re-materialize a channels-last tensor in default dim order.  Replace it
       with ``_clone_dim_order`` that preserves the producer's layout so
       downstream consumers (e.g. convolution) still see NHWC.

    2. **Float conv NHWC materialization** — float Cortex-M conv kernels need
       NHWC input.  If a ``convolution`` node's input is still in default NCHW
       order, insert an explicit ``_clone_dim_order`` to channels-last.

    3. **as_strided_copy replacement** — ``aten.as_strided_copy`` used as a
       layout-only identity (same shape, 1×1 spatial) is replaced with an
       explicit ``_clone_dim_order`` to the target layout.

    4. **view_copy default-order materialization** — a ``view_copy`` from 4-D
       channels-last to 2-D (e.g. classifier tail) needs the input in default
       order first.  Insert or patch a preceding ``_clone_dim_order``.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False

        modified |= self._fix_alias_copy(graph)
        modified |= self._fix_float_conv_inputs(graph)
        modified |= self._fix_as_strided_copy(graph)
        modified |= self._fix_view_copy_inputs(graph)

        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()
        return PassResult(graph_module, modified)

    # ------------------------------------------------------------------
    # 1. alias_copy layout preservation
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_alias_copy(graph: torch.fx.Graph) -> bool:
        modified = False
        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or node.target != exir_ops.edge.aten.alias_copy.default
            ):
                continue

            input_node = node.args[0]
            if not isinstance(input_node, torch.fx.Node):
                continue
            input_val = input_node.meta.get("val")
            output_val = node.meta.get("val")
            if input_val is None or output_val is None:
                continue
            if not (
                hasattr(input_val, "dim_order") and hasattr(output_val, "dim_order")
            ):
                continue
            if input_val.ndim != 4 or output_val.ndim != 4:
                continue
            if tuple(input_val.shape) != tuple(output_val.shape):
                continue

            input_dim_order = _get_dim_order(input_val)
            output_dim_order = _get_dim_order(output_val)
            if input_dim_order is None or input_dim_order == output_dim_order:
                continue

            if input_dim_order == _NHWC_DIM_ORDER:
                meta_val = input_val.contiguous(memory_format=torch.channels_last)
            elif input_dim_order == _DEFAULT_DIM_ORDER:
                meta_val = input_val.contiguous()
            else:
                meta_val = input_val

            clone = _make_clone_node(
                graph,
                node,
                input_node,
                list(input_dim_order),
                meta_val,
                insert_after=True,
            )
            clone.meta = node.meta.copy()
            clone.meta["val"] = meta_val
            node.replace_all_uses_with(clone)
            graph.erase_node(node)
            modified = True
        return modified

    # ------------------------------------------------------------------
    # 2. Float conv NHWC materialization
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_float_conv_inputs(graph: torch.fx.Graph) -> bool:
        modified = False
        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or node.target != exir_ops.edge.aten.convolution.default
            ):
                continue

            input_node = node.args[0]
            if not isinstance(input_node, torch.fx.Node):
                continue
            input_val = input_node.meta.get("val")
            if not isinstance(input_val, torch.Tensor):
                continue
            if input_val.ndim != 4 or input_val.dtype not in (
                torch.float32,
                torch.float16,
            ):
                continue

            dim_order = _get_dim_order(input_val)
            if dim_order != _DEFAULT_DIM_ORDER:
                continue

            meta_val = input_val.contiguous(memory_format=torch.channels_last)
            clone = _make_clone_node(
                graph,
                node,
                input_node,
                list(_NHWC_DIM_ORDER),
                meta_val,
            )
            new_args = list(node.args)
            new_args[0] = clone
            node.args = tuple(new_args)
            modified = True
        return modified

    # ------------------------------------------------------------------
    # 3. as_strided_copy replacement
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_as_strided_copy(graph: torch.fx.Graph) -> bool:
        modified = False
        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or node.target != exir_ops.edge.aten.as_strided_copy.default
            ):
                continue

            input_node = node.args[0]
            if not isinstance(input_node, torch.fx.Node):
                continue
            input_val = input_node.meta.get("val")
            output_val = node.meta.get("val")
            if input_val is None or output_val is None:
                continue
            if input_val.ndim != 4 or output_val.ndim != 4:
                continue
            if tuple(input_val.shape) != tuple(output_val.shape):
                continue
            if tuple(input_val.shape[-2:]) != (1, 1):
                continue

            input_dim_order = _get_dim_order(input_val)
            output_dim_order = _get_dim_order(output_val)
            if input_dim_order is None or output_dim_order is None:
                continue
            if input_dim_order == _DEFAULT_DIM_ORDER:
                continue
            if output_dim_order != _DEFAULT_DIM_ORDER:
                continue

            clone = _make_clone_node(
                graph,
                node,
                input_node,
                list(_DEFAULT_DIM_ORDER),
                output_val,
                insert_after=True,
            )
            clone.meta = node.meta.copy()
            node.replace_all_uses_with(clone)
            modified = True
        return modified

    # ------------------------------------------------------------------
    # 4. view_copy default-order materialization
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_view_copy_inputs(graph: torch.fx.Graph) -> bool:
        modified = False
        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or node.target != exir_ops.edge.aten.view_copy.default
            ):
                continue

            input_node = node.args[0]
            if not isinstance(input_node, torch.fx.Node):
                continue
            input_val = input_node.meta.get("val")
            output_val = node.meta.get("val")
            if input_val is None or output_val is None:
                continue
            if input_val.ndim != 4 or output_val.ndim != 2:
                continue

            input_dim_order = _get_dim_order(input_val)
            if input_dim_order is None:
                continue

            # If the input is already a _clone_dim_order with no requested
            # order, patch it in place to request default order.  If it already
            # requests default order, the view is protected and there is
            # nothing more to do.
            if (
                input_node.op == "call_function"
                and input_node.target
                == exir_ops.edge.dim_order_ops._clone_dim_order.default
            ):
                requested_dim_order = input_node.kwargs.get("dim_order")
                if requested_dim_order == list(_DEFAULT_DIM_ORDER):
                    if isinstance(input_val, torch.Tensor):
                        input_node.meta = input_node.meta.copy()
                        input_node.meta["val"] = input_val.contiguous()
                    continue
                if not requested_dim_order:
                    input_node.kwargs = {
                        "non_blocking": False,
                        "dim_order": list(_DEFAULT_DIM_ORDER),
                    }
                    if isinstance(input_val, torch.Tensor):
                        input_node.meta = input_node.meta.copy()
                        input_node.meta["val"] = input_val.contiguous()
                    modified = True
                    continue
            else:
                meta_val = (
                    input_val.contiguous()
                    if isinstance(input_val, torch.Tensor)
                    else input_val
                )
                # Always materialize default order before flatten/view from 4-D
                # to 2-D, even if the current metadata still says default
                # order.  Earlier passes may later replace an upstream conv/pool
                # with a Cortex-M op whose fake/meta output is channels-last;
                # having the explicit clone already in the graph keeps the
                # subsequent view_copy legal during ExportPass fake execution
                # and keeps the graph convertible to ExecuTorch out variants:
                #
                #   NHWC producer -> relu -> clone(default) -> view_copy [N, C*H*W]
                #
                # Without the clone, fake tensor view validation can fail on a
                # channels-last-strided tensor before the late normalization pass
                # gets a chance to run.
                clone = _make_clone_node(
                    graph,
                    node,
                    input_node,
                    list(_DEFAULT_DIM_ORDER),
                    meta_val,
                )
                node.replace_input_with(input_node, clone)
                modified = True
        return modified
