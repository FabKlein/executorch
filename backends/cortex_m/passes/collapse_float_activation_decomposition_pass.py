# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Collapse exported float hardswish / hardsigmoid decompositions back into
single Cortex-M activation ops.

PyTorch export decomposes hardswish and hardsigmoid into arithmetic
(add/clamp/clamp/div ± mul).  This pass pattern-matches those subgraphs and
replaces them with ``cortex_m::activation_f{32,16}`` so the graph stays on the
CMSIS-NN float path.

Hardswish is matched first because it wraps the hardsigmoid sub-pattern with
an outer multiply: ``hardswish(x) = hardsigmoid(x) * x``.  Remaining
standalone hardsigmoid nodes are caught in a second sweep.

The two exported shapes handled here are:

    hardsigmoid:
      x -> add(+3) -> clamp_min(0) -> clamp_max(6) -> div(6)

    hardswish:
      x ------------------------------+
                                      |
      x -> add(+3) -> clamp -> div(6) +-> mul

    or equivalently:
      x -----------------------+
                               |
      x -> add(+3) -> clamp ---+-> mul -> div(6)
"""

from __future__ import annotations

import math

import torch
from executorch.backends.cortex_m.float_activation_constants import (
    CMSIS_FLOAT_ACT_HARDSIGMOID,
    CMSIS_FLOAT_ACT_HARDSWISH,
)
from executorch.backends.cortex_m.passes.float_activation_pattern_utils import (
    match_clamp_chain,
    resolve_scalar_value,
)
from executorch.backends.cortex_m.passes.float_capabilities import (
    CortexMFloatCapabilities,
    get_cortex_m_float_capabilities,
    get_optional_cortex_m_float_op,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class CollapseFloatActivationDecompositionPass(ExportPass):
    """Collapse decomposed float hardswish / hardsigmoid back into
    ``cortex_m::activation_f{32,16}``."""

    def __init__(
        self,
        exported_program,
        capabilities: CortexMFloatCapabilities | None = None,
    ) -> None:
        super().__init__()
        self.exported_program = exported_program
        self.capabilities = capabilities or get_cortex_m_float_capabilities()

    # ------------------------------------------------------------------
    # Pattern helpers
    # ------------------------------------------------------------------

    def _enabled_mul_targets(self) -> tuple[object, ...]:
        # Hardswish can be collapsed before or after generic float mul lowering:
        #
        #   early pipeline:  aten.mul(x, hardsigmoid(x))
        #   late pipeline:   cortex_m::mul_f16/f32(x, hardsigmoid(x), ...)
        #
        # Accept the raw aten form so a f16 export does not briefly require
        # f32 elementwise support just because export materialized hardswish
        # constants in float32.
        targets = [exir_ops.edge.aten.mul.Tensor]
        if self.capabilities.is_float_dtype_enabled(torch.float16):
            op = get_optional_cortex_m_float_op("mul_f16")
            if op is not None:
                targets.append(op)
        if self.capabilities.is_float_dtype_enabled(torch.float32):
            op = get_optional_cortex_m_float_op("mul_f32")
            if op is not None:
                targets.append(op)
        return tuple(targets)

    def _match_hardsigmoid_div(
        self, node: torch.fx.Node
    ) -> tuple[torch.fx.Node, torch.dtype] | None:
        """Match ``clamp(clamp(x + 3, min=0), max=6) / 6``."""
        if node.op != "call_function" or node.target != exir_ops.edge.aten.div.Tensor:
            return None

        divisor_val = resolve_scalar_value(self.exported_program, node.args[1])
        if divisor_val is None or not math.isclose(
            divisor_val, 6.0, rel_tol=0.0, abs_tol=1e-6
        ):
            return None

        clamp2 = node.args[0]
        if not isinstance(clamp2, torch.fx.Node):
            return None
        # Delegate the clamp/add details to the shared helper so hardsigmoid and
        # hardswish keep exactly one definition of the "x + 3, clamp 0..6"
        # sub-pattern.
        return match_clamp_chain(
            clamp2,
            self.exported_program,
            add_val=3.0,
            min_val=0.0,
            max_val=6.0,
        )

    @staticmethod
    def _unwrap_promoted_hardswish_input(
        x_node: torch.fx.Node, dtype: torch.dtype
    ) -> tuple[torch.fx.Node, torch.dtype]:
        """Recover the original f16 tensor from export's hardswish f32 island.

        Some TorchScript exports implement f16 hardswish by first promoting the
        activation input to f32:

            x_f16 -> _to_dim_order_copy(dtype=f32) -> add/clamp/mul/div

        The following linear still expects f16 weights/input, so collapsing this
        pattern to activation_f32 would create a fake mixed-dtype backend graph.
        If the promote source is an f16 tensor with the same shape, use it as
        the activation input and lower to activation_f16.
        """

        if dtype != torch.float32:
            return x_node, dtype
        if (
            x_node.op != "call_function"
            or x_node.target != exir_ops.edge.dim_order_ops._to_dim_order_copy.default
            or x_node.kwargs.get("dtype") != torch.float32
        ):
            return x_node, dtype
        source = x_node.args[0]
        if not isinstance(source, torch.fx.Node):
            return x_node, dtype
        source_val = source.meta.get("val")
        promoted_val = x_node.meta.get("val")
        if not isinstance(source_val, torch.Tensor) or not isinstance(
            promoted_val, torch.Tensor
        ):
            return x_node, dtype
        if source_val.dtype != torch.float16:
            return x_node, dtype
        if tuple(source_val.shape) != tuple(promoted_val.shape):
            return x_node, dtype
        return source, torch.float16

    # -- hardswish: mul(hardsigmoid(x), x)  -or-  div(mul(x, clamp(…)), 6) --

    def _match_hardswish_mul(
        self, node: torch.fx.Node
    ) -> tuple[torch.fx.Node, torch.dtype] | None:
        if node.op != "call_function" or node.target not in self._enabled_mul_targets():
            return None

        lhs, rhs = node.args[0], node.args[1]
        if not isinstance(lhs, torch.fx.Node) or not isinstance(rhs, torch.fx.Node):
            return None

        lhs_match = self._match_hardsigmoid_div(lhs)
        if lhs_match is not None and rhs is lhs_match[0]:
            return self._unwrap_promoted_hardswish_input(*lhs_match)

        rhs_match = self._match_hardsigmoid_div(rhs)
        if rhs_match is not None and lhs is rhs_match[0]:
            return self._unwrap_promoted_hardswish_input(*rhs_match)

        return None

    def _match_hardswish_div(
        self, node: torch.fx.Node
    ) -> tuple[torch.fx.Node, torch.dtype] | None:
        if node.op != "call_function" or node.target != exir_ops.edge.aten.div.Tensor:
            return None

        divisor_val = resolve_scalar_value(self.exported_program, node.args[1])
        if divisor_val is None or not math.isclose(
            divisor_val, 6.0, rel_tol=0.0, abs_tol=1e-6
        ):
            return None

        mul_node = node.args[0]
        if not isinstance(mul_node, torch.fx.Node):
            return None
        if mul_node.target not in self._enabled_mul_targets():
            return None

        x_node, clamp2 = mul_node.args[0], mul_node.args[1]
        if not isinstance(x_node, torch.fx.Node) or not isinstance(
            clamp2, torch.fx.Node
        ):
            return None
        match = match_clamp_chain(
            clamp2,
            self.exported_program,
            add_val=3.0,
            min_val=0.0,
            max_val=6.0,
            required_x_node=x_node,
        )
        if match is None:
            return None
        return self._unwrap_promoted_hardswish_input(*match)

    # ------------------------------------------------------------------
    # Graph rewrite
    # ------------------------------------------------------------------

    def _replace_node(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        x_node: torch.fx.Node,
        dtype: torch.dtype,
        act_code: int,
    ) -> None:
        # Replace only the root of the matched arithmetic tree.  The internal
        # add/clamp/div/mul nodes become dead and are removed by eliminate_dead_code.
        #
        #   before: users -> div/mul(...decomposed activation...)
        #   after:  users -> cortex_m::activation_f{16,32}(x, act_code)
        op_name = "activation_f32" if dtype == torch.float32 else "activation_f16"
        self.capabilities.require_float_dtype_enabled(dtype, f"cortex_m::{op_name}")
        replacement_op = get_optional_cortex_m_float_op(op_name)
        with graph.inserting_before(node):
            new_node = graph.create_node(
                "call_function",
                replacement_op,
                args=(x_node, act_code, 0.0),
                kwargs={},
            )
            new_node.meta = node.meta.copy()
            new_node.name = node.name
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False

        # First sweep: hardswish (longer pattern, consumes the hardsigmoid
        # sub-pattern so it won't false-match in the second sweep).
        for node in list(graph.nodes):
            match = self._match_hardswish_mul(node)
            if match is None:
                match = self._match_hardswish_div(node)
            if match is None:
                continue
            self._replace_node(graph, node, *match, CMSIS_FLOAT_ACT_HARDSWISH)
            modified = True

        # Second sweep: standalone hardsigmoid.
        for node in list(graph.nodes):
            match = self._match_hardsigmoid_div(node)
            if match is None:
                continue
            self._replace_node(graph, node, *match, CMSIS_FLOAT_ACT_HARDSIGMOID)
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()

        return PassResult(graph_module, modified)
