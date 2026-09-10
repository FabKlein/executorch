# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fold Cortex-M float linear + native batch norm into one linear op.

This pass is part of the Cortex-M CMSIS-NN float lowering pipeline. It looks for
already-lowered Cortex-M linear ops followed by Cortex-M native inference batch
norm and rewrites the constant linear weight/bias offline:

    cortex_m::linear(x, W, b) -> cortex_m::batch_norm_native(...)

becomes:

    cortex_m::linear(x, W_folded, b_folded)

Doing this before linear weight packing keeps the math visible as ordinary
rank-2 weights, avoids a runtime batch-norm op, and keeps the eventual CMSIS-NN
linear call on the packed fast path.
"""

from __future__ import annotations

import torch
from executorch.backends.cortex_m.float_capabilities import (
    get_optional_cortex_m_float_op,
)
from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    get_param_tensor,
    is_param_node,
)
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export.graph_signature import InputKind


class FoldBatchNormIntoLinearPass(ExportPass):
    """Fold dense inference BN into preceding Cortex-M linear constants.

    A linear layer followed by inference batch norm can be rewritten offline:

        y = x @ W.T + b
        z[o] = (y[o] - mean[o]) * scale[o] + beta[o]

    becomes:

        W_folded[o, :] = W[o, :] * scale[o]
        b_folded[o] = (b[o] - mean[o]) * scale[o] + beta[o]

    The pass runs before ConvertToCortexMPass packs linear weights, so it only
    handles ordinary rank-2 weights:

        cortex_m::linear(..., weight[O,I], packed=False)
          -> cortex_m::batch_norm_native(...)

    and leaves already-packed linear constants alone.
    """

    def __init__(self, exported_program) -> None:
        super().__init__()
        self.exported_program = exported_program

    def _get_tensor(self, arg):
        if not isinstance(arg, torch.fx.Node):
            return None
        if is_param_node(self.exported_program, arg):
            return get_param_tensor(self.exported_program, arg)
        value = arg.meta.get("val")
        if isinstance(value, torch.Tensor):
            return value
        return None

    def _optional_targets(self) -> dict[str, object]:
        return {
            name: target
            for name in (
                "linear_f32",
                "linear_f16",
                "batch_norm_native_f32",
                "batch_norm_native_f16",
            )
            if (target := get_optional_cortex_m_float_op(name)) is not None
        }

    def _fold_linear_bn(
        self, linear_node: torch.fx.Node, bn_node: torch.fx.Node
    ) -> bool:
        targets = self._optional_targets()
        if linear_node.target == targets.get("linear_f32"):
            linear_replacement = targets["linear_f32"]
            dtype = torch.float32
        elif linear_node.target == targets.get("linear_f16"):
            linear_replacement = targets["linear_f16"]
            dtype = torch.float16
        else:
            return False

        # Do not try to fold after linear packing.  The packed weight is a
        # backend-specific flat buffer; folding is intentionally done while the
        # math is still visible as W[O, I].
        if (
            len(linear_node.args) >= 4
            and isinstance(linear_node.args[3], bool)
            and linear_node.args[3]
        ):
            return False

        weight_node = linear_node.args[1]
        bias_node = linear_node.args[2]
        weight = self._get_tensor(weight_node)
        if weight is None or weight.ndim != 2:
            return False

        gamma = self._get_tensor(bn_node.args[1])
        beta = self._get_tensor(bn_node.args[2])
        running_mean = self._get_tensor(bn_node.args[3])
        running_var = self._get_tensor(bn_node.args[4])
        eps = float(bn_node.args[5])

        if any(t is None for t in (gamma, beta, running_mean, running_var)):
            return False
        if any(t.ndim != 1 for t in (gamma, beta, running_mean, running_var)):
            return False

        out_features = weight.shape[0]
        if any(
            t.shape[0] != out_features for t in (gamma, beta, running_mean, running_var)
        ):
            return False

        gamma = gamma.to(torch.float32)
        beta = beta.to(torch.float32)
        running_mean = running_mean.to(torch.float32)
        running_var = running_var.to(torch.float32)
        weight_f32 = weight.to(torch.float32)

        if bias_node is None:
            bias_f32 = torch.zeros_like(gamma, dtype=torch.float32)
        else:
            bias = self._get_tensor(bias_node)
            if bias is None or bias.ndim != 1 or bias.shape[0] != out_features:
                return False
            bias_f32 = bias.to(torch.float32)

        scale = gamma / torch.sqrt(running_var + eps)
        folded_weight = weight_f32 * scale.reshape(-1, 1)
        folded_bias = (bias_f32 - running_mean) * scale + beta

        if dtype == torch.float16:
            folded_weight = folded_weight.to(torch.float16)
            folded_bias = folded_bias.to(torch.float16)
        else:
            folded_weight = folded_weight.to(torch.float32)
            folded_bias = folded_bias.to(torch.float32)

        with linear_node.graph.inserting_after(weight_node):
            new_weight = create_constant_placeholder(
                self.exported_program,
                linear_node.graph,
                f"{linear_node.name}_{weight_node.name}_weight_bn_folded",
                InputKind.PARAMETER,
                folded_weight,
            )
        with linear_node.graph.inserting_after(new_weight):
            new_bias = create_constant_placeholder(
                self.exported_program,
                linear_node.graph,
                f"{linear_node.name}_bias_bn_folded",
                InputKind.PARAMETER,
                folded_bias,
            )

        new_args = list(linear_node.args)
        new_args[1] = new_weight
        new_args[2] = new_bias

        with linear_node.graph.inserting_before(linear_node):
            new_linear = linear_node.graph.create_node(
                "call_function",
                linear_replacement,
                args=tuple(new_args),
                kwargs={},
            )
            new_linear.meta = bn_node.meta.copy()
            new_linear.name = bn_node.name

        bn_node.replace_all_uses_with(new_linear)
        linear_node.graph.erase_node(bn_node)
        linear_node.graph.erase_node(linear_node)
        return True

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        graph = graph_module.graph
        # FloatOpRewritePass replaces getitem(native_bn, 0) with the Cortex-M
        # single-output BN op, but the original aten tuple producer can remain
        # briefly as dead code.  Remove it before matching so the linear really
        # appears as:
        #
        #   cortex_m::linear -> cortex_m::batch_norm_native
        #
        # instead of having one live BN user plus one dead aten BN tuple user.
        graph.eliminate_dead_code()

        targets = self._optional_targets()
        batch_norm_targets = {
            target
            for key, target in targets.items()
            if key.startswith("batch_norm_native_")
        }
        linear_targets = {
            target for key, target in targets.items() if key.startswith("linear_")
        }

        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in batch_norm_targets:
                continue
            linear_node = node.args[0]
            if not isinstance(linear_node, torch.fx.Node):
                continue
            if (
                linear_node.op != "call_function"
                or linear_node.target not in linear_targets
            ):
                continue
            if len(linear_node.users) != 1:
                continue
            if self._fold_linear_bn(linear_node, node):
                modified = True

        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()

        return PassResult(graph_module, modified)
