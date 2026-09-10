# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export.graph_signature import InputKind


class FoldBatchNormIntoConvPass(ExportPass):
    """
    Fold cortex_m::batch_norm_native_* into the preceding cortex_m conv/depthwise
    conv by rewriting the weight and bias constants offline.
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
                "conv2d_f32",
                "conv2d_f16",
                "depthwise_conv2d_f32",
                "depthwise_conv2d_f16",
                "batch_norm_native_f32",
                "batch_norm_native_f16",
            )
            if (target := get_optional_cortex_m_float_op(name)) is not None
        }

    def _fold_conv_bn(self, conv_node: torch.fx.Node, bn_node: torch.fx.Node) -> bool:
        targets = self._optional_targets()
        if conv_node.target == targets.get("conv2d_f32"):
            conv_replacement = targets["conv2d_f32"]
            dtype = torch.float32
            weight_scale_dim = 0
        elif conv_node.target == targets.get("conv2d_f16"):
            conv_replacement = targets["conv2d_f16"]
            dtype = torch.float16
            weight_scale_dim = 0
        elif conv_node.target == targets.get("depthwise_conv2d_f32"):
            conv_replacement = targets["depthwise_conv2d_f32"]
            dtype = torch.float32
            weight_scale_dim = 3
        elif conv_node.target == targets.get("depthwise_conv2d_f16"):
            conv_replacement = targets["depthwise_conv2d_f16"]
            dtype = torch.float16
            weight_scale_dim = 3
        else:
            return False

        weight_node = conv_node.args[1]
        bias_node = conv_node.args[2]
        weight = self._get_tensor(weight_node)
        if weight is None:
            return False

        gamma = self._get_tensor(bn_node.args[1])
        beta = self._get_tensor(bn_node.args[2])
        running_mean = self._get_tensor(bn_node.args[3])
        running_var = self._get_tensor(bn_node.args[4])
        eps = float(bn_node.args[5])

        if any(t is None for t in (gamma, beta, running_mean, running_var)):
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
            if bias is None:
                return False
            bias_f32 = bias.to(torch.float32)

        scale = gamma / torch.sqrt(running_var + eps)

        reshape = [1] * weight_f32.ndim
        reshape[weight_scale_dim] = scale.numel()
        folded_weight = weight_f32 * scale.reshape(reshape)
        folded_bias = (bias_f32 - running_mean) * scale + beta

        if dtype == torch.float16:
            folded_weight = folded_weight.to(torch.float16)
            folded_bias = folded_bias.to(torch.float16)
        else:
            folded_weight = folded_weight.to(torch.float32)
            folded_bias = folded_bias.to(torch.float32)

        with conv_node.graph.inserting_after(weight_node):
            new_weight = create_constant_placeholder(
                self.exported_program,
                conv_node.graph,
                conv_node.name + "_weight_bn_folded",
                InputKind.PARAMETER,
                folded_weight,
            )
        with conv_node.graph.inserting_after(new_weight):
            new_bias = create_constant_placeholder(
                self.exported_program,
                conv_node.graph,
                conv_node.name + "_bias_bn_folded",
                InputKind.PARAMETER,
                folded_bias,
            )

        new_args = list(conv_node.args)
        new_args[1] = new_weight
        new_args[2] = new_bias

        with conv_node.graph.inserting_before(conv_node):
            new_conv = conv_node.graph.create_node(
                "call_function",
                conv_replacement,
                args=tuple(new_args),
                kwargs={},
            )
            new_conv.meta = bn_node.meta.copy()
            new_conv.name = bn_node.name

        bn_node.replace_all_uses_with(new_conv)
        conv_node.graph.erase_node(bn_node)
        conv_node.graph.erase_node(conv_node)
        return True

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        graph = graph_module.graph

        targets = self._optional_targets()
        batch_norm_targets = {
            target
            for key, target in targets.items()
            if key.startswith("batch_norm_native_")
        }
        conv_targets = {
            target
            for key, target in targets.items()
            if key.startswith("conv2d_") or key.startswith("depthwise_conv2d_")
        }

        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in batch_norm_targets:
                continue
            conv_node = node.args[0]
            if not isinstance(conv_node, torch.fx.Node):
                continue
            if conv_node.op != "call_function" or conv_node.target not in conv_targets:
                continue
            if len(conv_node.users) != 1:
                continue
            if self._fold_conv_bn(conv_node, node):
                modified = True

        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()

        return PassResult(graph_module, modified)
