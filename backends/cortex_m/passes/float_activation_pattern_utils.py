# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
from executorch.backends.transforms.utils import get_param_tensor, is_param_node
from executorch.exir.dialects._ops import ops as exir_ops


def resolve_scalar_value(exported_program, arg) -> float | None:
    if not isinstance(arg, torch.fx.Node):
        try:
            return float(arg)
        except Exception:
            return None
    if is_param_node(exported_program, arg):
        try:
            value = get_param_tensor(exported_program, arg)
            if value.numel() == 1:
                return float(value.item())
        except Exception:
            pass
    value = arg.meta.get("val")
    if value is None:
        return None
    try:
        if getattr(value, "numel", lambda: 0)() == 1:
            return float(value.item())
    except Exception:
        return None
    return None


def match_clamp_chain(
    clamp2_node: torch.fx.Node,
    exported_program,
    *,
    add_val: float,
    min_val: float,
    max_val: float,
    required_x_node: torch.fx.Node | None = None,
) -> tuple[torch.fx.Node, torch.dtype] | None:
    if clamp2_node.target != exir_ops.edge.aten.clamp.default:
        return None
    if len(clamp2_node.args) < 3 or clamp2_node.args[1] is not None:
        return None
    clamp2_max = resolve_scalar_value(exported_program, clamp2_node.args[2])
    if clamp2_max is None or not math.isclose(
        clamp2_max, max_val, rel_tol=0.0, abs_tol=1e-6
    ):
        return None

    clamp1 = clamp2_node.args[0]
    if (
        not isinstance(clamp1, torch.fx.Node)
        or clamp1.target != exir_ops.edge.aten.clamp.default
        or len(clamp1.args) < 2
    ):
        return None
    clamp1_min = resolve_scalar_value(exported_program, clamp1.args[1])
    if clamp1_min is None or not math.isclose(
        clamp1_min, min_val, rel_tol=0.0, abs_tol=1e-6
    ):
        return None

    add_node = clamp1.args[0]
    if (
        not isinstance(add_node, torch.fx.Node)
        or add_node.target != exir_ops.edge.aten.add.Tensor
    ):
        return None

    x_node = add_node.args[0]
    if not isinstance(x_node, torch.fx.Node):
        return None
    if required_x_node is not None and x_node is not required_x_node:
        return None

    addend_val = resolve_scalar_value(exported_program, add_node.args[1])
    if addend_val is None or not math.isclose(
        addend_val, add_val, rel_tol=0.0, abs_tol=1e-6
    ):
        return None

    input_val = x_node.meta.get("val")
    if input_val is None or input_val.dtype not in (torch.float32, torch.float16):
        return None

    return x_node, input_val.dtype
