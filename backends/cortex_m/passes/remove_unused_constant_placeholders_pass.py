# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from executorch.backends.transforms.utils import delete_constant_placeholder
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export.graph_signature import InputKind


class RemoveUnusedConstantPlaceholdersPass(ExportPass):
    """
    Drop stale lifted constants/parameters left behind by Cortex-M rewrites.

    Some passes replace an aten weight with a backend-specific constant, e.g.

        conv.weight [OIHW]   new_weight_nhwc [OHWI]
               |             |
                old aten conv
                       |
                       v
              cortex_m::conv2d(new_weight_nhwc)

    FX dead-code elimination does not erase unused placeholders, so the old
    lifted parameter can remain in the graph signature and serialized program.
    Removing it keeps placeholder order, graph signature, and constant storage
    aligned for the runtime.
    """

    def __init__(self, exported_program) -> None:
        super().__init__()
        self.exported_program = exported_program

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        removable_kinds = {
            InputKind.PARAMETER,
            InputKind.BUFFER,
            InputKind.CONSTANT_TENSOR,
        }
        input_specs_by_name = {
            spec.arg.name: spec
            for spec in self.exported_program.graph_signature.input_specs
        }
        modified = False

        for node in list(graph.nodes):
            if node.op != "placeholder" or node.users:
                continue

            input_spec = input_specs_by_name.get(node.name)
            if input_spec is None or input_spec.kind not in removable_kinds:
                continue

            delete_constant_placeholder(self.exported_program, node)
            modified = True

        if modified:
            graph_module.recompile()

        return PassResult(graph_module, modified)
