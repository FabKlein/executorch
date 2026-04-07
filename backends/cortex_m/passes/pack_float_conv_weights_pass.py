# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.cortex_m.passes.float_capabilities import (
    get_optional_cortex_m_float_op,
)
from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    get_param_tensor,
    is_param_node,
)
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export.graph_signature import InputKind

from .weight_packing import pack_nt_t_weights_to_nt_n_packed


class PackFloatConvWeightsPass(ExportPass):
    """
    Offline-pack constant Cortex-M float conv weights after batch-norm folding.

    ConvertToCortexMPass keeps conv weights in ordinary OHWI rank-4 form so
    FoldBatchNormIntoConvPass can still rewrite them. This pass runs later and
    converts constant conv weights into the CMSIS-NN NT_N packed RHS layout used
    by the matmul-backed float convolution path.
    """

    def __init__(self, exported_program) -> None:
        super().__init__()
        self.exported_program = exported_program

    def _can_pack_conv_node(self, node: torch.fx.Node) -> bool:
        # The CMSIS-NN packed RHS path is a matmul-oriented contract: every
        # output pixel is computed from an im2row-style input patch matrix
        # multiplied by a flattened filter matrix.
        #
        #   input patch matrix [rows, KH*KW*IC]
        #          x
        #   packed weight RHS [OC, KH*KW*IC]
        #
        # The actual runtime flow still depends on the CMSIS-NN kernel family:
        #
        #   1x1 conv:
        #     each output pixel is a simple IC-wide dot product.
        #
        #   1xN conv with input height == 1:
        #     CMSIS may choose the specialized conv1d helper.
        #
        #   generic KHxKW conv:
        #     CMSIS packs input patches at runtime, then calls the same
        #     matmul-backed helper. Prepacking the constant RHS here removes
        #     the slow NT_T gather pattern from that inner matmul.
        #
        # Keep the eligibility rules aligned with op_float_conv2d.cpp; the
        # Python pass sets weight_is_packed=True, and the C++ kernel validates
        # that the shape can really be consumed as a packed CMSIS RHS.
        if len(node.args) < 13:
            return False

        input_node = node.args[0]
        stride = node.args[3]
        padding = node.args[4]
        dilation = node.args[5]
        if not isinstance(input_node, torch.fx.Node):
            return False

        input_fake = get_first_fake_tensor(input_node)
        output_fake = get_first_fake_tensor(node)
        if (
            input_fake is None
            or input_fake.ndim != 4
            or output_fake is None
            or output_fake.ndim != 4
        ):
            return False

        if bool(node.args[6]):
            return False

        weight_node = node.args[1]
        if not isinstance(weight_node, torch.fx.Node):
            return False
        weight_tensor = get_param_tensor(self.exported_program, weight_node)
        if weight_tensor is None or weight_tensor.ndim != 4:
            return False
        out_channels = int(weight_tensor.shape[0])
        kernel_h = int(weight_tensor.shape[1])
        kernel_w = int(weight_tensor.shape[2])
        weight_in_channels = int(weight_tensor.shape[3])

        if weight_in_channels != int(input_fake.shape[1]):
            return False
        if kernel_h <= 0 or kernel_w <= 0:
            return False

        stride_h, stride_w = int(stride[0]), int(stride[1])
        padding_h, padding_w = int(padding[0]), int(padding[1])
        dilation_h, dilation_w = int(dilation[0]), int(dilation[1])

        if (
            stride_h <= 0
            or stride_w <= 0
            or padding_h < 0
            or padding_w < 0
            or dilation_h != 1
            or dilation_w != 1
        ):
            return False

        # Only pack when CMSIS-NN is guaranteed to stay on a packed-weight
        # path. If CMSIS falls back to the scalar/direct OHWI loop, passing a
        # packed filter would be interpreted as ordinary weights and produce
        # incorrect output.
        if kernel_h == 1 and kernel_w == 1 and padding_h == 0 and padding_w == 0:
            return True

        input_h = int(input_fake.shape[2])
        output_h = int(output_fake.shape[2])
        output_w = int(output_fake.shape[3])
        if (
            int(input_fake.shape[0]) == 1
            and input_h == 1
            and output_h == 1
            and kernel_h == 1
            and kernel_w in (3, 5)
            and stride_h == 1
            and stride_w == 1
            and padding_h == 0
            and padding_w == 0
        ):
            return True

        patch_len = kernel_h * kernel_w * weight_in_channels
        output_positions = output_h * output_w
        return patch_len >= 16 and out_channels >= 8 and output_positions >= 8

    def _pack_weight(self, weight_ohwi: torch.Tensor) -> torch.Tensor:
        # CMSIS matmul-backed float conv expects the filter RHS as a logical
        # [out_channels, kh * kw * in_channels] matrix packed into the
        # NT_N block layout used by arm_nn_mat_mult_nt_n_packed_*.
        #
        # ET still stores conv weights in ordinary OHWI form at this point so
        # earlier passes (especially BN folding) can reason about the original
        # kernel shape. This helper performs the final one-way conversion from
        # ordinary OHWI weights into the flat packed buffer that the runtime
        # kernel can pass directly to CMSIS without repacking again.
        #
        # Shape view:
        #
        #   OHWI weight tensor
        #     [O, H, W, I]
        #         |
        #         | flatten spatial/input dims
        #         v
        #   logical RHS matrix
        #     [O, H*W*I]
        #         |
        #         | pack output rows in blocks of 8 (f16) / 4 (f32)
        #         v
        #   flat packed buffer
        #     [block][k][lane]
        #
        # Tiny example (f16 block size = 8):
        #
        #   rows 0..7, cols 0..3
        #     row0: a0 a1 a2 a3
        #     row1: b0 b1 b2 b3
        #     ...
        #
        #   becomes:
        #     k=0: a0 b0 c0 d0 e0 f0 g0 h0
        #     k=1: a1 b1 c1 d1 e1 f1 g1 h1
        #     ...
        out_channels, kernel_h, kernel_w, in_channels = weight_ohwi.shape
        rhs_nt = weight_ohwi.reshape(out_channels, kernel_h * kernel_w * in_channels)
        return pack_nt_t_weights_to_nt_n_packed(
            rhs_nt, out_channels, int(rhs_nt.shape[1])
        )

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        # This pass runs after ConvertToCortexMPass and after BN folding.
        #
        # By this stage the graph already contains cortex_m::conv2d_f16/f32
        # nodes, and any Conv+BN fusion has finished mutating the rank-4 weight
        # tensors. We can therefore safely replace constant conv weights with a
        # packed rank-1 buffer without breaking earlier graph rewrites.
        #
        # For each constant Cortex-M float conv weight:
        #   1. read the original parameter tensor from the ExportedProgram,
        #   2. pack it offline into CMSIS NT_N format,
        #   3. create a new constant placeholder holding the packed bytes,
        #   4. rewrite the conv node to point at the packed constant and record
        #      the original logical kernel shape in the extra metadata args.
        #
        # The runtime wrapper then sees weight_is_packed=True and calls the
        # packed CMSIS kernel path directly, avoiding runtime weight repacking.
        #
        # Before:
        #   const weight [O,H,W,I] -> cortex_m::conv2d(..., weight_is_packed=False)
        #
        # After:
        #   const packed_weight [flat] -> cortex_m::conv2d(
        #       ...,
        #       weight_is_packed=True,
        #       packed_output_channels=O,
        #       packed_kernel_height=H,
        #       packed_kernel_width=W,
        #       packed_kernel_input_channels=I)
        graph = graph_module.graph
        modified = False

        conv_targets = {
            target
            for name in ("conv2d_f32", "conv2d_f16")
            if (target := get_optional_cortex_m_float_op(name)) is not None
        }

        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in conv_targets:
                continue

            if len(node.args) < 13:
                continue
            if not self._can_pack_conv_node(node):
                continue

            weight_node = node.args[1]
            weight_is_packed = node.args[6]
            if weight_is_packed is True:
                continue
            if not isinstance(weight_node, torch.fx.Node):
                continue
            if not is_param_node(self.exported_program, weight_node):
                continue

            weight_tensor = get_param_tensor(self.exported_program, weight_node)
            if weight_tensor is None or weight_tensor.ndim != 4:
                continue

            out_channels, kernel_h, kernel_w, in_channels = (
                int(weight_tensor.shape[0]),
                int(weight_tensor.shape[1]),
                int(weight_tensor.shape[2]),
                int(weight_tensor.shape[3]),
            )
            packed_weight = self._pack_weight(weight_tensor)

            # Rewrite the graph from a standard OHWI weight to a backend-specific
            # packed constant:
            #
            #   before:
            #
            #     %w_ohwi = placeholder[target=conv_weight_nhwc]
            #     %y = cortex_m::conv2d_f16(
            #         %x, %w_ohwi, ..., weight_is_packed=False, 0, 0, 0, 0)
            #
            #   after:
            #
            #     %w_ohwi = placeholder[target=conv_weight_nhwc]     # now dead
            #     %w_pack = placeholder[target=conv_weight_packed]
            #     %y = cortex_m::conv2d_f16(
            #         %x, %w_pack, ..., weight_is_packed=True, O, H, W, I)
            #
            # A later cleanup pass removes the now-unused %w_ohwi placeholder
            # from both the FX graph and the exported-program graph signature.
            with graph.inserting_after(weight_node):
                packed_weight_node = create_constant_placeholder(
                    self.exported_program,
                    graph,
                    node.name + "_weight_packed",
                    InputKind.PARAMETER,
                    packed_weight,
                )

            new_args = list(node.args)
            new_args[1] = packed_weight_node
            new_args[6] = True
            new_args[7] = out_channels
            new_args[8] = kernel_h
            new_args[9] = kernel_w
            new_args[10] = in_channels
            node.args = tuple(new_args)
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()

        return PassResult(graph_module, modified)
