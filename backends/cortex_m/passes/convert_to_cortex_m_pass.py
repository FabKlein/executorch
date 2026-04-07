# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cortex-M graph-rewrite pass for ops that need graph surgery.

Handles float conv, linear, and BMM rewrites that create constant placeholders
or pack weights. These cannot live in call_operator-based passes because
ExportPass.call_operator() can replace targets and arguments but cannot create
new graph nodes. Quantized lowering is owned by AtenToCortexMPass.
"""

import executorch.backends.cortex_m.ops.operators  # noqa

import torch
import torch.fx
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.cortex_m.passes.float_capabilities import (
    CortexMFloatCapabilities,
    get_cortex_m_float_capabilities,
)
from executorch.backends.cortex_m.passes.passes_utils import (
    is_channels_last,
    is_float_depthwise_conv,
    quantize_multiplier_aot,
)
from executorch.backends.cortex_m.passes.weight_packing import (
    pack_nt_t_weights_to_nt_n_packed,
)

from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    get_param_tensor,
    is_param_node,
)

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.export.graph_signature import InputKind
from torch.fx.passes.infra.pass_manager import PassResult


class ConvertToCortexMPass(ExportPass):
    """Lower float conv / linear / BMM to cortex_m ops via graph surgery."""

    _FLOAT_FUSED_ACTIVATION_TAG = "cortex_m_float_activation"

    def __init__(
        self,
        exported_program,
        capabilities: CortexMFloatCapabilities | None = None,
    ) -> None:
        super().__init__()
        self.exported_program = exported_program
        self.capabilities = capabilities or get_cortex_m_float_capabilities()

    def _require_float_dtype(self, dtype: torch.dtype, op_name: str) -> None:
        self.capabilities.require_float_dtype_enabled(dtype, op_name)

    def _get_default_float_activation_bounds(
        self, dtype: torch.dtype
    ) -> tuple[float, float]:
        finfo = torch.finfo(dtype)
        return float(finfo.min), float(finfo.max)

    def _get_float_activation_bounds(
        self, dtype: torch.dtype, node_meta: dict
    ) -> tuple[float, float]:
        default_min, default_max = self._get_default_float_activation_bounds(dtype)
        custom_meta = node_meta.get("custom", {})
        bounds = custom_meta.get(self._FLOAT_FUSED_ACTIVATION_TAG)
        if bounds is None:
            return default_min, default_max

        activation_min, activation_max = bounds
        activation_min = (
            default_min if activation_min is None else float(activation_min)
        )
        activation_max = (
            default_max if activation_max is None else float(activation_max)
        )
        return max(default_min, activation_min), min(default_max, activation_max)

    def _compute_kernel_sum(self, weights, bias, input_offset, weight_offset):
        """
        Computes the precomputed kernel sum term (bias optional)
            a * sum_j(wij + b) + ci

        for i = (1, ..., n), where j indexes the input activations.
        """
        weights_transposed = weights.T
        weights_int32 = weights_transposed.to(torch.int32)
        offset_weights = weights_int32 + weight_offset
        kernel_sum = torch.sum(offset_weights, dim=0, keepdim=True, dtype=torch.int32)
        kernel_sum_offset = kernel_sum * input_offset

        if bias is not None:
            kernel_sum_offset += bias

        return kernel_sum_offset

    def _get_batch_size_from_conv(self, conv_node: torch.fx.Node):
        """
        Extract batch size from convolution node's output shape.

        Returns None if shape metadata is unavailable, which can occur when
        processing nodes created earlier in the same pass iteration.

        For Conv2d operations, output_batch_size always equals input_batch_size.
        Conv2d outputs are always 4D (N, C, H, W) in the edge dialect.
        """
        try:
            if "val" in conv_node.meta:
                output_shape = conv_node.meta["val"].shape
                return output_shape[0]
        except (AttributeError, TypeError):
            pass
        return None

    def _get_float_linear_replacement(self, node):
        if (
            len(node.args) >= 4
            and isinstance(node.args[3], bool)
            and node.args[3] is True
        ):
            return None

        input_node = node.args[0]
        weight_node = node.args[1]
        bias_node = node.args[2] if len(node.args) > 2 else None

        input_tensor = get_first_fake_tensor(input_node)
        weight_tensor = get_param_tensor(self.exported_program, weight_node)
        if input_tensor is None or weight_tensor is None:
            return None

        if input_tensor.dtype not in (torch.float32, torch.float16):
            return None
        if weight_tensor.dtype != input_tensor.dtype or weight_tensor.dim() != 2:
            return None

        out_features = int(weight_tensor.shape[0])
        activation_min = float(node.args[-2])
        activation_max = float(node.args[-1])

        # For constant float linear weights we can repack once during lowering
        # instead of paying the conversion cost every time the runtime kernel
        # executes.
        #
        # Before:
        #   aten.linear(input, weight[O, I], bias)
        #
        # After:
        #   cortex_m::linear_f16/f32(
        #       input,
        #       packed_weight[flat],
        #       bias,
        #       weight_is_packed=True,
        #       packed_out_features=O,
        #       activation_min,
        #       activation_max)
        #
        # The runtime wrapper then sees weight_is_packed=True and directly uses
        # the packed CMSIS fully-connected path without repacking the constant
        # RHS again.
        packed_weight = pack_nt_t_weights_to_nt_n_packed(
            weight_tensor, out_features, int(weight_tensor.shape[1])
        )
        with node.graph.inserting_after(weight_node):
            packed_weight_node = create_constant_placeholder(
                self.exported_program,
                node.graph,
                node.name + "_weight_packed",
                InputKind.PARAMETER,
                packed_weight,
            )

        return node.target, (
            input_node,
            packed_weight_node,
            bias_node,
            True,
            out_features,
            activation_min,
            activation_max,
        )

    def _get_linear_replacement(self, node):
        """
        Let
        - yi be the output activations (y1, ... yn)
        - xj be the input activations (x1, ... xm)
        - wij be the weights (w11, ... wnm)
        - a be the input offset
        - b be the weight offset
        - ci be the bias

        Then the linear operation can be written as:
        yi = sum_j((xj + a) * (wij + b)) + ci
        = sum_j(xj*wij + xj*b + a*wij + a*b) + ci
        = sum_j(xj*wij) + sum_j(xj)*b + (a * sum_j(wij + b) + ci)
        = sum_j(xj*wij) + sum_j(xj)*b + kernel_sum

        where kernel_sum is precomputed aot.
        """
        input_scale = node.meta["input_qparams"][0].scale
        input_zp = node.meta["input_qparams"][0].zp
        weight_scale = node.meta["input_qparams"][1].scale
        weight_zp = node.meta["input_qparams"][1].zp
        output_scale = node.meta["output_qparams"][0].scale
        output_zp = node.meta["output_qparams"][0].zp
        output_min = node.meta["output_qparams"][0].qmin
        output_max = node.meta["output_qparams"][0].qmax

        quantized_multiplier, quantized_shift = quantize_multiplier_aot(
            (input_scale * weight_scale) / output_scale
        )

        # TODO: Add support for configuring the backend to support other extensions.
        # Kernel sum is only used in the CMSIS-NN implementation for the MVE extension,
        # so this should be optional.
        weights = node.args[1]
        weights_tensor = get_param_tensor(self.exported_program, weights)
        bias_tensor = (
            get_param_tensor(self.exported_program, node.args[2])
            if len(node.args) > 2
            else None
        )
        kernel_sum_tensor = self._compute_kernel_sum(
            weights_tensor, bias_tensor, -input_zp, -weight_zp
        )
        with node.graph.inserting_after(weights):
            kernel_sum = create_constant_placeholder(
                self.exported_program,
                node.graph,
                node.name + "_kernel_sum",
                InputKind.PARAMETER,
                kernel_sum_tensor,
            )

        args = (
            node.args[0],
            weights,
            None,
            kernel_sum,
            -input_zp,
            -weight_zp,
            output_zp,
            [quantized_multiplier],
            [quantized_shift],
            output_max,
            output_min,
        )

        return exir_ops.edge.cortex_m.quantized_linear.default, args

    def _get_convolution_replacement(self, node):
        (
            x,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        ) = node.args

        input_scale = node.meta["input_qparams"][0].scale
        input_zero_point = node.meta["input_qparams"][0].zp
        weight_scales = node.meta["input_qparams"][1].scale
        if not isinstance(weight_scales, list):
            fake_weight_tensor = get_first_fake_tensor(weight)
            weight_scales = [weight_scales] * fake_weight_tensor.shape[0]

        output_qparams = node.meta["output_qparams"][0]
        output_scale = output_qparams.scale
        output_zero_point = output_qparams.zp
        output_qmin = output_qparams.qmin
        output_qmax = output_qparams.qmax

        quantized_multipliers = []
        quantized_shifts = []
        for weight_scale in weight_scales:
            quantized_multiplier, quantized_shift = quantize_multiplier_aot(
                input_scale * weight_scale / output_scale
            )
            quantized_multipliers.append(quantized_multiplier)
            quantized_shifts.append(quantized_shift)

        param_weight_tensor = get_param_tensor(self.exported_program, weight)
        if param_weight_tensor is None:
            raise RuntimeError(
                f"Expected convolution weight parameter tensor for node {node.name}."
            )

        # Detect depthwise convolution:
        # Depthwise means groups == in_channels, out_channels == K * in_channels
        # Weight shape is [out_ch, in_ch_per_group, H, W]
        in_channels = param_weight_tensor.shape[1] * groups
        out_channels = param_weight_tensor.shape[0]
        is_depthwise = (
            groups > 1 and (in_channels == groups) and (out_channels % in_channels == 0)
        )

        # Only use DW path if batch_size==1, as CMSIS-NN DW falls back to
        # unoptimized implementation otherwise.
        batch_size = self._get_batch_size_from_conv(node)

        # TODO(#16347): It is likely but not certain that the un-optimized
        # CMSIS-NN DW conv or the one without any SIMD is less efficient that
        # the corresponding CMSIS-NN conv. We should benchmark and update the
        # constraints.
        # optimal_dw_conv_constraints = (batch_size == 1) and (
        #    (in_channels == out_channels and dilation == [1, 1]) or (in_channels == 1)
        # )
        use_depthwise_conv = is_depthwise and (batch_size == 1)

        if use_depthwise_conv:
            # For depthwise: OIHW -> IHWO which gives [1, H, W, C_OUT] for CMSIS-NN
            # PyTorch depthwise weight is [out_ch, 1, H, W], permute to [1, H, W, out_ch]
            # The permute achieves the desired logical layout (IHWO). CMSIS-NN expects
            # weights in physically contiguous memory after the permute (not in channels-last)
            # so we use contiguous() here.
            weight_permuted = param_weight_tensor.permute(1, 2, 3, 0).contiguous()
        else:
            # For regular conv: OIHW -> OHWI
            # The permute achieves the desired logical layout (OHWI). CMSIS-NN expects
            # weights in physically contiguous memory after the permute (not in channels-last)
            # so we use contiguous() here.
            weight_permuted = param_weight_tensor.permute(0, 2, 3, 1).contiguous()

        with node.graph.inserting_after(weight):
            weight_nhwc = create_constant_placeholder(
                self.exported_program,
                node.graph,
                node.name + "_weight_nhwc",
                InputKind.PARAMETER,
                weight_permuted,
            )

            quantized_multiplier_tensor = create_constant_placeholder(
                self.exported_program,
                node.graph,
                node.name + "_quantized_multiplier",
                InputKind.PARAMETER,
                torch.tensor(quantized_multipliers, dtype=torch.int32),
            )

            quantized_shift_tensor = create_constant_placeholder(
                self.exported_program,
                node.graph,
                node.name + "_quantized_shift",
                InputKind.PARAMETER,
                torch.tensor(quantized_shifts, dtype=torch.int32),
            )

        if use_depthwise_conv:
            # Compute depth_multiplier for depthwise convolution
            # For depthwise: output_channels = input_channels * depth_multiplier

            if out_channels % in_channels != 0:
                raise ValueError(
                    f"Depthwise conv: output_channels ({out_channels}) must be "
                    f"divisible by input_channels ({in_channels})"
                )
            depth_multiplier = out_channels // in_channels

            new_args = (
                x,
                weight_nhwc,
                bias,
                stride,
                padding,
                dilation,
                depth_multiplier,
                -input_zero_point,
                output_zero_point,
                quantized_multiplier_tensor,
                quantized_shift_tensor,
                output_qmin,
                output_qmax,
            )
            return exir_ops.edge.cortex_m.quantized_depthwise_conv2d.default, new_args
        else:
            # Use regular convolution operator
            new_args = (
                x,
                weight_nhwc,
                bias,
                stride,
                padding,
                dilation,
                -input_zero_point,
                output_zero_point,
                quantized_multiplier_tensor,
                quantized_shift_tensor,
                output_qmin,
                output_qmax,
            )
            return exir_ops.edge.cortex_m.quantized_conv2d.default, new_args

    def _get_float_convolution_replacement(self, node):
        (
            x,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        ) = node.args

        if transposed:
            return None

        input_tensor = get_first_fake_tensor(x)
        output_tensor = get_first_fake_tensor(node)
        weight_tensor = get_param_tensor(self.exported_program, weight)
        if input_tensor is None or output_tensor is None or weight_tensor is None:
            return None

        dtype = input_tensor.dtype
        if dtype not in (torch.float32, torch.float16):
            return None
        if output_tensor.dtype != dtype or weight_tensor.dtype != dtype:
            return None
        if input_tensor.ndim != 4 or output_tensor.ndim != 4 or weight_tensor.ndim != 4:
            return None
        # NormalizeDimOrderPass already materializes NHWC for float conv inputs.
        if not is_channels_last(input_tensor):
            return None

        out_channels = weight_tensor.shape[0]
        in_channels = weight_tensor.shape[1] * groups
        is_depthwise = is_float_depthwise_conv(in_channels, out_channels, groups)
        activation_min, activation_max = self._get_float_activation_bounds(
            dtype, node.meta
        )

        with node.graph.inserting_after(weight):
            if is_depthwise:
                # The float backend also treats a single-input-channel regular
                # conv (groups == 1, C_IN == 1) as depthwise-equivalent:
                #
                #   PyTorch:  conv2d, C_IN=1, C_OUT=K, groups=1
                #   Cortex-M: depthwise_conv2d, depth_multiplier=K
                #
                # This is why traces can show depthwise_conv2d for a source
                # layer that was not explicitly grouped in the original model.
                weight_permuted = weight_tensor.permute(1, 2, 3, 0).contiguous()
                # This pass may run more than once.  FX node names can be
                # reused after earlier rewrites erase nodes, so naming the
                # generated constant only from node.name can accidentally reuse
                # another layer's transformed weight.  Include the source
                # weight placeholder name to keep constants layer-specific:
                #
                #   conv_1st weight -> aten_convolution_default_b_conv_1st_weight_weight_nhwc
                #   mid_conv weight -> aten_convolution_default_b_mid_conv_a_weight_weight_nhwc
                weight_nhwc = create_constant_placeholder(
                    self.exported_program,
                    node.graph,
                    f"{node.name}_{weight.name}_weight_nhwc",
                    InputKind.PARAMETER,
                    weight_permuted,
                )
                depth_multiplier = out_channels // in_channels
                new_args = (
                    x,
                    weight_nhwc,
                    bias,
                    stride,
                    padding,
                    dilation,
                    depth_multiplier,
                    activation_min,
                    activation_max,
                )
                op = (
                    exir_ops.edge.cortex_m.depthwise_conv2d_f32.default
                    if dtype == torch.float32
                    else exir_ops.edge.cortex_m.depthwise_conv2d_f16.default
                )
                self._require_float_dtype(
                    dtype,
                    (
                        "cortex_m::depthwise_conv2d_f32"
                        if dtype == torch.float32
                        else "cortex_m::depthwise_conv2d_f16"
                    ),
                )
            elif groups == 1:
                weight_permuted = weight_tensor.permute(0, 2, 3, 1).contiguous()
                # Keep conv weights in standard OHWI form here.
                #
                # FoldBatchNormIntoConvPass runs later and still expects
                # ordinary rank-4 conv weights so it can rewrite/fold BN into
                # the preceding constant weight/bias tensors. Packing 1x1 conv
                # weights at this stage turns them into a backend-specific
                # rank-1 blob too early and breaks that pass.
                #
                # Offline packed constants remain enabled for linear / constant
                # batch_matmul. For conv, we keep the existing runtime packing
                # path for now until a post-BN-fold conv-packing pass is added.
                # See the depthwise branch above: include the original weight
                # node name so a second ConvertToCortexMPass run cannot collide
                # with constants created during the first run.
                weight_nhwc = create_constant_placeholder(
                    self.exported_program,
                    node.graph,
                    f"{node.name}_{weight.name}_weight_nhwc",
                    InputKind.PARAMETER,
                    weight_permuted,
                )
                new_args = (
                    x,
                    weight_nhwc,
                    bias,
                    stride,
                    padding,
                    dilation,
                    False,
                    0,
                    0,
                    0,
                    0,
                    activation_min,
                    activation_max,
                )
                op = (
                    exir_ops.edge.cortex_m.conv2d_f32.default
                    if dtype == torch.float32
                    else exir_ops.edge.cortex_m.conv2d_f16.default
                )
                self._require_float_dtype(
                    dtype,
                    (
                        "cortex_m::conv2d_f32"
                        if dtype == torch.float32
                        else "cortex_m::conv2d_f16"
                    ),
                )
            else:
                return None

        return op, new_args

    def _get_float_transpose_conv2d_replacement(self, node):
        (
            x,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        ) = node.args

        if not transposed or groups != 1:
            return None

        input_tensor = get_first_fake_tensor(x)
        output_tensor = get_first_fake_tensor(node)
        weight_tensor = get_param_tensor(self.exported_program, weight)
        if input_tensor is None or output_tensor is None or weight_tensor is None:
            return None

        dtype = input_tensor.dtype
        if dtype not in (torch.float32, torch.float16):
            return None
        if output_tensor.dtype != dtype or weight_tensor.dtype != dtype:
            return None
        if input_tensor.ndim != 4 or output_tensor.ndim != 4 or weight_tensor.ndim != 4:
            return None
        if not is_channels_last(input_tensor):
            return None

        activation_min, activation_max = self._get_float_activation_bounds(
            dtype, node.meta
        )
        weight_permuted = weight_tensor.permute(1, 2, 3, 0).contiguous()

        with node.graph.inserting_after(weight):
            weight_nhwc = create_constant_placeholder(
                self.exported_program,
                node.graph,
                node.name + "_weight_nhwc",
                InputKind.PARAMETER,
                weight_permuted,
            )

        new_args = (
            x,
            weight_nhwc,
            bias,
            stride,
            padding,
            output_padding,
            dilation,
            activation_min,
            activation_max,
        )
        op = (
            exir_ops.edge.cortex_m.transpose_conv2d_f32.default
            if dtype == torch.float32
            else exir_ops.edge.cortex_m.transpose_conv2d_f16.default
        )
        self._require_float_dtype(
            dtype,
            (
                "cortex_m::transpose_conv2d_f32"
                if dtype == torch.float32
                else "cortex_m::transpose_conv2d_f16"
            ),
        )
        return op, new_args

    def _get_transpose_conv2d_replacement(self, node):
        """
        Transform aten.convolution with transposed=True to cortex_m.quantized_transpose_conv2d
        """
        (
            x,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        ) = node.args

        input_scale = node.meta["input_qparams"][0].scale
        input_zero_point = node.meta["input_qparams"][0].zp
        weight_scales = node.meta["input_qparams"][1].scale

        # For transposed conv: weight shape is (in_channels, out_channels/groups, H, W)
        # We need requantization params for each output channel
        weight_tensor = get_first_fake_tensor(weight)
        if not isinstance(weight_scales, list):
            # weight_tensor.shape[1] is out_channels for transposed conv
            num_output_channels = weight_tensor.shape[1]
            weight_scales = [weight_scales] * num_output_channels

        output_qparams = node.meta["output_qparams"][0]
        output_scale = output_qparams.scale
        output_zero_point = output_qparams.zp
        output_qmin = output_qparams.qmin
        output_qmax = output_qparams.qmax

        # Compute per-channel requantization parameters
        quantized_multipliers = []
        quantized_shifts = []
        for weight_scale in weight_scales:
            quantized_multiplier, quantized_shift = quantize_multiplier_aot(
                input_scale * weight_scale / output_scale
            )
            quantized_multipliers.append(quantized_multiplier)
            quantized_shifts.append(quantized_shift)

        # CRITICAL: Weight layout transformation for transposed conv
        # PyTorch ConvTranspose2d: (in_channels, out_channels/groups, H, W)
        # CMSIS-NN expects: (out_channels, H, W, in_channels) = OHWI
        # Permutation: (1, 2, 3, 0)
        weight_tensor_param = get_param_tensor(self.exported_program, weight)
        if weight_tensor_param is None:
            raise RuntimeError(
                f"Expected transpose conv weight parameter tensor for node {node.name}."
            )
        weight_permuted = weight_tensor_param.permute(1, 2, 3, 0).contiguous()

        with node.graph.inserting_after(weight):
            weight_nhwc = create_constant_placeholder(
                self.exported_program,
                node.graph,
                node.name + "_weight_nhwc",
                InputKind.PARAMETER,
                weight_permuted,
            )

            quantized_multiplier_tensor = create_constant_placeholder(
                self.exported_program,
                node.graph,
                node.name + "_quantized_multiplier",
                InputKind.PARAMETER,
                torch.tensor(quantized_multipliers, dtype=torch.int32),
            )

            quantized_shift_tensor = create_constant_placeholder(
                self.exported_program,
                node.graph,
                node.name + "_quantized_shift",
                InputKind.PARAMETER,
                torch.tensor(quantized_shifts, dtype=torch.int32),
            )

        new_args = (
            x,
            weight_nhwc,
            bias,
            stride,
            padding,
            output_padding,  # output_padding is NEW for transposed conv
            dilation,
            -input_zero_point,
            output_zero_point,
            quantized_multiplier_tensor,
            quantized_shift_tensor,
            output_qmin,
            output_qmax,
        )
        return exir_ops.edge.cortex_m.quantized_transpose_conv2d.default, new_args

    def _get_bmm_replacement(self, node):
        lhs_scale = node.meta["input_qparams"][0].scale
        lhs_zp = node.meta["input_qparams"][0].zp
        rhs_scale = node.meta["input_qparams"][1].scale
        rhs_zp = node.meta["input_qparams"][1].zp
        output_scale = node.meta["output_qparams"][0].scale
        output_zp = node.meta["output_qparams"][0].zp

        output_mult, output_shift = quantize_multiplier_aot(
            (lhs_scale * rhs_scale) / output_scale
        )

        lhs_node = node.args[0]
        rhs_node = node.args[1]

        is_constant_rhs = is_param_node(self.exported_program, rhs_node)
        if is_constant_rhs:
            rhs_tensor = get_param_tensor(self.exported_program, rhs_node)
            rhs_transposed_tensor = rhs_tensor.permute(0, 2, 1).contiguous()
            with node.graph.inserting_after(rhs_node):
                rhs_transposed = create_constant_placeholder(
                    self.exported_program,
                    node.graph,
                    node.name + "_rhs_transposed",
                    InputKind.PARAMETER,
                    rhs_transposed_tensor,
                )
        else:
            with node.graph.inserting_before(node):
                rhs_transposed = node.graph.create_node(
                    "call_function",
                    target=exir_ops.edge.cortex_m.transpose.default,
                    args=(rhs_node, [0, 2, 1]),
                )

        args = (
            lhs_node,
            -lhs_zp,
            rhs_transposed,
            -rhs_zp,
            output_zp,
            output_mult,
            output_shift,
        )
        return exir_ops.edge.cortex_m.quantized_batch_matmul.default, args

    def _get_float_bmm_replacement(self, node):
        lhs_node = node.args[0]
        rhs_node = node.args[1]

        lhs_fake = get_first_fake_tensor(lhs_node)
        rhs_fake = get_first_fake_tensor(rhs_node)
        if lhs_fake.dtype not in (torch.float32, torch.float16):
            return None
        if lhs_fake.dtype != rhs_fake.dtype:
            return None
        if lhs_fake.dim() != 3 or rhs_fake.dim() != 3:
            return None

        rhs_cols = int(rhs_fake.shape[2])
        is_constant_rhs = is_param_node(self.exported_program, rhs_node)
        if is_constant_rhs:
            # Constant float BMM RHS can also be prepared offline.
            #
            # ExecuTorch `bmm(lhs, rhs)` sees:
            #   lhs = [B, M, K]
            #   rhs = [B, K, N]
            #
            # The Cortex-M kernel contract wants the RHS presented as a
            # transposed logical matrix per batch:
            #   rhs_transposed = [B, N, K]
            #
            # We therefore:
            #   1. transpose each batch RHS to [N, K]
            #   2. pack each [N, K] slice into NT_N format
            #   3. concatenate the packed slices into one flat constant buffer
            #
            # Before:
            #   aten.bmm(lhs[B,M,K], rhs[B,K,N])
            #
            # After:
            #   cortex_m::batch_matmul_f16/f32(
            #       lhs[B,M,K],
            #       packed_rhs[flat],
            #       rhs_is_packed=True,
            #       rhs_cols=K,
            #       activation_min,
            #       activation_max)
            #
            # The runtime wrapper then skips per-inference RHS packing and calls
            # the packed CMSIS batch-matmul path directly.
            rhs_tensor = get_param_tensor(self.exported_program, rhs_node)
            rhs_transposed_tensor = rhs_tensor.permute(0, 2, 1).contiguous()
            block_cols = 8 if lhs_fake.dtype == torch.float16 else 4
            can_pack_rhs = int(rhs_transposed_tensor.shape[0]) == 1 or (
                rhs_cols % block_cols == 0
            )
            with node.graph.inserting_after(rhs_node):
                if can_pack_rhs:
                    packed_rhs = torch.cat(
                        [
                            pack_nt_t_weights_to_nt_n_packed(
                                rhs_transposed_tensor[b],
                                rhs_cols,
                                int(rhs_transposed_tensor.shape[2]),
                            )
                            for b in range(int(rhs_transposed_tensor.shape[0]))
                        ]
                    ).contiguous()
                    rhs_name = node.name + "_rhs_transposed_packed"
                    rhs_value = packed_rhs
                else:
                    # CMSIS-NN advances multi-batch RHS pointers by the unpacked
                    # [N, K] stride. NT_N packing pads N up to a dtype-specific
                    # block size, so for multi-batch RHS with padded N we must
                    # keep the ordinary transposed layout for correctness.
                    rhs_name = node.name + "_rhs_transposed"
                    rhs_value = rhs_transposed_tensor
                rhs_transposed = create_constant_placeholder(
                    self.exported_program,
                    node.graph,
                    rhs_name,
                    InputKind.PARAMETER,
                    rhs_value,
                )
            rhs_is_packed = can_pack_rhs
        else:
            with node.graph.inserting_before(node):
                rhs_transposed = node.graph.create_node(
                    "call_function",
                    target=exir_ops.edge.cortex_m.transpose.default,
                    args=(rhs_node, [0, 2, 1]),
                )
            rhs_is_packed = False

        op = (
            exir_ops.edge.cortex_m.batch_matmul_f32.default
            if lhs_fake.dtype == torch.float32
            else exir_ops.edge.cortex_m.batch_matmul_f16.default
        )
        self._require_float_dtype(
            lhs_fake.dtype,
            (
                "cortex_m::batch_matmul_f32"
                if lhs_fake.dtype == torch.float32
                else "cortex_m::batch_matmul_f16"
            ),
        )
        activation_min, activation_max = self._get_float_activation_bounds(
            lhs_fake.dtype, node.meta
        )
        args = (
            lhs_node,
            rhs_transposed,
            rhs_is_packed,
            rhs_cols,
            activation_min,
            activation_max,
        )
        return op, args

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue

            has_qparams = bool(
                node.meta.get("input_qparams") and node.meta.get("output_qparams")
            )
            # Current upstream quantized graph surgery is centralized in
            # AtenToCortexMPass. Keep the legacy helpers below unreachable until
            # they can be removed separately without obscuring this rebase.
            if has_qparams:
                continue

            match node.target:
                case exir_ops.edge.aten.linear.default:
                    # Float linear is first lowered by FloatOpRewritePass;
                    # weight packing happens below on the cortex_m target.
                    continue
                case exir_ops.edge.aten.convolution.default:
                    transposed = node.args[6] if len(node.args) > 6 else False
                    replacement = (
                        self._get_float_transpose_conv2d_replacement(node)
                        if transposed
                        else self._get_float_convolution_replacement(node)
                    )
                    if replacement is None:
                        continue
                    op, args = replacement
                case exir_ops.edge.aten.bmm.default:
                    replacement = self._get_float_bmm_replacement(node)
                    if replacement is None:
                        continue
                    op, args = replacement
                case (
                    exir_ops.edge.cortex_m.linear_f32.default
                    | exir_ops.edge.cortex_m.linear_f16.default
                ):
                    replacement = self._get_float_linear_replacement(node)
                    if replacement is None:
                        continue
                    op, args = replacement
                case _:
                    continue

            with graph_module.graph.inserting_before(node):
                cortex_m_op = graph_module.graph.create_node(
                    "call_function",
                    target=op,
                    args=args,
                    kwargs={},
                )
                cortex_m_op.meta = dict(node.meta)

                node.replace_all_uses_with(cortex_m_op)
                graph_module.graph.erase_node(node)

            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
