# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test.common import parametrize, XfailIfNoCorstone300
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)


class CortexMConv2DFloat(torch.nn.Module):
    def __init__(self, dtype: torch.dtype, *args, bias: bool = False, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(*args, bias=bias, **kwargs)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.linspace(
                    -1.0,
                    1.0,
                    steps=self.conv.weight.numel(),
                    dtype=torch.float32,
                ).reshape_as(self.conv.weight)
            )
            if bias:
                self.conv.bias.copy_(
                    torch.linspace(
                        -0.25,
                        0.25,
                        steps=self.conv.bias.numel(),
                        dtype=torch.float32,
                    )
                )
        self.conv = self.conv.to(dtype)
        suffix = "f32" if dtype == torch.float32 else "f16"
        # Float lowering intentionally maps C_IN=1, groups=1 convolutions to
        # depthwise_conv2d: the operation is equivalent to depthwise with
        # depth_multiplier == out_channels and can use the CMSIS-NN DW path.
        op_kind = (
            "depthwise_conv2d"
            if self.conv.groups == self.conv.in_channels
            or (self.conv.groups == 1 and self.conv.in_channels == 1)
            else "conv2d"
        )
        self.ops_before_transforms = {
            "executorch_exir_dialects_edge__ops_aten_convolution_default": 1
        }
        self.ops_after_transforms = {
            f"executorch_exir_dialects_edge__ops_cortex_m_{op_kind}_{suffix}_default": 1
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CortexMConv2DFloatX3(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(3, 3, 3, padding=1, bias=False)
        for idx, conv in enumerate((self.conv1, self.conv2, self.conv3)):
            with torch.no_grad():
                conv.weight.copy_(
                    torch.linspace(
                        -1.0 + idx * 0.2,
                        1.0 + idx * 0.2,
                        steps=conv.weight.numel(),
                        dtype=torch.float32,
                    ).reshape_as(conv.weight)
                )
        self.conv1 = self.conv1.to(dtype)
        self.conv2 = self.conv2.to(dtype)
        self.conv3 = self.conv3.to(dtype)
        suffix = "f32" if dtype == torch.float32 else "f16"
        self.ops_before_transforms = {
            "executorch_exir_dialects_edge__ops_aten_convolution_default": 3
        }
        self.ops_after_transforms = {
            f"executorch_exir_dialects_edge__ops_cortex_m_conv2d_{suffix}_default": 3
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class CortexMDepthwiseConv2DFloat(torch.nn.Module):
    def __init__(self, dtype: torch.dtype, *args, bias: bool = False, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(*args, bias=bias, **kwargs)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.linspace(
                    -1.0,
                    1.0,
                    steps=self.conv.weight.numel(),
                    dtype=torch.float32,
                ).reshape_as(self.conv.weight)
            )
            if bias:
                self.conv.bias.copy_(
                    torch.linspace(
                        -0.25,
                        0.25,
                        steps=self.conv.bias.numel(),
                        dtype=torch.float32,
                    )
                )
        self.conv = self.conv.to(dtype)
        suffix = "f32" if dtype == torch.float32 else "f16"
        self.ops_before_transforms = {
            "executorch_exir_dialects_edge__ops_aten_convolution_default": 1
        }
        self.ops_after_transforms = {
            f"executorch_exir_dialects_edge__ops_cortex_m_depthwise_conv2d_{suffix}_default": 1
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


test_cases = {
    "f32_conv2d": McuTestCase(
        CortexMConv2DFloat(torch.float32, 2, 4, 3),
        (
            ramp_tensor(1, 5, (1, 2, 5, 5))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_conv2d_stride": McuTestCase(
        CortexMConv2DFloat(torch.float32, 3, 4, (1, 2), stride=2),
        (
            ramp_tensor(-3, 3, (1, 3, 8, 8))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_conv2d_padding": McuTestCase(
        CortexMConv2DFloat(torch.float32, 3, 2, 3, padding=(1, 2)),
        (
            ramp_tensor(-2, 2, (1, 3, 5, 6))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_conv2d_dilation": McuTestCase(
        CortexMConv2DFloat(torch.float32, 1, 4, 3, dilation=(2, 2)),
        (
            ramp_tensor(-1, 1, (1, 1, 8, 8))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_conv2d_bias": McuTestCase(
        CortexMConv2DFloat(torch.float32, 5, 4, (1, 2), bias=True),
        (
            ramp_tensor(-3, 3, (1, 5, 10, 10))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_conv2d_x3": McuTestCase(
        CortexMConv2DFloatX3(torch.float32),
        (
            ramp_tensor(-1, 1, (1, 3, 8, 8))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_conv2d_1x1_stride_h_w_1_2": McuTestCase(
        CortexMConv2DFloat(torch.float32, 3, 5, 1, stride=(1, 2), bias=True),
        (
            ramp_tensor(-2, 2, (1, 3, 17, 11))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_conv2d_1x3_nhwc_tuned": McuTestCase(
        CortexMConv2DFloat(torch.float32, 16, 16, (1, 3)),
        (
            ramp_tensor(-1, 1, (1, 16, 1, 21))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_conv2d_1x5_nhwc_tuned": McuTestCase(
        CortexMConv2DFloat(torch.float32, 16, 16, (1, 5)),
        (
            ramp_tensor(-1, 1, (1, 16, 1, 21))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_conv2d_2x2_common": McuTestCase(
        CortexMConv2DFloat(torch.float32, 4, 5, (2, 2)),
        (
            ramp_tensor(-2, 2, (1, 4, 6, 7))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_conv2d_patch_gemm_packed": McuTestCase(
        CortexMConv2DFloat(torch.float32, 4, 8, (2, 2)),
        (
            ramp_tensor(-2, 2, (1, 4, 6, 7))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_conv2d_3x3_pad1_common": McuTestCase(
        CortexMConv2DFloat(torch.float32, 2, 4, 3, padding=1, bias=True),
        (
            ramp_tensor(-2, 2, (1, 2, 3, 6))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_depthwise_conv2d": McuTestCase(
        CortexMDepthwiseConv2DFloat(torch.float32, 4, 4, 3, groups=4),
        (
            ramp_tensor(-1, 1, (1, 4, 8, 8))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_depthwise_conv2d_multiplier": McuTestCase(
        CortexMDepthwiseConv2DFloat(torch.float32, 3, 6, 3, groups=3),
        (
            ramp_tensor(-1, 1, (1, 3, 8, 8))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_depthwise_conv2d_single_channel_first_layer": McuTestCase(
        CortexMDepthwiseConv2DFloat(torch.float32, 1, 8, 3, groups=1),
        (
            ramp_tensor(-1, 1, (1, 1, 8, 8))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_depthwise_conv2d_stride": McuTestCase(
        CortexMDepthwiseConv2DFloat(torch.float32, 4, 4, 3, stride=2, groups=4),
        (
            ramp_tensor(-2, 2, (1, 4, 8, 8))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_depthwise_conv2d_padding": McuTestCase(
        CortexMDepthwiseConv2DFloat(torch.float32, 2, 2, 5, padding=2, groups=2),
        (
            ramp_tensor(-2, 2, (1, 2, 5, 5))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_depthwise_conv2d_bias": McuTestCase(
        CortexMDepthwiseConv2DFloat(
            torch.float32, 3, 3, 3, padding=1, groups=3, bias=True
        ),
        (
            ramp_tensor(-3, 3, (1, 3, 6, 6))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_depthwise_conv2d_1x1": McuTestCase(
        CortexMDepthwiseConv2DFloat(torch.float32, 4, 8, 1, groups=4),
        (
            ramp_tensor(-1, 1, (1, 4, 8, 8))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_depthwise_conv2d_smallc_nhwc": McuTestCase(
        CortexMDepthwiseConv2DFloat(torch.float32, 6, 6, 3, groups=6),
        (
            ramp_tensor(-1, 1, (1, 6, 9, 11))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_depthwise_conv2d_1x3_optimized": McuTestCase(
        CortexMDepthwiseConv2DFloat(torch.float32, 8, 8, (1, 3), groups=8),
        (
            ramp_tensor(-1, 1, (1, 8, 1, 21))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_depthwise_conv2d_2x5_optimized": McuTestCase(
        CortexMDepthwiseConv2DFloat(torch.float32, 8, 8, (2, 5), groups=8),
        (
            ramp_tensor(-1, 1, (2, 8, 2, 21))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_depthwise_conv2d_3x3_stride2_pad_h1_w0": McuTestCase(
        CortexMDepthwiseConv2DFloat(
            torch.float32, 4, 4, 3, stride=(2, 2), padding=(1, 0), groups=4
        ),
        (
            ramp_tensor(-1, 1, (1, 4, 5, 5))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_depthwise_conv2d_null_bias": McuTestCase(
        CortexMDepthwiseConv2DFloat(
            torch.float32, 4, 4, 3, stride=(2, 2), padding=(1, 0), groups=4, bias=False
        ),
        (
            ramp_tensor(-1, 1, (1, 4, 5, 5))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_conv2d": McuTestCase(
        CortexMConv2DFloat(torch.float16, 2, 4, 3),
        (
            ramp_tensor(1, 5, (1, 2, 5, 5))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_conv2d_stride": McuTestCase(
        CortexMConv2DFloat(torch.float16, 3, 4, (1, 2), stride=2),
        (
            ramp_tensor(-3, 3, (1, 3, 8, 8))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_conv2d_padding": McuTestCase(
        CortexMConv2DFloat(torch.float16, 3, 2, 3, padding=(1, 2)),
        (
            ramp_tensor(-2, 2, (1, 3, 5, 6))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_conv2d_dilation": McuTestCase(
        CortexMConv2DFloat(torch.float16, 1, 4, 3, dilation=(2, 2)),
        (
            ramp_tensor(-1, 1, (1, 1, 8, 8))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_conv2d_bias": McuTestCase(
        CortexMConv2DFloat(torch.float16, 5, 4, (1, 2), bias=True),
        (
            ramp_tensor(-3, 3, (1, 5, 10, 10))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_conv2d_x3": McuTestCase(
        CortexMConv2DFloatX3(torch.float16),
        (
            ramp_tensor(-1, 1, (1, 3, 8, 8))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_conv2d_1x1_stride_h_w_1_2": McuTestCase(
        CortexMConv2DFloat(torch.float16, 3, 5, 1, stride=(1, 2), bias=True),
        (
            ramp_tensor(-2, 2, (1, 3, 17, 11))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_conv2d_1x3_nhwc_tuned": McuTestCase(
        CortexMConv2DFloat(torch.float16, 16, 16, (1, 3)),
        (
            ramp_tensor(-1, 1, (1, 16, 1, 21))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_conv2d_1x5_nhwc_tuned": McuTestCase(
        CortexMConv2DFloat(torch.float16, 16, 16, (1, 5)),
        (
            ramp_tensor(-1, 1, (1, 16, 1, 21))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_conv2d_2x2_common": McuTestCase(
        CortexMConv2DFloat(torch.float16, 4, 5, (2, 2)),
        (
            ramp_tensor(-2, 2, (1, 4, 6, 7))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_conv2d_patch_gemm_packed": McuTestCase(
        CortexMConv2DFloat(torch.float16, 4, 8, (2, 2)),
        (
            ramp_tensor(-2, 2, (1, 4, 6, 7))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_conv2d_3x3_pad1_common": McuTestCase(
        CortexMConv2DFloat(torch.float16, 2, 4, 3, padding=1, bias=True),
        (
            ramp_tensor(-2, 2, (1, 2, 3, 6))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_depthwise_conv2d": McuTestCase(
        CortexMDepthwiseConv2DFloat(torch.float16, 4, 4, 3, groups=4),
        (
            ramp_tensor(-1, 1, (1, 4, 8, 8))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_depthwise_conv2d_multiplier": McuTestCase(
        CortexMDepthwiseConv2DFloat(torch.float16, 3, 6, 3, groups=3),
        (
            ramp_tensor(-1, 1, (1, 3, 8, 8))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_depthwise_conv2d_single_channel_first_layer": McuTestCase(
        CortexMDepthwiseConv2DFloat(torch.float16, 1, 8, 3, groups=1),
        (
            ramp_tensor(-1, 1, (1, 1, 8, 8))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_depthwise_conv2d_stride": McuTestCase(
        CortexMDepthwiseConv2DFloat(torch.float16, 4, 4, 3, stride=2, groups=4),
        (
            ramp_tensor(-2, 2, (1, 4, 8, 8))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_depthwise_conv2d_padding": McuTestCase(
        CortexMDepthwiseConv2DFloat(torch.float16, 2, 2, 5, padding=2, groups=2),
        (
            ramp_tensor(-2, 2, (1, 2, 5, 5))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_depthwise_conv2d_bias": McuTestCase(
        CortexMDepthwiseConv2DFloat(
            torch.float16, 3, 3, 3, padding=1, groups=3, bias=True
        ),
        (
            ramp_tensor(-3, 3, (1, 3, 6, 6))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_depthwise_conv2d_1x1": McuTestCase(
        CortexMDepthwiseConv2DFloat(torch.float16, 4, 8, 1, groups=4),
        (
            ramp_tensor(-1, 1, (1, 4, 8, 8))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_depthwise_conv2d_smallc_nhwc": McuTestCase(
        CortexMDepthwiseConv2DFloat(torch.float16, 6, 6, 3, groups=6),
        (
            ramp_tensor(-1, 1, (1, 6, 9, 11))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_depthwise_conv2d_1x3_optimized": McuTestCase(
        CortexMDepthwiseConv2DFloat(torch.float16, 8, 8, (1, 3), groups=8),
        (
            ramp_tensor(-1, 1, (1, 8, 1, 21))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_depthwise_conv2d_2x5_optimized": McuTestCase(
        CortexMDepthwiseConv2DFloat(torch.float16, 8, 8, (2, 5), groups=8),
        (
            ramp_tensor(-1, 1, (2, 8, 2, 21))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_depthwise_conv2d_3x3_stride2_pad_h1_w0": McuTestCase(
        CortexMDepthwiseConv2DFloat(
            torch.float16, 4, 4, 3, stride=(2, 2), padding=(1, 0), groups=4
        ),
        (
            ramp_tensor(-1, 1, (1, 4, 5, 5))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_depthwise_conv2d_null_bias": McuTestCase(
        CortexMDepthwiseConv2DFloat(
            torch.float16, 4, 4, 3, stride=(2, 2), padding=(1, 0), groups=4, bias=False
        ),
        (
            ramp_tensor(-1, 1, (1, 4, 5, 5))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
}


packed_regular_conv_test_cases = {
    key: test_cases[key]
    for key in (
        "f32_conv2d_patch_gemm_packed",
        "f16_conv2d_patch_gemm_packed",
    )
}


def _packed_cortex_m_conv_nodes(exported_program):
    return [
        node
        for node in exported_program.graph_module.graph.nodes
        if node.op == "call_function"
        and "cortex_m.conv2d" in str(node.target)
        and len(node.args) >= 11
        and node.args[6] is True
    ]


def _float_compare_kwargs(test_case):
    dtype = next(test_case.model.parameters()).dtype
    if dtype != torch.float16:
        return {}

    if isinstance(test_case.model, CortexMConv2DFloatX3):
        return {"atol": 6e-1, "rtol": 2e-2}

    if isinstance(
        test_case.model, CortexMConv2DFloat
    ) and test_case.model.conv.kernel_size == (1, 5):
        return {"atol": 6e-2, "rtol": 2e-2}

    return {"atol": 2e-2, "rtol": 2e-2}


@parametrize("test_case", test_cases)
def test_dialect_conv_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.check_count(test_case.model.ops_before_transforms)
    tester.run_passes()
    tester.check_count(test_case.model.ops_after_transforms)
    tester.run_method_and_compare_outputs(
        inputs=test_case.example_inputs,
        **_float_compare_kwargs(test_case),
    )


@parametrize("test_case", packed_regular_conv_test_cases)
def test_dialect_conv_float_packs_regular_conv_weights(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.run_passes()

    packed_nodes = _packed_cortex_m_conv_nodes(tester.get_artifact().exported_program())
    assert len(packed_nodes) == 1

    packed_node = packed_nodes[0]
    assert int(packed_node.args[7]) == 8  # out_channels
    assert int(packed_node.args[8]) == 2  # kernel_height
    assert int(packed_node.args[9]) == 2  # kernel_width
    assert int(packed_node.args[10]) == 4  # in_channels


@XfailIfNoCorstone300
@parametrize("test_case", test_cases)
def test_implementation_conv_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.run_passes()
    tester.to_executorch()
    tester.serialize()
    tester.run_method_and_compare_outputs(
        inputs=test_case.example_inputs,
        **_float_compare_kwargs(test_case),
    )
