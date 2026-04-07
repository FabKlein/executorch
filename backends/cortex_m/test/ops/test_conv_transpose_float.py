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


class CortexMConvTranspose2DFloat(torch.nn.Module):
    def __init__(self, dtype: torch.dtype, *args, bias: bool = False, **kwargs):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(*args, bias=bias, **kwargs)
        with torch.no_grad():
            self.conv_transpose.weight.copy_(
                torch.linspace(
                    -1.0,
                    1.0,
                    steps=self.conv_transpose.weight.numel(),
                    dtype=torch.float32,
                ).reshape_as(self.conv_transpose.weight)
            )
            if bias:
                self.conv_transpose.bias.copy_(
                    torch.linspace(
                        -0.25,
                        0.25,
                        steps=self.conv_transpose.bias.numel(),
                        dtype=torch.float32,
                    )
                )
        self.conv_transpose = self.conv_transpose.to(dtype)
        suffix = "f32" if dtype == torch.float32 else "f16"
        self.ops_before_transforms = {
            "executorch_exir_dialects_edge__ops_aten_convolution_default": 1
        }
        self.ops_after_transforms = {
            f"executorch_exir_dialects_edge__ops_cortex_m_transpose_conv2d_{suffix}_default": 1
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose(x)


def _channels_last_case(
    model: torch.nn.Module, start: float, end: float, shape: tuple[int, ...]
) -> McuTestCase:
    dtype = next(model.parameters()).dtype
    return McuTestCase(
        model,
        (
            ramp_tensor(start, end, shape)
            .to(dtype)
            .contiguous(memory_format=torch.channels_last),
        ),
    )


test_cases = {
    "f32_conv_transpose2d_basic": _channels_last_case(
        CortexMConvTranspose2DFloat(torch.float32, 2, 4, 3), 1, 5, (1, 2, 5, 5)
    ),
    "f32_conv_transpose2d_stride_2": _channels_last_case(
        CortexMConvTranspose2DFloat(torch.float32, 3, 6, kernel_size=3, stride=2),
        0,
        10,
        (1, 3, 4, 4),
    ),
    "f32_conv_transpose2d_stride_asym": _channels_last_case(
        CortexMConvTranspose2DFloat(torch.float32, 2, 4, kernel_size=3, stride=(2, 3)),
        -5,
        5,
        (1, 2, 4, 4),
    ),
    "f32_conv_transpose2d_padding_1": _channels_last_case(
        CortexMConvTranspose2DFloat(torch.float32, 2, 4, kernel_size=3, padding=1),
        0,
        20,
        (1, 2, 5, 5),
    ),
    "f32_conv_transpose2d_padding_asym": _channels_last_case(
        CortexMConvTranspose2DFloat(torch.float32, 2, 4, kernel_size=3, padding=(2, 1)),
        -10,
        10,
        (1, 2, 6, 6),
    ),
    "f32_conv_transpose2d_output_padding_1": _channels_last_case(
        CortexMConvTranspose2DFloat(
            torch.float32, 2, 4, kernel_size=3, stride=2, output_padding=1
        ),
        0,
        15,
        (1, 2, 5, 5),
    ),
    "f32_conv_transpose2d_output_padding_asym": _channels_last_case(
        CortexMConvTranspose2DFloat(
            torch.float32, 3, 6, kernel_size=4, stride=2, output_padding=(1, 0)
        ),
        5,
        25,
        (1, 3, 4, 4),
    ),
    "f32_conv_transpose2d_bias": _channels_last_case(
        CortexMConvTranspose2DFloat(torch.float32, 4, 8, kernel_size=3, bias=True),
        -20,
        20,
        (1, 4, 6, 6),
    ),
    "f32_conv_transpose2d_bias_single_out": _channels_last_case(
        CortexMConvTranspose2DFloat(
            torch.float32, 5, 1, kernel_size=3, stride=2, bias=True
        ),
        0,
        50,
        (1, 5, 4, 4),
    ),
    "f32_conv_transpose2d_dilation_2": _channels_last_case(
        CortexMConvTranspose2DFloat(torch.float32, 2, 4, kernel_size=3, dilation=2),
        0,
        30,
        (1, 2, 8, 8),
    ),
    "f32_conv_transpose2d_groups_2": _channels_last_case(
        CortexMConvTranspose2DFloat(torch.float32, 4, 8, kernel_size=3, groups=2),
        -15,
        15,
        (1, 4, 5, 5),
    ),
    "f32_conv_transpose2d_depthwise": _channels_last_case(
        CortexMConvTranspose2DFloat(torch.float32, 4, 4, kernel_size=3, groups=4),
        0,
        40,
        (1, 4, 6, 6),
    ),
    "f32_conv_transpose2d_kernel_1x1": _channels_last_case(
        CortexMConvTranspose2DFloat(torch.float32, 3, 6, kernel_size=1, stride=2),
        0,
        12,
        (1, 3, 4, 4),
    ),
    "f32_conv_transpose2d_kernel_asym": _channels_last_case(
        CortexMConvTranspose2DFloat(torch.float32, 2, 4, kernel_size=(2, 4)),
        -8,
        8,
        (1, 2, 5, 5),
    ),
    "f32_conv_transpose2d_kernel_5x5": _channels_last_case(
        CortexMConvTranspose2DFloat(torch.float32, 2, 4, kernel_size=5, stride=2),
        0,
        25,
        (1, 2, 6, 6),
    ),
    "f32_conv_transpose2d_single_channel_in": _channels_last_case(
        CortexMConvTranspose2DFloat(torch.float32, 1, 8, kernel_size=3, stride=2),
        0,
        16,
        (1, 1, 4, 4),
    ),
    "f32_conv_transpose2d_single_channel_out": _channels_last_case(
        CortexMConvTranspose2DFloat(torch.float32, 8, 1, kernel_size=3, stride=2),
        -40,
        40,
        (1, 8, 5, 5),
    ),
    "f32_conv_transpose2d_large_channels": _channels_last_case(
        CortexMConvTranspose2DFloat(torch.float32, 16, 32, kernel_size=3),
        -50,
        50,
        (1, 16, 4, 4),
    ),
    "f32_conv_transpose2d_large_spatial": _channels_last_case(
        CortexMConvTranspose2DFloat(torch.float32, 3, 6, kernel_size=3, stride=2),
        -100,
        100,
        (1, 3, 16, 16),
    ),
    "f32_conv_transpose2d_batch_2": _channels_last_case(
        CortexMConvTranspose2DFloat(torch.float32, 2, 4, kernel_size=3),
        0,
        80,
        (2, 2, 5, 5),
    ),
    "f32_conv_transpose2d_small_input": _channels_last_case(
        CortexMConvTranspose2DFloat(torch.float32, 4, 8, kernel_size=3, stride=2),
        0,
        8,
        (1, 4, 2, 2),
    ),
    "f32_conv_transpose2d_complex": _channels_last_case(
        CortexMConvTranspose2DFloat(
            torch.float32,
            4,
            8,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=1,
            bias=True,
        ),
        -30,
        30,
        (1, 4, 6, 6),
    ),
    "f32_conv_transpose2d_all_params": _channels_last_case(
        CortexMConvTranspose2DFloat(
            torch.float32,
            3,
            6,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            dilation=2,
            bias=True,
        ),
        0,
        60,
        (1, 3, 8, 8),
    ),
}

for name, case in list(test_cases.items()):
    if name.startswith("f32_"):
        f16_name = name.replace("f32_", "f16_", 1)
        model = case.model
        # Reconstruct equivalent float16 module from conv_transpose settings.
        conv = model.conv_transpose
        f16_model = CortexMConvTranspose2DFloat(
            torch.float16,
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            output_padding=conv.output_padding,
            groups=conv.groups,
            dilation=conv.dilation,
            bias=conv.bias is not None,
        )
        ex = case.example_inputs[0]
        test_cases[f16_name] = McuTestCase(
            f16_model,
            (ex.to(torch.float16).contiguous(memory_format=torch.channels_last),),
        )


xfails_dialect = {
    "f32_conv_transpose2d_groups_2": "Grouped transpose conv not supported in float ET path",
    "f32_conv_transpose2d_depthwise": "Depthwise transpose conv not supported in float ET path",
    "f16_conv_transpose2d_groups_2": "Grouped transpose conv not supported in float ET path",
    "f16_conv_transpose2d_depthwise": "Depthwise transpose conv not supported in float ET path",
}

xfails_implementation = {
    "f32_conv_transpose2d_groups_2": "Grouped transpose conv not supported in float ET path",
    "f32_conv_transpose2d_depthwise": "Depthwise transpose conv not supported in float ET path",
    "f32_conv_transpose2d_output_padding_1": "output_padding path still fails in float implementation",
    "f32_conv_transpose2d_output_padding_asym": "output_padding path still fails in float implementation",
    "f32_conv_transpose2d_complex": "Combined transpose-conv parameters still fail in float implementation",
    "f32_conv_transpose2d_all_params": "Combined transpose-conv parameters still fail in float implementation",
    "f16_conv_transpose2d_groups_2": "Grouped transpose conv not supported in float ET path",
    "f16_conv_transpose2d_depthwise": "Depthwise transpose conv not supported in float ET path",
    "f16_conv_transpose2d_output_padding_1": "output_padding not supported in float ET path",
    "f16_conv_transpose2d_output_padding_asym": "output_padding not supported in float ET path",
    "f16_conv_transpose2d_complex": "Uses output_padding which is not supported in float ET path",
    "f16_conv_transpose2d_all_params": "Combines unsupported transpose-conv parameters in float ET path",
}


def _float_compare_kwargs(test_case):
    dtype = next(test_case.model.parameters()).dtype
    if dtype == torch.float16:
        # Transpose-conv accumulates a bit more half-precision drift on FVP than
        # the tighter default allclose tolerances allow.
        return {"atol": 4e-2, "rtol": 2e-2}
    return {}


@parametrize("test_case", test_cases, xfails=xfails_dialect)
def test_dialect_conv_transpose2d_float(test_case):
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


@XfailIfNoCorstone300
@parametrize("test_case", test_cases, xfails=xfails_implementation)
def test_implementation_conv_transpose2d_float(test_case):
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
