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


class CortexMConvActivationFloat(torch.nn.Module):
    def __init__(
        self,
        dtype: torch.dtype,
        kind: str,
        *,
        depthwise: bool = False,
        transpose: bool = False,
    ) -> None:
        super().__init__()
        self.kind = kind
        suffix = "f32" if dtype == torch.float32 else "f16"

        if transpose:
            self.op = torch.nn.ConvTranspose2d(3, 4, kernel_size=3, stride=2, bias=True)
            op_name = f"executorch_exir_dialects_edge__ops_cortex_m_transpose_conv2d_{suffix}_default"
        elif depthwise:
            self.op = torch.nn.Conv2d(
                4, 4, kernel_size=3, padding=1, groups=4, bias=True
            )
            op_name = f"executorch_exir_dialects_edge__ops_cortex_m_depthwise_conv2d_{suffix}_default"
        else:
            self.op = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=True)
            op_name = (
                f"executorch_exir_dialects_edge__ops_cortex_m_conv2d_{suffix}_default"
            )

        with torch.no_grad():
            self.op.weight.copy_(
                torch.linspace(
                    -1.0,
                    1.0,
                    steps=self.op.weight.numel(),
                    dtype=torch.float32,
                ).reshape_as(self.op.weight)
            )
            self.op.bias.copy_(
                torch.linspace(
                    -0.25,
                    0.25,
                    steps=self.op.bias.numel(),
                    dtype=torch.float32,
                )
            )

        self.op = self.op.to(dtype)

        activation_before = (
            "executorch_exir_dialects_edge__ops_aten_relu_default"
            if kind == "relu"
            else "executorch_exir_dialects_edge__ops_aten_hardtanh_default"
        )
        self.ops_before_transforms = {
            "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
            activation_before: 1,
        }
        self.ops_after_transforms = {op_name: 1}
        self.ops_removed_after_transforms = [activation_before]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.op(x)
        if self.kind == "relu":
            return torch.relu(x)
        return torch.nn.functional.hardtanh(x, min_val=-0.5, max_val=0.5)


test_cases = {
    "f32_conv_relu": McuTestCase(
        CortexMConvActivationFloat(torch.float32, "relu"),
        (
            ramp_tensor(-1, 1, (1, 3, 6, 6))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_conv_relu": McuTestCase(
        CortexMConvActivationFloat(torch.float16, "relu"),
        (
            ramp_tensor(-1, 1, (1, 3, 6, 6))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_depthwise_hardtanh": McuTestCase(
        CortexMConvActivationFloat(torch.float32, "hardtanh", depthwise=True),
        (
            ramp_tensor(-1, 1, (1, 4, 6, 6))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_depthwise_hardtanh": McuTestCase(
        CortexMConvActivationFloat(torch.float16, "hardtanh", depthwise=True),
        (
            ramp_tensor(-1, 1, (1, 4, 6, 6))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_transpose_conv_relu": McuTestCase(
        CortexMConvActivationFloat(torch.float32, "relu", transpose=True),
        (
            ramp_tensor(-1, 1, (1, 3, 5, 5))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_transpose_conv_relu": McuTestCase(
        CortexMConvActivationFloat(torch.float16, "relu", transpose=True),
        (
            ramp_tensor(-1, 1, (1, 3, 5, 5))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_conv_activation_fusion_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.check_count(test_case.model.ops_before_transforms)
    tester.run_passes()
    tester.check_count(test_case.model.ops_after_transforms)
    tester.check_not(getattr(test_case.model, "ops_removed_after_transforms", []))
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs)


@XfailIfNoCorstone300
@parametrize("test_case", test_cases)
def test_implementation_conv_activation_fusion_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.run_passes()
    tester.to_executorch()
    tester.serialize()
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs)
