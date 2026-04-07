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


class CortexMAvgPool2dFloat(torch.nn.Module):
    def __init__(self, dtype, kernel_size, stride, padding=0):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(
            kernel_size,
            stride,
            padding,
            ceil_mode=False,
            count_include_pad=False,
        )
        suffix = "f32" if dtype == torch.float32 else "f16"
        self.ops_before_transforms = {
            "executorch_exir_dialects_edge__ops_aten_avg_pool2d_default": 1,
        }
        self.ops_after_transforms = {
            f"executorch_exir_dialects_edge__ops_cortex_m_avg_pool2d_{suffix}_default": 1,
        }

    def forward(self, x):
        return self.pool(x)


test_cases = {
    "f32_avgpool_2x2": McuTestCase(
        CortexMAvgPool2dFloat(torch.float32, kernel_size=2, stride=2),
        (
            ramp_tensor(0, 15, (1, 1, 4, 4))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_avgpool_3x3_s1": McuTestCase(
        CortexMAvgPool2dFloat(torch.float32, kernel_size=3, stride=1, padding=1),
        (
            ramp_tensor(-4, 4, (1, 2, 3, 3))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_avgpool_3x3_s2_pad1": McuTestCase(
        CortexMAvgPool2dFloat(torch.float32, kernel_size=3, stride=2, padding=1),
        (
            ramp_tensor(-3, 5, (1, 2, 4, 4))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_avgpool_2x2": McuTestCase(
        CortexMAvgPool2dFloat(torch.float16, kernel_size=2, stride=2),
        (
            ramp_tensor(0, 15, (1, 1, 4, 4))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_avgpool_3x3_s1": McuTestCase(
        CortexMAvgPool2dFloat(torch.float16, kernel_size=3, stride=1, padding=1),
        (
            ramp_tensor(-4, 4, (1, 2, 3, 3))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_avgpool_3x3_s2_pad1": McuTestCase(
        CortexMAvgPool2dFloat(torch.float16, kernel_size=3, stride=2, padding=1),
        (
            ramp_tensor(-3, 5, (1, 2, 4, 4))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_avg_pool2d_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.check_count(test_case.model.ops_before_transforms)
    tester.run_passes()
    tester.check_count(test_case.model.ops_after_transforms)
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs)


@XfailIfNoCorstone300
@parametrize("test_case", test_cases)
def test_implementation_avg_pool2d_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.run_passes()
    tester.to_executorch()
    tester.serialize()
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs)
