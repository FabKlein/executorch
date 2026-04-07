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


class CortexMFloatMul(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_mul_f32_default": 1,
    }

    def forward(self, x, y):
        return x * y


class CortexMHalfMul(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_mul_f16_default": 1,
    }

    def forward(self, x, y):
        return x * y


class CortexMFloatMulReLU(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
        "executorch_exir_dialects_edge__ops_aten_relu_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_mul_f32_default": 1,
    }

    ops_removed_after_transforms = [
        "executorch_exir_dialects_edge__ops_aten_relu_default",
    ]

    def forward(self, x, y):
        return torch.relu(x * y)


class CortexMHalfMulHardtanh(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
        "executorch_exir_dialects_edge__ops_aten_hardtanh_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_mul_f16_default": 1,
    }

    ops_removed_after_transforms = [
        "executorch_exir_dialects_edge__ops_aten_hardtanh_default",
    ]

    def __init__(self, min_val=-0.5, max_val=0.5):
        super().__init__()
        self.act = torch.nn.Hardtanh(min_val=min_val, max_val=max_val)

    def forward(self, x, y):
        return self.act(x * y)


test_cases = {
    "f32_rank_1": McuTestCase(
        CortexMFloatMul(),
        (
            ramp_tensor(-2.0, 2.0, (8,)).to(torch.float32),
            ramp_tensor(2.0, -2.0, (8,)).to(torch.float32),
        ),
    ),
    "f32_rank_4_small": McuTestCase(
        CortexMFloatMul(),
        (
            ramp_tensor(-0.1, 0.1, (2, 2, 2, 2)).to(torch.float32),
            ramp_tensor(0.2, -0.2, (2, 2, 2, 2)).to(torch.float32),
        ),
    ),
    "f32_channels_last": McuTestCase(
        CortexMFloatMul(),
        (
            ramp_tensor(-5.0, 5.0, (1, 4, 8, 8))
            .to(torch.float32)
            .to(memory_format=torch.channels_last),
            ramp_tensor(-3.0, 3.0, (1, 4, 8, 8))
            .to(torch.float32)
            .to(memory_format=torch.channels_last),
        ),
    ),
    "f32_broadcast_channels_1": McuTestCase(
        CortexMFloatMul(),
        (
            ramp_tensor(-2.0, 2.0, (1, 8, 1, 1))
            .to(torch.float32)
            .to(memory_format=torch.channels_last),
            ramp_tensor(-5.0, 5.0, (1, 8, 5, 5))
            .to(torch.float32)
            .to(memory_format=torch.channels_last),
        ),
    ),
    "f32_broadcast_channels_2": McuTestCase(
        CortexMFloatMul(),
        (
            ramp_tensor(-5.0, 5.0, (2, 8, 5, 5))
            .to(torch.float32)
            .to(memory_format=torch.channels_last),
            ramp_tensor(-2.0, 2.0, (1, 8, 1, 1))
            .to(torch.float32)
            .to(memory_format=torch.channels_last),
        ),
    ),
    "f16_rank_1": McuTestCase(
        CortexMHalfMul(),
        (
            ramp_tensor(-2.0, 2.0, (8,)).to(torch.float16),
            ramp_tensor(2.0, -2.0, (8,)).to(torch.float16),
        ),
    ),
    "f16_rank_4_small": McuTestCase(
        CortexMHalfMul(),
        (
            ramp_tensor(-0.1, 0.1, (2, 2, 2, 2)).to(torch.float16),
            ramp_tensor(0.2, -0.2, (2, 2, 2, 2)).to(torch.float16),
        ),
    ),
    "f16_channels_last": McuTestCase(
        CortexMHalfMul(),
        (
            ramp_tensor(-2.0, 2.0, (1, 4, 4, 4))
            .to(torch.float16)
            .to(memory_format=torch.channels_last),
            ramp_tensor(-1.0, 1.0, (1, 4, 4, 4))
            .to(torch.float16)
            .to(memory_format=torch.channels_last),
        ),
    ),
    "f16_broadcast_channels_1": McuTestCase(
        CortexMHalfMul(),
        (
            ramp_tensor(-2.0, 2.0, (1, 8, 1, 1))
            .to(torch.float16)
            .to(memory_format=torch.channels_last),
            ramp_tensor(-5.0, 5.0, (1, 8, 5, 5))
            .to(torch.float16)
            .to(memory_format=torch.channels_last),
        ),
    ),
    "f16_broadcast_channels_2": McuTestCase(
        CortexMHalfMul(),
        (
            ramp_tensor(-5.0, 5.0, (2, 8, 5, 5))
            .to(torch.float16)
            .to(memory_format=torch.channels_last),
            ramp_tensor(-2.0, 2.0, (1, 8, 1, 1))
            .to(torch.float16)
            .to(memory_format=torch.channels_last),
        ),
    ),
    "f32_relu_rank_1": McuTestCase(
        CortexMFloatMulReLU(),
        (
            ramp_tensor(-2.0, 2.0, (8,)).to(torch.float32),
            ramp_tensor(2.0, -2.0, (8,)).to(torch.float32),
        ),
    ),
    "f16_hardtanh_rank_1": McuTestCase(
        CortexMHalfMulHardtanh(),
        (
            ramp_tensor(-2.0, 2.0, (8,)).to(torch.float16),
            ramp_tensor(2.0, -2.0, (8,)).to(torch.float16),
        ),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_mul_float(test_case):
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
def test_implementation_mul_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.run_passes()
    tester.to_executorch()
    tester.serialize()
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs)
