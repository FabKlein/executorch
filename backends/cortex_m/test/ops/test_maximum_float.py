# Copyright 2026 Arm Limited and/or its affiliates.
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


class CortexMFloatMaximum(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_maximum_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_maximum_default": 1,
    }

    def forward(self, x, y):
        return torch.maximum(x, y)


class CortexMHalfMaximum(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_maximum_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_maximum_default": 1,
    }

    def forward(self, x, y):
        return torch.maximum(x, y)


test_cases = {
    "f32_tensor_small": McuTestCase(
        CortexMFloatMaximum(),
        (
            torch.tensor([[1.0, -2.0], [3.5, -4.5]], dtype=torch.float32),
            torch.tensor([[0.5, -1.0], [4.0, -3.5]], dtype=torch.float32),
        ),
    ),
    "f32_broadcast": McuTestCase(
        CortexMFloatMaximum(),
        (
            ramp_tensor(-2, 2, (2, 1, 2)).to(torch.float32),
            ramp_tensor(-3, 3, (1, 2, 1)).to(torch.float32),
        ),
    ),
    "f32_broadcast_scalar": McuTestCase(
        CortexMFloatMaximum(),
        (
            torch.tensor(1.0, dtype=torch.float32),
            ramp_tensor(-6, 6, (4, 1, 1, 3)).to(torch.float32),
        ),
    ),
    "f16_tensor_small": McuTestCase(
        CortexMHalfMaximum(),
        (
            torch.tensor([[1.0, -2.0], [3.5, -4.5]], dtype=torch.float16),
            torch.tensor([[0.5, -1.0], [4.0, -3.5]], dtype=torch.float16),
        ),
    ),
    "f16_broadcast_rank4": McuTestCase(
        CortexMHalfMaximum(),
        (
            ramp_tensor(-4, 4, (1, 2, 3, 1)).to(torch.float16),
            ramp_tensor(-6, 6, (4, 1, 1, 3)).to(torch.float16),
        ),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_maximum_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.check_count(test_case.model.ops_before_transforms)
    tester.run_passes()
    tester.check_count(test_case.model.ops_after_transforms)
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs)


@XfailIfNoCorstone300
@parametrize("test_case", test_cases)
def test_implementation_maximum_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.run_passes()
    tester.to_executorch()
    tester.serialize()
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs)
