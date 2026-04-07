# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.arm.test.common import (
    parametrize,
    xfail_type,
    XfailIfNoCorstone300,
)
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)


class CortexMSoftmaxF32(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten__softmax_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_softmax_f32_default": 1,
    }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=self.dim)


class CortexMSoftmaxF16(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten__softmax_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_softmax_f16_default": 1,
    }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=self.dim)


test_cases = {
    "f32_rank1": McuTestCase(
        CortexMSoftmaxF32(dim=-1),
        (ramp_tensor(-4, 4, (16,)).to(torch.float32),),
    ),
    "f32_rank2": McuTestCase(
        CortexMSoftmaxF32(dim=-1),
        (ramp_tensor(-8, 8, (4, 8)).to(torch.float32),),
    ),
    "f32_rank3": McuTestCase(
        CortexMSoftmaxF32(dim=-1),
        (ramp_tensor(-2, 2, (2, 3, 4)).to(torch.float32),),
    ),
    "f32_dim_not_last": McuTestCase(
        CortexMSoftmaxF32(dim=1),
        (ramp_tensor(-2, 2, (2, 3, 4)).to(torch.float32),),
    ),
    "f16_rank1": McuTestCase(
        CortexMSoftmaxF16(dim=-1),
        (ramp_tensor(-4, 4, (16,)).to(torch.float16),),
    ),
    "f16_rank2": McuTestCase(
        CortexMSoftmaxF16(dim=-1),
        (ramp_tensor(-8, 8, (4, 8)).to(torch.float16),),
    ),
    "f16_rank3": McuTestCase(
        CortexMSoftmaxF16(dim=-1),
        (ramp_tensor(-2, 2, (2, 3, 4)).to(torch.float16),),
    ),
    "f16_dim_not_last": McuTestCase(
        CortexMSoftmaxF16(dim=1),
        (ramp_tensor(-2, 2, (2, 3, 4)).to(torch.float16),),
    ),
}


xfail_cases_dialect: dict[str, xfail_type] = {
    "f32_dim_not_last": (
        "Float softmax stays in ATen when dim isn’t the last dimension",
        Exception,
    ),
    "f16_dim_not_last": (
        "Float softmax stays in ATen when dim isn’t the last dimension",
        Exception,
    ),
}

xfail_cases_impl: dict[str, xfail_type] = {
    "f32_dim_not_last": (
        "Float softmax on Cortex-M currently supports only the last dimension",
        Exception,
    ),
    "f16_dim_not_last": (
        "Float softmax on Cortex-M currently supports only the last dimension",
        Exception,
    ),
}


@parametrize("test_case", test_cases, xfails=xfail_cases_dialect)
def test_dialect_softmax_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.check_count(test_case.model.ops_before_transforms)
    tester.run_passes()
    tester.check_count(test_case.model.ops_after_transforms)
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs)


@XfailIfNoCorstone300
@parametrize("test_case", test_cases, xfails=xfail_cases_impl)
def test_implementation_softmax_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.run_passes()
    tester.to_executorch()
    tester.serialize()
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs)
