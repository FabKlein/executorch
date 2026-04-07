# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test.common import parametrize, XfailIfNoCorstone300
from executorch.backends.cortex_m.test.tester import CortexMTester, McuTestCase


class CortexMBmmFloat(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        suffix = "f32" if dtype == torch.float32 else "f16"
        self.ops_before_transforms = {
            "executorch_exir_dialects_edge__ops_aten_bmm_default": 1,
        }
        self.ops_after_transforms = {
            f"executorch_exir_dialects_edge__ops_cortex_m_batch_matmul_{suffix}_default": 1,
            "executorch_exir_dialects_edge__ops_cortex_m_transpose_default": 1,
        }

    def forward(self, lhs, rhs):
        return torch.bmm(lhs, rhs)


class CortexMBmmFloatConstantRhs(torch.nn.Module):
    def __init__(self, dtype: torch.dtype, rhs_shape: tuple[int, int, int]):
        super().__init__()
        rhs = torch.linspace(
            -1.0,
            1.0,
            steps=rhs_shape[0] * rhs_shape[1] * rhs_shape[2],
            dtype=torch.float32,
        ).reshape(rhs_shape)
        self.register_buffer("rhs", rhs.to(dtype))
        suffix = "f32" if dtype == torch.float32 else "f16"
        self.ops_before_transforms = {
            "executorch_exir_dialects_edge__ops_aten_bmm_default": 1,
        }
        self.ops_after_transforms = {
            f"executorch_exir_dialects_edge__ops_cortex_m_batch_matmul_{suffix}_default": 1,
        }

    def forward(self, lhs):
        return torch.bmm(lhs, self.rhs)


def _ramp(dtype: torch.dtype, shape: tuple[int, int, int], start: float, end: float):
    steps = shape[0] * shape[1] * shape[2]
    return torch.linspace(start, end, steps=steps, dtype=dtype).reshape(shape)


test_cases = {
    "f32_bmm_small": McuTestCase(
        CortexMBmmFloat(torch.float32),
        (
            _ramp(torch.float32, (1, 2, 3), -1.0, 1.0),
            _ramp(torch.float32, (1, 3, 4), -0.5, 1.5),
        ),
    ),
    "f32_bmm_square": McuTestCase(
        CortexMBmmFloat(torch.float32),
        (
            _ramp(torch.float32, (2, 4, 4), -1.0, 2.0),
            _ramp(torch.float32, (2, 4, 4), -2.0, 1.0),
        ),
    ),
    "f32_bmm_large_batch": McuTestCase(
        CortexMBmmFloat(torch.float32),
        (
            _ramp(torch.float32, (8, 3, 5), -1.0, 1.0),
            _ramp(torch.float32, (8, 5, 2), -0.75, 0.75),
        ),
    ),
    "f32_bmm_single_batch": McuTestCase(
        CortexMBmmFloat(torch.float32),
        (
            _ramp(torch.float32, (1, 8, 16), -1.0, 1.0),
            _ramp(torch.float32, (1, 16, 8), -0.5, 0.5),
        ),
    ),
    "f32_bmm_tall_skinny": McuTestCase(
        CortexMBmmFloat(torch.float32),
        (
            _ramp(torch.float32, (2, 16, 1), -1.0, 1.0),
            _ramp(torch.float32, (2, 1, 16), -1.0, 1.0),
        ),
    ),
    "f32_bmm_wide_short": McuTestCase(
        CortexMBmmFloat(torch.float32),
        (
            _ramp(torch.float32, (2, 1, 16), -1.0, 1.0),
            _ramp(torch.float32, (2, 16, 1), -1.0, 1.0),
        ),
    ),
    "f32_bmm_rectangular_medium": McuTestCase(
        CortexMBmmFloat(torch.float32),
        (
            _ramp(torch.float32, (2, 3, 7), -1.0, 1.0),
            _ramp(torch.float32, (2, 7, 5), -0.75, 1.25),
        ),
    ),
    "f32_bmm_batch3_rectangular": McuTestCase(
        CortexMBmmFloat(torch.float32),
        (
            _ramp(torch.float32, (3, 2, 5), -1.5, 0.5),
            _ramp(torch.float32, (3, 5, 4), -0.25, 1.75),
        ),
    ),
    "f16_bmm_small": McuTestCase(
        CortexMBmmFloat(torch.float16),
        (
            _ramp(torch.float16, (1, 2, 3), -1.0, 1.0),
            _ramp(torch.float16, (1, 3, 4), -0.5, 1.5),
        ),
    ),
    "f16_bmm_square": McuTestCase(
        CortexMBmmFloat(torch.float16),
        (
            _ramp(torch.float16, (2, 4, 4), -1.0, 2.0),
            _ramp(torch.float16, (2, 4, 4), -2.0, 1.0),
        ),
    ),
    "f16_bmm_large_batch": McuTestCase(
        CortexMBmmFloat(torch.float16),
        (
            _ramp(torch.float16, (8, 3, 5), -1.0, 1.0),
            _ramp(torch.float16, (8, 5, 2), -0.75, 0.75),
        ),
    ),
    "f16_bmm_single_batch": McuTestCase(
        CortexMBmmFloat(torch.float16),
        (
            _ramp(torch.float16, (1, 8, 16), -1.0, 1.0),
            _ramp(torch.float16, (1, 16, 8), -0.5, 0.5),
        ),
    ),
    "f16_bmm_tall_skinny": McuTestCase(
        CortexMBmmFloat(torch.float16),
        (
            _ramp(torch.float16, (2, 16, 1), -1.0, 1.0),
            _ramp(torch.float16, (2, 1, 16), -1.0, 1.0),
        ),
    ),
    "f16_bmm_wide_short": McuTestCase(
        CortexMBmmFloat(torch.float16),
        (
            _ramp(torch.float16, (2, 1, 16), -1.0, 1.0),
            _ramp(torch.float16, (2, 16, 1), -1.0, 1.0),
        ),
    ),
    "f16_bmm_rectangular_medium": McuTestCase(
        CortexMBmmFloat(torch.float16),
        (
            _ramp(torch.float16, (2, 3, 7), -1.0, 1.0),
            _ramp(torch.float16, (2, 7, 5), -0.75, 1.25),
        ),
    ),
    "f16_bmm_batch3_rectangular": McuTestCase(
        CortexMBmmFloat(torch.float16),
        (
            _ramp(torch.float16, (3, 2, 5), -1.5, 0.5),
            _ramp(torch.float16, (3, 5, 4), -0.25, 1.75),
        ),
    ),
}


const_rhs_test_cases = {
    "f32_const_rhs_small": McuTestCase(
        CortexMBmmFloatConstantRhs(torch.float32, (1, 3, 4)),
        (_ramp(torch.float32, (1, 2, 3), -1.0, 1.0),),
    ),
    "f32_const_rhs_rectangular": McuTestCase(
        CortexMBmmFloatConstantRhs(torch.float32, (2, 7, 5)),
        (_ramp(torch.float32, (2, 3, 7), -1.0, 1.0),),
    ),
    "f16_const_rhs_small": McuTestCase(
        CortexMBmmFloatConstantRhs(torch.float16, (1, 3, 4)),
        (_ramp(torch.float16, (1, 2, 3), -1.0, 1.0),),
    ),
    "f16_const_rhs_rectangular": McuTestCase(
        CortexMBmmFloatConstantRhs(torch.float16, (2, 7, 5)),
        (_ramp(torch.float16, (2, 3, 7), -1.0, 1.0),),
    ),
}


def _compare_kwargs(test_case):
    lhs = test_case.example_inputs[0]
    if lhs.dtype != torch.float16:
        return {}
    if test_case is test_cases["f16_bmm_single_batch"]:
        return {"atol": 2.5e-3, "rtol": 2.5e-3}
    return {}


@parametrize("test_case", test_cases)
def test_dialect_batch_matmul_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.check_count(test_case.model.ops_before_transforms)
    tester.run_passes()
    tester.check_count(test_case.model.ops_after_transforms)
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs)


@parametrize("test_case", const_rhs_test_cases)
def test_dialect_batch_matmul_float_const_rhs(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.check_count(test_case.model.ops_before_transforms)
    tester.run_passes()
    tester.check_count(test_case.model.ops_after_transforms)
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs)


@XfailIfNoCorstone300
@parametrize("test_case", test_cases)
def test_implementation_batch_matmul_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.run_passes()
    tester.to_executorch()
    tester.serialize()
    tester.run_method_and_compare_outputs(
        inputs=test_case.example_inputs, **_compare_kwargs(test_case)
    )


@XfailIfNoCorstone300
@parametrize("test_case", const_rhs_test_cases)
def test_implementation_batch_matmul_float_const_rhs(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.run_passes()
    tester.to_executorch()
    tester.serialize()
    tester.run_method_and_compare_outputs(
        inputs=test_case.example_inputs, **_compare_kwargs(test_case)
    )
