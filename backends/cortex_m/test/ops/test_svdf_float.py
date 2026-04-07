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


def _svdf_weights(
    dtype: torch.dtype, unit_count: int, rank: int, input_size: int, memory_size: int
):
    feature_batches = unit_count * rank
    weights_feature = (
        torch.linspace(
            -0.5,
            0.5,
            steps=feature_batches * input_size,
            dtype=torch.float32,
        )
        .reshape(feature_batches, input_size)
        .to(dtype)
    )
    weights_time = (
        torch.linspace(
            -0.4,
            0.4,
            steps=feature_batches * memory_size,
            dtype=torch.float32,
        )
        .reshape(feature_batches, memory_size)
        .to(dtype)
    )
    bias = torch.linspace(-0.25, 0.25, steps=unit_count, dtype=torch.float32).to(dtype)
    return weights_feature, weights_time, bias


class CortexMSVDFFloat(torch.nn.Module):
    def __init__(
        self,
        dtype: torch.dtype,
        input_size: int,
        unit_count: int,
        rank: int,
        memory_size: int,
        batch_size: int,
        time_major: bool,
    ):
        super().__init__()
        self.dtype = dtype
        self.rank = rank
        self.time_major = time_major
        self.input_activation_min = -1.0
        self.input_activation_max = 1.0
        self.output_activation_min = -100.0
        self.output_activation_max = 100.0
        weights_feature, weights_time, bias = _svdf_weights(
            dtype, unit_count, rank, input_size, memory_size
        )
        feature_batches = unit_count * rank
        self.register_buffer("weights_feature", weights_feature)
        self.register_buffer("weights_time", weights_time)
        self.register_buffer("bias", bias)
        self.register_buffer(
            "initial_state",
            torch.zeros((batch_size, feature_batches, memory_size), dtype=dtype),
        )

        suffix = "f32" if dtype == torch.float32 else "f16"
        op_name = f"executorch_exir_dialects_edge__ops_cortex_m_svdf_{suffix}_default"
        self.ops_before_transforms = {op_name: 1}
        self.ops_after_transforms = {op_name: 1}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        if self.dtype == torch.float32:
            return torch.ops.cortex_m.svdf_f32.default(
                x,
                self.initial_state,
                self.weights_feature,
                self.weights_time,
                self.bias,
                self.time_major,
                self.rank,
                self.input_activation_min,
                self.input_activation_max,
                self.output_activation_min,
                self.output_activation_max,
            )
        return torch.ops.cortex_m.svdf_f16.default(
            x,
            self.initial_state,
            self.weights_feature,
            self.weights_time,
            self.bias,
            self.time_major,
            self.rank,
            self.input_activation_min,
            self.input_activation_max,
            self.output_activation_min,
            self.output_activation_max,
        )


test_cases = {
    "f32_time_major_small": McuTestCase(
        CortexMSVDFFloat(
            torch.float32,
            input_size=16,
            unit_count=6,
            rank=2,
            memory_size=4,
            batch_size=1,
            time_major=True,
        ),
        (ramp_tensor(-1.5, 1.5, (3, 1, 16)).to(torch.float32),),
    ),
    "f32_batch_major_batch2": McuTestCase(
        CortexMSVDFFloat(
            torch.float32,
            input_size=20,
            unit_count=8,
            rank=2,
            memory_size=3,
            batch_size=2,
            time_major=False,
        ),
        (ramp_tensor(-1.25, 1.25, (2, 4, 20)).to(torch.float32),),
    ),
    "f16_time_major_small": McuTestCase(
        CortexMSVDFFloat(
            torch.float16,
            input_size=16,
            unit_count=6,
            rank=2,
            memory_size=4,
            batch_size=1,
            time_major=True,
        ),
        (ramp_tensor(-1.5, 1.5, (3, 1, 16)).to(torch.float16),),
    ),
    "f16_batch_major_batch2": McuTestCase(
        CortexMSVDFFloat(
            torch.float16,
            input_size=20,
            unit_count=8,
            rank=2,
            memory_size=3,
            batch_size=2,
            time_major=False,
        ),
        (ramp_tensor(-1.25, 1.25, (2, 4, 20)).to(torch.float16),),
    ),
}


def _float_compare_kwargs(test_case):
    if test_case.model.dtype == torch.float16:
        return {"atol": 2e-2, "rtol": 2e-2}
    return {"atol": 2e-4, "rtol": 2e-4}


@parametrize("test_case", test_cases)
def test_dialect_svdf_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.check_count(test_case.model.ops_before_transforms)
    tester.run_passes()
    tester.check_count(test_case.model.ops_after_transforms)
    tester.run_method_and_compare_outputs(
        inputs=test_case.example_inputs, **_float_compare_kwargs(test_case)
    )


@XfailIfNoCorstone300
@parametrize("test_case", test_cases)
def test_implementation_svdf_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.run_passes()
    tester.to_executorch()
    tester.serialize()
    tester.run_method_and_compare_outputs(
        inputs=test_case.example_inputs, **_float_compare_kwargs(test_case)
    )
