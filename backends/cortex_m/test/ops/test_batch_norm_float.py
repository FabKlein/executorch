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


class CortexMBatchNormFloat(torch.nn.Module):
    def __init__(self, dtype: torch.dtype, channels: int):
        super().__init__()
        self.dtype = dtype
        self.register_buffer(
            "scale",
            torch.linspace(0.75, 1.25, steps=channels, dtype=torch.float32).to(dtype),
        )
        self.register_buffer(
            "bias",
            torch.linspace(-0.2, 0.2, steps=channels, dtype=torch.float32).to(dtype),
        )
        suffix = "f32" if dtype == torch.float32 else "f16"
        op_name = (
            f"executorch_exir_dialects_edge__ops_cortex_m_batch_norm_{suffix}_default"
        )
        self.ops_before_transforms = {op_name: 1}
        self.ops_after_transforms = {op_name: 1}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.dtype).contiguous(memory_format=torch.channels_last)
        if self.dtype == torch.float32:
            return torch.ops.cortex_m.batch_norm_f32.default(x, self.scale, self.bias)
        return torch.ops.cortex_m.batch_norm_f16.default(x, self.scale, self.bias)


class CortexMNativeBatchNormFloat(torch.nn.Module):
    def __init__(self, dtype: torch.dtype, channels: int):
        super().__init__()
        self.dtype = dtype
        self.bn = torch.nn.BatchNorm2d(
            channels, affine=True, track_running_stats=True
        ).to(dtype)
        self.bn.eval()
        self.bn.weight.data.copy_(
            torch.linspace(0.75, 1.25, steps=channels, dtype=torch.float32).to(dtype)
        )
        self.bn.bias.data.copy_(
            torch.linspace(-0.2, 0.2, steps=channels, dtype=torch.float32).to(dtype)
        )
        self.bn.running_mean.copy_(
            torch.linspace(-0.3, 0.3, steps=channels, dtype=torch.float32).to(dtype)
        )
        self.bn.running_var.copy_(
            torch.linspace(0.5, 1.5, steps=channels, dtype=torch.float32).to(dtype)
        )
        suffix = "f32" if dtype == torch.float32 else "f16"
        self.ops_before_transforms = {
            "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default": 1
        }
        self.ops_after_transforms = {
            f"executorch_exir_dialects_edge__ops_cortex_m_batch_norm_native_{suffix}_default": 1
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.dtype).contiguous(memory_format=torch.channels_last)
        return self.bn(x)


test_cases = {
    "f32_basic": McuTestCase(
        CortexMBatchNormFloat(torch.float32, channels=6),
        (
            ramp_tensor(-1.5, 1.5, (1, 6, 4, 4))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_batch2": McuTestCase(
        CortexMBatchNormFloat(torch.float32, channels=4),
        (
            ramp_tensor(-2.0, 2.0, (2, 4, 3, 5))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_basic": McuTestCase(
        CortexMBatchNormFloat(torch.float16, channels=6),
        (
            ramp_tensor(-1.5, 1.5, (1, 6, 4, 4))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_batch2": McuTestCase(
        CortexMBatchNormFloat(torch.float16, channels=4),
        (
            ramp_tensor(-2.0, 2.0, (2, 4, 3, 5))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
}

native_test_cases = {
    "f32_native_basic": McuTestCase(
        CortexMNativeBatchNormFloat(torch.float32, channels=6),
        (
            ramp_tensor(-1.5, 1.5, (1, 6, 4, 4))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_native_batch2": McuTestCase(
        CortexMNativeBatchNormFloat(torch.float32, channels=4),
        (
            ramp_tensor(-2.0, 2.0, (2, 4, 3, 5))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_native_basic": McuTestCase(
        CortexMNativeBatchNormFloat(torch.float16, channels=6),
        (
            ramp_tensor(-1.5, 1.5, (1, 6, 4, 4))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_native_batch2": McuTestCase(
        CortexMNativeBatchNormFloat(torch.float16, channels=4),
        (
            ramp_tensor(-2.0, 2.0, (2, 4, 3, 5))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
}


def _float_compare_kwargs(test_case):
    if test_case.model.dtype == torch.float16:
        return {"atol": 2e-3, "rtol": 2e-3}
    return {"atol": 1e-5, "rtol": 1e-5}


@parametrize("test_case", test_cases)
def test_dialect_batch_norm_float(test_case):
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
def test_implementation_batch_norm_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.run_passes()
    tester.to_executorch()
    tester.serialize()
    tester.run_method_and_compare_outputs(
        inputs=test_case.example_inputs, **_float_compare_kwargs(test_case)
    )


@parametrize("test_case", native_test_cases)
def test_dialect_native_batch_norm_float(test_case):
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
@parametrize("test_case", native_test_cases)
def test_implementation_native_batch_norm_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.run_passes()
    tester.to_executorch()
    tester.serialize()
    tester.run_method_and_compare_outputs(
        inputs=test_case.example_inputs, **_float_compare_kwargs(test_case)
    )
