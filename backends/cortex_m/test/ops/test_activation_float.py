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


class CortexMActivationFloat(torch.nn.Module):
    def __init__(self, dtype, kind: str):
        super().__init__()
        self.dtype = dtype
        self.kind = kind
        suffix = "f32" if dtype == torch.float32 else "f16"

        aten_name = {
            "relu": "executorch_exir_dialects_edge__ops_aten_relu_default",
            "sigmoid": "executorch_exir_dialects_edge__ops_aten_sigmoid_default",
            "tanh": "executorch_exir_dialects_edge__ops_aten_tanh_default",
            "hardswish": "executorch_exir_dialects_edge__ops_aten_hardswish_default",
            "leaky_relu": "executorch_exir_dialects_edge__ops_aten_leaky_relu_default",
            "relu6": "executorch_exir_dialects_edge__ops_aten_hardtanh_default",
        }[kind]

        self.ops_before_transforms = {aten_name: 1}
        self.ops_after_transforms = {
            f"executorch_exir_dialects_edge__ops_cortex_m_activation_{suffix}_default": 1,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.dtype)
        if self.kind == "relu":
            return torch.relu(x)
        if self.kind == "sigmoid":
            return torch.sigmoid(x)
        if self.kind == "tanh":
            return torch.tanh(x)
        if self.kind == "hardswish":
            return torch.nn.functional.hardswish(x)
        if self.kind == "leaky_relu":
            return torch.nn.functional.leaky_relu(x, negative_slope=0.125)
        if self.kind == "relu6":
            return torch.nn.functional.hardtanh(x, min_val=0.0, max_val=6.0)
        raise AssertionError(f"Unsupported activation kind {self.kind}")


test_cases = {
    "f32_relu": McuTestCase(
        CortexMActivationFloat(torch.float32, "relu"),
        (ramp_tensor(-4, 4, (2, 3, 4)).to(torch.float32),),
    ),
    "f32_sigmoid": McuTestCase(
        CortexMActivationFloat(torch.float32, "sigmoid"),
        (ramp_tensor(-6, 6, (3, 5)).to(torch.float32),),
    ),
    "f32_tanh": McuTestCase(
        CortexMActivationFloat(torch.float32, "tanh"),
        (ramp_tensor(-4, 4, (4, 4)).to(torch.float32),),
    ),
    "f32_hardswish": McuTestCase(
        CortexMActivationFloat(torch.float32, "hardswish"),
        (ramp_tensor(-5, 5, (1, 2, 3, 4)).to(torch.float32),),
    ),
    "f32_leaky_relu": McuTestCase(
        CortexMActivationFloat(torch.float32, "leaky_relu"),
        (ramp_tensor(-3, 3, (2, 3, 4)).to(torch.float32),),
    ),
    "f32_relu6": McuTestCase(
        CortexMActivationFloat(torch.float32, "relu6"),
        (ramp_tensor(-3, 9, (2, 3, 4)).to(torch.float32),),
    ),
    "f16_relu": McuTestCase(
        CortexMActivationFloat(torch.float16, "relu"),
        (ramp_tensor(-4, 4, (2, 3, 4)).to(torch.float16),),
    ),
    "f16_sigmoid": McuTestCase(
        CortexMActivationFloat(torch.float16, "sigmoid"),
        (ramp_tensor(-6, 6, (3, 5)).to(torch.float16),),
    ),
    "f16_tanh": McuTestCase(
        CortexMActivationFloat(torch.float16, "tanh"),
        (ramp_tensor(-4, 4, (4, 4)).to(torch.float16),),
    ),
    "f16_hardswish": McuTestCase(
        CortexMActivationFloat(torch.float16, "hardswish"),
        (ramp_tensor(-5, 5, (1, 2, 3, 4)).to(torch.float16),),
    ),
    "f16_leaky_relu": McuTestCase(
        CortexMActivationFloat(torch.float16, "leaky_relu"),
        (ramp_tensor(-3, 3, (2, 3, 4)).to(torch.float16),),
    ),
    "f16_relu6": McuTestCase(
        CortexMActivationFloat(torch.float16, "relu6"),
        (ramp_tensor(-3, 9, (2, 3, 4)).to(torch.float16),),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_activation_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.check_count(test_case.model.ops_before_transforms)
    tester.run_passes()
    tester.check_count(test_case.model.ops_after_transforms)
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs)


@XfailIfNoCorstone300
@parametrize("test_case", test_cases)
def test_implementation_activation_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.run_passes()
    tester.to_executorch()
    tester.serialize()
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs)
