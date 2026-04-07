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


class CortexMLinearFloat(torch.nn.Module):
    def __init__(
        self, dtype: torch.dtype, in_features: int, out_features: int, bias: bool
    ):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        with torch.no_grad():
            weight = torch.linspace(
                -1.0,
                1.0,
                steps=out_features * in_features,
                dtype=torch.float32,
            ).reshape(out_features, in_features)
            self.linear.weight.copy_(weight)
            if bias:
                self.linear.bias.copy_(
                    torch.linspace(-0.25, 0.25, steps=out_features, dtype=torch.float32)
                )
        self.linear = self.linear.to(dtype)

        suffix = "f32" if dtype == torch.float32 else "f16"
        self.ops_before_transforms = {
            "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
        }
        self.ops_after_transforms = {
            f"executorch_exir_dialects_edge__ops_cortex_m_linear_{suffix}_default": 1,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class CortexMLinearFloatX3(torch.nn.Module):
    def __init__(
        self, dtype: torch.dtype, in_features: int, out_features: int, bias: bool
    ):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        with torch.no_grad():
            weight = torch.linspace(
                -1.0,
                1.0,
                steps=out_features * in_features,
                dtype=torch.float32,
            ).reshape(out_features, in_features)
            self.linear.weight.copy_(weight)
            if bias:
                self.linear.bias.copy_(
                    torch.linspace(-0.25, 0.25, steps=out_features, dtype=torch.float32)
                )
        self.linear = self.linear.to(dtype)

        suffix = "f32" if dtype == torch.float32 else "f16"
        self.ops_before_transforms = {
            "executorch_exir_dialects_edge__ops_aten_linear_default": 3,
        }
        self.ops_after_transforms = {
            f"executorch_exir_dialects_edge__ops_cortex_m_linear_{suffix}_default": 3,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.linear(x)
        x = self.linear(x)
        return x


class CortexMLinearBatchNormFloat(torch.nn.Module):
    def __init__(self, dtype: torch.dtype, in_features: int, out_features: int):
        super().__init__()
        self.dtype = dtype
        self.linear = torch.nn.Linear(in_features, out_features, bias=True)
        self.bn = torch.nn.BatchNorm1d(
            out_features, affine=True, track_running_stats=True
        )
        with torch.no_grad():
            weight = torch.linspace(
                -0.8,
                0.8,
                steps=out_features * in_features,
                dtype=torch.float32,
            ).reshape(out_features, in_features)
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(
                torch.linspace(-0.2, 0.2, steps=out_features, dtype=torch.float32)
            )
            self.bn.weight.copy_(
                torch.linspace(0.7, 1.3, steps=out_features, dtype=torch.float32)
            )
            self.bn.bias.copy_(
                torch.linspace(-0.3, 0.3, steps=out_features, dtype=torch.float32)
            )
            self.bn.running_mean.copy_(
                torch.linspace(-0.4, 0.4, steps=out_features, dtype=torch.float32)
            )
            self.bn.running_var.copy_(
                torch.linspace(0.6, 1.6, steps=out_features, dtype=torch.float32)
            )
        self.linear = self.linear.to(dtype)
        self.bn = self.bn.to(dtype).eval()

        suffix = "f32" if dtype == torch.float32 else "f16"
        self.ops_before_transforms = {
            "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
            "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default": 1,
        }
        self.ops_after_transforms = {
            f"executorch_exir_dialects_edge__ops_cortex_m_linear_{suffix}_default": 1,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.linear(x.to(dtype=self.dtype)))


test_cases = {
    "f32_rank1": McuTestCase(
        CortexMLinearFloat(torch.float32, 4, 3, bias=False),
        (ramp_tensor(-1, 1, (4,)).to(torch.float32),),
    ),
    "f32_rank2": McuTestCase(
        CortexMLinearFloat(torch.float32, 4, 3, bias=False),
        (ramp_tensor(-1.5, 1.5, (2, 4)).to(torch.float32),),
    ),
    "f32_rank3_bias": McuTestCase(
        CortexMLinearFloat(torch.float32, 4, 3, bias=True),
        (ramp_tensor(-2, 2, (2, 2, 4)).to(torch.float32),),
    ),
    "f32_rank4": McuTestCase(
        CortexMLinearFloat(torch.float32, 4, 2, bias=False),
        (ramp_tensor(-2.5, 2.5, (2, 1, 3, 4)).to(torch.float32),),
    ),
    "f32_rank5": McuTestCase(
        CortexMLinearFloat(torch.float32, 4, 2, bias=False),
        (ramp_tensor(-3, 3, (2, 1, 2, 3, 4)).to(torch.float32),),
    ),
    "f32_x3": McuTestCase(
        CortexMLinearFloatX3(torch.float32, 4, 4, bias=False),
        (ramp_tensor(-1, 1, (2, 4)).to(torch.float32),),
    ),
    "f32_linear_bn": McuTestCase(
        CortexMLinearBatchNormFloat(torch.float32, 4, 3),
        (ramp_tensor(-1, 1, (2, 4)).to(torch.float32),),
    ),
    "f16_rank1": McuTestCase(
        CortexMLinearFloat(torch.float16, 4, 3, bias=False),
        (ramp_tensor(-1, 1, (4,)).to(torch.float16),),
    ),
    "f16_rank2": McuTestCase(
        CortexMLinearFloat(torch.float16, 4, 3, bias=False),
        (ramp_tensor(-1.5, 1.5, (2, 4)).to(torch.float16),),
    ),
    "f16_rank3_bias": McuTestCase(
        CortexMLinearFloat(torch.float16, 4, 3, bias=True),
        (ramp_tensor(-2, 2, (2, 2, 4)).to(torch.float16),),
    ),
    "f16_rank4": McuTestCase(
        CortexMLinearFloat(torch.float16, 4, 2, bias=False),
        (ramp_tensor(-2.5, 2.5, (2, 1, 3, 4)).to(torch.float16),),
    ),
    "f16_rank5": McuTestCase(
        CortexMLinearFloat(torch.float16, 4, 2, bias=False),
        (ramp_tensor(-3, 3, (2, 1, 2, 3, 4)).to(torch.float16),),
    ),
    "f16_x3": McuTestCase(
        CortexMLinearFloatX3(torch.float16, 4, 4, bias=False),
        (ramp_tensor(-1, 1, (2, 4)).to(torch.float16),),
    ),
    "f16_linear_bn": McuTestCase(
        CortexMLinearBatchNormFloat(torch.float16, 4, 3),
        (ramp_tensor(-1, 1, (2, 4)).to(torch.float16),),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_linear_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.check_count(test_case.model.ops_before_transforms)
    tester.run_passes()
    tester.check_count(test_case.model.ops_after_transforms)
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs)


@XfailIfNoCorstone300
@parametrize("test_case", test_cases)
def test_implementation_linear_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.run_passes()
    tester.to_executorch()
    tester.serialize()
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs)
