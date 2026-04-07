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


class CortexMAddmmFloat(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.bias = torch.nn.Parameter(
            torch.linspace(-0.25, 0.25, steps=5, dtype=torch.float32)
        )
        self.weight = torch.nn.Parameter(
            torch.linspace(-1.0, 1.0, steps=20, dtype=torch.float32).reshape(5, 4)
        )
        self.bias.data = self.bias.data.to(dtype)
        self.weight.data = self.weight.data.to(dtype)
        suffix = "f32" if dtype == torch.float32 else "f16"
        self.ops_before_transforms = {
            "executorch_exir_dialects_edge__ops_aten_addmm_default": 1,
        }
        self.ops_after_transforms = {
            f"executorch_exir_dialects_edge__ops_cortex_m_linear_{suffix}_default": 1,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addmm(self.bias, x, self.weight.t())


class CortexMGlobalMeanFloat(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        suffix = "f32" if dtype == torch.float32 else "f16"
        self.ops_before_transforms = {
            "executorch_exir_dialects_edge__ops_aten_mean_dim": 1,
        }
        self.ops_after_transforms = {
            f"executorch_exir_dialects_edge__ops_cortex_m_avg_pool2d_{suffix}_default": 1,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(-2, -1), keepdim=True)


class CortexMDecomposedHardswishFloat(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        suffix = "f32" if dtype == torch.float32 else "f16"
        self.ops_before_transforms = {
            "executorch_exir_dialects_edge__ops_aten_add_Tensor": 1,
            "executorch_exir_dialects_edge__ops_aten_clamp_default": 2,
            "executorch_exir_dialects_edge__ops_aten_div_Tensor": 1,
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
        }
        self.ops_after_transforms = {
            f"executorch_exir_dialects_edge__ops_cortex_m_activation_{suffix}_default": 1,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.clamp(torch.clamp(x + 3.0, min=0.0), max=6.0) / 6.0


class CortexMDecomposedHardsigmoidFloat(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        suffix = "f32" if dtype == torch.float32 else "f16"
        self.ops_before_transforms = {
            "executorch_exir_dialects_edge__ops_aten_add_Tensor": 1,
            "executorch_exir_dialects_edge__ops_aten_clamp_default": 2,
            "executorch_exir_dialects_edge__ops_aten_div_Tensor": 1,
        }
        self.ops_after_transforms = {
            f"executorch_exir_dialects_edge__ops_cortex_m_activation_{suffix}_default": 1,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.clamp(x + 3.0, min=0.0), max=6.0) / 6.0


class CortexMConvBatchNormFoldFloat(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        suffix = "f32" if dtype == torch.float32 else "f16"
        self.conv = torch.nn.Conv2d(4, 6, kernel_size=3, padding=1, bias=True)
        self.bn = torch.nn.BatchNorm2d(6)
        self.conv.weight.data = self.conv.weight.data.to(dtype)
        self.conv.bias.data = self.conv.bias.data.to(dtype)
        self.bn.weight.data = self.bn.weight.data.to(dtype)
        self.bn.bias.data = self.bn.bias.data.to(dtype)
        self.bn.running_mean.data = self.bn.running_mean.data.to(dtype)
        self.bn.running_var.data = self.bn.running_var.data.to(dtype)
        self.bn.eval()
        self.ops_before_transforms = {
            "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
            "executorch_exir_dialects_edge__ops_aten__native_batch_norm_legit_no_training_default": 1,
        }
        self.ops_after_transforms = {
            f"executorch_exir_dialects_edge__ops_cortex_m_conv2d_{suffix}_default": 1,
            f"executorch_exir_dialects_edge__ops_cortex_m_batch_norm_native_{suffix}_default": 0,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


test_cases = {
    "f32_addmm": McuTestCase(
        CortexMAddmmFloat(torch.float32),
        (ramp_tensor(-1.0, 1.0, (2, 4)).to(torch.float32),),
    ),
    "f16_addmm": McuTestCase(
        CortexMAddmmFloat(torch.float16),
        (ramp_tensor(-1.0, 1.0, (2, 4)).to(torch.float16),),
    ),
    "f32_mean_dim_global_pool": McuTestCase(
        CortexMGlobalMeanFloat(torch.float32),
        (
            ramp_tensor(-2.0, 2.0, (1, 3, 7, 7))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_mean_dim_global_pool": McuTestCase(
        CortexMGlobalMeanFloat(torch.float16),
        (
            ramp_tensor(-2.0, 2.0, (1, 3, 7, 7))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_decomposed_hardswish": McuTestCase(
        CortexMDecomposedHardswishFloat(torch.float32),
        (ramp_tensor(-4.0, 4.0, (1, 8, 7, 7)).to(torch.float32),),
    ),
    "f16_decomposed_hardswish": McuTestCase(
        CortexMDecomposedHardswishFloat(torch.float16),
        (ramp_tensor(-4.0, 4.0, (1, 8, 7, 7)).to(torch.float16),),
    ),
    "f32_decomposed_hardsigmoid": McuTestCase(
        CortexMDecomposedHardsigmoidFloat(torch.float32),
        (ramp_tensor(-4.0, 4.0, (1, 8, 7, 7)).to(torch.float32),),
    ),
    "f16_decomposed_hardsigmoid": McuTestCase(
        CortexMDecomposedHardsigmoidFloat(torch.float16),
        (ramp_tensor(-4.0, 4.0, (1, 8, 7, 7)).to(torch.float16),),
    ),
    "f32_conv_batch_norm_fold": McuTestCase(
        CortexMConvBatchNormFoldFloat(torch.float32),
        (
            ramp_tensor(-2.0, 2.0, (1, 4, 8, 8))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_conv_batch_norm_fold": McuTestCase(
        CortexMConvBatchNormFoldFloat(torch.float16),
        (
            ramp_tensor(-2.0, 2.0, (1, 4, 8, 8))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
}


def _float_compare_kwargs(test_case):
    if test_case.model.ops_after_transforms.keys() and any(
        "_f16_" in op_name for op_name in test_case.model.ops_after_transforms
    ):
        return {"atol": 2e-3, "rtol": 2e-3}
    return {}


@parametrize("test_case", test_cases)
def test_dialect_mobilenet_patterns_float(test_case):
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
def test_implementation_mobilenet_patterns_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.run_passes()
    tester.to_executorch()
    tester.serialize()
    tester.run_method_and_compare_outputs(
        inputs=test_case.example_inputs, **_float_compare_kwargs(test_case)
    )
