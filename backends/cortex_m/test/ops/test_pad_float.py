# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F
from executorch.backends.arm.test.common import parametrize, XfailIfNoCorstone300
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)


class CortexMPadFloat(torch.nn.Module):
    def __init__(self, dtype, padding, value=0.0):
        super().__init__()
        self.dtype = dtype
        self.padding = padding
        self.value = value
        suffix = "f32" if dtype == torch.float32 else "f16"
        self.ops_before_transforms = {
            "executorch_exir_dialects_edge__ops_aten_constant_pad_nd_default": 1,
        }
        self.ops_after_transforms = {
            f"executorch_exir_dialects_edge__ops_cortex_m_pad_{suffix}_default": 1,
        }

    def forward(self, x):
        x = x.to(dtype=self.dtype)
        return F.pad(x, self.padding, mode="constant", value=self.value)


test_cases = {
    "f32_pad_rank4_all_dims": McuTestCase(
        CortexMPadFloat(torch.float32, (1, 1, 2, 2, 1, 0, 0, 1)),
        (
            ramp_tensor(-0.5, 0.5, (1, 2, 3, 4))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f32_pad_rank2": McuTestCase(
        CortexMPadFloat(torch.float32, (1, 2, 3, 4)),
        (ramp_tensor(-1.0, 1.0, (3, 5)).to(torch.float32),),
    ),
    "f32_pad_nonzero_value": McuTestCase(
        CortexMPadFloat(torch.float32, (1, 1), value=0.5),
        (ramp_tensor(-1.0, 1.0, (2, 4)).to(torch.float32),),
    ),
    "f32_pad_rank4_last_two_dims_channels_last": McuTestCase(
        CortexMPadFloat(torch.float32, (1, 2, 3, 4)),
        (
            ramp_tensor(-1.0, 1.0, (1, 3, 4, 5))
            .to(torch.float32)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_pad_rank4_all_dims": McuTestCase(
        CortexMPadFloat(torch.float16, (1, 1, 2, 2, 1, 0, 0, 1)),
        (
            ramp_tensor(-0.5, 0.5, (1, 2, 3, 4))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
    "f16_pad_rank2": McuTestCase(
        CortexMPadFloat(torch.float16, (1, 2, 3, 4)),
        (ramp_tensor(-1.0, 1.0, (3, 5)).to(torch.float16),),
    ),
    "f16_pad_nonzero_value": McuTestCase(
        CortexMPadFloat(torch.float16, (1, 1), value=0.5),
        (ramp_tensor(-1.0, 1.0, (2, 4)).to(torch.float16),),
    ),
    "f16_pad_rank4_last_two_dims_channels_last": McuTestCase(
        CortexMPadFloat(torch.float16, (1, 2, 3, 4)),
        (
            ramp_tensor(-1.0, 1.0, (1, 3, 4, 5))
            .to(torch.float16)
            .contiguous(memory_format=torch.channels_last),
        ),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_pad_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.check_count(test_case.model.ops_before_transforms)
    tester.run_passes()
    tester.check_count(test_case.model.ops_after_transforms)
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs)


@XfailIfNoCorstone300
@parametrize("test_case", test_cases)
def test_implementation_pad_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.run_passes()
    tester.to_executorch()
    tester.serialize()
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs)
