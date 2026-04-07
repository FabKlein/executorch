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


OPS_BEFORE_TRANSFORMS = {
    "executorch_exir_dialects_edge__ops_aten_permute_copy_default": 1,
}

OPS_AFTER_TRANSFORMS = {
    "executorch_exir_dialects_edge__ops_cortex_m_transpose_default": 1,
}


class CortexMFloatPermute(torch.nn.Module):
    ops_before_transforms = OPS_BEFORE_TRANSFORMS
    ops_after_transforms = OPS_AFTER_TRANSFORMS

    def __init__(self, perms):
        super().__init__()
        self.perms = perms

    def forward(self, x):
        return x.permute(self.perms)


class CortexMFloatTranspose(torch.nn.Module):
    ops_before_transforms = OPS_BEFORE_TRANSFORMS
    ops_after_transforms = OPS_AFTER_TRANSFORMS

    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class CortexMFloatT(torch.nn.Module):
    ops_before_transforms = OPS_BEFORE_TRANSFORMS
    ops_after_transforms = OPS_AFTER_TRANSFORMS

    def forward(self, x):
        return x.t()


test_cases = {
    "f32_permute_nhwc_to_nchw": McuTestCase(
        CortexMFloatPermute((0, 3, 1, 2)),
        (ramp_tensor(-0.5, 0.5, (2, 3, 4, 2)).to(torch.float32),),
    ),
    "f32_transpose_1_2": McuTestCase(
        CortexMFloatTranspose(1, 2),
        (ramp_tensor(-1.0, 1.0, (1, 3, 4)).to(torch.float32),),
    ),
    "f32_t_operator": McuTestCase(
        CortexMFloatT(),
        (ramp_tensor(-0.5, 0.5, (4, 2)).to(torch.float32),),
    ),
    "f16_permute_nchw_to_nhwc_neg_index": McuTestCase(
        CortexMFloatPermute((0, -2, -1, -3)),
        (ramp_tensor(10, 100, (2, 3, 4, 2)).to(torch.float16),),
    ),
    "f16_transpose_0_1": McuTestCase(
        CortexMFloatTranspose(0, 1),
        (ramp_tensor(-2.0, 2.0, (2, 3, 4, 3)).to(torch.float16),),
    ),
    "f16_t_operator": McuTestCase(
        CortexMFloatT(),
        (ramp_tensor(-1.0, 1.0, (2, 4)).to(torch.float16),),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_transpose_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.check_count(test_case.model.ops_before_transforms)
    tester.run_passes()
    tester.check_count(test_case.model.ops_after_transforms)
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs)


@XfailIfNoCorstone300
@parametrize("test_case", test_cases)
def test_implementation_transpose_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.run_passes()
    tester.to_executorch()
    tester.serialize()
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs)
