# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test.common import parametrize, XfailIfNoCorstone300
from executorch.backends.cortex_m.test.tester import CortexMTester, McuTestCase


class CortexMMaxPool2dFloat(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_max_pool2d_with_indices_default": 1,
    }

    ops_after_transforms_f32 = {
        "executorch_exir_dialects_edge__ops_cortex_m_max_pool2d_f32_default": 1,
    }
    ops_after_transforms_f16 = {
        "executorch_exir_dialects_edge__ops_cortex_m_max_pool2d_f16_default": 1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


def _channels_last(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous(memory_format=torch.channels_last)


test_cases = {
    "f32_maxpool_2x2": McuTestCase(
        CortexMMaxPool2dFloat(kernel_size=2, stride=2),
        (
            _channels_last(
                torch.linspace(0.0, 15.0, steps=16, dtype=torch.float32).reshape(
                    1, 1, 4, 4
                )
            ),
        ),
    ),
    "f32_maxpool_3x3_s1": McuTestCase(
        CortexMMaxPool2dFloat(kernel_size=3, stride=1, padding=1),
        (
            _channels_last(
                torch.linspace(-4.0, 4.0, steps=18, dtype=torch.float32).reshape(
                    1, 2, 3, 3
                )
            ),
        ),
    ),
    "f32_maxpool_3x3_s2_pad1": McuTestCase(
        CortexMMaxPool2dFloat(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        (
            _channels_last(
                torch.linspace(-3.0, 5.0, steps=32, dtype=torch.float32).reshape(
                    1, 2, 4, 4
                )
            ),
        ),
    ),
    "f16_maxpool_2x2": McuTestCase(
        CortexMMaxPool2dFloat(kernel_size=2, stride=2),
        (
            _channels_last(
                torch.linspace(0.0, 15.0, steps=16, dtype=torch.float16).reshape(
                    1, 1, 4, 4
                )
            ),
        ),
    ),
    "f16_maxpool_3x3_s1": McuTestCase(
        CortexMMaxPool2dFloat(kernel_size=3, stride=1, padding=1),
        (
            _channels_last(
                torch.linspace(-4.0, 4.0, steps=18, dtype=torch.float16).reshape(
                    1, 2, 3, 3
                )
            ),
        ),
    ),
    "f16_maxpool_3x3_s2_pad1": McuTestCase(
        CortexMMaxPool2dFloat(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        (
            _channels_last(
                torch.linspace(-3.0, 5.0, steps=32, dtype=torch.float16).reshape(
                    1, 2, 4, 4
                )
            ),
        ),
    ),
}


fallback_test_cases = {
    "f32_maxpool_ceil": McuTestCase(
        CortexMMaxPool2dFloat(kernel_size=3, stride=2, padding=1, ceil_mode=True),
        (
            _channels_last(
                torch.linspace(-2.0, 2.0, steps=16, dtype=torch.float32).reshape(
                    1, 1, 4, 4
                )
            ),
        ),
    ),
    "f16_maxpool_dilation": McuTestCase(
        CortexMMaxPool2dFloat(kernel_size=2, stride=1, padding=0, dilation=2),
        (
            _channels_last(
                torch.linspace(-2.0, 2.0, steps=36, dtype=torch.float16).reshape(
                    1, 1, 6, 6
                )
            ),
        ),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_max_pool2d_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    expected_after = (
        test_case.model.ops_after_transforms_f32
        if test_case.example_inputs[0].dtype == torch.float32
        else test_case.model.ops_after_transforms_f16
    )
    tester.export()
    tester.to_edge()
    tester.check(test_case.model.ops_before_transforms)
    tester.run_passes()
    tester.check_not(["executorch_exir_dialects_edge__ops_aten_max_pool2d_default"])
    tester.check_not(
        ["executorch_exir_dialects_edge__ops_aten_max_pool2d_with_indices_default"]
    )
    tester.check(expected_after)
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs)


@parametrize("test_case", fallback_test_cases)
def test_dialect_max_pool2d_float_fallback(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.run_passes()
    tester.check(
        {
            "executorch_exir_dialects_edge__ops_aten_max_pool2d_default": 1,
        }
    )
    tester.check_not(
        [
            "executorch_exir_dialects_edge__ops_cortex_m_max_pool2d_f32_default",
            "executorch_exir_dialects_edge__ops_cortex_m_max_pool2d_f16_default",
        ]
    )
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs)


@XfailIfNoCorstone300
@parametrize("test_case", test_cases)
def test_implementation_max_pool2d_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.run_passes()
    tester.to_executorch()
    tester.serialize()
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs)
